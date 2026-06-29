"""nff/sofa/hinge_objective.py — hinge loss and its gradient, in one place.

Client-side (runs in kgnn_mac; never imports ``Sofa``). Owns the three-term hinge
objective and, crucially, the assembly of its gradient from three sources of
different cost::

    loss = w_fatigue · ε_plastic/ε_yield   + w_mat · area/area₀   + w_gap · (gap/gap₀)²
             └─ oracle FD Jacobian ─┘          └─ analytic FD ─┘     └─ closed-form ─┘

Only the strain term needs physics: its gradient is the SOFA oracle's finite-
difference Jacobian row (the expensive 54-sim call). The material term is a central
FD on the cheap analytic hinge area (microsecond shoelace evals, no SOFA); the gap
term is an exact derivative. ``loss_and_grad`` evaluates the loss once and returns
the summed 9-vector gradient plus an ``aux`` dict of reporting quantities, so the
optimizer loop never has to know how any single term is differentiated.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from nff.sofa import oracle_payload as op
from nff.sofa.fatigue import cycles_to_failure
from nff.sofa.hinge_geometry import compute_hinge_geometry

_GAP_IDX = op.PARAM_NAMES.index('gap')


# ── Parameter-vector helpers (name ↔ value) ────────────────────────────────────

def phys_from_params(params: np.ndarray) -> dict:
    """Parameter vector → name→value dict (keys == Tesseract schema names)."""
    return {n: float(v) for n, v in zip(op.PARAM_NAMES, params)}


def _bezier_from_phys(phys: dict) -> dict:
    """Physical param dict → bezier_params for compute_hinge_geometry / the oracle."""
    return {
        's0_top': phys['s0_top'], 's0_bot': phys['s0_bot'],
        's1_top': phys['s1_top'], 's1_bot': phys['s1_bot'],
        'bc_up_xy': [phys['bcu_x'], phys['bcu_y']],
        'bc_lo_xy': [phys['bcl_x'], phys['bcl_y']],
    }


# ── Analytic hinge area + its gradient (no SOFA) ───────────────────────────────

def hinge_area(phys: dict, cs) -> float:
    """Hinge-strip area [m²] from the Bézier boundary — analytic, no SOFA.

    Lens area between the upper and lower arcs (shoelace on a sampled boundary);
    a cheap regulariser that rewards lean hinges.
    """
    geo = compute_hinge_geometry(cs, gap=phys['gap'], bezier_params=_bezier_from_phys(phys))

    def _bez(p0, c, p2, n=40):
        t = np.linspace(0.0, 1.0, n)[:, None]
        return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * c + t ** 2 * p2

    total = 0.0
    for hd in geo['hinge_data']:
        up = _bez(hd['p0_top'], hd['bc_up'], hd['p1_top'])
        lo = _bez(hd['p0_bot'], hd['bc_lo'], hd['p1_bot'])
        poly = np.vstack([up, lo[::-1]])
        x, y = poly[:, 0], poly[:, 1]
        total += 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return float(total)


def area_grad(params: np.ndarray, cs, eps: float = 1e-5) -> np.ndarray:
    """d(hinge_area)/d(param) via central FD on the cheap analytic area.

    Same central-difference recipe the SOFA oracle uses for the strain Jacobian,
    but applied to ``hinge_area`` — a microsecond geometry eval, not a simulation —
    so the 2·n_params evaluations cost nothing.
    """
    g = np.zeros(len(params))
    for i in range(len(params)):
        pp = params.copy(); pp[i] += eps
        pm = params.copy(); pm[i] -= eps
        g[i] = (hinge_area(phys_from_params(pp), cs)
                - hinge_area(phys_from_params(pm), cs)) / (2 * eps)
    return g


# ── Objective: weights + references + loss/grad assembly ───────────────────────

@dataclass
class HingeObjective:
    """Three-term hinge objective with its gradient-assembly logic.

    Holds the loss weights, the plasticity/fatigue criteria, and the normalisation
    references fixed from the initial design. Build with :func:`build_objective`.
    """

    cs: Any
    # Loss weights.
    w_fatigue: float
    w_mat: float
    w_gap: float
    # Material / failure criteria.
    eps_yield: float
    eps_frac: float
    fat_ef: float          # Coffin-Manson ductility coefficient ε_f'
    fat_c: float           # Coffin-Manson ductility exponent c (< 0)
    n_target: float        # target cycles-to-failure (reporting)
    # Normalisation references (fixed from the initial design).
    gap_ref: float
    area_ref: float
    area_min: float        # area floor — stop pulling material below this

    def loss_and_grad(self, params: np.ndarray, fwd: dict,
                      jac: dict) -> tuple[float, np.ndarray, dict]:
        """Evaluate loss and assemble its gradient from all three term sources.

        Args:
            params: current 9-vector of hinge design parameters.
            fwd: oracle ``apply`` result (strain/stress scalars).
            jac: oracle ``jacobian`` result; expects the
                ``smooth_principal_strain`` row vs every param.

        Returns:
            ``(loss, grad, aux)`` — scalar loss, summed 9-vector gradient, and a
            dict of per-term losses and reporting scalars for history/printing.
        """
        phys     = phys_from_params(params)
        max_vm   = float(fwd['max_von_mises_stress'])
        strain   = float(fwd['smooth_principal_strain'])   # design objective (KS-smooth)
        true_eps = float(fwd['max_principal_strain'])      # honest peak, reporting only
        area     = hinge_area(phys, self.cs)
        gp       = phys['gap']

        # loss = w_fatigue·ε_plastic/ε_yield + w_mat·area/area₀ + w_gap·(gap/gap₀)²
        eps_p = max(0.0, strain - self.eps_yield)
        n_f   = cycles_to_failure(eps_p, self.fat_ef, self.fat_c)
        l_fat = self.w_fatigue * eps_p / self.eps_yield
        l_mat = self.w_mat * area / self.area_ref
        l_gap = self.w_gap * (gp / self.gap_ref) ** 2
        loss  = l_fat + l_mat + l_gap

        # ── Gradient assembly: one term per source, summed by linearity ───────
        n_p = len(op.PARAM_NAMES)

        # Strain term — oracle's FD Jacobian row (the only SOFA-backed gradient).
        dstrain = np.array([
            float(jac['smooth_principal_strain'][ki])
            if 'smooth_principal_strain' in jac else 0.0
            for ki in op.PARAM_NAMES
        ])
        # ε_plastic = max(0, ε − ε_yield) has a kink at yield: subgradient is dstrain
        # while plastic, 0 below — below yield only material/gap steer.
        d_fat = (self.w_fatigue / self.eps_yield) \
            * (1.0 if strain > self.eps_yield else 0.0) * dstrain

        # Material term — cheap central FD on the analytic area (no SOFA).
        # Degeneracy guard: stop pulling at the floor, else the hinge collapses.
        d_mat = (np.zeros(n_p) if area <= self.area_min
                 else (self.w_mat / self.area_ref) * area_grad(params, self.cs))

        # Gap term — exact analytic derivative of (gap/gap₀)²; touches only gap.
        d_gap = np.zeros(n_p)
        d_gap[_GAP_IDX] = self.w_gap * 2.0 * gp / self.gap_ref ** 2

        grad = d_fat + d_mat + d_gap

        aux = {
            'l_fat': l_fat, 'l_mat': l_mat, 'l_gap': l_gap,
            'strain': strain, 'true_eps': true_eps, 'eps_p': eps_p,
            'n_f': n_f, 'max_vm': max_vm, 'area': area,
        }
        return loss, grad, aux


def build_objective(cfg: dict, cs, init_params: np.ndarray) -> HingeObjective:
    """Parse weights/criteria from ``cfg`` and fix references from the initial design."""
    sofa_cfg = cfg.get('sofa', {})
    mat_cfg  = cfg.get('material', {})
    loss_cfg = cfg.get('loss', {})

    eps_yield = float(mat_cfg.get('yield_strain',
                      float(mat_cfg.get('yield_strength', 50e6)) /
                      float(mat_cfg.get('young_modulus', 3.5e9))))

    gap_ref  = max(float(sofa_cfg.get('gap_initial', 0.003)), 1e-6)
    area_ref = max(hinge_area(phys_from_params(init_params), cs), 1e-12)

    return HingeObjective(
        cs        = cs,
        w_fatigue = float(loss_cfg.get('w_fatigue', loss_cfg.get('w_strain', 5.0))),
        w_mat     = float(loss_cfg.get('w_mat', 2.0)),
        w_gap     = float(loss_cfg.get('w_gap', 0.5)),
        eps_yield = eps_yield,
        eps_frac  = float(mat_cfg.get('fracture_strain', 0.045)),
        fat_ef    = float(mat_cfg.get('fatigue_ductility_coeff', 0.05)),
        fat_c     = float(mat_cfg.get('fatigue_ductility_exp', -0.6)),
        n_target  = float(loss_cfg.get('target_cycles', 100.0)),
        gap_ref   = gap_ref,
        area_ref  = area_ref,
        area_min  = float(loss_cfg.get('min_hinge_area_m2', 20e-6)),
    )
