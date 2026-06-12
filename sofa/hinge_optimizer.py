"""
sofa/hinge_optimizer.py — Corner-hinge cubic Bézier optimizer via Tesseract HTTP.

Optimizes 9 hinge parameters using the Tesseract physics oracle (SOFA FEM
running in Docker). All 13 parameters live directly in physical space (metres)
and are updated by a numpy Adam step; gap + the four edge reaches are kept
positive by projecting (clipping) to a small floor after each step. No softplus
reparametrisation — that froze gap/reaches ~500× slower than the control points.

Objective — a lean hinge that survives MANY closing cycles (low-cycle fatigue),
keeping the face gap small:
  loss = w_fatigue · ε_plastic / ε_yield    (minimise plastic strain → maximise N_f)
       + w_mat     · hinge_area / area₀      (lean hinge — minimise material)
       + w_gap     · (gap / gap₀)²           (keep the face gap small/controlled)
ε_plastic = max(0, ε_max − ε_yield) is the per-fold plastic strain (only strain
above yield fatigues). Cycles-to-failure via Coffin-Manson N_f = ½·(ε_p/ε_f')^(1/c)
(reported; PLA constants uncertain). Minimising ε_p pushes toward an elastic design
(ε < yield → ~unlimited cycles). The strain comes from the oracle's max principal
strain (FD Jacobian); material + gap are analytic client-side FD (no SOFA call).

Parameters (physical, metres) — gap + 4 reaches (positive, floored) + 8 free CPs:
  gap                — rigid face separation along ê_arm [m]
  s0_top, s0_bot     — upper/lower endpoint reach on face 0 [m]
  s1_top, s1_bot     — upper/lower endpoint reach on face 1 [m]
  bcu, bcl             — single interior CP for the upper/lower (quadratic) arcs [m]
  CP defaults come from the symmetric mesh geometry at the initial gap.

Gradient cost per epoch:
  POST /apply    →  1 forward pass (rotation mode)
  POST /jacobian →  26 central-FD perturbations × 1 mode = 26 SOFA sims (strain only)
  Total: ~27 SOFA simulations per optimizer step.

Usage
-----
    # Start the Tesseract server:
    docker run -p 8000:8000 nff-sofa-oracle

    conda run -n kgnn_mac python sofa/hinge_optimizer.py \\
        --config data/configs/sofa/hinge_opt_2face.yaml \\
        [--n-epochs 30] [--lr 0.05] [--tesseract-url http://localhost:8000]

Outputs  data/outputs/hinge_opt/<timestamp>_<config>/
    config.yaml         — copy of driving config
    convergence.npz     — per-epoch arrays for all 13 params + losses
    convergence.png     — convergence figure
"""

from __future__ import annotations

import argparse
import datetime
import os
import pathlib
import shutil
import sys
import types

import numpy as np
import requests
import yaml


# JAX CPU + x64 — needed only for CentroidalState.from_tessellation().
os.environ.setdefault('JAX_PLATFORM_NAME', 'cpu')
import jax
jax.config.update('jax_enable_x64', True)

REPO = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from nff.topology.core    import UnitPattern
from nff.topology.builder import build_tessellation
from nff.stages.state     import CentroidalState

PATTERNS_FILE = REPO / 'data' / 'library' / 'patterns.yaml'
OUTPUTS_DIR   = REPO / 'data' / 'outputs' / 'hinge_opt'

TESSERACT_DEFAULT_URL = 'http://localhost:8000'

# Parameter layout: 5 positive (softplus) followed by 8 free CP coords.
# Names match the Tesseract InputSchema field names exactly.
_POS_NAMES   = ['gap', 's0_top', 's0_bot', 's1_top', 's1_bot']
_FREE_NAMES  = ['bcu_x', 'bcu_y', 'bcl_x', 'bcl_y']   # one CP per (quadratic) arc
_PARAM_NAMES = _POS_NAMES + _FREE_NAMES            # 9, == _JAC_INPUTS order

_JAC_INPUTS  = list(_PARAM_NAMES)


# ── Numerics ──────────────────────────────────────────────────────────────────

def _phys(params: np.ndarray) -> dict:
    """13-vector of physical params → name→value dict (keys == schema names)."""
    return {n: float(v) for n, v in zip(_PARAM_NAMES, params)}


def _bezier_from_phys(phys: dict) -> dict:
    """Physical param dict → bezier_params for compute_hinge_geometry / the oracle."""
    return {
        's0_top': phys['s0_top'], 's0_bot': phys['s0_bot'],
        's1_top': phys['s1_top'], 's1_bot': phys['s1_bot'],
        'bc_up_xy': [phys['bcu_x'], phys['bcu_y']],
        'bc_lo_xy': [phys['bcl_x'], phys['bcl_y']],
    }


def _hinge_area(phys: dict, cs) -> float:
    """Hinge-strip area [m²] from the Bézier boundary — analytic, no SOFA.

    Area of the lens between the upper and lower arcs (shoelace on a sampled
    boundary). Cheap regulariser term that rewards lean hinges.
    """
    from nff.sofa.mesh_builder_gmsh import compute_hinge_geometry
    geo = compute_hinge_geometry(cs, gap=phys['gap'], bezier_params=_bezier_from_phys(phys))

    def _bez(p0, c, p2, n=40):
        t = np.linspace(0.0, 1.0, n)[:, None]
        return (1-t)**2*p0 + 2*(1-t)*t*c + t**2*p2

    total = 0.0
    for hd in geo['hinge_data']:
        up = _bez(hd['p0_top'], hd['bc_up'], hd['p1_top'])
        lo = _bez(hd['p0_bot'], hd['bc_lo'], hd['p1_bot'])
        poly = np.vstack([up, lo[::-1]])
        x, y = poly[:, 0], poly[:, 1]
        total += 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return float(total)


def _area_grad(params: np.ndarray, cs, eps: float = 1e-5) -> np.ndarray:
    """d(hinge_area)/d(param) via central FD on the cheap analytic area."""
    g = np.zeros(len(params))
    for i in range(len(params)):
        pp = params.copy(); pp[i] += eps
        pm = params.copy(); pm[i] -= eps
        g[i] = (_hinge_area(_phys(pp), cs) - _hinge_area(_phys(pm), cs)) / (2 * eps)
    return g


def _decode_array(v) -> np.ndarray:
    """Decode a Tesseract array-output JSON value into a numpy array."""
    import base64
    if isinstance(v, dict) and v.get('object_type') == 'array':
        shape, dtype, data = tuple(v['shape']), v['dtype'], v['data']
        if data.get('encoding') == 'base64':
            buf = base64.b64decode(data['buffer'])
            return np.frombuffer(buf, dtype=dtype).reshape(shape).copy()
        return np.asarray(data['buffer'], dtype=dtype).reshape(shape)
    return np.asarray(v)


class _NumpyAdam:
    """Minimal Adam optimizer in numpy — no JAX or optax dependency."""

    def __init__(self, lr: float, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m: np.ndarray | None = None
        self.v: np.ndarray | None = None
        self.t: int = 0

    def update(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grads
        self.v = self.b2 * self.v + (1 - self.b2) * grads ** 2
        m_hat = self.m / (1 - self.b1 ** self.t)
        v_hat = self.v / (1 - self.b2 ** self.t)
        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ── Tesseract HTTP client ─────────────────────────────────────────────────────

def _call_tesseract_apply(url: str, payload: dict) -> dict:
    """POST /apply and return the OutputSchema as a plain dict."""
    try:
        resp = requests.post(f"{url}/apply", json={"inputs": payload}, timeout=600)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Cannot connect to Tesseract server at {url}.\n"
            "Start the server: docker run -p 8000:8000 nff-sofa-oracle"
        )
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(
            f"Tesseract /apply failed ({resp.status_code}):\n{resp.text[:800]}"
        ) from e


def _call_tesseract_jacobian(
    url: str,
    payload: dict,
    jac_inputs: list[str],
    jac_outputs: list[str],
) -> dict:
    """POST /jacobian and return {output: {input: float}} Jacobian dict.

    Tesseract HTTP body convention:
      {"inputs": <InputSchema dict>, "jac_inputs": [...], "jac_outputs": [...]}
    """
    body = {"inputs": payload, "jac_inputs": jac_inputs, "jac_outputs": jac_outputs}
    try:
        resp = requests.post(f"{url}/jacobian", json=body, timeout=1800)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Cannot connect to Tesseract server at {url}.\n"
            "Start the server: docker run -p 8000:8000 nff-sofa-oracle"
        )
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(
            f"Tesseract /jacobian failed ({resp.status_code}):\n{resp.text[:800]}"
        ) from e


# ── Pattern loading ───────────────────────────────────────────────────────────

def _load_pattern(pattern_name: str) -> UnitPattern:
    with open(PATTERNS_FILE) as f:
        patterns = yaml.safe_load(f)
    if pattern_name not in patterns:
        raise ValueError(f"Pattern '{pattern_name}' not found in {PATTERNS_FILE}")
    raw = patterns[pattern_name]
    internal_hinges = []
    for h in raw.get('internal_hinges', []):
        hc = dict(h)
        if 'angle_factor' in hc:
            hc['angle'] = float(hc.pop('angle_factor')) * float(np.pi)
        internal_hinges.append(hc)
    return UnitPattern(
        vertices=np.array(raw['vertices'], dtype=float),
        faces=raw['faces'],
        internal_hinges=internal_hinges,
        external_hinges=raw.get('external_hinges', []),
    )


# ── CentroidalState setup ─────────────────────────────────────────────────────

def build_physical_cs(cfg: dict) -> types.SimpleNamespace:
    """Build a SimpleNamespace with the CS fields needed for Tesseract payloads.

    The tessellation is built from the pattern at JAX-normalized scale, then
    scaled by face_size_m to produce SI-unit coordinates for SOFA.
    """
    tess_cfg = cfg.get('tessellation', {})
    sofa_cfg = cfg.get('sofa', {})
    bc_cfg   = cfg.get('boundary_conditions', {})

    pattern_name = tess_cfg.get('pattern', 'unit_2face')
    nx = int(tess_cfg.get('width', 1))
    ny = int(tess_cfg.get('height', 1))

    pattern      = _load_pattern(pattern_name)
    tessellation = build_tessellation(pattern, nx=nx, ny=ny)

    n_faces      = len(tessellation.faces)
    clamped_face = int(bc_cfg.get('clamped_face', 0))
    loaded_face  = int(bc_cfg.get('loaded_face', n_faces - 1))

    tessellation.set_face_dofs(clamped_face, [0, 1, 2])
    tessellation.set_face_load(loaded_face, dof_id=2, value=1.0)

    cs = CentroidalState.from_tessellation(tessellation)

    face_size_m = float(sofa_cfg.get('face_size_m', 0.100))

    return types.SimpleNamespace(
        face_centroids             = np.array(cs.face_centroids)        * face_size_m,
        centroid_node_vectors      = np.array(cs.centroid_node_vectors) * face_size_m,
        hinge_node_pairs           = np.array(cs.hinge_node_pairs,           dtype=np.int32),
        hinge_adj_info             = np.array(cs.hinge_adj_info,             dtype=np.int32),
        constrained_face_DOF_pairs = np.array(cs.constrained_face_DOF_pairs, dtype=np.int32),
        loaded_face_DOF_pairs      = np.array(cs.loaded_face_DOF_pairs,      dtype=np.int32),
    )


# ── Payload builder ───────────────────────────────────────────────────────────

def _build_tesseract_payload(
    cs: types.SimpleNamespace,
    phys: dict,
    cfg: dict,
    clamped_faces: list,
    loaded_faces: list,
) -> dict:
    """Assemble the full Tesseract InputSchema as a JSON-serializable dict."""
    sofa_cfg = cfg.get('sofa', {})
    mat_cfg  = cfg.get('material', {})

    return {
        # CS topology
        "face_centroids":        cs.face_centroids.tolist(),
        "centroid_node_vectors": cs.centroid_node_vectors.tolist(),
        "hinge_node_pairs":      cs.hinge_node_pairs.tolist(),
        "hinge_adj_info":        cs.hinge_adj_info.tolist(),
        "clamped_faces":         clamped_faces,
        "loaded_faces":          loaded_faces,
        # Differentiable hinge params (9-param corner-hinge quadratic Bézier)
        **{name: float(phys[name]) for name in _PARAM_NAMES},
        # Mesh resolution
        "sheet_thickness":      float(sofa_cfg.get('sheet_thickness', 0.001)),
        "n_z":                  int(sofa_cfg.get('n_z', 2)),
        # Loading
        "rotation_angle_deg":     float(sofa_cfg.get('rotation_angle_deg', -5.0)),
        "applied_moment":         0.0,
        "loading_mode":           'rotation',
        "shear_displacement_m":   float(sofa_cfg.get('shear_displacement_m', 0.005)),
        "tension_displacement_m": float(sofa_cfg.get('tension_displacement_m', 0.005)),
        "skip_secondary_modes":   bool(sofa_cfg.get('skip_secondary_modes', False)),
        "n_steps":                int(sofa_cfg.get('n_steps', 500)),
        "fem_method":             str(sofa_cfg.get('fem_method', 'polar')),
        "rotation_pivot_auto":    bool(sofa_cfg.get('rotation_pivot_auto', True)),
        # Material
        "young_modulus":  float(mat_cfg.get('young_modulus', 3.5e9)),
        "poisson_ratio":  float(mat_cfg.get('poisson_ratio', 0.36)),
        "yield_strength": float(mat_cfg.get('yield_strength', 50e6)),
        # FD step
        "fd_eps": float(sofa_cfg.get('fd_eps', 1e-5)),
    }


# ── Optimization loop ─────────────────────────────────────────────────────────

def run_optimization(
    cfg: dict,
    n_epochs: int,
    lr: float,
    tesseract_url: str,
    out_dir: pathlib.Path,
) -> dict:
    sofa_cfg = cfg.get('sofa', {})
    mat_cfg  = cfg.get('material', {})

    cs_static = build_physical_cs(cfg)

    # ── Positive-param initial values (gap + 4 reaches) ───────────────────────
    gap_init   = float(sofa_cfg.get('gap_initial',   0.003))
    reach_init = float(sofa_cfg.get('reach_initial', gap_init))
    pos_init = np.array([
        gap_init,
        float(sofa_cfg.get('s0_top_initial', reach_init)),
        float(sofa_cfg.get('s0_bot_initial', reach_init)),
        float(sofa_cfg.get('s1_top_initial', reach_init)),
        float(sofa_cfg.get('s1_bot_initial', reach_init)),
    ], dtype=np.float64)

    # ── Free CP initial values — symmetric mesh geometry at the initial gap ───
    from nff.sofa.mesh_builder_gmsh import compute_hinge_geometry
    _hd0 = compute_hinge_geometry(cs_static, gap=gap_init)['hinge_data'][0]
    _cp_default = {
        'bcu_x': _hd0['bc_up'][0], 'bcu_y': _hd0['bc_up'][1],
        'bcl_x': _hd0['bc_lo'][0], 'bcl_y': _hd0['bc_lo'][1],
    }
    # Optional concave start: bow each arc's CP INWARD (toward the hinge axis) by
    # initial_concave_bow_m instead of the default outward bulge — a stretched,
    # concave starting shape to steer the optimizer out of the thick-bulb basin.
    _bow = float(sofa_cfg.get('initial_concave_bow_m', 0.0))
    if _bow > 0:
        for (kx, ky), pa, pb in [(('bcu_x', 'bcu_y'), 'p0_top', 'p1_top'),
                                 (('bcl_x', 'bcl_y'), 'p0_bot', 'p1_bot')]:
            mid = 0.5 * (np.asarray(_hd0[pa]) + np.asarray(_hd0[pb]))
            out = np.array([_cp_default[kx], _cp_default[ky]]) - mid     # default = outward
            inward = mid - _bow * out / (np.linalg.norm(out) + 1e-12)    # flip → inward
            _cp_default[kx], _cp_default[ky] = float(inward[0]), float(inward[1])

    free_init = np.array([
        float(sofa_cfg.get(f'{name}_initial', _cp_default[name])) for name in _FREE_NAMES
    ], dtype=np.float64)

    # Extract clamped/loaded face indices from CS DOF arrays.
    clamped_faces = sorted({int(f) for f in cs_static.constrained_face_DOF_pairs[:, 0]})
    loaded_faces  = sorted({int(f) for f in cs_static.loaded_face_DOF_pairs[:, 0]})

    # ── Parameters live directly in physical space (metres) ───────────────────
    # No softplus — Adam moves every coordinate at a comparable ~lr/step rate, so
    # gap and the boundary reaches actually move. Positivity is enforced by
    # projecting (clipping) gap + reaches to a floor after each step.
    params   = np.concatenate([pos_init, free_init]).astype(np.float64)
    pos_mask = np.array([n in _POS_NAMES for n in _PARAM_NAMES])  # gap + 4 reaches
    floor    = float(sofa_cfg.get('param_floor', 0.0005))         # 0.5 mm min

    # ── Loss weights + the plasticity / low-cycle-fatigue criterion ───────────
    loss_cfg = cfg.get('loss', {})
    w_fatigue = float(loss_cfg.get('w_fatigue', loss_cfg.get('w_strain', 5.0)))  # min plastic strain
    w_mat     = float(loss_cfg.get('w_mat', 2.0))   # material (lean hinge)
    w_gap     = float(loss_cfg.get('w_gap', 0.5))   # keep the face gap small/controlled
    eps_frac  = float(mat_cfg.get('fracture_strain', 0.045))
    # Plasticity: strain above yield is plastic; only the plastic part fatigues.
    eps_yield = float(mat_cfg.get('yield_strain',
                      float(mat_cfg.get('yield_strength', 50e6)) /
                      float(mat_cfg.get('young_modulus', 3.5e9))))
    # Coffin-Manson low-cycle fatigue: ε_p = ε_f'·(2N)^c  →  N_f = ½·(ε_p/ε_f')^(1/c).
    # PLA constants are uncertain (scattered literature) — treat N_f as ballpark.
    fat_ef = float(mat_cfg.get('fatigue_ductility_coeff', 0.05))   # ε_f'
    fat_c  = float(mat_cfg.get('fatigue_ductility_exp', -0.6))     # c (< 0)
    n_target = float(loss_cfg.get('target_cycles', 100.0))         # reporting reference

    def _cycles_to_failure(eps_p: float) -> float:
        if eps_p <= 1e-9:
            return float('inf')   # stays elastic → effectively unlimited cycles
        return 0.5 * (eps_p / fat_ef) ** (1.0 / fat_c)

    gap_ref  = max(gap_init, 1e-6)
    area_ref = max(_hinge_area(_phys(params), cs_static), 1e-12)
    # Degeneracy guard: stop rewarding material reduction below this floor — a
    # too-small hinge is unmanufacturable and makes the SOFA solve ill-conditioned.
    area_min = float(loss_cfg.get('min_hinge_area_m2', 20e-6))   # 20 mm²

    optimizer = _NumpyAdam(lr)
    history: dict = {k: [] for k in _PARAM_NAMES + [
        'total_loss', 'loss_fatigue', 'loss_mat', 'loss_gap',
        'max_strain', 'plastic_strain', 'cycles_Nf', 'max_vm_rot', 'hinge_area']}

    _p0 = _phys(params)
    print(f"\nHinge optimization (9-param quadratic Bézier, physical space): {n_epochs} epochs, lr={lr}")
    print(f"  Tesseract URL: {tesseract_url}")
    print(f"  loss = {w_fatigue}·ε_plastic/ε_y + {w_mat}·area/area₀ + {w_gap}·(gap/gap₀)²")
    print(f"  PLA: ε_yield={eps_yield*100:.2f}%  ε_fracture={eps_frac*100:.1f}%  "
          f"Coffin-Manson(ε_f'={fat_ef}, c={fat_c}); target ≥ {n_target:.0f} cycles")
    print(f"  Initial: gap={_p0['gap']*1e3:.2f} mm  "
          f"reach s0=({_p0['s0_top']*1e3:.2f},{_p0['s0_bot']*1e3:.2f}) "
          f"s1=({_p0['s1_top']*1e3:.2f},{_p0['s1_bot']*1e3:.2f}) mm  "
          f"area={area_ref*1e6:.1f} mm²\n")

    def _get_val(v):
        if isinstance(v, dict):
            if 'data' in v and 'buffer' in v['data']: return float(v['data']['buffer'])
            if 'value' in v: return float(v['value'])
        return float(v)

    def _save_convergence():
        np.savez(
            str(out_dir / 'convergence.npz'),
            **{name: np.array(history[name]) for name in _PARAM_NAMES},
            total_loss      = np.array(history['total_loss']),
            loss_fatigue    = np.array(history['loss_fatigue']),
            loss_mat        = np.array(history['loss_mat']),
            loss_gap        = np.array(history['loss_gap']),
            max_strain      = np.array(history['max_strain']),
            plastic_strain  = np.array(history['plastic_strain']),
            cycles_Nf       = np.array(history['cycles_Nf']),
            max_vm_rot      = np.array(history['max_vm_rot']),
            hinge_area      = np.array(history['hinge_area']),
            fracture_strain = np.array(eps_frac),
            yield_strain    = np.array(eps_yield),
            target_cycles   = np.array(n_target),
            stress          = np.array(history['max_vm_rot']),   # backward-compat alias
        )

    for epoch in range(n_epochs):
        phys    = _phys(params)
        payload = _build_tesseract_payload(cs_static, phys, cfg, clamped_faces, loaded_faces)

        # ── Oracle calls (apply + strain Jacobian) — robust to a hung sim ──────
        try:
            fwd = _call_tesseract_apply(tesseract_url, payload)
            jac = _call_tesseract_jacobian(
                tesseract_url, payload, _JAC_INPUTS, ['max_principal_strain'])
        except Exception as ex:
            print(f"  epoch {epoch+1}: oracle call failed ({type(ex).__name__}: {ex}); "
                  "stopping early — keeping the epochs completed so far.")
            break

        max_vm  = _get_val(fwd['max_von_mises_stress'])
        strain  = _get_val(fwd['max_principal_strain'])
        area    = _hinge_area(phys, cs_static)
        gp      = phys['gap']

        # loss = w_fatigue·ε_plastic/ε_yield  +  w_mat·area/area₀  +  w_gap·(gap/gap₀)²
        # Only the PLASTIC part of the strain fatigues; minimising it maximises the
        # cycles-to-failure N_f (and pushes the design toward elastic = ~unlimited life).
        eps_p    = max(0.0, strain - eps_yield)
        n_f      = _cycles_to_failure(eps_p)
        l_fat    = w_fatigue * eps_p / eps_yield
        l_mat    = w_mat * area / area_ref
        l_gap    = w_gap * (gp / gap_ref) ** 2
        loss     = l_fat + l_mat + l_gap

        # ── Gradients ─────────────────────────────────────────────────────────
        def _jac_val(output_key, inp_key):
            return _get_val(jac[output_key][inp_key]) if output_key in jac else 0.0
        dstrain  = np.array([_jac_val('max_principal_strain', ki) for ki in _JAC_INPUTS])
        # d(ε_p)/d(param) = dstrain when plastic (strain > yield), else 0.
        d_fat    = (w_fatigue / eps_yield) * (1.0 if strain > eps_yield else 0.0) * dstrain
        # Material + gap terms are analytic / cheap-FD (no SOFA). Degeneracy guard:
        # stop the material pull once the hinge is at the min area (else it collapses).
        n_p      = len(_PARAM_NAMES)
        d_mat    = (np.zeros(n_p) if area <= area_min
                    else (w_mat / area_ref) * _area_grad(params, cs_static))
        d_gap    = np.zeros(n_p); d_gap[0] = w_gap * 2.0 * gp / gap_ref ** 2
        grad     = d_fat + d_mat + d_gap

        params = optimizer.update(params, grad)
        params[pos_mask] = np.maximum(params[pos_mask], floor)   # project to positivity

        # ── Record + save every epoch (a crash never loses everything) ────────
        for name in _PARAM_NAMES:
            history[name].append(phys[name])
        history['total_loss'].append(loss)
        history['loss_fatigue'].append(l_fat)
        history['loss_mat'].append(l_mat)
        history['loss_gap'].append(l_gap)
        history['max_strain'].append(strain)
        history['plastic_strain'].append(eps_p)
        history['cycles_Nf'].append(min(n_f, 1e9))   # cap inf for storage
        history['max_vm_rot'].append(max_vm)
        history['hinge_area'].append(area)
        _save_convergence()

        _nf_s = '∞' if not np.isfinite(n_f) else f'{n_f:.1f}'
        print(f"  epoch {epoch+1:3d}/{n_epochs}  "
              f"loss={loss:.3f} (fat={l_fat:.2f} mat={l_mat:.2f} gap={l_gap:.2f})  "
              f"ε_max={strain*100:.2f}% ε_p={eps_p*100:.2f}% N_f={_nf_s}cyc  "
              f"σ_max={max_vm/1e6:.0f}MPa area={area*1e6:.1f}mm²  "
              f"gap={phys['gap']*1e3:.3f} mm  "
              f"s0=({phys['s0_top']*1e3:.2f},{phys['s0_bot']*1e3:.2f}) "
              f"s1=({phys['s1_top']*1e3:.2f},{phys['s1_bot']*1e3:.2f}) mm  "
              f"bc_up=({phys['bcu_x']*1e3:.2f},{phys['bcu_y']*1e3:.2f})  "
              f"bc_lo=({phys['bcl_x']*1e3:.2f},{phys['bcl_y']*1e3:.2f}) mm")

    if not history['total_loss']:
        print("No epochs completed — aborting before final-state capture.")
        return history

    # ── Final state at the best (lowest-loss) design — capture field for viz ──
    best_idx  = int(np.argmin(history['total_loss']))
    best_phys = {name: float(history[name][best_idx]) for name in _PARAM_NAMES}
    print(f"\nCapturing final-state field at best design (epoch {best_idx + 1}) ...")
    payload = _build_tesseract_payload(cs_static, best_phys, cfg, clamped_faces, loaded_faces)
    payload['return_fields']        = True
    payload['skip_secondary_modes'] = True   # rotation field only
    try:
        fwd = _call_tesseract_apply(tesseract_url, payload)
        np.savez(
            str(out_dir / 'final_state.npz'),
            von_mises_field = _decode_array(fwd['von_mises_field']),
            deformed_nodes  = _decode_array(fwd['deformed_nodes']),
            mesh_tets       = _decode_array(fwd['mesh_tets']),
            best_idx        = np.array(best_idx),
            # CS topology so the visualizer can rebuild the initial/optimal meshes.
            face_centroids             = cs_static.face_centroids,
            centroid_node_vectors      = cs_static.centroid_node_vectors,
            hinge_node_pairs           = cs_static.hinge_node_pairs,
            hinge_adj_info             = cs_static.hinge_adj_info,
            constrained_face_DOF_pairs = cs_static.constrained_face_DOF_pairs,
            loaded_face_DOF_pairs      = cs_static.loaded_face_DOF_pairs,
        )
        _nf = history['cycles_Nf'][best_idx]
        print(f"  final_state.npz saved (best epoch {best_idx + 1}: "
              f"ε_max={history['max_strain'][best_idx]*100:.2f}%, "
              f"ε_p={history['plastic_strain'][best_idx]*100:.2f}%, "
              f"N_f={'∞' if _nf >= 1e9 else f'{_nf:.0f}'} cyc, "
              f"area={history['hinge_area'][best_idx]*1e6:.1f} mm²).")
    except Exception as ex:
        print(f"  WARNING: final-state field capture failed ({ex}); "
              "visualizer will skip the von Mises panel.")

    return history



# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description='Optimize SOFA hinge dimensions via Tesseract HTTP.')
    p.add_argument('--config',
                   default='data/configs/sofa/hinge_opt_2face.yaml',
                   help='Path to YAML config.')
    p.add_argument('--n-epochs',    type=int,   default=None,
                   help='Override optimization.n_epochs from config.')
    p.add_argument('--lr',          type=float, default=None,
                   help='Override optimization.learning_rate from config.')
    p.add_argument('--tesseract-url', default=TESSERACT_DEFAULT_URL,
                   help='URL of the running Tesseract server (Docker).')
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    opt_cfg  = cfg.get('optimization', {})
    n_epochs = args.n_epochs if args.n_epochs is not None else int(opt_cfg.get('n_epochs', 30))
    lr       = args.lr       if args.lr       is not None else float(opt_cfg.get('learning_rate', 0.05))
    url      = args.tesseract_url

    # Verify server is reachable before committing to a long run.
    try:
        requests.get(f"{url}/health", timeout=30)
    except (requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout):
        print(f"ERROR: cannot reach Tesseract server at {url}")
        print("Start the server:  docker run -p 8000:8000 -e TESSERACT_RUNTIME_SERVE_HOST=0.0.0.0 nff-sofa-oracle:latest serve")
        raise SystemExit(1)

    config_name = pathlib.Path(args.config).stem
    timestamp   = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir     = OUTPUTS_DIR / f'{timestamp}_{config_name}'
    out_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(args.config, out_dir / 'config.yaml')

    history = run_optimization(cfg, n_epochs, lr, url, out_dir)

    best_idx = int(np.argmin(history['total_loss']))
    b = lambda k: history[k][best_idx] * 1e3
    _nfb = history['cycles_Nf'][best_idx]
    print(f'\nBest (epoch {best_idx + 1}): '
          f'ε_max={history["max_strain"][best_idx]*100:.2f}%  '
          f'ε_p={history["plastic_strain"][best_idx]*100:.2f}%  '
          f'N_f={"∞" if _nfb >= 1e9 else f"{_nfb:.0f}"} cyc  '
          f'area={history["hinge_area"][best_idx]*1e6:.1f} mm²  '
          f'gap={b("gap"):.3f} mm  '
          f's0=({b("s0_top"):.2f},{b("s0_bot"):.2f}) s1=({b("s1_top"):.2f},{b("s1_bot"):.2f}) mm  '
          f'bc_up=({b("bcu_x"):.2f},{b("bcu_y"):.2f}) bc_lo=({b("bcl_x"):.2f},{b("bcl_y"):.2f}) mm')
    print(f'Results saved → {out_dir}')


if __name__ == '__main__':
    main()
