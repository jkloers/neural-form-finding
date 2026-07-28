"""Standard 80 gsm A1 copy/printer paper: orthotropic-elastic, tensile-tear failure.

Tier-1 physically-faithful paper (hypotheses H2 orthotropy, H3 buckling, H6 tearing). The
fold is **elastic-until-tear** by default: geometric nonlinearity (NLGEOM) lets the ligament
buckle out of plane and relieve bending strain, and the hinge fails by **tensile tearing** at
the ligament root once the max principal strain reaches the strain-to-break -- not by metal
ductile necking. This matches how the surrogate is consumed (Stage-2 solves the loaded
equilibrium of W(a,s,theta), not unloaded spring-back), so modelling permanent set is not
required for a first faithful model.

The permanent CREASE set (H5) is available as an isotropic-J2 ``*PLASTIC`` block, gated by
``Hypotheses.crease_softening``. CAVEAT (validate on the first real ccx run): CalculiX's
built-in von-Mises plasticity assumes ISOTROPIC elasticity; combining it with orthotropic
``*ELASTIC`` may require a UMAT. The default (crease off) is fully native and robust.

Material axes: 1 = MD (machine direction), 2 = CD (cross), 3 = ZD (thickness). ``*ORIENTATION``
sets MD relative to the secondary cut (+x), rotated by ``Hypotheses.orientation_deg``.

Parameters: standard 80 g/m^2 office copy paper (see PAPER_80GSM for values + provenance).
All stresses/moduli in MPa (N/mm^2); strains dimensionless.
"""

from __future__ import annotations

import numpy as np

from nff.rve.damage import max_principal_strain, stress_triaxiality
from nff.rve.materials.base import Hypotheses, Material

# Standard 80 gsm A1 copy/printer paper. Values are mid-range handbook figures for uncoated
# wood-free copy paper at ~50% RH; every one carries real spread (±20-40%) and should be
# refined against the actual stock. Axes: 1=MD, 2=CD, 3=ZD (thickness).
PAPER_80GSM = dict(
    # orthotropic elasticity [MPa]
    E_MD=6800.0,     # machine-direction Young's modulus (~6-8 GPa); stiff along the web
    E_CD=3400.0,     # cross-direction modulus; E_MD/E_CD ~= 2 (typical draw anisotropy)
    E_ZD=40.0,       # through-thickness modulus (fiber mat is soft in ZD; ~E_MD/170)
    nu12=0.30,       # in-plane MD-CD Poisson ratio
    nu13=0.01,       # out-of-plane Poisson ratios ~ 0 (poorly known, tiny effect: nu31 ~ nu13*E3/E1)
    nu23=0.01,
    G12=2000.0,      # in-plane shear modulus
    G13=20.0,        # transverse (interlaminar) shear moduli -- low, uncertain
    G23=20.0,
    # crease plasticity (only emitted when Hypotheses.crease_softening) -- isotropic J2 surrogate
    sigma_y=20.0,    # crease-onset yield (~0.55 * MD tensile strength)
    sigma_u=35.0,    # MD tensile strength [MPa]
    eps_p_u=0.017,   # plastic strain at strength (total strain-to-break ~2%)
    # tensile-tear failure (H6), bending/triaxiality-aware. eps_tear0 is the PHYSICAL uniaxial
    # tensile strain-to-break of 80 gsm copy paper (~2-4%); the bending tolerance is carried by the
    # criterion itself (membrane_tear_from_frame: through-thickness-averaged strain removes the
    # bending gradient) + the fracture locus eps_f(eta). NOT tuned to match observation -- set to
    # the material value and let the physics decide (measured 2026-07-11: membrane+triaxiality cut
    # the raw-surface conservatism ~40-50%; the residual gap vs the observed 60 deg fold is the
    # pure-rotation-spine over-straining the ligament and/or a higher bending-regime fracture strain
    # -- calibrate eps_tear0/k against the physical print).
    eps_tear0=0.03,
    k=1.5,           # triaxiality sensitivity of the fracture locus (Johnson-Cook-like)
)


def fracture_locus(eta: np.ndarray, eps_f0: float, k: float,
                   eta_floor: float = -1.0 / 3.0, eps_f_cap: float = 3.0) -> np.ndarray:
    """Triaxiality-dependent tear strain, normalised so ``eps_f(1/3) = eps_f0`` (uniaxial tension).

    Johnson-Cook-like. ``k`` sets how fast the tolerable strain falls with tension / rises with
    shear-compression; ``eta_floor`` is the compression cutoff and ``eps_f_cap`` bounds the rise.
    Paper-only: the elasto-plastic materials use the constant-``eps_f`` plastic-dissipation measure
    in :mod:`nff.rve.damage` instead.
    """
    eta_c = np.maximum(np.asarray(eta, float), eta_floor)
    return np.minimum(eps_f0 * np.exp(-k * (eta_c - 1.0 / 3.0)), eps_f_cap)


def _column_average(coords: np.ndarray, arrays, tol: float = 0.05):
    """Average per-node ``arrays`` over through-thickness columns (nodes sharing an in-plane x,y).

    Groups nodes by their reference in-plane position (rounded to ``tol`` mm) -- one group per
    extruded column -- and returns each array averaged within its column, one row per column. This
    removes the bending strain gradient (tension outer / compression inner), leaving the membrane
    (mid-surface) response.
    """
    key = np.round(coords[:, :2] / tol).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    n = int(inv.max()) + 1
    cnt = np.bincount(inv, minlength=n)
    out = []
    for A in arrays:
        A = np.asarray(A, float)
        acc = np.stack([np.bincount(inv, weights=A[:, c], minlength=n) for c in range(A.shape[1])],
                       axis=1)
        out.append(acc / cnt[:, None])
    return out


def _tear_margin(principal: np.ndarray, S, eps_tear0: float, k: float, q: float) -> float:
    """Robust percentile of the triaxiality-aware tear damage ``D = principal / eps_f(eta)``."""
    if S is not None and np.size(S):
        S = np.asarray(S, float)
        if S.ndim == 2 and S.shape[1] >= 6 and S.shape[0] == principal.shape[0]:
            eps_f = fracture_locus(stress_triaxiality(S[:, :6]), eps_tear0, k)
            return float(np.percentile(principal / eps_f, q))
    return float(np.percentile(principal, q)) / eps_tear0


def membrane_tear_from_frame(frame: dict, coords, *, eps_tear0: float = 0.03, k: float = 1.5,
                             q: float = 99.0) -> float:
    """Bending- and triaxiality-aware tear margin ``D`` (>=1 => tear).

    Averages strain and stress through the thickness (per in-plane column) so the bending gradient
    cancels and the membrane (mid-surface) state remains, then applies the tear locus. A pure fold
    (no membrane stretch) accumulates little membrane strain and does not tear regardless of fold
    angle -- matching that paper creases to 180 deg without tearing. With ``coords=None`` this
    degrades to the raw surface criterion.
    """
    E = frame.get("TOSTRAIN")
    if E is None or not np.size(E):
        return float("nan")
    E = np.asarray(E, float)
    S = frame.get("STRESS")
    if coords is None:
        return _tear_margin(max_principal_strain(E), S, eps_tear0, k, q)
    coords = np.asarray(coords, float)
    if S is not None and np.size(S) and np.asarray(S).shape == E.shape:
        E_col, S_col = _column_average(coords, [E, np.asarray(S, float)])
    else:
        (E_col,) = _column_average(coords, [E]); S_col = None
    return _tear_margin(max_principal_strain(E_col), S_col, eps_tear0, k, q)


class PaperOrthotropic(Material):
    """Orthotropic-elastic 80 gsm copy paper with a tensile-tear criterion.

    PARKED, and the one material that does NOT use the shared plastic-dissipation damage: paper is
    modelled as elastic-until-tear, so it emits no PEEQ at all and ``<PEEQ>_lig`` is undefined for
    it. Its membrane tear margin lives here rather than in :mod:`nff.rve.damage` so that the shared
    module carries exactly one damage definition.
    """

    name = "PAPER"
    _orient = "ORIENT_MD"     # *ORIENTATION name binding MD to the local material x-axis

    def __init__(self, params: dict | None = None):
        self.params = dict(PAPER_80GSM if params is None else params)
        self.eps_tear0 = self.params["eps_tear0"]
        self.k = self.params["k"]

    @classmethod
    def from_dict(cls, params: dict) -> "PaperOrthotropic":
        return cls(params)

    def _orientation_card(self, hyp: Hypotheses) -> str:
        """*ORIENTATION rotating MD by orientation_deg about z (local x = MD, xy-plane holds CD)."""
        th = np.radians(hyp.orientation_deg)
        c, s = float(np.cos(th)), float(np.sin(th))
        return (f"*ORIENTATION, NAME={self._orient}, SYSTEM=RECTANGULAR\n"
                f"{c:.6f}, {s:.6f}, 0.0, {-s + 0.0:.6f}, {c:.6f}, 0.0")

    def constitutive_cards(self, hyp: Hypotheses, *, elastic_only: bool = False) -> str:
        m = self.params
        lines = [
            f"*MATERIAL, NAME={self.name}",
            "*ELASTIC, TYPE=ENGINEERING CONSTANTS",
            (f"{m['E_MD']:.1f}, {m['E_CD']:.1f}, {m['E_ZD']:.1f}, {m['nu12']:.3f}, "
             f"{m['nu13']:.3f}, {m['nu23']:.3f}, {m['G12']:.1f}, {m['G13']:.1f}"),
            f"{m['G23']:.1f}, 0.0",                                    # G23 + temperature
        ]
        if not elastic_only and hyp.crease_softening:                 # H5 crease permanent set
            lines += ["*PLASTIC",
                      f"{m['sigma_y']:.1f}, 0.0",
                      f"{m['sigma_u']:.1f}, {m['eps_p_u']:.4f}"]
        lines.append(self._orientation_card(hyp))
        return "\n".join(lines)

    def section_cards(self, elset: str, hyp: Hypotheses) -> str:
        return f"*SOLID SECTION, ELSET={elset}, MATERIAL={self.name}, ORIENTATION={self._orient}"

    def el_file_fields(self, *, elastic_only: bool = False) -> str:
        # tensile-tear needs total strain (E); stress (S) kept for diagnostics/triaxiality
        return "E, S"

    @property
    def eps_f(self) -> float:
        return self.eps_tear0

    @property
    def yield_strain(self) -> float:
        return self.params["sigma_y"] / self.params["E_MD"]

    def damage(self, frame: dict, hyp: Hypotheses, *, xyz, conn, w_lig: float) -> float:
        """Membrane tear margin (``>=1`` => tear). ``conn``/``w_lig`` are unused: the column
        average already restricts to the deforming ligament by construction."""
        return membrane_tear_from_frame(frame, xyz, eps_tear0=self.eps_tear0, k=self.k, q=99.0)
