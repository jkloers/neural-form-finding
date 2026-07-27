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

from nff.rve.damage import membrane_tear_from_frame
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


class PaperOrthotropic(Material):
    """Orthotropic-elastic 80 gsm copy paper with a tensile-tear failure criterion (H6)."""

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

    def failure(self, frame: dict, hyp: Hypotheses, *, coords=None, q: float = 99.0) -> float:
        return membrane_tear_from_frame(frame, coords, eps_tear0=self.eps_tear0, k=self.k, q=q)
