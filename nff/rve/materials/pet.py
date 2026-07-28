"""PET (polyethylene terephthalate) sheet: isotropic elastic + multi-point J2 plasticity.

Calibrated to the Series-1 / kirigami tensile campaign on a real ~0.5 mm PET sheet
(``docs/physical_calibration_series1_tensile_protocol.md`` §13; batch
``data/experiments/raw/kirigami_20260723``). Structurally a :class:`SteelJ2` (isotropic
``*ELASTIC`` + ``*PLASTIC`` + ductile-damage failure, H6=ductile_D), but with:

* a **multi-point** ``*PLASTIC`` table (steel is bilinear) anchored on the measured yield
  and the **cold-draw** point (true stress = lambda * plateau, true strain = ln lambda);
* a large **fracture strain** ``eps_f0`` — PET cold-draws to eps_true >= 1.2 without breaking
  (the folding-hinge regime: plastic "damage" / permanent set, not fracture).

Calibration provenance (9 drawn specimens, MD+CD, 1 mm/min):
    yield      upper-yield peak 44 MPa (the flow-curve start; 0.2%-offset yield ~48.8). Isotropic.
    hardening  multi-point true-stress cold-draw curve (plateau 33.6 eng -> true 110 @ eps 1.19,
               lambda 3.28), extended to true 200 @ eps 1.784 by the 2026-07-27 run-to-break;
               see PET_PLASTIC. Validated on the w=18mm hinge (opening 319 vs 287 N exp); that
               validation is UNAFFECTED by the extension -- peak force lands at PEEQ 0.188,
               and PEEQ only passes 1.189 well past the peak, at a = 4.8 mm.
    E          3.0 GPa -- MEASURED, not assumed. Raw crosshead reads ~1.0 GPa, but the load
               train adds C = 7.77 um/N in series (calibrated against the hinge fillet-hole
               video, ``nff.calibration.compliance``). Removing it and taking the INITIAL
               tangent (1-5 MPa; PET bends away from linear within a few MPa) gives
               3.02 +- 0.91 GPa over the 10 coupons.
    eps_f0     1.784 -- MEASURED (2026-07-27, n=1), no longer a floor. First coupon taken to fracture:
               A0 = 9.170 mm^2 (18.34 x 0.50), tear section 8.56 x 0.18 = 1.5408 mm^2, so
               eps_f = ln(A0/A_f) = 1.784. The tear section had drawn well past the natural draw
               ratio (A0/A_f = 5.95 vs lambda ~ 3.3), almost entirely by further WIDTH reduction,
               after ~26 min under plateau load -- so part of that strain is creep, and 1.784 is a
               slow-rate fracture strain.
"""
from __future__ import annotations

from nff.rve.damage import damage_from_frame
from nff.rve.materials.base import Hypotheses, Material

# true stress [MPa], true plastic strain -- multi-point cold-draw flow curve rebuilt from the 9 raw
# coupon curves (2026-07-26). Neck-propagation extraction: at constant plateau force each shoulder
# point carries true stress sigma_true(eps)=sigma_plateau*exp(eps) up to eps=ln(lambda) (plastic
# incompressibility). The real post-yield DIP (~34 MPa) is flattened to the 44 MPa upper-yield peak
# because CalculiX *PLASTIC must be non-decreasing. Validated coupon-only: w=18mm hinge opening
# 319 N sim vs 287 N experiment (11%), a large improvement on the old 2-point [(48.8,0),(109,1.154)]
# straight line (379 N, 32%). See memory/project_physical_calibration.
#
# Extended past eps=1.189 on 2026-07-27 from the first run-to-break (tensile_break_20260727). Same
# constant-force construction, same measured plateau (33.6 MPa eng, reproduced to 0.1% on that
# specimen), carried out to the tear at eps = ln(A0/A_f) = 1.784 where the section carried 199 N/mm^2
# true. The four added points are an INTERPOLATION between two measured endpoints, not four
# measurements: only the plateau force and the final cross-section were observed. Before this
# extension CalculiX held 110.2 MPa flat above eps 1.189, which under-predicted anything driven past
# the natural draw ratio. Nothing in the hinge operating regime reaches that far -- a 57 deg fold
# peaks at PEEQ 0.12 -- so the extension is insurance against deep-draw excursions, not a change
# to any validated prediction.
PET_PLASTIC = [
    (44.0, 0.000), (44.0, 0.100), (44.0, 0.272), (45.3, 0.300), (55.3, 0.500),
    (67.6, 0.700), (82.5, 0.900), (100.8, 1.100), (110.2, 1.189),
    (123.3, 1.300), (143.2, 1.450), (166.4, 1.600), (200.0, 1.784),
]

PET = dict(E=3000.0, nu=0.40, plastic=PET_PLASTIC)   # MPa; E is a literature placeholder


class PETIsotropic(Material):
    """Isotropic elastoplastic PET (linear elastic + multi-point J2 cold-draw hardening).

    Failure is ``D = PEEQ / eps_f(eta)`` (H6 = ductile_D) with ``eps_f0`` measured on our own
    coupon. **The triaxiality term is switched off for PET: k = 0, so eps_f = eps_f0 everywhere.**

    The locus ``eps_f0 * exp(-k (eta - 1/3))`` is a metals construction. It models void growth
    under hydrostatic tension, and its ``k = 1.5`` came from mild steel -- neither transfers to a
    cold-drawing thermoplastic, whose failure is crazing and fibrillation. Nothing in the
    literature offers a PET value: polymer fracture loci are calibrated per material from notched
    specimen sets, and are often not even monotonic in eta (Lode angle matters too).

    Switching it off costs nothing here. Measured triaxiality in the RVE runs is eta ~ 0.33 (fold)
    to 0.41 (shear) -- both essentially uniaxial tension, because the critical fibre of a fold is
    the outer surface in bending, not shear. Across that band the exponential moves under 10%,
    less than the uncertainty on ``eps_f0`` itself (n = 1). The term was introduced for steel to
    stop a constant eps_f = 0.25 condemning shear-dominated folds; our folds are not shear
    dominated, so it never does the job it was added for.

    To turn it back on, measure it: one double-edge-notched coupon (eta ~ 0.5-0.6) run on the same
    protocol gives a second point on the locus and hence ``k``.
    """

    name = "PET"

    def __init__(self, params: dict | None = None, *, eps_f0: float = 1.784, k: float = 0.0):
        self.params = dict(PET if params is None else params)
        self.eps_f0 = eps_f0                       # fracture strain at uniaxial tension (measured, n=1)
        self.k = k                                 # triaxiality sensitivity -- OFF for PET, see class docstring

    @classmethod
    def from_dict(cls, params: dict) -> "PETIsotropic":
        return cls(params)

    def constitutive_cards(self, hyp: Hypotheses, *, elastic_only: bool = False) -> str:
        m = self.params
        lines = [f"*MATERIAL, NAME={self.name}", "*ELASTIC", f"{m['E']:.1f}, {m['nu']:.3f}"]
        if not elastic_only:
            lines.append("*PLASTIC")
            lines += [f"{sig:.1f}, {eps:.3f}" for sig, eps in m["plastic"]]
        return "\n".join(lines)

    def section_cards(self, elset: str, hyp: Hypotheses) -> str:
        return f"*SOLID SECTION, ELSET={elset}, MATERIAL={self.name}"

    def el_file_fields(self, *, elastic_only: bool = False) -> str:
        return "E, S" if elastic_only else "E, PEEQ, S"

    def failure(self, frame: dict, hyp: Hypotheses, *, coords=None, q: float = 99.0) -> float:
        return damage_from_frame(frame, eps_f0=self.eps_f0, k=self.k, q=q)
