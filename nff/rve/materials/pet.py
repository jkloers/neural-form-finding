"""PET (polyethylene terephthalate) sheet: isotropic elastic + multi-point J2 plasticity.

Calibrated to the Series-1 / kirigami tensile campaign on a real ~0.5 mm PET sheet
(``docs/physical_calibration_series1_tensile_protocol.md`` §13; batch
``data/experiments/raw/kirigami_20260723``). Structurally a :class:`SteelJ2` (isotropic
``*ELASTIC`` + ``*PLASTIC``, sharing the one plastic-dissipation damage measure), but with:

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
    E          2.5 GPa, from the COMPLIANCE-FREE video ladder (``nff.calibration.ladder``), which
               reads 1.8-2.7 GPa across the coupons. The earlier "3.02 +- 0.91 GPa" is RETIRED: it
               applied the HINGE fixture's series compliance (C = 7.77 um/N) to COUPON data, and C
               is not a machine constant -- the same coupon also fits C = 4.81, and batch slopes
               scatter +-11%, i.e. specimen seating varies by more than the correction. Low stakes
               either way: E 3.0 -> 1.2 GPa moves the hinge force ~2%, the response being
               plasticity-dominated.
    eps_f0     1.784 -- MEASURED (2026-07-27, n=1), no longer a floor. First coupon taken to fracture:
               A0 = 9.170 mm^2 (18.34 x 0.50), tear section 8.56 x 0.18 = 1.5408 mm^2, so
               eps_f = ln(A0/A_f) = 1.784. The tear section had drawn well past the natural draw
               ratio (A0/A_f = 5.95 vs lambda ~ 3.3), almost entirely by further WIDTH reduction,
               after ~26 min under plateau load -- so part of that strain is creep, and 1.784 is a
               slow-rate fracture strain.
"""
from __future__ import annotations

from nff.rve.damage import plastic_damage
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

PET = dict(E=2500.0, nu=0.40, plastic=PET_PLASTIC)   # MPa; nu still UNMEASURED


class PETIsotropic(Material):
    """Isotropic elastoplastic PET (linear elastic + multi-point J2 cold-draw hardening).

    Damage is the shared measure ``Delta = <PEEQ>_lig / eps_f`` (:mod:`nff.rve.damage`) with
    ``eps_f`` measured on our own coupon. For PET's flat cold-draw plateau that average IS the
    normalized plastic dissipation in the ligament -- the irreversibility the design loss exists to
    push down.

    **No triaxiality locus.** The Johnson-Cook form ``eps_f0 * exp(-k (eta - 1/3))`` is a metals
    construction: it models void nucleation and growth under hydrostatic tension, and its
    ``k = 1.5`` came from mild steel. Neither transfers to a cold-drawing thermoplastic, which
    fails by crazing and fibrillation. Nothing in the literature offers a PET value -- polymer
    fracture loci are calibrated per material from notched specimen sets, and are often not even
    monotonic in eta (the Lode angle matters too). Rather than ship an assumed steel constant, the
    campaign RECORDS ``<eta>`` (``nff.rve.damage.mean_triaxiality``) so the constant-``eps_f``
    choice stays audited: measured triaxiality in the fold/shear RVE runs is eta ~ 0.33-0.41, both
    essentially uniaxial tension, because the critical fibre of a fold is the outer surface in
    bending rather than shear.

    If PET ever needs pressure sensitivity, the honest place for it is a pressure-modified yield
    surface in the constitutive law (polymers do yield ~10-20% differently in tension and
    compression), not a fracture locus bolted onto post-processing.
    """

    name = "PET"

    def __init__(self, params: dict | None = None, *, eps_f0: float = 1.784):
        self.params = dict(PET if params is None else params)
        self._eps_f = eps_f0                       # fracture strain, MEASURED (2026-07-27, n=1)

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

    @property
    def eps_f(self) -> float:
        return self._eps_f

    @property
    def yield_strain(self) -> float:
        return self.params["plastic"][0][0] / self.params["E"]     # upper-yield peak / E

    def damage(self, frame: dict, hyp: Hypotheses, *, xyz, conn, w_lig: float) -> float:
        return plastic_damage(frame, xyz, conn, w_lig, self._eps_f)
