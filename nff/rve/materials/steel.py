"""S235 mild steel: isotropic linear-elastic + bilinear von-Mises (J2) plasticity.

The original CalculiX material, relocated behind the :class:`Material` interface. The
emitted deck is byte-identical to the legacy inline cards (see
``tests/test_rve_material_deck.py``) — this file only moves the material-specific
strings out of the deck writer, it does not change the physics.
"""

from __future__ import annotations

from nff.rve.damage import damage_from_frame
from nff.rve.materials.base import Hypotheses, Material

STEEL = dict(E=210_000.0, nu=0.30, sigma_y=235.0, Et=2100.0)   # MPa; isotropic hardening Et = E/100


class SteelJ2(Material):
    """Isotropic elastoplastic S235 steel (linear elastic + bilinear J2 hardening).

    Failure is triaxiality-dependent ductile damage ``D = PEEQ / eps_f(eta)`` (H6 = ductile_D).
    """

    name = "STEEL"

    def __init__(self, params: dict | None = None, *, eps_f0: float = 0.25, k: float = 1.5):
        self.params = dict(STEEL if params is None else params)
        self.eps_f0 = eps_f0                       # fracture strain at uniaxial tension
        self.k = k                                 # triaxiality sensitivity of the fracture locus

    @classmethod
    def from_dict(cls, params: dict) -> "SteelJ2":
        return cls(params)

    def constitutive_cards(self, hyp: Hypotheses, *, elastic_only: bool = False) -> str:
        m = self.params
        lines = [f"*MATERIAL, NAME={self.name}", "*ELASTIC", f"{m['E']:.1f}, {m['nu']:.3f}"]
        if not elastic_only:
            lines += ["*PLASTIC",
                      f"{m['sigma_y']:.1f}, 0.0",
                      f"{m['sigma_y'] + m['Et'] * 0.5:.1f}, 0.5"]   # hardening slope Et
        return "\n".join(lines)

    def section_cards(self, elset: str, hyp: Hypotheses) -> str:
        return f"*SOLID SECTION, ELSET={elset}, MATERIAL={self.name}"

    def el_file_fields(self, *, elastic_only: bool = False) -> str:
        return "E, S" if elastic_only else "E, PEEQ, S"

    def failure(self, frame: dict, hyp: Hypotheses, *, coords=None, q: float = 99.0) -> float:
        return damage_from_frame(frame, eps_f0=self.eps_f0, k=self.k, q=q)
