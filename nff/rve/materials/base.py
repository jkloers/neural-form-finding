"""Material + modeling-hypothesis abstractions for the single-hinge RVE oracle.

The FEA oracle (``nff.rve.ccx_solver``) owns the material-AGNOSTIC machinery: mesh,
deck skeleton, solve loop, parse. Everything material-SPECIFIC — the constitutive
cards, the section card, the requested output fields, and (Phase 2) the failure
criterion — lives behind the :class:`Material` interface so a new material plugs in
without touching the deck writer, mesh, solver, or parser.

The physical modeling assumptions are made explicit as named toggles on
:class:`Hypotheses` (rather than hidden in code paths), mapping to the hinge-model
hypotheses H1-H7:

    H1  buckling_localized_at_hinge  rigid panels; all out-of-plane + inelastic action
                                     confined to the ligament (justifies the RVE)
    H2  orientation_deg             orthotropic material direction rel. to the cut
    H3  (always on)                 bending-dominated fold, NLGEOM, seeded imperfection
    H4  tc_asymmetry               tension/compression asymmetry across the fold
    H5  crease_softening           irreversible localized crease (stiffness loss)
    H6  failure_mode               "ductile_D" (metal) | "tensile_tear" (paper)
    H7  (always on)                quasi-static, fixed environment

    element_family                 "solid" (C3D15 prisms, current) | "shell" (thin-sheet)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Hypotheses:
    """The explicit modeling assumptions a hinge RVE is evaluated under.

    Defaults reproduce the legacy steel campaign (isotropic solid, no asymmetry,
    ductile-damage failure). A material implementation reads only the toggles it
    honours and ignores the rest.
    """

    buckling_localized_at_hinge: bool = True   # H1: rigid panels, seed+refine only at ligament
    element_family: str = "solid"              # "solid" (C3D15) | "shell" (S8R)
    orientation_deg: float = 0.0               # H2: material MD direction rel. to secondary cut (+x)
    tc_asymmetry: bool = False                 # H4: tension/compression asymmetry
    crease_softening: bool = False             # H5: irreversible crease (stiffness loss)
    failure_mode: str = "ductile_D"            # H6: "ductile_D" | "tensile_tear"


class Material(ABC):
    """A CalculiX material as seen by the deck writer.

    A material emits the three material-specific pieces of the input deck. Everything
    else in the deck (nodes, elements, steps, boundary conditions, node output) is
    material-agnostic and owned by the deck writer.
    """

    name: str

    @abstractmethod
    def constitutive_cards(self, hyp: Hypotheses, *, elastic_only: bool = False) -> str:
        """The ``*MATERIAL`` block (elasticity + inelasticity + any ``*ORIENTATION``).

        Returned as one multi-line string; the deck writer appends it verbatim. When
        ``elastic_only`` is set, the inelastic cards are omitted (elastic smoke path).
        """

    @abstractmethod
    def section_cards(self, elset: str, hyp: Hypotheses) -> str:
        """The ``*SOLID SECTION`` / ``*SHELL SECTION`` card binding ``elset`` to this material."""

    @abstractmethod
    def el_file_fields(self, *, elastic_only: bool = False) -> str:
        """The ``*EL FILE`` field list (e.g. ``"E, PEEQ, S"``) written per output frame."""

    @abstractmethod
    def failure(self, frame: dict, hyp: Hypotheses, *, coords=None, q: float = 99.0) -> float:
        """Scalar failure margin ``D`` for a parsed output frame (``D >= 1`` => fracture).

        The physical criterion is material-specific (H6): ductile plastic-strain damage for a
        metal, bending/triaxiality-aware tearing for paper. ``frame`` is a parsed ``.frd`` frame
        (fields per the material's :meth:`el_file_fields`); ``coords`` are the reference node
        positions, supplied when the criterion needs through-thickness (membrane) averaging.
        Returns NaN when the required fields are absent.
        """
