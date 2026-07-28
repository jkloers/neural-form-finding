"""Tests for the paper RVE material (nff.rve.materials.paper).

Phase 2: orthotropic-elastic 80 gsm copy paper with a tensile-tear failure criterion. The
deck cards are pinned (no ccx needed) and the tear margin is checked against the strain math.

Run: JAX_PLATFORMS=cpu conda run -n kgnn_mac python -m pytest tests/test_rve_material_paper.py -q
"""
import numpy as np
import pytest

from nff.rve.materials import (PAPER_80GSM, Hypotheses, Material, PaperOrthotropic,
                               coerce_material, get_material)


def test_paper_is_a_registered_material():
    assert isinstance(PaperOrthotropic(), Material)
    assert isinstance(get_material("paper"), PaperOrthotropic)
    assert isinstance(coerce_material("paper"), PaperOrthotropic)


def test_paper_elastic_deck_is_orthotropic_engineering_constants():
    cards = PaperOrthotropic().constitutive_cards(Hypotheses())
    assert "*MATERIAL, NAME=PAPER" in cards
    assert "*ELASTIC, TYPE=ENGINEERING CONSTANTS" in cards
    # first data line: E_MD, E_CD, E_ZD, nu12, nu13, nu23, G12, G13  (8 values)
    assert "6800.0, 3400.0, 40.0, 0.300, 0.010, 0.010, 2000.0, 20.0" in cards
    # second data line: G23, temperature
    assert "20.0, 0.0" in cards
    # *ORIENTATION binds MD to the material x-axis
    assert "*ORIENTATION, NAME=ORIENT_MD, SYSTEM=RECTANGULAR" in cards


def test_paper_default_is_elastic_no_plastic():
    """Native default (crease off) must NOT emit *PLASTIC (the ortho+J2 combo is the caveat)."""
    assert "*PLASTIC" not in PaperOrthotropic().constitutive_cards(Hypotheses())


def test_paper_crease_softening_emits_plastic_block():
    cards = PaperOrthotropic().constitutive_cards(Hypotheses(crease_softening=True))
    assert "*PLASTIC" in cards
    assert "20.0, 0.0" in cards          # crease-onset yield
    assert "35.0, 0.0170" in cards       # tensile strength at plastic strain 0.017


def test_orientation_rotates_MD():
    unrot = PaperOrthotropic().constitutive_cards(Hypotheses(orientation_deg=0.0))
    assert "1.000000, 0.000000, 0.0, 0.000000, 1.000000, 0.0" in unrot   # identity, no -0.0
    rot = PaperOrthotropic().constitutive_cards(Hypotheses(orientation_deg=90.0))
    assert "0.000000, 1.000000, 0.0, -1.000000, 0.000000, 0.0" in rot     # MD -> +y


def test_paper_section_references_orientation():
    sec = PaperOrthotropic().section_cards("EALL", Hypotheses())
    assert sec == "*SOLID SECTION, ELSET=EALL, MATERIAL=PAPER, ORIENTATION=ORIENT_MD"


def test_paper_field_output_is_strain_and_stress():
    assert PaperOrthotropic().el_file_fields() == "E, S"   # tensile-tear needs strain, not PEEQ


def test_paper_tear_failure_uniaxial_no_stress():
    # strain-only frame (no stress) -> locus collapses to eps_tear0; strain 2x eps_tear0 -> D=2
    eps = 2.0 * PAPER_80GSM["eps_tear0"]
    frame = {"TOSTRAIN": np.tile([eps, 0.0, 0.0, 0.0, 0.0, 0.0], (30, 1))}
    D = PaperOrthotropic().damage(frame, Hypotheses(), xyz=None, conn=None, w_lig=0.0)     # coords=None -> surface, no stress
    assert D == pytest.approx(2.0, rel=1e-6)
    # no strain field -> NaN (never a spurious pass/fail)
    assert np.isnan(PaperOrthotropic().damage({"STRESS": np.zeros((3, 6))}, Hypotheses(), xyz=None, conn=None, w_lig=0.0))


def test_paper_triaxiality_relief_in_shear():
    # same principal strain, but a shear-dominated stress (eta<1/3) tolerates MORE -> lower D
    eps = 0.04
    E = np.tile([eps, 0.0, 0.0, 0.0, 0.0, 0.0], (20, 1))
    S_uni = np.tile([100.0, 0.0, 0.0, 0.0, 0.0, 0.0], (20, 1))     # uniaxial tension, eta=+1/3
    S_shear = np.tile([0.0, 0.0, 0.0, 80.0, 0.0, 0.0], (20, 1))    # pure shear, eta~0
    D_uni = PaperOrthotropic().damage({"TOSTRAIN": E, "STRESS": S_uni}, Hypotheses(), xyz=None, conn=None, w_lig=0.0)
    D_shear = PaperOrthotropic().damage({"TOSTRAIN": E, "STRESS": S_shear}, Hypotheses(), xyz=None, conn=None, w_lig=0.0)
    assert D_uni == pytest.approx(eps / PAPER_80GSM["eps_tear0"], rel=1e-6)   # eta=1/3 -> eps_tear0
    assert D_shear < D_uni                                          # shear is more tear-tolerant


def test_paper_membrane_cancels_bending_gradient():
    # a pure bending state (tension top / compression bottom, zero mid-plane) -> membrane D ~ 0,
    # far below the surface D that a per-node criterion would report.
    coords = np.array([[0.0, -8.0, z] for z in (0.0, 0.05, 0.10)] * 4, float)
    strain = []
    for _ in range(4):
        strain += [[+0.06, 0, 0, 0, 0, 0], [0.0, 0, 0, 0, 0, 0], [-0.06, 0, 0, 0, 0, 0]]
    frame = {"TOSTRAIN": np.array(strain, float)}
    D_mem = PaperOrthotropic().damage(frame, Hypotheses(), xyz=coords, conn=None, w_lig=0.0)
    D_surf = PaperOrthotropic().damage(frame, Hypotheses(), xyz=None, conn=None, w_lig=0.0)        # no coords -> surface
    assert D_mem < 1e-6              # membrane strain averages to ~0 through the thickness
    assert D_surf > 1.0             # the surface sees the full 6% tension
