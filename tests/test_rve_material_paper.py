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


def test_paper_tear_failure_from_principal_strain():
    # uniaxial +x strain of 0.03 -> max principal 0.03; D = 0.03 / eps_tear(0.015) = 2.0
    eps = 0.03
    frame = {"TOSTRAIN": np.tile([eps, 0.0, 0.0, 0.0, 0.0, 0.0], (30, 1))}
    D = PaperOrthotropic().failure(frame, Hypotheses())
    assert D == pytest.approx(eps / PAPER_80GSM["eps_tear"], rel=1e-6)
    assert D == pytest.approx(2.0, rel=1e-6)
    # no strain field -> NaN (never a spurious pass/fail)
    assert np.isnan(PaperOrthotropic().failure({"STRESS": np.zeros((3, 6))}, Hypotheses()))
