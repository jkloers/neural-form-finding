"""Golden tests for the material/hypothesis deck refactor (nff.rve.materials).

Phase 1 relocated the material-specific CalculiX cards out of the deck writer and behind
the ``Material`` interface. These tests pin the emitted steel deck byte-for-byte so the
refactor is provably physics-neutral (``ccx`` need not run — the deck text is the contract).

Run: JAX_PLATFORMS=cpu conda run -n kgnn_mac python -m pytest tests/test_rve_material_deck.py -q
"""
import numpy as np
import pytest

from nff.rve.materials import STEEL, Hypotheses, SteelJ2, coerce_material
from nff.rve.materials.base import Material

# The exact legacy blocks the inline deck writer used to emit for S235 steel.
_LEGACY_CONSTITUTIVE = (
    "*MATERIAL, NAME=STEEL\n"
    "*ELASTIC\n"
    "210000.0, 0.300\n"
    "*PLASTIC\n"
    "235.0, 0.0\n"
    "1285.0, 0.5"
)
_LEGACY_CONSTITUTIVE_ELASTIC = "*MATERIAL, NAME=STEEL\n*ELASTIC\n210000.0, 0.300"
_LEGACY_SECTION = "*SOLID SECTION, ELSET=EALL, MATERIAL=STEEL"


def test_steel_constitutive_cards_byte_identical():
    steel = SteelJ2()
    assert steel.constitutive_cards(Hypotheses()) == _LEGACY_CONSTITUTIVE
    assert steel.constitutive_cards(Hypotheses(), elastic_only=True) == _LEGACY_CONSTITUTIVE_ELASTIC


def test_steel_section_and_field_cards():
    steel = SteelJ2()
    assert steel.section_cards("EALL", Hypotheses()) == _LEGACY_SECTION
    assert steel.el_file_fields() == "E, PEEQ, S"
    assert steel.el_file_fields(elastic_only=True) == "E, S"


def test_hardening_point_matches_Et_formula():
    # second *PLASTIC point = (sigma_y + Et*0.5, 0.5); guards against a silent slope regression.
    m = dict(STEEL, sigma_y=300.0, Et=4000.0)
    cards = SteelJ2(m).constitutive_cards(Hypotheses())
    assert "2300.0, 0.5" in cards            # 300 + 4000*0.5 = 2300
    assert "300.0, 0.0" in cards


def test_coerce_material_from_dict_and_passthrough():
    m = coerce_material(STEEL)
    assert isinstance(m, SteelJ2) and m.params == STEEL
    assert coerce_material(m) is m           # a Material passes through unchanged
    assert isinstance(coerce_material("steel"), SteelJ2)


def test_coerce_material_rejects_garbage():
    with pytest.raises(TypeError):
        coerce_material(42)


def test_steel_is_a_material():
    assert isinstance(SteelJ2(), Material)


def test_steel_damage_is_the_shared_plastic_dissipation():
    """The material damage hook must be exactly the shared <PEEQ>_lig/eps_f measure."""
    from nff.rve.damage import plastic_damage
    from tests.test_rve_damage import unit_wedge_mesh

    xyz, conn = unit_wedge_mesh(w_lig=4.0)
    rng = np.random.default_rng(0)
    frame = {"PEEQ": np.abs(rng.normal(0.1, 0.05, size=len(xyz)))}
    expected = plastic_damage(frame, xyz, conn, 4.0, SteelJ2().eps_f)
    got = SteelJ2().damage(frame, Hypotheses(), xyz=xyz, conn=conn, w_lig=4.0)
    assert got == expected
    # no plastic field at all -> NaN, never a spurious 0
    assert np.isnan(SteelJ2().damage({"STRESS": np.zeros((len(xyz), 6))}, Hypotheses(),
                                     xyz=xyz, conn=conn, w_lig=4.0))


def test_full_generated_deck_contains_legacy_steel_block():
    """End-to-end: a real gmsh-built deck must carry the exact legacy material + section cards."""
    pytest.importorskip("gmsh")                   # solver-side dep; skip where unavailable
    import tempfile

    from nff.rve.ccx_solver import prepare_job
    from nff.rve.geometry import RVEParams

    with tempfile.TemporaryDirectory() as d:
        meta = prepare_job(RVEParams(w_lig=8.0), angle_deg=30.0, n_steps=2, n_through=1,
                           workdir=d)
        deck = open(meta["job"] + ".inp").read()
    assert _LEGACY_CONSTITUTIVE in deck
    assert _LEGACY_SECTION in deck
    assert "E, PEEQ, S" in deck                    # plastic field-output path
    assert deck.count("*ELEMENT, TYPE=C3D15") == 1
