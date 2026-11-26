"""The hinge-energy switch returns a HingeModel of the kind the config asks for."""
import os

import jax
import pytest

jax.config.update("jax_enable_x64", True)

ROM = "data/configs/closed/quickstart_4x4_rom.yaml"
SURROGATE = "data/configs/closed/circle_4x4_surrogate.yaml"


def _build(path):
    from nff.config.experiment import load_and_parse_config
    from nff.closed.setup import build_closed_initial_state, init_closed_les_params
    from nff.hinge import build_hinge_model

    cfg = load_and_parse_config(path)
    state, _ = build_closed_initial_state(cfg)
    params, sf = init_closed_les_params(cfg)
    return cfg, build_hinge_model(cfg, sf, state, params)


def test_rom_injects_no_bond_energy():
    from nff.hinge import HingeModel

    _, hinge = _build(ROM)
    assert isinstance(hinge, HingeModel) and hinge.kind == "rom"
    assert hinge.bond_energy_fn is None              # Stage 2 falls back to the springs
    bond_energy, *_ = hinge                          # still unpacks as the historical 5-tuple
    assert bond_energy is None and len(tuple(hinge)) == 5


def test_surrogate_injects_a_bond_energy_and_tracks_geometry():
    from nff.config.experiment import load_and_parse_config

    ck = load_and_parse_config(SURROGATE).hinge_model.checkpoint
    if not os.path.exists(ck):
        pytest.skip(f"{ck} not present (data/ is gitignored)")
    _, hinge = _build(SURROGATE)
    assert hinge.kind == "surrogate"
    assert callable(hinge.bond_energy_fn) and callable(hinge.geometry_fn)
    assert callable(hinge.probe_fn) and callable(hinge.damage_fn)
