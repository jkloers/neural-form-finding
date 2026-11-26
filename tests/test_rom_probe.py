"""The ROM's kinematic hinge probe (``nff.closed.nff.hinge.rom.build_rom_probe_fn``).

Axial compression is not a surrogate prediction: ``eta_a = a / w_lig`` is read off the corotated
local hinge frame, so it is available to the linear-spring model too. Before this probe existed the
ROM branch returned ``(None,)*5`` and ``loss_weights.compression`` was silently skipped whatever its
value, which is the bug these tests exist to keep fixed.

Two guarantees, both load-bearing:

1. The probe and the offline path extractor reduce the SAME deployment to the SAME coordinates.
   If they drift, in-loop compression and the reported paths describe different sheets.
2. Building the probe does not perturb the solve. The probe is a read-out, not a physics change,
   so every ROM run predating it must stay reproducible to the bit.
"""
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)   # the exact-agreement assertions below need float64
import pytest

from nff.config.experiment import load_and_parse_config
from nff.closed.setup import build_closed_initial_state, init_closed_les_params
from nff.hinge import build_hinge_model
from nff.hinge.paths import build_hinge_geometry, extract_hinge_paths, resolve_length_scale
from nff.stages.pipeline import forward_pipeline


CFG = "data/configs/closed/sheet_4x8ft_rom.yaml"


def _deploy(compression):
    """Deploy the untrained ROM sheet with the probe on (compression != 0) or off."""
    c = load_and_parse_config(CFG)
    object.__setattr__(c.training.loss_weights, 'compression', float(compression))
    state, _ = build_closed_initial_state(c)
    c.topology['init_noise'] = 0.0
    c.topology['init_seed'] = 0
    params, static_features = init_closed_les_params(c)
    bond_energy, probe_fn, geometry_fn, _, _ = build_hinge_model(c, static_features, state, params)
    results = forward_pipeline(
        state, c.target, c.validity, c.physics, map_type=c.mapping.type, map_params=params,
        static_features=static_features, load_specs=c.topology['loads'],
        bond_energy_fn=bond_energy,
        hinge_geometry=(geometry_fn(params)
                        if (geometry_fn is not None and bond_energy is not None) else None))
    return c, state, static_features, params, results, bond_energy, probe_fn, geometry_fn


@pytest.fixture(scope="module")
def probe_on():
    return _deploy(1.0)


def test_rom_branch_builds_no_probe_unless_a_term_asks_for_one():
    """Gating on the weight is what keeps every pre-existing ROM config untouched."""
    *_, bond_energy, probe_fn, geometry_fn = _deploy(0.0)
    assert bond_energy is None and probe_fn is None and geometry_fn is None


def test_rom_probe_is_built_when_compression_is_weighted(probe_on):
    _, _, _, _, _, bond_energy, probe_fn, geometry_fn = probe_on
    assert bond_energy is None, "the ROM must not acquire a learned bond energy"
    assert probe_fn is not None and geometry_fn is not None


def test_rom_probe_emits_only_kinematic_keys(probe_on):
    """``D`` and ``ood`` have no linear-spring counterpart; emitting zeros would be a silent lie.

    ``nff/training/loss.py`` disables a term whose key is absent, so this omission IS the mechanism
    that keeps ``damage`` and ``ood`` off under the ROM.
    """
    c, state, sf, params, results, _, probe_fn, geometry_fn = probe_on
    out = probe_fn(results['solution'].fields, results['valid_state'].centroid_node_vectors,
                   geometry_fn(params), results['reference_bond_vectors'])
    assert set(out) == {'eta_a', 'eta_s', 'theta'}


def test_rom_probe_matches_the_offline_path_extractor(probe_on):
    """The in-loop reduction and the reported paths must be the same numbers."""
    c, state, sf, params, results, _, probe_fn, geometry_fn = probe_on
    out = probe_fn(results['solution'].fields, results['valid_state'].centroid_node_vectors,
                   geometry_fn(params), results['reference_bond_vectors'])
    paths = extract_hinge_paths(results['solution'], results['valid_state'],
                                build_hinge_geometry(c, sf, state, params),
                                resolve_length_scale(c), results['reference_bond_vectors'])
    for key, col in (('eta_a', 0), ('eta_s', 1), ('theta', 2)):
        probe = np.asarray(out[key])
        # the extractor prepends the undeformed state as step 0; the probe only sees solved steps
        extracted = np.asarray(paths.eta[1:, :, col])
        assert probe.shape == extracted.shape
        np.testing.assert_allclose(probe, extracted, rtol=0, atol=1e-9)


def test_building_the_probe_does_not_change_the_solve(probe_on):
    """A read-out must not move the physics -- bit-identical, not merely close.

    The ROM's ``ligament_energy_linearized`` takes ``hinge_geometry`` as an ignored ``**kwargs``
    entry, so threading it in would be numerically inert anyway; the loss keeps it out of
    ``control_params`` regardless, and this pins that down.
    """
    *_, results_off, _, _, _ = _deploy(0.0)
    fields_on = np.asarray(probe_on[4]['solution'].fields)
    fields_off = np.asarray(results_off['solution'].fields)
    assert np.array_equal(fields_off, fields_on)


def test_surrogate_only_weights_raise_under_the_rom():
    """Silently ignoring a requested term is how the compression bug survived; refuse instead."""
    c = load_and_parse_config(CFG)
    object.__setattr__(c.training.loss_weights, 'compression', 1.0)
    object.__setattr__(c.training.loss_weights, 'damage', 1.0)
    state, _ = build_closed_initial_state(c)
    params, static_features = init_closed_les_params(c)
    with pytest.raises(ValueError, match="damage"):
        build_hinge_model(c, static_features, state, params)
