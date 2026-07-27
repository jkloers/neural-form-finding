"""Displacement-controlled Stage-2 actuation (``nff.stages.physics.displacement``).

The load-bearing guarantee is the ROUND TRIP: prescribing the motion that a force-controlled solve
produced must return the SAME equilibrium and hand back the force that produced it. If that holds,
displacement control is the same physics read from the other end, not a second model.
"""
import os

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)   # the round-trip assertions below are tight
import jax.numpy as jnp
import pytest

from nff.stages.physics.displacement import (
    parse_displacement_specs, build_displacement_control, PrescribedDisplacement)


CFG = "data/configs/closed/sheet_4x8ft_rom.yaml"
TOP_FACES = [2, 5, 8]        # the loaded (top) edge of the 3x3 sheet
LOAD_PER_TILE = 60.0         # N, as configured


# ── Spec parsing ──────────────────────────────────────────────────────────────

def test_face_list_expands_to_one_entry_per_dof():
    got = parse_displacement_specs([{'face': [2, 5, 8], 'dof': 1, 'value': 0.46}])
    assert got == (PrescribedDisplacement(2, 1, 0.46),
                   PrescribedDisplacement(5, 1, 0.46),
                   PrescribedDisplacement(8, 1, 0.46))


def test_scalar_face_and_degrees():
    (got,) = parse_displacement_specs([{'face': 4, 'dof': 2, 'value_deg': 30.0}])
    assert got.face == 4 and got.dof == 2
    assert got.value == pytest.approx(np.pi / 6)


def test_empty_specs_are_force_control():
    assert parse_displacement_specs(None) == ()
    assert build_displacement_control([], np.zeros((0, 2), dtype=np.int32)) is None


@pytest.mark.parametrize("spec, msg", [
    ({'face': 1, 'dof': 3, 'value': 0.1}, "dof must be 0, 1 or 2"),
    ({'face': 1, 'dof': 1, 'value': 0.1, 'value_deg': 5.0}, "exactly one of"),
    ({'face': 1, 'dof': 1}, "exactly one of"),
    ({'face': 1, 'dof': 1, 'value_deg': 5.0}, "'value_deg' applies to dof 2"),
])
def test_malformed_specs_raise(spec, msg):
    with pytest.raises(ValueError, match=msg):
        parse_displacement_specs([spec])


def test_duplicate_face_dof_raises():
    with pytest.raises(ValueError, match="prescribed twice"):
        parse_displacement_specs([{'face': [2, 2], 'dof': 1, 'value': 0.4}])


# ── Constraint-set assembly ───────────────────────────────────────────────────

def test_clamped_rows_come_first_and_prescribed_ramp():
    """Row order is load-bearing: build_constrained_kinematics scatters values in this order."""
    clamped = np.array([[0, 1], [3, 0]], dtype=np.int32)
    dc = build_displacement_control([{'face': 2, 'dof': 1, 'value': 0.5}], clamped)

    np.testing.assert_array_equal(dc.constrained_face_DOF_pairs,
                                  np.array([[0, 1], [3, 0], [2, 1]], dtype=np.int32))
    np.testing.assert_allclose(np.asarray(dc.constrained_DOFs_fn(1.0)), [0.0, 0.0, 0.5])
    np.testing.assert_allclose(np.asarray(dc.constrained_DOFs_fn(0.5)), [0.0, 0.0, 0.25])
    np.testing.assert_allclose(np.asarray(dc.constrained_DOFs_fn(0.0)), [0.0, 0.0, 0.0])


def test_prescribing_a_clamped_dof_raises():
    clamped = np.array([[2, 1]], dtype=np.int32)
    with pytest.raises(ValueError, match="both clamped and prescribed"):
        build_displacement_control([{'face': 2, 'dof': 1, 'value': 0.5}], clamped)


def test_prescribing_a_loaded_dof_raises():
    """A force on a constrained DOF does no work — build_loading would silently drop it."""
    with pytest.raises(ValueError, match="external load and a prescribed displacement"):
        build_displacement_control([{'face': 2, 'dof': 1, 'value': 0.5}],
                                   np.zeros((0, 2), dtype=np.int32),
                                   loaded_face_DOF_pairs=np.array([[2, 1]], dtype=np.int32))


# ── The round trip, on the 4 ft x 8 ft sheet ──────────────────────────────────

@pytest.fixture(scope="module")
def sheet():
    """Untrained 4x8ft ROM sheet: force-controlled deploy + the pieces to re-run it."""
    if not os.path.exists(CFG):
        pytest.skip(f"{CFG} not present (data/ is gitignored)")
    import dataclasses
    from nff.config.experiment import load_and_parse_config
    from nff.closed.setup import build_closed_initial_state, init_closed_les_params
    from nff.stages.pipeline import forward_pipeline

    config = load_and_parse_config(CFG)
    state, _ = build_closed_initial_state(config)
    params, sf = init_closed_les_params(config)

    def deploy(st, physics_cfg, load_specs):
        return forward_pipeline(st, config.target, config.validity, physics_cfg,
                                map_type=config.mapping.type, map_params=params,
                                static_features=sf, load_specs=load_specs)

    forced = deploy(state, config.physics, config.topology.get('loads', []))
    return config, state, deploy, forced, dataclasses.replace


def test_force_control_reports_no_reactions(sheet):
    """Backward compatibility: nothing prescribed -> nothing changes, no reaction report."""
    _, _, _, forced, _ = sheet
    assert forced['reactions'] is None


def test_prescribing_the_force_solution_reproduces_it_and_returns_the_force(sheet):
    """The round trip. Impose the dy that 60 N/tile produced; expect the same shape and 60 N back."""
    config, state, deploy, forced, replace = sheet
    forced_disp = np.asarray(forced['solution'].fields[-1])
    imposed = forced_disp[TOP_FACES, 1]                     # per-face dy at full deployment

    # Same problem, actuated the other way: drop the loads from the state (they live there, not in
    # load_specs) and prescribe the measured motion instead.
    unloaded = state._replace(loaded_face_DOF_pairs=np.zeros((0, 2), dtype=np.int32),
                              load_values=jnp.zeros(0, dtype=float))
    physics_cfg = replace(config.physics, prescribed_displacements=tuple(
        {'face': int(f), 'dof': 1, 'value': float(u)} for f, u in zip(TOP_FACES, imposed)))
    pulled = deploy(unloaded, physics_cfg, [])

    # 1. The prescribed DOFs hold exactly the imposed values.
    pulled_disp = np.asarray(pulled['solution'].fields[-1])
    np.testing.assert_allclose(pulled_disp[TOP_FACES, 1], imposed, rtol=0, atol=1e-12)

    # 2. Every other DOF relaxes to the SAME equilibrium — the two actuations are one physics.
    np.testing.assert_allclose(pulled_disp, forced_disp, rtol=1e-4, atol=1e-6)

    # 3. The reaction hands the applied force back.
    rr = pulled['reactions']
    np.testing.assert_array_equal(np.asarray(rr.face_DOF_pairs),
                                  np.array([[f, 1] for f in TOP_FACES], dtype=np.int32))
    np.testing.assert_allclose(np.asarray(rr.values)[-1], LOAD_PER_TILE, rtol=2e-3)
    assert np.asarray(rr.values).shape == (config.physics.num_load_steps, len(TOP_FACES))


def test_reaction_grows_monotonically_along_the_ramp(sheet):
    """The step history is a force-displacement curve: pulling further costs more."""
    config, state, deploy, _, replace = sheet
    unloaded = state._replace(loaded_face_DOF_pairs=np.zeros((0, 2), dtype=np.int32),
                              load_values=jnp.zeros(0, dtype=float))
    physics_cfg = replace(config.physics, prescribed_displacements=(
        {'face': TOP_FACES, 'dof': 1, 'value': 0.46},))
    total = np.asarray(deploy(unloaded, physics_cfg, [])['reactions'].values).sum(axis=1)
    assert np.all(np.diff(total) > 0.0)


def test_reactions_are_a_real_force_field_in_global_equilibrium(sheet):
    """With no external load, every reaction (clamps + prescribed) must sum to zero.

    This is the check that the readout is a FORCE and not just a derivative: the crosshead pushes
    the sheet exactly as hard as the clamp pushes back.
    """
    config, state, deploy, _, replace = sheet
    from nff.stages.physics.energy import build_potential_energy
    from nff.stages.physics.params import ReferenceGeometry, build_control_params
    from nff.stages.physics.displacement import compute_reaction_forces

    unloaded = state._replace(loaded_face_DOF_pairs=np.zeros((0, 2), dtype=np.int32),
                              load_values=jnp.zeros(0, dtype=float))
    physics_cfg = replace(config.physics, prescribed_displacements=(
        {'face': TOP_FACES, 'dof': 1, 'value': 0.46},))
    res = deploy(unloaded, physics_cfg, [])
    vs, disp = res['valid_state'], res['solution'].fields[-1]

    geometry = ReferenceGeometry.from_centroidal_state(vs)
    energy_fn = build_potential_energy(bond_connectivity=geometry.bond_connectivity,
                                       linearized_strains=physics_cfg.linearized_strains,
                                       use_contact=physics_cfg.use_contact)
    control_params = build_control_params(
        geometry=geometry, k_stretch=vs.k_stretch, k_shear=vs.k_shear, k_rot=vs.k_rot,
        density=vs.density, k_contact=physics_cfg.k_contact, min_angle=physics_cfg.min_angle,
        cutoff_angle=physics_cfg.cutoff_angle, use_contact=physics_cfg.use_contact)

    clamped = np.asarray(state.constrained_face_DOF_pairs, dtype=np.int32).reshape(-1, 2)
    prescribed = np.asarray(res['reactions'].face_DOF_pairs, dtype=np.int32)
    every = np.vstack([clamped, prescribed])
    R = np.asarray(compute_reaction_forces(energy_fn, disp, control_params, every))

    for dof in (0, 1):
        assert abs(R[every[:, 1] == dof].sum()) < 1e-3, f"dof {dof} reactions do not balance"


def test_a_jitted_training_step_runs_under_displacement_control():
    """Regression: the imposed values must close over as a compile-time constant.

    Built with `jnp` inside the trace they are a TRACER, and jaxopt's custom_vjp cannot capture a
    closed-over tracer — the whole training step failed to lower. NumPy closes over as an XLA
    constant instead. Only a real jitted `value_and_grad` step exercises this path.
    """
    if not os.path.exists(CFG):
        pytest.skip(f"{CFG} not present (data/ is gitignored)")
    import dataclasses
    from nff.config.experiment import load_and_parse_config
    from nff.closed.setup import build_closed_initial_state, init_closed_les_params
    from nff.training.trainer import create_train_step, TrainState

    config = load_and_parse_config(CFG)
    state, _ = build_closed_initial_state(config)
    state = state._replace(loaded_face_DOF_pairs=np.zeros((0, 2), dtype=np.int32),
                           load_values=jnp.zeros(0, dtype=float))
    params, sf = init_closed_les_params(config)
    physics_cfg = dataclasses.replace(config.physics, prescribed_displacements=(
        {'face': TOP_FACES, 'dof': 1, 'value': 0.46},))

    optimizer, step = create_train_step(
        state, config.target, config.validity, physics_cfg, config.training,
        map_type=config.mapping.type, use_jit=True, load_specs=[], static_features=sf)
    ts = TrainState(params=params, opt_state=optimizer.init(params), rng=jax.random.PRNGKey(0))
    ts, loss, _ = step(ts)

    assert np.isfinite(float(loss))
    # The design actually moved: gradients reached map_params through the prescribed-motion solve.
    assert not np.allclose(np.asarray(ts.params['z']), np.asarray(params['z']))


def test_updated_lagrangian_is_rejected(sheet):
    """UL maps each increment back with the total t — an imposed value would be inconsistent."""
    config, state, deploy, _, replace = sheet
    unloaded = state._replace(loaded_face_DOF_pairs=np.zeros((0, 2), dtype=np.int32),
                              load_values=jnp.zeros(0, dtype=float))
    physics_cfg = replace(config.physics, updated_lagrangian=True, prescribed_displacements=(
        {'face': TOP_FACES, 'dof': 1, 'value': 0.46},))
    with pytest.raises(NotImplementedError, match="updated_lagrangian"):
        deploy(unloaded, physics_cfg, [])
