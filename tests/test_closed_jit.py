"""The closed training step must compile ONCE and never retrace.

The step is the whole pipeline -- Stage-0 map, the Stage-2 jaxopt L-BFGS load-stepping scan, the
surrogate MLP, the loss terms and the optax update -- fused into a single XLA program. Compiling it
is expensive (1-2 min), so a silent retrace does not fail anything; it just multiplies the run's
wall time by the number of times it happens, and a 200-epoch run that recompiles per epoch looks
exactly like a slow one.

Nothing guarded this. ``tests/test_design_gradient.py`` covers jit-vs-eager gradient AGREEMENT (the
incident where ``jnp.abs`` returned an exact gradient eagerly and 176/226 NaN under ``jax.jit``),
and ``tests/test_closed_pipeline_smoke.py`` covers finiteness across two steps -- but neither
notices if every step pays a fresh compile.

Run with JAX forced to CPU (as the closed pipeline requires):
    JAX_PLATFORMS=cpu pytest tests/test_closed_jit.py -q
"""
import os

import jax
import pytest

jax.config.update("jax_enable_x64", True)

CONFIG = "data/configs/closed/rect_a4_beam_surrogate_v2.yaml"


def _build_step():
    from nff.config.experiment import load_and_parse_config
    from nff.closed.setup import (
        build_closed_initial_state, init_closed_les_params, build_surrogate_energy)
    from nff.training.trainer import create_train_step, TrainState

    cfg = load_and_parse_config(CONFIG)
    ck = getattr(cfg.hinge_model, "checkpoint", None)
    if not (ck and os.path.exists(ck)):
        pytest.skip("surrogate checkpoint not present")

    state0, _ = build_closed_initial_state(cfg)
    params, sf = init_closed_les_params(cfg)
    bond, probe, geometry_fn, _damage_fn, w_lig_logit0 = build_surrogate_energy(
        cfg, sf, state0, params)
    if w_lig_logit0 is not None:
        params = {**params, "w_lig_logit": w_lig_logit0}

    optimizer, step = create_train_step(
        state0, cfg.target, cfg.validity, cfg.physics, cfg.training,
        map_type=cfg.mapping.type, use_jit=True, load_specs=cfg.topology.get("loads", []),
        static_features=sf, bond_energy_fn=bond, hinge_probe_fn=probe,
        hinge_geometry_fn=geometry_fn)
    return step, TrainState(params=params, opt_state=optimizer.init(params),
                            rng=jax.random.PRNGKey(0))


@pytest.mark.skipif(not os.path.exists(CONFIG), reason="closed config not present")
def test_train_step_is_jitted_and_compiles_exactly_once():
    """Three steps, one compile.

    A retrace here means some part of the step changed shape, dtype or pytree structure between
    epochs -- most plausibly ``params`` gaining or losing a key (``w_lig_logit`` is added once,
    before the step is built, precisely so this stays invariant).
    """
    step, st = _build_step()
    assert hasattr(step, "_cache_size"), (
        "create_train_step(use_jit=True) did not return a jitted callable -- the whole step, "
        "including the Stage-2 solve, is meant to compile into one XLA program")

    st, _loss, _aux = step(st)
    assert step._cache_size() == 1, "the first call should produce exactly one compiled variant"

    for _ in range(2):
        st, _loss, _aux = step(st)
        assert step._cache_size() == 1, (
            f"the step RETRACED: cache grew to {step._cache_size()} variants. Every retrace pays "
            "the full pipeline compile again.")


@pytest.mark.skipif(not os.path.exists(CONFIG), reason="closed config not present")
def test_stage2_solve_is_inside_the_jit():
    """The Stage-2 L-BFGS must be compiled, not dispatched op-by-op.

    Lowering the step and finding the load-stepping ``scan`` in the compiled program is the direct
    evidence: if the solve had escaped the trace it could not appear there.
    """
    step, st = _build_step()
    text = step.lower(st).as_text()
    assert "while" in text or "scan" in text, (
        "no loop construct in the lowered step -- the Stage-2 load-stepping scan and its L-BFGS "
        "iterations are not inside the compiled program")
