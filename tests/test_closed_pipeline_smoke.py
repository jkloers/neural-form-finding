"""End-to-end JIT smoke for the closed surrogate pipeline (integration; ~1-2 min compile).

Guards the exact failure class this pipeline has hit repeatedly: a JIT ``value_and_grad`` through the
Stage-2 physics + design-tracked surrogate geometry that goes NaN (custom_vjp / degenerate
normalization). If two real training steps stay finite and the loss does not diverge, the whole
threading -> solver -> backward chain is healthy.

Run with JAX forced to CPU (as the closed pipeline requires):
    JAX_PLATFORMS=cpu pytest tests/test_closed_pipeline_smoke.py -q
"""
import os
import math

import jax
import pytest

jax.config.update("jax_enable_x64", True)

CONFIG = "data/configs/closed/rect_a4_beam_surrogate_v2.yaml"


@pytest.mark.skipif(not os.path.exists(CONFIG), reason="closed config not present")
def test_closed_pipeline_trains_finite_and_non_diverging():
    from nff.config.experiment import load_and_parse_config
    from nff.scripts.closed_setup import (
        build_closed_initial_state, init_closed_les_params, build_surrogate_energy)
    from nff.training.trainer import create_train_step, TrainState

    cfg = load_and_parse_config(CONFIG)
    ck = getattr(cfg.hinge_model, "checkpoint", None)
    if not (ck and os.path.exists(ck)):
        pytest.skip("surrogate checkpoint not present")

    state0, _ = build_closed_initial_state(cfg)
    params, sf = init_closed_les_params(cfg)
    bond, stab, geometry_fn, w_lig_logit0 = build_surrogate_energy(cfg, sf, state0, params)
    if w_lig_logit0 is not None:
        params = {**params, "w_lig_logit": w_lig_logit0}

    optimizer, step = create_train_step(
        state0, cfg.target, cfg.validity, cfg.physics, cfg.training,
        map_type=cfg.mapping.type, use_jit=True, load_specs=cfg.topology.get("loads", []),
        static_features=sf, bond_energy_fn=bond, stability_fn=stab, hinge_geometry_fn=geometry_fn)
    st = TrainState(params=params, opt_state=optimizer.init(params), rng=jax.random.PRNGKey(0))

    losses = []
    for _i in range(2):
        st, loss, _aux = step(st)
        losses.append(float(loss))

    assert all(math.isfinite(l) for l in losses), losses          # JIT-NaN regression guard
    assert losses[1] <= losses[0] + 1e-6                          # learning, not diverging
