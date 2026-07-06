"""Regression tests for the surrogate's feature normalization robustness.

A zero-variance training feature (e.g. a spine-only dataset with a=s=0) produced feat_std ~1e-8, so
`(raw-mean)/std` had a ~1e8 gradient that overflowed to NaN once that feature was differentiated
(the Phase 2b design->(a,s)->energy path) under XLA fusion. The fix follows sklearn's StandardScaler:
a constant feature gets scale 1.0. These tests pin that behaviour at both the compute and load sites.
"""
import os
import numpy as np
import jax, jax.numpy as jnp
import pytest

from nff.models.hinge_surrogate import _robust_std, _features, load_hinge_surrogate


def test_robust_std_constant_feature_gets_unit_scale():
    out = np.asarray(_robust_std(np.array([0.0, 1e-9, 0.5, 2.0])))
    assert out[0] == 1.0 and out[1] == 1.0          # (near-)constant -> scale 1, never 1/eps
    assert out[2] == 0.5 and out[3] == 2.0          # informative features untouched


def test_features_jit_grad_finite_with_degenerate_feature():
    """The exact failure mode: a/s stds were ~0. After the floor the JIT gradient must be finite."""
    std = jnp.asarray(_robust_std(np.array([0.0, 0.0, 0.2, 0.9, 0.6])))   # a, s degenerate
    stats = {"feat_mean": jnp.zeros(5, dtype=jnp.float64), "feat_std": std}
    f = lambda u: jnp.sum(_features(u[None], jnp.array([[5.0, 1.5]]), stats) ** 2)
    g = jax.jit(jax.grad(f))(jnp.array([0.05, 0.03, 0.1]))
    assert np.all(np.isfinite(np.asarray(g)))


def test_load_hinge_surrogate_floors_degenerate_std():
    """No degenerate std survives load -> older/under-exercised checkpoints stay numerically safe."""
    ck = "data/outputs/hinge_surrogate_v2.pkl"
    if not os.path.exists(ck):
        pytest.skip("surrogate checkpoint not present")
    _, stats, _ = load_hinge_surrogate(ck)
    assert np.all(np.asarray(stats["feat_std"]) >= 1e-6)
