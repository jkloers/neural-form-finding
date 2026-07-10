"""The surrogate's defining guarantees -- W(u;g) = W_scale * ||h(u,g) - h(0,g)||^2.

These hold BY CONSTRUCTION (squared form, zero at u=0) for ANY parameters, so a random net suffices:
  * W >= 0                     (norm squared -> no spurious negative-energy minimizer),
  * W(0, g) = 0                (rigid-body motion is free),
  * dW/du(0, g) = 0            (zero internal force at rest).
The Stage-2 forward solve + IFT backward rely on all three; pin them.
"""
import numpy as np
import jax
import pytest

from nff.models.hinge_surrogate import (
    init_hinge_surrogate, compute_norm_stats, apply_hinge_energy, apply_hinge_force)


def _net_and_stats(feat_dim, seed=0):
    rng = np.random.default_rng(seed)
    n = 256
    a, s, th = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
    w_lig = rng.uniform(1.0, 20.0, n)
    alpha = rng.uniform(0.5, 2.6, n)
    W = rng.uniform(0.0, 5.0, n)
    fr = rng.uniform(0.1, 0.3, n) if feat_dim == 6 else None
    stats = compute_norm_stats(a, s, th, w_lig, alpha, W, fillet_ratio=fr)
    net = init_hinge_surrogate(jax.random.PRNGKey(seed), hidden=(16, 16), m_out=8, feat_dim=feat_dim)
    return net, stats


def _rand_ug(feat_dim, seed=1):
    rng = np.random.default_rng(seed)
    u = rng.normal(size=(64, 3)) * np.array([2.0, 1.5, 0.3])
    g_cols = [rng.uniform(1.0, 20.0, 64), rng.uniform(0.5, 2.6, 64)]
    if feat_dim == 6:
        g_cols.append(rng.uniform(0.1, 0.3, 64))
    return u, np.stack(g_cols, axis=-1)


@pytest.mark.parametrize("feat_dim", [5, 6])
def test_energy_nonnegative(feat_dim):
    net, stats = _net_and_stats(feat_dim)
    u, g = _rand_ug(feat_dim)
    W = np.asarray(apply_hinge_energy(net, u, g, stats))
    assert np.all(W >= 0.0)


@pytest.mark.parametrize("feat_dim", [5, 6])
def test_energy_and_force_vanish_at_rest(feat_dim):
    net, stats = _net_and_stats(feat_dim)
    _, g = _rand_ug(feat_dim)
    u0 = np.zeros((g.shape[0], 3))
    W0 = np.asarray(apply_hinge_energy(net, u0, g, stats))
    F0 = np.asarray(apply_hinge_force(net, u0, g, stats))
    assert np.max(np.abs(W0)) < 1e-9          # W(0, g) = 0
    assert np.max(np.abs(F0)) < 1e-8          # dW/du(0, g) = 0
