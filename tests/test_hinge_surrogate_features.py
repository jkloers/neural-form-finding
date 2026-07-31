"""The surrogate's feature-parameterization contract (``stats["feat_kind"]``).

Training moved from the raw kinematics ``[a, s, theta, log w_lig, alpha]`` to the dimensionless
``[a/w_lig, s/w_lig, ...]`` the oracle actually samples in. Every checkpoint trained before that
carries no ``feat_kind`` key, and a dimensional net fed dimensionless inputs is silently wrong
rather than an error -- so the legacy default is pinned here, at the one function that decides it.
"""
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from nff.models.hinge_surrogate import (FEAT_KIND_ETA, FEAT_KIND_RAW, _features,
                                        compute_norm_stats)

_U = jnp.array([[1.7, -0.8, 0.42], [-3.1, 2.2, 0.95]])
_G = jnp.array([[8.0, 1.55], [21.0, 1.72]])
_IDENTITY = {"feat_mean": jnp.zeros(5, dtype=jnp.float64),      # standardization off, so the test
             "feat_std": jnp.ones(5, dtype=jnp.float64)}        # reads the raw layout directly


def _expected(u, g, dimensionless):
    a, s = (u[:, 0] / g[:, 0], u[:, 1] / g[:, 0]) if dimensionless else (u[:, 0], u[:, 1])
    return np.stack([a, s, u[:, 2], np.log(g[:, 0]), g[:, 1]], -1)


def test_absent_feat_kind_is_the_legacy_raw_layout():
    """A checkpoint predating the key must keep its dimensional features -- exactly."""
    out = np.asarray(_features(_U, _G, _IDENTITY))
    assert np.array_equal(out, _expected(np.asarray(_U), np.asarray(_G), dimensionless=False))


def test_eta_layout_divides_the_translations_by_the_ligament_width():
    stats = {**_IDENTITY, "feat_kind": FEAT_KIND_ETA}
    out = np.asarray(_features(_U, _G, stats))
    assert np.allclose(out, _expected(np.asarray(_U), np.asarray(_G), dimensionless=True))
    # the two kinds must actually differ, or the test above proves nothing
    assert not np.allclose(out, np.asarray(_features(_U, _G, _IDENTITY)))


def test_explicit_raw_matches_the_absent_default():
    stats = {**_IDENTITY, "feat_kind": FEAT_KIND_RAW}
    assert np.array_equal(np.asarray(_features(_U, _G, stats)),
                          np.asarray(_features(_U, _G, _IDENTITY)))


def test_compute_norm_stats_records_the_kind_and_standardizes_that_same_layout():
    """The stats and the features they standardize come from one definition; drift would show as a
    non-zero mean on the training features themselves."""
    rng = np.random.default_rng(0)
    n = 512
    a, s, th = rng.normal(0, 2, n), rng.normal(0, 1, n), rng.uniform(0, 1.4, n)
    w, al = rng.uniform(5, 25, n), rng.uniform(1.4, 1.9, n)
    for kind, dimensionless in ((FEAT_KIND_ETA, True), (FEAT_KIND_RAW, False)):
        stats = compute_norm_stats(a, s, th, w, al, rng.uniform(1, 100, n), feat_kind=kind)
        assert stats["feat_kind"] == kind
        u = jnp.asarray(np.stack([a, s, th], -1))
        g = jnp.asarray(np.stack([w, al], -1))
        feats = np.asarray(_features(u, g, stats))
        assert np.allclose(feats.mean(0), 0.0, atol=1e-9)
        assert np.allclose(feats.std(0), 1.0, atol=1e-9)
        # and the standardization was built on the layout the kind names
        raw = _expected(np.asarray(u), np.asarray(g), dimensionless)
        assert np.allclose(feats * np.asarray(stats["feat_std"]) + np.asarray(stats["feat_mean"]), raw)


def test_default_kind_for_new_training_is_dimensionless():
    stats = compute_norm_stats(np.zeros(4), np.zeros(4), np.ones(4), np.full(4, 9.0),
                               np.ones(4), np.ones(4))
    assert stats["feat_kind"] == FEAT_KIND_ETA
