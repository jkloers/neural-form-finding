"""The 1-DOF departure measures, gated on fields whose answer is known in closed form."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax                                                            # noqa: E402
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp                                               # noqa: E402

from nff.closed.one_dof import (_kabsch_2d, amplify_departure, arrival_fractions,  # noqa: E402
                                decay_length, distribution_stats, fit_configuration, fit_one_dof,
                                graph_distance, localisation_ratio, relative_rotations)
from nff.closed.shield_kinematics import (_checkerboard_signs, build_deploy_plan,  # noqa: E402
                                          deploy_tiles)
from nff.topology.closed_builder_jax import (boundary_flat_from_logits,  # noqa: E402
                                             build_boundary_edges, build_deploy_structure,
                                             solve_cut_vertices_jax)


def _grid_hinges(M, N):
    """Four-neighbour hinge list on an M x N tile grid, face id = i * N + j."""
    pairs = []
    for i in range(M):
        for j in range(N):
            k = i * N + j
            if i + 1 < M:
                pairs.append([k, (i + 1) * N + j])
            if j + 1 < N:
                pairs.append([k, i * N + j + 1])
    return np.asarray(pairs, dtype=np.int64)


# ── the angle fit ─────────────────────────────────────────────────────────────

def test_fit_recovers_an_exact_one_dof_field():
    M, N = 6, 4
    sigma = _checkerboard_signs(M, N).astype(float)
    psi, swing = -0.31, 0.02
    theta = swing + sigma * psi
    fit = fit_one_dof(theta, sigma)
    assert np.isclose(fit.psi, psi)
    assert np.isclose(fit.swing, swing)
    assert fit.rms < 1e-15
    assert np.isclose(fit.r2, 1.0)


def test_fit_separates_swing_from_psi():
    """A pure rigid rotation must land entirely in swing, leaving psi at zero."""
    sigma = _checkerboard_signs(6, 4).astype(float)
    fit = fit_one_dof(np.full(sigma.size, 0.07), sigma)
    assert np.isclose(fit.swing, 0.07)
    assert abs(fit.psi) < 1e-15


def test_fit_handles_an_unbalanced_checkerboard():
    """An odd tile count cannot balance, and an odd sheet is what a centred probe needs."""
    sigma = _checkerboard_signs(5, 5).astype(float)
    assert int((sigma > 0).sum()) * 2 != sigma.size          # genuinely unbalanced
    psi, swing = -0.29, 0.013
    fit = fit_one_dof(swing + sigma * psi, sigma)
    assert np.isclose(fit.psi, psi)
    assert np.isclose(fit.swing, swing)
    assert fit.rms < 1e-14


def test_balanced_fit_agrees_with_the_least_squares_path():
    """The closed form is a shortcut, not a different estimator."""
    sigma = _checkerboard_signs(6, 4).astype(float)
    rng = np.random.default_rng(0)
    theta = 0.02 + sigma * -0.31 + 0.001 * rng.standard_normal(sigma.size)
    fit = fit_one_dof(theta, sigma)
    design = np.column_stack([np.ones_like(sigma), sigma])
    (swing_ls, psi_ls), *_ = np.linalg.lstsq(design, theta, rcond=None)
    assert np.isclose(fit.swing, swing_ls, rtol=0, atol=1e-15)
    assert np.isclose(fit.psi, psi_ls, rtol=0, atol=1e-15)


def test_fit_rejects_non_unit_signs():
    with pytest.raises(ValueError, match=r"\+-1"):
        fit_one_dof(np.zeros(4), np.array([1.0, 0.0, -1.0, 1.0]))


def test_residual_is_the_non_one_dof_part():
    sigma = _checkerboard_signs(6, 4).astype(float)
    extra = np.zeros(sigma.size)
    extra[7] = 0.01
    fit = fit_one_dof(sigma * -0.3 + extra, sigma)
    assert fit.r2 < 1.0
    assert np.argmax(np.abs(fit.residual)) == 7


# ── localisation ──────────────────────────────────────────────────────────────

def test_localisation_ratio_is_one_for_the_pure_mechanism():
    M, N = 6, 4
    sigma = _checkerboard_signs(M, N).astype(float)
    psi = -0.3
    theta = sigma * psi
    assert np.isclose(localisation_ratio(theta, _grid_hinges(M, N), psi), 1.0)


def test_localisation_ratio_grows_when_one_hinge_carries_the_rotation():
    M, N = 6, 4
    sigma = _checkerboard_signs(M, N).astype(float)
    theta = sigma * -0.3
    theta[0] *= 3.0
    fit = fit_one_dof(theta, sigma)
    assert localisation_ratio(theta, _grid_hinges(M, N), fit.psi) > 1.5


def test_relative_rotations_are_signed():
    rel = relative_rotations(np.array([0.0, 0.5]), np.array([[0, 1], [1, 0]]))
    assert np.allclose(rel, [0.5, -0.5])


# ── spatial structure ─────────────────────────────────────────────────────────

def test_graph_distance_is_hop_count_from_the_source():
    pairs = np.array([[0, 1], [1, 2], [2, 3]])
    d = graph_distance(pairs, 4, [0])
    assert np.array_equal(d, [0, 1, 2, 3])


def test_graph_distance_marks_unreachable_tiles():
    d = graph_distance(np.array([[0, 1]]), 3, [0])
    assert d[2] == -1


def test_decay_length_recovers_a_known_exponential():
    d = np.arange(12)
    lam = 2.5
    v = 3.0 * np.exp(-d / lam)
    out = decay_length(v, d)
    assert np.isclose(out['lam'], lam, rtol=1e-8)
    assert np.isclose(out['inv_lam'], 1.0 / lam, rtol=1e-8)
    assert out['r2'] > 0.999


def test_decay_length_reports_no_decay_for_a_flat_field():
    out = decay_length(np.full(10, 0.4), np.arange(10))
    assert out['lam'] == float('inf')
    assert abs(out['inv_lam']) < 1e-12


def test_decay_length_drops_zeros_rather_than_clipping_them():
    d = np.arange(6)
    v = np.array([1.0, 0.5, 0.25, 0.0, 0.0, 0.0])
    assert decay_length(v, d)['n'] == 3


# ── arrival ───────────────────────────────────────────────────────────────────

def test_proportional_history_gives_one_arrival_for_every_tile():
    sigma = _checkerboard_signs(4, 4).astype(float)
    lf = np.linspace(0.0, 1.0, 11)
    hist = np.outer(lf, sigma * -0.4)
    tau = arrival_fractions(hist, sigma, lf, frac=0.5)
    assert np.allclose(tau, 0.5)


def test_a_late_tile_arrives_later():
    sigma = _checkerboard_signs(4, 4).astype(float)
    lf = np.linspace(0.0, 1.0, 21)
    hist = np.outer(lf, sigma * -0.4)
    hist[:, 3] = (lf ** 4) * sigma[3] * -0.4          # this tile lags
    tau = arrival_fractions(hist, sigma, lf, frac=0.5)
    assert tau[3] > 0.7
    assert np.isclose(tau[0], 0.5)


def test_tiles_that_never_move_are_nan_not_zero():
    sigma = _checkerboard_signs(4, 4).astype(float)
    lf = np.linspace(0.0, 1.0, 11)
    hist = np.outer(lf, sigma * -0.4)
    hist[:, 5] = 0.0
    tau = arrival_fractions(hist, sigma, lf)
    assert np.isnan(tau[5])


# ── the configuration fit ─────────────────────────────────────────────────────

def test_configuration_fit_recovers_psi_from_a_rigid_deployment():
    M, N = 6, 5
    struct = build_deploy_structure(M, N)
    bnd = build_boundary_edges(struct, 1.0, 1.0)
    bflat = boundary_flat_from_logits(bnd, jnp.asarray(bnd['init_logits']))
    flat = np.asarray(solve_cut_vertices_jax(struct, bflat, 0.45)[
        np.asarray(struct['corner_pid'])])
    plan = build_deploy_plan(struct, M, N, spacing=1.0, spacing_y=1.0)
    psi_true = -0.33
    target = np.asarray(deploy_tiles(plan, flat, psi_true))
    out = fit_configuration(plan, flat, target)
    assert np.isclose(out['psi'], psi_true, atol=1e-5)
    assert out['rmse'] < 1e-7


def test_configuration_fit_is_invariant_to_a_rigid_move():
    M, N = 6, 5
    struct = build_deploy_structure(M, N)
    bnd = build_boundary_edges(struct, 1.0, 1.0)
    bflat = boundary_flat_from_logits(bnd, jnp.asarray(bnd['init_logits']))
    flat = np.asarray(solve_cut_vertices_jax(struct, bflat, 0.45)[
        np.asarray(struct['corner_pid'])])
    plan = build_deploy_plan(struct, M, N, spacing=1.0, spacing_y=1.0)
    target = np.asarray(deploy_tiles(plan, flat, -0.33))
    a = np.deg2rad(23.0)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    moved = target.reshape(-1, 2) @ R.T + np.array([4.0, -7.0])
    out = fit_configuration(plan, flat, moved.reshape(target.shape))
    assert out['rmse'] < 1e-7


def test_configuration_fit_returns_the_aligned_ideal_it_fitted():
    """The reported ``nodes`` must BE the configuration the reported rmse refers to."""
    M, N = 6, 5
    struct = build_deploy_structure(M, N)
    bnd = build_boundary_edges(struct, 1.0, 1.0)
    bflat = boundary_flat_from_logits(bnd, jnp.asarray(bnd['init_logits']))
    flat = np.asarray(solve_cut_vertices_jax(struct, bflat, 0.45)[
        np.asarray(struct['corner_pid'])])
    plan = build_deploy_plan(struct, M, N, spacing=1.0, spacing_y=1.0)
    target = np.asarray(deploy_tiles(plan, flat, -0.33))
    out = fit_configuration(plan, flat, target)
    assert out['nodes'].shape == target.shape
    err = np.linalg.norm(out['nodes'].reshape(-1, 2) - target.reshape(-1, 2), axis=1)
    assert np.isclose(float(np.sqrt(np.mean(err ** 2))), out['rmse'], rtol=0, atol=1e-12)
    assert err.max() < 1e-6


# ── exaggerating a departure ──────────────────────────────────────────────────

def _rigidly_moved_squares(n=7, seed=0):
    """``n`` unit squares, and the same squares each given its own small rigid motion."""
    rng = np.random.default_rng(seed)
    base = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    ideal = np.stack([base + np.array([2.0 * k, 0.0]) for k in range(n)])
    moved = np.empty_like(ideal)
    for k in range(n):
        a = 0.03 * rng.standard_normal()
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        c = ideal[k].mean(axis=0)
        moved[k] = (ideal[k] - c) @ R.T + c + 0.02 * rng.standard_normal(2)
    return ideal, moved


def _edge_lengths(tiles):
    closed = np.concatenate([tiles, tiles[:, :1]], axis=1)
    return np.linalg.norm(np.diff(closed, axis=1), axis=-1)


def test_amplifying_by_one_reproduces_the_measured_configuration():
    ideal, actual = _rigidly_moved_squares()
    assert np.allclose(amplify_departure(ideal, actual, 1.0), actual, rtol=0, atol=1e-13)


def test_amplifying_by_zero_gives_the_reference():
    ideal, actual = _rigidly_moved_squares()
    assert np.allclose(amplify_departure(ideal, actual, 0.0), ideal, rtol=0, atol=1e-13)


@pytest.mark.parametrize("factor", [2.0, 5.0, 20.0])
def test_amplification_keeps_every_panel_congruent(factor):
    """The point of scaling the rigid motion: an exaggerated panel is still the same panel."""
    ideal, actual = _rigidly_moved_squares()
    out = amplify_departure(ideal, actual, factor)
    assert np.abs(_edge_lengths(out) - _edge_lengths(ideal)).max() < 1e-12


def test_amplification_scales_a_pure_offset_exactly():
    """The two components scale exactly; only their VECTOR SUM is not proportional, so they are
    tested apart. Here the departure is translation alone."""
    base = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    ideal = np.stack([base, base + 2.0])
    shift = np.array([0.013, -0.021])
    out = amplify_departure(ideal, ideal + shift, 6.0)
    assert np.allclose(out - ideal, 6.0 * shift, rtol=0, atol=1e-13)


def test_amplification_scales_a_pure_rotation_angle_exactly():
    """And here it is rotation alone, read back as an angle rather than as a displacement."""
    base = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    ideal = np.stack([base, base + 2.0])
    a = 0.05
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    turned = np.stack([(t - t.mean(0)) @ R.T + t.mean(0) for t in ideal])
    out = amplify_departure(ideal, turned, 6.0)
    for k in range(ideal.shape[0]):
        Rk, _ = _kabsch_2d(ideal[k], out[k])
        assert np.isclose(float(np.arctan2(Rk[1, 0], Rk[0, 0])), 6.0 * a, rtol=0, atol=1e-13)


def test_amplification_rejects_a_shape_mismatch():
    ideal, actual = _rigidly_moved_squares()
    with pytest.raises(ValueError, match="shape mismatch"):
        amplify_departure(ideal, actual[:-1], 3.0)


# ── distribution ──────────────────────────────────────────────────────────────

def test_distribution_stats_on_a_uniform_field():
    s = distribution_stats(np.full(50, 2.0))
    assert np.isclose(s['n_eff'], 50.0)
    assert abs(s['gini']) < 1e-12
    assert abs(s['iqr']) < 1e-12


def test_distribution_stats_flag_a_single_hot_entry():
    v = np.zeros(50)
    v[0] = 1.0
    s = distribution_stats(v)
    assert np.isclose(s['n_eff'], 1.0)
    assert s['gini'] > 0.9
