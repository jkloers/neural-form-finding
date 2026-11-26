"""Gate tests for the local-probe pocket measurements."""

import numpy as np
import pytest

from nff_paper.closed_support.pocket import (angular_profile, centre_face, decay_ratios,
                               decomposition_precision, energy_decomposition,
                               load_specs, mechanism_split, pocket_shape, rigid_resultants,
                               scatter_hinge_to_tiles, scatter_load, self_equilibrate,
                               subspace_split, tile_adjacency)


PAIR = np.array([[0.0, 0.0], [1.0, 0.0]])


# ── self-equilibration ────────────────────────────────────────────────────────

def test_pinch_pair_is_already_self_equilibrated():
    """Opposed forces along the line joining two tiles carry no resultant, so nothing changes."""
    load = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    assert np.allclose(rigid_resultants(load, PAIR), 0.0, atol=1e-14)
    assert np.allclose(self_equilibrate(load, PAIR), load, atol=1e-14)


def test_shear_pair_carries_a_net_moment_until_it_is_corrected():
    """Opposed forces ACROSS the line are a couple; without the fix the gauge would react."""
    load = np.array([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])
    before = rigid_resultants(load, PAIR)
    assert np.isclose(before[0], 0.0) and np.isclose(before[1], 0.0)
    assert abs(before[2]) > 0.5                                    # a real net moment

    fixed = self_equilibrate(load, PAIR)
    assert np.allclose(rigid_resultants(fixed, PAIR), 0.0, atol=1e-13)
    # The forces survive, reduced, and the moment is paid for by a couple on each tile.
    assert fixed[0, 1] > 0.5 and np.isclose(fixed[0, 1], -fixed[1, 1])
    assert abs(fixed[0, 2]) > 0.1 and np.isclose(fixed[0, 2], fixed[1, 2])


def test_self_equilibration_nulls_every_resultant_for_arbitrary_loads():
    rng = np.random.default_rng(7)
    xy = rng.normal(size=(5, 2))
    for _ in range(10):
        fixed = self_equilibrate(rng.normal(size=(5, 3)), xy)
        assert np.allclose(rigid_resultants(fixed, xy), 0.0, atol=1e-12)


def test_self_equilibration_is_a_projection():
    """Applying it twice must change nothing, or the correction is not minimal."""
    rng = np.random.default_rng(3)
    xy, load = rng.normal(size=(4, 2)), rng.normal(size=(4, 3))
    once = self_equilibrate(load, xy)
    assert np.allclose(self_equilibrate(once, xy), once, atol=1e-13)


def test_resultants_do_not_depend_on_the_moment_origin_once_forces_balance():
    load = self_equilibrate(np.array([[0.0, 2.0, 0.0], [0.0, -2.0, 0.0]]), PAIR)
    assert np.allclose(rigid_resultants(load, PAIR, origin=np.array([9.0, -4.0])), 0.0, atol=1e-12)


def test_a_single_tile_cannot_carry_a_self_equilibrated_probe():
    with pytest.raises(ValueError, match="at least two tiles"):
        self_equilibrate(np.array([[1.0, 0.0, 0.0]]), np.array([[0.0, 0.0]]))


# ── plumbing into the pipeline ────────────────────────────────────────────────

def test_load_specs_emits_each_face_dof_pair_once():
    """A repeated (face, dof) pair overwrites downstream instead of summing."""
    specs = load_specs(np.array([[0.8, 0.0, 0.4], [-0.8, 0.0, 0.4]]), [10, 11])
    assert len(specs) == 4
    assert len({(s['face'], s['dof']) for s in specs}) == 4
    assert {s['dof'] for s in specs} == {0, 2}                     # the zero fy is dropped


def test_scatter_load_places_the_cluster_and_leaves_the_rest_alone():
    full = scatter_load(np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 0.0]]), [2, 5], n_faces=7)
    assert full.shape == (7, 3)
    assert np.allclose(full[2], [1.0, 2.0, 3.0])
    assert np.allclose(full[5], [-1.0, 0.0, 0.0])
    assert np.allclose(np.delete(full, [2, 5], axis=0), 0.0)


def test_tile_adjacency_reduces_node_bonds_to_tile_pairs():
    # 4 corners per tile: node 9 is tile 2 corner 1, node 17 is tile 4 corner 1.
    pairs = tile_adjacency(np.array([[9, 17]]), n_nodes=4)
    assert pairs.tolist() == [[2, 4]]


def test_scatter_hinge_to_tiles_conserves_the_total():
    pairs = np.array([[0, 1], [1, 2], [2, 3]])
    vals = np.array([2.0, 4.0, 6.0])
    tiles = scatter_hinge_to_tiles(vals, pairs, n_faces=4)
    assert np.isclose(tiles.sum(), vals.sum())
    assert np.allclose(tiles, [1.0, 3.0, 5.0, 3.0])


def test_centre_face_matches_the_col_times_N_plus_row_convention():
    assert centre_face(5, 5) == 2 * 5 + 2
    assert centre_face(21, 21) == 10 * 21 + 10
    with pytest.raises(ValueError, match="both dimensions odd"):
        centre_face(6, 5)


# ── mechanism versus pocket ───────────────────────────────────────────────────

def test_mechanism_split_is_exact_and_reconstructs_the_response():
    rng = np.random.default_rng(11)
    mode = rng.normal(size=40)
    perp = rng.normal(size=40)
    perp -= perp @ mode / (mode @ mode) * mode
    response = 3.0 * mode / np.linalg.norm(mode) + perp

    split = mechanism_split(response, mode)
    assert np.isclose(split.amplitude, 3.0)
    assert np.allclose(split.mechanism + split.pocket, response, atol=1e-12)
    assert np.allclose(split.pocket, perp, atol=1e-12)
    assert np.isclose(split.share, 9.0 / (9.0 + perp @ perp))


def test_a_pure_mechanism_response_leaves_no_pocket():
    mode = np.array([1.0, -1.0, 1.0, -1.0])
    split = mechanism_split(-2.5 * mode, mode)
    assert np.isclose(split.share, 1.0)
    assert np.allclose(split.pocket, 0.0, atol=1e-14)


def test_subspace_split_removes_the_whole_span_not_just_one_vector():
    """Two nearly parallel global directions: removing one leaves the other behind."""
    rng = np.random.default_rng(19)
    a = rng.normal(size=30)
    b = a + 0.05 * rng.normal(size=30)                             # nearly parallel to a
    local = rng.normal(size=30)
    modes = np.vstack([a, b])
    q, _ = np.linalg.qr(modes.T)
    local -= q @ (q.T @ local)                                     # genuinely outside the span

    response = 4.0 * a - 2.0 * b + local
    split = subspace_split(response, modes)
    assert np.allclose(split.pocket, local, atol=1e-10)
    assert np.allclose(split.removed + split.pocket, response, atol=1e-12)
    # One vector alone cannot do it.
    assert not np.allclose(mechanism_split(response, a).pocket, local, atol=1e-6)


def test_subspace_split_is_idempotent_and_orientation_free():
    rng = np.random.default_rng(23)
    modes = rng.normal(size=(3, 20))
    response = rng.normal(size=20)
    once = subspace_split(response, modes)
    assert np.allclose(subspace_split(once.pocket, modes).pocket, once.pocket, atol=1e-12)
    scaled = subspace_split(response, modes * np.array([[2.0], [-5.0], [0.1]]))
    assert np.allclose(scaled.pocket, once.pocket, atol=1e-12)
    assert np.isclose(scaled.share, once.share)


def test_mode_scaling_does_not_change_the_split():
    rng = np.random.default_rng(5)
    mode, response = rng.normal(size=12), rng.normal(size=12)
    a = mechanism_split(response, mode)
    b = mechanism_split(response, 1000.0 * mode)
    assert np.isclose(a.share, b.share)
    assert np.allclose(a.pocket, b.pocket, atol=1e-12)


# ── separating the probe's own stress from its redistribution ─────────────────

def _quadratic_energy(u, k, g):
    """A stand-in for the stored energy: quadratic in the displacement, per element."""
    return 0.5 * k * u ** 2 + g * u


def test_decomposition_is_an_exact_identity():
    rng = np.random.default_rng(31)
    k, g = rng.uniform(0.5, 4.0, 20), rng.normal(size=20)
    bg, delta = rng.normal(size=20), 0.01 * rng.normal(size=20)
    e = [_quadratic_energy(bg + s * delta, k, g) for s in (1.0, 0.0, -1.0)]
    own, redist = energy_decomposition(*e)
    assert np.allclose(own + redist, e[0] - e[1], atol=1e-15)


def test_own_scales_quadratically_and_redistribution_linearly():
    """The identity the whole magnitude claim rests on."""
    rng = np.random.default_rng(37)
    k, g = rng.uniform(0.5, 4.0, 16), rng.normal(size=16)
    bg, direction = rng.normal(size=16), rng.normal(size=16)

    def parts(mag):
        e = [_quadratic_energy(bg + s * mag * direction, k, g) for s in (1.0, 0.0, -1.0)]
        return energy_decomposition(*e)

    own1, red1 = parts(1e-4)
    own2, red2 = parts(3e-4)                                       # exactly 3x the probe
    assert np.allclose(own2, 9.0 * own1, rtol=1e-9)                # quadratic
    assert np.allclose(red2, 3.0 * red1, rtol=1e-9)                # linear


def test_both_normalised_shapes_are_magnitude_independent():
    """Four decades of probe magnitude must leave both shapes untouched."""
    rng = np.random.default_rng(41)
    k, g = rng.uniform(0.5, 4.0, 24), rng.normal(size=24)
    bg, direction = rng.normal(size=24), rng.normal(size=24)

    # Bounded below by cancellation, above by nonlinearity; this is the usable window.
    shapes = []
    for mag in (1e-4, 1e-3, 1e-2, 1e-1):
        e = [_quadratic_energy(bg + s * mag * direction, k, g) for s in (1.0, 0.0, -1.0)]
        own, redist = energy_decomposition(*e)
        shapes.append((own / np.abs(own).max(), redist / np.abs(redist).max()))
    for own, redist in shapes[1:]:
        assert np.allclose(own, shapes[0][0], atol=1e-7)
        assert np.allclose(redist, shapes[0][1], atol=1e-12)


def test_a_too_small_probe_is_flagged_as_round_off():
    """`own` is second order but differenced from first-order energies, so it cancels away."""
    rng = np.random.default_rng(47)
    k, g = rng.uniform(0.5, 4.0, 24), rng.normal(size=24)
    bg, direction = rng.normal(size=24), rng.normal(size=24)

    def precision(mag):
        e = [_quadratic_energy(bg + s * mag * direction, k, g) for s in (1.0, 0.0, -1.0)]
        return decomposition_precision(*e)

    tiny, sane = precision(1e-9), precision(1e-2)
    assert tiny["own_rel"] > 1e-3                        # nothing but round-off left
    assert sane["own_rel"] < 1e-9                        # comfortably resolved
    assert sane["redist_rel"] < tiny["redist_rel"]       # the linear part degrades far more slowly


def test_the_total_alone_does_drift_with_magnitude():
    """Why the split is needed: the sum of the two is not homogeneous."""
    rng = np.random.default_rng(43)
    k, g = rng.uniform(0.5, 4.0, 24), rng.normal(size=24)
    bg, direction = rng.normal(size=24), rng.normal(size=24)

    def total_shape(mag):
        t = (_quadratic_energy(bg + mag * direction, k, g) - _quadratic_energy(bg, k, g))
        return t / np.abs(t).max()

    assert not np.allclose(total_shape(1e-6), total_shape(1e1), atol=1e-3)


# ── decay against distance ────────────────────────────────────────────────────

def test_decay_ratios_recover_a_planted_exponential():
    dist = np.repeat(np.arange(8), 5)
    lam = 2.0
    values = 4.0 * np.exp(-dist / lam)
    prof = decay_ratios(values, dist)
    assert np.isclose(prof.decay_length, lam, rtol=1e-9)
    assert prof.r2 > 0.999999
    assert np.allclose(prof.ratios[:5], np.exp(1.0 / lam), rtol=1e-9)


def test_decay_ratios_report_the_step_to_the_first_neighbours():
    dist = np.repeat(np.arange(4), 3)
    values = np.repeat([1.0, 0.125, 0.04, 0.013], 3)
    prof = decay_ratios(values, dist)
    assert np.isclose(prof.ratios[0], 8.0)
    assert np.isclose(prof.ratios[1], 0.125 / 0.04)


def test_decay_fit_ignores_shells_outside_the_requested_range():
    """The outer shells sit on the sheet's floor and must not enter the fit."""
    dist = np.repeat(np.arange(8), 4)
    clean = 3.0 * np.exp(-dist / 2.0)
    floored = np.maximum(clean, 0.3)                               # a boundary floor
    assert np.isclose(decay_ratios(floored, dist, fit_hops=(1, 2, 3)).decay_length, 2.0, rtol=1e-6)
    assert decay_ratios(floored, dist, fit_hops=(5, 6, 7)).decay_length == float("inf")


def test_decay_ratios_survive_a_flat_field():
    dist = np.repeat(np.arange(6), 3)
    prof = decay_ratios(np.ones(dist.size), dist)
    assert prof.decay_length == float("inf")
    assert np.allclose(prof.ratios, 1.0)


# ── the shape of the pocket ───────────────────────────────────────────────────

def _grid_points(half=4.0, n=81):
    g = np.linspace(-half, half, n)
    xx, yy = np.meshgrid(g, g, indexing='ij')
    return np.column_stack([xx.ravel(), yy.ravel()])


def test_pocket_shape_recovers_a_planted_angle_and_aspect_ratio():
    xy = _grid_points()
    angle, a, b = np.deg2rad(30.0), 1.6, 0.4
    c, s = np.cos(angle), np.sin(angle)
    u = xy[:, 0] * c + xy[:, 1] * s
    v = -xy[:, 0] * s + xy[:, 1] * c
    w = np.exp(-(u ** 2 / a ** 2 + v ** 2 / b ** 2))

    shape = pocket_shape(w, xy, origin=np.zeros(2))
    assert np.isclose(shape.angle, angle, atol=1e-3)
    assert np.isclose(shape.aspect, a / b, rtol=2e-2)
    assert np.isclose(shape.semi_major, a / np.sqrt(2.0), rtol=2e-2)


def test_a_round_pocket_has_unit_aspect_and_no_direction_bias():
    xy = _grid_points()
    w = np.exp(-(xy[:, 0] ** 2 + xy[:, 1] ** 2))
    shape = pocket_shape(w, xy, origin=np.zeros(2))
    assert np.isclose(shape.aspect, 1.0, rtol=1e-6)
    assert np.allclose(shape.offset, 0.0, atol=1e-12)


def test_gyration_grows_with_the_reach_of_the_field():
    xy = _grid_points()
    near = pocket_shape(np.exp(-(xy ** 2).sum(axis=1) / 0.5), xy, np.zeros(2))
    far = pocket_shape(np.exp(-(xy ** 2).sum(axis=1) / 4.0), xy, np.zeros(2))
    assert far.gyration > near.gyration


def test_offset_reports_a_pocket_that_leans_to_one_side():
    xy = _grid_points()
    w = np.exp(-((xy[:, 0] - 1.5) ** 2 + xy[:, 1] ** 2))
    shape = pocket_shape(w, xy, origin=np.zeros(2))
    assert shape.offset[0] > 1.0 and abs(shape.offset[1]) < 1e-9


def test_a_zero_field_has_no_shape():
    xy = _grid_points(half=1.0, n=5)
    with pytest.raises(ValueError, match="identically zero"):
        pocket_shape(np.zeros(xy.shape[0]), xy, np.zeros(2))


# ── directional reach ─────────────────────────────────────────────────────────

def test_angular_profile_finds_a_planted_lobe():
    """A ring of near tiles plus a spur of far ones in one direction."""
    ring_ang = np.linspace(-np.pi, np.pi, 36, endpoint=False)
    ring = np.column_stack([np.cos(ring_ang), np.sin(ring_ang)])
    spur = np.column_stack([5.0 * np.cos(np.deg2rad([10.0, 15.0, 20.0])),
                            5.0 * np.sin(np.deg2rad([10.0, 15.0, 20.0]))])
    xy = np.vstack([ring, spur])
    w = np.ones(xy.shape[0])

    prof = angular_profile(w, xy, origin=np.zeros(2), n_sectors=12)
    best = int(np.nanargmax(prof['reach']))
    assert abs(prof['angle'][best]) < np.deg2rad(45.0)
    assert prof['reach'][best] > 2.0
    assert np.isclose(np.nanmin(prof['reach']), 1.0)


def test_angular_profile_is_flat_for_an_isotropic_field():
    ang = np.linspace(-np.pi, np.pi, 72, endpoint=False)
    xy = np.column_stack([2.0 * np.cos(ang), 2.0 * np.sin(ang)])
    prof = angular_profile(np.ones(xy.shape[0]), xy, np.zeros(2), n_sectors=12)
    assert np.allclose(prof['reach'], 2.0)
    assert (prof['n'] == 6).all()


def test_the_probe_tile_itself_does_not_enter_the_profile():
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    prof = angular_profile(np.ones(5), xy, origin=np.zeros(2), n_sectors=4)
    assert int(prof['n'].sum()) == 4
