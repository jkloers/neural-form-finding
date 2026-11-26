"""Statistics of the load path -- especially the two that silently invert their answer."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax                                                            # noqa: E402
jax.config.update("jax_enable_x64", True)

from nff.closed.force_stats import (bipartite_parity, channel_concentration,  # noqa: E402
                                    checkerboard_amplitude, cut_family_split,
                                    elongation_alternation, energy_split, lattice_autocorrelation,
                                    mode_boundary_shear, neighbour_correlation, polynomial_detrend,
                                    force_clustering, region_mask, rotation_departure,
                                    route_directness, support_band_pressure,
                                    clustered_groups, family_contiguity,
                                    motif_families, patch_labels, tile_pressure,
                                    tile_shape)
from nff.closed.load_paths import ForceField, interface_decomposition, virial_stress  # noqa: E402


# ── fixtures ──────────────────────────────────────────────────────────────────

def _grid_field(nx=6, ny=4, t_scale=1.0, pressure=None):
    """A rectangular grid of unit tiles, hinged to its four-neighbours."""
    xs, ys = np.meshgrid(np.arange(nx, dtype=float), np.arange(ny, dtype=float), indexing='ij')
    face_xy = np.column_stack([xs.ravel(), ys.ravel()])
    n_faces = face_xy.shape[0]
    index = np.arange(n_faces).reshape(nx, ny)

    pairs = []
    for i in range(nx):
        for j in range(ny):
            if i + 1 < nx:
                pairs.append((index[i, j], index[i + 1, j]))
            if j + 1 < ny:
                pairs.append((index[i, j], index[i, j + 1]))
    pairs = np.asarray(pairs, dtype=np.int32)
    n = pairs.shape[0]

    corner = np.array([[-0.4, -0.4], [0.4, -0.4], [0.4, 0.4], [-0.4, 0.4]])
    span = face_xy[pairs[:, 1]] - face_xy[pairs[:, 0]]
    axis = span / np.linalg.norm(span, axis=-1, keepdims=True)
    # `t` is the force on face j, and `axial = -(t . axis)`, so a POSITIVE `t_scale` has to point
    # from j back toward i for the interface to read as tension.
    t = -t_scale * axis
    hinge_xy = 0.5 * (face_xy[pairs[:, 0]] + face_xy[pairs[:, 1]])
    areas = np.ones(n_faces)

    sigma = (virial_stress(face_xy, areas, pairs, hinge_xy, -t, t) if pressure is None
             else np.einsum('f,ij->fij', np.asarray(pressure, dtype=float), np.eye(2)))

    return ForceField(
        face_xy=face_xy, face_nodes=face_xy[:, None, :] + corner[None], face_areas=areas,
        hinge_xy=hinge_xy, face_pairs=pairs, t_lig=t, m_lig=np.zeros((n, 2)),
        t_con=np.zeros((n, 2)), m_con=np.zeros((n, 2)), hinge_axis=axis,
        axial=-np.sum(t * axis, axis=-1),
        transverse=t[:, 0] * axis[:, 1] - t[:, 1] * axis[:, 0], energy=np.zeros(n),
        sigma=sigma, f_ext=np.zeros((n_faces, 3)), residual=np.zeros((n_faces, 3)),
        free_mask=np.ones((n_faces, 3), dtype=bool), boundary_xy=np.zeros((0, 2)), checks={})


# ── the detrending trap ───────────────────────────────────────────────────────

def test_checkerboard_is_invisible_until_the_smooth_trend_is_removed():
    """The whole point of the detrend, as a regression test.

    A field that is a strong smooth ramp PLUS a weak alternating background is the real case. On the
    raw field the ramp dominates: neighbouring tiles correlate positively and the checkerboard reads
    as absent, or even as its own opposite. Removing the ramp recovers it.
    """
    field = _grid_field(nx=8, ny=6)
    parity, is_bipartite = bipartite_parity(field)
    assert is_bipartite                                     # a four-neighbour grid is two-colourable

    ramp = 10.0 * field.face_xy[:, 0]                       # the load's doing
    values = ramp + 0.5 * parity                            # ... plus the mechanism's

    raw = neighbour_correlation(field, values)
    result = checkerboard_amplitude(field, values)

    assert raw > 0.5                                        # the ramp makes neighbours look alike
    assert result['neighbour_corr_detrended'] < -0.5        # they alternate once it is gone
    # The SIGN of the projection is arbitrary: it flips with which colour class parity calls +1.
    # Only the magnitude is meaningful, and it is what the manuscript quotes.
    assert abs(result['corr_detrended']) > 0.9              # the mode is recovered outright
    assert abs(result['corr_raw']) < abs(result['corr_detrended'])


def test_polynomial_detrend_removes_a_polynomial_exactly():
    field = _grid_field(nx=6, ny=5)
    x, y = field.face_xy[:, 0], field.face_xy[:, 1]
    smooth = 3.0 + 2.0 * x - 0.5 * y + 0.25 * x * y
    assert np.allclose(polynomial_detrend(smooth, field.face_xy), 0.0, atol=1e-9)


def test_checkerboard_amplitude_reports_zero_on_a_purely_smooth_field():
    field = _grid_field(nx=6, ny=5)
    smooth = field.face_xy[:, 0] ** 2
    result = checkerboard_amplitude(field, smooth)
    assert result['variance_share'] < 0.05


# ── the local-motif trap ──────────────────────────────────────────────────────

def test_a_local_motif_survives_only_in_the_sub_block():
    """A period-two motif confined to one region does not survive a whole-lattice statistic."""
    lattice = np.tile(np.linspace(0.0, 1.0, 12)[:, None], (1, 6))       # smooth everywhere
    lattice[:4, :] += 0.5 * ((-1.0) ** np.arange(4))[:, None]           # ... except the top block

    whole = lattice_autocorrelation(lattice)['axis_0']
    block = lattice_autocorrelation(lattice, block=(slice(0, 4), slice(None)))['axis_0']

    assert whole['lag_1'] > 0.0                                          # the trend wins globally
    assert block['lag_1'] < -0.5                                         # the motif wins locally
    assert block['lag_2'] > 0.5                                          # period two, not noise


def test_lattice_autocorrelation_reads_a_pure_period_two_field():
    lattice = np.tile(((-1.0) ** np.arange(10))[:, None], (1, 4))
    axis_0 = lattice_autocorrelation(lattice)['axis_0']
    assert axis_0['lag_1'] == pytest.approx(-1.0, abs=1e-9)
    assert axis_0['lag_2'] == pytest.approx(1.0, abs=1e-9)


# ── the rest ──────────────────────────────────────────────────────────────────

def test_tile_pressure_is_positive_under_tension():
    """Sign convention, tied to the one fixed in load_paths: positive normal stress is tension."""
    pulled = _grid_field(nx=4, ny=3, t_scale=1.0)
    assert tile_pressure(pulled).mean() > 0
    pushed = _grid_field(nx=4, ny=3, t_scale=-1.0)
    assert tile_pressure(pushed).mean() < 0


def test_effective_count_equals_the_number_of_equal_carriers():
    """N_eff of k equal values is k, and of one dominant value is one."""
    field = _grid_field(nx=4, ny=3)
    dec = interface_decomposition(field)

    dec = dict(dec)
    dec['f_n'] = np.array([1.0, 1.0, 1.0, 1.0] + [0.0] * (len(dec['f_n']) - 4))
    dec['f_t'] = np.zeros_like(dec['f_n'])
    assert channel_concentration(dec)['tension']['n_eff'] == pytest.approx(4.0)

    dec['f_n'] = np.array([1.0] + [0.0] * (len(dec['f_n']) - 1))
    assert channel_concentration(dec)['tension']['n_eff'] == pytest.approx(1.0)


def test_a_straight_tension_run_is_perfectly_direct():
    """A route along a straight line of tiles has arc length equal to its chord."""
    field = _grid_field(nx=6, ny=1, t_scale=1.0)
    dec = interface_decomposition(field)
    summary = route_directness(field, dec, threshold=0.0)
    assert summary['n_routes'] == 1
    assert summary['directness_median'] == pytest.approx(1.0, abs=1e-9)
    assert summary['n_looping'] == 0


def test_support_band_selects_the_strip_between_the_two_supports():
    """The band spans the supports and excludes the supported tiles themselves."""
    field = _grid_field(nx=5, ny=5)
    corners = [0, 4]                                    # (0,0) and (0,4): one column apart in y
    result = support_band_pressure(field, corners)
    assert result['support_faces'] == [0, 4]
    assert 0 < result['n_band'] < field.n_faces


def test_support_band_needs_two_distinct_supports():
    field = _grid_field(nx=4, ny=4)
    assert support_band_pressure(field, [3])['n_band'] == 0
    assert support_band_pressure(field, [3, 3])['n_band'] == 0


def test_tile_shape_reads_a_known_rectangle():
    """A 2-by-1 tile lying along +x has aspect ratio two and a long axis at zero degrees."""
    nodes = np.array([[[-1.0, -0.5], [1.0, -0.5], [1.0, 0.5], [-1.0, 0.5]]])
    aspect, angle = tile_shape(nodes)
    assert aspect[0] == pytest.approx(2.0)
    assert angle[0] % 180.0 == pytest.approx(0.0, abs=1e-9)


def test_tile_shape_rejects_non_quadrilaterals():
    with pytest.raises(ValueError):
        tile_shape(np.zeros((2, 3, 2)))


def test_family_contiguity_reads_regions_above_chance():
    """Two solid blocks are clustered; the same labels shuffled are not."""
    lab = np.zeros((8, 8), dtype=int)
    lab[:, 4:] = 1
    out = family_contiguity(lab)
    assert out['ratio'] > 1.5
    assert out['n_regions'] == 2


def test_family_contiguity_reads_a_checkerboard_as_BELOW_chance():
    """The trap: a period-two motif is real structure, and a plain neighbour count calls it
    nothing. It must come out below one, never near one."""
    ii, jj = np.indices((8, 8))
    out = family_contiguity((ii + jj) % 2)
    assert out['observed'] == 0.0
    assert out['ratio'] < 0.1


def test_family_contiguity_is_near_one_for_a_shuffled_field():
    rng = np.random.default_rng(0)
    lab = rng.integers(0, 3, size=(14, 9))
    assert 0.7 < family_contiguity(lab)['ratio'] < 1.3


def test_family_contiguity_rejects_a_non_lattice():
    with pytest.raises(ValueError):
        family_contiguity(np.zeros(9, dtype=int))


def test_motif_families_recovers_two_separated_blobs():
    blob = np.array([[0.0, 0.0], [0.1, -0.1], [-0.1, 0.05]])
    x = np.vstack([blob, blob + 8.0])
    labels, centroids = motif_families(x, 2)
    assert set(np.bincount(labels)) == {3}
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4] == labels[5]
    assert labels[0] != labels[3]
    # Centroids come back in the ORIGINAL units, not standardised ones.
    assert np.isclose(np.sort(centroids[:, 0])[0], blob[:, 0].mean(), atol=0.2)


def test_motif_families_is_deterministic_without_a_seed():
    """The figure must redraw identically, so the initialisation carries no randomness."""
    rng = np.random.RandomState(3)
    x = rng.randn(60, 2)
    first, _ = motif_families(x, 3)
    second, _ = motif_families(x, 3)
    assert np.array_equal(first, second)


def test_motif_families_rejects_one_family_and_a_flat_input():
    with pytest.raises(ValueError):
        motif_families(np.zeros((10, 2)), 1)
    with pytest.raises(ValueError):
        motif_families(np.zeros(10), 2)
    with pytest.raises(ValueError):
        motif_families(np.zeros((3, 2)), 4)


def test_clustered_groups_keeps_an_L_and_drops_an_isolated_pair():
    lab = np.array([[0, 0, 1, 1],
                    [0, 1, 1, 1],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1]])
    mask, groups = clustered_groups(lab, min_size=3)
    assert mask[0, 0] and mask[1, 0] and mask[0, 1]        # the L of family 0
    assert groups[0] == (1, 13)
    assert (1, 3) in groups or (0, 3) in groups
    lab_pair = np.zeros((4, 4), dtype=int)
    lab_pair[0, 0] = lab_pair[0, 1] = 1
    mask_pair, _ = clustered_groups(lab_pair, min_size=3)
    assert not mask_pair[0, 0] and not mask_pair[0, 1]


def test_clustered_groups_marks_nothing_on_a_checkerboard():
    """Alternation is not clustering: every site's neighbours are the other family."""
    ii, jj = np.indices((6, 6))
    mask, groups = clustered_groups((ii + jj) % 2, min_size=3)
    assert not mask.any()
    assert groups == []


def test_clustered_groups_rejects_a_non_lattice():
    with pytest.raises(ValueError):
        clustered_groups(np.zeros(9, dtype=int))


def test_patch_labels_separates_components_touching_only_at_a_corner():
    mask = np.zeros((8, 8), dtype=bool)
    mask[0:4, 0:4] = True
    mask[4:8, 4:8] = True
    labels = patch_labels(mask)
    assert labels.max() == 2
    assert labels[0, 0] != labels[7, 7]
    assert (labels > 0).sum() == 32


def test_elongation_alternation_separates_aligned_from_alternating():
    """Ninety degrees apart is the rotating-rectangle motif; aligned tiles are not it."""
    field = _grid_field(nx=4, ny=4)
    aligned = np.array([[[-1.0, -0.4], [1.0, -0.4], [1.0, 0.4], [-1.0, 0.4]]] * field.n_faces)
    field = field._replace(face_nodes=aligned) if hasattr(field, '_replace') else field
    object.__setattr__(field, 'face_nodes', aligned)
    assert elongation_alternation(field)['neighbour_angle_median'] == pytest.approx(0.0, abs=1e-6)


def test_cut_family_split_separates_the_two_orientations():
    """A grid cuts in two orthogonal families and the split must find both."""
    field = _grid_field(nx=5, ny=5)
    dec = interface_decomposition(field)
    split = cut_family_split(field, dec)
    assert split['a']['n'] > 0 and split['b']['n'] > 0
    assert split['a']['n'] + split['b']['n'] == len(dec['f_n'])


def _split_toy():
    """Two panels displaced arbitrarily, enough to load all three springs."""
    from types import SimpleNamespace
    cnv = np.array([[[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]] * 2)
    state = SimpleNamespace(centroid_node_vectors=cnv,
                            bond_connectivity=np.array([[1, 4]], dtype=np.int32))
    disp = np.array([[0.013, -0.021, 0.034], [-0.011, 0.022, -0.029]])
    return state, disp, np.array([[1.0, 0.0]])


def test_energy_split_is_exactly_additive():
    """The property the whole split rests on, and the one the CLI checks against the solve.

    The three springs are independent quadratics, so evaluating the solver's own bond energy with
    the other two stiffnesses zeroed returns the remaining term exactly. Reconstructing the split
    from a strain measure of one's own choosing is what this exists to avoid: a corotated
    reconstruction disagreed with these runs' stored energy by up to 7%.
    """
    from nff.stages.physics.energy import ligament_energy_linearized
    from nff.utils.linalg import face_to_node_kinematics_fn
    import jax.numpy as jnp

    state, disp, ref = _split_toy()
    ks = dict(k_stretch=2.0, k_shear=3.0, k_rot=5.0)
    split = energy_split(state, disp, ref, **ks)

    cnv = jnp.asarray(state.centroid_node_vectors)
    nodes = face_to_node_kinematics_fn(jnp.asarray(disp), cnv).reshape((8, 3))
    bond = state.bond_connectivity
    full = float(np.asarray(ligament_energy_linearized(
        (nodes[bond[:, 0]], nodes[bond[:, 1]]), reference_vector=jnp.asarray(ref), **ks)).sum())

    assert split['total'] == pytest.approx(full, rel=1e-12)
    assert sum(split['totals'].values()) == pytest.approx(full, rel=1e-12)
    assert sum(split['shares'].values()) == pytest.approx(1.0)
    assert all(v >= 0 for v in split['totals'].values())


def test_energy_split_is_quadratic_in_each_stiffness_alone():
    """Doubling one stiffness doubles only its own term, which is what makes the split a split."""
    state, disp, ref = _split_toy()
    base = energy_split(state, disp, ref, k_stretch=2.0, k_shear=3.0, k_rot=5.0)
    stiffer = energy_split(state, disp, ref, k_stretch=2.0, k_shear=3.0, k_rot=10.0)

    assert stiffer['totals']['rotation'] == pytest.approx(2 * base['totals']['rotation'], rel=1e-12)
    assert stiffer['totals']['axial'] == pytest.approx(base['totals']['axial'], rel=1e-12)
    assert stiffer['totals']['shear'] == pytest.approx(base['totals']['shear'], rel=1e-12)

    dead = energy_split(state, disp, ref, k_stretch=2.0, k_shear=3.0, k_rot=0.0)
    assert dead['totals']['rotation'] == pytest.approx(0.0, abs=1e-15)


def test_rotation_departure_tracks_the_force_a_hinge_carries():
    """A sheet built so the stiff springs wake up only where the force is large.

    The manuscript's claim is a MONOTONE crossover, so the statistic that carries it is the rank
    correlation, not a bin ratio: a threshold and a gradual ramp both raise the top bin.
    """
    field = _grid_field(nx=6, ny=4)
    dec = dict(interface_decomposition(field))
    n = len(dec['f_mag'])
    force = np.linspace(0.1, 1.0, n)
    dec['f_mag'] = force

    per_hinge = {'rotation': np.ones(n),
                 'axial': 0.5 * force ** 2,          # stiff channels grow with the load
                 'shear': 0.5 * force ** 2}
    result = rotation_departure(field, dec, per_hinge)

    assert result['spearman'] > 0.99
    assert result['share_strongest'] > result['share_weakest']
    assert result['ratio'] > 1.0
    assert result['share_by_force_quantile'] == sorted(result['share_by_force_quantile'])


def test_rotation_departure_is_flat_when_the_split_does_not_depend_on_load():
    field = _grid_field(nx=6, ny=4)
    dec = dict(interface_decomposition(field))
    n = len(dec['f_mag'])
    dec['f_mag'] = np.linspace(0.1, 1.0, n)
    per_hinge = {'rotation': np.ones(n), 'axial': 0.1 * np.ones(n), 'shear': 0.1 * np.ones(n)}

    result = rotation_departure(field, dec, per_hinge)
    assert result['ratio'] == pytest.approx(1.0, abs=1e-9)
    assert result['share_overall'] == pytest.approx(0.2 / 1.2, rel=1e-9)


def test_force_clustering_separates_a_compact_region_from_a_scattered_one():
    """Contiguous strong tiles score above chance; strong tiles spread out do not."""
    field = _grid_field(nx=8, ny=8)
    dec = dict(interface_decomposition(field))
    pairs = np.asarray(field.face_pairs)
    xs = field.face_xy[:, 0]

    def _clustering(strong_tiles):
        # Drive scatter_to_tiles by putting all the force on interfaces inside the chosen set.
        inside = strong_tiles[pairs[:, 0]] & strong_tiles[pairs[:, 1]]
        dec['f_mag'] = np.where(inside, 1.0, 1e-6)
        return force_clustering(field, dec, quantile=0.75)['ratio']

    compact = _clustering(xs < 2)                       # one contiguous strip, 16 of 64 tiles
    scattered = _clustering((np.arange(field.n_faces) % 4) == 0)   # same count, spread out

    assert compact > 1.5
    assert scattered < compact


def test_region_mask_selects_a_slab():
    field = _grid_field(nx=6, ny=3)
    left = region_mask(field, (0.0, 1 / 3))
    assert left.sum() > 0
    assert field.face_xy[left, 0].max() < field.face_xy[~left, 0].max()


# ── does shear sit on the tension/compression boundary? ───────────────────────

def _split_sheet(nx=6, ny=4, shear_on_boundary=True):
    """Left half in tension, right half in compression, with the crossing hinges made shear.

    The positive control for :func:`mode_boundary_shear`. A sheet where shear genuinely does
    separate the two regions, so a measure that reports nothing here is broken rather than
    reporting an absence.
    """
    import dataclasses

    field = _grid_field(nx=nx, ny=ny,
                        pressure=np.where(np.arange(nx * ny) // ny < nx // 2, 1.0, -1.0))
    pairs = np.asarray(field.face_pairs)
    axis = np.asarray(field.hinge_axis)
    side = (np.arange(nx * ny) // ny) < nx // 2
    crossing = side[pairs[:, 0]] != side[pairs[:, 1]]

    t = -np.asarray(field.t_lig, dtype=float).copy()
    if shear_on_boundary:                     # perpendicular force -> |f_t| > |f_n|
        t[crossing] = np.column_stack([-axis[crossing, 1], axis[crossing, 0]])
    return dataclasses.replace(field, t_lig=-t), crossing


def test_mode_boundary_shear_detects_a_sheet_where_shear_really_does_separate():
    field, crossing = _split_sheet()
    result = mode_boundary_shear(field, interface_decomposition(field), n_null=400)

    assert result['n_boundary'] == int(crossing.sum())
    assert result['coverage'] == pytest.approx(1.0)          # every boundary hinge is shear
    assert result['enrichment'] > 1.5
    assert result['z'] > 2.0
    # The topological half: deleting the shear hinges must leave pressure-pure components.
    assert result['cutset_purity'] == pytest.approx(1.0)
    assert result['cutset_z'] > 2.0


def test_mode_boundary_shear_reports_no_separation_when_the_boundary_carries_normal_force():
    """Same two regions, but the crossing hinges are normal. The cut-set test must not fire."""
    field, _ = _split_sheet(shear_on_boundary=False)
    result = mode_boundary_shear(field, interface_decomposition(field), n_null=400)

    assert result['coverage'] == pytest.approx(0.0)
    assert result['cutset_purity'] < 1.0


def test_mode_boundary_shear_is_nan_without_a_boundary_and_still_reports_the_cut_set():
    """A sheet of one pressure sign has no mode boundary at all, and must not divide by zero."""
    field = _grid_field(nx=5, ny=4, pressure=np.ones(20))
    result = mode_boundary_shear(field, interface_decomposition(field), n_null=200)

    assert result['n_boundary'] == 0
    assert np.isnan(result['enrichment'])
    assert np.isfinite(result['cutset_purity'])
    assert result['cutset_purity'] == pytest.approx(1.0)      # one sign, so every component is pure


def test_mode_boundary_shear_survives_an_all_shear_sheet():
    """Every interface shear-dominated: the enrichment is exactly one, and nothing raises."""
    import dataclasses

    field = _grid_field(nx=5, ny=4,
                        pressure=np.where(np.arange(20) % 2 == 0, 1.0, -1.0))
    axis = np.asarray(field.hinge_axis)
    field = dataclasses.replace(field, t_lig=np.column_stack([-axis[:, 1], axis[:, 0]]))
    result = mode_boundary_shear(field, interface_decomposition(field), n_null=200)

    assert result['p_shear'] == pytest.approx(1.0)
    assert result['enrichment'] == pytest.approx(1.0)
