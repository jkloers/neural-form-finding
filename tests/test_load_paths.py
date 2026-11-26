"""Guards for the load-path extraction and the currents drawn from it.

Two things can go wrong silently here. The forces can be subtly wrong -- a missed lever arm, a
dropped channel, a sign flip -- and still produce a plausible picture, so the physics is pinned to
cases with a known answer. And the trajectory integrator can shred a principal-direction field,
because those are sign-ambiguous LINE fields rather than vector fields; that failure is checked
against a trajectory whose exact shape is known.
"""

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from nff.closed.load_paths import (ForceField, _gini, branch_frame, corotate_axis,  # noqa: E402
                                   deployed_nodes, edge_modes, interface_decomposition,
                                   ligament_pair_forces, orientation_confound,
                                   polygon_areas, resolve_frame, virial_stress)
from nff.closed.load_path_viz import (chaikin, mode_intensity,  # noqa: E402
                                      shear_planes, zone_colours)
from nff.stages.physics.energy import ligament_energy_linearized  # noqa: E402


K = 1000.0
N_NODES = 4
# Two unit tiles side by side, joined by one hinge. Face 0's node 0 sits at (0.5, 0); face 1's
# node 2 sits at (1.5, 0); the reference bond between them is (1, 0).
CNV = jnp.array([[[0.5, 0.0], [0.0, 0.5], [-0.5, 0.0], [0.0, -0.5]]] * 2)
CENTROIDS = jnp.array([[0.0, 0.0], [2.0, 0.0]])
BOND_PAIRS = np.array([[0, N_NODES + 2]])
BOND_KWARGS = dict(reference_vector=jnp.array([[1.0, 0.0]]),
                   k_stretch=jnp.array([K]), k_shear=jnp.array([K]), k_rot=jnp.array([K]))


def pair_forces(face_disp):
    return ligament_pair_forces(jnp.asarray(face_disp), CNV, BOND_PAIRS,
                                ligament_energy_linearized, BOND_KWARGS)


# ── the physics ───────────────────────────────────────────────────────────────

def test_axial_pull_gives_the_analytic_spring_force():
    """One hinge, one known answer. Everything downstream inherits this sign convention."""
    d = 0.01
    f_i, f_j, energy = pair_forces([[0.0, 0.0, 0.0], [d, 0.0, 0.0]])

    assert float(energy[0]) == pytest.approx(K * d ** 2 / 2, rel=1e-10)
    # Stretched: the ligament pulls face j back toward face i, i.e. in -x.
    assert float(f_j[0, 0]) == pytest.approx(-K * d, rel=1e-10)
    assert float(f_i[0, 0]) == pytest.approx(+K * d, rel=1e-10)


def test_newtons_third_law_holds_for_an_arbitrary_state():
    """Translational invariance of the pair energy; a broken lever arm would not disturb it, but a
    broken gather would."""
    rng = np.random.default_rng(0)
    f_i, f_j, _ = pair_forces(rng.normal(scale=0.05, size=(2, 3)))
    assert np.allclose(np.asarray(f_i)[:, :2], -np.asarray(f_j)[:, :2], atol=1e-12)


def test_rigid_body_motion_carries_no_force():
    """Pure translation and pure rotation must both be free. This is the check that the rigid-tile
    kinematics are chained correctly: rotating a tile moves its NODES, and a missing lever arm
    would show up here as a spurious force under pure rotation."""
    f_i, f_j, energy = pair_forces([[0.3, -0.2, 0.0], [0.3, -0.2, 0.0]])
    assert float(energy[0]) == pytest.approx(0.0, abs=1e-18)
    assert np.allclose(np.asarray(f_j), 0.0, atol=1e-12)

    omega = 0.37
    rot = jnp.array([[jnp.cos(omega), -jnp.sin(omega)], [jnp.sin(omega), jnp.cos(omega)]])
    moved = CENTROIDS @ rot.T - CENTROIDS
    spun = jnp.concatenate([moved, jnp.full((2, 1), omega)], axis=-1)

    f_i, f_j, energy = pair_forces(spun)
    assert float(energy[0]) == pytest.approx(0.0, abs=1e-18)
    assert np.allclose(np.asarray(f_j), 0.0, atol=1e-10)


def test_moment_appears_on_the_tile_that_rotates():
    """Rotating one tile alone must load the hinge and produce a reaction moment on that tile."""
    f_i, f_j, energy = pair_forces([[0.0, 0.0, 0.0], [0.0, 0.0, 0.05]])
    assert float(energy[0]) > 0.0
    assert abs(float(f_j[0, 2])) > 0.0
    # The moments are NOT equal and opposite: they differ by the lever arm of the shared force.
    assert not np.isclose(float(f_i[0, 2]), -float(f_j[0, 2]), atol=1e-9)


def test_virial_stress_is_positive_under_tension():
    """A tile pulled outward at its right edge must register positive (tensile) sigma_xx."""
    face_xy = np.array([[0.0, 0.0]])
    hinge_xy = np.array([[0.5, 0.0]])
    pairs = np.array([[0, 0]])
    pull = np.array([[10.0, 0.0, 0.0]])          # force on the tile, pointing +x, applied at +x

    sigma = virial_stress(face_xy, np.array([1.0]), pairs, hinge_xy, pull, np.zeros((1, 3)))
    assert sigma[0, 0, 0] > 0
    assert sigma[0, 0, 0] == pytest.approx(0.5 * 10.0, rel=1e-12)


def test_deployed_nodes_and_areas_follow_the_rigid_tile():
    """A 90-degree tile rotation must move its nodes and leave its area untouched."""
    disp = jnp.array([[0.0, 0.0, jnp.pi / 2], [0.0, 0.0, 0.0]])
    nodes = deployed_nodes(CENTROIDS, CNV, disp)
    assert np.allclose(nodes[0, 0], [0.0, 0.5], atol=1e-12)      # (0.5,0) rotated by +90 deg
    assert np.allclose(polygon_areas(nodes), polygon_areas(deployed_nodes(
        CENTROIDS, CNV, jnp.zeros((2, 3)))), atol=1e-12)


def test_gini_spans_even_to_concentrated():
    assert _gini(np.ones(50)) == pytest.approx(0.0, abs=1e-12)
    assert _gini(np.array([0.0] * 999 + [1.0])) > 0.99
    assert _gini(np.zeros(10)) == 0.0                             # no force at all must not divide


def _toy_field(t_values, face_pairs=None, face_xy=None, hinge_axis=None, hinge_xy=None):
    """Tiles joined by hinges carrying ``t_values``; four in a row at x = 0,1,2,3 by default."""
    face_xy = (np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]) if face_xy is None
               else np.asarray(face_xy, dtype=float))
    n_faces = face_xy.shape[0]
    corner = np.array([[-0.4, -0.4], [0.4, -0.4], [0.4, 0.4], [-0.4, 0.4]])
    t = np.atleast_2d(np.asarray(t_values, dtype=float))
    pairs = np.array([[0, 1], [1, 2], [2, 3]]) if face_pairs is None else np.asarray(face_pairs)
    n = t.shape[0]
    # Resolved in the branch frame exactly as the extractor does, so the fixture can carry shear.
    span = face_xy[pairs[:, 1]] - face_xy[pairs[:, 0]]
    axis = span / np.linalg.norm(span, axis=-1, keepdims=True)
    areas = np.full(n_faces, 0.64)
    hinge_mid = (0.5 * (face_xy[pairs[:, 0]] + face_xy[pairs[:, 1]]) if hinge_xy is None
                 else np.asarray(hinge_xy, dtype=float))
    return ForceField(
        face_xy=face_xy, face_nodes=face_xy[:, None, :] + corner[None],
        face_areas=areas,
        hinge_xy=hinge_mid,
        face_pairs=pairs,
        t_lig=t, m_lig=np.zeros((n, 2)), t_con=np.zeros((n, 2)), m_con=np.zeros((n, 2)),
        hinge_axis=(np.zeros((n, 2)) if hinge_axis is None
                    else np.asarray(hinge_axis, dtype=float).reshape(n, 2)),
        axial=-np.sum(t * axis, axis=-1),
        transverse=t[:, 0] * axis[:, 1] - t[:, 1] * axis[:, 0], energy=np.zeros(n),
        # The virial stress is BUILT from the same forces rather than zero-filled. Anything reading
        # the net state of a tile -- the signed pressure, and the compression fill derived from it
        # -- reads `sigma`, and a zero-filled fixture silently reports an unstressed sheet.
        sigma=virial_stress(face_xy, areas, pairs, hinge_mid, -t, t),
        f_ext=np.zeros((n_faces, 3)),
        residual=np.zeros((n_faces, 3)),
        free_mask=np.ones((n_faces, 3), dtype=bool), boundary_xy=np.zeros((0, 2)), checks={})


# ── the flow field ────────────────────────────────────────────────────────────

def _circular_line_field(resolution=161):
    grid = ff.build_grid(np.array([[-1.5, -1.5], [1.5, 1.5]]), resolution=resolution, pad=0.0)
    X, Y = grid.meshgrid()
    r = np.maximum(np.hypot(X, Y), 1e-9)
    return grid, np.stack([-Y / r, X / r], axis=-1)


# ── Interface decomposition ───────────────────────────────────────────────────────────────────

def test_branch_frame_is_orthonormal_and_points_from_i_to_j():
    """The frame is the line of centres, because a point hinge has no contact plane."""
    field = _toy_field([[-1.0, 0.0], [-2.0, 0.0], [-3.0, 0.0]])
    n_hat, t_hat = branch_frame(field)
    assert np.allclose(np.linalg.norm(n_hat, axis=-1), 1.0)
    assert np.allclose(np.sum(n_hat * t_hat, axis=-1), 0.0, atol=1e-14)
    assert np.allclose(n_hat, np.array([[1.0, 0.0]] * 3))     # tiles are laid out along +x


def test_f0_sums_the_tile_force_magnitudes_and_never_divides_by_zero():
    """The normaliser is the SIZE of the load set, not its resultant.

    A self-equilibrated couple loads the sheet perfectly well while its resultant vanishes into
    float noise, so normalising by the resultant divides by ~1e-14 and inflates every share
    without bound.
    """
    field = _toy_field([[-1.0, 0.0]])
    assert field.f0 == 1.0                                    # no load applied: falls back to 1
    field.f_ext[1, 0] = 3.0
    field.f_ext[2, 1] = 4.0
    assert field.f0 == pytest.approx(7.0)                     # 3 + 4, not the resultant 5

    couple = _toy_field([[-1.0, 0.0]])
    couple.f_ext[1, 0] = 2.0
    couple.f_ext[2, 0] = -2.0
    assert couple.f0 == pytest.approx(4.0)                    # resultant is zero, the load is not


def test_shear_dominated_flag_follows_the_branch_frame():
    """A force perpendicular to the line of centres is pure shear, whatever its magnitude."""
    field = _toy_field([[0.0, 7.0], [-7.0, 0.0]], face_pairs=np.array([[0, 1], [1, 2]]))
    dec = interface_decomposition(field)
    assert dec['shear_dominated'][0] and not dec['shear_dominated'][1]
    assert dec['f_t'][0] == pytest.approx(-7.0)
    assert dec['f_n'][0] == pytest.approx(0.0, abs=1e-14)


def test_normalised_quantities_scale_out_the_applied_load():
    """Two sheets carrying different absolute loads must be comparable after normalisation."""
    field = _toy_field([[-2.0, 0.0], [-4.0, 0.0], [-6.0, 0.0]])
    field.f_ext[3, 0] = 8.0
    dec = interface_decomposition(field)
    assert np.allclose(dec['f_n_norm'], np.array([2.0, 4.0, 6.0]) / 8.0)


# ── The orientation confound, and the frame-free force-pair glyph ─────────────────────────────

def test_orientation_confound_catches_a_uniaxial_lattice():
    """The normal/shear label must be flagged when it is really reporting hinge ORIENTATION.

    A uniform +x force through a square lattice: hinges lying along x read as normal, hinges lying
    across x read as shear, purely because of where they sit. This is the confound that split one
    load path across two panels, so it has to be measurable rather than remembered.
    """
    face_xy, pairs, t = [], [], []
    for i in range(4):                                   # a row of tiles joined both ways
        for j in range(4):
            face_xy.append([float(i), float(j)])
    face_xy = np.array(face_xy)
    index = lambda i, j: i * 4 + j                       # noqa: E731
    for i in range(4):
        for j in range(4):
            if i + 1 < 4:
                pairs.append([index(i, j), index(i + 1, j)])
                t.append([-5.0, 0.0])                    # along the branch -> reads as normal
            if j + 1 < 4:
                pairs.append([index(i, j), index(i, j + 1)])
                t.append([-5.0, 0.0])                    # across the branch -> reads as shear

    field = _toy_field(t, face_pairs=np.array(pairs), face_xy=face_xy)
    field.f_ext[0, 0] = 10.0                             # the load axis is +x
    assert orientation_confound(field) > 0.8




# ── The load-path figure: tension routes over compression/shear regions ───────────────────────

def test_mode_intensity_attributes_to_both_tiles_of_a_hinge():
    """Per-TILE totals, so the shading is exact and cannot smear a mode across a void."""
    field = _toy_field([[0.0, 6.0]], face_pairs=np.array([[0, 1]]))   # pure shear, one hinge
    field.f_ext[3, 0] = 12.0
    compression, shear = mode_intensity(field, interface_decomposition(field))
    assert shear[0] == pytest.approx(0.5) and shear[1] == pytest.approx(0.5)
    assert np.allclose(compression, 0.0)


def test_zone_colours_shade_compression_only():
    """Shear no longer competes for the fill channel; it has its own representation (slip planes).

    Tile 0-1 is compressed, 1-2 is pure shear. Only the compressed pair may be shaded, or a reader
    cannot tell "this tile is sheared" from "this tile is compressed but a neighbour won the tile".
    """
    field = _toy_field([[8.0, 0.0], [0.0, 8.0], [0.0, 0.0]])
    field.f_ext[3, 0] = 8.0
    dec = interface_decomposition(field)
    assert dec['f_n'][0] < 0 and dec['shear_dominated'][1]     # compression, then pure shear
    colours, shaded = zone_colours(field, dec)
    assert len(colours) == field.n_faces
    assert shaded == 2                                        # exactly the two compressed tiles
    assert all(len(c) == 3 for c in colours)                  # plain RGB: no per-face alpha


def _slip_toy(sign=1.0):
    """Two rows of tiles zigzagging across one plane, sliding past each other.

    A single ROW cannot show a slip plane along its own direction -- sliding along a row is
    separation, not shear. A real plane alternates from side to side, and the transmitted force
    alternates sign with that orientation, which is precisely what the side-flip in
    ``shear_planes`` has to undo.
    """
    face_xy = np.array([[0.0, 0.0], [0.5, 1.0], [1.0, 0.0], [1.5, 1.0]])
    pairs = np.array([[0, 1], [1, 2], [2, 3]])
    t = sign * np.array([[-6.0, 0.0], [6.0, 0.0], [-6.0, 0.0]])   # alternates with the zigzag
    field = _toy_field(t, face_pairs=pairs, face_xy=face_xy)
    field.f_ext[3, 0] = 6.0
    return field


def _plane_toy(offset=0.0):
    """Three shear interfaces on one line, each with its cut axis ACROSS that line.

    Pure shear by construction: the force runs along the line, the hinge axis across it. ``offset``
    lifts the middle interface off the line, to test that non-collinear ones do not merge.
    """
    face_xy = np.array([[0.0, 1.0], [0.0, -1.0], [1.0, 1.0], [1.0, -1.0],
                        [2.0, 1.0], [2.0, -1.0]])
    pairs = np.array([[0, 1], [2, 3], [4, 5]])
    hinges = np.array([[0.0, 0.0], [1.0, offset], [2.0, 0.0]])
    axis = np.tile([[0.0, 1.0]], (3, 1))                  # cut axis is +y ...
    t = np.tile([[5.0, 0.0]], (3, 1))                     # ... force is +x: pure shear
    return _toy_field(t, face_pairs=pairs, face_xy=face_xy, hinge_axis=axis, hinge_xy=hinges)


def test_slip_plane_runs_NORMAL_to_the_hinge_axis():
    """The definition, asserted directly -- shear is normal to the hinge axis.

    The previous construction chained interfaces by tile adjacency and came out 81 degrees off,
    essentially perpendicular to the truth, so this is the regression that matters most.
    """
    field = _plane_toy()
    dec = interface_decomposition(field)
    assert dec['shear_dominated'].all()

    planes = shear_planes(field, dec, np.ones(field.n_hinges, dtype=bool))
    assert len(planes) == 1
    plane = planes[0]
    assert plane['n_hinges'] == 3                          # all three merged onto one line

    for h in range(field.n_hinges):
        axis = dec['n_hat'][h]
        assert abs(float(plane['direction'] @ axis)) < 1e-9, "plane is not normal to the hinge axis"


def test_non_collinear_shear_interfaces_do_not_merge():
    """Parallel is not enough -- they must lie on the SAME line to be one plane."""
    apart = shear_planes(_plane_toy(offset=2.0),
                         interface_decomposition(_plane_toy(offset=2.0)),
                         np.ones(3, dtype=bool))
    assert len(apart) > 1


def test_shear_plane_slip_sense_flips_with_the_shear_direction():
    """The sense is computed from the transmitted force, so reversing it must reverse the arrows."""
    keep = np.ones(3, dtype=bool)
    a, b = _slip_toy(+1.0), _slip_toy(-1.0)
    sa = shear_planes(a, interface_decomposition(a), keep)[0]['slip']
    sb = shear_planes(b, interface_decomposition(b), keep)[0]['slip']
    assert sa == -sb


def test_chaikin_smooths_without_leaving_the_control_hull():
    """The guard against the smoothing-spline failure: a corner cut can never bow outside."""
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [2.0, 1.0]])
    out = chaikin(pts, iterations=3)
    assert out.shape[0] > pts.shape[0]
    assert out[:, 0].min() >= pts[:, 0].min() - 1e-12
    assert out[:, 0].max() <= pts[:, 0].max() + 1e-12
    assert out[:, 1].min() >= pts[:, 1].min() - 1e-12
    assert out[:, 1].max() <= pts[:, 1].max() + 1e-12
    assert np.allclose(out[0], pts[0]) and np.allclose(out[-1], pts[-1])


def test_edge_modes_labels_the_three_channels():
    field = _toy_field([[-4.0, 0.0], [4.0, 0.0], [0.0, 4.0]])
    modes = edge_modes(interface_decomposition(field))
    assert list(modes) == ['tension', 'compression', 'shear']


# ── The frame the force is resolved in ────────────────────────────────────────────────────────

def test_cut_frame_overrides_the_branch_frame_and_can_invert_the_label():
    """THE regression: the two frames disagree exactly when the cut axis is across the branch.

    A hinge whose ligament runs PERPENDICULAR to the line of centres classifies oppositely in the
    two frames -- pure tension along the cut reads as pure shear across the branch. The cut frame
    must win, because it is the frame `eta_a`/`eta_s` and the compression loss term are read off.
    """
    t = np.array([[0.0, -5.0]])                      # force along +y on face j... resolved below
    across = np.array([[0.0, 1.0]])                  # cut axis perpendicular to the +x branch
    field = _toy_field(t, face_pairs=np.array([[0, 1]]), hinge_axis=across)

    branch = interface_decomposition(_toy_field(t, face_pairs=np.array([[0, 1]])))
    cut = interface_decomposition(field)

    assert branch['frame'] == 'branch' and cut['frame'] == 'cut'
    assert branch['shear_dominated'][0] and not cut['shear_dominated'][0]   # the label inverts
    assert abs(cut['f_n'][0]) == pytest.approx(5.0)                          # pure tension on the cut
    assert cut['f_t'][0] == pytest.approx(0.0, abs=1e-12)


def test_missing_cut_axis_falls_back_to_the_branch_frame():
    """A replot without the cut axis must still work, and must say which frame it used."""
    field = _toy_field([[-4.0, 0.0]], face_pairs=np.array([[0, 1]]))
    n_hat, _, source = resolve_frame(field)
    assert source == 'branch'
    assert np.allclose(n_hat, [[1.0, 0.0]])


def test_corotation_turns_the_cut_axis_with_the_tiles():
    """The cut axis is a REFERENCE quantity; the force is deployed, so it must be corotated.

    Matches `hinge_kinematics`, which corotates by the MEAN of the two tiles' rotations before
    reading (a, s) -- otherwise the figure and the loss sit in two frames sharing one name.
    """
    axis = np.array([[1.0, 0.0]])
    pairs = np.array([[0, 1]])
    quarter = np.zeros((2, 3))
    quarter[:, 2] = np.pi / 2                        # both tiles rotate 90 degrees
    turned = corotate_axis(axis, quarter, pairs)
    assert np.allclose(turned, [[0.0, 1.0]], atol=1e-12)

    half = np.zeros((2, 3))
    half[0, 2], half[1, 2] = 0.0, np.pi / 2          # mean rotation is 45 degrees
    assert np.allclose(corotate_axis(axis, half, pairs),
                       [[np.cos(np.pi / 4), np.sin(np.pi / 4)]], atol=1e-12)


def test_slip_plane_extends_past_its_end_interfaces_into_the_voids():
    """The trace must reach beyond the outermost contacts, so it crosses the flanking voids.

    A plane that stopped exactly on its end contacts would read as a corner-to-corner connector
    rather than a plane cutting through the sheet.
    """
    field = _plane_toy()
    dec = interface_decomposition(field)
    plane = shear_planes(field, dec, np.ones(field.n_hinges, dtype=bool))[0]

    along = (np.asarray(field.hinge_xy) - plane['points'].mean(axis=0)) @ plane['direction']
    ends = (plane['points'] - plane['points'].mean(axis=0)) @ plane['direction']
    assert ends.min() < along.min() - 1e-9
    assert ends.max() > along.max() + 1e-9
    assert plane['points'].shape[0] > 2                    # a smoothed trace, not a bare segment


def test_slip_plane_direction_survives_arbitrary_axis_SIGNS():
    """A cut axis is a LINE, not a vector: its sign is arbitrary, and averaging signed vectors
    cancels them into a direction ~90 degrees off.

    Measured before the fix: a plane's own contacts sat 9 tiles along its "normal" and 0.3 along its
    "direction", so the trace zigzagged sideways to reach them at 3x the straight-line length. Same
    trap that shredded the principal-direction tracer.
    """
    field = _plane_toy()
    dec = interface_decomposition(field)

    flipped = _plane_toy()
    flipped.hinge_axis[1] = -flipped.hinge_axis[1]        # flip ONE axis: same line, other sign
    dec_flipped = interface_decomposition(flipped)

    a = shear_planes(field, dec, np.ones(3, dtype=bool))[0]
    b = shear_planes(flipped, dec_flipped, np.ones(3, dtype=bool))[0]

    assert abs(float(a['direction'] @ b['direction'])) > 0.999   # same line either way
    for plane in (a, b):
        span = np.linalg.norm(plane['points'][-1] - plane['points'][0])
        length = np.linalg.norm(np.diff(plane['points'], axis=0), axis=1).sum()
        assert length / span < 1.05, "trace wanders off its own plane"
