"""The exactness identity the katagami warp rests on: bilinear image of a segment IS a quadratic."""

import numpy as np
import pytest

from legacy.katagami.topology_art.quad_warp import (
    assert_ccw_convex_fn,
    bilinear_fn,
    path_points_fn,
    sagitta_fn,
    segment_bezier_fn,
    transform_path_fn,
    warp_polyline_fn,
)


def random_convex_quads(n, seed=0):
    """(n, 4, 2) convex CCW quads: sorted angles on a circle, then a random affine with det > 0."""
    rng = np.random.default_rng(seed)
    angles = np.sort(rng.uniform(0.0, 2.0 * np.pi, size=(n, 4)), axis=1)
    base = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
    out = np.empty_like(base)
    for i in range(n):
        while True:
            matrix = rng.normal(size=(2, 2))
            if np.linalg.det(matrix) > 0.2:
                break
        out[i] = base[i] @ matrix.T + rng.normal(size=2)
    assert_ccw_convex_fn(out, name="random quads")
    return out


def test_segment_image_is_exactly_quadratic():
    """The whole feature rests on this: a straight (u, v) segment maps to an exact quadratic."""
    rng = np.random.default_rng(1)
    quads = random_convex_quads(64, seed=1)
    t = np.linspace(0.0, 1.0, 64)[:, None]
    worst = 0.0
    for quad in quads:
        p0, p1 = rng.random(2), rng.random(2)
        b0, b1, b2 = segment_bezier_fn(quad, p0, p1)
        curve = (1 - t) ** 2 * b0 + 2 * (1 - t) * t * b1 + t**2 * b2
        truth = bilinear_fn(quad, p0 + t * (p1 - p0))
        diameter = np.linalg.norm(quad - quad.mean(axis=0), axis=1).max()
        worst = max(worst, float(np.abs(curve - truth).max() / diameter))
    assert worst < 1e-13, f"relative deviation {worst:.3e}"


def test_corners_map_to_corners():
    quads = random_convex_quads(16, seed=2)
    cell = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    for quad in quads:
        assert np.allclose(bilinear_fn(quad, cell), quad, atol=1e-14)


def test_unit_cell_edges_map_to_straight_lines():
    """Cell-boundary segments have du*dv = 0, so their images are exactly straight and emit as L."""
    cell = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    for quad in random_convex_quads(32, seed=3):
        path = warp_polyline_fn(quad, cell, closed=True)
        assert [kind for kind, _ in path.segments] == ["L"] * 4
        for i in range(4):
            b0, b1, b2 = segment_bezier_fn(quad, cell[i], cell[(i + 1) % 4])
            assert sagitta_fn(b0, b1, b2) < 1e-15


def test_diagonals_are_curved_unless_the_quad_is_a_parallelogram():
    """Curvature comes only from d = Q0 - Q1 + Q2 - Q3; a parallelogram warps affinely."""
    square = np.array([[0.0, 0.0], [2.0, 0.0], [3.0, 1.0], [1.0, 1.0]])  # parallelogram
    b0, b1, b2 = segment_bezier_fn(square, [0.0, 0.0], [1.0, 1.0])
    assert sagitta_fn(b0, b1, b2) < 1e-15

    skew = np.array([[0.0, 0.0], [2.0, 0.0], [1.2, 1.0], [0.0, 1.0]])
    b0, b1, b2 = segment_bezier_fn(skew, [0.0, 0.0], [1.0, 1.0])
    assert sagitta_fn(b0, b1, b2) > 1e-3


def test_sagitta_is_the_exact_parametric_deviation_and_bounds_the_geometric_one():
    rng = np.random.default_rng(4)
    t = np.linspace(0.0, 1.0, 2001)[:, None]
    for quad in random_convex_quads(24, seed=4):
        p0, p1 = rng.random(2), rng.random(2)
        b0, b1, b2 = segment_bezier_fn(quad, p0, p1)
        curve = (1 - t) ** 2 * b0 + 2 * (1 - t) * t * b1 + t**2 * b2
        chord = b0 + t * (b2 - b0)
        flatness = sagitta_fn(b0, b1, b2)

        parametric = float(np.linalg.norm(curve - chord, axis=1).max())
        assert parametric == pytest.approx(flatness, rel=1e-6, abs=1e-12)

        direction = b2 - b0
        length = float(np.linalg.norm(direction))
        if length < 1e-9:
            continue
        normal = np.array([-direction[1], direction[0]]) / length
        assert float(np.abs((curve - chord) @ normal).max()) <= flatness + 1e-12


def test_warp_commutes_with_affine_maps():
    """Warping a transformed quad equals transforming the warped path -- exact, not approximate."""
    rng = np.random.default_rng(5)
    poly = rng.random((7, 2))
    matrix = np.array([[0.8, -0.35], [0.2, 1.1]])
    offset = np.array([3.0, -2.0])
    for quad in random_convex_quads(16, seed=5):
        direct = warp_polyline_fn(quad @ matrix.T + offset, poly)
        carried = transform_path_fn(warp_polyline_fn(quad, poly), matrix, offset)
        assert np.abs(path_points_fn(direct) - path_points_fn(carried)).max() < 1e-12


def test_warped_geometry_stays_inside_its_own_quad():
    """Convexity means the bilinear image of the unit cell is the quad, so nothing can leak out."""
    shapely = pytest.importorskip("shapely.geometry")
    rng = np.random.default_rng(6)
    poly = rng.random((9, 2))
    for quad in random_convex_quads(32, seed=6):
        cell = shapely.Polygon(quad).buffer(1e-9)
        pts = path_points_fn(warp_polyline_fn(quad, poly, closed=True), per_segment=32)
        assert cell.covers(shapely.LineString(pts))


def test_assert_ccw_convex_rejects_a_folded_quad():
    folded = np.array([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
    with pytest.raises(ValueError, match="convex"):
        assert_ccw_convex_fn(folded)
