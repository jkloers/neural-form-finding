"""A continuous net authored globally and warped cell by cell: does it stay one pattern?"""

import numpy as np
import pytest

from legacy.katagami.topology_art.net import (
    NETS,
    build_net_fn,
    clip_to_cells_fn,
    warp_net_fn,
)
from legacy.katagami.topology_art.quad_warp import path_points_fn

COLS, ROWS = 8, 6


def _mesh(distortion=0.0):
    """A control mesh over ``COLS x ROWS`` cells; distorted but still convex.

    Deliberately NOT separable. A displacement of the form ``f(y)`` in x and ``g(x)`` in y leaves
    every cell an exact parallelogram, whose bilinear map is affine -- so straight lines stay
    straight and the warp never produces a curve to test.
    """
    y, x = np.meshgrid(np.arange(ROWS + 1, dtype=float), np.arange(COLS + 1, dtype=float),
                       indexing="ij")
    return np.stack([x + distortion * np.sin(0.7 * x) * np.cos(0.6 * y),
                     y + distortion * np.cos(0.5 * x) * np.sin(0.8 * y)], axis=-1)


def test_every_piece_lands_inside_exactly_one_cell():
    """The premise of the whole construction: a piece is warped by ONE cell's map."""
    for name in sorted(NETS):
        for polyline in build_net_fn(name, COLS, ROWS, scale=2.0):
            for piece in clip_to_cells_fn(polyline):
                cells = np.floor(0.5 * (piece[:-1] + piece[1:])).astype(int)
                assert (cells == cells[0]).all(), f"{name}: a piece spans more than one cell"


def test_a_vertex_sitting_exactly_on_a_boundary_still_splits():
    """The bug this test exists for: a crossing at t=0 has no interior root to find.

    A splitter that looks only for strict interior crossings walks straight through such a vertex
    and hands back a piece spanning two cells. Sampled circles land on the lattice constantly, so
    this is the common case rather than a corner one.
    """
    exact = np.array([[0.5, 0.5], [1.0, 0.5], [1.5, 0.5], [2.0, 0.5], [2.5, 0.5]])
    pieces = clip_to_cells_fn(exact)
    assert len(pieces) == 3, f"expected one piece per cell, got {len(pieces)}"
    for piece in pieces:
        cells = np.floor(0.5 * (piece[:-1] + piece[1:])).astype(int)
        assert (cells == cells[0]).all()


@pytest.mark.parametrize("name", sorted(NETS))
def test_the_pieces_rejoin_to_machine_precision(name):
    """THE invariant. Two cells must compute their shared edge identically.

    Where a net crosses a cell boundary the two warps have to land on the same point. If they do
    not, the net is not continuous and the whole construction is unsound -- so this is measured
    rather than argued from the algebra.
    """
    mesh = _mesh(0.3)
    paths = warp_net_fn(build_net_fn(name, COLS, ROWS, scale=2.0), mesh)
    ends = np.array([[np.asarray(p.start, float), np.asarray(p.endpoint, float)]
                     for p in paths]).reshape(-1, 2)

    # Pair each endpoint with its nearest neighbour; the ones that should meet are those that very
    # nearly do, and the worst of those gaps is the residual.
    from scipy.spatial import cKDTree

    pairs = cKDTree(ends).query_pairs(1e-6, output_type="ndarray")
    assert len(pairs) > 10, "a net must cross cell boundaries; almost nothing joined here"
    worst = np.linalg.norm(ends[pairs[:, 0]] - ends[pairs[:, 1]], axis=1).max()
    assert worst < 1e-11, f"{name}: pieces meet {worst:.2e} apart"


def test_a_segment_along_a_cell_boundary_stays_exactly_straight():
    """A bilinear image is straight when du*dv is zero, and the writer must emit it as a line."""
    mesh = _mesh(0.3)
    along = [np.array([[0.0, 2.0], [1.0, 2.0], [2.0, 2.0]])]  # runs down a row of cell edges
    for path in warp_net_fn(along, mesh):
        assert all(kind == "L" for kind, _ in path.segments), "boundary segments must be lines"


def test_the_net_genuinely_bends_away_from_the_boundaries():
    """The counterpart: off the cell edges the warp is quadratic, not a polyline.

    Asked of a whole net rather than of one chosen segment: a single cell can happen to be an
    affine parallelogram, where a straight line stays exactly straight and rightly comes back as
    a line.
    """
    mesh = _mesh(0.4)
    kinds = [kind for path in warp_net_fn(build_net_fn("asanoha", COLS, ROWS, scale=2.0), mesh)
             for kind, _ in path.segments]
    assert "Q" in kinds and "L" in kinds


@pytest.mark.parametrize("name", ["asanoha", "kikko", "kagome", "shippo"])
def test_the_net_is_one_connected_piece_once_it_has_width(name):
    """A net of struts needs no tuning to be cuttable -- being connected is what a net IS.

    Contrast the scattered elements, where a whole round went into making the sheet hold together.
    """
    shapely = pytest.importorskip("shapely.geometry")
    from shapely.ops import unary_union

    from legacy.katagami.stencil import min_bridge_fn

    mesh = _mesh(0.25)
    paths = warp_net_fn(build_net_fn(name, COLS, ROWS, scale=2.0), mesh)
    width = 0.16
    region = unary_union([shapely.LineString(path_points_fn(p, per_segment=8))
                          .buffer(0.5 * width, cap_style=1, join_style=1) for p in paths])
    biggest = max(region.geoms, key=lambda g: g.area) if region.geom_type == "MultiPolygon" \
        else region
    assert biggest.area > 0.9 * region.area, f"{name}: the net is not essentially one piece"
    # With round caps the narrowest place in a net of uniform struts IS the strut. Measured on the
    # main body only: the lattice is generated past the sheet, and eroding a border stub away
    # changes the component count for a reason that has nothing to do with the strut.
    assert min_bridge_fn(biggest, upper=0.5, tolerance=0.002) == pytest.approx(width, rel=0.06)


def test_flat_caps_pinch_where_pieces_meet_at_a_slight_angle():
    """Why the buffer uses round caps, and where the pinch actually is.

    Not at a sharp corner -- a right angle is fine, the two rectangles overlap generously. The
    damage is at a SLIGHT bend, which is exactly what the net produces: a piece ends at a cell
    boundary and the next starts with a slightly different heading, so two flat caps meet at a
    small angle and leave a notch that eats most of the strut.
    """
    shapely = pytest.importorskip("shapely.geometry")
    from shapely.ops import unary_union

    from legacy.katagami.stencil import min_bridge_fn

    bend = np.deg2rad(12.0)
    pieces = [shapely.LineString([(0.0, 0.0), (10.0, 0.0)]),
              shapely.LineString([(10.0, 0.0),
                                  (10.0 + 10.0 * np.cos(bend), 10.0 * np.sin(bend))])]
    flat = unary_union([line.buffer(1.0, cap_style=2, join_style=1) for line in pieces])
    rounded = unary_union([line.buffer(1.0, cap_style=1, join_style=1) for line in pieces])
    assert min_bridge_fn(flat, upper=4.0, tolerance=0.01) < 1.9, "the notch should show"
    assert min_bridge_fn(rounded, upper=4.0, tolerance=0.01) == pytest.approx(2.0, rel=0.06)


def test_a_folded_control_cell_is_refused():
    """A folded cell warps the net inside out rather than failing, so it is checked."""
    from legacy.katagami.net_field import control_mesh_fn

    # Four tiles in a 2x2 grid whose centroids cross over into a bow tie.
    centroids = np.array([[0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    tiles = np.repeat(centroids[:, None, :], 4, axis=1)
    with pytest.raises(ValueError, match="convex"):
        control_mesh_fn(tiles, (2, 2))


def test_the_control_mesh_is_conforming_where_the_tiles_are_not():
    """The reason the centroids are used and the tiles are not: shared edges, no hanging nodes."""
    from legacy.katagami.net_field import control_mesh_fn

    rng = np.random.default_rng(0)
    grid = (6, 5)
    base = np.stack(np.meshgrid(np.arange(grid[0], dtype=float),
                                np.arange(grid[1], dtype=float), indexing="ij"), axis=-1)
    jitter = base + 0.12 * rng.normal(size=base.shape)
    tiles = np.repeat(jitter.reshape(-1, 2)[:, None, :], 4, axis=1)
    mesh = control_mesh_fn(tiles, grid)
    from legacy.katagami.topology_art.quad_warp import bilinear_fn

    def quad(row, column):
        return np.array([mesh[row, column], mesh[row, column + 1],
                         mesh[row + 1, column + 1], mesh[row + 1, column]])

    # Walk the shared edge from both sides and demand the same world points. This is the property
    # the tile mesh cannot offer -- its edges meet at T-junctions, so the two sides disagree.
    v = np.linspace(0.0, 1.0, 11)
    worst = 0.0
    for row in range(mesh.shape[0] - 1):
        for column in range(mesh.shape[1] - 2):
            left = bilinear_fn(quad(row, column), np.stack([np.ones_like(v), v], axis=1))
            right = bilinear_fn(quad(row, column + 1), np.stack([np.zeros_like(v), v], axis=1))
            worst = max(worst, float(np.abs(left - right).max()))
    # Both sides reduce to the same linear interpolation of the same two control points, but they
    # reach it by a different order of operations, so agreement is to the last bit rather than bit
    # for bit.
    assert worst < 1e-14, f"neighbouring cells disagree on their shared edge by {worst:.2e}"
