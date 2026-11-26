"""Katagami composition: the corner-port rule, the seam it buys, and an exact SVG round-trip."""

import os

import numpy as np
import pytest

from nff.topology.motif import CORNERS, MOTIFS, build_motif_fn, check_corner_ports_fn
from nff.topology.quad_warp import BezierPath, bilinear_fn, path_points_fn, warp_polyline_fn
from nff.topology.svg_export import fit_to_page_fn, path_data_fn, write_svg_fn

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGMAT_RUN = os.path.join(REPO, "data/outputs/runs/run_20260806_183646_figmat_1_deploy")


@pytest.mark.parametrize("name", sorted(MOTIFS))
@pytest.mark.parametrize("detail", [1, 2, 3])
def test_every_motif_obeys_the_corner_port_rule(name, detail):
    """A stroke may touch the cell boundary only at a corner; mid-edge points have no partner."""
    report = check_corner_ports_fn(build_motif_fn(name, detail=detail))
    assert report["ok"], (
        f"{name} detail {detail}: {len(report['mid_edge'])} mid-edge points, "
        f"{len(report['outside'])} outside the cell"
    )


@pytest.mark.parametrize("name", sorted(set(MOTIFS) - {"rosette"}))
def test_connecting_motifs_reach_every_corner(name):
    """A motif meant to tile the sheet must actually use all four corners, or the field breaks up."""
    assert check_corner_ports_fn(build_motif_fn(name))["ports"] == (0, 1, 2, 3)


def test_detail_levels_share_identical_corner_ports():
    """Level of detail may vary tile to tile only because the ports do not depend on it."""
    for name in sorted(set(MOTIFS) - {"rosette"}):
        ports = {check_corner_ports_fn(build_motif_fn(name, detail=d))["ports"] for d in (1, 2, 3)}
        assert len(ports) == 1, f"{name} changes its ports with detail: {ports}"


def test_page_fit_preserves_aspect_and_lands_inside_the_page():
    matrix, offset = fit_to_page_fn((0.0, 0.0, 15.0, 10.0), (420.0, 297.0), margin_mm=12.0)
    assert matrix[0, 0] == pytest.approx(matrix[1, 1])
    assert matrix[0, 1] == 0.0 and matrix[1, 0] == 0.0
    corners = np.array([[0.0, 0.0], [15.0, 0.0], [15.0, 10.0], [0.0, 10.0]]) @ matrix.T + offset
    assert corners.min() >= 12.0 - 1e-9
    assert corners[:, 0].max() <= 420.0 - 12.0 + 1e-9
    assert corners[:, 1].max() <= 297.0 - 12.0 + 1e-9


def test_cubic_elevation_is_exact():
    """``--curve c`` must be a degree elevation, not a refit."""
    b0, b1, b2 = np.array([0.0, 0.0]), np.array([1.0, 2.0]), np.array([3.0, 0.5])
    path = BezierPath(start=b0, segments=(("Q", np.array([b1, b2])),))
    t = np.linspace(0, 1, 101)[:, None]
    quad = (1 - t) ** 2 * b0 + 2 * (1 - t) * t * b1 + t**2 * b2
    c1, c2 = b0 + (2 / 3) * (b1 - b0), b2 + (2 / 3) * (b1 - b2)
    cubic = (1 - t) ** 3 * b0 + 3 * (1 - t) ** 2 * t * c1 + 3 * (1 - t) * t**2 * c2 + t**3 * b2
    assert np.abs(quad - cubic).max() < 1e-13
    assert path_data_fn(path, curve="c").startswith("M0,0 C")


def test_svg_roundtrips_through_the_repo_parser(tmp_path):
    """Emit with ``--curve c`` and re-read with the project's own SVG parser.

    The parser supports ``M/L/C/Z`` but not ``Q``, which is exactly why the cubic option exists.
    """
    shield_target = pytest.importorskip("nff.closed.shield_target")
    quad = np.array([[0.0, 0.0], [2.0, 0.3], [1.7, 1.6], [0.2, 1.2]])
    path = warp_polyline_fn(quad, np.array([[0.0, 0.0], [0.6, 0.35], [1.0, 1.0]]))
    out = tmp_path / "roundtrip.svg"
    write_svg_fn(str(out), {"motif": [path]}, width_mm=10.0, height_mm=10.0,
                 y_up=False, curve="c", precision=9)

    parsed, _ = shield_target.parse_svg_paths(str(out))
    assert len(parsed) == 1
    got = parsed[0]["path"].vertices * np.array([1.0, -1.0])

    expected = [np.asarray(path.start, float)]
    current = expected[0]
    for _, points in path.segments:
        control, end = points
        expected.extend([current + (2 / 3) * (control - current),
                         end + (2 / 3) * (control - end), end])
        current = end
    assert np.abs(got - np.array(expected)).max() < 1e-8


# ── integration against the real sheet ───────────────────────────────────────────────────────

pytestmark_run = pytest.mark.skipif(
    not os.path.isdir(FIGMAT_RUN), reason="figmat run folder not present"
)


@pytest.fixture(scope="module")
def figmat_tiles():
    if not os.path.isdir(FIGMAT_RUN):
        pytest.skip("figmat run folder not present")
    from nff.closed.katagami import load_flat_tiles_fn

    tiles, meta = load_flat_tiles_fn(run_dir=FIGMAT_RUN)
    return tiles, meta


def test_flat_tiles_partition_the_sheet(figmat_tiles):
    """Zero gap and zero overlap: the reason a motif can be laid across the whole sheet."""
    tiles, meta = figmat_tiles
    span = tiles.reshape(-1, 2).max(axis=0) - tiles.reshape(-1, 2).min(axis=0)
    assert float(meta["areas"].sum()) == pytest.approx(float(span[0] * span[1]), rel=1e-9)


def test_interior_corners_are_shared_by_exactly_two_tiles(figmat_tiles):
    """The structural fact the whole design rests on."""
    from nff.closed.katagami import shared_corners_fn

    tiles, _ = figmat_tiles
    valence = shared_corners_fn(tiles)["valence"]
    assert set(np.unique(valence).tolist()) <= {1, 2}
    assert int((valence == 1).sum()) == 4  # only the four sheet corners


def test_corner_join_residual_is_machine_zero(figmat_tiles):
    """The seam test: strokes terminate at corners, so this bounds how far two of them can land."""
    from nff.closed.katagami import corner_join_residual_fn

    tiles, _ = figmat_tiles
    span = tiles.reshape(-1, 2).max(axis=0) - tiles.reshape(-1, 2).min(axis=0)
    assert corner_join_residual_fn(tiles) < 1e-12 * float(np.hypot(*span))


def test_motif_never_leaves_its_own_tile(figmat_tiles):
    """Convexity guarantees containment; check it on the real, badly skewed tiles."""
    shapely = pytest.importorskip("shapely.geometry")
    tiles, _ = figmat_tiles
    motif = build_motif_fn("shippo")
    for quad in tiles[::7]:
        cell = shapely.Polygon(quad).buffer(1e-9 * float(np.abs(quad).max()))
        for stroke in motif.strokes:
            pts = path_points_fn(
                warp_polyline_fn(quad, stroke.points, closed=stroke.closed), per_segment=6)
            assert cell.covers(shapely.LineString(pts))


def test_ribbon_ports_land_on_the_shared_world_points(figmat_tiles):
    """The phase-2 seam invariant: a junction closes exactly at NONZERO width.

    Position, direction and half-width at a corner all belong to the corner, so both tiles must
    produce the same pair of outline endpoints, ``p +/- w * perp(d)``. Checked on the two tiles that
    actually share a corner on the real sheet.
    """
    from nff.closed.katagami import diagonal_chains_fn
    from nff.topology.motif import CORNERS
    from nff.topology.ribbon import (
        build_chord_fn, build_ribbon_fn, ribbon_port_points_fn, width_profile_fn,
    )

    tiles, _ = figmat_tiles
    structure = diagonal_chains_fn(tiles)
    positions = structure["corners"]["positions"]
    edges = structure["edges"]
    scale = float(np.abs(tiles).max())

    half_width = 0.03
    worst = 0.0
    checked = 0
    for chain in structure["chains"]:
        nodes, walked = chain["nodes"], chain["edges"]
        tangents, _ = _chain_frames_for_test(positions[nodes], chain["closed"])
        for step, edge_index in enumerate(walked):
            node_a, _, tile, (corner_a, corner_b) = edges[edge_index]
            start, end = (corner_a, corner_b) if nodes[step] == node_a else (corner_b, corner_a)
            quad = tiles[tile]
            centre = build_chord_fn(quad, CORNERS[start], CORNERS[end],
                                    tangents[step], tangents[step + 1], n_samples=8)
            widths = width_profile_fn(9, half_width, half_width)
            outline = build_ribbon_fn(quad, centre, widths,
                                      end_directions=(tangents[step], tangents[step + 1]))
            world = bilinear_fn(quad, outline)
            got = np.array([world[0], world[-1]])           # the two port points at the start
            want = ribbon_port_points_fn(half_width, positions[nodes[step]], tangents[step])
            worst = max(worst, float(np.abs(np.sort(got, axis=0) - np.sort(want, axis=0)).max()))
            checked += 1
    assert checked > 100
    assert worst < 1e-9 * scale, f"port mismatch {worst:.3e}"


def _chain_frames_for_test(positions, closed):
    from nff.closed.katagami import _chain_frames

    return _chain_frames(positions, closed)


def test_diagonal_chains_are_long_and_cover_every_tile(figmat_tiles):
    """The chains are found, not invented: a 2-regular corner graph falls into few long chains."""
    from nff.closed.katagami import diagonal_chains_fn

    tiles, _ = figmat_tiles
    structure = diagonal_chains_fn(tiles)
    covered = sorted(e for chain in structure["chains"] for e in chain["edges"])
    assert covered == list(range(2 * len(tiles)))  # every diagonal used exactly once
    lengths = [len(c["edges"]) for c in structure["chains"]]
    assert len(lengths) < 30
    assert float(np.median(lengths)) >= 5.0
    assert max(lengths) >= len(tiles) // 2


def test_every_tile_puts_geometry_on_all_four_of_its_corners(figmat_tiles):
    """Each tile must reach each of its own corners, or the interlocking net has holes in it.

    Checked per tile on the warped geometry, not on the unit cell, so it also covers the claim that
    the bilinear map sends cell corners to quad corners exactly.
    """
    tiles, _ = figmat_tiles
    motif = build_motif_fn("shippo")
    worst = 0.0
    for quad in tiles:
        visited = []
        for stroke in motif.strokes:
            path = warp_polyline_fn(quad, stroke.points, closed=stroke.closed)
            visited.append(path.start)
            visited.extend(points[-1] for _, points in path.segments)
        visited = np.asarray(visited)
        gaps = np.linalg.norm(quad[:, None, :] - visited[None, :, :], axis=2).min(axis=1)
        worst = max(worst, float(gaps.max()))
    assert worst < 1e-12 * float(np.abs(tiles).max())


@pytest.mark.parametrize("kind", ["shear", "saddle", "pinch", "vortex2", "lens"])
def test_new_deformation_fields_build_a_usable_sheet(kind):
    """A field is only a design lever if the sheet it makes is still a valid tiling."""
    from nff.closed.katagami import build_r_field_fn, load_flat_tiles_fn

    logits = build_r_field_fn((9, 9), kind=kind, seed=1, amplitude=2.0)
    assert logits.shape == (9, 9)
    assert np.isfinite(logits).all()
    assert np.abs(logits).max() == pytest.approx(2.0)

    tiles, meta = load_flat_tiles_fn(synthetic=(8, 8), field_kind=kind, seed=1, amplitude=1.8)
    assert meta["n_tiles"] == 64
    edges = np.roll(tiles, -1, axis=1) - tiles
    cross = (np.roll(edges, 1, axis=1)[..., 0] * edges[..., 1]
             - np.roll(edges, 1, axis=1)[..., 1] * edges[..., 0])
    assert (cross > 0).all() or (cross < 0).all(), "a non-convex tile breaks the bilinear map"


def test_a_ground_population_leaves_the_main_packing_untouched():
    """The two scales must be comparable, so adding a ground may not move a single figure."""
    from nff.closed.scale_field import build_scale_field_fn
    from nff.closed.katagami import load_flat_tiles_fn

    tiles, _ = load_flat_tiles_fn(synthetic=(8, 6), field_kind="banded", seed=2, amplitude=1.8)
    plain, _ = build_scale_field_fn(tiles, density=0.5, size_frac=2.0, smoothing=6, seed=2)
    layered, report = build_scale_field_fn(tiles, density=0.5, size_frac=2.0, smoothing=6, seed=2,
                                           ground_scale=0.3)
    figures = [e for e in layered if e.population == "main"]
    assert report["n_ground"] > 0
    assert len(figures) == len(plain)
    for placed, alone in zip(figures, plain):
        assert np.allclose(placed.leaf.centre, alone.leaf.centre)
        assert placed.leaf.length == alone.leaf.length
        assert placed.leaf.angle == alone.leaf.angle
    # The ground goes down first, so no figure may be painted before any of it.
    assert all(e.population == "ground" for e in layered[:report["n_ground"]])


def test_element_width_follows_the_tile_when_asked_and_not_otherwise():
    from nff.closed.scale_field import build_scale_field_fn, tile_anisotropy_fn
    from nff.closed.katagami import load_flat_tiles_fn

    tiles, _ = load_flat_tiles_fn(synthetic=(8, 6), field_kind="lens", seed=2, amplitude=2.2)
    assert tile_anisotropy_fn(tiles).min() >= 1.0 - 1e-12

    _, flat = build_scale_field_fn(tiles, density=0.4, smoothing=6, seed=2, aniso_power=0.0)
    _, varied = build_scale_field_fn(tiles, density=0.4, smoothing=6, seed=2, aniso_power=1.0)
    assert flat["width_min"] == flat["width_max"]
    assert varied["width_max"] > 1.2 * varied["width_min"]


def _lens_tiles(shape=(8, 6)):
    from nff.closed.katagami import load_flat_tiles_fn

    tiles, meta = load_flat_tiles_fn(synthetic=shape, field_kind="lens", seed=3, amplitude=1.8)
    return tiles, (meta["M"], meta["N"])


def test_a_strict_lattice_puts_exactly_one_element_on_every_tile():
    """Regular alignment means regular INCIDENCE: each tile carries one element, once."""
    from nff.closed.scale_field import build_scale_field_fn

    tiles, grid = _lens_tiles()
    elements, report = build_scale_field_fn(tiles, grid=grid, placement="lattice",
                                            regularity=1.0, smoothing=6, seed=3)
    assert report["n_elements"] == len(tiles)
    assert sorted(e.tile for e in elements) == list(range(len(tiles)))


def test_the_lattice_needs_the_grid_and_says_so():
    from nff.closed.scale_field import build_scale_field_fn

    tiles, _ = _lens_tiles()
    with pytest.raises(ValueError, match="grid"):
        build_scale_field_fn(tiles, placement="lattice", smoothing=6)


def test_regularity_is_a_dial_from_the_lattice_toward_disorder():
    """Each step down must move elements further from their sites, or the dial does nothing."""
    from nff.closed.scale_field import build_scale_field_fn

    tiles, grid = _lens_tiles()
    sites = {e.tile: e.leaf.centre for e in build_scale_field_fn(
        tiles, grid=grid, placement="lattice", regularity=1.0, smoothing=6, seed=3)[0]}
    drift = []
    for regularity in (1.0, 0.8, 0.5, 0.0):
        elements, _ = build_scale_field_fn(tiles, grid=grid, placement="lattice",
                                           regularity=regularity, smoothing=6, seed=3)
        drift.append(float(np.mean([np.linalg.norm(e.leaf.centre - sites[e.tile])
                                    for e in elements])))
    assert drift[0] == 0.0
    assert all(one < other for one, other in zip(drift, drift[1:])), drift


def test_the_bond_offsets_alternate_rows_and_nothing_else():
    from nff.closed.scale_field import build_scale_field_fn

    tiles, grid = _lens_tiles()
    common = dict(grid=grid, placement="lattice", regularity=1.0, smoothing=6, seed=3)
    square = {e.tile: e.leaf.centre for e in build_scale_field_fn(tiles, bond=0.0, **common)[0]}
    shingle = {e.tile: e.leaf.centre for e in build_scale_field_fn(tiles, bond=1.0, **common)[0]}
    cols = grid[1]
    moved = [tile for tile in square if not np.allclose(square[tile], shingle[tile])]
    assert moved, "a half-cell bond must move something"
    assert all((tile // cols) % 2 == 1 for tile in moved), "only alternate rows may shift"


def test_placement_stays_on_poisson_by_default_even_when_a_grid_is_given():
    """The grid is information, not an instruction: existing renders must not change under it."""
    from nff.closed.scale_field import build_scale_field_fn

    tiles, grid = _lens_tiles()
    without, _ = build_scale_field_fn(tiles, density=0.5, smoothing=6, seed=3)
    with_grid, _ = build_scale_field_fn(tiles, density=0.5, smoothing=6, seed=3, grid=grid)
    assert len(without) == len(with_grid)
    for one, other in zip(without, with_grid):
        assert np.allclose(one.leaf.centre, other.leaf.centre)


def test_the_lattice_decides_which_elements_fuse_rather_than_discovering_it():
    """The lattice does not connect the sheet by itself; it makes the connection a decision.

    Overlap depth is the knob, and it has a WINDOW: too little and neighbours never meet, too much
    and their cores eat the web between them. What the lattice buys is that almost every contact is
    a lattice neighbour, so the connectivity can be reasoned about at all -- and that a tuned
    overlap then gives a single piece with a bridge worth cutting.
    """
    from nff.closed.scale_field import build_scale_field_fn, transform_elements_fn
    from nff.closed.stencil import measure_stencil_fn, resolve_paper_fn
    from nff.topology.leaf import ElementSpec
    from nff.topology.svg_export import fit_to_page_fn

    pytest.importorskip("shapely.geometry")
    tiles, grid = _lens_tiles((14, 10))
    page = (0.0, 0.0, 420.0, 297.0)
    spec = ElementSpec(rim=0.0, rim_abs=5.0, duty=1.0, core_shift=0.6)
    def score(size_frac, **kwargs):
        elements, report = build_scale_field_fn(tiles, grid=grid, size_frac=size_frac,
                                                smoothing=14, seed=3, **kwargs)
        matrix, offset = fit_to_page_fn(report["sheet_bounds"], (420.0, 297.0), margin_mm=0.0)
        placed = transform_elements_fn(elements, scale=float(matrix[0, 0]), offset=offset)
        paper, pieces = resolve_paper_fn(placed, {"main": spec}, page=page)
        return measure_stencil_fn(paper, pieces, page=page, grid=grid,
                                  tile_of=[e.tile for e in placed])

    scattered = score(2.3, placement="poisson", density=0.30)
    lattice = dict(placement="lattice", regularity=1.0, bond=1.0)
    assert score(2.3, **lattice)["lattice_incidence"] > scattered["lattice_incidence"]

    tuned = score(2.6, **lattice)
    assert tuned["n_components"] == 1
    assert tuned["min_bridge_mm"] > 1.5
    assert tuned["min_bridge_mm"] > scattered["min_bridge_mm"]

    # Both ends of the window fail, which is why the overlap has to be chosen rather than maximised.
    assert score(2.3, **lattice)["largest_share"] < 0.9  # neighbours never reach each other
    assert score(3.2, **lattice)["n_components"] > 1     # cores eat the web between them


@pytest.mark.parametrize("kind", ["swell", "terrace"])
def test_the_calm_fields_build_a_usable_sheet(kind):
    from nff.closed.katagami import build_r_field_fn, load_flat_tiles_fn

    logits = build_r_field_fn((9, 9), kind=kind, seed=1, amplitude=1.2)
    assert np.isfinite(logits).all()

    tiles, _ = load_flat_tiles_fn(synthetic=(20, 10), field_kind=kind, seed=3,
                                  amplitude=1.2, aspect=1.4)
    edges = np.roll(tiles, -1, axis=1) - tiles
    cross = (np.roll(edges, 1, axis=1)[..., 0] * edges[..., 1]
             - np.roll(edges, 1, axis=1)[..., 1] * edges[..., 0])
    assert (cross > 0).all() or (cross < 0).all()


def test_a_still_flow_comes_from_the_tile_pitch_not_from_a_swirl_free_field():
    """Removing the swirl from the r-field does NOT calm the flow, and this pins why.

    The heading is each tile's principal axis, so a defect appears wherever a tile lands
    near-isotropic -- whatever the field's formula looks like. At square pitch the gentle fields are
    no quieter than a two-vortex one. A tall pitch elongates every tile the same way and settles all
    of them at once, which is what the calm presets actually rely on.
    """
    from nff.closed.katagami import load_flat_tiles_fn
    from nff.closed.scale_field import build_scale_field_fn

    def opposed(kind, aspect):
        tiles, meta = load_flat_tiles_fn(synthetic=(20, 10), field_kind=kind, seed=3,
                                         amplitude=1.2, aspect=aspect)
        _, report = build_scale_field_fn(tiles, grid=(meta["M"], meta["N"]), placement="lattice",
                                         regularity=1.0, smoothing=30, seed=3)
        return report["opposed_neighbours"]

    assert opposed("swell", 1.0) > 4, "no angular term in the field buys nothing on its own"
    assert opposed("terrace", 1.0) > 4
    assert opposed("swell", 1.4) <= 2, "a tall pitch is what settles the field"
    assert opposed("terrace", 1.4) <= 2
    # `banded` is the one field quiet at any pitch, which is why it stays the fallback.
    assert opposed("banded", 1.0) == 0


def test_a_square_uniform_sheet_has_no_direction_but_a_tall_one_does():
    """The tile IS the direction sensor, and a square tile is isotropic.

    Worth pinning because the failure is silent: the axis comes back as numerical noise rather than
    as an error, and the sheet quietly fills with elements pointing at each other.
    """
    from nff.closed.katagami import load_flat_tiles_fn
    from nff.closed.scale_field import build_scale_field_fn, tile_anisotropy_fn

    def measure(aspect):
        tiles, meta = load_flat_tiles_fn(synthetic=(20, 14), field_kind="uniform", seed=3,
                                         amplitude=0.0, aspect=aspect)
        _, report = build_scale_field_fn(tiles, grid=(meta["M"], meta["N"]),
                                         placement="lattice", smoothing=30, seed=3)
        return tile_anisotropy_fn(tiles).max(), report["opposed_neighbours"]

    square_ratio, square_opposed = measure(1.0)
    tall_ratio, tall_opposed = measure(1.6)
    assert square_ratio == pytest.approx(1.0, abs=1e-9), "a square tile has no long axis"
    assert square_opposed > 0, "so its heading field is noise"
    assert tall_ratio > 1.5 and tall_opposed == 0, "a rectangular pitch settles it completely"


def test_the_outline_is_off_unless_a_panel_asks_for_it(tmp_path):
    """The regression that shipped: a default of 0.5 drew the rim on every panel that omitted it."""
    from nff.closed.scale_field import PALETTES, Element
    from nff.scripts.figures.katagami_contact_sheet import write_panel_svg_fn
    from nff.topology.leaf import ElementSpec, Leaf

    placed = [Element(Leaf(centre=np.array([50.0, 50.0]), angle=0.2, length=40.0), "main", 0)]
    specs = {"main": ElementSpec(rim=0.0, rim_abs=3.0, duty=1.0)}

    # The writer always emits a stroke-width from its own default style, so the question is
    # whether a stroke COLOUR was asked for, not whether the attribute is present.
    bare = tmp_path / "bare.svg"
    write_panel_svg_fn(placed, specs, str(bare))
    assert 'stroke="none"' in bare.read_text()

    webbed = tmp_path / "webbed.svg"
    write_panel_svg_fn(placed, specs, str(webbed), web=0.5)
    assert f'stroke="{PALETTES["sumi"]["edge"]}"' in webbed.read_text()
    assert bare.read_text() != webbed.read_text()


@pytest.mark.parametrize("palette", ["sumi", "kakishibu", "ai"])
def test_every_palette_reaches_the_vector_output(palette, tmp_path):
    from nff.closed.scale_field import PALETTES, Element
    from nff.scripts.figures.katagami_contact_sheet import write_panel_svg_fn
    from nff.topology.leaf import ElementSpec, Leaf

    placed = [Element(Leaf(centre=np.array([50.0, 50.0]), angle=0.2, length=40.0), "main", 0)]
    out = tmp_path / f"{palette}.svg"
    write_panel_svg_fn(placed, {"main": ElementSpec(rim=0.0, rim_abs=3.0, duty=1.0)}, str(out),
                       colours=PALETTES[palette])
    text = out.read_text()
    assert PALETTES[palette]["fill"] in text
    assert PALETTES[palette]["ground"] in text


def test_sparse_elements_cut_only_when_they_are_read_as_holes():
    """A sparse motif taken as paper is a field of islands; taken as holes it is one sheet."""
    pytest.importorskip("shapely.geometry")
    from nff.closed.katagami import load_flat_tiles_fn
    from nff.closed.scale_field import build_scale_field_fn, transform_elements_fn
    from nff.closed.stencil import measure_stencil_fn, resolve_paper_fn
    from nff.topology.leaf import ElementSpec
    from nff.topology.svg_export import fit_to_page_fn

    page = (0.0, 0.0, 420.0, 297.0)
    tiles, meta = load_flat_tiles_fn(synthetic=(21, 10), field_kind="uniform", seed=3,
                                     amplitude=0.0, aspect=1.5)
    elements, report = build_scale_field_fn(
        tiles, grid=(meta["M"], meta["N"]), placement="lattice", regularity=0.86, bond=1.0,
        size_frac=1.15, leaf_width=0.22, leaf_bow=0.05, smoothing=30, seed=3)
    matrix, offset = fit_to_page_fn(report["sheet_bounds"], (420.0, 297.0), margin_mm=0.0)
    placed = transform_elements_fn(elements, scale=float(matrix[0, 0]), offset=offset)

    needles = {"main": ElementSpec(rings=1, rim=0.0, invert=True)}
    as_paper, pieces = resolve_paper_fn(placed, needles, page=page, ground="hole")
    assert as_paper.is_empty, "inverted elements contribute no paper of their own"

    as_holes, pieces = resolve_paper_fn(placed, needles, page=page, ground="paper")
    ground = measure_stencil_fn(as_holes, pieces, page=page)
    assert ground["n_components"] == 1, "read as holes, the ground is a single sheet"
    # Fewer holes than needles: jitter lets some touch and merge, and the bleed carries others off
    # the page. What matters is that the great majority survive as separate cuts.
    assert 0.7 * len(placed) < ground["n_holes"] <= len(placed)
    assert ground["coverage"] > 0.5, "a komon ground is mostly paper"
