"""The white web's guarantees: the slot between holes, the narrowest band, and the thread.

Each test pins one claim the constructions make BEFORE anything is drawn, because the whole point of
them is that connectivity and slot width stop being measurements of a finished sheet.
"""

import numpy as np
import pytest
from shapely.geometry import LineString, Polygon, box

from legacy.katagami.scale_field import Element
from legacy.katagami.web import (
    _components as _as_parts,
    _folded_cells,
    build_cell_web_fn,
    build_gapped_web_fn,
    element_dark_fn,
    element_pale_fn,
    chain_thread_fn,
    element_dark_fn,
    extend_mesh_fn,
    fans_opened_fn,
    fit_leaf_fn,
    island_census_fn,
    island_tabs_fn,
    keep_largest_fn,
    keep_threaded_fn,
    mst_thread_fn,
    polyline_region_fn,
    resolve_dark_fn,
    bridge_faces_fn,
    concentric_leaves_fn,
    element_contours_fn,
    inscribed_disc_fn,
    lloyd_sites_fn,
    round_holes_fn,
    stroke_field_fn,
    stroke_weight_report_fn,
    thread_cost_fn,
    lattice_edges_fn,
    min_width_fn,
    opening_retention_fn,
    rim_for_slot_fn,
    rim_pinch_fn,
    territories_fn,
    thread_region_fn,
)
from legacy.katagami.cell_motifs import UNIT_MOTIFS, fit_motif_fn
from legacy.katagami.topology_art.leaf import ElementSpec, Leaf, leaf_polygon_fn

PAGE = (0.0, 0.0, 120.0, 90.0)


def _lattice(rows=6, cols=5, pitch=18.0, bond=0.5, angle=0.4, length=30.0):
    """A staggered lattice of leaves, deliberately overlapping their neighbours."""
    elements = []
    for row in range(rows):
        for column in range(cols):
            centre = np.array([12.0 + pitch * (row + bond * (column % 2)),
                               12.0 + pitch * column])
            elements.append(Element(Leaf(centre=centre, angle=angle, length=length,
                                         width=0.62, bow=0.22),
                                    "main", row * cols + column))
    return elements


# ---------------------------------------------------------------- the element


def test_dark_bands_partition_the_element():
    """Two rings give outline minus core; four give two disjoint annuli, not two discs."""
    leaf = Leaf(centre=np.zeros(2), angle=0.0, length=40.0, width=0.62, bow=0.22)
    outline = Polygon(leaf_polygon_fn(leaf, n=200))

    two = element_dark_fn(leaf, ElementSpec(rings=2, rim_abs=3.0, duty=1.0), samples=200)
    pale = element_pale_fn(leaf, ElementSpec(rings=2, rim_abs=3.0, duty=1.0), samples=200)
    assert two.area + pale.area == pytest.approx(outline.area, rel=1e-3)

    four = element_dark_fn(leaf, ElementSpec(rings=4, rim_abs=6.0, duty=0.5), samples=200)
    assert len(list(four.geoms)) == 2, "four rings must leave two separate dark bands"
    assert four.area < two.area, "spending a band on pale cannot darken the element"


def test_more_than_two_rings_encloses_an_island():
    """The double contour traps its own white, which is why it needs a thread at all."""
    leaf = Leaf(centre=np.array([60.0, 45.0]), angle=0.0, length=40.0, width=0.62, bow=0.22)
    dark = element_dark_fn(leaf, ElementSpec(rings=4, rim_abs=6.0, duty=0.5), samples=200)
    outer = max(dark.geoms, key=lambda part: part.area)
    assert len(outer.interiors) == 1, "the outer band must be an annulus"

    white = box(*PAGE).difference(dark)
    parts = sorted(getattr(white, "geoms", [white]), key=lambda part: part.area)
    assert len(parts) == 2, "the band must cut the white in two"
    assert parts[0].within(Polygon(outer.interiors[0])), "the small piece is the trapped island"


# ---------------------------------------------------------------- the rim, solved


@pytest.mark.parametrize("length", [10.0, 40.0, 250.0])
def test_the_rim_pinch_is_similarity_invariant(length):
    """The shortfall is a property of the proportions, which is what lets it be solved once."""
    spec = ElementSpec(rings=2, duty=1.0, core_shift=0.3)
    pinch = rim_pinch_fn(spec, width=0.62, bow=0.22)
    leaf = Leaf(centre=np.zeros(2), angle=0.7, length=length, width=0.62, bow=0.22)
    band = 0.06 * length
    pale = element_pale_fn(leaf, ElementSpec(rings=2, rim_abs=band, duty=1.0, core_shift=0.3),
                           samples=300)
    measured = min_width_fn(pale, upper=4.0 * band, tolerance=1e-3 * band) / band
    assert measured == pytest.approx(pinch, rel=0.05)


def test_the_solved_rim_delivers_the_slot_it_was_asked_for():
    """A nested core inset evenly still narrows at the tip, so the rim must be solved, not chosen."""
    spec = ElementSpec(rings=2, duty=1.0, core_shift=0.6)
    slot = 2.5
    rim = rim_for_slot_fn(spec, width=0.62, bow=0.22, slot=slot)
    assert rim > slot, "the tip pinch always costs something"
    leaf = Leaf(centre=np.zeros(2), angle=0.0, length=48.0, width=0.62, bow=0.22)
    pale = element_pale_fn(leaf, ElementSpec(rings=2, rim_abs=rim, duty=1.0, core_shift=0.6),
                           samples=300)
    assert min_width_fn(pale, upper=3.0 * slot, tolerance=0.01) == pytest.approx(slot, rel=0.06)


# ---------------------------------------------------------------- the width measure


@pytest.mark.parametrize("width", [1.0, 3.5, 7.0])
def test_minimum_width_reads_a_known_strip(width):
    """The measure has to return the width of something whose width is not in question."""
    strip = box(0.0, 0.0, 200.0, width)
    assert min_width_fn(strip, upper=12.0, tolerance=0.02) == pytest.approx(width, rel=0.05)


def test_retention_sees_a_thin_arm_the_component_count_cannot():
    """One piece carrying a thin arm is exactly the case a severance measure calls healthy."""
    body = box(0.0, 0.0, 20.0, 20.0)
    whisker = box(20.0, 9.8, 60.0, 10.2)
    joined = body.union(whisker)
    assert len(list(getattr(joined, "geoms", [joined]))) == 1
    assert opening_retention_fn(joined, 2.0) < 0.99
    assert min_width_fn(joined, upper=8.0) < 1.0


def test_retention_is_an_area_and_says_so():
    """Below its budget a thin feature is invisible, which is why the slot is claimed per CHANNEL.

    The measure weighs by area, so an arm holding well under one per cent of the white does not move
    it. That is the reason the strict claim is stated as a distance between holes and verified as one,
    with retention carried alongside as how much of the sheet sits at or above the slot.
    """
    body = box(0.0, 0.0, 200.0, 200.0)
    hair = box(200.0, 99.9, 210.0, 100.1)
    assert opening_retention_fn(body.union(hair), 2.0) > 0.999


# ---------------------------------------------------------------- territories


def test_territories_keep_the_slot_from_each_other_and_the_page():
    """Half a slot off each of two cells sharing an edge is the whole guarantee."""
    sites = np.array([[x, y] for x in (20.0, 45.0, 70.0, 95.0) for y in (20.0, 45.0, 70.0)])
    clearance = 4.0
    cells = territories_fn(sites, page=PAGE, clearance=clearance, frame=clearance)
    assert all(not cell.is_empty for cell in cells)
    for one in range(len(cells)):
        for other in range(one + 1, len(cells)):
            gap = cells[one].distance(cells[other])
            assert gap >= clearance - 1e-6, f"cells {one},{other} only {gap:.3f} apart"
    frame = box(*PAGE).difference(box(*PAGE).buffer(-clearance))
    for cell in cells:
        assert cell.disjoint(frame.buffer(-1e-6)), "a framed cell may not reach the page edge"


def test_fitting_shrinks_only_as_far_as_it_must():
    """A leaf already inside its territory is returned untouched; the arcs are never resampled."""
    territory = box(0.0, 0.0, 60.0, 60.0)
    inside = Leaf(centre=np.array([30.0, 30.0]), angle=0.3, length=20.0, width=0.5, bow=0.2)
    assert fit_leaf_fn(inside, territory) is inside

    big = Leaf(centre=np.array([30.0, 30.0]), angle=0.3, length=200.0, width=0.5, bow=0.2)
    fitted = fit_leaf_fn(big, territory)
    assert fitted is not None and fitted.length < big.length
    assert fitted.width == big.width and fitted.bow == big.bow, "only a similarity is allowed"
    assert territory.contains(Polygon(leaf_polygon_fn(fitted, n=200)))


# ---------------------------------------------------------------- the constructions


def test_gapped_occlusion_holds_the_slot_between_every_pair_of_holes():
    """Growing each occluder by half the slot is what makes the channel a decision."""
    elements = _lattice()
    slot = 3.0
    spec = ElementSpec(rings=2, rim_abs=rim_for_slot_fn(
        ElementSpec(rings=2, duty=1.0, core_shift=0.3), width=0.62, bow=0.22, slot=slot),
        duty=1.0, core_shift=0.3)
    paper, holes, report = build_gapped_web_fn(elements, {"main": spec}, page=PAGE, clearance=slot)

    assert report["rim_ok"], "the element's own border must not be the narrowest white"
    assert report["islands_dropped"] == 0
    live = [hole for hole in holes if not hole.is_empty]
    worst = min(live[one].distance(live[other])
                for one in range(len(live)) for other in range(one + 1, len(live)))
    assert worst >= slot - 1e-6, f"two holes came within {worst:.3f} of a {slot} slot"
    # Pointwise width cannot be claimed and is not: wherever three holes meet at an angle the white
    # between them tapers to a corner, so a little of it is thinner than the slot by construction.
    # What is claimed is the channel, and almost all of the area sits at or above it. The fraction
    # depends on how much of the element the border takes: on this deliberately small lattice the
    # rim is a seventh of the leaf, against a fourteenth on the A3 sheets, where it reads 99.7%.
    assert opening_retention_fn(paper, slot) > 0.95


def test_a_cell_web_is_one_piece_whatever_the_element_does():
    """Confinement, not overlap depth, is what decides the piece count here."""
    spec = ElementSpec(rings=1, rim=0.0, invert=True)
    for size in (12.0, 30.0, 90.0):
        elements = _lattice(length=size)
        paper, holes, skeleton, report = build_cell_web_fn(
            elements, {"main": spec}, page=PAGE, clearance=3.0, mode="crop", frame=3.0)
        assert len(list(getattr(paper, "geoms", [paper]))) == 1, f"length {size} broke the sheet"
        assert report["islands_dropped"] == 0
        assert min_width_fn(skeleton, upper=9.0) == pytest.approx(3.0, rel=0.08)


# ---------------------------------------------------------------- the thread


def test_the_grid_thread_is_connected_and_regular():
    """A spanning subgraph of the lattice, with one valence in the body of the sheet."""
    rows, cols = 6, 5
    tile_of = np.arange(rows * cols)
    edges, report = lattice_edges_fn(tile_of, (rows, cols), mode="grid")
    assert report["thread_connected"]
    assert report["n_edges"] == rows * (cols - 1) + cols * (rows - 1)
    assert report["degree_max"] == 4 and report["degree_min"] == 2
    interior = [d for tile, d in enumerate(
        [sum(tile in edge for edge in edges) for tile in tile_of])
        if 0 < tile // cols < rows - 1 and 0 < tile % cols < cols - 1]
    assert set(interior) == {4}, "an interior site must meet exactly four others"


def test_the_thread_width_is_the_bridge_it_promises():
    """Round caps, so a shallow bend does not notch the strut below its own width."""
    sites = np.array([[10.0, 10.0], [50.0, 12.0], [90.0, 40.0]])
    region = thread_region_fn(sites, [(0, 1), (1, 2)], width=2.6, page=PAGE)
    assert min_width_fn(region, upper=8.0, tolerance=0.02) == pytest.approx(2.6, rel=0.05)


def test_keeping_the_threaded_part_drops_exactly_what_hangs_free():
    held = box(0.0, 0.0, 20.0, 20.0)
    loose = box(50.0, 50.0, 60.0, 60.0)
    kept, report = keep_threaded_fn(held.union(loose), box(5.0, 5.0, 15.0, 15.0))
    assert report["islands_dropped"] == 1
    assert report["islands_area"] == pytest.approx(loose.area)
    assert kept.area == pytest.approx(held.area)


# ---------------------------------------------------------------- the armature (phase 9)


def _four_band(slot=2.5, length=48.0, centre=(60.0, 45.0)):
    """One double-contour element: four bands, so it traps a pale island."""
    probe = ElementSpec(rings=4, duty=0.5, core_shift=0.35)
    rim = rim_for_slot_fn(probe, width=0.62, bow=0.22, slot=slot)
    spec = ElementSpec(rings=4, rim_abs=rim, duty=0.5, core_shift=0.35)
    leaf = Leaf(centre=np.array(centre, float), angle=0.4, length=length, width=0.62, bow=0.22)
    return Element(leaf, "main", 0), spec


def test_the_census_finds_the_trapped_island_and_names_its_host():
    element, spec = _four_band()
    paper = box(*PAGE).difference(element_dark_fn(element.leaf, spec, samples=200))
    report, islands = island_census_fn(paper, [element], {"main": spec})
    assert report["n_islands"] == 1 and report["n_hosts"] == 1
    assert report["islands_unhosted"] == 0
    assert islands[0][1] == 0, "the island belongs to the only element on the sheet"


def test_one_nick_per_island_is_sufficient():
    """The whole nesting hangs off a single radial cut, because every band contains the centre."""
    elements = _lattice(rows=4, cols=4, length=26.0)
    probe = ElementSpec(rings=4, duty=0.5, core_shift=0.35)
    spec = ElementSpec(rings=4, rim_abs=rim_for_slot_fn(probe, width=0.62, bow=0.22, slot=2.0),
                       duty=0.5, core_shift=0.35)
    specs = {"main": spec}
    paper = box(*PAGE).difference(resolve_dark_fn(elements, specs, page=PAGE))
    report, islands = island_census_fn(paper, elements, specs)
    assert report["n_islands"] > 0, "the fixture must actually trap something"

    sheet = max(_as_parts(paper), key=lambda part: part.area)
    tabs, tab_report = island_tabs_fn(islands, elements, sheet, width=2.0, page=PAGE)
    assert tab_report["n_tabs"] == report["n_islands"], "one nick per island, no more"
    held, dropped = keep_largest_fn(paper.union(tabs))
    assert len(_as_parts(held)) == 1 and dropped["islands_dropped"] == 0


def test_one_nick_per_island_is_an_upper_bound_and_not_the_minimum():
    """Sufficient, but redundant: an overlapping field lets one cut serve several islands.

    A nick runs radially through its host element, and the elements overlap, so the cut frequently
    crosses a neighbour's island on the way out. Measured on this fixture only 8 of 19 nicks are
    individually load bearing. One per island is therefore a bound that always works rather than the
    smallest armature that would.
    """
    elements = _lattice(rows=4, cols=4, length=26.0)
    probe = ElementSpec(rings=4, duty=0.5, core_shift=0.35)
    spec = ElementSpec(rings=4, rim_abs=rim_for_slot_fn(probe, width=0.62, bow=0.22, slot=2.0),
                       duty=0.5, core_shift=0.35)
    specs = {"main": spec}
    paper = box(*PAGE).difference(resolve_dark_fn(elements, specs, page=PAGE))
    census, islands = island_census_fn(paper, elements, specs)
    sheet = max(_as_parts(paper), key=lambda part: part.area)

    # Collectively necessary: with no armature at all every island is loose.
    assert keep_largest_fn(paper)[1]["islands_dropped"] == census["n_islands"]

    load_bearing = 0
    for drop in range(len(islands)):
        tabs, _ = island_tabs_fn(islands[:drop] + islands[drop + 1:], elements, sheet,
                                 width=2.0, page=PAGE)
        load_bearing += keep_largest_fn(paper.union(tabs))[1]["islands_dropped"] == 1
    assert 0 < load_bearing < len(islands), (
        f"{load_bearing} of {len(islands)} nicks load bearing: some must matter, and on an "
        "overlapping field some must not")


def test_the_spanning_tree_is_the_cheapest_connected_armature():
    sites = np.array([[x, y] for x in (20.0, 45.0, 70.0, 95.0) for y in (20.0, 45.0, 70.0)])
    edges, report = mst_thread_fn(sites)
    assert report["n_edges"] == len(sites) - 1
    assert report["thread_connected"]
    grid_edges, grid_report = lattice_edges_fn(np.arange(len(sites)), (4, 3), mode="grid")
    assert len(edges) < len(grid_edges), "a tree must be cheaper than the lattice it spans"


def test_the_chain_armature_follows_the_tiling_and_holds_its_width():
    from legacy.katagami.katagami import load_flat_tiles_fn

    tiles, meta = load_flat_tiles_fn(synthetic=(6, 5), amplitude=1.2, seed=3, field_kind="banded")
    polylines, report = chain_thread_fn(tiles)
    assert report["n_chains"] > 0
    assert report["chain_cells_max"] >= report["chain_cells_median"]
    region = polyline_region_fn(polylines, width=2.6)
    assert min_width_fn(region, upper=8.0, tolerance=0.02) == pytest.approx(2.6, rel=0.06)


def test_extending_the_mesh_keeps_the_interior_and_reaches_past_it():
    """Centroids are interior points, so an unextended net leaves the border with no armature."""
    mesh = np.stack(np.meshgrid(np.linspace(0.0, 4.0, 5), np.linspace(0.0, 3.0, 4)), axis=-1)
    grown = extend_mesh_fn(mesh, rings=1)
    assert grown.shape == (mesh.shape[0] + 2, mesh.shape[1] + 2, 2)
    assert np.allclose(grown[1:-1, 1:-1], mesh), "the interior may not move"
    assert grown[..., 0].min() < mesh[..., 0].min() and grown[..., 0].max() > mesh[..., 0].max()
    assert _folded_cells(grown) == 0, "a regular mesh must stay convex when padded"


def test_the_armature_opens_the_closed_contour_it_crosses():
    """A nested comb draws closed rings; a cut stencil's arcs terminate. The armature is the cut."""
    element, spec = _four_band()
    specs = {"main": spec}
    centre = np.asarray(element.leaf.centre, float)
    axis = np.array([np.cos(element.leaf.angle), np.sin(element.leaf.angle)])
    across = LineString([centre - 1.2 * element.leaf.length * axis,
                         centre + 1.2 * element.leaf.length * axis]).buffer(1.5)
    assert fans_opened_fn([element], specs, across)["rings_opened"] == 1

    far = box(0.0, 0.0, 4.0, 4.0)
    missed = fans_opened_fn([element], specs, far)
    assert missed["rings_closed"] == 1 and missed["rings_opened"] == 0


def test_an_armature_inside_existing_white_costs_nothing():
    """The ideal a routing aims at: the guarantee is already drawn."""
    paper = box(0.0, 0.0, 60.0, 60.0)
    inside = box(10.0, 10.0, 20.0, 20.0)
    assert thread_cost_fn(paper, inside, page=PAGE) == pytest.approx(0.0)
    outside = box(80.0, 10.0, 90.0, 20.0)
    assert thread_cost_fn(paper, outside, page=PAGE) > 0.0


# ---------------------------------------------------------------- the line register


def test_the_even_rule_holds_its_weight_and_the_brush_rule_does_not():
    """Same paths, opposite weight rules. The difference is the tip, and it is measurable."""
    leaf = Leaf(centre=np.array([60.0, 45.0]), angle=0.4, length=48.0, width=0.62, bow=0.22)
    element = Element(leaf, "main", 0)
    width = 2.5

    even = stroke_field_fn([element], page=PAGE, width=width, kind="even")
    assert stroke_weight_report_fn(even, width=width)["ink_min_mm"] == pytest.approx(width,
                                                                                    rel=0.05)
    assert stroke_weight_report_fn(even, width=width)["ink_at_weight"] > 0.95

    varying = stroke_field_fn([element], page=PAGE, width=width, kind="varying")
    report = stroke_weight_report_fn(varying, width=width)
    assert report["ink_min_mm"] < 0.9 * width, "a brush stroke must thin where the flanks converge"
    assert report["ink_at_weight"] < 0.95


def test_the_offset_rule_keeps_a_constant_gap_and_the_leaf_rule_pinches():
    """Two ways to space concentric contours, and they differ at the tip by the known factor."""
    leaf = Leaf(centre=np.zeros(2), angle=0.0, length=80.0, width=0.62, bow=0.22)

    offset = element_contours_fn(leaf, spacing=6.0, count=3, rule="offset", samples=300)
    assert len(offset) == 3
    for outer, inner in zip(offset, offset[1:]):
        gap = Polygon(outer).exterior.distance(Polygon(inner).exterior)
        assert gap == pytest.approx(6.0, rel=0.06), "a geometric offset must hold its spacing"

    # The leaf rule insets absolutely in both dimensions, but the flanks converge at the tip, so the
    # gap measured there comes out well under nominal -- the same pinch the rim has.
    nested = element_contours_fn(leaf, spacing=6.0, count=3, rule="leaf", samples=300)
    worst = min(Polygon(outer).exterior.distance(Polygon(inner).exterior)
                for outer, inner in zip(nested, nested[1:]))
    assert worst < 0.85 * 6.0


def test_occlusion_is_what_keeps_the_ground_connected():
    """A planar arrangement of closed loops has one face per cell; terminating the strokes opens it."""
    elements = _lattice(rows=4, cols=4, length=30.0)
    mesh = stroke_field_fn(elements, page=PAGE, width=2.0, kind="even", occlude=False)
    drawing = stroke_field_fn(elements, page=PAGE, width=2.0, kind="even", occlude=True)
    faces_mesh = len(_as_parts(box(*PAGE).difference(mesh)))
    faces_drawing = len(_as_parts(box(*PAGE).difference(drawing)))
    assert faces_mesh > faces_drawing, "closed loops must divide the ground more than open ones do"


def test_bridging_faces_joins_the_ground_without_erasing_the_drawing():
    """Bridge each face to its NEAREST NEIGHBOUR: a loop drawing has no dominant face to aim at."""
    elements = _lattice(rows=4, cols=4, length=30.0)
    width = 2.0
    ink = stroke_field_fn(elements, page=PAGE, width=width, kind="even", occlude=True)
    before = len(_as_parts(box(*PAGE).difference(ink)))
    assert before > 1, "the fixture must actually divide the ground"

    bridged, report = bridge_faces_fn(ink, page=PAGE, width=width)
    assert report["faces_left"] == 1, "the ground must come out as one piece"
    # The guarantee has to be cheap. Sizing a gap from the distance to the LARGEST face instead of
    # the nearest one erased the drawing outright, so this is the regression that catches it.
    assert bridged.area > 0.85 * ink.area, "bridging must not consume the drawing"


def test_a_stroke_that_is_the_paper_needs_the_opposite_device():
    """Stroke as paper strands whole loops; stroke as slot strands faces of the ground."""
    apart = [Element(Leaf(centre=np.array([25.0 + 40.0 * i, 45.0]), angle=0.0, length=22.0,
                          width=0.62, bow=0.22), "main", i) for i in range(3)]
    ink = stroke_field_fn(apart, page=PAGE, width=2.0, kind="even")
    assert len(_as_parts(ink)) == 3, "loops that do not touch are three pieces of paper"

    sites = np.array([element.leaf.centre for element in apart], float)
    armature = thread_region_fn(sites, [(0, 1), (1, 2)], width=2.0, page=PAGE)
    joined, dropped = keep_largest_fn(ink.union(armature))
    assert dropped["islands_dropped"] == 0 and len(_as_parts(joined)) == 1


# ---------------------------------------------------------------- the cell web


def _sites_grid(nx=5, ny=4, pitch=22.0):
    return np.array([[14.0 + pitch * i, 12.0 + pitch * j] for i in range(nx) for j in range(ny)])


def test_a_breathing_slot_is_guaranteed_by_its_SMALLEST_value():
    """Two neighbours leave (c_i + c_j)/2, so an array clearance is only as strong as its minimum."""
    sites = _sites_grid()
    slots = np.linspace(2.0, 6.0, len(sites))
    cells = territories_fn(sites, page=PAGE, clearance=slots, frame=float(slots.min()))
    live = [cell for cell in cells if not cell.is_empty]
    worst = min(live[one].distance(live[other])
                for one in range(len(live)) for other in range(one + 1, len(live)))
    assert worst >= slots.min() - 1e-6, f"a pair came within {worst:.3f} of a {slots.min()} floor"
    # And it really does breathe: the widest-slotted cell must have given up more than the narrowest.
    assert cells[-1].area / cells[0].area < 1.0


def test_rounding_a_hole_can_only_widen_the_channels():
    """An opening is a subset, so the guarantee cannot be weakened by softening a corner."""
    cell = box(0.0, 0.0, 40.0, 30.0)
    assert round_holes_fn(cell, 0.0).equals(cell)
    rounded = round_holes_fn(cell, 5.0)
    assert rounded.within(cell.buffer(1e-9)) and rounded.area < cell.area
    # A square's corners become quarter-circles of the radius, so the area lost is the corner excess.
    assert cell.area - rounded.area == pytest.approx(4 * 25.0 - np.pi * 25.0, rel=0.05)


def test_relaxing_the_sites_evens_the_cells_out():
    """Lloyd is the other end of the size-contrast dial, so it has to actually reduce the spread."""
    rng = np.random.default_rng(0)
    sites = np.column_stack([rng.uniform(12.0, 108.0, 40), rng.uniform(10.0, 80.0, 40)])
    relaxed, report = lloyd_sites_fn(sites, page=PAGE, passes=8)
    assert report["area_spread_after"] < report["area_spread_before"]
    assert relaxed.shape == sites.shape


def test_the_inscribed_disc_sits_inside_its_cell_and_touches_it():
    cell = box(0.0, 0.0, 40.0, 24.0)
    disc = inscribed_disc_fn(cell, np.array([20.0, 12.0]))
    assert disc.within(cell.buffer(1e-9))
    # The largest circle in a 40x24 rectangle has radius 12, so it must touch the long sides.
    assert cell.boundary.distance(disc.centroid) == pytest.approx(12.0, rel=1e-3)
    assert disc.area == pytest.approx(np.pi * 144.0, rel=0.01)


def test_the_fill_dial_opens_the_web_without_moving_the_slot():
    """Filling less widens every channel, but the floor the sheet is guaranteed at does not move."""
    elements = [Element(Leaf(centre=site, angle=0.3, length=26.0, width=0.62, bow=0.22),
                        "main", index) for index, site in enumerate(_sites_grid())]
    spec = ElementSpec(rings=1, rim=0.0, invert=True)
    slot = 3.0
    full = build_cell_web_fn(elements, {"main": spec}, page=PAGE, clearance=slot, mode="crop",
                             frame=slot)[3]
    half = build_cell_web_fn(elements, {"main": spec}, page=PAGE, clearance=slot, mode="crop",
                             frame=slot, fill=0.5)[3]
    assert half["gap_median_mm"] > full["gap_median_mm"], "less fill must open the web"
    assert half["gap_min_mm"] >= slot - 1e-6 and full["gap_min_mm"] >= slot - 1e-6
    # `at_clearance` IS the fill axis: it collapses as the holes pull away from their cell walls.
    assert half["at_clearance"] < full["at_clearance"]


# ---------------------------------------------------------------- the motif vocabulary


@pytest.mark.parametrize("name", sorted(set(UNIT_MOTIFS) - {"ring"}))
def test_no_motif_encloses_its_own_white(name):
    """A motif with a hole in it sheds one loose piece per cell, and the picture will not show it.

    The rosette was exactly this: petals ringing an empty centre dropped 255 islands on a 280-cell
    sheet and looked perfectly healthy. Only `ring` is allowed to enclose anything, and it is in the
    registry precisely as the control.
    """
    shape = UNIT_MOTIFS[name]()
    parts = list(getattr(shape, "geoms", [shape]))
    assert all(not part.interiors for part in parts), f"{name} encloses white"


def test_the_ring_control_really_does_enclose_white():
    assert len(UNIT_MOTIFS["ring"]().interiors) == 1


@pytest.mark.parametrize("name", sorted(UNIT_MOTIFS))
def test_every_motif_stays_inside_its_cell_at_any_reach(name):
    """The guarantee never looks at the shape, only at whether it stayed in its own territory."""
    cell = Polygon([(0.0, 0.0), (30.0, 2.0), (28.0, 24.0), (4.0, 20.0)])
    site = np.array([15.0, 12.0])
    for reach in (0.6, 1.0, 2.5):
        hole = fit_motif_fn(name, cell, site=site, angle=0.5, reach=reach)
        assert hole.within(cell.buffer(1e-6)), f"{name} left its cell at reach {reach}"
    # Reach 1 is the inscribed size, so it must touch the cell but not exceed it.
    inscribed = fit_motif_fn(name, cell, site=site, angle=0.5, reach=1.0)
    assert inscribed.area > 0.0
    assert cell.exterior.distance(inscribed) < 0.5, f"{name} did not grow to meet its cell"
