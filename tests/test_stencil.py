"""The painted field resolved into the object a blade would leave behind."""

import numpy as np
import pytest

shapely = pytest.importorskip("shapely.geometry")

from shapely.geometry import Polygon, box  # noqa: E402
from shapely.ops import unary_union  # noqa: E402

from nff.closed.scale_field import Element  # noqa: E402
from nff.closed.stencil import (  # noqa: E402
    contact_graph_fn,
    cut_paths_fn,
    measure_stencil_fn,
    min_bridge_fn,
    prune_dust_fn,
    resolve_paper_fn,
    solve_slot_for_bridge_fn,
)
from nff.topology.leaf import ElementSpec, Leaf, element_layers_fn, leaf_polygon_fn  # noqa: E402
from nff.topology.quad_warp import CompoundPath  # noqa: E402
from nff.topology.svg_export import path_data_fn  # noqa: E402

PAGE = (0.0, 0.0, 100.0, 100.0)
SPEC = ElementSpec(rim=0.0, rim_abs=3.0, duty=1.0)


def _field(*leaves):
    return [Element(leaf, "main", index) for index, leaf in enumerate(leaves)]


def test_the_resolver_agrees_with_asking_which_layer_is_on_top():
    """An independent check: for a point, find the topmost element covering it and read its ink.

    The resolver works by subtracting later outlines; this decides the same question by direct
    lookup, so agreement is evidence about the construction rather than a restatement of it.
    """
    leaves = [Leaf(centre=np.array([40.0, 50.0]), angle=0.3, length=40.0, width=0.62, bow=0.22),
              Leaf(centre=np.array([58.0, 55.0]), angle=1.1, length=36.0, width=0.62, bow=0.22),
              Leaf(centre=np.array([50.0, 38.0]), angle=-0.6, length=44.0, width=0.62, bow=0.22)]
    elements = _field(*leaves)
    paper, _ = resolve_paper_fn(elements, {"main": SPEC}, page=PAGE, samples=192)

    layers = [element_layers_fn(leaf, SPEC) for leaf in leaves]
    outlines = [Polygon(leaf_polygon_fn(entry[0][0], n=192)) for entry in layers]
    cores = [Polygon(leaf_polygon_fn(entry[1][0], n=192)) if len(entry) > 1 else None
             for entry in layers]

    def pale_by_lookup(point):
        probe = shapely.Point(*point)
        for index in range(len(leaves) - 1, -1, -1):  # topmost first
            if outlines[index].contains(probe):
                return not (cores[index] is not None and cores[index].contains(probe))
        return False

    rng = np.random.default_rng(0)
    probes = rng.uniform([10, 10], [90, 90], size=(4000, 2))
    disagreements = 0
    for point in probes:
        # Points within a hair of an edge are decided by the sampling, not by the construction.
        if paper.boundary.distance(shapely.Point(*point)) < 0.15:
            continue
        if paper.contains(shapely.Point(*point)) != pale_by_lookup(point):
            disagreements += 1
    assert disagreements == 0, f"{disagreements} points resolved differently from a direct lookup"


def test_a_later_core_punches_through_an_earlier_leaf():
    """Painter's order is the whole subtlety: the top element's hole is a hole in everything."""
    under = Leaf(centre=np.array([50.0, 50.0]), angle=0.0, length=40.0, width=0.62, bow=0.22)
    over = Leaf(centre=np.array([56.0, 50.0]), angle=0.0, length=40.0, width=0.62, bow=0.22)
    paper, _ = resolve_paper_fn(_field(under, over), {"main": SPEC}, page=PAGE, samples=192)
    top_core = Polygon(leaf_polygon_fn(element_layers_fn(over, SPEC)[1][0], n=192))
    assert paper.intersection(top_core).area < 1e-6 * top_core.area


def test_the_narrowest_neck_is_what_erosion_finds():
    """Two blocks joined by a neck of known width: the measure must return that width."""
    for neck in (1.0, 2.5, 4.0):
        shape = unary_union([box(0, 0, 20, 20), box(20, 10 - neck / 2, 26, 10 + neck / 2),
                             box(26, 0, 46, 20)])
        assert min_bridge_fn(shape, upper=8.0, tolerance=0.02) == pytest.approx(neck, abs=0.1)


def test_a_region_that_never_breaks_reports_the_bracket_rather_than_a_wrong_number():
    assert min_bridge_fn(box(0, 0, 50, 50), upper=6.0) == pytest.approx(6.0)


def test_dust_is_dropped_and_declared():
    region = unary_union([box(0, 0, 30, 30), box(50, 50, 50.4, 50.4)])
    kept, report = prune_dust_fn(region, min_area=1.0)
    assert report["dust_dropped"] == 1
    assert report["dust_area"] == pytest.approx(0.16, rel=1e-6)
    assert kept.geom_type == "Polygon"


def test_a_buried_element_is_not_counted_as_a_loose_fragment():
    """An element hidden under later ones contributes nothing; calling it isolated would mislead."""
    big = Polygon(box(0, 0, 20, 20))
    pieces = [Polygon(), big, Polygon(box(19, 5, 30, 15))]
    report = contact_graph_fn(pieces)
    assert report["buried"] == 1
    assert report["isolated"] == 0
    assert report["n_contacts"] == 1


def test_lattice_incidence_separates_neighbour_contacts_from_accidental_ones():
    grid = (3, 3)
    touching = [Polygon(box(0, 0, 10, 10)), Polygon(box(9, 0, 19, 10))]
    assert contact_graph_fn(touching, grid=grid, tile_of=[0, 1])["lattice_incidence"] == 1.0
    assert contact_graph_fn(touching, grid=grid, tile_of=[0, 8])["lattice_incidence"] == 0.0


def test_the_cut_file_carries_holes_inside_their_own_component():
    ring = Polygon(box(0, 0, 20, 20).exterior.coords, [box(5, 5, 10, 10).exterior.coords])
    assert len(ring.interiors) == 1, "the fixture itself must have a hole to carry"
    paths = cut_paths_fn(ring)
    assert len(paths) == 1 and isinstance(paths[0], CompoundPath)
    assert len(paths[0].subpaths) == 2, "the hole must share the component's own path"
    data = path_data_fn(paths[0], precision=4)
    assert data.count("M") == 2 and data.count("Z") == 2


def test_the_solver_returns_the_narrowest_slot_that_still_meets_the_target():
    """The bridge is not the slot, so hitting a target is a search; it must not overshoot blindly."""
    def measure(slot):
        return {"min_bridge_mm": 0.4 * slot}

    slot, report, met = solve_slot_for_bridge_fn(measure, 2.0, low=1.0, high=10.0, steps=8)
    assert met and report["min_bridge_mm"] >= 2.0
    assert slot == pytest.approx(5.0, abs=0.1)

    _, _, impossible = solve_slot_for_bridge_fn(measure, 99.0, low=1.0, high=10.0, steps=4)
    assert not impossible


def test_measurements_carry_the_sheet_size_with_them():
    paper = box(10, 10, 90, 90)
    report = measure_stencil_fn(paper, [paper], page=PAGE)
    assert report["n_components"] == 1
    assert report["coverage"] == pytest.approx(0.64)
    assert report["min_bridge_per_width"] == pytest.approx(report["min_bridge_mm"] / 100.0)
