"""What the artwork would be if it were cut: the painted image resolved into real regions.

An overlapping field of opaque elements is a picture, not an object. Painter's order hides the fact
that two leaves lying across each other are, in paper, one piece. This module resolves the paint
into the two regions that actually exist -- the ink that stays and the holes that fall out -- and
measures whether the result would survive a blade.

**Pale is the paper.** The choice is forced, not aesthetic: on a measured sheet the pale region came
out as one connected piece holding 99.97% of the ink, while the dark region was 560 loose fragments.
Reading it the other way would hand you a box of confetti.

A point is pale when the topmost element covering it paints it pale, so the paper is

    union over i of  (outline_i  minus  core_i  minus  every outline painted after i)

which is exact, and cheap once an ``STRtree`` restricts each element to the few later ones that
actually reach it.

⚠ The boolean operations cannot preserve circular arcs. The artwork keeps its exact arcs; anything
this module returns is a dense polyline, and the sampling density is the only approximation in the
cut file.
"""

import numpy as np
from jaxtyping import Float
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from shapely.strtree import STRtree

from nff_paper.topology_art.leaf import element_layers_fn, leaf_polygon_fn
from nff_paper.topology_art.quad_warp import BezierPath, CompoundPath


def _as_components(region):
    """A Polygon or MultiPolygon as a plain list, empty geometries dropped."""
    if region.is_empty:
        return []
    return list(region.geoms) if region.geom_type == "MultiPolygon" else [region]


def resolve_paper_fn(elements, specs, *, page=None, ground="hole", samples: int = 64):
    """The painted field as the region that would remain after cutting.

    Args:
        elements: ``Element`` records in paint order, already in page coordinates.
        specs: ``{population: ElementSpec}``, the same inking the renderer uses.
        page: ``(min_x, min_y, max_x, max_y)`` to clip to, or None to keep the overhang.
        ground: what the untouched page is. ``'hole'`` is the shingled reading, where the elements
            carry the paper. ``'paper'`` is the komon reading, where the ground is a whole sheet and
            the elements are holes punched in it -- the only way a SPARSE motif can be cut at all,
            since sparse elements taken as paper are islands with nothing holding them.
        samples: outline samples per element; the cut file's only quantisation.

    Returns:
        ``(paper, pieces)`` -- the resolved region, and each element's own contribution to it in
        paint order, which is what the contact graph is built from. When the ground is the paper
        those contributions are usually empty, and the components of the ground are the measure
        that means something.
    """
    outlines, papers = [], []
    for element in elements:
        layers = element_layers_fn(element.leaf, specs[element.population])
        outline = Polygon(leaf_polygon_fn(layers[0][0], n=samples)).buffer(0)
        core = (Polygon(leaf_polygon_fn(layers[1][0], n=samples)).buffer(0)
                if len(layers) > 1 else None)
        outlines.append(outline)
        # Whichever layer is inked as paper is the part this element contributes. Inverting swaps
        # them, so an inverted element gives back only its core, or nothing when it has none.
        if layers[0][1] == "pale":
            papers.append(outline.difference(core) if core is not None else outline)
        else:
            papers.append(core if core is not None else Polygon())

    tree = STRtree(outlines)
    pieces = []
    for index, contribution in enumerate(papers):
        # Only elements painted LATER can cover this one, and only those that reach it.
        above = [outlines[other] for other in tree.query(outlines[index]) if other > index]
        if above and not contribution.is_empty:
            contribution = contribution.difference(unary_union(above))
        pieces.append(contribution)

    parts = [piece for piece in pieces if not piece.is_empty]
    if ground == "paper":
        window = box(*page) if page is not None else box(*unary_union(outlines).bounds)
        parts.append(window.difference(unary_union(outlines)))
    paper = unary_union(parts) if parts else Polygon()
    if page is not None:
        window = box(*page)
        paper = paper.intersection(window)
        pieces = [piece.intersection(window) for piece in pieces]
    return paper, pieces


def prune_dust_fn(paper, *, min_area: float):
    """Drop fragments too small to survive handling, and say what was dropped.

    Reported rather than silent: a stencil that quietly sheds pieces is not the same object as one
    that never had them.
    """
    components = _as_components(paper)
    kept = [component for component in components if component.area >= min_area]
    dropped = len(components) - len(kept)
    return unary_union(kept) if kept else paper, {
        "dust_dropped": dropped,
        "dust_area": float(sum(c.area for c in components if c.area < min_area)),
    }


def min_bridge_fn(paper, *, upper: float = 8.0, tolerance: float = 0.05):
    """The narrowest neck in the ink, by eroding until the region falls apart.

    Eroding by ``w/2`` from both sides severs any neck thinner than ``w``, so the width at which the
    component count first rises is the narrowest bridge holding the sheet together. This is the
    number that decides whether a design can be cut, and no analytic expression for it exists once
    the elements overlap arbitrarily.

    Returns:
        The narrowest bridge width, or ``upper`` when nothing breaks up to that width.
    """
    intact = len(_as_components(paper))
    if intact == 0:
        return 0.0

    def survives(width):
        eroded = paper.buffer(-0.5 * width)
        # A neck may vanish entirely rather than split, so a DROP in count is a failure too.
        return len(_as_components(eroded)) == intact

    if survives(upper):
        return float(upper)
    low, high = 0.0, float(upper)
    while high - low > tolerance:
        middle = 0.5 * (low + high)
        if survives(middle):
            low = middle
        else:
            high = middle
    return float(low)


def contact_graph_fn(pieces, *, grid=None, tile_of=None):
    """Which elements are fused to which, and how regular that incidence is.

    A stencil's calm comes from every element meeting the same few neighbours in the same way. The
    degree distribution says how many, and ``lattice_incidence`` says whether they are the lattice's
    own neighbours or whichever elements happened to land nearby.

    Args:
        pieces: visible pale part of each element.
        grid: ``(M, N)`` panel grid, when the elements were placed on it.
        tile_of: tile index per element, aligned with ``pieces``.
    """
    tree = STRtree(pieces)
    degree = np.zeros(len(pieces), dtype=int)
    # An element wholly covered by later ones contributes nothing to the object. It is BURIED, not
    # isolated, and counting the two together would drag the degree statistics toward zero and hide
    # the fragments that actually would fall off the sheet.
    buried = np.array([piece.is_empty for piece in pieces], dtype=bool)
    pairs = set()
    for index, piece in enumerate(pieces):
        if buried[index]:
            continue
        for other in tree.query(piece):
            if other <= index or buried[other]:
                continue
            if piece.intersects(pieces[other]):
                pairs.add((index, int(other)))
                degree[index] += 1
                degree[other] += 1

    visible_degree = degree[~buried]
    report = {
        "n_contacts": len(pairs),
        "degree_min": int(visible_degree.min()) if visible_degree.size else 0,
        "degree_median": float(np.median(visible_degree)) if visible_degree.size else 0.0,
        "degree_max": int(visible_degree.max()) if visible_degree.size else 0,
        "isolated": int((visible_degree == 0).sum()),
        "buried": int(buried.sum()),
    }
    if grid is not None and tile_of is not None:
        rows, cols = grid
        coordinates = np.stack([np.asarray(tile_of) // cols, np.asarray(tile_of) % cols], axis=1)
        on_lattice = 0
        for one, other in pairs:
            step = np.abs(coordinates[one] - coordinates[other])
            if step.sum() == 1 or (step == 1).all():  # edge- or corner-adjacent tiles
                on_lattice += 1
        report["lattice_incidence"] = on_lattice / max(len(pairs), 1)
        report["n_tiles_used"] = int(len(np.unique(tile_of)))
    return report


def measure_stencil_fn(paper, pieces, *, page, grid=None, tile_of=None, min_bridge_upper=8.0):
    """Every number that decides whether a design is a cuttable object."""
    components = _as_components(paper)
    areas = sorted((component.area for component in components), reverse=True)
    holes = sum(len(component.interiors) for component in components)
    cut_length = sum(component.exterior.length + sum(ring.length for ring in component.interiors)
                     for component in components)
    window = box(*page)
    report = {
        "n_components": len(components),
        "largest_share": float(areas[0] / sum(areas)) if areas else 0.0,
        "coverage": float(paper.area / window.area),
        "n_holes": int(holes),
        "cut_length_mm": float(cut_length),
        "min_bridge_mm": min_bridge_fn(paper, upper=min_bridge_upper),
    }
    # A bridge width means nothing without the sheet it sits on. Carried as a fraction of the page
    # too, so a design proved on A3 can be read off at the size it would really be cut at.
    report["min_bridge_per_width"] = report["min_bridge_mm"] / (page[2] - page[0])
    report.update(contact_graph_fn(pieces, grid=grid, tile_of=tile_of))
    return report


def solve_slot_for_bridge_fn(measure_for_slot, target, *, low=1.0, high=10.0, steps=7):
    """Find the slot width whose sheet holds a bridge of at least ``target``.

    The narrowest neck is not the rim. It falls where the cores of two DIFFERENT elements pass close
    to each other, so it depends on the packing as well as on the border, and no closed form
    predicts it -- measured on one lattice the bridge came out near 0.43 of the slot. Widening the
    slot fixes it, at the cost of a paler sheet, so this is a bisection with the trade-off reported
    rather than a formula.

    Args:
        measure_for_slot: ``slot -> report`` from :func:`measure_stencil_fn`.
        target: required minimum bridge width.
        low, high: slot widths to bracket with.
        steps: bisection steps; each one resolves the whole sheet.

    Returns:
        ``(slot, report, met)`` for the narrowest slot found that meets the target, or the widest
        tried when nothing does -- with ``met`` saying which happened.
    """
    best = measure_for_slot(high)
    if best["min_bridge_mm"] < target:
        return high, best, False
    lower, upper, kept = low, high, best
    for _ in range(steps):
        middle = 0.5 * (lower + upper)
        report = measure_for_slot(middle)
        if report["min_bridge_mm"] >= target:
            upper, kept = middle, report
        else:
            lower = middle
    return upper, kept, True


def _ring_path(coordinates: Float[np.ndarray, "n 2"]) -> BezierPath:
    points = np.asarray(coordinates, float)[:-1]  # shapely repeats the first point
    return BezierPath(start=points[0],
                      segments=tuple(("L", np.array([point])) for point in points[1:]),
                      closed=True)


def cut_paths_fn(paper) -> list:
    """The resolved region as fill paths, each component carrying its own holes.

    One ``CompoundPath`` per component, because ``fill-rule: evenodd`` cannot punch a hole across
    separate SVG path elements.
    """
    return [CompoundPath(subpaths=(_ring_path(np.asarray(component.exterior.coords)),
                                   *(_ring_path(np.asarray(ring.coords))
                                     for ring in component.interiors)))
            for component in _as_components(paper)]
