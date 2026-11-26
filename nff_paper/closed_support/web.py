"""The white region as a DECIDED object: connectivity by construction, incidence by lattice.

Everything before this resolved the painted field into paper and then measured whether the paper
happened to hold together. On a scattered sheet it held by luck; on a lattice it held inside a window
of overlap depth, with the narrowest neck falling wherever two cores of DIFFERENT elements passed
close. Connectivity was a discovery either way, and a bridge width was something to bisect for.

The bound was there all along and nobody stated it. A later silhouette erases the earlier element's
whole rim, and each core sits its own rim inside its own silhouette, so any two holes are at least
one RIM apart whichever was painted first. What ate it is the leaf's tip, where the flanks converge
and the band drops to a fixed fraction of the nominal rim -- 0.435 of it at ``core_shift=0.6``, which
is exactly the 0.43 the earlier work measured and attributed to the packing. That fraction depends
only on the PROPORTIONS, so ``rim_for_slot_fn`` solves it once instead of bisecting a whole sheet.

Four constructions build on that, in the order they are worth reaching for.

**The gapped web** (``build_gapped_web_fn``) is the recommendation. Grow every occluding silhouette
by half the slot before subtracting it and the pairwise bound becomes ``rim + slot/2``, robust rather
than incidental. The cut is still made by a leaf's own arc, so nothing is faceted and the shingling
survives: the drawing is unchanged and the channel is now a decision.

**The cell web** (``build_cell_web_fn``) is the most regular and costs the leaf. One element per
site, confined to its own Voronoi territory eroded by half the slot, so the white IS the skeleton
thickened to one width, with the lattice's own junctions -- valence 3 on a staggered bond, 4 on a
square one. A leaf much larger than its cell is cropped away to the cell, and the sheet then reads as
paving rather than as scales.

**The thread** (``build_thread_web_fn``) is what a double contour needs. More than two bands enclose
a pale island, and an enclosed island is a separate piece of paper however wide its border: measured
52 pieces on one sheet. Real Ise-katagami answers this with 糸入れ, silk threads laid between the
laminated sheets to hold exactly those islands; here the thread is a spanning subgraph of the
lattice, unioned into the paper.

**The skeleton web** (``build_skeleton_web_fn``) leaves the drawing free and guarantees only a
spanning web beneath it. Measured worst of the four: the web's own junctions come out narrow.

⚠ Pointwise width is NOT claimed. Wherever three holes meet at an angle the white between them
tapers to a corner, so some of it is thinner than the slot by construction. The claim is a distance
between holes, verified as one, with ``opening_retention_fn`` carried alongside as how much of the
white area sits at the slot.
"""

from functools import lru_cache

import numpy as np
from jaxtyping import Float
from shapely.geometry import LineString, MultiPoint, Point, Polygon, box
from shapely.affinity import scale as affine_scale
from shapely.ops import unary_union, voronoi_diagram
from shapely.strtree import STRtree

from nff_paper.topology_art.leaf import (
    ElementSpec,
    Leaf,
    element_layers_fn,
    inner_leaf_fn,
    leaf_polygon_fn,
    rim_fraction_fn,
    transform_leaf_fn,
)


def _components(region):
    if region.is_empty:
        return []
    return list(region.geoms) if region.geom_type == "MultiPolygon" else [region]


# ---------------------------------------------------------------- the element, fully nested


def element_dark_fn(leaf: Leaf, spec, *, samples: int = 64):
    """The element's holes: every dark band of the nesting, resolved into one region.

    The existing paper resolver reads only the outline and the first core, so a comb of bands is
    drawn but cut as though it were a single ring. Alternating bands are what a double-contour leaf
    IS, so they are resolved here instead: each dark layer contributes the annulus between itself and
    the next layer inward, and the innermost contributes its whole disc.
    """
    layers = element_layers_fn(leaf, spec)
    rings = [Polygon(leaf_polygon_fn(shape, n=samples)).buffer(0) for shape, _ in layers]
    bands = []
    for index, (_, tag) in enumerate(layers):
        if tag != "dark":
            continue
        inner = rings[index + 1] if index + 1 < len(rings) else None
        bands.append(rings[index] if inner is None else rings[index].difference(inner))
    return unary_union(bands) if bands else Polygon()


def resolve_dark_fn(elements, specs, *, page, samples: int = 64):
    """The sheet's holes under painter's order, with the full nesting kept.

    An element painted later replaces whatever lies under its silhouette, so its own dark bands are
    all that survive there. The untouched ground is a hole too: these constructions read the sheet as
    figures cut out of nothing rather than as paper carrying figures.
    """
    outlines, darks = [], []
    for element in elements:
        layers = element_layers_fn(element.leaf, specs[element.population])
        outlines.append(Polygon(leaf_polygon_fn(layers[0][0], n=samples)).buffer(0))
        darks.append(element_dark_fn(element.leaf, specs[element.population], samples=samples))

    tree = STRtree(outlines)
    parts = []
    for index, dark in enumerate(darks):
        above = [outlines[other] for other in tree.query(outlines[index]) if other > index]
        if above and not dark.is_empty:
            dark = dark.difference(unary_union(above))
        if not dark.is_empty:
            parts.append(dark)
    parts.append(box(*page).difference(unary_union(outlines)))
    return unary_union(parts).intersection(box(*page))


# ---------------------------------------------------------------- territories


def voronoi_cells_fn(sites: Float[np.ndarray, "n 2"], *, page, pad: float = 20.0) -> list:
    """The Voronoi cell of each site, clipped to the page and returned in the site's own order.

    ``voronoi_diagram`` hands the cells back in whatever order it likes, so each one is matched to
    the site it contains rather than assumed to be aligned.
    """
    window = box(*page)
    cells = list(voronoi_diagram(MultiPoint([tuple(site) for site in sites]),
                                 envelope=window.buffer(pad)).geoms)
    tree = STRtree(cells)
    ordered = []
    for site in sites:
        point = Point(site)
        found = [cells[index] for index in tree.query(point) if cells[index].intersects(point)]
        ordered.append(found[0].intersection(window) if found else Polygon())
    return ordered


def lloyd_sites_fn(sites: Float[np.ndarray, "n 2"], *, page, passes: int = 8):
    """Relax the sites toward the centroids of their own cells: a centroidal Voronoi diagram.

    Each pass moves every site to the middle of the region it owns, which evens the cell areas out and
    drives the diagram toward a honeycomb. It is the opposite extreme from letting the tiling's own
    size field through, and having both makes "how much do the cells differ" a dial rather than a
    property of the sheet.

    Returns:
        ``(sites, report)`` with the area spread before and after, so the relaxation is measured.
    """
    sites = np.asarray(sites, float).copy()

    def spread(points):
        areas = np.array([cell.area for cell in voronoi_cells_fn(points, page=page)
                          if not cell.is_empty])
        return float(areas.max() / max(areas.min(), 1e-12)) if areas.size else 0.0

    before = spread(sites)
    for _ in range(max(passes, 0)):
        moved = []
        for site, cell in zip(sites, voronoi_cells_fn(sites, page=page)):
            moved.append(np.asarray(cell.centroid.coords[0]) if not cell.is_empty else site)
        sites = np.array(moved, float)
    return sites, {"lloyd_passes": int(passes), "area_spread_before": before,
                   "area_spread_after": spread(sites)}


def territories_fn(sites: Float[np.ndarray, "n 2"], *, page, clearance,
                   frame: float | None = None):
    """One Voronoi cell per site, eroded so that any two of them are ``clearance`` apart.

    Neighbouring cells share a whole edge, so taking half the slot off each leaves exactly the slot
    between them. That is the entire guarantee: whatever shape a hole takes inside its own cell, it
    cannot come nearer than the slot to any other hole, and it cannot reach across the sheet to
    separate it.

    Args:
        sites: (n, 2) hole centres, in page coordinates.
        page: ``(min_x, min_y, max_x, max_y)``.
        clearance: the slot between neighbouring holes. A scalar holds one slot across the sheet; an
            array of one value per site lets the web BREATHE, broad where the tiles are stretched and
            fine where they are packed. Two neighbours then leave ``(c_i + c_j)/2`` between them, so
            the sheet's guarantee is the smallest value given, not the average.
        frame: width of solid paper to keep around the page, or None to let the border holes run off
            the edge and the sheet read full bleed.
            ⚠⚠ With no frame the erosion still insets every border territory by HALF the slot, so the
            strip of skeleton along the page edge is only half a slot wide and becomes the narrowest
            white on the sheet. That is what made `skeleton_under` measure 0.80 mm against a 1.6 mm
            slot. Pass a frame of at least the clearance whenever the slot is being claimed.

    Returns:
        A list of polygons aligned with ``sites``; a cell too small to survive the erosion comes back
        empty rather than as an error, since a site crowded by its neighbours simply carries no hole.
    """
    slots = np.broadcast_to(np.asarray(clearance, float), (len(sites),))
    window = box(*page)
    cells = voronoi_cells_fn(sites, page=page, pad=4.0 * float(slots.max()) + 10.0)
    limit = window if frame is None else window.buffer(-float(frame))

    territories = []
    for cell, slot in zip(cells, slots):
        clipped = cell.intersection(limit)
        # Mitred, because a Voronoi cell is convex and its true inner offset is the intersection of
        # the offset half planes. A round join would eat the corners the leaf needs.
        eroded = clipped.buffer(-0.5 * float(slot), join_style="mitre", mitre_limit=8.0)
        territories.append(max(_components(eroded), key=lambda part: part.area, default=Polygon()))
    return territories


def round_holes_fn(hole, radius: float):
    """Round a hole's corners without ever narrowing the channel around it.

    A morphological opening: erode by the radius and dilate back, which turns every convex corner
    into an arc of that radius. Safe for the guarantee by construction -- an opening is a SUBSET of
    what it opens, so the hole only shrinks and every channel between holes can only widen.
    """
    if radius <= 0.0 or hole.is_empty:
        return hole
    return hole.buffer(-radius, join_style="round").buffer(radius, join_style="round")


def inscribed_disc_fn(territory, site=None):
    """The largest circle centred in a territory: the plainest komon hole there is.

    Centred on the site when it lies inside its own cell, which it does by definition of a Voronoi
    diagram, and on the cell's representative point otherwise.
    """
    if territory.is_empty:
        return Polygon()
    centre = Point(site) if site is not None else territory.representative_point()
    if not territory.contains(centre):
        centre = territory.representative_point()
    return centre.buffer(territory.boundary.distance(centre))


def fit_leaf_fn(leaf: Leaf, territory, *, samples: int = 48, steps: int = 14) -> Leaf | None:
    """The largest concentric copy of ``leaf`` that fits inside ``territory``.

    Shrinking rather than clipping, because a leaf differs from its own copy by a similarity and
    nothing else: the arcs stay exact, so the element remains the element. Clipping would answer the
    same containment question with straight facets where the cell happened to fall.
    """
    if territory.is_empty:
        return None

    def fits(scale):
        shrunk = transform_leaf_fn(leaf, scale=scale,
                                  offset=(1.0 - scale) * np.asarray(leaf.centre, float))
        return territory.contains(Polygon(leaf_polygon_fn(shrunk, n=samples)))

    if fits(1.0):
        return leaf
    low, high = 0.0, 1.0
    for _ in range(steps):
        middle = 0.5 * (low + high)
        if fits(middle):
            low = middle
        else:
            high = middle
    if low <= 1e-3:
        return None
    return transform_leaf_fn(leaf, scale=low,
                            offset=(1.0 - low) * np.asarray(leaf.centre, float))


# ---------------------------------------------------------------- the thread


def lattice_edges_fn(tile_of, grid, *, mode: str = "delaunay",
                     sites: Float[np.ndarray, "n 2"] | None = None):
    """The thread's graph over the sites, and how regular its incidence is.

    ``grid`` mode takes the tiling's own row and column adjacency, which is unambiguous and gives a
    valence of four inside the sheet. ``delaunay`` mode reads the neighbours off the geometry, which
    is what a staggered bond wants: the half-cell offset makes each site meet six others, and no
    combinatorial rule states that without knowing which way the offset runs.

    Returns:
        ``(edges, report)`` with edges as index pairs into the site array.
    """
    tile_of = np.asarray(tile_of, int)
    count = len(tile_of)
    edges = set()
    if mode == "grid":
        rows, cols = grid
        at = {int(tile): index for index, tile in enumerate(tile_of)}
        for index, tile in enumerate(tile_of):
            row, column = int(tile) // cols, int(tile) % cols
            for step_row, step_col in ((0, 1), (1, 0)):
                other = (row + step_row) * cols + (column + step_col)
                if row + step_row < rows and column + step_col < cols and other in at:
                    edges.add((index, at[other]))
    elif mode == "delaunay":
        from scipy.spatial import Delaunay

        triangles = Delaunay(np.asarray(sites, float)).simplices
        for triangle in triangles:
            for one, other in ((0, 1), (1, 2), (2, 0)):
                a, b = int(triangle[one]), int(triangle[other])
                edges.add((min(a, b), max(a, b)))
    else:
        raise ValueError(f"unknown thread mode {mode!r}")

    edges = sorted(edges)
    degree = np.zeros(count, dtype=int)
    for one, other in edges:
        degree[one] += 1
        degree[other] += 1

    # A spanning check, not a hope: the thread only guarantees one piece of paper if the thread
    # itself is one piece.
    seen = np.zeros(count, dtype=bool)
    neighbours = [[] for _ in range(count)]
    for one, other in edges:
        neighbours[one].append(other)
        neighbours[other].append(one)
    stack, seen[0] = [0], True
    while stack:
        current = stack.pop()
        for other in neighbours[current]:
            if not seen[other]:
                seen[other] = True
                stack.append(other)

    values, counts = np.unique(degree, return_counts=True)
    return edges, {
        "n_edges": len(edges),
        "thread_connected": bool(seen.all()),
        "degree_median": float(np.median(degree)),
        "degree_min": int(degree.min()),
        "degree_max": int(degree.max()),
        "degree_histogram": {int(v): int(c) for v, c in zip(values, counts)},
    }


def thread_region_fn(sites, edges, *, width: float, page=None):
    """The thread given a width: the paper it contributes.

    Round caps, for the reason the nets already showed -- two segments meeting at a shallow angle
    with flat caps leave a notch that pinches the strut well under its own width, so the bridge would
    no longer be the number the design named.
    """
    sites = np.asarray(sites, float)
    strokes = [LineString([sites[one], sites[other]]).buffer(
        0.5 * width, cap_style="round", join_style="round") for one, other in edges]
    region = unary_union(strokes) if strokes else Polygon()
    return region.intersection(box(*page)) if page is not None else region


def keep_largest_fn(paper):
    """Keep the sheet and report what fell off it.

    Safer than seeding on the armature, because an armature need not be one piece. A net generated
    past the sheet leaves boundary fragments once the overhang is cropped -- 34 of them on one asanoha
    panel -- and seeding on the whole armature then keeps whatever paper a stray fragment happens to
    touch, so the sheet comes out in two pieces. The largest component is the sheet in every
    construction here, and a local armature such as a nick per island is unaffected because a nick
    that works joins that main component anyway.
    """
    parts = _components(paper)
    if not parts:
        return paper, {"islands_dropped": 0, "islands_area": 0.0}
    keep = max(parts, key=lambda part: part.area)
    dropped = [part for part in parts if part is not keep]
    return keep, {"islands_dropped": len(dropped),
                  "islands_area": float(sum(part.area for part in dropped))}


def keep_threaded_fn(paper, thread):
    """Keep only the paper the thread actually holds, and say what that cost.

    A pale fragment that touches no thread is a piece that falls out. Dropping it is the cut
    decision, not a defect: it becomes part of the hole, and the sheet is then one piece by
    construction rather than one piece if measured.
    """
    parts = _components(paper)
    kept = [part for part in parts if part.intersects(thread)]
    dropped = [part for part in parts if not part.intersects(thread)]
    return unary_union(kept) if kept else Polygon(), {
        "islands_dropped": len(dropped),
        "islands_area": float(sum(part.area for part in dropped)),
    }


# ---------------------------------------------------------------- the two constructions


def build_cell_web_fn(elements, specs, *, page, clearance, mode: str = "crop",
                      frame: float | None = None, tab_mm: float = 0.0, corner_radius: float = 0.0,
                      fill: float = 1.0, holes=None, samples: int = 64):
    """Holes confined one per territory: the white web of a thickened Voronoi skeleton.

    The skeleton, ``window`` minus every territory, is the load-bearing part of the claim. It is
    connected because the Voronoi adjacency graph is, and it is at least ``clearance`` wide because
    each territory gave up half the slot, so anything attached to it is held. Whatever is not
    attached is not paper, and is reported as dropped rather than left in the drawing.

    Args:
        elements: one ``Element`` per site, already in page coordinates.
        specs: ``{population: ElementSpec}``.
        clearance: slot between neighbouring holes; the sheet's narrowest channel by construction.
        mode: ``'crop'`` intersects each element with its territory, so a large element fills its
            cell and the web becomes the skeleton at uniform width. ``'fit'`` shrinks it instead,
            keeping the leaf a leaf and leaving the cell corners as part of the web. ``'disc'`` puts
            the cell's largest inscribed circle there, the plainest komon reading of all.
        frame: solid border to keep around the page, or None for full bleed.
        tab_mm: width of a radial tab per element, or 0 for none. A cropped element has its bands
            opened onto the skeleton where the cell cut them, so the tab matters only for the few
            elements that fall wholly inside their own territory.
        corner_radius: round every hole's corners by this much, turning Voronoi facets into pebbles.
            It cannot weaken the sheet: an opening is a subset, so the holes only shrink.
        holes: a hole already chosen per cell, bypassing ``mode`` entirely. The guarantee does not
            care what the shape is, only that it stays inside its own territory, so a motif
            vocabulary can live outside this module and still inherit the whole claim.
        fill: how much of the room available to it each hole actually takes, scaled about its own
            site. This is the dial along the whole family, and it is needed because ``fit`` already
            shrinks a leaf to its cell: past that point ``size_frac`` does nothing, so a sparse sheet
            cannot be reached by making the element smaller. 1 fills, 0.5 halves, and the web widens
            to match.

    Returns:
        ``(paper, holes, skeleton, report)``.
    """
    sites = np.array([element.leaf.centre for element in elements], float)
    territories = territories_fn(sites, page=page, clearance=clearance, frame=frame)
    if holes is not None:
        # A caller that has already chosen a shape per cell. The guarantee does not care what the
        # shape is, only that it stays inside its own territory, so the motif vocabulary lives
        # outside this module entirely.
        holes = [round_holes_fn(hole, corner_radius) for hole in holes]
        return _cell_web_from_holes(elements, holes, territories, sites, page=page,
                                    clearance=clearance, mode=mode, tab_mm=tab_mm,
                                    corner_radius=corner_radius, fill=fill)

    holes = []
    for element, territory, site in zip(elements, territories, sites):
        spec = specs[element.population]
        if mode == "fit":
            fitted = fit_leaf_fn(element.leaf, territory, samples=samples // 2)
            dark = (element_dark_fn(fitted, spec, samples=samples)
                    if fitted is not None else Polygon())
        elif mode == "crop":
            dark = element_dark_fn(element.leaf, spec, samples=samples).intersection(territory)
        elif mode == "disc":
            dark = inscribed_disc_fn(territory, site)
        else:
            raise ValueError(f"unknown cell-web mode {mode!r}")
        if fill != 1.0 and not dark.is_empty:
            dark = affine_scale(dark, xfact=fill, yfact=fill, origin=Point(site))
        holes.append(round_holes_fn(dark, corner_radius))

    return _cell_web_from_holes(elements, holes, territories, sites, page=page,
                                clearance=clearance, mode=mode, tab_mm=tab_mm,
                                corner_radius=corner_radius, fill=fill)


def _cell_web_from_holes(elements, holes, territories, sites, *, page, clearance, mode, tab_mm,
                         corner_radius, fill):
    """Assemble and measure the sheet once the hole of every cell has been decided."""
    window = box(*page)
    skeleton = window.difference(unary_union([t for t in territories if not t.is_empty]))
    paper = window.difference(unary_union([hole for hole in holes if not hole.is_empty]))
    if tab_mm > 0.0:
        paper = paper.union(tab_region_fn(elements, width=tab_mm, page=page))
    paper, islands = keep_threaded_fn(paper, skeleton)

    slots = np.broadcast_to(np.asarray(clearance, float), (len(sites),))
    report = {"construction": f"cell/{mode}", "clearance_mm": float(slots.min()),
              "clearance_max_mm": float(slots.max()), "corner_radius_mm": float(corner_radius),
              "fill": float(fill),
              "tab_mm": float(tab_mm), "n_sites": len(sites),
              "n_holes": int(sum(not h.is_empty for h in holes))}
    # The guarantee is the SMALLEST slot given, never the average: two neighbours leave
    # (c_i + c_j)/2 between them, so a breathing web is only as strong as its finest channel.
    report.update(channel_report_fn(holes, sites, clearance=float(slots.min())))
    report.update(islands)
    return paper, holes, skeleton, report


def build_thread_web_fn(elements, specs, *, page, grid, tile_of, thread_mm: float,
                        mode: str = "delaunay", clearance: float | None = None,
                        cell_mode: str = "fit", frame: float | None = None, samples: int = 64):
    """The nested element kept whole, with a regular thread holding every island it encloses.

    Args:
        thread_mm: thread width, which is the sheet's narrowest bridge.
        mode: thread graph, ``'grid'`` or ``'delaunay'``.
        clearance: when given, the elements are first confined to Voronoi territories exactly as in
            the cell web, so the thread holds islands rather than rescuing accidental overlaps. When
            None the elements overlap freely and the thread is all that guarantees the sheet.

    Returns:
        ``(paper, thread, report)``.
    """
    sites = np.array([element.leaf.centre for element in elements], float)
    edges, thread_report = lattice_edges_fn(tile_of, grid, mode=mode, sites=sites)
    thread = thread_region_fn(sites, edges, width=thread_mm, page=page)

    if clearance is None:
        dark = resolve_dark_fn(elements, specs, page=page, samples=samples)
        paper = box(*page).difference(dark)
        holes = []
        cell = {}
    else:
        paper, holes, _, cell = build_cell_web_fn(elements, specs, page=page, clearance=clearance,
                                                  mode=cell_mode, frame=frame, samples=samples)
        cell = {key: cell[key] for key in ("clearance_mm", "gap_min_mm", "gap_median_mm",
                                          "at_clearance")}

    paper = paper.union(thread)
    paper, islands = keep_threaded_fn(paper, thread)
    report = {"construction": f"thread/{mode}", "thread_mm": float(thread_mm),
              "n_sites": len(sites), "n_holes": int(sum(not h.is_empty for h in holes))}
    report.update(thread_report)
    report.update(islands)
    report.update(cell)
    return paper, thread, report


def element_pale_fn(leaf: Leaf, spec, *, samples: int = 200):
    """The white inside one element: its silhouette less every dark band."""
    outline = Polygon(leaf_polygon_fn(leaf, n=samples)).buffer(0)
    dark = element_dark_fn(leaf, spec, samples=samples)
    return outline if dark.is_empty else outline.difference(dark)


@lru_cache(maxsize=512)
def _pinch(width, bow, rings, duty, core_shift, core_width_delta, probe, samples):
    canonical = Leaf(centre=np.zeros(2), angle=0.0, length=1.0, width=width, bow=bow)
    spec = ElementSpec(rings=rings, rim=probe, duty=duty, core_shift=core_shift,
                       core_width_delta=core_width_delta)
    pale = element_pale_fn(canonical, spec, samples=samples)
    return min_width_fn(pale, upper=4.0 * probe, tolerance=1e-4 * probe) / probe


def rim_pinch_fn(spec, *, width: float, bow: float, probe: float | None = None,
                 samples: int = 200) -> float:
    """Narrowest white inside an element, as a fraction of its NOMINAL rim.

    A nested core is inset from the outline by the same absolute amount all round, but the flanks
    converge at the tip, so the band the eye and the blade see there is narrower than the number that
    produced it. The shortfall is a property of the PROPORTIONS alone, invariant under a similarity,
    so it is measured once on the canonical leaf and applies to every element on the sheet.

    This is what turns the sheet's narrowest white from a measurement into an arithmetic. The bridge
    of an overlapping field had to be bisected for because the neck fell where two different cores
    passed close and no formula predicted it; under the gapped construction that distance is set by
    the slot, and the only other candidate is this one.

    ⚠ The probe band must shrink with the number of rings. A comb of eleven bands insets its innermost
    ring by five of them, so probing a deep comb at a fixed 0.08 of the length leaves no element at
    all and the ratio comes back meaningless. The default scales the probe so the whole nesting always
    occupies the same share of the leaf.
    """
    if probe is None:
        probe = 0.08 / max(1.0, 0.5 * (int(spec.rings) - 1))
    return float(_pinch(round(float(width), 4), round(float(bow), 4), int(spec.rings),
                        round(float(spec.duty), 4), round(float(spec.core_shift), 4),
                        round(float(spec.core_width_delta), 4), float(probe), int(samples)))


def rim_for_slot_fn(spec, *, width: float, bow: float, slot: float, **kwargs) -> float:
    """The absolute rim whose narrowest band comes out at ``slot``."""
    return float(slot / max(rim_pinch_fn(spec, width=width, bow=bow, **kwargs), 1e-6))


def build_gapped_web_fn(elements, specs, *, page, clearance: float, samples: int = 64):
    """Painter's order with the occluders dilated: the shingle kept, every channel at the slot.

    The favourite reading crops each element's core with the silhouettes painted after it, which is
    what makes a field of leaves read as scales. The channel it leaves between two cores is then
    whatever the packing happened to give -- measured on one lattice at 0.43 of the slot, and needing
    a bisection to hit a target.

    Dilating every occluding silhouette by half the slot before subtracting fixes it outright. Core
    ``j`` sits at least its own rim inside silhouette ``j``, and core ``i`` is now cut at least half a
    slot OUTSIDE it, so any two cores are a slot apart whatever the packing does. The cut is still
    made by a leaf's own arc rather than by a cell boundary, so nothing is faceted and the shingling
    survives intact.

    The rim must therefore be at least half the slot; below that the element's own border is the
    narrowest white on the sheet and the guarantee is the rim, not the slot. ``rim_ok`` reports it.

    Returns:
        ``(paper, holes, report)``.
    """
    outlines, darks, rims = [], [], []
    for element in elements:
        spec = specs[element.population]
        layers = element_layers_fn(element.leaf, spec)
        outlines.append(Polygon(leaf_polygon_fn(layers[0][0], n=samples)).buffer(0))
        darks.append(element_dark_fn(element.leaf, spec, samples=samples))
        rims.append(rim_fraction_fn(element.leaf.length, rim_frac=spec.rim, rim_abs=spec.rim_abs)
                    * element.leaf.length)

    # The occluder is the silhouette grown by half the slot. Grown once and reused, since every
    # earlier element asks the same question of it.
    grown = [outline.buffer(0.5 * clearance, join_style="round") for outline in outlines]
    tree = STRtree(grown)

    holes = []
    for index, dark in enumerate(darks):
        above = [grown[other] for other in tree.query(grown[index]) if other > index]
        if above and not dark.is_empty:
            dark = dark.difference(unary_union(above))
        holes.append(dark)

    window = box(*page)
    ground = window.difference(unary_union(outlines))
    dark_region = unary_union([ground, *[hole for hole in holes if not hole.is_empty]])
    paper = window.difference(dark_region)

    parts = _components(paper)
    keep = max(parts, key=lambda part: part.area, default=Polygon())
    dropped = [part for part in parts if part is not keep]
    report = {
        "construction": "gapped", "clearance_mm": float(clearance),
        "n_sites": len(elements),
        "rim_min_mm": float(min(rims)) if rims else 0.0,
        "rim_ok": bool(rims and min(rims) >= 0.5 * clearance),
        "islands_dropped": len(dropped),
        "islands_area": float(sum(part.area for part in dropped)),
    }
    return keep, holes, report


def open_paper_fn(paper, width: float):
    """Drop every white feature narrower than ``width``, keeping the rest as it was.

    An erosion followed by the matching dilation. The spanning web is a lower bound on the sheet's
    strength; this is the other half of the same claim, since after it NO white anywhere is thinner
    than the slot, so the narrowest neck in the object is a design number rather than a measurement.
    It is paid for in the drawing: a rim finer than the slot is not thinned, it is gone.
    """
    return paper.buffer(-0.5 * width, join_style="round").buffer(0.5 * width, join_style="round")


def opening_retention_fn(paper, width: float) -> float:
    """Fraction of the white that survives an opening at ``width``.

    The existing bridge measure erodes until the component count changes, so it sees SEVERANCE and
    not thinning: a sheet that is one piece stays one piece while every thin arm on it erodes to
    nothing, and the measurement then reports the erosion cap instead of the width. Retention asks
    the question the claim actually makes -- is any white thinner than the slot -- and answers it as
    an area, which cannot be fooled by the count.
    """
    if paper.is_empty:
        return 0.0
    return float(open_paper_fn(paper, width).area / paper.area)


def min_width_fn(paper, *, retention: float = 0.99, upper: float = 12.0,
                 tolerance: float = 0.05) -> float:
    """The widest opening the white survives whole: its narrowest feature, near enough.

    Bisected on retention rather than on connectivity, for the reason above. The tolerance on the
    retained fraction absorbs the buffer's own polygonal approximation of a circle.
    """
    if paper.is_empty:
        return 0.0
    if opening_retention_fn(paper, upper) >= retention:
        return float(upper)
    low, high = 0.0, float(upper)
    while high - low > tolerance:
        middle = 0.5 * (low + high)
        if opening_retention_fn(paper, middle) >= retention:
            low = middle
        else:
            high = middle
    return float(low)


def skeleton_region_fn(sites, *, page, clearance: float, frame: float | None = None):
    """The Voronoi skeleton at the slot width: a spanning web, connected because the diagram is.

    Extracted so that a skeleton can be used as an armature on the same footing as a thread or a net,
    rather than only inside the construction that first needed it.
    """
    territories = territories_fn(sites, page=page, clearance=clearance, frame=frame)
    live = [territory for territory in territories if not territory.is_empty]
    return box(*page).difference(unary_union(live)) if live else box(*page)


def build_skeleton_web_fn(elements, specs, *, page, clearance: float,
                          frame: float | None = "clearance", open_mm: float = 0.0,
                          samples: int = 64):
    """The drawing left free, with the channels between neighbouring elements guaranteed.

    Confining each hole to its own cell is stricter than the object requires: holes that merge across
    cells still fall out, so what has to be connected is not the region between every pair of
    elements but a web that spans the sheet. The Voronoi skeleton at the slot width is exactly such a
    web -- connected because the Voronoi adjacency graph is, and ``clearance`` wide because every
    territory gave up half the slot -- and it runs where the white channels between neighbouring
    elements already run, so it curves with the flow and costs almost no ink.

    Above it the elements overlap as they like. Nothing is cropped, so the shingling and the exact
    arcs survive, and the pale islands a nested element encloses are opened onto the skeleton.

    Args:
        frame: solid border to keep. Defaults to the clearance itself, because with no frame the
            border strip of the skeleton is only HALF the slot and is then the narrowest white on the
            sheet -- measured at 0.80 mm against a 1.6 mm slot before this default was fixed. Pass
            None explicitly for a full-bleed sheet, and expect the border to hold the claim back.

    Returns:
        ``(paper, skeleton, report)``.
    """
    if frame == "clearance":
        frame = float(clearance)
    sites = np.array([element.leaf.centre for element in elements], float)
    window = box(*page)
    skeleton = skeleton_region_fn(sites, page=page, clearance=clearance, frame=frame)

    dark = resolve_dark_fn(elements, specs, page=page, samples=samples)
    paper = window.difference(dark)
    if open_mm > 0.0:
        paper = open_paper_fn(paper, open_mm)
    paper = paper.union(skeleton)
    paper, islands = keep_threaded_fn(paper, skeleton)

    report = {"construction": "skeleton", "clearance_mm": float(clearance),
              "frame_mm": 0.0 if frame is None else float(frame),
              "open_mm": float(open_mm), "n_sites": len(sites),
              "skeleton_share": float(skeleton.area / window.area)}
    report.update(islands)
    return paper, skeleton, report


def channel_report_fn(holes, sites, *, clearance: float, tolerance: float = 0.05):
    """Width of the white channel between every pair of neighbouring holes.

    "Strict" is the minimum of these and "regular" is how many of them sit at that minimum, so a
    construction claiming both has to show both. A mean and a standard deviation will not do it: a
    few pairs on the convex hull are arbitrarily far apart and drag both, while the body of the sheet
    is at the slot exactly. ``at_clearance`` counts the channels actually holding the slot.

    Neighbours are read from the Delaunay graph of the sites, the dual of the territories, which
    names exactly the pairs a channel runs between.
    """
    from scipy.spatial import Delaunay

    empty = {"gap_min_mm": 0.0, "gap_median_mm": 0.0, "gap_p95_mm": 0.0, "at_clearance": 0.0}
    sites = np.asarray(sites, float)
    if len(sites) < 4:
        return empty
    pairs = set()
    for triangle in Delaunay(sites).simplices:
        for one, other in ((0, 1), (1, 2), (2, 0)):
            a, b = int(triangle[one]), int(triangle[other])
            pairs.add((min(a, b), max(a, b)))
    gaps = np.asarray([holes[one].distance(holes[other]) for one, other in sorted(pairs)
                       if not holes[one].is_empty and not holes[other].is_empty], float)
    if not gaps.size:
        return empty
    return {
        "gap_min_mm": float(gaps.min()),
        "gap_median_mm": float(np.median(gaps)),
        "gap_p95_mm": float(np.percentile(gaps, 95)),
        "at_clearance": float((gaps <= clearance * (1.0 + tolerance)).mean()),
    }


def tab_region_fn(elements, *, width: float, overshoot: float = 0.06, reach: float = 0.0,
                  page=None):
    """One tab per element, along its own axis, from outside the base to the given reach.

    Every band of a nested element contains the element's centre, so a single radial cut from beyond
    the base inward crosses all of them at once and threads the whole nesting onto whatever lies
    outside the silhouette. Placed at the same parametric position in every element's own frame, so
    the device is one repeated rule rather than a patch applied where a measurement complained.
    """
    strokes = []
    for element in elements:
        leaf = element.leaf
        axis = np.array([np.cos(leaf.angle), np.sin(leaf.angle)])
        centre = np.asarray(leaf.centre, float)
        start = centre - (0.5 + overshoot) * leaf.length * axis
        end = centre + reach * leaf.length * axis
        strokes.append(LineString([start, end]).buffer(
            0.5 * width, cap_style="round", join_style="round"))
    region = unary_union(strokes) if strokes else Polygon()
    return region.intersection(box(*page)) if page is not None else region


# ---------------------------------------------------------------- the LINE register
#
# Everything above treats an element as an area: a silhouette, bands inside it, figure against ground.
# A stencil can also be a DRAWING -- strokes tracing loops, with no solid mass anywhere -- and then the
# element survives only as its outline. The two readings of such a sheet are genuinely different
# objects: the stroke is either the slot cut into a nearly solid sheet, or it is the whole sheet.


def concentric_leaves_fn(leaf: Leaf, *, spacing: float, count: int = 1) -> list:
    """A leaf and its inward copies at a constant absolute spacing.

    Inset absolutely rather than scaled, for the reason the rim already had: a scaled copy takes the
    same FRACTION off length and width, and a leaf is far longer than it is wide, so the gap between
    two contours would open at the ends and close at the sides.
    """
    leaves, current = [leaf], leaf
    for _ in range(max(count, 1) - 1):
        inner = inner_leaf_fn(current, spacing / max(current.length, 1e-9))
        if inner is None:
            break
        leaves.append(inner)
        current = inner
    return leaves


def element_contours_fn(leaf: Leaf, *, spacing: float, count: int = 1, rule: str = "offset",
                        samples: int = 96) -> list:
    """The contours of one element, as closed polylines.

    Args:
        rule: ``'offset'`` takes the true geometric inward offset, so the gap between contours is
            exactly ``spacing`` the whole way round. ``'leaf'`` uses nested LEAVES instead, which
            inherits the tip pinch: the inset is absolute in both dimensions, but the flanks converge
            at the tip and the gap measured there comes out at about 0.72 of nominal. Both are worth
            having -- the leaf rule keeps every contour an exact leaf, the offset rule keeps the
            spacing honest -- but they are not the same drawing and the difference is at the tip.
    """
    outline = leaf_polygon_fn(leaf, n=samples)
    if rule == "leaf":
        return [leaf_polygon_fn(contour, n=samples)
                for contour in concentric_leaves_fn(leaf, spacing=spacing, count=count)]
    if rule != "offset":
        raise ValueError(f"unknown contour rule {rule!r}")

    contours = [outline]
    solid = Polygon(outline).buffer(0)
    for step in range(1, max(count, 1)):
        inner = solid.buffer(-step * spacing, join_style="round")
        parts = _components(inner)
        if not parts:
            break
        biggest = max(parts, key=lambda part: part.area)
        contours.append(np.asarray(biggest.exterior.coords)[:-1])
    return contours


def element_stroke_fn(leaf: Leaf, *, width: float, spacing: float | None = None, count: int = 1,
                      kind: str = "even", rule: str = "offset", samples: int = 96):
    """One element as STROKES along its contours rather than as filled bands.

    Args:
        width: nominal stroke weight.
        spacing: centre-to-centre distance between contours, defaulting to three widths.
        count: how many concentric contours to stroke.
        kind: ``'even'`` buffers the contour itself, so the weight is exactly ``width`` the whole way
            round, as a pen or a blade gives. ``'varying'`` takes the band between a leaf and its own
            inset copy, which is what the filled elements do: the flanks converge at the tip, so the
            stroke thins there to a fixed fraction of its nominal weight and reads as a brush lifting.
        rule: how the concentric contours are spaced; see :func:`element_contours_fn`.
    """
    spacing = 3.0 * width if spacing is None else spacing
    if kind == "even":
        strokes = [LineString(np.vstack([contour, contour[:1]])).buffer(
            0.5 * width, cap_style="round", join_style="round")
            for contour in element_contours_fn(leaf, spacing=spacing, count=count, rule=rule,
                                               samples=samples)]
    elif kind == "varying":
        # The brush rule is defined on the LEAF, because the thinning it is being compared for is a
        # property of the nested leaf's own geometry.
        strokes = []
        for contour in concentric_leaves_fn(leaf, spacing=spacing, count=count):
            inner = inner_leaf_fn(contour, width / max(contour.length, 1e-9))
            solid = Polygon(leaf_polygon_fn(contour, n=samples)).buffer(0)
            strokes.append(solid if inner is None
                           else solid.difference(Polygon(leaf_polygon_fn(inner, n=samples))))
    else:
        raise ValueError(f"unknown stroke kind {kind!r}")
    return unary_union(strokes) if strokes else Polygon()


def stroke_field_fn(elements, *, page, width: float, spacing: float | None = None, count: int = 1,
                    kind: str = "even", rule: str = "offset", occlude: bool = True,
                    samples: int = 96):
    """Every element's contours as one region of ink.

    Args:
        occlude: whether a later element's whole SILHOUETTE erases the strokes under it.

    ⚠⚠ Occlusion is what makes this a drawing rather than a mesh. With it, each stroke terminates
    where the next leaf covers it, so the loops are cut open and the ground stays connected -- the
    look of overlapping brush strokes. Without it every contour stays a closed circuit, and a planar
    arrangement of closed circuits divides the ground into one cell per face: measured **1115 separate
    pieces** on a sheet of 280 elements, against 52 with occlusion on. Which reading is being cut
    decides whether that matters, so it is a parameter and not a default to forget.
    """
    strokes, silhouettes = [], []
    for element in elements:
        strokes.append(element_stroke_fn(element.leaf, width=width, spacing=spacing, count=count,
                                         kind=kind, rule=rule, samples=samples))
        silhouettes.append(Polygon(leaf_polygon_fn(element.leaf, n=samples)).buffer(0))

    if occlude:
        tree = STRtree(silhouettes)
        kept = []
        for index, stroke in enumerate(strokes):
            above = [silhouettes[other] for other in tree.query(silhouettes[index])
                     if other > index]
            kept.append(stroke.difference(unary_union(above)) if above and not stroke.is_empty
                        else stroke)
        strokes = kept

    live = [stroke for stroke in strokes if not stroke.is_empty]
    region = unary_union(live) if live else Polygon()
    return region.intersection(box(*page))


def bridge_strokes_fn(ink, elements, *, spacing: float, width: float, margin: float = 0.15,
                      samples: int = 96, page=None):
    """Interrupt the strokes at intervals, so the ground they enclose reconnects.

    The dual of the thread, and the device the other polarity needs. When a stroke is the SLOT rather
    than the paper, a closed circuit of strokes cuts the ground into one piece per face -- a planar
    arrangement of loops on 280 elements leaves 349 of them. Threading cannot help: there is nothing
    isolated to hold, the ground is simply divided. What reconnects it is leaving the slot UNCUT here
    and there, which is what a carver does and what the interrupted slots of a real komon are.

    Args:
        spacing: arclength between bridges along each contour.
        width: the stroke weight being interrupted; the gap is sized from it.
        margin: extra gap beyond the stroke weight, as a fraction of it.

    Returns:
        ``(interrupted_ink, report)``.
    """
    radius = 0.5 * width * (1.0 + margin)
    gaps = []
    for element in elements:
        outline = leaf_polygon_fn(element.leaf, n=samples)
        closed = np.vstack([outline, outline[:1]])
        steps = np.linalg.norm(np.diff(closed, axis=0), axis=1)
        arc = np.concatenate([[0.0], np.cumsum(steps)])
        if arc[-1] <= spacing:
            continue
        # Sampled by arclength, so the bridges sit at an even pitch along the contour however the
        # outline's curvature bunches its vertices.
        wanted = np.arange(0.5 * spacing, arc[-1], spacing)
        points = np.stack([np.interp(wanted, arc, closed[:, 0]),
                           np.interp(wanted, arc, closed[:, 1])], axis=1)
        gaps.extend(Point(point).buffer(radius) for point in points)

    if not gaps:
        return ink, {"n_bridges": 0, "bridge_spacing_mm": float(spacing)}
    cut = unary_union(gaps)
    interrupted = ink.difference(cut)
    if page is not None:
        interrupted = interrupted.intersection(box(*page))
    return interrupted, {"n_bridges": len(gaps), "bridge_spacing_mm": float(spacing)}


def bridge_faces_fn(ink, *, page, width: float, margin: float = 0.6, rounds: int = 12):
    """Punch a gap through the stroke wherever it strands a face of the ground.

    The exact dual of the nick, and the pairing is worth stating: when a stroke is the PAPER the
    drawing traps islands and they are held by ADDING paper; when a stroke is the SLOT the drawing
    divides the ground into faces and they are joined by REMOVING slot. One device per stranded piece
    either way.

    ⚠⚠ Each face is bridged to its NEAREST NEIGHBOUR, never to the largest face. A drawing of closed
    loops has no dominant face at all -- on one measured sheet the ground was 349 cells whose biggest
    held 1% of the page -- so "the distance to the main component" is halfway across the sheet, and
    sizing a gap from it punches a hole that erases the drawing. Bridging neighbours keeps every gap
    the width of the one stroke it crosses, and halves the face count each round.

    ⚠ An even pitch of bridges along each contour will not do this job on its own: a face bounded by
    several crossing strokes can receive no bridge at all, and 2470 evenly spaced gaps still left 14
    pieces. Use the even pitch for the look and this for the guarantee.

    Returns:
        ``(interrupted_ink, report)``.
    """
    from shapely.ops import nearest_points

    window = box(*page)
    cap = 0.75 * width + margin
    gaps, rounds_used = [], 0
    for _ in range(rounds):
        parts = _components(window.difference(ink))
        if len(parts) <= 1:
            break
        rounds_used += 1
        tree = STRtree(parts)
        fresh, bridged = [], set()
        for index, part in enumerate(parts):
            other = int(tree.query_nearest(part, exclusive=True, all_matches=False)[0])
            key = (min(index, other), max(index, other))
            if key in bridged:
                continue
            bridged.add(key)
            here, there = nearest_points(part, parts[other])
            centre = np.array([0.5 * (here.x + there.x), 0.5 * (here.y + there.y)])
            fresh.append(Point(centre).buffer(min(0.5 * here.distance(there) + margin, cap)))
        if not fresh:
            break
        gaps.extend(fresh)
        ink = ink.difference(unary_union(fresh))

    return ink, {"n_face_bridges": len(gaps), "bridge_rounds": rounds_used,
                 "faces_left": len(_components(window.difference(ink)))}


def stroke_weight_report_fn(region, *, width: float):
    """Whether the ink really holds the weight it was drawn at, and how much of it is thinner.

    The even rule should come back at the width it was given; the varying rule should come back below
    it, by the tip pinch. Reporting the same two numbers for both is what makes that a measurement
    rather than an assertion.
    """
    return {"stroke_mm": float(width),
            "ink_min_mm": min_width_fn(region, upper=4.0 * width),
            "ink_at_weight": opening_retention_fn(region, 0.95 * width)}


# ---------------------------------------------------------------- the armature
#
# An armature is any region unioned into the paper to hold what the drawing alone leaves loose. The
# ladder below runs from the cheapest possible one -- a nick per trapped island -- to a whole komon
# net, and every rung is measured the same way, so the guarantee becomes a trade against tone rather
# than a fixed device.


def island_census_fn(paper, elements, specs, *, samples: int = 64):
    """Which white pieces are trapped, and inside which element.

    The largest component is the sheet; everything else is an island, attributed to the SMALLEST
    silhouette containing it, since a leaf lying across another traps the island in its own band and
    not in the one underneath.

    Returns:
        ``(report, islands)`` where ``islands`` is ``[(component, host_element_index)]`` with -1 for
        an island no silhouette contains. That list is what decides how many nicks are needed and
        where they go, so it is returned rather than counted and thrown away.
    """
    parts = _components(paper)
    if not parts:
        return {"n_islands": 0, "islands_area": 0.0, "n_hosts": 0, "islands_unhosted": 0}, []
    sheet = max(parts, key=lambda part: part.area)
    loose = [part for part in parts if part is not sheet]

    outlines = [Polygon(leaf_polygon_fn(
        element_layers_fn(element.leaf, specs[element.population])[0][0], n=samples)).buffer(0)
        for element in elements]
    tree = STRtree(outlines)

    islands = []
    for island in loose:
        probe = island.representative_point()
        found = [int(i) for i in tree.query(probe) if outlines[i].contains(probe)]
        host = min(found, key=lambda i: outlines[i].area) if found else -1
        islands.append((island, host))

    hosts = {host for _, host in islands if host >= 0}
    report = {
        "n_islands": len(islands),
        "islands_area": float(sum(island.area for island, _ in islands)),
        "n_hosts": len(hosts),
        "islands_unhosted": sum(1 for _, host in islands if host < 0),
    }
    return report, islands


def island_tabs_fn(islands, elements, sheet, *, width: float, reach: float = 1.6, page=None):
    """One nick per trapped island, cut along its host element's own axis.

    Minimal in count: an island needs exactly one attachment, and a radial cut from the island out
    through the base crosses every band of the nesting at once, so one nick frees the whole stack.

    ⚠ The nick has to REACH the sheet, and a fixed overshoot past the base does not: outside its own
    silhouette an element may sit in a neighbour's core, which is hole, not paper. The segment is
    therefore run outward until it leaves the dark, by subtracting the sheet from the ray and keeping
    the piece still attached to the island. Where the ray never arrives, the shortest segment to the
    sheet is used instead and counted, so the fallback is visible rather than silent.

    Args:
        reach: how far to cast the ray, in leaf lengths.
    """
    from shapely.ops import nearest_points

    strokes, fallbacks = [], 0
    for island, host in islands:
        start = island.representative_point()
        if host >= 0:
            leaf = elements[host].leaf
            axis = np.array([np.cos(leaf.angle), np.sin(leaf.angle)])
            ray = LineString([start, np.asarray(start.coords)[0] - reach * leaf.length * axis])
            outside = ray.difference(sheet)
            pieces = list(outside.geoms) if outside.geom_type == "MultiLineString" else [outside]
            attached = [piece for piece in pieces
                        if not piece.is_empty and piece.distance(start) < 1e-6]
            # Attached and shorter than the whole ray means the ray genuinely arrived somewhere.
            if attached and attached[0].length < 0.98 * ray.length:
                strokes.append(attached[0])
                continue
        fallbacks += 1
        strokes.append(LineString(nearest_points(island, sheet)))

    region = unary_union([stroke.buffer(0.5 * width, cap_style="round", join_style="round")
                          for stroke in strokes]) if strokes else Polygon()
    if page is not None:
        region = region.intersection(box(*page))
    return region, {"n_tabs": len(strokes), "tab_fallbacks": fallbacks}


def _delaunay_edges(sites):
    from scipy.spatial import Delaunay

    pairs = set()
    for triangle in Delaunay(np.asarray(sites, float)).simplices:
        for one, other in ((0, 1), (1, 2), (2, 0)):
            a, b = int(triangle[one]), int(triangle[other])
            pairs.add((min(a, b), max(a, b)))
    return sorted(pairs)


def mst_thread_fn(sites):
    """The cheapest spanning armature over the sites: ``n-1`` struts instead of grid mode's ``2n``.

    Built over the Delaunay edges rather than all pairs, which contains a minimum spanning tree of the
    complete graph and keeps the struts between actual neighbours.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import minimum_spanning_tree

    sites = np.asarray(sites, float)
    edges = _delaunay_edges(sites)
    lengths = [float(np.linalg.norm(sites[one] - sites[other])) for one, other in edges]
    rows = [one for one, _ in edges]
    cols = [other for _, other in edges]
    graph = coo_matrix((lengths, (rows, cols)), shape=(len(sites), len(sites)))
    tree = minimum_spanning_tree(graph).tocoo()
    kept = sorted((min(int(a), int(b)), max(int(a), int(b)))
                  for a, b in zip(tree.row, tree.col))
    return kept, {"n_edges": len(kept), "thread_connected": len(kept) == len(sites) - 1,
                  "strut_length_mm": float(tree.data.sum())}


def polyline_region_fn(polylines, *, width: float, page=None):
    """Buffer a set of polylines into paper, round-capped.

    Round, for the reason the nets already showed: two pieces meeting at a shallow angle with flat
    caps leave a notch that pinches the strut to well under its own width, so the bridge would stop
    being the number the design named.
    """
    strokes = [LineString(np.asarray(points, float)).buffer(
        0.5 * width, cap_style="round", join_style="round")
        for points in polylines if len(np.asarray(points)) >= 2]
    region = unary_union(strokes) if strokes else Polygon()
    return region.intersection(box(*page)) if page is not None else region


def chain_thread_fn(tiles, *, scale: float = 1.0, offset=(0.0, 0.0), rel_tol: float = 1e-9):
    """One continuous stroke per diagonal chain of the tiling.

    Nothing is invented here. Pairing each tile's opposite corners and following them through the
    corners they share gives a 2-regular graph, so the sheet's own diagonals fall into disjoint chains
    and cycles. A chain runs corner to corner, which is BETWEEN the elements rather than through their
    centres, and a line that runs for twenty cells is read as a line while the grid it crosses is not
    read at all.

    Returns:
        ``(polylines, report)`` already in page coordinates.
    """
    from nff_paper.closed_support.katagami import diagonal_chains_fn

    structure = diagonal_chains_fn(np.asarray(tiles, float), rel_tol=rel_tol)
    positions = structure["corners"]["positions"]
    polylines, lengths = [], []
    for chain in structure["chains"]:
        points = positions[chain["nodes"]]
        if len(points) < 2:
            continue
        polylines.append(scale * points + np.asarray(offset, float))
        lengths.append(len(chain["edges"]))
    closed = sum(1 for chain in structure["chains"] if chain["closed"])
    return polylines, {
        "n_chains": len(polylines),
        "chain_cells_median": float(np.median(lengths)) if lengths else 0.0,
        "chain_cells_max": int(max(lengths)) if lengths else 0,
        "chains_closed": closed,
        "strut_length_mm": float(sum(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()
                                     for p in polylines)),
    }


def extend_mesh_fn(mesh: Float[np.ndarray, "rows cols 2"], *, rings: int = 1):
    """Pad a control mesh outward by linear extrapolation.

    ⚠⚠ Tile centroids are INTERIOR points, so a net built on the centroid mesh stops about half a
    pitch short of the sheet on every side. As artwork that is a fringe; as an armature it is a
    failure exactly where a sheet is weakest, because the border elements would have nothing holding
    them. One extrapolated ring carries the net past the edge, and the overhang is cropped to the
    page like any other bleed.
    """
    padded = np.asarray(mesh, float)
    for _ in range(int(rings)):
        padded = np.concatenate([(2.0 * padded[0] - padded[1])[None], padded,
                                 (2.0 * padded[-1] - padded[-2])[None]], axis=0)
        padded = np.concatenate([(2.0 * padded[:, 0] - padded[:, 1])[:, None], padded,
                                 (2.0 * padded[:, -1] - padded[:, -2])[:, None]], axis=1)
    return padded


def _folded_cells(mesh):
    """How many cells of a control mesh are not convex; a folded cell warps its net inside out."""
    corners = np.stack([mesh[:-1, :-1], mesh[:-1, 1:], mesh[1:, 1:], mesh[1:, :-1]], axis=2)
    edges = np.roll(corners, -1, axis=2) - corners
    previous = np.roll(edges, 1, axis=2)
    cross = previous[..., 0] * edges[..., 1] - previous[..., 1] * edges[..., 0]
    return int((~((cross > 0).all(axis=2) | (cross < 0).all(axis=2))).sum())


def net_thread_fn(tiles, grid, name, *, width: float, scale: float = 2.0, rings: int = 1,
                  transform=(1.0, (0.0, 0.0)), page=None, per_segment: int = 10):
    """A komon net as the armature: the thing holding the sheet is itself a motif.

    A net is one connected piece by construction and its narrowest bridge is its own strut width, so
    it supplies the guarantee with no tuning at all. Used under an OUTLINED element it stays visible,
    which is the difference from the earlier attempt to put a net under opaque figures.

    ⚠⚠ ``transform`` must be the transform the ELEMENTS were placed with. The net panels fit to the
    centroid mesh's bounds while element layouts fit to the sheet's, and unioning two regions built
    under different transforms slides the net off the leaves.

    Args:
        transform: ``(scale, offset)`` as applied to the element field.
    """
    from nff_paper.closed_support.net_field import control_mesh_fn
    from nff_paper.topology_art.net import build_net_fn, warp_net_fn
    from nff_paper.topology_art.quad_warp import path_points_fn

    mesh = extend_mesh_fn(control_mesh_fn(np.asarray(tiles, float), grid), rings=rings)
    rows, cols = mesh.shape[0] - 1, mesh.shape[1] - 1
    paths = warp_net_fn(build_net_fn(name, cols, rows, scale=scale), mesh)

    factor, offset = transform
    polylines = [factor * path_points_fn(path, per_segment=per_segment) + np.asarray(offset, float)
                 for path in paths]
    region = polyline_region_fn(polylines, width=width, page=page)
    return region, {
        "net": name, "n_pieces": len(paths), "cells": rows * cols,
        "folded_cells": _folded_cells(mesh),
        "strut_length_mm": float(sum(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()
                                     for p in polylines)),
    }


def thread_cost_fn(paper, armature, *, page) -> float:
    """What the guarantee costs in tone: the paper the armature adds, as a fraction of the page.

    Zero for an armature that lies entirely inside white the drawing already had, which is the ideal
    a routing is aiming at.
    """
    window = box(*page)
    added = armature.intersection(window).difference(paper)
    return float(added.area / window.area)


def fans_opened_fn(elements, specs, armature, *, samples: int = 64):
    """How many closed contours the armature cut open.

    The nested comb draws bundles of CLOSED rings with an oval eye; a cut stencil's are OPEN fans
    whose arcs terminate. An armature crossing the nesting terminates every arc at the same
    parametric place, so it converts one into the other without needing a partial-outline element.
    A band counts as opened when it no longer encloses anything once the armature is taken out of it.
    """
    closed = opened = 0
    for element in elements:
        dark = element_dark_fn(element.leaf, specs[element.population], samples=samples)
        for band in _components(dark):
            if not band.interiors:
                continue
            closed += 1
            cut = band.difference(armature)
            if all(not part.interiors for part in _components(cut)):
                opened += 1
    return {"rings_closed": closed, "rings_opened": opened,
            "fans_fraction": float(opened / max(closed, 1))}


# ---------------------------------------------------------------- drawing


def render_regions_fn(layers, out_png, *, page, ground="#141414", dpi: int = 150,
                      figure_width_in: float = 7.0):
    """Paint ``[(region, colour)]`` in order, holes and all: a picture of objects, not of elements.

    More than one region because a diagnostic needs it. The island census only means something drawn,
    and it has to distinguish the sheet from the pieces that would fall out of it.
    """
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path

    from nff.utils import figstyle

    figstyle.use_agg()
    import matplotlib.pyplot as plt

    figstyle.apply_charter(**{"axes.grid": False})

    def patch_of(region, colour):
        vertices, codes = [], []
        for component in _components(region):
            for ring in (component.exterior, *component.interiors):
                points = np.asarray(ring.coords)[:-1]
                vertices.extend(points)
                codes.extend([Path.MOVETO] + [Path.LINETO] * (len(points) - 1))
                vertices.append(points[0])
                codes.append(Path.CLOSEPOLY)
        if not vertices:
            return None
        return PathPatch(Path(np.asarray(vertices), codes), facecolor=colour, edgecolor="none")

    span_x, span_y = page[2] - page[0], page[3] - page[1]
    figure, axis = plt.subplots(figsize=(figure_width_in, figure_width_in * span_y / span_x))
    figure.patch.set_facecolor(ground)
    axis.set_facecolor(ground)
    for region, colour in layers:
        patch = patch_of(region, colour)
        if patch is not None:
            axis.add_patch(patch)
    axis.set_xlim(page[0], page[2])
    axis.set_ylim(page[1], page[3])
    axis.set_aspect("equal")
    axis.axis("off")
    figure.savefig(out_png, dpi=dpi, bbox_inches="tight", pad_inches=0.0, facecolor=ground)
    plt.close(figure)
    return out_png


def render_region_fn(paper, out_png, *, page, fill="#F2EFE6", ground="#141414",
                     dpi: int = 150, figure_width_in: float = 7.0, thread=None):
    """One region, painted as the paper.

    ``thread`` overpaints the armature where it lies inside the paper, which is the only way to see
    whether it reads as part of the pattern or as a grid laid over it.
    """
    layers = [(paper, fill)]
    if thread is not None:
        layers.append((thread.intersection(paper), fill))
    return render_regions_fn(layers, out_png, page=page, ground=ground, dpi=dpi,
                             figure_width_in=figure_width_in)
