"""Motifs for the cell web: a shape per Voronoi territory, fitted and turned by the flow.

The cell web's guarantee has nothing to do with what the hole looks like. A territory is its site's
Voronoi cell eroded by half the slot, so ANY shape contained in it keeps every neighbouring pair a
slot apart and leaves the white as one connected skeleton. The motif is therefore free, and this
module is the vocabulary.

Each motif is written once as a UNIT shape -- centred on the origin, roughly unit radius, drawn along
+x -- and then rotated to the local flow and grown until it just fits its cell. Writing them that way
means a motif never has to know about cells, sizes or slots, and adding one is a single function
returning a polygon.

⚠ One motif in the registry deliberately breaks the sheet: ``ring`` encloses its own white, and an
enclosed island is a separate piece of paper however wide the web around it. It is kept because the
failure is worth seeing, and because it is the same island problem the threads solved elsewhere.

Pure geometry over shapely; imports nothing above ``nff.topology``.
"""

import numpy as np
from shapely.affinity import rotate as affine_rotate
from shapely.affinity import scale as affine_scale
from shapely.affinity import translate as affine_translate
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from legacy.katagami.topology_art.leaf import Leaf, leaf_polygon_fn


def _regular(sides: int, *, radius: float = 1.0, phase: float = 0.0) -> Polygon:
    angles = phase + np.arange(sides) * (2.0 * np.pi / sides)
    return Polygon(np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1))


def unit_disc_fn() -> Polygon:
    return Point(0.0, 0.0).buffer(1.0, quad_segs=48)


def unit_polygon_fn(sides: int = 6) -> Polygon:
    """A regular polygon. Three sides is uroko, the scale pattern; six is the tortoiseshell cell."""
    return _regular(sides, phase=np.pi / 2.0)


def unit_star_fn(points: int = 6, *, inner: float = 0.45) -> Polygon:
    step = np.pi / points
    angles = np.arange(2 * points) * step + np.pi / 2.0
    radii = np.where(np.arange(2 * points) % 2 == 0, 1.0, inner)
    return Polygon(np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1))


def unit_crescent_fn(*, bite: float = 0.86, offset: float = 0.52) -> Polygon:
    """A lune: one disc with a second bitten out of its side."""
    return unit_disc_fn().difference(Point(offset, 0.0).buffer(bite, quad_segs=48))


def unit_magatama_fn(*, turns: float = 0.62, head: float = 0.46, samples: int = 96) -> Polygon:
    """The comma bead: a tail that sweeps round and swells into a head.

    Built as a centreline of decreasing curvature carrying a half-width that grows from nothing at the
    tail to ``head`` at the round end, which is what a magatama actually is. A constant-width buffer
    cannot express it, so the outline is walked explicitly.
    """
    t = np.linspace(0.0, 1.0, samples)
    angle = np.pi * turns * (1.0 - t) * 2.0
    radius = 0.42 + 0.34 * t
    centre = np.stack([radius * np.cos(angle), radius * np.sin(angle)], axis=1)
    width = head * t**0.75

    tangent = np.gradient(centre, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-12)
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
    outline = np.vstack([centre + width[:, None] * normal,
                         (centre - width[:, None] * normal)[::-1]])
    shape = Polygon(outline).buffer(0)
    return max(shape.geoms, key=lambda part: part.area) if shape.geom_type == "MultiPolygon" \
        else shape


def unit_rosette_fn(petals: int = 5, *, reach: float = 0.52, petal: float = 0.60) -> Polygon:
    """Petals round a centre: five is sakura, the cherry blossom.

    ⚠⚠ The petals MUST cover the middle. At ``reach`` 0.58 with ``petal`` 0.46 they ring an empty
    centre, and the sheet then sheds one small disc per flower -- measured at 255 dropped islands on a
    280-cell sheet, invisible in the picture and obvious in the count. A disc of the reach radius is
    unioned in as a backstop so the flower is solid whatever the two radii are set to.
    """
    angles = np.pi / 2.0 + np.arange(petals) * (2.0 * np.pi / petals)
    return unary_union([Point(reach * np.cos(a), reach * np.sin(a)).buffer(petal, quad_segs=32)
                        for a in angles] + [Point(0.0, 0.0).buffer(reach, quad_segs=32)])


def unit_slit_fn(*, thickness: float = 0.20) -> Polygon:
    """A bar along the flow: the plainest komon of all, and the cheapest to cut."""
    return Polygon([(-1.0, -thickness), (1.0, -thickness), (1.0, thickness), (-1.0, thickness)])


def unit_cross_fn(*, thickness: float = 0.22, arm: float = 0.9) -> Polygon:
    bar = Polygon([(-arm, -thickness), (arm, -thickness), (arm, thickness), (-arm, thickness)])
    return bar.union(affine_rotate(bar, 90.0, origin=(0.0, 0.0)))


def unit_ring_fn(*, inner: float = 0.55) -> Polygon:
    """⚠ Encloses its own white. Kept as the control that shows what the guarantee does NOT cover."""
    return unit_disc_fn().difference(Point(0.0, 0.0).buffer(inner, quad_segs=48))


def unit_cluster_fn(count: int = 3, *, reach: float = 0.55, dot: float = 0.34) -> Polygon:
    """Several holes in one cell, which is how a fine komon gets its density."""
    if count <= 1:
        return Point(0.0, 0.0).buffer(dot + reach, quad_segs=32)
    angles = np.pi / 2.0 + np.arange(count) * (2.0 * np.pi / count)
    return unary_union([Point(reach * np.cos(a), reach * np.sin(a)).buffer(dot, quad_segs=32)
                        for a in angles])


def unit_leaf_fn(*, width: float = 0.62, bow: float = 0.22) -> Polygon:
    """The element the rest of the study is built on, as a unit shape."""
    return Polygon(leaf_polygon_fn(Leaf(centre=np.zeros(2), angle=0.0, length=2.0,
                                        width=width, bow=bow), n=96))


def unit_ellipse_fn(*, squash: float = 0.55) -> Polygon:
    return affine_scale(unit_disc_fn(), xfact=1.0, yfact=squash, origin=(0.0, 0.0))


def unit_vesica_fn(*, offset: float = 0.62) -> Polygon:
    """The lens two overlapping circles leave: a pointed oval, and a leaf without a direction."""
    return Point(-offset, 0.0).buffer(1.0, quad_segs=48).intersection(
        Point(offset, 0.0).buffer(1.0, quad_segs=48))


def unit_teardrop_fn(*, head: float = 0.62) -> Polygon:
    """A round head drawn to a point: the convex hull of a disc and the tip beyond it."""
    return unary_union([Point(-0.3, 0.0).buffer(head, quad_segs=48),
                        Polygon([(-0.3, head), (1.0, 0.0), (-0.3, -head)])]).convex_hull


def unit_stadium_fn(*, thickness: float = 0.34) -> Polygon:
    """A bar with round ends: a slit that will not tear at its corners."""
    return LineString([(-1.0 + thickness, 0.0), (1.0 - thickness, 0.0)]).buffer(
        thickness, quad_segs=32)


def unit_squircle_fn(*, power: float = 4.0, samples: int = 160) -> Polygon:
    """A superellipse: the square softened toward the circle, or the circle stiffened."""
    t = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False)
    return Polygon(np.stack([np.sign(np.cos(t)) * np.abs(np.cos(t)) ** (2.0 / power),
                             np.sign(np.sin(t)) * np.abs(np.sin(t)) ** (2.0 / power)], axis=1))


def unit_sector_fn(*, spread: float = 100.0, samples: int = 64) -> Polygon:
    """A fan: the circular sector, and the unit of a seigaiha wave."""
    angles = np.radians(np.linspace(-spread / 2.0, spread / 2.0, samples))
    arc = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    return Polygon(np.vstack([[[0.0, 0.0]], arc]))


def unit_chevron_fn(*, thickness: float = 0.30, spread: float = 0.9) -> Polygon:
    """The arrow feather, yagasuri: a V that points along the flow."""
    outer = [(-spread, 1.0), (spread, 0.0), (-spread, -1.0)]
    inner = [(-spread + thickness * 1.8, 0.0)]
    return Polygon(outer + inner)


def unit_matsuba_fn(*, thickness: float = 0.13, spread: float = 0.42) -> Polygon:
    """Pine needles: two blades from one root, which is what keeps them a single piece."""
    root = np.array([-1.0, 0.0])
    blades = [LineString([root, root + np.array([2.0 * np.cos(sign * spread),
                                                 2.0 * np.sin(sign * spread)])]).buffer(
        thickness, cap_style="round") for sign in (+1.0, -1.0)]
    return unary_union(blades)


def unit_gourd_fn(*, upper: float = 0.44, lower: float = 0.62) -> Polygon:
    """Hyotan, the double gourd: a small head over a fuller body."""
    return unary_union([Point(0.0, 1.0 - upper).buffer(upper, quad_segs=32),
                        Point(0.0, lower - 1.0).buffer(lower, quad_segs=32),
                        Polygon([(-0.22, 0.4), (0.22, 0.4), (0.22, -0.2), (-0.22, -0.2)])])


def unit_hexagram_fn() -> Polygon:
    """Kikkou's star: two triangles crossed, solid rather than outlined."""
    up = _regular(3, phase=np.pi / 2.0)
    return up.union(affine_rotate(up, 180.0, origin=(0.0, 0.0)))


def unit_tomoe_fn(count: int = 3, *, turns: float = 0.55) -> Polygon:
    """Mitsudomoe: commas chasing each other round a common centre.

    The heads are pulled in to meet at the middle, and a central disc is unioned in, because three
    commas ringing an empty centre would shed a piece per cell exactly as the rosette did.
    """
    comma = affine_scale(unit_magatama_fn(turns=turns), xfact=0.72, yfact=0.72, origin=(0.0, 0.0))
    arms = [affine_rotate(comma, 360.0 * index / count, origin=(0.0, 0.0)) for index in range(count)]
    return unary_union(arms + [Point(0.0, 0.0).buffer(0.3, quad_segs=32)])


def unit_wave_fn(*, crests: int = 3, depth: float = 0.34, samples: int = 120) -> Polygon:
    """A band with a scalloped edge: the crest of a seigaiha read as a solid."""
    x = np.linspace(-1.0, 1.0, samples)
    top = 0.15 + depth * np.cos(crests * np.pi * x)
    return Polygon(np.vstack([np.stack([x, top], axis=1),
                              np.stack([x[::-1], np.full(samples, -0.55)], axis=1)]))


def unit_bars_fn(count: int = 2, *, thickness: float = 0.16, pitch: float = 0.5) -> Polygon:
    """Parallel strokes. Separate pieces of hole, which is legal: only WHITE may not be enclosed."""
    offsets = (np.arange(count) - 0.5 * (count - 1)) * pitch
    return unary_union([Polygon([(-1.0, y - thickness), (1.0, y - thickness),
                                 (1.0, y + thickness), (-1.0, y + thickness)]) for y in offsets])


def unit_diamond_fn(*, squash: float = 0.6) -> Polygon:
    return affine_scale(_regular(4, phase=0.0), xfact=1.0, yfact=squash, origin=(0.0, 0.0))


def unit_trapezoid_fn(*, top: float = 0.45) -> Polygon:
    return Polygon([(-1.0, -0.7), (1.0, -0.7), (top, 0.8), (-top, 0.8)])


UNIT_MOTIFS = {
    "disc": unit_disc_fn,
    "triangle": lambda: unit_polygon_fn(3),
    "square": lambda: unit_polygon_fn(4),
    "hexagon": lambda: unit_polygon_fn(6),
    "star": lambda: unit_star_fn(6),
    "star5": lambda: unit_star_fn(5, inner=0.40),
    "crescent": unit_crescent_fn,
    "magatama": unit_magatama_fn,
    "sakura": lambda: unit_rosette_fn(5),
    "rosette8": lambda: unit_rosette_fn(8, reach=0.60, petal=0.46),
    "slit": unit_slit_fn,
    "cross": unit_cross_fn,
    "ring": unit_ring_fn,
    "cluster3": lambda: unit_cluster_fn(3),
    "cluster7": lambda: unit_cluster_fn(7, reach=0.62, dot=0.26),
    "leaf": unit_leaf_fn,
    # -- the wider vocabulary
    "pentagon": lambda: unit_polygon_fn(5),
    "heptagon": lambda: unit_polygon_fn(7),
    "octagon": lambda: unit_polygon_fn(8),
    "decagon": lambda: unit_polygon_fn(10),
    "star3": lambda: unit_star_fn(3, inner=0.42),
    "star4": lambda: unit_star_fn(4, inner=0.42),
    "star8": lambda: unit_star_fn(8, inner=0.55),
    "star12": lambda: unit_star_fn(12, inner=0.68),
    "ellipse": unit_ellipse_fn,
    "vesica": unit_vesica_fn,
    "teardrop": unit_teardrop_fn,
    "stadium": unit_stadium_fn,
    "squircle": unit_squircle_fn,
    "sector": unit_sector_fn,
    "chevron": unit_chevron_fn,
    "matsuba": unit_matsuba_fn,
    "gourd": unit_gourd_fn,
    "hexagram": unit_hexagram_fn,
    "mitsudomoe": unit_tomoe_fn,
    "futatsudomoe": lambda: unit_tomoe_fn(2),
    "wave": unit_wave_fn,
    "bars2": lambda: unit_bars_fn(2),
    "bars3": lambda: unit_bars_fn(3, thickness=0.11, pitch=0.42),
    "diamond": unit_diamond_fn,
    "trapezoid": unit_trapezoid_fn,
    "kiku": lambda: unit_rosette_fn(12, reach=0.62, petal=0.34),
    "clover": lambda: unit_rosette_fn(3, reach=0.50, petal=0.62),
    "rosette6": lambda: unit_rosette_fn(6, reach=0.55, petal=0.52),
}


def fit_motif_fn(name: str, territory, *, site, angle: float = 0.0, reach: float = 1.0,
                 steps: int = 14, params: dict | None = None):
    """A unit motif rotated to the flow, centred on the site, and sized against its cell.

    The inscribed size is bisected for rather than solved, because "the largest copy that fits" has no
    closed form for an arbitrary shape in an arbitrary convex cell. Fourteen steps put it within 0.01%
    of the cell's own size, far finer than the slot.

    Args:
        reach: multiples of that inscribed size. **1 is the whole motif, and it is SPARSE** -- a shape
            inscribed in a polygon is much smaller than the polygon, so every motif lands around 80%
            paper at reach 1. Above 1 the motif is allowed to outgrow its cell and is then cropped by
            it, which is what carries the family to the dense end: the black keeps the motif's
            character at its corners while filling the cell like a paving stone. Below 1 it pulls
            further in and the web opens.

    The crop is what keeps the guarantee intact at any reach: the result is intersected with the
    territory, so it can never leave the cell however far it is grown.
    """
    if territory.is_empty:
        return Polygon()
    unit = UNIT_MOTIFS[name](**(params or {}))
    turned = affine_rotate(unit, angle, origin=(0.0, 0.0), use_radians=True)
    reference = affine_translate(turned, float(site[0]), float(site[1]))

    def at(scale):
        return affine_scale(reference, xfact=scale, yfact=scale, origin=Point(site))

    # An upper bound that certainly fails: the cell cannot hold a copy wider than its own bounds.
    low, high = 0.0, max(territory.bounds[2] - territory.bounds[0],
                         territory.bounds[3] - territory.bounds[1]) + 1.0
    if territory.contains(at(high)):
        low = high
    else:
        for _ in range(steps):
            middle = 0.5 * (low + high)
            if territory.contains(at(middle)):
                low = middle
            else:
                high = middle
    if low <= 0.0:
        return Polygon()
    grown = at(low * reach)
    return grown if reach <= 1.0 else grown.intersection(territory)


def motif_holes_fn(territories, sites, angles, *, name: str, reach: float = 1.0,
                   detail: dict | None = None, params: dict | None = None):
    """One motif per territory, optionally choosing the motif by how big the cell is.

    Args:
        detail: ``{"large": name, "small": name, "quantile": q}`` to give the roomier cells a more
            elaborate figure and the cramped ones a plain one. A carver does exactly this, and it is
            the only way an elaborate motif survives on a sheet whose cells differ tenfold in area.
    """
    areas = np.array([territory.area for territory in territories], float)
    chosen = [name] * len(territories)
    if detail:
        cut = float(np.quantile(areas[areas > 0], detail.get("quantile", 0.5))) if (areas > 0).any() \
            else 0.0
        chosen = [detail["large"] if area >= cut else detail["small"] for area in areas]
    return [fit_motif_fn(pick, territory, site=site, angle=angle, reach=reach, params=params)
            for pick, territory, site, angle in zip(chosen, territories, sites, angles)]
