"""The katagami element: a leaf, rounded at the base and drawn to a point at the tip.

The element has to be **directional**. A shape that is symmetric end to end -- a lens, a crescent --
gives the eye no way to read which way a scale lies, so a field of them looks like clutter however
carefully it is packed. A leaf has a base and a tip, so a whole field of them can shingle in one
consistent direction, which is what makes the reference stencils legible.

The construction is three circular arcs:

- a **base circle** of radius ``r``, giving the rounded end;
- two **flank arcs**, each tangent to the base circle at its tangent point and passing through the
  tip. Tangency at the base is what stops the outline showing a corner where flank meets base; the
  two flanks meet at the tip at an angle, which is what makes the tip a point.

Everything is exact under a similarity, so each leaf is written as three SVG ``A`` commands and stays
smooth at any magnification.

Pure numpy; this module imports nothing from ``nff`` outside ``nff.topology``.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from jaxtyping import Float

from nff.topology.quad_warp import BezierPath


@dataclass(frozen=True)
class Leaf:
    """One element. ``angle`` points from base to TIP, so the shape carries a direction."""

    centre: Float[np.ndarray, "2"]
    angle: float
    length: float
    width: float = 0.46  # base diameter as a fraction of length; must stay below 1
    bow: float = 0.30    # 0 gives straight flanks (a teardrop); larger bows them outward


def transform_leaf_fn(leaf: Leaf, *, scale: float, offset) -> Leaf:
    """Apply a similarity at the parameter level, so the arcs never need resampling."""
    return Leaf(
        centre=scale * np.asarray(leaf.centre, float) + np.asarray(offset, float),
        angle=leaf.angle,
        length=scale * leaf.length,
        width=leaf.width,
        bow=leaf.bow,
    )


def _frame(leaf):
    """Base circle, tangent points and tip, in world coordinates."""
    radius = 0.5 * leaf.width * leaf.length
    axis = np.array([np.cos(leaf.angle), np.sin(leaf.angle)])
    side = np.array([-axis[1], axis[0]])
    tip = np.asarray(leaf.centre, float) + 0.5 * leaf.length * axis
    base = np.asarray(leaf.centre, float) - 0.5 * leaf.length * axis + radius * axis

    span = float(np.linalg.norm(tip - base))
    # The tangent-from-tip point is the ONE place where a flank tangent to the base circle is forced
    # to be straight, because there the tip lies on the tangent line itself. Sliding the shoulder
    # further round the base is what lets the flank bow while still meeting the base without a
    # corner, so ``bow`` is a genuine degree of freedom rather than a cosmetic fudge.
    theta_tangent = float(np.arccos(np.clip(radius / span, -1.0, 1.0)))
    theta = theta_tangent + float(np.clip(leaf.bow, 0.0, 0.95)) * (np.pi - theta_tangent)
    touch = [base + radius * (np.cos(theta) * axis + sign * np.sin(theta) * side)
             for sign in (+1.0, -1.0)]
    return base, radius, tip, touch


def _flank_circle(touch, base, tip):
    """Circle through ``tip`` and tangent to the base circle at ``touch``.

    Its centre lies on the base-circle normal through the tangent point, which is exactly the
    condition that makes flank and base meet without a corner.
    """
    normal = (touch - base) / np.linalg.norm(touch - base)
    offset = touch - tip
    denominator = float(offset @ normal)
    if abs(denominator) < 1e-12:
        return None, None  # flank is straight
    radius = -float(offset @ offset) / (2.0 * denominator)
    return touch + radius * normal, abs(radius)


def _pick_direction(a0, a1, *, minor=None, via=None):
    """Which way round the circle to go, by size or by a point the arc must pass through.

    Choosing by "major or minor" is only safe when the answer cannot flip. It does flip here: once
    the shoulders slide past ninety degrees the arc round the BACK of the base stops being the major
    one, and a size test silently returns the arc across the front instead. Naming the point the arc
    has to pass through is unambiguous at any shoulder angle.
    """
    full = 2.0 * np.pi
    ccw_span = (a1 - a0) % full
    if via is not None:
        take_ccw = ((via - a0) % full) < ccw_span
    else:
        take_ccw = (ccw_span <= np.pi) if minor else (ccw_span > np.pi)
    return take_ccw, ccw_span if take_ccw else full - ccw_span


def _arc_flags(start, end, centre, *, minor=True, via=None):
    """``(radius, large_arc, sweep)`` for the arc between two points about a known centre."""
    start, end, centre = (np.asarray(v, float) for v in (start, end, centre))
    radius = float(np.linalg.norm(start - centre))
    a0 = float(np.arctan2(*(start - centre)[::-1]))
    a1 = float(np.arctan2(*(end - centre)[::-1]))
    via_angle = None if via is None else float(np.arctan2(*(np.asarray(via, float) - centre)[::-1]))
    take_ccw, span = _pick_direction(a0, a1, minor=minor, via=via_angle)
    return radius, int(span > np.pi), int(take_ccw)


def leaf_path_fn(leaf: Leaf) -> BezierPath:
    """The element as a closed path of three exact arcs: flank, base, flank."""
    base, radius, tip, touch = _frame(leaf)

    axis = np.array([np.cos(leaf.angle), np.sin(leaf.angle)])
    back = base - radius * axis

    def arc(start, end, centre, *, minor=True, via=None):
        if centre is None:
            return ("L", np.array([end]))
        arc_radius, large_arc, sweep = _arc_flags(start, end, centre, minor=minor, via=via)
        return ("A", np.array([[arc_radius, arc_radius], [large_arc, sweep], end]))

    lower_centre, _ = _flank_circle(touch[1], base, tip)
    upper_centre, _ = _flank_circle(touch[0], base, tip)
    return BezierPath(
        start=tip,
        segments=(
            arc(tip, touch[1], lower_centre, minor=True),
            arc(touch[1], touch[0], base, via=back),      # round the BACK of the base
            arc(touch[0], tip, upper_centre, minor=True),
        ),
        closed=True,
    )


def _sample_arc(start, end, centre, *, minor=True, via=None, n=48):
    start, end, centre = (np.asarray(v, float) for v in (start, end, centre))
    radius = float(np.linalg.norm(start - centre))
    a0 = float(np.arctan2(*(start - centre)[::-1]))
    a1 = float(np.arctan2(*(end - centre)[::-1]))
    via_angle = None if via is None else float(np.arctan2(*(np.asarray(via, float) - centre)[::-1]))
    take_ccw, span = _pick_direction(a0, a1, minor=minor, via=via_angle)
    angles = a0 + np.linspace(0.0, span if take_ccw else -span, n)
    return centre + radius * np.stack([np.cos(angles), np.sin(angles)], axis=1)


def leaf_polygon_fn(leaf: Leaf, *, n: int = 48) -> Float[np.ndarray, "m 2"]:
    """Densely sampled outline. For the raster preview, tests and packing checks only."""
    base, radius, tip, touch = _frame(leaf)
    centre_lower, _ = _flank_circle(touch[1], base, tip)
    centre_upper, _ = _flank_circle(touch[0], base, tip)
    axis = np.array([np.cos(leaf.angle), np.sin(leaf.angle)])
    lower = _sample_arc(tip, touch[1], centre_lower, minor=True, n=n)
    around = _sample_arc(touch[1], touch[0], base, via=base - radius * axis, n=n)
    upper = _sample_arc(touch[0], tip, centre_upper, minor=True, n=n)
    return np.concatenate([lower[:-1], around[:-1], upper[:-1]], axis=0)


def leaf_tip_fn(leaf: Leaf) -> Float[np.ndarray, "2"]:
    """The tip point, which is where the accent dot sits."""
    axis = np.array([np.cos(leaf.angle), np.sin(leaf.angle)])
    return np.asarray(leaf.centre, float) + 0.5 * leaf.length * axis


def dot_path_fn(centre: Float[np.ndarray, "2"], radius: float) -> BezierPath:
    """A full circle as two exact semicircular arcs.

    SVG cannot express a whole circle in one arc command -- start and end would coincide and the
    command is defined to be a no-op -- so a circle is always two.
    """
    centre = np.asarray(centre, float)
    left = centre - np.array([radius, 0.0])
    right = centre + np.array([radius, 0.0])
    return BezierPath(
        start=left,
        segments=(
            ("A", np.array([[radius, radius], [0, 1], right])),
            ("A", np.array([[radius, radius], [0, 1], left])),
        ),
        closed=True,
    )


def dot_placement_fn(width: float, bow: float, radius_frac: float, *,
                     margin: float = 0.015, samples: int = 400) -> tuple[float, float]:
    """Where the largest disc of the requested size fits, as close to the tip as possible.

    Returned in units of leaf length, measured along the axis from the leaf's centre. Because every
    leaf is the same shape up to a similarity, this is solved once on the canonical leaf and then
    scaled, rather than per element.

    The leaf narrows toward the tip, so a large disc and a tip-ward position are in direct conflict.
    This walks the disc forward until it would break the outline, and shrinks it only if it cannot
    fit anywhere.
    """
    canonical = Leaf(centre=np.zeros(2), angle=0.0, length=1.0, width=width, bow=bow)
    outline = leaf_polygon_fn(canonical, n=samples)
    axis_x = np.linspace(-0.5, 0.5, samples)
    clearance = np.array([np.linalg.norm(outline - np.array([x, 0.0]), axis=1).min()
                          for x in axis_x])

    radius = float(radius_frac)
    while radius > 1e-4:
        feasible = np.flatnonzero(clearance >= radius + margin)
        if feasible.size:
            return float(axis_x[feasible[-1]]), radius
        radius *= 0.92
    return 0.0, radius


@dataclass(frozen=True)
class ElementSpec:
    """How one leaf is inked: the nested bands, and the shape of the core inside them.

    Kept apart from ``Leaf`` because it says nothing about where an element is or how big it is.
    That separation is what lets a whole tonal range be drawn over ONE packing, so the comparison
    is of the ink alone.
    """

    rings: int = 2               # 2 is a pale shape with an inked core; 9+ reads as a comb
    rim: float = 0.075           # band width as a fraction of leaf length
    rim_abs: float = 0.0         # if positive, band width in world units instead
    duty: float = 0.5            # pale fraction of each band period
    # Both of the next two are fractions of the rim the core actually has to move within, not
    # lengths or angles. A shift of half a leaf length is meaningless when the border is a
    # fortieth of one; measured against the rim, 1 means the core just touches the outline at the
    # base, and the whole interesting range is [0, 1) whatever the rim happens to be.
    core_shift: float = 0.0      # slide the core toward the base; 1 spends the whole rim
    core_turn: float = 0.0       # rotate the core off the leaf's axis; 1 spends the whole rim
    core_width_delta: float = 0.0
    core_bow_delta: float = 0.0
    invert: bool = False         # a dark leaf holding a pale core, on the pale ground


def element_layers_fn(leaf: Leaf, spec: ElementSpec) -> list[tuple[Leaf, str]]:
    """One element as its painted layers, outermost first, each tagged ``'pale'`` or ``'dark'``.

    The single place the nesting is decided. Vector output and raster preview both consume this, so
    a change to the element cannot land in one and miss the other.
    """
    outer, inner = ("dark", "pale") if spec.invert else ("pale", "dark")
    layers = [(leaf, outer)]
    band = rim_fraction_fn(leaf.length, rim_frac=spec.rim, rim_abs=spec.rim_abs)
    if band <= 0.0:
        return layers

    span = max(spec.rings - 1, 1)
    for ring in range(1, max(spec.rings, 1)):
        # Bands alternate ink, but need not be equal: `duty` sets how much of each period stays
        # pale, which is what separates a thin cut slot from a 50/50 stripe.
        period, phase = divmod(ring, 2)
        # Deeper rings take proportionally more of the shift, so a comb sweeps instead of sliding
        # bodily. With two rings the single core takes all of it.
        reach = ring / span
        rim = band * (period + (spec.duty if phase else 0.0))
        # The rim to this ring IS the clearance it has, so it sets what a full shift or turn means.
        # A turn moves the core's ends sideways by about half its length times the angle, so
        # spending the whole rim that way is an angle of twice it; the margin keeps the containment
        # check a backstop rather than the thing that decides the range.
        travel = 0.95 * rim
        nested = inner_leaf_fn(
            leaf, rim,
            shift=spec.core_shift * reach * travel, turn=spec.core_turn * reach * 2.0 * travel,
            width_delta=spec.core_width_delta * reach, bow_delta=spec.core_bow_delta * reach)
        if nested is None:
            break
        layers.append((nested, inner if phase else outer))
    return layers


def _points_inside_fn(polygon: Float[np.ndarray, "m 2"],
                      points: Float[np.ndarray, "k 2"]) -> np.ndarray:
    """Winding number of a closed polygon about each point; true where the point is inside.

    A ray-crossing count would do for a simple polygon, but the winding sum is branch-free and
    vectorises over both the polygon and the points at these sizes.
    """
    head = np.asarray(polygon, float)
    tail = np.roll(head, -1, axis=0)
    first = head[None, :, :] - points[:, None, :]
    second = tail[None, :, :] - points[:, None, :]
    cross = first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]
    dot = (first * second).sum(axis=-1)
    return np.abs(np.arctan2(cross, dot).sum(axis=1)) > np.pi


@lru_cache(maxsize=8192)
def _core_clears_outline(width, bow, rim, shift, turn, width_delta, bow_delta,
                         margin, samples=160):
    """Whether a core of these proportions sits strictly inside its leaf, with clearance.

    Containment is invariant under a similarity and every leaf is the same shape up to one, so this
    is decided once per set of PROPORTIONS on the canonical leaf rather than per element. The
    arguments are dimensionless for exactly that reason, and the cache is what keeps the check off
    the per-element path on a sheet of a few thousand.
    """
    outer = Leaf(centre=np.zeros(2), angle=0.0, length=1.0, width=width, bow=bow)
    inner = _core_leaf(outer, rim, shift, turn, width_delta, bow_delta)
    if inner is None:
        return False
    outline = leaf_polygon_fn(outer, n=samples)
    probe = leaf_polygon_fn(inner, n=samples // 2)
    if not _points_inside_fn(outline, probe).all():
        return False
    gaps = np.linalg.norm(probe[:, None, :] - outline[None, :, :], axis=2).min(axis=1)
    return bool(gaps.min() >= margin)


def _core_leaf(leaf, rim, shift, turn, width_delta, bow_delta):
    """The nested leaf itself, before any containment question is asked."""
    scale = 1.0 - 2.0 * rim
    inner_width = (leaf.width - 2.0 * rim) / scale if scale > 0.0 else -1.0
    if scale < 0.1 or inner_width + width_delta < 0.05 or inner_width + width_delta >= 1.0:
        return None
    axis = np.array([np.cos(leaf.angle), np.sin(leaf.angle)])
    return Leaf(
        centre=np.asarray(leaf.centre, float) - shift * leaf.length * axis,
        angle=leaf.angle + turn,
        length=scale * leaf.length,
        width=inner_width + width_delta,
        bow=float(np.clip(leaf.bow + bow_delta, 0.0, 0.95)),
    )


def inner_leaf_fn(leaf: Leaf, rim: float, *, shift: float = 0.0, turn: float = 0.0,
                  width_delta: float = 0.0, bow_delta: float = 0.0,
                  margin: float = 0.004) -> Leaf | None:
    """The leaf's dark interior: a nested leaf leaving a rim of roughly constant width.

    A uniformly scaled copy will not do. The rim would then eat the same FRACTION of the length and
    of the width, and since a leaf is much longer than it is wide that leaves a fat rim at the ends
    and a thin one at the sides. Taking the rim off both dimensions in absolute terms instead gives
    an inner width of ``(width - 2*rim) / (1 - 2*rim)``, and an even border.

    An even border is not the only interesting one. Sliding the core toward the BASE widens the rim
    into a highlight at the tip and thins it at the base, which is how a cut leaf is actually drawn:
    the element then reads as a leaf rather than as its own outline.

    Args:
        rim: border width as a fraction of leaf length.
        shift: how far to slide the core toward the base, in units of leaf length.
        turn: rotation of the core off the leaf's own axis [radians].
        width_delta, bow_delta: let the core's silhouette differ from the outline's, from a slender
            spine to a core that swells at the shoulders.
        margin: clearance the core must keep from the outline, in units of leaf length.

    Returns:
        The core, or None when these proportions leave no room for one -- which is the right answer
        rather than an error: a slot as wide as the motif leaves solid paper behind.
    """
    if any((shift, turn, width_delta, bow_delta)):
        key = tuple(round(float(v), 4) for v in
                    (leaf.width, leaf.bow, rim, shift, turn, width_delta, bow_delta, margin))
        if not _core_clears_outline(*key):
            return None
    return _core_leaf(leaf, rim, shift, turn, width_delta, bow_delta)


def rim_fraction_fn(length: float, *, rim_frac: float = 0.0, rim_abs: float = 0.0,
                    max_frac: float = 0.24) -> float:
    """Border width as a fraction of THIS leaf's length, from either a relative or absolute slot.

    A rim given as a fraction scales with the element, so across the sheet's size spread the border
    varies by that same factor and the size field reads partly as a change of line weight. A blade
    cuts one slot width whatever the size of the motif; ``rim_abs`` reproduces that, and size then
    reads purely as scale. The small elements go pale and the large ones dark, which is the constant
    slot doing its work rather than a defect to be clamped away.
    """
    if rim_abs > 0.0:
        return float(min(rim_abs / max(length, 1e-12), max_frac))
    return float(rim_frac)
