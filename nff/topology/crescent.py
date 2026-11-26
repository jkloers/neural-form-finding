"""The katagami element: a crescent bounded by two circular arcs.

One shape is used everywhere in the scale field; only its position, angle and size change. That
discipline is the reference stencils' own, and it buys something concrete: because every instance
differs from every other by a SIMILARITY, a circle stays a circle, so each element can be written as
exact SVG elliptical arcs instead of a polyline approximation. The curves are then smooth at any
magnification, which is the difference between work that reads as Japanese and work that does not.

A crescent is the region between two arcs sharing their two endpoints (the cusps). The outer arc
bulges further than the inner one, so the shape is thickest at the middle and comes to a point at
each cusp. A bulge above half the chord makes the arc longer than a semicircle, which is what turns
a leaf into a hook.

Pure numpy; this module imports nothing from ``nff`` outside ``nff.topology``.
"""

from dataclasses import dataclass

import numpy as np
from jaxtyping import Float

from nff.topology.quad_warp import BezierPath


@dataclass(frozen=True)
class Crescent:
    """One element: a chord of length ``length`` at ``angle``, bowed by two arcs.

    ``bulge_outer`` and ``bulge_inner`` are sagittas as a fraction of the chord, both measured to the
    same side, so their difference is the crescent's waist thickness.
    """

    centre: Float[np.ndarray, "2"]
    angle: float
    length: float
    bulge_outer: float = 0.62
    bulge_inner: float = 0.30

    @property
    def cusps(self) -> Float[np.ndarray, "2 2"]:
        half = 0.5 * self.length * np.array([np.cos(self.angle), np.sin(self.angle)])
        return np.stack([self.centre - half, self.centre + half])


def transform_crescent_fn(crescent: Crescent, *, scale: float, offset) -> Crescent:
    """Apply a similarity. Kept at the parameter level so the arcs never have to be resampled."""
    return Crescent(
        centre=scale * np.asarray(crescent.centre, float) + np.asarray(offset, float),
        angle=crescent.angle,
        length=scale * crescent.length,
        bulge_outer=crescent.bulge_outer,
        bulge_inner=crescent.bulge_inner,
    )


def _arc_from_sagitta(start, end, sagitta):
    """Circle through ``start`` and ``end`` bowed by ``sagitta``, positive to the LEFT of the chord.

    Returns ``(radius, large_arc_flag, sweep_flag)`` in the world (y-up) frame. The whole drawing is
    emitted into one y-flipping group, so handedness stays consistent across every primitive.
    """
    start, end = np.asarray(start, float), np.asarray(end, float)
    chord = end - start
    length = float(np.linalg.norm(chord))
    if length < 1e-15 or abs(sagitta) < 1e-15:
        raise ValueError("degenerate arc: zero chord or zero sagitta")

    radius = (0.25 * length**2 + sagitta**2) / (2.0 * abs(sagitta))
    large_arc = int(abs(sagitta) > 0.5 * length)
    # Bowing to the left of the chord is a clockwise sweep about the centre, which lies to the right.
    sweep = int(sagitta < 0.0)
    return radius, large_arc, sweep


def crescent_path_fn(crescent: Crescent) -> BezierPath:
    """The element as a closed path of two exact arcs.

    Segments are ``("A", [[rx, ry], [large_arc, sweep], [x_end, y_end]])``, which the SVG writer emits
    verbatim as an ``A`` command.
    """
    cusp_a, cusp_b = crescent.cusps
    outer_sagitta = crescent.bulge_outer * crescent.length
    inner_sagitta = crescent.bulge_inner * crescent.length

    radius_out, large_out, sweep_out = _arc_from_sagitta(cusp_a, cusp_b, outer_sagitta)
    # Returning along the chord flips what "left" means, so the same absolute bow is now negative.
    radius_in, large_in, sweep_in = _arc_from_sagitta(cusp_b, cusp_a, -inner_sagitta)

    return BezierPath(
        start=cusp_a,
        segments=(
            ("A", np.array([[radius_out, radius_out], [large_out, sweep_out], cusp_b])),
            ("A", np.array([[radius_in, radius_in], [large_in, sweep_in], cusp_a])),
        ),
        closed=True,
    )


def _sample_arc(start, end, sagitta, n):
    start, end = np.asarray(start, float), np.asarray(end, float)
    chord = end - start
    length = float(np.linalg.norm(chord))
    tangent = chord / length
    normal = np.array([-tangent[1], tangent[0]])
    apex = 0.5 * (start + end) + sagitta * normal
    radius = (0.25 * length**2 + sagitta**2) / (2.0 * abs(sagitta))
    centre = apex - np.sign(sagitta) * radius * normal

    a0, apex_angle, a1 = (float(np.arctan2(*(point - centre)[::-1]))
                          for point in (start, apex, end))
    # Pick the way round that actually passes through the apex. A sign test on the chord is not
    # enough: once the sagitta exceeds half the chord the arc is the major one, and the naive answer
    # silently returns its complement -- a shallower crescent that still looks plausible.
    full = 2.0 * np.pi
    span_ccw = (a1 - a0) % full
    if (apex_angle - a0) % full < span_ccw:
        sweep = np.linspace(a0, a0 + span_ccw, n)
    else:
        sweep = np.linspace(a0, a0 - ((a0 - a1) % full), n)
    return centre + radius * np.stack([np.cos(sweep), np.sin(sweep)], axis=1)


def crescent_polygon_fn(crescent: Crescent, *, n: int = 64) -> Float[np.ndarray, "m 2"]:
    """Densely sampled outline. For the raster preview, tests and packing checks only, never output."""
    cusp_a, cusp_b = crescent.cusps
    outer = _sample_arc(cusp_a, cusp_b, crescent.bulge_outer * crescent.length, n)
    inner = _sample_arc(cusp_b, cusp_a, -crescent.bulge_inner * crescent.length, n)
    return np.concatenate([outer[:-1], inner[:-1]], axis=0)


def crescent_area_fn(crescent: Crescent) -> float:
    """Closed-form area: the difference of the two circular segments the arcs cut off."""

    def segment_area(sagitta):
        length = crescent.length
        radius = (0.25 * length**2 + sagitta**2) / (2.0 * abs(sagitta))
        half_angle = np.arcsin(np.clip(0.5 * length / radius, -1.0, 1.0))
        theta = 2.0 * (np.pi - half_angle) if abs(sagitta) > 0.5 * length else 2.0 * half_angle
        return 0.5 * radius**2 * (theta - np.sin(theta))

    return float(segment_area(crescent.bulge_outer * crescent.length)
                 - segment_area(crescent.bulge_inner * crescent.length))
