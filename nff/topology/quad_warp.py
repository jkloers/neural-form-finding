"""Exact bilinear warp of unit-cell geometry onto a convex quadrilateral.

A straight segment in the unit cell maps to an EXACT quadratic Bezier under the bilinear
square-to-quad map, so motif geometry can be carried onto an irregular quad tiling with no sampling
and no flattening anywhere in the pipeline.

Writing the bilinear form as

    P(u, v) = Q0 + u*a + v*b + u*v*d,
    a = Q1 - Q0,  b = Q3 - Q0,  d = Q0 - Q1 + Q2 - Q3

and substituting an affine ``u(t), v(t)`` leaves ``u*v`` as the only nonlinear term, so ``P(t)`` is a
quadratic polynomial. That is the whole reason this module exists: degree 2 in, degree 2 out.

``d`` is the quad's deviation from a parallelogram. It is also the entire source of curvature: the
image of a segment is straight exactly when ``du*dv*d`` vanishes, which is why unit-cell edges (one
of ``du``, ``dv`` zero) always come out straight -- the property motif continuity relies on.

Pure numpy; this module imports nothing from ``nff``.
"""

from dataclasses import dataclass

import numpy as np
from jaxtyping import Float


@dataclass(frozen=True)
class BezierPath:
    """An exact path of line and quadratic-Bezier segments in world units.

    The single currency between the warp, the SVG writer and any consumer. ``segments`` holds
    ``("L", (1, 2))`` or ``("Q", (2, 2))`` entries, each giving the control points AFTER the current
    point; ``("Q", [c, p])`` is control point then endpoint.
    """

    start: Float[np.ndarray, "2"]
    segments: tuple[tuple[str, Float[np.ndarray, "k 2"]], ...]
    closed: bool = False

    @property
    def endpoint(self) -> Float[np.ndarray, "2"]:
        """Final on-curve point (the start point again when the path is closed)."""
        if self.closed or not self.segments:
            return self.start
        return self.segments[-1][1][-1]


@dataclass(frozen=True)
class CompoundPath:
    """Several closed subpaths emitted as ONE fill region.

    Needed because ``fill-rule: evenodd`` cannot punch a hole across separate SVG ``<path>``
    elements: an outline and the hole inside it have to share a single ``d`` string.
    """

    subpaths: tuple[BezierPath, ...]

    @property
    def closed(self) -> bool:
        return True


def bilinear_fn(
    quad: Float[np.ndarray, "4 2"],
    uv: Float[np.ndarray, "n 2"],
) -> Float[np.ndarray, "n 2"]:
    """Map unit-cell coordinates onto a quad ordered BL, BR, TR, TL.

    The same map used by ``nff.closed.shield_design._tile_sample_points`` and
    ``nff.closed.shield_render.tile_colours``.

    Args:
        quad: (4, 2) corners in CCW order BL, BR, TR, TL.
        uv: (n, 2) coordinates in the unit cell.

    Returns:
        (n, 2) world positions.
    """
    quad = np.asarray(quad, float)
    uv = np.atleast_2d(np.asarray(uv, float))
    u = uv[:, 0:1]
    v = uv[:, 1:2]
    q0, q1, q2, q3 = quad
    return q0 + u * (q1 - q0) + v * (q3 - q0) + (u * v) * (q0 - q1 + q2 - q3)


def jacobian_fn(
    quad: Float[np.ndarray, "4 2"],
    uv: Float[np.ndarray, "2"],
) -> Float[np.ndarray, "2 2"]:
    """Jacobian ``[dP/du, dP/dv]`` of the bilinear map at a unit-cell point.

    Analytic, and the bridge between the two coordinate systems the artwork lives in: motifs are
    authored in the cell, but junction conditions (a shared direction, a shared width) are world
    quantities, so they have to be pulled back through this.
    """
    quad = np.asarray(quad, float)
    u, v = float(uv[0]), float(uv[1])
    q0, q1, q2, q3 = quad
    d = q0 - q1 + q2 - q3
    return np.stack([(q1 - q0) + v * d, (q3 - q0) + u * d], axis=1)


def pull_back_fn(
    quad: Float[np.ndarray, "4 2"],
    uv: Float[np.ndarray, "2"],
    world_vector: Float[np.ndarray, "2"],
    *,
    eps: float = 1e-12,
) -> Float[np.ndarray, "2"]:
    """Cell-space vector whose image under the bilinear map is ``world_vector``.

    Solves ``J @ x = world_vector``. A collapsed tile makes ``J`` near-singular; rather than emit a
    wild control point, fall back to the least-squares solution, which degrades gracefully.
    """
    jac = jacobian_fn(quad, uv)
    if abs(float(np.linalg.det(jac))) < eps * float(np.abs(jac).max() ** 2 + eps):
        return np.linalg.lstsq(jac, np.asarray(world_vector, float), rcond=None)[0]
    return np.linalg.solve(jac, np.asarray(world_vector, float))


def solve_offset_fn(
    quad: Float[np.ndarray, "4 2"],
    uv: Float[np.ndarray, "2"],
    world_offset: Float[np.ndarray, "2"],
    *,
    iterations: int = 3,
) -> Float[np.ndarray, "2"]:
    """Cell-space step from ``uv`` whose image is displaced by exactly ``world_offset``.

    The linear pull-back ``J^-1 @ world_offset`` is only first-order correct: the bilinear map also
    contributes ``x_u * x_v * d``. That term cancels between the two sides of a ribbon, so the band's
    tip-to-tip width is exact either way, but it does NOT cancel between two different tiles -- so a
    junction built on the linear solve misses by a fraction of the stroke width. Newton removes it.

    Solves ``J @ x + x_u*x_v*d = world_offset`` from the linear estimate; converges in two steps at
    the offsets a stroke uses.
    """
    quad = np.asarray(quad, float)
    target = np.asarray(world_offset, float)
    q0, q1, q2, q3 = quad
    d = q0 - q1 + q2 - q3
    jac = jacobian_fn(quad, uv)

    x = pull_back_fn(quad, uv, target)
    for _ in range(iterations):
        residual = jac @ x + (x[0] * x[1]) * d - target
        derivative = jac + np.stack([x[1] * d, x[0] * d], axis=1)
        if abs(float(np.linalg.det(derivative))) < 1e-14:
            break
        x = x - np.linalg.solve(derivative, residual)
    return x


def _quadratic_coefficients(quad, p0_uv, p1_uv):
    """Monomial coefficients ``(A, B, C)`` of the segment image ``P(t) = A + B t + C t^2``."""
    quad = np.asarray(quad, float)
    u0, v0 = (float(x) for x in p0_uv)
    u1, v1 = (float(x) for x in p1_uv)
    du, dv = u1 - u0, v1 - v0
    q0, q1, q2, q3 = quad
    a = q1 - q0
    b = q3 - q0
    d = q0 - q1 + q2 - q3
    coef_a = q0 + u0 * a + v0 * b + (u0 * v0) * d
    coef_b = du * a + dv * b + (u0 * dv + v0 * du) * d
    coef_c = (du * dv) * d
    return coef_a, coef_b, coef_c


def segment_bezier_fn(
    quad: Float[np.ndarray, "4 2"],
    p0_uv: Float[np.ndarray, "2"],
    p1_uv: Float[np.ndarray, "2"],
) -> tuple[Float[np.ndarray, "2"], Float[np.ndarray, "2"], Float[np.ndarray, "2"]]:
    """Exact quadratic-Bezier control points of a straight unit-cell segment's image.

    Args:
        quad: (4, 2) corners BL, BR, TR, TL.
        p0_uv: (2,) segment start in the unit cell.
        p1_uv: (2,) segment end in the unit cell.

    Returns:
        ``(b0, b1, b2)``, each (2,). Exact, not a fit.
    """
    coef_a, coef_b, coef_c = _quadratic_coefficients(quad, p0_uv, p1_uv)
    return coef_a, coef_a + 0.5 * coef_b, coef_a + coef_b + coef_c


def sagitta_fn(
    b0: Float[np.ndarray, "2"],
    b1: Float[np.ndarray, "2"],
    b2: Float[np.ndarray, "2"],
) -> float:
    """Flatness of a quadratic Bezier: ``max_t |curve(t) - chord(t)| = |2*b1 - b0 - b2| / 4``.

    Exact for the PARAMETRIC deviation, since ``curve(t) - chord(t) = t*(1-t)*(2*b1 - b0 - b2)``
    and ``t*(1-t)`` peaks at ``1/4``. The perpendicular distance from the curve to the chord is at
    most this, so the value is a conservative straightness test -- which is the direction that
    matters when deciding whether a segment may be emitted as a line.
    """
    return float(np.linalg.norm(2.0 * np.asarray(b1) - np.asarray(b0) - np.asarray(b2)) / 4.0)


def warp_polyline_fn(
    quad: Float[np.ndarray, "4 2"],
    uv_points: Float[np.ndarray, "n 2"],
    *,
    closed: bool = False,
    straight_tol: float = 1e-12,
) -> BezierPath:
    """Warp a unit-cell polyline onto a quad as an exact line/Bezier path.

    A segment whose sagitta falls below ``straight_tol`` is emitted as a line. That is not an
    optimisation: segments along the cell boundary have ``du*dv = 0``, so their images are exactly
    straight, and emitting them as lines keeps shared tile edges free of spurious curvature.

    Args:
        quad: (4, 2) corners BL, BR, TR, TL.
        uv_points: (n, 2) polyline vertices in the unit cell; for a closed path do not repeat the
            first vertex.
        closed: close the path back onto its first vertex.
        straight_tol: sagitta below which a segment is emitted as a line [world units].

    Returns:
        A ``BezierPath`` in world units.
    """
    uv = np.atleast_2d(np.asarray(uv_points, float))
    if len(uv) < 2:
        raise ValueError(f"a polyline needs at least 2 points, got {len(uv)}")
    order = list(range(len(uv))) + ([0] if closed else [])

    start = bilinear_fn(quad, uv[order[0]])[0]
    segments = []
    for i0, i1 in zip(order[:-1], order[1:]):
        b0, b1, b2 = segment_bezier_fn(quad, uv[i0], uv[i1])
        if sagitta_fn(b0, b1, b2) < straight_tol:
            segments.append(("L", np.array([b2])))
        else:
            segments.append(("Q", np.array([b1, b2])))
    return BezierPath(start=start, segments=tuple(segments), closed=closed)


def transform_path_fn(
    path: BezierPath,
    matrix: Float[np.ndarray, "2 2"],
    offset: Float[np.ndarray, "2"],
) -> BezierPath:
    """Apply an affine map to a path.

    Exact for any affine map, because the affine image of a Bezier's control points is precisely
    the control polygon of the image curve. Used for page fitting, which is a similarity.
    """
    matrix = np.asarray(matrix, float)
    offset = np.asarray(offset, float)

    if isinstance(path, CompoundPath):
        return CompoundPath(tuple(transform_path_fn(p, matrix, offset) for p in path.subpaths))

    def apply(points):
        return np.asarray(points, float) @ matrix.T + offset

    return BezierPath(
        start=apply(path.start[None])[0],
        segments=tuple((kind, apply(pts)) for kind, pts in path.segments),
        closed=path.closed,
    )


def path_points_fn(path: BezierPath, *, per_segment: int = 24) -> Float[np.ndarray, "m 2"]:
    """Sample a path densely. For tests, previews and containment checks only, never for output."""
    if isinstance(path, CompoundPath):
        return np.concatenate([path_points_fn(p, per_segment=per_segment) for p in path.subpaths])
    t = np.linspace(0.0, 1.0, per_segment + 1)[1:, None]
    out = [path.start[None]]
    current = path.start
    for kind, pts in path.segments:
        if kind == "L":
            out.append(current + t * (pts[0] - current))
            current = pts[0]
        else:
            ctrl, end = pts
            out.append((1 - t) ** 2 * current + 2 * (1 - t) * t * ctrl + t**2 * end)
            current = end
    return np.concatenate(out, axis=0)


def assert_ccw_convex_fn(tiles: Float[np.ndarray, "n 4 2"], *, name: str = "tiles") -> None:
    """Raise unless every quad is convex and counter-clockwise.

    The bilinear map only stays inside its own cell for convex quads, so this guards the containment
    guarantee the whole composition relies on.
    """
    tiles = np.asarray(tiles, float)
    if tiles.ndim != 3 or tiles.shape[1:] != (4, 2):
        raise ValueError(f"{name} must have shape (n, 4, 2), got {tiles.shape}")
    edge = np.roll(tiles, -1, axis=1) - tiles
    nxt = np.roll(edge, -1, axis=1)
    cross = edge[:, :, 0] * nxt[:, :, 1] - edge[:, :, 1] * nxt[:, :, 0]
    bad = np.flatnonzero(~(cross > 0).all(axis=1))
    if bad.size:
        raise ValueError(
            f"{name}: {bad.size} of {len(tiles)} quads are not convex CCW (first: {bad[:5].tolist()})"
        )
