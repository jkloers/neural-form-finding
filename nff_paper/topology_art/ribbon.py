"""Chord centrelines and variable-width ribbons authored in the unit cell.

A stroke crossing an irregular tiling can only look continuous if neighbouring tiles agree, at the
corner they share, on three things: where the stroke is, which way it points, and how wide it is.
None of those may therefore be a property of a tile. They are properties of the CORNER, computed once
in world coordinates and handed to both tiles (``nff_paper.closed_support.katagami.build_corner_frames_fn``).

This module is the cell-space half of that contract. Authoring stays in the unit cell so the exact
segment-to-quadratic-Bezier property survives and containment stays free, while the world-space
junction conditions are pulled back through the analytic Jacobian:

    required cell tangent  =  J^-1 @ world_direction
    required cell offset   =  the exact solve of  J @ x + x_u*x_v*d = half_width * perp(direction)

The offset needs the exact solve, not the linear ``J^-1`` estimate. The bilinear map's second-order
term cancels between the two sides of one ribbon, so the linear version still gives an exact
tip-to-tip width -- but it does not cancel between two DIFFERENT tiles, and a junction built on it
misses by a few percent of the stroke width. ``solve_offset_fn`` removes that.

Because the offset at a port is then computed from quantities the two tiles share, both tiles'
ribbon outlines land on exactly the same two points. That is what makes the junction fluid at
NONZERO width, rather than having to taper to nothing.

Pure numpy; this module imports nothing from ``nff`` outside ``nff.topology``.
"""

import numpy as np
from jaxtyping import Float

from nff_paper.topology_art.quad_warp import jacobian_fn, pull_back_fn, solve_offset_fn


def _unit(vector, eps=1e-15):
    norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    return vector / np.maximum(norm, eps)


def _perp(vector):
    return np.stack([-vector[..., 1], vector[..., 0]], axis=-1)


def cell_tangent_fn(
    quad: Float[np.ndarray, "4 2"],
    corner_uv: Float[np.ndarray, "2"],
    world_direction: Float[np.ndarray, "2"],
    *,
    inward: bool = True,
) -> Float[np.ndarray, "2"]:
    """Unit cell-space tangent whose image points along ``world_direction``.

    ``inward`` flips the sign so the tangent leads into the cell, which is what a control handle
    needs; the world direction itself is shared between the two tiles and carries no such preference.
    """
    tangent = _unit(pull_back_fn(quad, corner_uv, np.asarray(world_direction, float)))
    if inward and float(np.dot(tangent, np.array([0.5, 0.5]) - np.asarray(corner_uv, float))) < 0.0:
        tangent = -tangent
    return tangent


def _clamped_handle(origin, direction, length, *, margin=0.92):
    """Longest handle along ``direction`` that keeps the control point inside the unit cell.

    Shortening the handle rather than clipping its coordinates is what preserves the tangent
    DIRECTION, and the direction is the half of the port contract the neighbouring tile also relies
    on. Clipping componentwise would quietly rotate the stroke away from its shared tangent.
    """
    limit = length
    for axis in (0, 1):
        step = float(direction[axis])
        if step > 1e-12:
            limit = min(limit, (1.0 - float(origin[axis])) / step)
        elif step < -1e-12:
            limit = min(limit, float(origin[axis]) / -step)
    return max(0.0, margin * limit)


def build_chord_fn(
    quad: Float[np.ndarray, "4 2"],
    corner_a: Float[np.ndarray, "2"],
    corner_b: Float[np.ndarray, "2"],
    direction_a: Float[np.ndarray, "2"],
    direction_b: Float[np.ndarray, "2"],
    *,
    handle: float = 0.42,
    n_samples: int = 24,
) -> Float[np.ndarray, "n 2"]:
    """Cell-space centreline joining two cell corners with prescribed world tangents.

    A cubic in the cell, sampled. Its control points are clamped into the unit cell, so the convex
    hull property keeps the whole centreline inside and the warped stroke inside its own tile.

    Args:
        quad: (4, 2) tile corners BL, BR, TR, TL.
        corner_a, corner_b: (2,) cell coordinates of the two ports.
        direction_a, direction_b: (2,) shared world directions at those ports.
        handle: control-handle length as a fraction of the port separation.
        n_samples: centreline samples.

    Returns:
        (n_samples + 1, 2) cell-space polyline, endpoints exactly at the two corners.
    """
    corner_a = np.asarray(corner_a, float)
    corner_b = np.asarray(corner_b, float)
    span = float(np.linalg.norm(corner_b - corner_a))
    tangent_a = cell_tangent_fn(quad, corner_a, direction_a)
    tangent_b = cell_tangent_fn(quad, corner_b, direction_b)
    ctrl_a = corner_a + _clamped_handle(corner_a, tangent_a, handle * span) * tangent_a
    ctrl_b = corner_b + _clamped_handle(corner_b, tangent_b, handle * span) * tangent_b

    t = np.linspace(0.0, 1.0, n_samples + 1)[:, None]
    return ((1 - t) ** 3 * corner_a + 3 * (1 - t) ** 2 * t * ctrl_a
            + 3 * (1 - t) * t**2 * ctrl_b + t**3 * corner_b)


def width_profile_fn(
    n: int, half_width_a: float, half_width_b: float, *, swell: float = 0.0
) -> Float[np.ndarray, "n"]:
    """World half-width along a chord: ends pinned, a swell in between.

    The swell uses ``sin(pi*s)``, which vanishes at both ends, so the fluctuation can be as strong as
    wanted without ever disturbing a junction.
    """
    s = np.linspace(0.0, 1.0, n)
    return (half_width_a + s * (half_width_b - half_width_a)) * (1.0 + swell * np.sin(np.pi * s))


def build_ribbon_fn(
    quad: Float[np.ndarray, "4 2"],
    centreline_uv: Float[np.ndarray, "n 2"],
    half_widths: Float[np.ndarray, "n"],
    *,
    closed: bool = False,
    end_directions: tuple | None = None,
) -> Float[np.ndarray, "m 2"]:
    """Closed cell-space outline of a ribbon of prescribed WORLD half-width.

    At every sample the offset direction is the world normal, pulled back into the cell, so the band
    has the requested width on the page rather than in the cell. The tile's own stretch therefore
    stops leaking into the line weight, which is what would otherwise make the tiling legible.

    The offset is exact, not approximate: with ``m`` the pulled-back unit normal,
    ``P(p + h*m) - P(p - h*m) = 2h * J * m`` identically, because the bilinear map's second-order
    term cancels between the two sides. So the band's tip-to-tip width is the requested one to
    machine precision on any quad.

    Args:
        quad: (4, 2) tile corners.
        centreline_uv: (n, 2) cell-space centreline.
        half_widths: (n,) world half-widths.
        closed: treat the centreline as a loop.
        end_directions: optional ``(d_first, d_last)`` world directions replacing the finite
            difference at the two ends. Supplying the SHARED port directions is what makes the two
            tiles' outlines land on the same two world points, so the junction closes exactly.

    Returns:
        (m, 2) closed cell-space outline, ready for ``warp_polyline_fn(..., closed=True)``.
    """
    centre = np.atleast_2d(np.asarray(centreline_uv, float))
    widths = np.asarray(half_widths, float).reshape(-1)
    if len(centre) != len(widths):
        raise ValueError(f"centreline has {len(centre)} points but {len(widths)} widths")

    tangents_uv = np.gradient(centre, axis=0) if not closed else (
        np.roll(centre, -1, axis=0) - np.roll(centre, 1, axis=0))
    if end_directions is not None and not closed:
        for index, direction in ((0, end_directions[0]), (-1, end_directions[1])):
            exact = pull_back_fn(quad, centre[index], np.asarray(direction, float))
            # The shared direction is an unoriented line; align it with the walk so the outline
            # does not fold back on itself at the tip.
            tangents_uv[index] = exact * np.sign(np.dot(exact, tangents_uv[index]) or 1.0)

    left = np.empty_like(centre)
    right = np.empty_like(centre)
    for i, (point, tangent_uv, half) in enumerate(zip(centre, tangents_uv, widths)):
        world_tangent = _unit(jacobian_fn(quad, point) @ tangent_uv)
        world_offset = half * _perp(world_tangent)
        left[i] = point + solve_offset_fn(quad, point, world_offset)
        right[i] = point + solve_offset_fn(quad, point, -world_offset)

    if closed:
        return np.concatenate([left, right[::-1]], axis=0)
    return np.concatenate([left, right[::-1]], axis=0)


def ribbon_port_points_fn(
    half_width: float,
    position: Float[np.ndarray, "2"],
    direction: Float[np.ndarray, "2"],
) -> Float[np.ndarray, "2 2"]:
    """The two world points a ribbon must hit at a port: ``p +/- w * perp(d)``.

    Computed from corner-owned quantities only, which is precisely why both tiles produce the same
    pair and the seam closes exactly.
    """
    normal = _perp(_unit(np.asarray(direction, float)))
    position = np.asarray(position, float)
    return np.stack([position + half_width * normal, position - half_width * normal])
