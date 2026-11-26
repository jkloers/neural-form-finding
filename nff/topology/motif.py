"""Katagami motifs authored in the unit cell, under the corner-port rule.

On a closed-builder sheet the tiles abut in T-junctions: an interior corner is shared by exactly two
tiles, but a tile edge is matched by a neighbour only rarely (7.7% of edge slots on the 15x10 figmat
sheet). So there is no consistent parameterisation along a shared edge, and a motif stroke that ends
at a mid-edge point has nothing to meet on the other side.

Hence the rule every motif here obeys: **a stroke may touch the cell boundary only at a corner.**
Corners are shared exactly, so corner-routed strokes join their neighbour to machine precision, and
the tiling stops being legible in the artwork. ``check_corner_ports_fn`` enforces it.

Curves are polylined here, at author time, with the deviation recorded in ``chord_tol_uv``. Under a
bilinear warp a circle is not a conic, so no exact curve representation survives downstream; by
discretising once at the source, every later step is exact with respect to the authored motif and
the single approximation is one stated number.

Pure numpy; this module imports nothing from ``nff``.
"""

from dataclasses import dataclass, field

import numpy as np
from jaxtyping import Float

CORNERS = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
CENTRE = np.array([0.5, 0.5])
EDGE_MIDPOINTS = np.array([[0.5, 0.0], [1.0, 0.5], [0.5, 1.0], [0.0, 0.5]])


@dataclass(frozen=True)
class Stroke:
    """A polyline in the unit cell. ``closed`` paths must not repeat their first point."""

    points: Float[np.ndarray, "n 2"]
    closed: bool = False


@dataclass(frozen=True)
class Motif:
    """A cell figure: polylines in ``[0, 1]^2`` plus the provenance a reader needs to trust it."""

    name: str
    strokes: tuple[Stroke, ...]
    detail: int = 2
    chord_tol_uv: float = 0.0
    params: dict = field(default_factory=dict)


# ── curve sampling ───────────────────────────────────────────────────────────────────────────


def _arc_segments(radius: float, sweep: float, tol: float) -> int:
    """Segments needed so a circular arc's chord deviation stays under ``tol`` [cell units].

    An ``n``-segment arc of radius ``r`` over total angle ``sweep`` deviates by
    ``r*(1 - cos(sweep/(2n)))``; invert that rather than guessing a segment count.
    """
    if radius <= tol or sweep <= 0.0:
        return 1
    half = np.arccos(np.clip(1.0 - tol / radius, -1.0, 1.0))
    return max(1, int(np.ceil(abs(sweep) / (2.0 * half))))


def _polar(centre, radius, angle):
    return np.asarray(centre) + np.stack([radius * np.cos(angle), radius * np.sin(angle)], axis=-1)


def _arc(centre, radius, a0, a1, tol):
    """Circular arc as a polyline, both endpoints included."""
    n = _arc_segments(radius, abs(a1 - a0), tol)
    return _polar(centre, radius, np.linspace(a0, a1, n + 1))


def _quad_poly(p0, control, p1, tol):
    """Quadratic Bezier as a polyline, sampled to a chord deviation under ``tol``.

    A quadratic lies inside the convex hull of its three control points, so keeping the control
    point strictly inside the cell is enough to keep the whole stroke inside -- no clipping, and no
    envelope that has to follow the cell outline and would drag its kinks into the artwork.
    """
    p0, control, p1 = (np.asarray(v, float) for v in (p0, control, p1))
    sag = float(np.linalg.norm(2.0 * control - p0 - p1) / 4.0)
    n = max(2, int(np.ceil(np.sqrt(max(sag, 0.0) / max(tol, 1e-12)))))
    t = np.linspace(0.0, 1.0, n + 1)[:, None]
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * control + t**2 * p1


def _bowed_edge(p0, p1, through, tol):
    """Corner-to-corner stroke bowed inward so that it passes through ``through`` at its midpoint.

    Staying clear of the edge is not decoration: two neighbouring tiles split their common boundary
    differently, so any feature that hugs an edge draws that mismatch, and with it the tiling.
    """
    p0, p1, through = (np.asarray(v, float) for v in (p0, p1, through))
    return _quad_poly(p0, 2.0 * through - 0.5 * (p0 + p1), p1, tol)


# ── motif factories ──────────────────────────────────────────────────────────────────────────


def build_grid_test(**_) -> Motif:
    """Cell border plus both diagonals. For tests only: the border must emit as straight lines."""
    return Motif(
        name="grid_test",
        strokes=(
            Stroke(CORNERS, closed=True),
            Stroke(CORNERS[[0, 2]]),
            Stroke(CORNERS[[1, 3]]),
        ),
    )


def build_asanoha(*, detail: int = 2, star: float = 0.62, **_) -> Motif:
    """Hemp leaf: the two diagonals, a central diamond, and a leaf tent over each edge.

    The diagonals are what carry the pattern across tiles -- they run corner to corner, so every
    stroke that reaches the boundary reaches it at a shared point. The diamond is built from the
    edge-midpoint directions rather than the corner directions on purpose: a figure built on the
    corner directions is a scaled copy of the cell and would draw the tiling back into the artwork.
    """
    diamond = CENTRE + star * (EDGE_MIDPOINTS - CENTRE)
    strokes = [Stroke(CORNERS[[0, 2]]), Stroke(CORNERS[[1, 3]])]
    if detail >= 1:
        strokes.append(Stroke(diamond, closed=True))
    if detail >= 2:
        for i in range(4):
            strokes.append(Stroke(np.array([CORNERS[i], diamond[i], CORNERS[(i + 1) % 4]])))
    if detail >= 3:
        strokes.append(Stroke(CENTRE + 0.42 * (diamond - CENTRE), closed=True))
    return Motif(
        name="asanoha", strokes=tuple(strokes), detail=detail, params={"star": star}
    )


def build_kikko(*, detail: int = 2, radius: float = 0.36, tol: float = 2e-3, **_) -> Motif:
    """Tortoiseshell: the diagonals plus a strictly interior octagon, nested by detail level."""
    angles = np.arange(8) * (np.pi / 4.0) + np.pi / 8.0
    strokes = [Stroke(CORNERS[[0, 2]]), Stroke(CORNERS[[1, 3]])]
    for level in range(max(1, detail)):
        strokes.append(Stroke(_polar(CENTRE, radius * (1.0 - 0.3 * level), angles), closed=True))
    return Motif(
        name="kikko",
        strokes=tuple(strokes),
        detail=detail,
        chord_tol_uv=0.0,
        params={"radius": radius},
    )


def build_seigaiha(*, detail: int = 2, tol: float = 1.5e-3, **_) -> Motif:
    """Wave scales: nested arcs about the cell centre, opening toward the corners.

    Kept strictly interior and paired with the diagonals, so the wave reads as a texture inside the
    net rather than as something that has to line up across a seam it cannot line up across.
    """
    strokes = [Stroke(CORNERS[[0, 2]]), Stroke(CORNERS[[1, 3]])]
    n_rings = 2 + max(0, detail - 1)
    for k in range(1, n_rings + 1):
        radius = 0.40 * k / n_rings
        strokes.append(Stroke(_arc(CENTRE, radius, 0.0, 2.0 * np.pi, tol)[:-1], closed=True))
    return Motif(
        name="seigaiha", strokes=tuple(strokes), detail=detail, chord_tol_uv=tol,
        params={"n_rings": n_rings},
    )


def build_karakusa(*, detail: int = 2, turn: float = 1.15, tol: float = 1.5e-3,
                   leaf: float = 0.13, **_) -> Motif:
    """Vine scroll: a tendril spiralling inward from each corner, with leaves and a centre bud.

    Each corner emits exactly one tendril, so the vine crosses every shared corner and grows
    continuously across the sheet.
    """
    strokes = []
    radial = CORNERS - CENTRE
    for i in range(4):
        a0 = float(np.arctan2(radial[i, 1], radial[i, 0]))
        control = _polar(CENTRE, 0.40, a0 + turn * 0.5)
        end = _polar(CENTRE, 0.15, a0 + turn)
        tendril = _quad_poly(CORNERS[i], control, end, tol)
        strokes.append(Stroke(tendril))
        if detail >= 2:
            anchor = tendril[max(1, int(0.62 * (len(tendril) - 1)))]
            outward = anchor - CENTRE
            outward = outward / max(np.linalg.norm(outward), 1e-12)
            side = np.array([-outward[1], outward[0]])
            hub = anchor + 0.55 * leaf * side - 0.15 * leaf * outward
            strokes.append(Stroke(_arc(hub, leaf * 0.5, 0.0, 2.0 * np.pi, tol)[:-1], closed=True))
    if detail >= 1:
        strokes.append(Stroke(_arc(CENTRE, 0.085, 0.0, 2.0 * np.pi, tol)[:-1], closed=True))
    return Motif(
        name="karakusa", strokes=tuple(strokes), detail=detail, chord_tol_uv=tol,
        params={"turn": turn, "leaf": leaf},
    )


def build_rosette(*, detail: int = 2, petals: int = 8, radius: float = 0.34,
                  tol: float = 1.2e-3, **_) -> Motif:
    """Four-fold blossom, strictly interior. Also the default corner node.

    Each petal is a pair of circular arcs meeting at the centre and at the petal tip, which is the
    cheapest closed shape that still reads as a flower rather than a polygon.
    """
    strokes = []
    for k in range(petals):
        axis = 2.0 * np.pi * k / petals
        tip = _polar(CENTRE, radius, axis)
        hub = 0.5 * (CENTRE + tip)
        half = 0.5 * float(np.linalg.norm(tip - CENTRE))
        for sign in (+1.0, -1.0):
            centre = hub + sign * 0.34 * radius * np.array([-np.sin(axis), np.cos(axis)])
            arc_r = float(np.hypot(half, 0.34 * radius))
            start = np.arctan2(*(CENTRE - centre)[::-1])
            end = np.arctan2(*(tip - centre)[::-1])
            if sign > 0:
                start, end = end, start
            strokes.append(Stroke(_arc(centre, arc_r, start, end, tol)))
    if detail >= 2:
        strokes.append(Stroke(_arc(CENTRE, 0.30 * radius, 0.0, 2.0 * np.pi, tol)[:-1], closed=True))
    return Motif(
        name="rosette", strokes=tuple(strokes), detail=detail, chord_tol_uv=tol,
        params={"petals": petals, "radius": radius},
    )


def build_shippo(*, detail: int = 2, bulge: float = 0.66, tol: float = 1.2e-3,
                 inner: float = 0.55, **_) -> Motif:
    """Seven treasures: a cushion through all four corners, bowed inward from the edges.

    The cushion is the answer to the seam problem in one stroke. It passes through every corner, so
    neighbouring cushions touch exactly and the sheet reads as one interlocking net; and it bows
    well clear of the edges, so the places where two tiles disagree about where their common
    boundary is are simply not drawn on.
    """
    ring = []
    for i in range(4):
        through = CENTRE + bulge * (EDGE_MIDPOINTS[i] - CENTRE)
        ring.append(_bowed_edge(CORNERS[i], CORNERS[(i + 1) % 4], through, tol)[:-1])
    strokes = [Stroke(np.concatenate(ring, axis=0), closed=True)]
    if detail >= 2:
        strokes.append(Stroke(CENTRE + inner * (np.concatenate(ring, axis=0) - CENTRE), closed=True))
    if detail >= 3:
        strokes.append(Stroke(_arc(CENTRE, 0.11, 0.0, 2.0 * np.pi, tol)[:-1], closed=True))
    return Motif(
        name="shippo", strokes=tuple(strokes), detail=detail, chord_tol_uv=tol,
        params={"bulge": bulge, "inner": inner},
    )


MOTIFS = {
    "shippo": build_shippo,
    "asanoha": build_asanoha,
    "kikko": build_kikko,
    "seigaiha": build_seigaiha,
    "karakusa": build_karakusa,
    "rosette": build_rosette,
    "grid_test": build_grid_test,
}


def build_motif_fn(name: str, *, detail: int = 2, **params) -> Motif:
    """Build a motif from the registry.

    Args:
        name: registry key; ``MOTIFS`` lists them.
        detail: level of detail, 1 (sparse) to 3 (dense). Corner ports are identical at every
            level, so detail may vary tile to tile without breaking continuity.
        **params: motif-specific overrides.

    Returns:
        A ``Motif`` in the unit cell.
    """
    if name not in MOTIFS:
        raise ValueError(f"unknown motif {name!r}; available: {sorted(MOTIFS)}")
    return MOTIFS[name](detail=detail, **params)


def check_corner_ports_fn(motif: Motif, *, tol: float = 1e-9) -> dict:
    """Verify the corner-port rule and that the motif stays inside the cell.

    Args:
        motif: the motif to check.
        tol: distance at which a point counts as touching the boundary or as being a corner.

    Returns:
        Dict with ``ok``, ``ports`` (corner indices actually used), ``mid_edge`` (offending points)
        and ``outside`` (points beyond the cell).
    """
    mid_edge, outside, ports = [], [], set()
    for stroke in motif.strokes:
        pts = np.atleast_2d(np.asarray(stroke.points, float))
        if (pts < -tol).any() or (pts > 1.0 + tol).any():
            outside.extend(pts[(pts < -tol).any(axis=1) | (pts > 1.0 + tol).any(axis=1)].tolist())
        on_boundary = np.minimum(pts, 1.0 - pts).min(axis=1) <= tol
        for point in pts[on_boundary]:
            distances = np.linalg.norm(CORNERS - point, axis=1)
            if distances.min() <= tol:
                ports.add(int(np.argmin(distances)))
            else:
                mid_edge.append(point.tolist())
    return {
        "ok": not mid_edge and not outside,
        "ports": tuple(sorted(ports)),
        "mid_edge": mid_edge,
        "outside": outside,
    }
