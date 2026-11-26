"""Continuous komon nets, authored globally and warped by a deformed control mesh.

Every motif in this repo so far lives inside one tile. It has to: the panel tiles partition the
sheet exactly but meet in T-junctions, so a stroke crossing a tile edge has no partner to meet on the
other side, and the corner-port rule exists to keep strokes away from edges entirely.

A net cannot be built that way. It is one pattern spanning the whole sheet, so it must cross cell
boundaries constantly. What makes it possible is choosing a **conforming** control mesh -- one whose
neighbouring cells share a whole edge, and agree on it. The grid of tile centroids is exactly that:
cell ``(i, j)`` spans ``c[i,j], c[i+1,j], c[i+1,j+1], c[i,j+1]``, so the edge between two cells is
the same segment computed from the same two points on both sides. Under the bilinear map that edge
is a linear interpolation of those two points from either cell, so the two warps agree there
identically and mid-edge crossings become legal.

The net is therefore authored ONCE in reference coordinates over the whole sheet, then cut at the
cell boundaries and warped piece by piece. No motif needs to know where the cells are, and no motif
needs ports designed for it.

Pure numpy; this module imports nothing from ``nff`` outside ``nff.topology``.
"""

import numpy as np
from jaxtyping import Float

from nff.topology.quad_warp import BezierPath, warp_polyline_fn

# A hexagonal lattice laid on a rectangular reference grid: one hexagon repeat spans 1 unit across
# and this much along. Kept exact rather than 0.866 so the lattice closes on itself.
ROOT3 = float(np.sqrt(3.0))
ROOT2 = float(np.sqrt(2.0))


def _dedupe(segments, places=7):
    """Drop segments already drawn, so a shared lattice edge is one strut and not two.

    A hexagon net written hexagon by hexagon draws every interior edge twice. Harmless on screen,
    but it doubles the piece count and, once the net is given a width and cut, the doubled edge
    becomes a doubled outline.
    """
    seen, out = set(), []
    for start, end in segments:
        key = tuple(sorted((tuple(np.round(start, places)), tuple(np.round(end, places)))))
        if key in seen:
            continue
        seen.add(key)
        out.append(np.array([start, end], float))
    return out


def _hex_lattice(cols, rows, radius):
    """Pointy-top hexagon centres in axial coordinates, covering the domain with a margin.

    The lattice is generated past the sheet on every side. A net that stops at the boundary leaves a
    ragged fringe of half-motifs; carried over and clipped, it leaves a cut edge.
    """
    step_x, step_y = ROOT3 * radius, 1.5 * radius
    down = int(np.floor(-2.0 * radius / step_y)) - 1
    up = int(np.ceil((rows + 2.0 * radius) / step_y)) + 1
    centres = []
    for r in range(down, up + 1):
        # Axial layout: each row shifts by half a step, which is what makes the rows interlock.
        left = int(np.floor((-2.0 * radius - 0.5 * step_x * r) / step_x)) - 1
        right = int(np.ceil((cols + 2.0 * radius - 0.5 * step_x * r) / step_x)) + 1
        for q in range(left, right + 1):
            centres.append((step_x * (q + 0.5 * r), step_y * r))
    return np.array(centres, float)


def _hexagon(centre, radius):
    """The six corners of one pointy-top hexagon, counter-clockwise from the lower right."""
    angles = np.pi / 6.0 + np.arange(6) * (np.pi / 3.0)
    return np.stack([centre[0] + radius * np.cos(angles),
                     centre[1] + radius * np.sin(angles)], axis=1)


def build_kikko(cols, rows, *, scale=2.0, **_):
    """Tortoiseshell: the plain hexagon lattice, each shared edge drawn once."""
    radius = scale / ROOT3
    segments = []
    for centre in _hex_lattice(cols, rows, radius):
        corners = _hexagon(centre, radius)
        segments.extend(zip(corners, np.roll(corners, -1, axis=0)))
    return _dedupe(segments)


def build_asanoha(cols, rows, *, scale=2.0, **_):
    """Hemp leaf: the hexagon lattice, each hexagon cut into six triangles by three diagonals.

    Straight struts throughout, which is what makes a deformation legible -- a warped straight line
    is unmistakable in a way that a warped curve is not.
    """
    radius = scale / ROOT3
    segments = []
    for centre in _hex_lattice(cols, rows, radius):
        corners = _hexagon(centre, radius)
        segments.extend(zip(corners, np.roll(corners, -1, axis=0)))
        # Three full chords through the centre, rather than six spokes: same figure, half the pieces.
        segments.extend((corners[k], corners[k + 3]) for k in range(3))
    return _dedupe(segments)


def build_kagome(cols, rows, *, scale=2.0, **_):
    """The basket weave: the trihexagonal lattice, triangles and hexagons together.

    Built as the MEDIAL of the hexagon lattice -- join the midpoints of each hexagon's six edges.
    Three families of full-length rulings would give the triangular lattice instead, which is a
    different pattern: it has no hexagons in it.
    """
    radius = scale / ROOT3
    segments = []
    for centre in _hex_lattice(cols, rows, radius):
        corners = _hexagon(centre, radius)
        midpoints = 0.5 * (corners + np.roll(corners, -1, axis=0))
        segments.extend(zip(midpoints, np.roll(midpoints, -1, axis=0)))
    return _dedupe(segments)


def build_shippo(cols, rows, *, scale=2.0, samples=64, **_):
    """Seven treasures: equal circles on a square lattice, each passing through its neighbours.

    Spacing ``r * sqrt(2)`` is what makes the overlaps cut the familiar four-petal eye. The circles
    are sampled rather than emitted as arcs, because the warp is bilinear and takes a circle to
    something that is no longer one.
    """
    step = scale
    radius = step / ROOT2
    # Closed explicitly: a ring sampled without its endpoint is an arc with a gap in it, and the
    # gap turns every circle into its own loose piece once the net is cut.
    angles = np.linspace(0.0, 2.0 * np.pi, samples + 1)
    ring = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    strokes = []
    for row in range(-2, int(np.ceil(rows / step)) + 3):
        for column in range(-2, int(np.ceil(cols / step)) + 3):
            centre = np.array([column * step, row * step], float)
            strokes.append(centre + radius * ring)
    return strokes


def build_sayagata(cols, rows, *, scale=2.0, **_):
    """Key fret: manji arms on a lattice turned through 45 degrees, linked into meanders.

    Every turn is a right angle, so the pattern states the frame it was drawn in, and under a
    deformation that frame visibly leans. This is the most explicit reading of a warp of the set.

    ⚠ This is a MEANDER approximation, not the true linked manji of sayagata proper. The arms run
    alongside each other and read as a key fret, but they do not close into the interlocking figure
    a carver would cut, and the field ends in loose arms rather than a resolved edge.
    """
    unit = scale / 4.0
    arm, hook = 2.0, 1.0
    first = np.array([[-arm, -hook], [-arm, 0.0], [arm, 0.0], [arm, hook]])
    second = np.array([[-hook, arm], [0.0, arm], [0.0, -arm], [hook, -arm]])

    turn = np.pi / 4.0
    rotate = np.array([[np.cos(turn), -np.sin(turn)], [np.sin(turn), np.cos(turn)]])
    step = 2.0 * arm * unit
    reach = int(np.ceil((cols + rows) / step)) + 3

    strokes = []
    for row in range(-reach, reach + 1):
        for column in range(-reach, reach + 1):
            # Half-step stagger on alternate rows: without it the arms sit corner to corner and
            # cross instead of running alongside one another.
            base = np.array([column * step + 0.5 * step * (row % 2), row * step], float)
            origin = base @ rotate.T
            if not (-scale <= origin[0] <= cols + scale and -scale <= origin[1] <= rows + scale):
                continue
            for stroke in (first, second):
                strokes.append((unit * stroke) @ rotate.T + origin)
    return strokes


NETS = {
    "asanoha": build_asanoha,
    "kikko": build_kikko,
    "kagome": build_kagome,
    "shippo": build_shippo,
    "sayagata": build_sayagata,
}


def build_net_fn(name: str, cols: float, rows: float, *, scale: float = 2.0, **params) -> list:
    """The net as polylines in reference coordinates over ``[0, cols] x [0, rows]``.

    Args:
        name: registry key; ``NETS`` lists them.
        cols, rows: extent of the reference domain, in control-mesh cells.
        scale: cells per motif repeat. Larger means a coarser net.
    """
    if name not in NETS:
        raise ValueError(f"unknown net {name!r}; available: {sorted(NETS)}")
    return NETS[name](cols, rows, scale=scale, **params)


def _insert_crossings(points, axis):
    """Add a vertex wherever the polyline passes through an integer line on ``axis``."""
    out = [points[0]]
    for start, end in zip(points[:-1], points[1:]):
        span = end[axis] - start[axis]
        if abs(span) > 1e-15:
            low, high = sorted((start[axis], end[axis]))
            for line in range(int(np.floor(low)) + 1, int(np.ceil(high))):
                t = (line - start[axis]) / span
                if 1e-12 < t < 1.0 - 1e-12:
                    crossing = start + t * (end - start)
                    # Snapped exactly onto the line, so both neighbouring cells read the identical
                    # parameter and their two warps meet to machine precision.
                    crossing[axis] = float(line)
                    out.append(crossing)
        out.append(end)
    return np.array(out, float)


def _segment_cells(points):
    """The cell each segment lies in, taken at its midpoint."""
    return np.floor(0.5 * (points[:-1] + points[1:])).astype(int)


def clip_to_cells_fn(polyline: Float[np.ndarray, "n 2"], *, closed: bool = False) -> list:
    """Cut a reference polyline into pieces, each lying in exactly one cell.

    Crossings are inserted first, and the polyline is then split wherever consecutive segments fall
    in different cells. Splitting on the segment's CELL rather than on the crossings alone is what
    makes a vertex that already sits exactly on a boundary work: such a crossing has ``t = 0`` and
    no interior root to find, so a crossing-only splitter walks straight through it and hands back a
    piece spanning two cells. Sampled circles land on the lattice constantly, so this is the common
    case and not a corner one.
    """
    points = np.asarray(polyline, float)
    if closed:
        points = np.vstack([points, points[:1]])
    for axis in (0, 1):
        points = _insert_crossings(points, axis)
    if len(points) < 2:
        return []

    cells = _segment_cells(points)
    pieces, start = [], 0
    for index in range(1, len(cells)):
        if not (cells[index] == cells[index - 1]).all():
            pieces.append(points[start:index + 1])
            start = index
    pieces.append(points[start:])
    return [piece for piece in pieces if len(piece) >= 2]


def _cell_of(piece):
    """Which cell a piece lies in. Every segment of it agrees, so the first one decides."""
    middle = 0.5 * (piece[0] + piece[1])
    return int(np.floor(middle[0])), int(np.floor(middle[1]))


def warp_net_fn(polylines, mesh: Float[np.ndarray, "rows cols 2"], *,
                closed: bool = False) -> list[BezierPath]:
    """Clip a reference net to the control mesh and warp each piece onto its own cell.

    Args:
        polylines: the net in reference coordinates.
        mesh: ``(rows, cols, 2)`` control points; cell ``(i, j)`` spans indices ``i..i+1``,
            ``j..j+1``.

    Returns:
        Exact ``BezierPath`` pieces in world units.
    """
    mesh = np.asarray(mesh, float)
    rows, cols = mesh.shape[0] - 1, mesh.shape[1] - 1
    paths = []
    for polyline in polylines:
        for piece in clip_to_cells_fn(polyline, closed=closed):
            column, row = _cell_of(piece)
            if not (0 <= column < cols and 0 <= row < rows):
                continue  # the net is generated past the sheet; the overhang is simply dropped
            quad = np.array([mesh[row, column], mesh[row, column + 1],
                             mesh[row + 1, column + 1], mesh[row + 1, column]])
            local = piece - np.array([column, row], float)
            paths.append(warp_polyline_fn(quad, np.clip(local, 0.0, 1.0)))
    return paths
