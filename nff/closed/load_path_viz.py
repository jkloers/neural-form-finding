"""The load-path figure: tension as routes, compression and shear as regions.

ONE figure. Earlier rounds carried a zone-flow view, a skeleton view, interpolated continuum
currents, a two-panel normal/shear split and an annotated analysis sheet; all of them are gone. What
survives is the reading that held up:

  * **Tension is drawn as PATHS** because it is path-like -- its connected runs reach 11-15 tiles.
  * **Compression is drawn as REGIONS** because it is not path-like: its links form components of at
    most 2-3 tiles, so chains would claim a continuity the data does not have.
  * **Shear is drawn as SLIP PLANES** -- a fault trace with opposed half-arrows, the structural-
    geology symbol. Its runs are short but near-perfectly COLLINEAR (elongation 3471), so they are
    planes, and the half-arrows show the sense in which the two sides slide past each other.
  * Three channels, three REPRESENTATIONS -- area, plane, route. Separating by representation beats
    separating by hue, and it frees the fill channel for compression alone: two modes sharing one
    fill forced a winner-takes-tile rule that no reader could invert.

⚠ The regions shade TILE POLYGONS, never an interpolated field. Tiles are the natural resolution of
this medium -- a rigid tile is where force arrives and leaves -- so a per-tile total is exact, it
cannot smear a mode across a void, and a short straight run of sheared tiles still renders as a short
straight band. Shear's runs are only 4-6 tiles but are near-perfectly collinear, and a smooth density
would have destroyed exactly that.

Boundary conditions come from ``visualization.draw_supports_and_loads`` -- the same code the run's
loading diagram uses, so the two figures cannot disagree about how the sheet is held.

⚠ A sheet held by a single pin has a zero-energy rigid rotation and swings bodily (36 deg measured),
which makes any load-path reading incoherent: the geometry no longer aligns with the force. Hold the
sheet against rotation before reading a load path from it.
"""

from typing import Optional

import numpy as np

from nff.closed.force_stats import mode_intensity, tension_routes, tile_pressure
from nff.closed.load_paths import ForceField, edge_modes, interface_decomposition
from nff.utils.figstyle import GREY, INK, ORANGE, TEAL, apply_charter

import matplotlib.pyplot as plt                                       # noqa: E402
from matplotlib.collections import LineCollection, PolyCollection    # noqa: E402


# From the article's CHARTER palette, as one WARM family. Tension takes PRINCETON ORANGE; RED stays
# reserved for the applied load, so a route is never mistaken for the thing driving it.
# ⚠ Compression was briefly on SIM (#08519c), which belongs to the experiment-vs-model palette and
# means MEASURED-vs-SIMULATED -- a distinction this figure does not draw. It is now the warm wash
# below; the line/field pair is separated by a white casing, not by spending another hue.
MODE_COLOUR = {'tension': ORANGE, 'compression': ORANGE, 'shear': TEAL}

# The compression FIELD takes the COMPRESSION END OF THE DIVERGING MAP, the same hue a signed
# stress plot of the same sheet gives a compressed tile. It was a warm amber while the wash was a
# per-interface quantity of its own; now that it is the negative part of the signed field, a
# separate hue would say the two panels show different things. Tension is separated from it by a
# white casing rather than by hue.
COMPRESSION_FILL = TEAL

LW_MAX = 7.5           # stroke width in points at ...
LW_AT = 1.2            # ... this multiple of the applied load


# ── The two channels ──────────────────────────────────────────────────────────────────────────

# ``mode_intensity`` is imported from nff.closed.force_stats, where it now lives with the rest of
# the per-tile readings. It is re-exported here because this is where callers expect it.


def zone_colours(field: ForceField, dec, *, strength=0.95, floor=0.0, contrast=1.0,
                 saturate_at=75.0):
    """(n_faces,) RGB per tile shaded by the COMPRESSIVE part of the net tile stress.

    ⚠ This is the NEGATIVE PART OF ONE SIGNED FIELD, not a channel of its own. Shading the
    one-sided per-interface compression instead -- the sum of ``max(-f_n, 0)`` over a tile's hinges
    -- answers a different question and disagrees: measured over five deployments, between 5% and
    40% of the tiles it painted most strongly were in NET TENSION, because a tile squeezed at one
    hinge and pulled at three registered as compressed. A panel drawn that way cannot be described
    as the compressive half of a signed stress plot, because it is not one.

    Normalised and ramped exactly as the signed field is, so a tile's shade here and its shade in a
    signed plot of the same sheet are the same function of the same number. Compression owns the
    fill channel outright; shear has its own representation (:func:`shear_planes`).

    Pre-multiplied against white rather than using per-face alpha, so the result is a plain colour
    list the tile renderer takes unchanged.

    Args:
        strength: how far a fully compressed tile travels from the paper toward the hue.
        floor: intensity below which a tile stays unpainted. Zero keeps the ramp linear.
        contrast: gamma on the normalised intensity. One keeps it linear, matching the signed map.
        saturate_at: percentile of ``|p|`` the ramp saturates at. The stress distribution is heavily
            skewed, so normalising by the maximum leaves the sheet uniformly pale.
    """
    pressure = tile_pressure(field)
    pressure = pressure / (float(np.abs(pressure).mean()) or 1.0)
    compression = np.maximum(-pressure, 0.0)
    reference = float(np.percentile(np.abs(pressure), saturate_at)) if pressure.size else 0.0
    comp_n = compression / max(reference, 1e-30)

    from matplotlib.colors import to_rgb
    base = np.array(to_rgb('#FBFAF7'))
    hue = np.array(to_rgb(COMPRESSION_FILL))
    colours, shaded = [], 0
    for face in range(field.n_faces):
        level = float(np.clip((comp_n[face] - floor) / (1.0 - floor), 0.0, 1.0)) ** contrast
        colours.append(tuple(base + strength * level * (hue - base)))
        shaded += int(level > 0.02)
    return colours, shaded


# ── Shear as slip planes ──────────────────────────────────────────────────────────────────────

def _mean_axis(vectors, reference=None):
    """Mean of a LINE field: sign-align before averaging.

    ⚠ A cut axis has no canonical sign, so ``slip_dir`` comes out as +v or -v arbitrarily per hinge.
    Averaging the signed vectors cancels them and returns a direction roughly PERPENDICULAR to the
    truth -- measured, that put a plane's own contacts 9 tiles along its "normal" and 0.3 along its
    "direction", so the trace had to zigzag sideways to reach them. This is the same line-field trap
    that shredded the principal-direction tracer: an orientation is not a vector.
    """
    v = np.asarray(vectors, dtype=float)
    ref = np.asarray(reference if reference is not None else v[0], dtype=float)
    flipped = v * np.where(v @ ref < 0, -1.0, 1.0)[:, None]
    mean = flipped.mean(axis=0)
    return mean / max(np.linalg.norm(mean), 1e-30)


def shear_planes(field: ForceField, dec, keep, *, half_length=0.55, merge_perp=0.30,
                 align_deg=5.0, max_gap=2.0, min_members=3):
    """Slip planes, built from the DEFINITION: each runs normal to its hinge's axis.

    ⚠ The previous construction chained hinges by tile-graph adjacency and was **81 degrees wrong**
    (median, 91% beyond 60 degrees) -- essentially perpendicular to the truth. In this lattice
    consecutive shear interfaces stack ALONG the hinge axis, so a chain of them runs across the plane
    rather than along it. Shear is by definition normal to the hinge axis, so the plane is built from
    that directly and cannot be off.

    Each shear interface contributes a segment through its own contact point along ``shear_dir``
    (= perp of the cut axis). Segments that are collinear -- parallel within ``align_deg`` and offset
    by less than ``merge_perp`` tiles -- merge into one plane. Because the segment extends either side
    of the contact, it crosses the VOIDS that flank it rather than running corner to corner: measured,
    91% of shear interfaces have another lying along their own slip direction at a median off-axis
    angle of 0.9 degrees, so these planes are a real feature of the sheet, not a drawing convention.

    Returns dicts with ``points`` (the trace), ``direction``, ``normal``, ``slip``, ``magnitude``,
    ``n_hinges``.
    """
    modes = edge_modes(dec)
    selected = np.flatnonzero(keep & (modes == 'shear'))
    if selected.size == 0:
        return []

    unit = field.tile_scale
    hinge_xy = np.asarray(field.hinge_xy)
    face_xy = np.asarray(field.face_xy)
    axis = dec['n_hat']                                   # the hinge (cut) axis
    slip_dir = np.stack([-axis[:, 1], axis[:, 0]], axis=-1)   # normal to it: the slip direction

    # ⚠ Group by SEEDED collinearity, never by transitive union. Union-find chains A~B and B~C, so
    # a group drifts arbitrarily far from any single line even though every LINK is within tolerance
    # -- measured, that produced "planes" whose contacts sat up to 5 tiles off their own fitted line,
    # and the trace zigzagged sideways to reach them (path length 3x the span). Seeding instead means
    # every member is within tolerance of THE SAME line, so a plane is always flat.
    order = np.argsort(-np.abs(dec['f_t'][selected]))
    remaining = list(selected[order])                     # strongest interface seeds first
    cos_tol = np.cos(np.radians(align_deg))
    groups = []
    while remaining:
        seed = remaining.pop(0)
        members, rest = [seed], []
        for h in remaining:
            aligned = abs(float(slip_dir[h] @ slip_dir[seed])) >= cos_tol
            offset = abs(float((hinge_xy[h] - hinge_xy[seed]) @ axis[seed]))
            (members if (aligned and offset <= merge_perp * unit) else rest).append(h)
        remaining = rest
        groups.append(np.array(members))

    # Collinearity alone spans the sheet: two interfaces at opposite edges are collinear, and a line
    # drawn between them runs over material that is not shearing at all -- which is what made the
    # planes look unrelated to the geometry around them. A plane is a LOCAL run, so split wherever
    # the sheared interfaces stop for more than `max_gap` tiles.
    runs = []
    for members in groups:
        hinges = np.asarray(members)
        direction = _mean_axis(slip_dir[hinges], slip_dir[hinges[0]])
        along = (hinge_xy[hinges] - hinge_xy[hinges].mean(axis=0)) @ direction
        order = np.argsort(along)
        current = [hinges[order[0]]]
        for prev, nxt in zip(order[:-1], order[1:]):
            if along[nxt] - along[prev] > max_gap * unit:
                runs.append(np.array(current))
                current = []
            current.append(hinges[nxt])
        runs.append(np.array(current))

    planes = []
    for hinges in runs:
        pts = hinge_xy[hinges]
        direction = _mean_axis(slip_dir[hinges], slip_dir[hinges[0]])
        normal = np.array([-direction[1], direction[0]])
        centre = pts.mean(axis=0)

        # Sense of slip: how the tile on the +normal side is pushed along the plane.
        slip = 0.0
        for h in hinges:
            j = int(field.face_pairs[h][1])
            side = np.sign(np.dot(face_xy[j] - centre, normal)) or 1.0
            slip += float(np.dot(np.asarray(field.t_total)[h], direction)) * side
        sense = 1.0 if slip >= 0 else -1.0

        # The trace follows the ACTUAL contacts, ordered along the plane, then extends past the end
        # ones so it visibly crosses the flanking voids. A pure straight segment is the fitted line
        # and reads as CAD; the contacts are collinear to ~1 degree anyway, so curving through them
        # costs no accuracy and looks like a fault trace instead of a ruler.
        # Follow the contacts, but damp their perpendicular scatter: a plane is flat by definition,
        # so the residual is measurement wobble, not shape. Keeping a fraction of it looks natural;
        # keeping all of it lets the trace wander off its own plane.
        # Extend by a fraction of the plane's OWN extent, capped. A fixed extension is up to 49% of
        # the drawn line on a short plane -- half the stroke would be invented, and it reads as an
        # arbitrary stub rather than a measured feature.
        along_raw = (pts - centre) @ direction
        reach = min(half_length * unit, 0.25 * float(np.ptp(along_raw) + 1e-12))

        along = (pts - centre) @ direction
        perp = (pts - centre) @ normal
        eased = centre + np.outer(along, direction) + 0.35 * np.outer(perp, normal)
        ordered = eased[np.argsort(along)]
        control = np.vstack([ordered[:1] - direction * reach,
                             ordered,
                             ordered[-1:] + direction * reach])
        trace = chaikin(control, iterations=2)

        planes.append({
            'points': trace,
            'direction': direction,
            'normal': normal,
            'slip': sense,
            'magnitude': float(np.abs(dec['f_t'][hinges]).mean() / field.f0),
            'n_hinges': int(hinges.size),
            'contacts': pts,                            # the interfaces the plane is MADE of
            # Fewer than `min_members` sheared interfaces is not evidence of a plane -- two adjacent
            # contacts span ~1 tile, and drawing that as a fault trace claims a band from a pair.
            # They stay as POINTS: still "shear happens here", without inventing a plane.
            'is_plane': bool(hinges.size >= min_members),
        })
    return planes


def _point_to_segment(point, a, b):
    """Distance from a point to the segment ``a-b``."""
    ab = np.asarray(b) - np.asarray(a)
    t = np.clip(np.dot(np.asarray(point) - a, ab) / max(np.dot(ab, ab), 1e-30), 0.0, 1.0)
    return float(np.linalg.norm(np.asarray(point) - (a + t * ab)))


def _segment_enters(start, end, poly, samples=9):
    """Does the straight segment pass through the interior of ``poly``?"""
    from matplotlib.path import Path

    path = Path(np.asarray(poly))
    for t in np.linspace(0.0, 1.0, samples + 2)[1:-1]:
        point = np.asarray(start) + t * (np.asarray(end) - np.asarray(start))
        if path.contains_point(point) and min(
                _point_to_segment(point, a, b)
                for a, b in zip(poly, np.roll(poly, -1, axis=0))) > 1e-9:
            return True
    return False


def _border_trace(field: ForceField, node_path, walk):
    """The slip trace as it physically is: a path ALONG THE TILE BORDERS.

    Two consecutive shear interfaces always share a tile, and both sit on that tile's outline, so the
    plane between them runs along that tile's EDGE. Joining the two contact points with a straight
    segment instead drives the trace through the tile's interior -- through solid material, where a
    slip plane cannot be. Walking the shorter way round the shared tile's polygon puts the trace back
    on the boundary between the two sides that are sliding.
    """
    nodes = np.asarray(field.face_nodes)
    hinge_xy = np.asarray(field.hinge_xy)

    trace = [hinge_xy[walk[0]]]
    for k in range(len(walk) - 1):
        start, end = hinge_xy[walk[k]], hinge_xy[walk[k + 1]]
        shared = int(node_path[k + 1])                  # the tile both interfaces belong to
        poly = nodes[shared]

        # Prefer the STRAIGHT segment: between two contacts the plane normally crosses a VOID, and
        # a void is not material. Detour only when the chord would actually pass through the shared
        # tile -- forcing a detour at every step is what made the trace wrap tiles instead of
        # running between the two sides.
        if not _segment_enters(start, end, poly):
            trace.append(end)
            continue

        n = poly.shape[0]
        a = int(np.argmin(np.linalg.norm(poly - start, axis=1)))
        b = int(np.argmin(np.linalg.norm(poly - end, axis=1)))
        # Of the two ways round, take the one that hugs the chord -- i.e. the side the plane is on.
        # The merely-shorter walk can go round the far side of the tile, away from the interface.
        best, best_cost = None, np.inf
        for step, count in ((1, (b - a) % n), (-1, (a - b) % n)):
            corners = [poly[(a + step * m) % n] for m in range(1, count)]
            cost = (np.mean([_point_to_segment(c, start, end) for c in corners])
                    if corners else 0.0)
            if cost < best_cost:
                best, best_cost = corners, cost
        trace.extend(best)
        trace.append(end)
    return np.asarray(trace)


def _half_arrow(ax, start, direction, length, barb_side, *, colour, lw, zorder):
    """One half-arrow: a shaft with a single barb, the structural-geology slip symbol.

    A full arrowhead would read as an applied force. The half-head says "this side slides this way",
    which is what a slip plane shows.
    """
    d = np.asarray(direction, dtype=float)
    d = d / max(np.linalg.norm(d), 1e-30)
    n = np.array([-d[1], d[0]]) * barb_side
    end = np.asarray(start) + d * length
    ax.plot([start[0], end[0]], [start[1], end[1]], color=colour, lw=lw,
            solid_capstyle='round', zorder=zorder)
    barb = end - d * (0.34 * length) + n * (0.24 * length)
    ax.plot([end[0], barb[0]], [end[1], barb[1]], color=colour, lw=lw,
            solid_capstyle='round', zorder=zorder)


def draw_shear_planes(ax, planes, unit, *, zorder=14, colour=None, weight_scale=1.0):
    """Draw each slip plane as a trace with opposed half-arrows on either side."""
    colour = colour or MODE_COLOUR['shear']
    drawn = 0
    for plane in planes:
        pts = plane['points']
        strength = float(np.clip(plane['magnitude'] / 0.5, 0.35, 1.0))
        if not plane.get('is_plane', True):
            # An isolated sheared interface, which on this lattice is the RULE rather than the
            # exception: fed every shear-dominated hinge with no threshold at all, the collinearity
            # test still yields 1 plane against 99 isolated contacts, and loosening it to a full
            # tile buys only 10 runs of at most 5 hinges. Shear here is distributed, not organised
            # into slip planes, so the mark is a point and the legend must say so. The size range is
            # wide enough that the point still carries the magnitude it transmits.
            spot = plane['contacts']
            ax.scatter(spot[:, 0], spot[:, 1], s=(8 + 38 * strength) * weight_scale ** 2,
                       c=colour, edgecolors='white', linewidths=0.7 * weight_scale,
                       zorder=zorder + 2)
            continue
        drawn += 1
        trace = pts
        # Taper toward the tips: a slip plane fades where it runs out of sheared interfaces, and a
        # bar of constant width claims it stops dead.
        t = np.linspace(-1.0, 1.0, trace.shape[0])
        profile = np.sqrt(np.clip(1.0 - 0.82 * t ** 2, 0.06, 1.0))
        widths = (1.2 + 4.2 * strength) * profile * weight_scale
        segs = np.stack([trace[:-1], trace[1:]], axis=1)
        ax.add_collection(LineCollection(segs, colors='white',
                                         linewidths=0.5 * (widths[:-1] + widths[1:])
                                         + 2.2 * weight_scale,
                                         alpha=0.8, zorder=zorder - 1,
                                         capstyle='round', joinstyle='round'))
        ax.add_collection(LineCollection(segs, colors=colour,
                                         linewidths=0.5 * (widths[:-1] + widths[1:]),
                                         alpha=0.92, zorder=zorder,
                                         capstyle='round', joinstyle='round'))

        # Anchor the line to the interfaces it represents. Without this the plane floats over the
        # tessellation with nothing tying it to the geometry -- which is exactly how it read.
        contacts = plane.get('contacts')
        if contacts is not None:
            ax.scatter(contacts[:, 0], contacts[:, 1], s=(16 + 26 * strength) * weight_scale ** 2,
                       c=colour, edgecolors='white', linewidths=0.8 * weight_scale,
                       zorder=zorder + 2)

        d, n, sense = plane['direction'], plane['normal'], plane['slip']
        length = 0.62 * unit * strength
        offset = 0.17 * unit
        mid = trace.mean(axis=0)
        # One pair per plane, at its midpoint: the two sides sliding opposite ways.
        for side in (+1.0, -1.0):
            start = mid + n * offset * side - d * sense * side * length / 2
            _half_arrow(ax, start, d * sense * side, length, -side,
                        colour=colour, lw=(1.0 + 2.2 * strength) * weight_scale,
                        zorder=zorder + 1)
    return drawn


def chaikin(points, iterations=3):
    """Corner-cutting subdivision: smooth a polyline without letting it leave its own hull.

    A smoothing SPLINE was the obvious choice and the wrong one -- on a wandering route it bows
    outside the material and draws a chain where there are no tiles. Chaikin only ever cuts corners,
    so the curve stays inside the control polygon by construction. Extra columns (a per-point force)
    are carried through the same weights, so widths stay attached to the geometry they describe.
    """
    pts = np.asarray(points, dtype=float)
    for _ in range(iterations):
        if pts.shape[0] < 3:
            break
        a, b = pts[:-1], pts[1:]
        cut = np.empty((2 * a.shape[0], pts.shape[1]))
        cut[0::2] = 0.75 * a + 0.25 * b
        cut[1::2] = 0.25 * a + 0.75 * b
        pts = np.vstack([pts[:1], cut, pts[-1:]])       # keep the true endpoints
    return pts


def draw_tapered(ax, pts_with_width, *, colour, lw_scale, alpha, zorder, lw_floor=0.8, casing=1.8):
    """Draw a smoothed polyline whose width follows the force carried along it.

    ``casing`` lays a white stroke underneath. The compression field is warm and the tension line is
    warm; separating them by hue would break the palette, so they are separated by an outline
    instead -- the standard cartographic fix for a line crossing a field of its own colour.
    """
    xy = pts_with_width[:, :2]
    w = pts_with_width[:, 2]
    segments = np.stack([xy[:-1], xy[1:]], axis=1)
    widths = np.maximum(lw_scale * 0.5 * (w[:-1] + w[1:]), lw_floor)
    if casing:
        ax.add_collection(LineCollection(segments, colors='white', linewidths=widths + casing,
                                         alpha=0.85, zorder=zorder - 1,
                                         capstyle='round', joinstyle='round'))
    ax.add_collection(LineCollection(segments, colors=colour, linewidths=widths, alpha=alpha,
                                     zorder=zorder, capstyle='round', joinstyle='round'))


def _edge_path(field: ForceField, h: int):
    """Centroid -> hinge -> centroid, so a route visibly crosses the gateway it uses."""
    i, j = field.face_pairs[h]
    return np.array([field.face_xy[i], field.hinge_xy[h], field.face_xy[j]])


def _edge_width(ratio: float, lw_max: float = LW_MAX) -> float:
    return lw_max * float(np.clip(ratio / LW_AT, 0.0, 1.0))


def draw_tension_routes(ax, field: ForceField, dec, keep, *, zorder=20, min_run=3,
                        weight_scale=1.0):
    """Tension as continuous routes; runs shorter than ``min_run`` tiles stay faint.

    Drawing every 2-tile fragment at full strength turns mid-sheet into a hatch that LOOKS like a
    path field and is not one -- on the single-clamp sheet the genuine runs are 12 and 11 tiles and
    everything else is a pair. Fragments remain, faintly, because they do carry load.
    """
    # The routes are selected in force_stats, so the spines drawn here and the directness reported
    # in the text are the same objects. Only the drawing lives below.
    network = tension_routes(field, dec, keep, min_run=min_run)
    if not network['graph_hinges']:
        return 0

    face_xy, hinge_xy = np.asarray(field.face_xy), np.asarray(field.hinge_xy)
    for route in network['routes']:
        node_path, hinges = route['node_path'], route['hinges']
        control = []
        for k, node in enumerate(node_path):
            if k + 1 < len(node_path):
                h = hinges[k]
                f = float(dec['f_norm'][h])
                control.append([*face_xy[node], f])
                control.append([*hinge_xy[h], f])          # the gateway it actually crosses
            else:
                control.append([*face_xy[node], control[-1][2]])
        draw_tapered(ax, chaikin(np.array(control)), colour=MODE_COLOUR['tension'],
                     lw_scale=LW_MAX / LW_AT * weight_scale, alpha=0.97, zorder=zorder + 1,
                     lw_floor=1.8 * weight_scale, casing=1.8 * weight_scale)

    # Everything not on a spine stays, faintly: it carries load, but it is not a route.
    spine_edges = set(network['spine_hinges'])
    for h in network['graph_hinges']:
        if h in spine_edges:
            continue
        path = chaikin(_edge_path(field, h), iterations=2)
        ax.plot(path[:, 0], path[:, 1], color=MODE_COLOUR['tension'], lw=1.1 * weight_scale,
                alpha=0.22, zorder=zorder, solid_capstyle='round')
    return len(network['routes'])


# ── The figure ────────────────────────────────────────────────────────────────────────────────

CLAMP_FILL = "#5A6B8C"     # slate: outside the warm compression ramp and the teal shear cue, so a
                           # held tile cannot be mistaken for either reading of the force network


def draw_load_path(ax, field: ForceField, *, tessellation=None, config=None,
                   threshold: Optional[float] = None, zones: bool = True,
                   highlight=None, highlight_colour: Optional[str] = None,
                   canvas_aspect: Optional[float] = None, label: Optional[str] = None,
                   clamp_as_fill: bool = False, weight_scale: float = 1.0,
                   shear_colour: Optional[str] = None) -> dict:
    """Draw one deployed sheet's load path into an EXISTING axes. Returns the stats dict.

    Split out of :func:`plot_load_path` so a multi-panel figure can place this reading beside
    another view of the same deployment instead of loading a separate PNG.

    Args:
        ax:           the axes to draw into. Its aspect and frame are set here.
        field:        the extracted ``ForceField``.
        tessellation: the DEPLOYED tessellation, for the shared boundary-condition drawing.
        config:       the run's config, for clamp/roller/load specs. Omit to skip the BC overlay.
        threshold:    share of F0 below which a link is faint. ``None`` uses the MEAN transmitted
                      force, the standard strong/weak partition of a force network -- a fixed
                      percentage keeps 96% of the links on this sheet and separates nothing.
        zones:        shade compression/shear regions. ``False`` gives the pure-route reading.
        highlight:    hinge index, or a sequence of them, to ring on the sheet.
        highlight_colour: ring colour; defaults to ``INK``.
        canvas_aspect: width/height of the drawing area, for the proportional window. ``None``
                      reads it off the axes, which is what a panel in a grid wants.
        weight_scale: multiplier on every stroke, marker and casing. The defaults suit a
                      full-page panel; a small multi-panel cell needs them proportionally thinner,
                      because point sizes do not shrink with the drawing.
        shear_colour: colour of the shear marks. ``None`` keeps ``MODE_COLOUR['shear']``. A panel
                      whose compression wash already spends the same hue needs a second one here,
                      or the two readings merge.
    """
    dec = interface_decomposition(field)
    if threshold is None:
        threshold = float(dec['f_norm'].mean())
    keep = dec['f_norm'] >= threshold

    ax.set_aspect('equal')
    ax.axis('off')

    if zones:
        colours, shaded = zone_colours(field, dec)
    else:
        colours, shaded = ['#F4F5F7'] * field.n_faces, 0
    # A CLAMP DRAWN AS A WALL is sized from the sheet span, not from the tile, so on a small sheet it
    # spans several tiles and stops naming which one is held. Filling the tile itself cannot be
    # ambiguous: the support IS the tile.
    clamped_ids = []
    if clamp_as_fill and config is not None:
        topo = config.topology
        clamped_ids = [int(i) for i in (topo.get('bc_clamped') or topo.get('clamped_faces') or [])]
        for i in clamped_ids:
            if 0 <= i < len(colours):
                colours[i] = CLAMP_FILL
    ax.add_collection(PolyCollection(list(field.face_nodes), facecolors=colours,
                                     edgecolors='#9CA3AB', linewidths=0.7 * weight_scale,
                                     zorder=2))

    planes = shear_planes(field, dec, keep)
    n_planes = draw_shear_planes(ax, planes, field.tile_scale, colour=shear_colour,
                                 weight_scale=weight_scale)
    routes = draw_tension_routes(ax, field, dec, keep, weight_scale=weight_scale)

    reach = np.zeros((0, 2))
    if config is not None and tessellation is not None:
        topo = config.topology
        x0, y0, x1, y1, span = draw_supports_and_loads_safe(
            ax, tessellation, [] if clamp_as_fill else (topo.get('bc_clamped') or topo.get('clamped_faces')),
            topo.get('loads'), clamped_dofs=topo.get('clamped_dofs'),
            y_roller_faces=topo.get('y_roller_faces'), x_roller_faces=topo.get('x_roller_faces'),
            weight_scale=weight_scale)
        # The helper returns the TESSELLATION bbox, but it also draws a support wall outside it and
        # load arrows further out still. Pad by the largest of those reaches or the window clips the
        # very annotations that say how the sheet is held.
        pad = 0.18 * span
        reach = np.array([[x0 - pad, y0 - pad], [x1 + pad, y1 + pad]])

    if canvas_aspect is None:
        box = ax.get_window_extent()
        canvas_aspect = float(box.width / max(box.height, 1e-30))
    _fit_axes(ax, field, reach, canvas_aspect)

    if highlight is not None:
        draw_hinge_marks(ax, field, highlight, colour=highlight_colour)
    if label:
        ax.text(0.012, 0.985, label, transform=ax.transAxes, va='top', ha='left',
                fontsize=12, color=GREY)

    return {'tension_routes': routes, 'shear_planes': n_planes,
                     'isolated_shear': sum(1 for p in planes if not p['is_plane']), 'shaded_tiles': shaded,
            'strong_links': int(keep.sum()), 'total_links': int(keep.size),
            'threshold': threshold}


def draw_hinge_marks(ax, field: ForceField, hinges, *, colour: Optional[str] = None, size=13.0):
    """Ring one or more hinges on a drawn sheet, by index into ``field.hinge_xy``.

    A RING rather than a filled dot on purpose: ``MODE_COLOUR`` already spends the palette on the
    three transmission modes, so a mark that differs only in hue reads as a fourth mode. A hollow
    ring on a white halo is the one shape nothing else in this drawing uses.
    """
    idx = np.atleast_1d(np.asarray(hinges, dtype=int))
    xy = np.asarray(field.hinge_xy)[idx]
    # ONE hollow BLACK ring. The old halo + rim + colour stack read as three concentric marks at
    # page size, and the white halo punched a hole in the tile it was supposed to sit on.
    for ms, edge, lw in ((size, colour or INK, 2.0),):
        ax.plot(xy[:, 0], xy[:, 1], 'o', ms=ms, markerfacecolor='none', markeredgecolor=edge,
                markeredgewidth=lw, zorder=40, linestyle='none')


def plot_load_path(field: ForceField, tessellation, out_png: str, *, config=None,
                   threshold: Optional[float] = None, zones: bool = True,
                   label: Optional[str] = None):
    """Draw the load path of one deployed sheet to its own PNG. Returns ``(out_png, stats)``.

    See :func:`draw_load_path` for the arguments; this wrapper only owns the figure lifecycle and
    the canvas sizing.
    """
    apply_charter()
    sheet = np.ptp(field.face_nodes.reshape(-1, 2), axis=0)
    aspect = float(sheet[1] / max(sheet[0], 1e-30))
    fig_w = 11.0
    plot_h = min(fig_w * aspect, 8.5)
    fig = plt.figure(figsize=(fig_w, plot_h), facecolor='white')
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])

    stats = draw_load_path(ax, field, tessellation=tessellation, config=config,
                           threshold=threshold, zones=zones, canvas_aspect=fig_w / plot_h,
                           label=label)

    fig.savefig(out_png, dpi=200, facecolor='white')
    plt.close(fig)
    return out_png, stats


def draw_supports_and_loads_safe(ax, tessellation, clamped_faces, load_specs, **kwargs):
    """Shared BC drawing, tolerant of a config that does not declare every support kind."""
    from nff.utils.visualization import draw_supports_and_loads
    return draw_supports_and_loads(ax, tessellation, clamped_faces or [], load_specs or [], **kwargs)


def _fit_axes(ax, field: ForceField, reach, canvas_aspect: float):
    """Window on the geometry, with a PROPORTIONAL per-axis margin.

    One absolute margin on a wide sheet inflates y more than x, breaking the aspect the canvas was
    sized for and letterboxing the plot.
    """
    pts = field.face_nodes.reshape(-1, 2)
    if len(reach):
        pts = np.vstack([pts, reach])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    lo, hi = lo - 0.06 * (hi - lo), hi + 0.06 * (hi - lo)

    box, centre = hi - lo, 0.5 * (lo + hi)
    if box[0] / box[1] < canvas_aspect:
        box[0] = box[1] * canvas_aspect
    else:
        box[1] = box[0] / canvas_aspect
    ax.set_xlim(centre[0] - box[0] / 2, centre[0] + box[0] / 2)
    ax.set_ylim(centre[1] - box[1] / 2, centre[1] + box[1] / 2)
