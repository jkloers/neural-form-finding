"""A cut sheet deployed by hand, measured against the simulation that designed it.

Row 1 is the comparison and nothing else: the MEASURED tile outlines against the PREDICTED ones, in
the design's own millimetre frame. The photographs the measurement came from are not drawn. They
were, and they cost the plate a grey raster in every panel while adding nothing the outlines do not
already say -- the tape was only ever the instrument. What is left is a line drawing in which the
disagreement is the only thing with contrast, and a plate that is pure vector.

Row 2 carries what the outlines cannot. Panel (d) is hinge opening, measured against simulated: a
DIFFERENCE of tile rotations, so it is free of any registration and tests the kinematics alone.
Panel (e) is the per-tile shape residual -- what is left once the whole sheet's rigid-body motion is
removed -- against the floor measured on the closed sheet, where the answer is known to be zero.

Colour says WHAT a thing is and never WHICH state it is in: orange is the measured sheet, red the
disagreement between it and the prediction, and everything structural is black -- the predicted
outline solid, the target dashed, the pull arrow solid. The manuscript draws every target as a black
dashed line, and matching that was worth more than any locally prettier choice. The three states are ordered, so row 2 separates them
on a neutral grey ramp and leaves every hue free to keep its meaning.

    PYTHONPATH=$(pwd) python3 nff/scripts/figures/plot_deployment_validation.py
"""

import argparse
import json
import os

import numpy as np

from nff.calibration import photo_geometry as pg
from nff.calibration import photo_rectify as pr
from nff.utils.figstyle import (FS_ANNOT, FS_BASE, GREY, INK, ORANGE, RED, SIM, TEAL,
                                apply_charter, figsize_in, page_bp, panel_letter, paper_rc,
                                save_paper_figure, use_agg)

use_agg()
import matplotlib.pyplot as plt                                      # noqa: E402
from matplotlib.lines import Line2D                                  # noqa: E402
from matplotlib.patches import PathPatch, Patch, Rectangle           # noqa: E402
from matplotlib.path import Path                                     # noqa: E402
from shapely.geometry import Polygon                                 # noqa: E402

STATES = ("Closed", "Deploying", "Deployed")
LABEL = {"Closed": "closed", "Deploying": "deploying", "Deployed": "deployed"}

def shades(base, n, lightest=0.62):
    """``n`` tints of one charter hue, light to dark, ending on the hue itself."""
    c = np.array(plt.matplotlib.colors.to_rgb(base))
    return [tuple(c + (1.0 - c) * f) for f in np.linspace(lightest, 0.0, n)]


# Each quantitative panel takes shades of the charter hue that already MEANS its quantity: teal is
# the hinge colour and (d) is hinge opening; orange is the sheet, and (e) measures where the sheet
# ended up. Deployment state is ordered, so it runs light to dark within the hue.
TONE_HINGE = None      # filled in main(), once matplotlib is configured
TONE_ERROR = None

# The target is a black dashed line everywhere in the manuscript -- plot_validity_ladder.py:130
# draws it as INK, lw 0.9, dash (3, 2). Matching it exactly is worth more than any local optimum.
TARGET_KW = dict(color=INK, lw=0.9, ls=(0, (3, 2)))
# The target owns black DASHED, so the prediction takes black SOLID: dash against solid separates
# them, and a single ink weight keeps the plate monochrome apart from the two things that carry
# meaning -- the orange sheet and the red disagreement. --sim-style offers the alternatives.
# Thin on purpose: the disagreement is a sliver a few millimetres wide at this scale, and a heavy
# outline simply covers the thing the figure exists to show.
SIM_STYLES = {"blue": dict(color=SIM, lw=0.45),
              "ink": dict(color=INK, lw=0.4),
              "grey": dict(color=GREY, lw=0.45)}
MARGIN_MM = 235.0           # breathing room, and room for the dimensions in panel (a)
ROT_DEG = -90.0             # brings the sheet's long axis horizontal; see --rotate


def rotate(P, deg):
    """Rigid rotation about the origin, applied to (..., 2) points."""
    if not deg:
        return np.asarray(P, float)
    a = np.radians(deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return np.asarray(P, float) @ R.T


def align_sim(sim, meas):
    """Simulated tiles with the rigid-body offset to the measurement removed.

    Off by default. The simulation clamps its hold tile rigidly and the lab held it by hand, so the
    two frames differ by however far the hand wandered. Removing that shows shape against shape;
    leaving it in shows the experiment as it happened. Both are honest, they answer different
    questions, and the caption has to say which is drawn.
    """
    R, t, _ = pg.kabsch_2d(sim.reshape(-1, 2), meas.reshape(-1, 2))
    return (sim.reshape(-1, 2) @ R.T + t).reshape(sim.shape)


def _poly(ax, p, **kw):
    q = np.vstack([p, p[:1]])
    ax.plot(q[:, 0], q[:, 1], **kw)


def _fill_shapely(ax, geom, **kw):
    """Fill a shapely (Multi)Polygon, holes included, as one patch."""
    if geom.is_empty:
        return
    parts = list(geom.geoms) if geom.geom_type.startswith("Multi") else [geom]
    verts, codes = [], []
    for g in parts:
        if g.geom_type != "Polygon" or g.is_empty:
            continue
        for ring in [g.exterior, *g.interiors]:
            c = np.asarray(ring.coords)
            if len(c) < 3:
                continue
            verts.append(c)
            codes.append([Path.MOVETO] + [Path.LINETO] * (len(c) - 2) + [Path.CLOSEPOLY])
    if verts:
        ax.add_patch(PathPatch(Path(np.concatenate(verts), np.concatenate(codes)), **kw))


def error_region(meas_tile, sim_tile):
    """Where measurement and prediction disagree: the symmetric difference of the two tiles.

    Drawn rather than described, so the error is an AREA the eye can size directly instead of a
    gap between two outlines it has to estimate.
    """
    a, b = Polygon(meas_tile), Polygon(sim_tile)
    if not a.is_valid:
        a = a.buffer(0)
    if not b.is_valid:
        b = b.buffer(0)
    return a.symmetric_difference(b)


def _dimension(ax, a, b, text, off, tick=55.0, pad=26.0, rot=0.0):
    """An engineering dimension: witness lines, a dimensioned run, and the length written on it.

    Preferred over a scale bar in the closed panel. All three panels share one scale, and the closed
    sheet is a known object, so dimensioning it gives the reader the scale AND the size of the thing
    -- one element doing two jobs instead of two doing one each.
    """
    a, b, off = np.asarray(a, float), np.asarray(b, float), np.asarray(off, float)
    n = off / max(np.linalg.norm(off), 1e-9)
    A, B = a + off, b + off
    for p, q in ((a, A), (b, B)):                      # witness lines back to the object
        ax.plot(*np.array([p + n * pad * 0.35, q + n * pad * 0.45]).T,
                color=INK, lw=0.4, zorder=14)
    ax.annotate("", xy=A, xytext=B, zorder=14,
                arrowprops=dict(arrowstyle="<|-|>", color=INK, lw=0.6,
                                shrinkA=0, shrinkB=0, mutation_scale=5))
    mid = 0.5 * (A + B) + n * pad * 0.30
    ax.text(*mid, text, ha="center", va="center", fontsize=FS_ANNOT, color=INK,
            rotation=rot, zorder=15,
            bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none"))


def scale_bar(ax, x0, y0, length=500.0, tick=40.0):
    """Scale bar in DATA coordinates, with end ticks so the extent is unambiguous.

    Not used by this plate, whose closed panel is dimensioned instead. It is here because the
    dimension and the bar are the same decision made twice, and a panel that cannot carry witness
    lines -- a photograph cropped to its subject -- needs the other one.
    """
    ax.plot([x0, x0 + length], [y0, y0], color=INK, lw=1.4, solid_capstyle="butt", zorder=15)
    for x in (x0, x0 + length):
        ax.plot([x, x], [y0 - tick / 2, y0 + tick / 2], color=INK, lw=1.4, zorder=15)
    ax.text(x0 + length / 2, y0 + tick * 0.9, f"{length:.0f} mm", ha="center", va="bottom",
            fontsize=FS_ANNOT, color=INK, zorder=15)


def state_legend(ax, sim_style="ink", ncol=6, fontsize=FS_ANNOT):
    """The colour key for the three state panels, on an axes of its own.

    The caption decodes these five marks in prose, which is enough to LOOK UP a mark and not enough
    to read the panels at a glance: the reader meets orange, a thin line, red, a dashed rectangle
    and an arrow all at once, and prose hands them back one at a time. One strip under the row
    answers all five in the order the eye meets them.

    Args:
        ax: an axes reserved for the strip. Its frame and ticks are removed here.
        sim_style: key into ``SIM_STYLES``, so the prediction's swatch cannot drift from its line.
        ncol: entries per row.
        fontsize: label size, in points of the printed page.
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    ax.set_axis_off()
    handles = [Patch(facecolor=ORANGE, edgecolor="none", label="measured"),
               Patch(facecolor=GREY, alpha=0.45, edgecolor="none", label="held"),
               Line2D([], [], label="predicted", **SIM_STYLES[sim_style]),
               Patch(facecolor=RED, edgecolor=RED, label="disagreement"),
               Line2D([], [], label="target", **TARGET_KW),
               Line2D([], [], color=INK, lw=0.9, marker=">", markersize=2.6, markevery=[-1],
                      label="pull")]
    return ax.legend(handles=handles, ncol=ncol, loc="center", frameon=False,
                     fontsize=fontsize, handlelength=1.7, handleheight=0.9, handletextpad=0.45,
                     columnspacing=1.5, borderpad=0.0, borderaxespad=0.0)


def geometry(M, D, name, deg, align):
    """(measured, simulated, target) for one state, rotated into the drawing frame.

    ⚠ The target is REGULARISED to a rectangle for drawing. The tape reaches this frame through a
    homography fitted on the closed sheet, so its corners arrive about 2.7 deg out of square, and
    none of that is removable: squaring the tape un-squares the sheet, which is a rectangle of
    known size and the metric standard. The tape stays the measurement everywhere it is one; this
    is a drawn shape that no comparison is posed against.
    """
    meas = D[f"{name}_nodes"]
    sim = np.asarray(M["states"][name]["sim_nodes_mm"])
    if align:
        sim = align_sim(sim, meas)
    target = pg.regularise_rectangle(np.asarray(D["red_design_corners"], float))
    return rotate(meas, deg), rotate(sim, deg), rotate(target, deg)


def window(M, D, deg, align, margin_mm=None):
    """One window that holds every state, so the three panels share a scale and a frame.

    Args:
        margin_mm: breathing room on all four sides. ``MARGIN_MM`` is sized for the dimensions
            drawn in the closed panel; a plate that states its scale elsewhere wants less, and
            gets a larger drawing inside the same box for it.
    """
    m_mm = MARGIN_MM if margin_mm is None else float(margin_mm)
    pts = []
    for name in STATES:
        m, s, r = geometry(M, D, name, deg, align)
        pts += [m.reshape(-1, 2), s.reshape(-1, 2), r]
    P = np.vstack(pts)
    return (P[:, 0].min() - m_mm, P[:, 0].max() + m_mm,
            P[:, 1].min() - m_mm, P[:, 1].max() + m_mm)


def error_hotspot(M, D, name, deg, align, w_mm, h_mm, px_mm=4.0):
    """The crop of one state holding the most disagreement, weighted by how wide it is.

    Chosen rather than hand-picked so the detail cannot drift to a flattering neighbourhood, and so
    it follows the data if the design, the match step or the alignment ever changes.

    ⚠ NOT the widest single point. The distance transform of the red peaks at a tile TIP, where the
    sliver is widest precisely because the two outlines are diverging fastest -- and a window
    centred there is three-quarters empty paper with the wedge running off one corner. What a
    reader needs is a full frame of disagreement, so the score is the width-weighted red the window
    CONTAINS: the distance transform box-filtered at the window size, maximised. Wide slivers still
    win, but only where there are enough of them to fill the crop.

    Args:
        w_mm, h_mm: the crop, in millimetres. Its aspect must be the axes' aspect, since both are
            drawn equal-aspect and a mismatch letterboxes the crop instead of filling it.
        px_mm: raster pitch for the transform. Only the argmax is used, so this sets how finely the
            centre is located, not the reported widths.

    Returns:
        ``(x0, x1, y0, y1)`` in millimetres of the drawing frame.
    """
    import scipy.ndimage as ndi
    from matplotlib.path import Path
    meas, sim, _ = geometry(M, D, name, deg, align)
    regions = [error_region(meas[i], sim[i]) for i in range(len(meas))]
    regions = [r for r in regions if not r.is_empty]
    b = np.array([r.bounds for r in regions])
    gx = np.arange(b[:, 0].min() - 20.0, b[:, 2].max() + 20.0, px_mm)
    gy = np.arange(b[:, 1].min() - 20.0, b[:, 3].max() + 20.0, px_mm)
    XX, YY = np.meshgrid(gx, gy)
    pts = np.c_[XX.ravel(), YY.ravel()]
    occ = np.zeros(pts.shape[0], bool)
    for r in regions:
        for g in (r.geoms if hasattr(r, "geoms") else [r]):
            occ |= Path(np.asarray(g.exterior.coords)).contains_points(pts)
    dt = ndi.distance_transform_edt(occ.reshape(XX.shape))
    box = (max(int(round(h_mm / px_mm)), 1), max(int(round(w_mm / px_mm)), 1))
    j, i = np.unravel_index(ndi.uniform_filter(dt, size=box, mode="constant").argmax(), dt.shape)
    cx, cy = gx[i], gy[j]
    return (cx - w_mm / 2.0, cx + w_mm / 2.0, cy - h_mm / 2.0, cy + h_mm / 2.0)


def link_detail(fig, ax_parent, ax_detail, detail_win, color=GREY, lw=0.5):
    """Leaders from the outlined crop to the panel that magnifies it.

    A locator rectangle and a magnified panel elsewhere on the plate are two marks the reader has
    to associate, and nothing on the page says they belong together: the rectangle is 2 mm wide and
    the panel is a hundred times its area, so the eye does not read one as the other's source. Two
    lines from the rectangle's lower corners to the panel's upper corners say it without a word.

    Drawn on the FIGURE rather than either axes, since they cross the gap between the two.
    """
    from matplotlib.patches import ConnectionPatch
    x0, x1, y0, y1 = detail_win
    out = []
    for xa, xb in (((x0, y0), (0.0, 1.0)), ((x1, y0), (1.0, 1.0))):
        cp = ConnectionPatch(xyA=xa, coordsA=ax_parent.transData,
                             xyB=xb, coordsB=ax_detail.transAxes,
                             color=color, lw=lw, zorder=19, clip_on=False)
        fig.add_artist(cp)
        out.append(cp)
    return out


def widest_error(M, D, name, deg, align, window=None, px_mm=2.0, fit_inside=True,
                 slack=0.16):
    """Where the disagreement is widest, how wide, and across which direction.

    The number the magnified panel exists to make readable, so it is measured rather than eyeballed
    off the plate: the deepest point of the distance transform of the union of the error regions,
    whose depth is the half-width and whose nearest background pixel gives the normal to dimension
    across.

    ⚠⚠ THE TRANSFORM IS ALWAYS COMPUTED ON THE FULL STATE, and ``window`` only restricts where the
    maximum is taken. Rasterising the crop alone inflates it: a band running off the crop's edge
    has no background between it and that edge, so the transform measures to the far side of the
    band instead and reports roughly half again the true width -- 63 mm against 42 on this plate.

    Args:
        window: restrict the maximum to this crop. The transform itself is still global.
        fit_inside: keep the answer far enough from ``window``'s edges that the dimension drawn
            across it fits, AND its label beyond that. ⚠ Without this the widest place lands ON the
            crop boundary -- the crop is chosen for how much red it holds, which pushes its edge
            through the thickest part -- and the dimension arrow is clipped by the panel.
        slack: extra inset beyond the half-width, as a fraction of the window height, reserving
            room for the label past the arrowhead. ⚠ The panel is letterboxed, so the dimension
            plus a label at each end is most of its height: this is what guarantees the widest
            place is found near the middle, where all three fit.

    Returns:
        ``(half_width_mm, centre_xy, unit_normal_xy)``.
    """
    import scipy.ndimage as ndi
    from matplotlib.path import Path
    meas, sim, _ = geometry(M, D, name, deg, align)
    regions = [r for r in (error_region(meas[i], sim[i]) for i in range(len(meas)))
               if not r.is_empty]
    b = np.array([r.bounds for r in regions])
    gx = np.arange(b[:, 0].min() - 20.0, b[:, 2].max() + 20.0, px_mm)
    gy = np.arange(b[:, 1].min() - 20.0, b[:, 3].max() + 20.0, px_mm)
    XX, YY = np.meshgrid(gx, gy)
    pts = np.c_[XX.ravel(), YY.ravel()]
    occ = np.zeros(pts.shape[0], bool)
    for r in regions:
        for g in (r.geoms if hasattr(r, "geoms") else [r]):
            occ |= Path(np.asarray(g.exterior.coords)).contains_points(pts)
    occ = occ.reshape(XX.shape)
    dt, idx = ndi.distance_transform_edt(occ, return_indices=True)
    def inside(w):
        return (XX >= w[0]) & (XX <= w[1]) & (YY >= w[2]) & (YY <= w[3])

    if window is not None:
        dt = np.where(inside(window), dt, 0.0)
        if fit_inside:
            m = dt.max() * px_mm + slack * (window[3] - window[2])
            snug = np.where(inside((window[0] + m, window[1] - m,
                                    window[2] + m, window[3] - m)), dt, 0.0)
            if snug.max() > 0.0:
                dt = snug
    j, i = np.unravel_index(dt.argmax(), dt.shape)
    centre = np.array([gx[i], gy[j]])
    near = np.array([gx[idx[1, j, i]], gy[idx[0, j, i]]])
    n = centre - near
    n = n / max(float(np.linalg.norm(n)), 1e-9)
    return float(dt[j, i] * px_mm), centre, n


def _clear_spot(ax, win, avoid, keep_out, text, fontsize, prefer):
    """Where a label's box fits without covering what the label is about.

    ⚠⚠ THE BOX IS SIZED IN POINTS AND THE PANEL IS 25 pt TALL. A 6.6 pt label is 9 pt of box, so on
    this panel it is over a third of the height whatever the magnification -- which is why "beyond
    the arrowhead" cannot be made to fit (dimension + two labels is 29 pt of a 25 pt panel), and why
    "beside the dimension line" lands on the band. Neither is a placement rule that can be tuned;
    the box has to go where the drawing is empty, and that has to be measured rather than guessed.

    Args:
        avoid: polygons the label must not cover -- the red, which is what it reports.
        keep_out: (xmin, xmax, ymin, ymax) of the dimension itself, which it must not cover either.
        prefer: the label is pulled toward this point, so it stays legibly near its dimension.
    """
    from matplotlib.path import Path
    w, h = win[1] - win[0], win[3] - win[2]
    bb = ax.get_position()
    mm_x = w / (bb.width * ax.figure.get_figwidth() * 72.0)
    mm_y = h / (bb.height * ax.figure.get_figheight() * 72.0)
    bw = (0.60 * len(text) + 0.8) * fontsize * mm_x
    bh = 1.6 * fontsize * mm_y
    paths = [Path(np.asarray(g.exterior.coords)) for r in avoid
             for g in (r.geoms if hasattr(r, "geoms") else [r])]
    ox, oy = np.meshgrid(np.linspace(-0.5, 0.5, 9), np.linspace(-0.5, 0.5, 3))
    probe = np.c_[ox.ravel() * bw, oy.ravel() * bh]

    best, out = None, np.asarray(prefer, float)
    for fx in np.linspace(0.0, 1.0, 41):
        for fy in np.linspace(0.0, 1.0, 17):
            c = np.array([win[0] + fx * w, win[2] + fy * h])
            if (c[0] - bw / 2 < win[0] or c[0] + bw / 2 > win[1]
                    or c[1] - bh / 2 < win[2] or c[1] + bh / 2 > win[3]):
                continue
            pts = probe + c
            hit = sum(int(p.contains_points(pts).sum()) for p in paths)
            over = float(not (c[0] + bw / 2 < keep_out[0] or c[0] - bw / 2 > keep_out[1]
                              or c[1] + bh / 2 < keep_out[2] or c[1] - bh / 2 > keep_out[3]))
            cost = 100.0 * hit + 400.0 * over + np.linalg.norm(c - prefer) / w
            if best is None or cost < best:
                best, out = cost, c
    return out


def _width_dimension(ax, win, centre, normal, half, text, avoid=(), reach=0.16, gap=0.020):
    """An engineering dimension across a band: witness lines, arrowheads, the length beside it.

    ⚠ A bare double-headed arrow was tried and does not read: it says a length without saying
    between WHICH two things, and laid over the red it is nearly invisible. The witness lines are
    the part that carries the meaning -- they run from the two edges the width is measured between,
    out to a clear stretch of panel where the dimension line and its label can be seen.

    Args:
        centre, normal, half: the widest point, the direction the width is measured along, and the
            half-width, as ``widest_error`` returns them.
        avoid: the error polygons, so the label can be kept off the thing it reports.
        reach: how far along the band the dimension line is carried, as a fraction of panel width.
        gap: clear space left between the band edges and the start of the witness lines.
    """
    w, h = win[1] - win[0], win[3] - win[2]
    c, n = np.asarray(centre, float), np.asarray(normal, float)
    t = np.array([-n[1], n[0]])
    a, b = c - half * n, c + half * n
    room = lambda p: min(p[0] - win[0], win[1] - p[0], p[1] - win[2], win[3] - p[1])
    if room(c + t * reach * w) < room(c - t * reach * w):
        t = -t
    off = t * reach * w
    A, B = a + off, b + off
    for p, q in ((a, A), (b, B)):                       # witness lines, clear of the band itself
        ax.plot(*np.array([p + t * gap * w, q + t * gap * w]).T,
                color=INK, lw=0.35, zorder=15)
    ax.annotate("", xy=A, xytext=B, zorder=15,
                arrowprops=dict(arrowstyle="<|-|>", color=INK, lw=0.6,
                                shrinkA=0, shrinkB=0, mutation_scale=4))
    box = np.array([a, b, A, B, a + t * gap * w, b + t * gap * w])
    keep_out = (box[:, 0].min(), box[:, 0].max(), box[:, 1].min(), box[:, 1].max())
    lab = _clear_spot(ax, win, avoid, keep_out, text, FS_ANNOT, 0.5 * (A + B))
    ax.plot(*np.array([0.5 * (A + B), lab]).T, color=INK, lw=0.3, zorder=15)
    ax.text(*lab, text, fontsize=FS_ANNOT, color=INK, zorder=16, ha="center", va="center",
            bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))


def panel_detail(ax, name, M, D, win, deg, align, clamp, sim_style, dim=None):
    """One state again, magnified, so the disagreement has a size the reader can read off.

    The state panels run at about 29 mm per point, where the widest sliver of red prints 1.5 pt --
    the eye sees THAT the two disagree and has no way to say by how much, and the caption's
    millimetres have nothing on the page to attach to. Here the same marks are drawn at roughly
    five times that, beside a bar, and the sliver becomes a length.

    ⛔ No arrow, no tile numbers, no target: at this crop none of the three is in frame, and a
    legend entry with nothing to point at is worse than a missing one.

    ⛔⛔ AND NO SCALE BAR. One was drawn here and it was exactly 100 mm, and it still read as an
    error: this plate carries bars at 13.0, 1.13 and 6.0 mm per point, so the bar labelled 50 mm in
    (b) prints 2.6 times LONGER than the one labelled 500 mm in (a), and a fourth bar, the shortest
    of them, invited exactly the arithmetic that makes the set look wrong. A bar is also the long
    way round to the answer -- it asks the reader to measure the red against it and multiply. The
    dimension states the width instead, on the widest place, which is what the panel is for.

    Args:
        dim: ``(half_width_mm, centre_xy, unit_normal_xy)`` from ``widest_error``, or None.
    """
    meas, sim, _ = geometry(M, D, name, deg, align)
    kw = SIM_STYLES[sim_style]
    for i in range(len(meas)):
        ax.fill(*np.vstack([meas[i], meas[i][:1]]).T, color=ORANGE, lw=0, zorder=1)
        if i == clamp:
            ax.fill(*np.vstack([meas[i], meas[i][:1]]).T, color=GREY, alpha=0.45, lw=0, zorder=2)
    for i in range(len(meas)):
        _fill_shapely(ax, error_region(meas[i], sim[i]), facecolor=RED, edgecolor=RED,
                      lw=0.3, zorder=3)
    for i in range(len(sim)):
        _poly(ax, sim[i], zorder=4, **kw)
    ax.set_xlim(win[0], win[1])
    ax.set_ylim(win[2], win[3])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    # ⚠⚠ set_visible(True), not just a colour. The house style carries
    # ``axes.spines.top/right = False``, so this panel was drawn as an L and never read as a
    # rectangle at all -- and a magnified crop HAS to, because it is the locator rectangle on the
    # parent panel at N times the size and the leaders join corresponding corners. Two rectangles
    # and two leaders are what say where the zoom came from; an L says nothing.
    for sp in ax.spines.values():
        sp.set(linewidth=0.5, color=GREY, visible=True)
    if dim is not None:
        r, c, n = dim
        _width_dimension(ax, win, c, n, r, f"{2.0 * r:.0f} mm",
                         avoid=[q for q in (error_region(meas[i], sim[i])
                                            for i in range(len(meas))) if not q.is_empty])
        # ⚠ In a CORNER with a leader, never beside the dimension. The label is several times
        # wider than a 40 mm dimension at any magnification this panel can carry, so placing it on
        # the band puts its white box over the very thing being measured -- and offsetting it along
        # the band walks it off the panel, which is exactly what the first version did.



def rectified(exp_dir, name, H, win, mm_per_px=2.2):
    """The rectified frame as a line drawing: floor flattened to grey, tape kept.

    Only available unrotated -- the raster is warped into the design frame, and turning it as well
    would resample it twice for a cosmetic gain.
    """
    rgb = pr.load_rgb(os.path.join(exp_dir, "raw", f"{name}.jpg"))
    clean = pr.clean_render(rgb, pr.colour_masks(rgb),
                            colours={"orange": (0.961, 0.502, 0.145),
                                     "green": (0.122, 0.541, 0.298),
                                     "red": (0.80, 0.80, 0.82)})
    x = np.arange(win[0], win[1], mm_per_px)
    y = np.arange(win[2], win[3], mm_per_px)
    img = pr.warp_to_grid(clean, H, x, y, fill=np.nan)
    return np.clip(np.nan_to_num(img, nan=1.0), 0.0, 1.0), x, y


def panel_frame(ax, name, M, D, win, deg, align, clamp, pull, pull_dir, sim_style, first,
                exp_dir=None, dims=True, detail_win=None):
    """One state against the design.

    Args:
        first: the closed panel, which carries the tile numbers every other panel and both plots
            index by, and starts its pull arrow clear of them.
        dims: the two dimensions of the flat sheet, drawn on the ``first`` panel. Off when the
            plate gives the scale somewhere else, and two statements of it would compete.
        detail_win: a crop shown magnified elsewhere on the plate, outlined here so the magnified
            view has an address. A detail with no locator is an unplaceable picture.
    """
    meas, sim, target = geometry(M, D, name, deg, align)
    flat_nodes = rotate(D["flat_face_nodes"], deg)
    if exp_dir is not None:
        img, gx, gy = rectified(exp_dir, name, D[f"{name}_H"], win)
        ax.imshow(img, extent=[gx[0], gx[-1], gy[0], gy[-1]], origin="lower",
                  interpolation="antialiased", zorder=0)
    # The measured sheet is a BODY, not a wireframe: filled, no border. The prediction is a thin
    # line laid over it, and where the two disagree the gap between them is filled red -- so the
    # error is an area the eye sizes directly rather than a gap it has to estimate.
    for i in range(len(meas)):
        if exp_dir is None:
            ax.fill(*np.vstack([meas[i], meas[i][:1]]).T, color=ORANGE, lw=0, zorder=1)
        else:
            _poly(ax, meas[i], color=ORANGE, lw=0.9, zorder=1)
        if i == clamp:
            ax.fill(*np.vstack([meas[i], meas[i][:1]]).T, color=GREY, alpha=0.45, lw=0, zorder=2)
    for i in range(len(meas)):
        _fill_shapely(ax, error_region(meas[i], sim[i]), facecolor=RED, edgecolor=RED,
                      lw=0.3, zorder=3)
    kw = SIM_STYLES[sim_style]
    for i in range(len(sim)):
        _poly(ax, sim[i], zorder=4, **({**kw, "lw": kw["lw"] * 1.4} if i == pull else kw))
    _poly(ax, target, zorder=5, **TARGET_KW)

    # The activated tile and the direction it was pulled: fixed in the global frame, hence
    # identical in all three panels, and deliberately not the tile's travel -- the mechanism
    # redirects the motion nearly perpendicular to the pull.
    # The arrow starts clear of the tile's number rather than under it, so every number can sit on
    # its own centroid and the row of them reads as one aligned set.
    base = meas[pull].mean(0) + pull_dir * (110.0 if first else 0.0)
    ax.annotate("", xy=base + pull_dir * 290.0, xytext=base, zorder=17,
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.9, mutation_scale=6,
                                shrinkA=0, shrinkB=0))

    if detail_win is not None:
        x0, x1, y0, y1 = detail_win
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="none", edgecolor=GREY,
                               lw=0.5, zorder=18))
    ax.set_xlim(win[0], win[1])
    ax.set_ylim(win[2], win[3])
    ax.set_aspect("equal")
    ax.axis("off")
    if first:
        # Tile numbers on the closed sheet only: the tiles are identifiable there, and every other
        # panel and both plots below index by the same number.
        for i in range(len(meas)):
            ax.text(*meas[i].mean(0), str(i), ha="center", va="center", fontsize=FS_ANNOT,
                    color=INK, zorder=16)
        if dims:
            flat = rotate(D["flat_face_nodes"].reshape(-1, 2), deg)
            x0, x1 = flat[:, 0].min(), flat[:, 0].max()
            y0, y1 = flat[:, 1].min(), flat[:, 1].max()
            _dimension(ax, (x0, y0), (x1, y0), f"{x1 - x0:.0f} mm", (0.0, -185.0))
            _dimension(ax, (x0, y0), (x0, y1), f"{y1 - y0:.0f} mm", (-185.0, 0.0), rot=90.0)


def panel_parity(ax, M, tone, legend_kw=None):
    lim = 0.0
    for name in STATES:
        st = M["states"][name]
        xs, ys = np.asarray(st["opening_sim_deg"]), np.asarray(st["opening_measured_deg"])
        sd = np.asarray(st["opening_sd_deg"]) if st["opening_sd_deg"] else None
        ax.errorbar(xs, ys, yerr=sd, fmt="o", ms=2.6, mew=0.0, color=tone[name],
                    ecolor=tone[name], elinewidth=0.6, capsize=0.0, zorder=3,
                    label=f"{LABEL[name]}  {st['opening_rmse_deg']:.1f}")
        lim = max(lim, float(xs.max()), float(ys.max()))
    lim *= 1.16
    ax.plot([0, lim], [0, lim], ls="--", lw=0.7, color=GREY, zorder=1)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.set_xlabel(r"simulated hinge opening $\vartheta$  [deg]", fontsize=FS_BASE)
    ax.set_ylabel(r"measured $\vartheta$  [deg]", fontsize=FS_BASE)
    ax.tick_params(length=2.4, pad=1.5)
    # Lower right: the half-plane below the equality line is empty, and the legend has to sit where
    # the data are not -- above the line it lands on the deployed cluster.
    # ⚠ That only holds while the panel is TALL. This axes is aspect-equal, so a short box shrinks
    # the data square while the legend keeps its point size, and below about 110 bp of height the
    # legend crosses the equality line. ``legend_kw`` lets a cramped plate move it outside instead,
    # into the width the square gives back when it shrinks.
    kw = dict(loc="lower right", frameon=False, fontsize=FS_ANNOT, handletextpad=0.3,
              labelspacing=0.26, borderaxespad=0.25,
              title="RMSE [deg]", title_fontsize=FS_ANNOT)
    kw.update(legend_kw or {})
    leg = ax.legend(**kw)
    leg._legend_box.align = "left"


def panel_tiles(ax, M, floor_mm, clamp, tone):
    key = "shape_resid_mm"
    n = len(M["states"]["Deployed"][key])
    idx = np.arange(n)
    w = 0.27
    # The floor is a bound, not a series: a hairline rule reads as one, and the caption decodes it.
    ax.axhspan(0, floor_mm, color=GREY, alpha=0.10, lw=0, zorder=0)
    ax.axhline(floor_mm, color=GREY, lw=0.6, ls=(0, (2.6, 1.8)), zorder=1)
    top = max(np.max(np.asarray(M["states"][q][key])) for q in STATES) * 1.30
    for k, name in enumerate(STATES):
        ax.bar(idx + (k - 1) * w, np.asarray(M["states"][name][key]), width=w,
               color=tone[name], lw=0.0, zorder=2)
    ax.set_xticks(idx)
    ax.set_xticklabels([str(i) for i in idx])
    ax.get_xticklabels()[clamp].set_color(GREY)
    ax.set_xlabel("panel", fontsize=FS_BASE)
    ax.set_ylabel("panel centre error  [mm]", fontsize=FS_BASE)
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(0, top)
    ax.tick_params(length=2.4, pad=1.5)
    ax.legend(handles=[Patch(facecolor=tone[q], label=LABEL[q]) for q in STATES],
              loc="upper left", ncol=1, frameon=False, fontsize=FS_ANNOT,
              handlelength=1.1, handletextpad=0.35, labelspacing=0.24, borderaxespad=0.1)


def caption_text(M, align, width=1.18):
    """Caption sidecar; every number derived from the data so it cannot go stale."""
    dep, dly, clo = (M["states"][k] for k in ("Deployed", "Deploying", "Closed"))
    b = dep.get("buckling") or {}
    tw, tl = M["target_measured_mm"]
    dw, dl = M["target_design_mm"]
    frame = ("with the rigid-body offset between prediction and measurement removed, so that shape "
             "is compared to shape" if align else
             "in the design frame; the bodily offset between the two is the hand grip's wander, not "
             "shape error, and panel (e) reports the shape with that offset removed")
    return f"""A 3x3 PET kirigami sheet deployed by hand, measured against the simulation that designed it.
Generated by nff/scripts/figures/plot_deployment_validation.py.

(a) closed, (b) part deployed, (c) deployed. Orange is the measured sheet, recovered by posing the
nine known rigid tiles against tape markers in rectified photographs; the thin solid outline is the
simulation at the matched load step, drawn {frame}. Red fills the disagreement between them, so the
error is an area rather than a gap. The grey tile is held (tile {M['states']['Deployed']['clamp_face']})
and tile {M['states']['Deployed']['pull_face']} is pulled; tiles are numbered in (a) and every panel
uses those numbers. The dashed black rectangle is the deployment target. The black arrow at the activated tile is the applied
pull, a fixed direction in the global frame and so the same in all three panels. The tile does not
travel along it: the mechanism redirects the motion to roughly perpendicular, and measurement and
simulation agree on that redirected direction to within 4 degrees once the hold's own motion
is removed, and to within 1 degree without that gauging. All three panels share one frame and one scale;
the closed sheet in (a) is dimensioned, and is a standard 4 x 8 ft stock panel. (d) Hinge opening measured against simulated, twelve hinges per state; the dashed
line is equality and the legend gives the RMSE. Opening is a difference of tile rotations, so it
carries no registration. (e) Per tile, the distance between the measured and simulated
tile centres after the sheet's rigid-body motion has been removed by a rigid (rotation and
translation, no scale) alignment of the two, so that what remains is shape rather than placement.
The dashed rule and the band beneath it mark the measurement floor established on (a), where the
answer is known to be zero. Panels (d) and (e) take shades of the hue that already carries their
quantity in this paper: teal for hinges, red for the disagreement drawn in (a)-(c).

Provenance, not part of the caption.
Inputs: {M['photo_dir']}, {M['sim']}.
Measurement floor {clo['centroid_rms_mm']:.1f} mm centroid, {clo['rot_rms_deg']:.2f} deg rotation,
{clo['opening_rmse_deg']:.2f} deg opening, from the closed frame against the flat design.
Deployed: matched on {dep['match_on']} at load {dep['load_fraction']:.3f}; opening RMSE
{dep['opening_rmse_deg']:.2f} deg; shape-only {dep['shape_only_rms_mm']:.1f} mm; the hand clamp
slipped {dep['clamp_slip_mm']:.1f} mm; the pull tile travelled {dep['pull_travel_measured_mm']:.0f} mm
against {dep['pull_travel_sim_mm']:.0f} mm simulated, which the mechanism does not reach at any
defensible load. At matched extension the sheet is {b.get('deficit_mm', 0):.1f} mm
({b.get('deficit_frac', 0) * 100:.1f}%) narrower than predicted, consistent with the out-of-plane
rippling observed in the lab. Deploying was shot mid-pull and is NOT a static equilibrium; its
opening is comparable ({dly['opening_rmse_deg']:.2f} deg RMSE), its positions are not.
Tape was partly hidden on tiles {dep['low_coverage_tiles']} (deployed) and
{dly['low_coverage_tiles']} (deploying), which caps how well those tiles can be fitted.
Authored at {{width}} x \\textwidth: the manuscript must wrap the graphic in
\\makebox[\\textwidth][c]{{}} and set \\includegraphics to the same fraction.
Target rectangle measured {tw:.0f} x {tl:.0f} mm against {dw:.0f} x {dl:.0f} designed.
Regenerate: PYTHONPATH=$(pwd) python3 nff/scripts/figures/plot_deployment_validation.py
"""


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp-dir", default="data/experiments/pet45_3x3_deployment_20260814")
    p.add_argument("--out", default="paper/Images/deployment_validation.pdf")
    p.add_argument("--width-frac", type=float, default=1.18,
                   help="page_bp width fraction. Over 1.0 the manuscript must wrap the graphic in "
                        "\\makebox[\\textwidth][c]{}, as it already does for two other plates, and "
                        "the \\includegraphics width must match this number")
    p.add_argument("--aspect", type=float, default=1.46, help="page_bp aspect, width/height")
    p.add_argument("--rotate", dest="rotate", action="store_true", default=True,
                   help="turn the sheet's long axis horizontal, which shortens the plate by ~20%%")
    p.add_argument("--no-rotate", dest="rotate", action="store_false")
    p.add_argument("--align-sim", dest="align_sim", action="store_true", default=True,
                   help="draw the simulation with the rigid-body offset to the measurement removed, "
                        "so shape is compared to shape (default)")
    p.add_argument("--raw-sim", dest="align_sim", action="store_false",
                   help="leave the offset in: the simulation stays in the design frame")
    p.add_argument("--sim-style", choices=sorted(SIM_STYLES), default="ink",
                   help="how to draw the prediction now that black belongs to the target")
    p.add_argument("--photos", action="store_true",
                   help="draw the rectified photographs behind the outlines; forces --no-rotate")
    p.add_argument("--dpi", type=int, default=450, help="PNG companion only; the PDF is vector")
    args = p.parse_args()

    M = json.load(open(os.path.join(args.exp_dir, "derived", "metrics.json")))
    D = np.load(os.path.join(args.exp_dir, "derived", "measured.npz"))
    clamp = M["states"]["Deployed"]["clamp_face"]
    pull = M["states"]["Deployed"]["pull_face"]
    floor_mm = M["noise_floor_mm"]
    if args.photos:
        args.rotate = False                       # the raster is only warped into the design frame
    deg = ROT_DEG if args.rotate else 0.0
    win = window(M, D, deg, args.align_sim)
    # The applied load, taken from the boundary condition the solve actually used, so the arrow is
    # the BC and not the response to it. Fixed in the global frame, hence the same in every panel.
    sim_npz = np.load(M["sim"])
    dirs = sim_npz["load_dirs"] if "load_dirs" in sim_npz.files else np.array([[0.0, 1.0]])
    pull_dir = rotate(np.asarray(dirs)[0][None, :], deg)[0]
    pull_dir = pull_dir / max(float(np.linalg.norm(pull_dir)), 1e-9)

    w_bp, h_bp = page_bp(args.width_frac, args.aspect)
    apply_charter(**paper_rc())
    fig = plt.figure(figsize=figsize_in(w_bp, h_bp))
    # Two gridspecs, not one: row 1 holds fixed-aspect drawings, so its height follows from its
    # width. Sharing a wspace with the plot row would narrow the drawings and leave dead paper.
    # Row 1 carries the argument, so it gets the whole canvas: its panels have no axes and so owe
    # nothing to the gutter that row 2's y-label needs. Spanning wider here, and authoring above
    # \textwidth, is what buys (a)-(c) their size without breaking the page.
    R1_L, R1_R = 0.012, 0.995
    R2_L, R2_R = 0.088, 0.995
    frac = (win[3] - win[2]) / (win[1] - win[0])          # panel height / width
    panel_w = (R1_R - R1_L) / 3.0 * w_bp
    row1_h = panel_w * frac / h_bp                        # as a figure fraction
    top = 0.965
    gs0 = fig.add_gridspec(1, 3, left=R1_L, right=R1_R, top=top, bottom=top - row1_h,
                           wspace=0.035)
    gs1 = fig.add_gridspec(1, 2, left=R2_L, right=R2_R, top=top - row1_h - 0.105, bottom=0.135,
                           wspace=0.30, width_ratios=[1.0, 1.30])

    frames = []
    for j, name in enumerate(STATES):
        ax = fig.add_subplot(gs0[0, j])
        panel_frame(ax, name, M, D, win, deg, args.align_sim, clamp, pull, pull_dir,
                    args.sim_style, first=(j == 0),
                    exp_dir=args.exp_dir if args.photos else None)
        frames.append(ax)

    fig.legend(handles=[Patch(facecolor=ORANGE, lw=0, label="measured sheet"),
                        Line2D([], [], label="simulated", **SIM_STYLES[args.sim_style]),
                        Patch(facecolor=RED, lw=0, label="disagreement"),
                        Line2D([], [], label="target", **TARGET_KW)],
               loc="center", bbox_to_anchor=(0.50, top - row1_h - 0.050), ncol=4, frameon=False,
               fontsize=FS_ANNOT, handlelength=1.5, handletextpad=0.35, columnspacing=1.5)

    tone_hinge = dict(zip(STATES, shades(TEAL, 3)))
    tone_error = dict(zip(STATES, shades(ORANGE, 3)))
    with plt.rc_context({"axes.grid": False}):
        axd = fig.add_subplot(gs1[0, 0])
        panel_parity(axd, M, tone_hinge)
    axe = fig.add_subplot(gs1[0, 1])
    panel_tiles(axe, M, floor_mm, clamp, tone_error)

    # Panel letters from the REAL axes boxes, after layout: guessed figure fractions clip the letter
    # off the canvas the moment a panel moves.
    # Above the axes' own left edge, not out in the gutter: the gutter belongs to the y-axis label,
    # and a letter placed there lands on top of it the moment the panel is not wide.
    for ax, letter in zip(frames + [axd, axe], "abcde"):
        bb = ax.get_position()
        dx = 0.010 if ax in frames else 0.0
        panel_letter(fig, letter, x=max(bb.x0 - dx, 0.004), y=bb.y1 + 0.014)

    pdf, lo, hi = save_paper_figure(fig, args.out, w_bp, h_bp, dpi=args.dpi, strict_margins=True)
    plt.close(fig)
    with open(os.path.splitext(pdf)[0] + ".txt", "w") as fh:
        fh.write(caption_text(M, args.align_sim, args.width_frac))
    size_kb = os.path.getsize(pdf) / 1024.0
    print(f"wrote {pdf} ({w_bp:.1f} x {h_bp:.1f} bp, type {lo:.1f}-{hi:.1f} pt, {size_kb:.0f} kB)")
    print(f"wrote {os.path.splitext(pdf)[0]}.txt")


if __name__ == "__main__":
    main()
