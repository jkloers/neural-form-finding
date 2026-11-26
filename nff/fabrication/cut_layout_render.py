"""Printable workshop sheets for a cut-pattern layout: A3 overview, A4 tables, A4 1:1 detail.

Deliberately not a paper figure. Everything is laid out in page millimetres so a stated drawing
scale is exact, the type is sized to be read at arm's length on a bench, and the palette is plain
black on white with red reserved for the two things that ruin a sheet -- the scaling caveat and the
grip marks.

Consumes the schedule from ``nff.fabrication.cut_layout`` and the geometry from
``nff.fabrication.cut_pattern``; writes PDFs.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Polygon, PathPatch
from matplotlib.path import Path
from shapely import affinity
from shapely.geometry import box as shapely_box

_A4_W, _A4_H = 210.0, 297.0
_A3_W, _A3_H = 297.0, 420.0

_INK = "#111111"
_GREY = "#9A9A9A"
_FAINT = "#DFDFDF"
_RED = "#C1121F"
_BLUE = "#245C9E"
_MONO = "DejaVu Sans Mono"


def _page(w_mm, h_mm):
    """Blank page whose data coordinates ARE page millimetres."""
    fig = plt.figure(figsize=(w_mm / 25.4, h_mm / 25.4), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w_mm)
    ax.set_ylim(0, h_mm)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def _frame(ax, x, y, w, h, *, lw=0.6, color=_INK, fc="none"):
    ax.add_patch(Rectangle((x, y), w, h, fill=fc != "none", facecolor=fc, edgecolor=color, lw=lw))


def _tick_values(span, step):
    return np.arange(0.0, span + 1e-6, step)


# ── A3 overview ────────────────────────────────────────────────────────────────────

def render_overview_a3(schedule, geom, out_pdf, *, draw_scale=8.0, meta=None, grid_mm=100.0,
                       label_points=True):
    """Overview drawing: the whole stock sheet at ``1:draw_scale`` with every mark identified.

    The drawing locates the marks; the coordinate tables carry the numbers. Ruler edges every
    ``grid_mm`` let a table value be sanity-checked against the picture before it is trusted.

    Args:
        schedule: output of ``build_layout_schedule``.
        geom: the matching ``build_cut_geometry`` dict (same scale, design frame).
        draw_scale: N in 1:N. 8 puts a 1219 x 2438 mm stock inside an A3 with room for the notes.
        meta: title-block rows as ``(label, value)`` pairs.
        grid_mm: reference grid / tick pitch [mm real].
        label_points: draw point IDs (turn off if the pattern is dense enough to overlap).
    """
    W, H = schedule["stock"]
    s = 1.0 / float(draw_scale)
    dw, dh = W * s, H * s

    fig, ax = _page(_A3_W, _A3_H)
    ox, oy = 46.0, 52.0                                    # page position of the datum corner
    right_x = ox + dw + 14.0                               # notes column
    right_w = _A3_W - right_x - 10.0

    def P(x, y):                                           # real mm (datum frame) -> page mm
        return ox + x * s, oy + y * s

    # ── reference grid + stock outline ──
    for gx in _tick_values(W, grid_mm):
        ax.plot(*zip(P(gx, 0), P(gx, H)), color=_FAINT, lw=0.3, zorder=0)
    for gy in _tick_values(H, grid_mm):
        ax.plot(*zip(P(0, gy), P(W, gy)), color=_FAINT, lw=0.3, zorder=0)
    _frame(ax, ox, oy, dw, dh, lw=1.2)

    # ── rulers: ticks every grid_mm, values every other one ──
    for i, gx in enumerate(_tick_values(W, grid_mm)):
        px, py = P(gx, 0)
        ax.plot([px, px], [oy - (2.6 if i % 2 == 0 else 1.4), oy], color=_INK, lw=0.5)
        if i % 2 == 0:
            ax.text(px, oy - 3.4, f"{gx:.0f}", ha="center", va="top", fontsize=4.6, color=_INK)
    for i, gy in enumerate(_tick_values(H, grid_mm)):
        px, py = P(0, gy)
        ax.plot([ox - (2.6 if i % 2 == 0 else 1.4), ox], [py, py], color=_INK, lw=0.5)
        if i % 2 == 0:
            ax.text(ox - 3.4, py, f"{gy:.0f}", ha="right", va="center", fontsize=4.6, color=_INK)
    ax.text(ox + dw / 2, oy - 8.5, "x  [mm from datum]", ha="center", va="top", fontsize=6,
            color=_INK)
    ax.text(ox - 9.5, oy + dh / 2, "y  [mm from datum]", ha="center", va="center", fontsize=6,
            color=_INK, rotation=90)

    # ── datum symbol ──
    ax.add_patch(Polygon([[ox, oy], [ox - 3.2, oy - 5.4], [ox + 3.2, oy - 5.4]],
                         closed=True, facecolor=_INK, edgecolor="none", zorder=6))
    ax.text(ox - 5.5, oy - 7.0, "DATUM  (0, 0)", ha="left", va="top", fontsize=5.4,
            color=_INK, fontweight="bold")

    # Type scales with how crowded the sheet is: a 3x3 carries 16 marks where a 5x5 carries 48, and
    # sizing both for the dense case wastes the legibility the sparse one could have.
    dense = len(schedule["points"]) > 30
    id_size, tile_size = (4.0, 4.2) if dense else (5.6, 6.0)

    # ── tile numbers, faint, so a grip instruction can name a tile ──
    for t in schedule.get("tiles", []):
        px, py = P(t["x"], t["y"])
        ax.text(px, py, str(t["face"]), ha="center", va="center", fontsize=tile_size, color=_GREY,
                zorder=2)

    # ── the cuts: drawn line (thin) + the material actually removed (thick) ──
    for c in schedule["cuts"]:
        ax.plot(*zip(P(c["x_from"], c["y_from"]), P(c["x_to"], c["y_to"])),
                color=_GREY, lw=0.25, ls=(0, (3, 2)), zorder=2)
        ax.plot(*zip(P(c["x_cut_from"], c["y_cut_from"]), P(c["x_cut_to"], c["y_cut_to"])),
                color=_INK, lw=0.9, solid_capstyle="butt", zorder=4)

    # ── relief holes, drawn oversize: at true scale a 4 mm hole is 0.5 mm on an A3 and vanishes ──
    for hh in schedule["holes"]:
        px, py = P(hh["x"], hh["y"])
        ax.add_patch(Circle((px, py), 0.6, facecolor="white", edgecolor=_INK, lw=0.35, zorder=5))

    # ── marking points ──
    r_pt = 0.9 if dense else 1.2
    for p in schedule["points"]:
        px, py = P(p["x"], p["y"])
        ax.add_patch(Circle((px, py), r_pt, facecolor="white", edgecolor=_BLUE, lw=0.5, zorder=6))
        ax.plot([px], [py], marker="+", ms=1.8, mew=0.4, color=_BLUE, zorder=7)
        if label_points:
            ax.text(px + r_pt + 0.5, py + 0.8, p["id"], ha="left", va="bottom", fontsize=id_size,
                    color=_BLUE, zorder=8)

    # ── grip marks ──
    for a in schedule.get("activation", []):
        px, py = P(a["x"], a["y"])
        col = _INK if a["role"] == "HOLD" else _RED
        ax.add_patch(Circle((px, py), 6.0, facecolor=col, edgecolor="none", alpha=0.16, zorder=9))
        ax.add_patch(Circle((px, py), 6.0, facecolor="none", edgecolor=col, lw=0.8, zorder=9))
        ax.text(px, py + 1.1, a["role"], ha="center", va="center", fontsize=5.4, color=col,
                fontweight="bold", zorder=10)
        ax.text(px, py - 2.0, f"tile {a['face']}", ha="center", va="center", fontsize=4.2,
                color=col, zorder=10)

    _overview_notes(ax, schedule, right_x, right_w, oy, dh, draw_scale, meta)
    _title_block(ax, schedule, 10.0, _A3_H - 34.0, _A3_W - 20.0, 24.0, draw_scale, meta)
    fig.savefig(out_pdf)
    plt.close(fig)
    return out_pdf


def _overview_notes(ax, schedule, x, w, oy, dh, draw_scale, meta):
    """Legend, scaling caveat, pre-cut checks and the summary, stacked down the notes column."""
    y = oy + dh
    sm = schedule["summary"]

    def head(text, yy):
        ax.text(x, yy, text, ha="left", va="top", fontsize=6.4, color=_INK, fontweight="bold")
        ax.plot([x, x + w], [yy - 2.2, yy - 2.2], color=_INK, lw=0.5)
        return yy - 5.2

    def line(text, yy, color=_INK, size=5.2, mono=False):
        ax.text(x, yy, text, ha="left", va="top", fontsize=size, color=color,
                family=_MONO if mono else None)
        return yy - (size * 0.30 + 1.35)

    y = head("LEGEND", y)
    swatch_x, text_x, row = x + 0.5, x + 10.0, 4.4
    for kind, label in [("cut", "cut  (knife, full depth)"),
                        ("line", "construction line  (do NOT cut)"),
                        ("point", "marking point  (POINTS table)"),
                        ("hole", f"relief hole  {sm['hole_drill_label']}")]:
        cy = y - 1.2
        if kind == "cut":
            ax.plot([swatch_x, swatch_x + 7], [cy, cy], color=_INK, lw=0.9)
        elif kind == "line":
            ax.plot([swatch_x, swatch_x + 7], [cy, cy], color=_GREY, lw=0.35, ls=(0, (3, 2)))
        elif kind == "point":
            ax.add_patch(Circle((swatch_x + 3.5, cy), 0.9, facecolor="white", edgecolor=_BLUE,
                                lw=0.5))
        else:
            ax.add_patch(Circle((swatch_x + 3.5, cy), 0.6, facecolor="white", edgecolor=_INK,
                                lw=0.4))
        ax.text(text_x, cy, label, ha="left", va="center", fontsize=5.2, color=_INK)
        y -= row
    y -= 1.5

    y = head("SEQUENCE", y)
    for tag, step in [("1.", "Square the stock; mark the datum corner."),
                      ("2.", "Measure the stock diagonal (CHECKS below)."),
                      ("3.", "Mark the ANCHOR points of each line only"),
                      ("", "(POINTS table: x/y, then both tape arcs)."),
                      ("4.", "Scribe the line through its two anchors."),
                      ("5.", "Step ALONG the line for its other points"),
                      ("", "(LINES table, station column). No ordinates."),
                      ("6.", "Verify the layout checks before cutting."),
                      ("7.", "Drill every relief hole."),
                      ("8.", "Cut hole-to-hole. Never cut past a hole.")]:
        y = line(f"{tag:<3s}{step}", y)
    y -= 2.0

    y = head("PRE-CUT CHECKS", y)
    for ch in schedule["checks"]:
        y = line(f"{ch['frm']:>6s} - {ch['to']:<6s} {ch['dist']:9.1f}", y, mono=True, size=5.0)
    y = line("measure before cutting; a bad mark is free", y, color=_GREY, size=4.6)
    y -= 2.0

    y = head("QUANTITIES", y)
    for k, v in [("marking points", f"{sm['n_points']}"),
                 ("of those, MEASURED", f"{sm['n_measured']}"),
                 ("cuts", f"{sm['n_cuts']}"),
                 ("relief holes", f"{sm['n_holes']}"),
                 ("total cut length", f"{sm['cut_len_total'] / 1000.0:.2f} m"),
                 ("ligament w_lig", f"{sm['w_lig_min']:.1f}-{sm['w_lig_max']:.1f} mm")]:
        y = line(f"{k:<18s}{v}", y, mono=True, size=5.0)
    y -= 2.0

    act = schedule.get("activation", [])
    if act:
        y = head("ACTIVATION", y)
        for a in act:
            y = line(f"{a['role']:<5s} tile {a['face']:<3d} at ({a['x']:.0f}, {a['y']:.0f})", y,
                     mono=True, size=5.0, color=_INK if a["role"] == "HOLD" else _RED)
        n_hold = sum(1 for a in act if a["role"] == "HOLD")
        n_pull = len(act) - n_hold
        y = line(f"{n_hold} held, {n_pull} pulled. Pull along the", y, color=_GREY, size=4.6)
        y = line("long axis. Nothing else restrained.", y, color=_GREY, size=4.6)
        for t in ((meta or {}).get("grip_note") or []):
            y = line(t, y, color=_GREY, size=4.6)

    # ── the caveat, anchored to the foot of the drawing so the column reads as a block ──
    caveat = (meta or {}).get("caveat", [])
    if not caveat:
        return
    box_h = 8.0 + 3.0 * len(caveat)
    top = max(oy + box_h, min(y - 4.0, oy + box_h + 40.0))
    _frame(ax, x, top - box_h, w, box_h, lw=0.8, color=_RED)
    ty = top - 2.6
    ax.text(x + 2.0, ty, "READ BEFORE CUTTING", ha="left", va="top", fontsize=5.8, color=_RED,
            fontweight="bold")
    ty -= 4.4
    for t in caveat:
        ax.text(x + 2.0, ty, t, ha="left", va="top", fontsize=4.6, color=_RED)
        ty -= 3.0


def _title_block(ax, schedule, x, y, w, h, draw_scale, meta):
    """Title strip across the head of the sheet."""
    _frame(ax, x, y, w, h, lw=0.9)
    W, H = schedule["stock"]
    sm = schedule["summary"]
    m = meta or {}
    ax.text(x + 3.0, y + h - 3.0, m.get("title", "KIRIGAMI CUT LAYOUT"), ha="left", va="top",
            fontsize=11.5, color=_INK, fontweight="bold")
    ax.text(x + 3.0, y + h - 11.5, m.get("subtitle", ""), ha="left", va="top", fontsize=6.2,
            color=_GREY)

    cells = [("STOCK", f"{W:.1f} × {H:.1f} mm"),
             ("MATERIAL", m.get("material", "")),
             ("SCALE", f"1:{draw_scale:g}"),
             ("PITCH", m.get("pitch", "")),
             ("CUTS / HOLES", f"{sm['n_cuts']} / {sm['n_holes']}"),
             ("DRILL", sm["hole_drill_label"]),
             ("DESIGN", m.get("source", "")),
             ("DATE", m.get("date", ""))]
    cw = w / 4.0
    for i, (k, v) in enumerate(cells):
        cx = x + (i % 4) * cw + 3.0
        cy = y + h - 15.0 - (i // 4) * 5.6
        ax.text(cx, cy, k, ha="left", va="top", fontsize=4.4, color=_GREY)
        ax.text(cx + 21.0, cy, v, ha="left", va="top", fontsize=5.4, color=_INK)


# ── A4 tables ──────────────────────────────────────────────────────────────────────

def render_tables_a4(schedule, out_pdf, *, meta=None):
    """Point / cut / hole tables, paginated onto A4 portrait with a tick column."""
    from matplotlib.backends.backend_pdf import PdfPages

    W, H = schedule["stock"]
    pt_hdr = (f"{'ID':<5s}{'origin':<10s}{'x':>9s}{'y':>9s}{'from A':>10s}{'from B':>10s}  "
              f"{'on edge':<12s}{'lines':>6s}  []")
    pt_rows = [
        f"{p['id']:<5s}{p['origin']:<10s}{p['x']:>9.1f}{p['y']:>9.1f}"
        f"{p['d_from_A']:>10.1f}{p['d_from_B']:>10.1f}  "
        f"{(p['edge'] + ' ' + format(p['along'], '.1f')) if p['edge'] else '-':<12s}"
        f"{p['n_lines']:>6d}  []"
        for p in schedule["points"]]

    # LINES: one block per cut. Set out the two ANCHOR points, scribe, then every remaining point on
    # that line is one measurement ALONG it.
    ln_hdr = (f"{'LINE':<6s}{'pt':>5s}{'station':>10s}{'step':>9s}  {'how to place it':<34s}[]")
    ln_rows = []
    for c in schedule["cuts"]:
        st = c["stations"]
        prev = 0.0
        origin = {p["id"]: p["origin"] for p in schedule["points"]}
        for k, (pid, s) in enumerate(st):
            og = origin.get(pid, "")
            how = ("MEASURE it (POINTS table)" if og == "MEASURE"
                   else f"already set on {og}" if og and not og.startswith(c["id"])
                   else f"step along from {st[0][0]}")
            ln_rows.append(f"{(c['id'] if k == 0 else ''):<6s}{pid:>5s}{s:>10.1f}"
                           f"{(s - prev if k else 0.0):>9.1f}  {how:<34s}[]")
            prev = s
        ln_rows.append(f"{'':<6s}{'':>5s}{'':>10s}{'':>9s}  "
                       f"CUT {c['cut_from_station']:.1f} -> {c['cut_to_station']:.1f} "
                       f"({c['cut_len']:.1f} mm)   []")
        ln_rows.append("")

    ct_hdr = (f"{'ID':<5s}{'from':>6s}{'to':>6s}  {'through':<16s}{'line':>9s}{'set.a':>8s}"
              f"{'set.b':>8s}{'CUT':>9s}{'ang':>7s}  []")
    ct_rows = [
        f"{c['id']:<5s}{c['from_id']:>6s}{c['to_id']:>6s}  "
        f"{(' '.join(c['through']) or '-'):<16s}{c['line_len']:>9.1f}"
        f"{c['setback_from']:>8.1f}{c['setback_to']:>8.1f}{c['cut_len']:>9.1f}"
        f"{c['angle_deg']:>7.1f}  []"
        for c in schedule["cuts"]]

    hl_hdr = f"{'ID':<5s}{'x':>9s}{'y':>9s}{'dia':>8s}{'w_lig':>8s}  []"
    hl_rows = [f"{h['id']:<5s}{h['x']:>9.1f}{h['y']:>9.1f}{h['dia_nominal']:>8.1f}"
               f"{h['w_lig']:>8.1f}  []" for h in schedule["holes"]]

    sections = [
        ("POINTS  —  mark these first",
         ["x, y are measured from the datum corner along the stock edges.",
          "from A / from B are tape distances from the two BOTTOM corners:",
          f"A = (0, 0) and B = ({W:.1f}, 0). Strike two arcs to place a point",
          "without squaring off a long edge. Use them to check every mark.",
          "'on edge' means the point lies on the factory edge at that distance.",
          "'lines' is how many construction lines must pass through the mark.",
          "You do NOT have to set every point out this way -- see the LINES table.",
          "Only the anchor points need ordinates; the rest come off the lines."],
         pt_hdr, pt_rows),
        ("LINES  —  work line by line, not point by point",
         ["Every point on a cut is COLLINEAR with the rest, to better than a",
          "thousandth of a millimetre. So set out the two ANCHOR points of a line",
          "from the POINTS table, scribe it, and then place every other point on",
          "that line with ONE measurement along it -- no second ordinate, no",
          "squaring off, and no error of its own.",
          "station = distance along the line from its first point. step = gap",
          "from the point above. The CUT line under each block gives the two",
          "stations to cut between; outside them is ligament, do not cut."],
         ln_hdr, ln_rows),
        ("CUTS  —  the same lines, summarised",
         ["line = full length between the two anchor points (scribe this).",
          "through = the marks this line must also pass through. If it misses",
          "one, a mark is wrong: fix it before cutting.",
          "set.a / set.b = ligament setback at each end: do NOT cut there.",
          "CUT = the length actually cut, hole centre to hole centre.",
          "A setback of 0.0 means the cut runs out to the sheet edge."],
         ct_hdr, ct_rows),
        ("RELIEF HOLES  —  drill before cutting",
         ["Each hole sits at a cut end. Drill it first, then cut into it:",
          "the drilled edge is the hinge's stress relief and must stay round.",
          "dia is the drill for THAT hole; w_lig is that hinge's ligament width.",
          "The hole tracks the ligament, so the sizes differ -- check each row."],
         hl_hdr, hl_rows),
    ]

    with PdfPages(out_pdf) as pdf:
        for title, notes, hdr, rows in sections:
            for n_page, page in enumerate(_paginate(rows, first=34, rest=52)):
                fig, ax = _page(_A4_W, _A4_H)
                y = _A4_H - 14.0
                ax.text(12.0, y, title + ("" if n_page == 0 else f"  (cont. {n_page + 1})"),
                        ha="left", va="top", fontsize=9.5, color=_INK, fontweight="bold")
                y -= 5.0
                ax.plot([12.0, _A4_W - 12.0], [y, y], color=_INK, lw=0.7)
                y -= 4.0
                if n_page == 0 and notes:
                    for n in notes:
                        ax.text(12.0, y, n, ha="left", va="top", fontsize=6.0, color=_GREY)
                        y -= 3.4
                    y -= 1.5
                ax.text(12.0, y, hdr, ha="left", va="top", fontsize=6.6, color=_INK,
                        family=_MONO, fontweight="bold")
                y -= 4.2
                for r in page:
                    ax.text(12.0, y, r, ha="left", va="top", fontsize=6.6, color=_INK,
                            family=_MONO)
                    y -= 4.05
                _footer(ax, meta)
                pdf.savefig(fig)
                plt.close(fig)
    return out_pdf


def _paginate(rows, *, first, rest):
    """Split rows into pages; the first page is shorter because it carries the notes."""
    if not rows:
        return [[]]
    pages, i = [rows[:first]], first
    while i < len(rows):
        pages.append(rows[i:i + rest])
        i += rest
    return pages


def _footer(ax, meta):
    m = meta or {}
    ax.text(12.0, 8.0, m.get("subtitle", ""), ha="left", va="bottom", fontsize=5.0, color=_GREY)
    ax.text(_A4_W - 12.0, 8.0, "all dimensions in mm", ha="right", va="bottom", fontsize=5.0,
            color=_GREY)


# ── A4 detail + 1:1 gauge ──────────────────────────────────────────────────────────

def _nearest(values, x):
    """The entry of ``values`` closest to ``x``."""
    return min(values, key=lambda v: abs(v - x))


def _fill_shapely(ax, g, **kw):
    """Fill a Shapely (Multi)Polygon, honouring interior rings."""
    for p in getattr(g, "geoms", [g]):
        if p.is_empty or not hasattr(p, "exterior"):
            continue
        verts, codes = [], []
        for ring in [p.exterior, *p.interiors]:
            xs, ys = ring.coords.xy
            verts += list(zip(xs, ys))
            codes += [Path.MOVETO] + [Path.LINETO] * (len(xs) - 2) + [Path.CLOSEPOLY]
        ax.add_patch(PathPatch(Path(verts, codes), **kw))


def render_detail_a4(schedule, geom, out_pdf, *, meta=None, detail_scale=3.0):
    """One real hinge at ``detail_scale``:1, a 1:1 gauge, and a print-calibration bar.

    The hinge is not a schematic: the actual Shapely geometry is clipped to a window around one
    ligament and rotated so the terminating cut lies horizontal, so what is dimensioned here is the
    same object the tables were measured from.

    The gauge is the reason this page exists. A ligament is around 12 mm and a relief hole around
    4 mm; both are easy to mark a millimetre wrong, and a millimetre is 8% of the ligament. The
    calibration bar comes first because a printer set to "fit to page" silently voids the gauge.
    """
    sm = schedule["summary"]
    info = geom["hinge_info"]
    med = float(np.median([h["w_lig"] for h in info]))
    hinge = min(info, key=lambda h: abs(float(h["w_lig"]) - med))
    wl, rho = float(hinge["w_lig"]), float(hinge["rho"])
    tip = np.asarray(hinge["tip"], float)
    u = np.asarray(hinge["dir"], float)
    u = u / (np.linalg.norm(u) + 1e-12)
    k = float(detail_scale)

    fig, ax = _page(_A4_W, _A4_H)
    ax.text(12.0, _A4_H - 14.0, "HINGE DETAIL — print at ACTUAL SIZE (100%)", ha="left",
            va="top", fontsize=9.5, color=_INK, fontweight="bold")
    ax.plot([12.0, _A4_W - 12.0], [_A4_H - 19.0, _A4_H - 19.0], color=_INK, lw=0.7)

    # ── print calibration, first, because everything below depends on it ──
    bar = 150.0
    bx, by = (_A4_W - bar) / 2.0, _A4_H - 30.0
    ax.plot([bx, bx + bar], [by, by], color=_INK, lw=1.2, solid_capstyle="butt")
    for t in np.arange(0, bar + 1e-6, 10.0):
        ax.plot([bx + t, bx + t], [by, by + (3.0 if t % 50 == 0 else 1.8)], color=_INK, lw=0.7)
    ax.text(_A4_W / 2, by - 3.0, "This bar must measure exactly 150 mm. If it does not, the "
            "printer rescaled the page and the 1:1 gauge below is void.",
            ha="center", va="top", fontsize=5.6, color=_RED)

    # ── the real hinge, clipped and squared up ──
    half = 1.6 * wl
    cx, cy = _A4_W / 2.0, _A4_H - 105.0
    win = shapely_box(tip[0] - 1.6 * half, tip[1] - 1.6 * half,
                      tip[0] + 1.6 * half, tip[1] + 1.6 * half)
    theta = float(np.degrees(np.arctan2(u[1], u[0])))

    def to_page(g):
        g = affinity.rotate(g, -theta, origin=(tip[0], tip[1]))
        g = affinity.scale(g, k, k, origin=(tip[0], tip[1]))
        return affinity.translate(g, cx - tip[0], cy - tip[1])

    page_win = shapely_box(cx - k * half, cy - k * half, cx + k * half, cy + k * half)
    _fill_shapely(ax, to_page(geom["sheet"].intersection(win)).intersection(page_win),
                  facecolor="#F2F2F2", edgecolor="none", zorder=1)
    _fill_shapely(ax, to_page(geom["cuts"].intersection(win)).intersection(page_win),
                  facecolor=_INK, edgecolor="none", zorder=2)
    _frame(ax, cx - k * half, cy - k * half, 2 * k * half, 2 * k * half, lw=0.4, color=_GREY)

    # ── dimensions, in page mm: the tip is at (cx, cy) and +x runs along the cut ──
    dim = dict(arrowstyle="<|-|>", color=_BLUE, lw=0.9, mutation_scale=6)
    vx = cx - k * wl                                        # the crossing (un-retracted vertex)
    off = k * 0.95 * wl
    ax.annotate("", (cx, cy + off), (vx, cy + off), arrowprops=dim)
    for px in (vx, cx):
        ax.plot([px, px], [cy, cy + off], color=_BLUE, lw=0.4)
    ax.text((vx + cx) / 2, cy + off + 1.4, f"ligament  {wl:.2f} mm", ha="center", va="bottom",
            fontsize=6.6, color=_BLUE)
    ax.plot([cx], [cy], marker="+", ms=5, mew=0.9, color="white", zorder=4)   # the drill centre

    lead = (cx + k * 0.85 * wl, cy - k * 1.15 * wl)
    ax.annotate("", (cx, cy), lead, arrowprops=dict(arrowstyle="-", color=_BLUE, lw=0.4))
    ax.text(lead[0] + 1.5, lead[1], f"relief hole  Ø{_nearest(sm['hole_drills'], 2 * rho):.1f} mm  "
            f"(exact Ø{2 * rho:.2f})   THIS hinge only", ha="left", va="center", fontsize=6.6,
            color=_BLUE, bbox=dict(facecolor="white", edgecolor="none", pad=1.0))

    ax.text(_A4_W / 2, cy - k * half - 3.5,
            f"scale {detail_scale:g}:1 — the crossing cut runs SOLID through the junction; this cut "
            f"stops {wl:.2f} mm short of it.", ha="center", va="top", fontsize=6.2, color=_INK)
    ax.text(_A4_W / 2, cy - k * half - 7.5,
            f"Slots are drawn at the {geom['w_c']:.2f} mm simulated kerf. A knife cut is thinner "
            f"than that: cut ON the line, and take every dimension to the line.",
            ha="center", va="top", fontsize=6.2, color=_GREY)

    # ── 1:1 gauge, cut it out and hold it against the sheet ──
    gw, gh = 170.0, 40.0
    gx, gy = (_A4_W - gw) / 2.0, 66.0
    _frame(ax, gx, gy, gw, gh, lw=0.5, color=_GREY)
    ax.text(gx + 3.0, gy + gh - 3.0, "1:1 GAUGE — cut out along the grey box, then hold it "
            "against the marked sheet", ha="left", va="top", fontsize=6.2, color=_INK,
            fontweight="bold")
    lig_y, lx = gy + 15.0, gx + 14.0
    for px in (lx, lx + wl):
        ax.plot([px, px], [lig_y - 6.0, lig_y + 6.0], color=_INK, lw=1.2)
    ax.annotate("", (lx, lig_y), (lx + wl, lig_y), arrowprops=dim)
    ax.text(lx + wl / 2, lig_y - 7.5, f"w_lig {wl:.1f}", ha="center", va="top", fontsize=5.6,
            color=_BLUE)
    for i, d in enumerate(sm["hole_drills"][:4]):
        cxi = gx + 74.0 + i * 22.0
        ax.add_patch(Circle((cxi, lig_y), d / 2.0, facecolor="none", edgecolor=_INK, lw=0.6))
        ax.text(cxi, lig_y - 7.5, f"Ø{d:.1f}", ha="center", va="top", fontsize=5.6, color=_INK)

    # ── the ligament spread across the sheet: one drill, many widths ──
    ty = 54.0
    for t in (f"Ligament widths on this sheet run {sm['w_lig_min']:.1f} to {sm['w_lig_max']:.1f} mm "
              f"(median {sm['w_lig_med']:.1f}). They are NOT all the same, and the width IS the "
              f"design:",
              f"take each one from the CUTS table. The relief hole tracks it "
              f"(exact Ø{sm['hole_dia_min']:.2f}-{sm['hole_dia_max']:.2f} mm), so EACH HOLE HAS ITS "
              f"OWN DRILL in the HOLES table -- do not drill them all the same."
              if len(sm["hole_drills"]) > 1 else
              f"take each one from the CUTS table. Relief holes are close enough that one "
              f"Ø{sm['hole_drills'][0]:.1f} mm drill covers every one."):
        ax.text(12.0, ty, t, ha="left", va="top", fontsize=6.0, color=_INK)
        ty -= 4.2
    ty -= 4.0
    for t in (meta or {}).get("caveat", []):
        ax.text(12.0, ty, t, ha="left", va="top", fontsize=5.6, color=_RED)
        ty -= 3.6

    _footer(ax, meta)
    fig.savefig(out_pdf)
    plt.close(fig)
    return out_pdf
