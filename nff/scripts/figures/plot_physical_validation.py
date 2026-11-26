"""The physical validation on one plate: what was built, what one ligament does, how well it matches.

This merges two plates that had grown into each other. The deployment plate drew the measured tile
outlines against the predicted ones and deliberately carried no photograph; the buckle plate then
reintroduced a photograph of the same sheet, in the same orientation, through the same homography,
purely to say WHERE its close-ups were taken. The sheet was therefore drawn twice, and the one
picture that shows what was actually cut and pulled was the smallest thing on either page.

So the photograph becomes the subject. It leads, at about four times the area it had, and the ring
on it points at the two panels beside it, which resolve that one ligament measured against predicted.
Underneath, the argument the outlines make: three states against the design, then the two
quantities the outlines cannot carry -- hinge opening, which is a DIFFERENCE of tile rotations and
so is free of registration, and the per-tile shape residual once the sheet's rigid-body motion is
removed.

★★★ THE PHOTOGRAPH IS RETOUCHED, AND THE PLATE SAYS SO. A rectification's empty corners, a specular
highlight on transparent PET, a weight and a cable are all reconstructed by diffusion, and every
one of them is listed with its area in the sidecar ``.txt``. This is safe for exactly one reason:
the homography was fitted on the UNMODIFIED photograph and every number on the plate descends from
that fit, so nothing reconstructed here can reach a measurement. ⛔ Do not carry this licence over
to panels (b) or (c). The close-up IS the evidence for the escape, and it is untouched beyond the
white balance and gamma the sidecar already declares.

Run:
    PYTHONPATH=$(pwd) python3 nff/scripts/figures/plot_physical_validation.py
"""

import argparse
import json
import os

import numpy as np

from nff.utils.figstyle import (GREY, INK, ORANGE, TEAL, apply_charter, figsize_in, page_bp,
                                panel_letter, paper_rc, save_paper_figure, use_agg)

use_agg()
import matplotlib.pyplot as plt                                          # noqa: E402
from matplotlib.patches import FancyBboxPatch                            # noqa: E402
from nff.calibration import photo_retouch as prt                         # noqa: E402
from nff.scripts.figures import plot_deployment_validation as dv         # noqa: E402
from nff.scripts.figures import plot_hinge_buckle_figure as hb           # noqa: E402


# ── the layout, in points of the printed page ─────────────────────────────────────────────────
#
# Authored in bp and converted to figure fractions at the end, never guessed as fractions. The
# canvas height FOLLOWS from the parts rather than being chosen: pick an aspect first and every
# panel is silently letterboxed to fit it.
#
# ⚠⚠ THE BUDGET IS THE TEXT BLOCK, IN BOTH DIRECTIONS, AND IT BINDS. This class gives
# \textwidth = 370.6 bp and \textheight = 552.7 pt = 550.6 bp, and a full-page float must hold the
# plate AND its caption inside that. A first version was authored at 578 bp tall -- taller than the
# whole text block before a single caption line -- and its caption ran off the page past the folio.
# A second sat at 1.2\textwidth and bled past the block on both sides.
#
# ⛔ THIS PLATE DOES NOT BLEED. It is authored at \textwidth and included at \textwidth, so an
# authored 8 pt is 8 pt on the page. Widening it means REGENERATING at the new --width-frac, never
# scaling it up in \includegraphics, which would shrink every label by the scale factor.
#
# Eight panels in one column is tight, so every other panel is sized against the hero rather than
# the hero against them: at 0.635 of the canvas the photograph is 235 x 173 bp, about four times
# the area the merged-away plate gave it, and what is left over sets the state row and the charts.
# ⚠ The hero cannot simply grow: hero + gap + close-ups spans the canvas, so a wider hero narrows
# the close-up column and drives their data aspect up until they are nearly square.
L, R = 0.024, 0.985         # the drawing rows owe nothing to a y-label gutter, so they span
CHART_L = 0.092             # the chart row does, and this is the gutter it needs
GAP_X = 0.016
TOP_PAD = 17.5              # panel letters and the row-1 titles; below ~12 the letters clip
TITLE_BP = 11.0             # the lower close-up's own title, between the stacked pair
GAP_ROW12 = 12.0
GAP_ROW23 = 13.0
BOTTOM_PAD = 25.0
# ⚠ THERE IS NOW A LEGEND STRIP, under the state row and inside the frame that groups it. It was
# left off until 2026-08-21 on the argument that the caption already decodes orange, the thin
# outline, red and the dashed rectangle. That is enough to LOOK a mark UP and not enough to READ
# the row: five marks arrive at once and prose returns them one at a time. It costs 12 bp, paid out
# of GAP_ROW12 and GAP_ROW23, which each carried 2 bp more than a panel letter needs.
# ── the reading strip ─────────────────────────────────────────────────────────────────
# Under the state row and inside the frame that groups it: the colour key on the left, and on the
# right the same deployed sheet magnified about five times, where the disagreement finally has a
# size. ⚠⚠ THE STATE PANELS RUN AT 29 mm PER POINT. The widest sliver of red on the plate is 43 mm,
# so it prints 1.5 pt there: the eye sees THAT the two disagree and cannot say by how much, and the
# millimetres in the text have nothing on the page to attach to. Five times, beside a 100 mm bar,
# turns it into a length. The crop is chosen by error_hotspot, never by hand, so it cannot drift to
# a flattering neighbourhood.
STRIP_H = 28.0
STRIP_GAP = 2.0             # state-row axes bottom -> the strip; the axes already carry white
DETAIL_W = 112.0            # the magnified panel, in points; the key takes what is left
DETAIL_GAP = 14.0
# ⚠⚠ THE MAGNIFIED PANEL IS A RECTANGLE, AND IT HAS TO READ AS ONE. It is the locator rectangle on
# (f) at five times the size, and the leaders join corresponding corners, so the two borders are
# the SAME rectangle drawn twice -- that correspondence is the whole argument for where the crop
# came from. It only works while all four of its sides are visible: run it out to the column edge
# and its right and bottom merge with the frame that groups (a)-(f), and it stops looking like an
# object and starts looking like a hole cut in the frame. These two inset it clear of both.
DETAIL_INSET_R = 5.0
DETAIL_INSET_B = 3.0
# dv.MARGIN_MM is 235 mm, sized to hold the two dimensions the closed panel used to draw. This
# plate states its scale on (a) instead, so the state row buys back 1.5 bp of height and, more to
# the point, draws the three sheets 11% larger inside the same boxes.
STATE_MARGIN_MM = 120.0

# ── the two rules that group the plate ─────────────────────────────────────────────────
#
# (a)-(f) are the sheet itself, drawn; (g)-(h) are what was measured off it, plotted. Eight panels
# in one column read as one undifferentiated stack, and a reader arriving at (g) has no signal that
# the argument has changed register. Two hairline frames say it without a word of caption.
#
# ⚠⚠ THE FRAMES ARE PAID FOR IN HEIGHT, AND THE PAGE HAS ABOUT 20 bp TO GIVE. A full-page float
# must hold the plate AND its caption inside \textheight = 550.6 bp, and the caption runs ~101 bp,
# so the plate ceiling is ~450 bp. The frames need head room above (a) for its letter, a gap
# between them, and a tail below the chart labels: about 20 bp in total. That is bought back by
# narrowing the hero 0.635 -> 0.615 and the state panels 0.315 -> 0.300, NOT by letting the plate
# grow -- at 456 bp the caption's last line lands past the folio.
BOX_OUT = 4.0               # canvas edge -> frame, and frame -> the column it holds
BOX_HEAD = 5.0              # the extra head room the frame needs to clear the letter of panel (a)
BOX_FOOT = 2.0              # row-2 axes bottom -> frame; the state axes already carry white there
BOX_GAP = 7.0               # between the two frames
BOX_TAIL = 3.0              # below the chart x-labels -> frame
BOX_KW = dict(facecolor="none", edgecolor=GREY, lw=0.5, alpha=0.55)


def layout(width_frac, frame_on, margin, hero_w=0.598, state_w=0.300, chart_h=112.0,
           level_deg=0.0, boxed=True):
    """Every box on the plate, in points, from the panels' own data aspects.

    Args:
        hero_w: the photograph's width as a fraction of the canvas.
        state_w: each state panel's width as a fraction of the canvas.
        chart_h: height of the two quantitative panels, in points.
        boxed: reserve the head room, the inter-frame gap and the tail the two grouping frames
            need, and return them under ``"box_draw"`` and ``"box_plot"``.

    Returns:
        ``(w_bp, h_bp, boxes, aspect_y)``; ``boxes`` maps a name to ``(x, y_from_top, w, h)`` in
        points, and ``aspect_y`` is the data aspect the two close-ups must be drawn at for the
        stacked pair to finish level with the hero.
    """
    w_bp = page_bp(width_frac, 1.0)[0]
    x0, x1, y0, y1, _, _ = hb.context_window(frame_on, margin, level_deg)
    hero_w_bp = hero_w * w_bp
    hero_h = hero_w_bp * (y1 - y0) / (x1 - x0)
    zoom_w = (R - L - hero_w - GAP_X) * w_bp
    zoom_h = 0.5 * (hero_h - TITLE_BP)
    # ⚠ The close-ups' data aspect is zoom_h / zoom_w, and a WIDER hero makes it LARGER, because a
    # wider hero is both taller and leaves a narrower column. Below about 0.5 the predicted dome is
    # cut off by its own panel edge. The single-row plate this merges ran at 0.77, so staying near
    # that is staying with a framing that is known to hold the mode.

    row2_w = state_w * w_bp
    win = _deployment_window()
    row2_h = row2_w * (win[3] - win[2]) / (win[1] - win[0])
    row2_gap = ((R - L) * w_bp - 3.0 * row2_w) / 2.0

    y = TOP_PAD if boxed else TOP_PAD - BOX_HEAD
    boxes = {"hero": (L * w_bp, y, hero_w_bp, hero_h),
             "meas": ((L + hero_w + GAP_X) * w_bp, y, zoom_w, zoom_h),
             "pred": ((L + hero_w + GAP_X) * w_bp, y + zoom_h + TITLE_BP, zoom_w, zoom_h)}
    y += hero_h + GAP_ROW12
    for j in range(3):
        boxes[f"state{j}"] = (L * w_bp + j * (row2_w + row2_gap), y, row2_w, row2_h)
    row2_bot = y + row2_h
    strip_y = row2_bot + STRIP_GAP
    strip_w = (R - L) * w_bp
    detail_x = L * w_bp + strip_w - DETAIL_INSET_R - DETAIL_W
    boxes["detail"] = (detail_x, strip_y, DETAIL_W, STRIP_H - DETAIL_INSET_B)
    boxes["legend"] = (L * w_bp, strip_y, detail_x - L * w_bp - DETAIL_GAP, STRIP_H)
    row2_bot = strip_y + STRIP_H
    y = row2_bot + GAP_ROW23 + (BOX_FOOT + BOX_GAP if boxed else 0.0)
    boxes["charts"] = (CHART_L * w_bp, y, (R - CHART_L) * w_bp, chart_h)
    h_bp = y + chart_h + BOTTOM_PAD + (BOX_TAIL + BOX_OUT if boxed else 0.0)
    if boxed:
        # Both frames span the same column, so the plate reads as two blocks of one document and
        # not as two figures that happen to share a page.
        fx, fw = L * w_bp - BOX_OUT, (R - L) * w_bp + 2.0 * BOX_OUT
        boxes["box_draw"] = (fx, BOX_OUT, fw, row2_bot + BOX_FOOT - BOX_OUT)
        top = row2_bot + BOX_FOOT + BOX_GAP
        boxes["box_plot"] = (fx, top, fw, h_bp - BOX_OUT - top)
    return w_bp, h_bp, boxes, zoom_h / zoom_w


def _deployment_window(align=True):
    """The frame the three state panels share, from the deployment measurement."""
    M, D = _deployment_data()
    return dv.window(M, D, dv.ROT_DEG, align, margin_mm=STATE_MARGIN_MM)


_CACHE: dict = {}


def _deployment_data(exp_dir="data/experiments/pet45_3x3_deployment_20260814"):
    if "dep" not in _CACHE:
        M = json.load(open(os.path.join(exp_dir, "derived", "metrics.json")))
        D = np.load(os.path.join(exp_dir, "derived", "measured.npz"))
        _CACHE["dep"] = (M, D)
    return _CACHE["dep"]


def hero_scale_bar(ax, at=(0.655, 0.185), length_mm=500.0, tick_mm=45.0):
    """A scale bar on the photograph, placed in the void the deployment opens.

    The photograph is the one panel that cannot be dimensioned: it is cropped to its subject, so
    there is no room outside the sheet for witness lines, and the sheet's own edges are the
    measurement rather than a known size. ``at`` is in axes fractions and lands the bar on floor
    between the two long panels, which is empty in the deployed state by construction.
    """
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    dv.scale_bar(ax, x0 + at[0] * (x1 - x0), y0 + at[1] * (y1 - y0),
                 length=length_mm, tick=tick_mm)


def place(fig, box, w_bp, h_bp):
    """A points box measured from the TOP, as the matplotlib axes rectangle it corresponds to."""
    x, y, w, h = box
    return fig.add_axes([x / w_bp, 1.0 - (y + h) / h_bp, w / w_bp, h / h_bp])


def build(args):
    M, D = _deployment_data(args.exp_dir)
    clamp = M["states"]["Deployed"]["clamp_face"]
    pull = M["states"]["Deployed"]["pull_face"]
    win = dv.window(M, D, dv.ROT_DEG, args.align_sim, margin_mm=STATE_MARGIN_MM)
    sim_npz = np.load(M["sim"])
    dirs = sim_npz["load_dirs"] if "load_dirs" in sim_npz.files else np.array([[0.0, 1.0]])
    pull_dir = dv.rotate(np.asarray(dirs)[0][None, :], dv.ROT_DEG)[0]
    pull_dir = pull_dir / max(float(np.linalg.norm(pull_dir)), 1e-9)

    shot = hb.SHOTS[args.variant]
    fit = hb.fit_shot_camera(shot, args.frames, args.theta_measured, view_r=args.view_r,
                             use_cache=not args.no_cache)
    d, k, cam = fit["frames"], fit["k"], fit["camera"]

    level_deg = hb.level_angle(args.level)
    w_bp, h_bp, boxes, aspect_y = layout(args.width_frac, args.frame_on, args.margin,
                                         hero_w=args.hero_w, state_w=args.state_w,
                                         chart_h=args.chart_h, level_deg=level_deg,
                                         boxed=not args.no_boxes)
    apply_charter(**paper_rc())
    fig = plt.figure(figsize=figsize_in(w_bp, h_bp))

    # ---- row 1: the sheet, and the one ligament it rings
    ledger: dict = {}
    ax_hero = place(fig, boxes["hero"], w_bp, h_bp)
    # ⚠ draw_target=True. The retouch repaints the taped rectangle NEUTRAL, so without the outline
    # the target survives in (a) only as a pale band a reader has no reason to read as anything --
    # while (d)-(f) name the same rectangle with a black dashed line. One target, one line style.
    hb.panel_context(ax_hero, marker_frac=args.marker_frac, r_win_mm=float(d["r_win"]),
                     frame_on=args.frame_on, margin=args.margin, retouch=not args.no_retouch,
                     draw_target=True, ledger=ledger, level_deg=level_deg)
    hero_scale_bar(ax_hero, at=(args.bar_x, args.bar_y), length_mm=args.bar_mm)

    ax_meas = place(fig, boxes["meas"], w_bp, h_bp)
    # ⚠ aspect_y is PASSED, not derived from the hero. The single-row plate took it from panel (a)
    # so that three panels read as one strip; here the pair is stacked beside a much larger picture
    # and must instead be the shape that makes the column finish level with it.
    hb.panel_photo(ax_meas, shot, fit["view_r"], level_deg=-cam["roll"], aspect_y=aspect_y,
                   target_px=args.target_px)
    hb.scale_bar(ax_meas, fit["view_r"], aspect_y=aspect_y)

    ax_pred = place(fig, boxes["pred"], w_bp, h_bp)
    rend = hb.panel_render(ax_pred, d, k, cam, fit["view_r"], tape=shot["tape"], dpi=args.dpi,
                           aspect_y=aspect_y, roll=0.0, use_cache=not args.no_cache,
                           target_px=args.target_px)
    hb.scale_bar(ax_pred, fit["view_r"], aspect_y=aspect_y)

    for ax, txt in ((ax_hero, "the deployed sheet"),
                    (ax_meas, "measured   $\\vartheta$ = %.1f$^\\circ$" % args.theta_measured),
                    (ax_pred, "predicted   $\\vartheta$ = %.1f$^\\circ$" % cam["theta"])):
        ax.set_title(txt, fontsize=8.0, color=INK, pad=3)

    # ---- row 2: the same sheet against the design, in three states
    dbox = boxes["detail"]
    detail_win = None if args.no_detail else dv.error_hotspot(
        M, D, args.detail_state, dv.ROT_DEG, args.align_sim,
        args.detail_mm_per_bp * dbox[2], args.detail_mm_per_bp * dbox[3])

    frames = []
    for j, name in enumerate(dv.STATES):
        ax = place(fig, boxes[f"state{j}"], w_bp, h_bp)
        # dims=False: the closed panel keeps its tile numbers, which every other panel and both
        # charts index by, but gives up its two dimensions. The scale is stated once, on (a).
        dv.panel_frame(ax, name, M, D, win, dv.ROT_DEG, args.align_sim, clamp, pull, pull_dir,
                       args.sim_style, first=(j == 0), dims=False,
                       detail_win=detail_win if name == args.detail_state else None)
        frames.append(ax)

    ncol = 3 if detail_win is not None else 6
    dv.state_legend(place(fig, boxes["legend"], w_bp, h_bp), sim_style=args.sim_style, ncol=ncol)
    if detail_win is not None:
        ax_det = place(fig, dbox, w_bp, h_bp)
        # ⚠ NOT the state's widest: fit_inside holds the answer clear of the crop edges so the
        # dimension and its label fit, so this is the widest place that can be annotated. The
        # sidecar reports both, or the two numbers look like a discrepancy.
        dim = dv.widest_error(M, D, args.detail_state, dv.ROT_DEG, args.align_sim,
                              window=detail_win)
        dv.panel_detail(ax_det, args.detail_state, M, D, detail_win, dv.ROT_DEG, args.align_sim,
                        clamp, args.sim_style, dim=dim)
        # The leaders say WHICH panel the crop came from, so the label does not: it carries the
        # magnification alone. ⚠ INSIDE the axes, not a title -- STRIP_GAP is 2 bp, so a title
        # lands back up in the state row, on the locator rectangle itself.
        mag = (win[1] - win[0]) / (detail_win[1] - detail_win[0])
        ax_det.text(0.985, 0.93, f"$\\times${mag:.0f}", transform=ax_det.transAxes,
                    ha="right", va="top", fontsize=6.6, color=INK,
                    bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))
        dv.link_detail(fig, frames[dv.STATES.index(args.detail_state)], ax_det, detail_win)

    # ---- row 3: what the outlines cannot carry
    cx, cy, cw, ch = boxes["charts"]
    gs = fig.add_gridspec(1, 2, left=cx / w_bp, right=(cx + cw) / w_bp,
                          top=1.0 - cy / h_bp, bottom=1.0 - (cy + ch) / h_bp,
                          wspace=0.30, width_ratios=[1.0, 1.30])
    with plt.rc_context({"axes.grid": False}):
        ax_par = fig.add_subplot(gs[0, 0])
        # ⚠ The parity axes is aspect-equal, so its data square is set by the row HEIGHT and the
        # legend keeps its point size: too short a row and the legend crosses the equality line.
        # Moving it outside was tried and is worse at \textwidth, where the width the square gives
        # back is exactly where panel (h)'s y-label lives. Buying the height instead is what works.
        dv.panel_parity(ax_par, M, dict(zip(dv.STATES, dv.shades(TEAL, 3))))
    ax_res = fig.add_subplot(gs[0, 1])
    dv.panel_tiles(ax_res, M, M["noise_floor_mm"], clamp,
                   dict(zip(dv.STATES, dv.shades(ORANGE, 3))))

    # Panel letters from the REAL axes boxes after layout: a guessed figure fraction clips the
    # letter off the canvas the moment a panel moves.
    for name in ("box_draw", "box_plot"):
        if name in boxes:
            x, y, w, h = boxes[name]
            fig.add_artist(FancyBboxPatch(
                (x / w_bp, 1.0 - (y + h) / h_bp), w / w_bp, h / h_bp,
                boxstyle="round,pad=0,rounding_size=0.008",
                transform=fig.transFigure, zorder=0, **BOX_KW))

    ordered = [ax_hero, ax_meas, ax_pred] + frames + [ax_par, ax_res]
    for ax, letter in zip(ordered, "abcdefgh"):
        bb = ax.get_position()
        # ⚠ 0.004, not 0.010. The letter has to sit INSIDE the frame that groups it, and the frame
        # is only BOX_OUT clear of the column; a letter hung the old distance to the left of panel
        # (a) landed on the rule itself.
        dx = 0.004 if ax in ordered[:6] else 0.0
        panel_letter(fig, letter, x=max(bb.x0 - dx, 0.004), y=min(bb.y1 + 0.012, 0.997))

    detail = None
    if detail_win is not None:
        widest = 2.0 * dv.widest_error(M, D, args.detail_state, dv.ROT_DEG, args.align_sim)[0]
        # ⚠ The PANEL's letter, not the state name's initial. "Deployed" is panel (f).
        detail = (round(mag), widest, "def"[dv.STATES.index(args.detail_state)], 2.0 * dim[0])
    return fig, w_bp, h_bp, dict(M=M, cam=cam, fit=fit, render=rend, frames=d, k=k,
                                 ledger=ledger, shot=shot, detail=detail)


def caption_text(info, args, w_bp, h_bp):
    """The provenance sidecar: every number, and every retouch, off the plate."""
    M, cam, d, k = info["M"], info["cam"], info["frames"], info["k"]
    clo = M["states"]["Closed"]
    dep = M["states"]["Deployed"]
    dly = M["states"]["Deploying"]
    b = M.get("buckle", {})
    uz = d["disp"][k][:, 2]
    lines = [
        "Figure. A cut sheet deployed by hand, and one of its ligaments leaving the plane.",
        "",
        "(a) The deployed sheet photographed in plan, rectified through the deployment homography",
        "and rotated so its long axis is horizontal. Orange is the measured sheet, the grey panel",
        "is held, the black dashed rectangle is the taped target, teal is the meshed window of",
        "every hinge, and the ring marks the one resolved in (b) and (c), the ligament joining",
        "panels 1 and 2. The scale bar is the plate's only statement of scale: the closed panel",
        "(d) no longer carries the two dimensions of the flat sheet.",
        "(b) That ligament photographed at the deployed state with the marking tape removed.",
        "(c) The same ligament in the three-dimensional solve, driven along its own measured route",
        "and drawn at true out-of-plane scale. (b) and (c) share one camera and one scale, both",
        "fixed by the relief hole drilled at the cut tip, which the photograph and the model have",
        "in common.",
        "(d) closed, (e) part deployed, (f) deployed: the measured sheet against the prediction,",
        "the latter drawn with the rigid-body offset between the two removed, with the",
        "disagreement filled red.",
        "(g) Hinge opening measured against predicted, twelve hinges per state.",
        "(h) Distance between measured and predicted panel centres, rigid-body motion removed.",
        "",
        "Provenance, not part of the caption.",
        f"Inputs: {M['photo_dir']}, {M['sim']}; close-up {info['shot']['image']} of",
        f"{hb.CLOSEUP}; oracle frame {k} of {args.frames}.",
        f"Measurement floor {clo['centroid_rms_mm']:.1f} mm centroid, {clo['rot_rms_deg']:.2f} deg",
        f"rotation, {clo['opening_rmse_deg']:.2f} deg opening, from the closed frame.",
        f"Deployed: matched on {dep['match_on']} at load {dep['load_fraction']:.3f}; opening RMSE",
        f"{dep['opening_rmse_deg']:.2f} deg; shape-only {dep['shape_only_rms_mm']:.1f} mm; the hand",
        f"clamp slipped {dep['clamp_slip_mm']:.1f} mm; the pull tile travelled",
        f"{dep['pull_travel_measured_mm']:.0f} mm against {dep['pull_travel_sim_mm']:.0f} mm",
        f"simulated. At matched extension the sheet is {b.get('deficit_mm', 0):.1f} mm",
        f"({b.get('deficit_frac', 0) * 100:.1f}%) narrower than predicted.",
        f"Deploying was shot mid-pull and is NOT a static equilibrium ({dly['opening_rmse_deg']:.2f}",
        "deg opening RMSE); its openings are comparable, its positions are not.",
        f"Camera of (c) fitted to (b) on the relief hole: axis ratio {info['render']['ratio']:.3f}",
        f"rendered against {fit_ratio(info):.3f} photographed, bearing",
        f"{info['render']['bearing_deg']:.1f} against {photo_bearing(info):.1f} deg.",
        f"Oracle at theta = {cam['theta']:.2f} deg: max out-of-plane {uz.max():.1f} mm =",
        f"{uz.max() / float(d['thickness']):.0f}x the {float(d['thickness']):.1f} mm gauge; peak",
        f"PEEQ {float(np.nanmax(d['peeq_nodal'][k])):.4f}, damage {float(d['damage'][k]):.2e}.",
        "",
        "★ Panel (a) is RETOUCHED. The homography was fitted on the unmodified photograph and every",
        "number above descends from that fit, so nothing reconstructed below reaches a measurement.",
        "Panels (b) and (c) are NOT retouched.",
    ]
    if info.get("detail") is not None:
        mag, widest, letter, shown = info["detail"]
        lines += [
            "",
            f"Under (d)-(f): the colour key, and the region outlined on ({letter}) drawn at "
            f"x{mag:.0f}. That crop is",
            "chosen as the window holding the most width-weighted disagreement, never by hand, so it",
            "cannot drift to a flattering neighbourhood. The widest sliver of red in that state is",
            f"{widest:.0f} mm: {widest / 29.4:.1f} pt in the state row, {widest / args.detail_mm_per_bp:.1f} pt in the",
            f"magnified panel, where it is dimensioned rather than left to a scale bar. The dimension",
            f"reads {shown:.0f} mm, not {widest:.0f}: it is drawn at the widest place far enough inside the crop for",
            "the dimension and its label to fit, which is not quite the widest place in the state.",
            "⛔ No scale bar here. The plate runs at 13.0, 1.13 and %.1f mm/pt, so its 50 mm bar" % args.detail_mm_per_bp,
            "prints 2.6x LONGER than its 500 mm one, and a fourth bar invites that arithmetic.",
        ]
    lines += prt.ledger_lines(info["ledger"])
    lines += ["",
              f"(a) is squared to the {args.level} and not to the design frame: it carries an extra",
              f"{hb.level_angle(args.level):+.2f} deg of in-plane rotation on top of ROT_DEG, because the sheet swings",
              "as it deploys and the target was taped at its own angle, so a photograph squared to the",
              "closed sheet shows nothing level. The swing is a result, and (d)-(f) are where it reads:",
              "a fixed target beside a sheet that has turned."]
    lines += [
        "",
        f"Authored at {args.width_frac} x \\textwidth ({w_bp:.1f} x {h_bp:.1f} bp): \\includegraphics",
        "must use the SAME fraction, or the type prints smaller than it was authored."
        + ("" if args.width_frac <= 1.0 else
           " Above 1.0 the manuscript must also wrap the graphic in \\makebox[\\textwidth][c]{}."),
        "Regenerate: PYTHONPATH=$(pwd) python3 nff/scripts/figures/plot_physical_validation.py",
    ]
    return "\n".join(lines) + "\n"


def fit_ratio(info):
    return float(info["fit"]["hole"].ratio)


def photo_bearing(info):
    return float(np.degrees(np.mod(-info["fit"]["hole"].angle, np.pi)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp-dir", default="data/experiments/pet45_3x3_deployment_20260814")
    ap.add_argument("--frames", default=hb.FRAMES)
    ap.add_argument("--out", default="paper/Images/physical_validation.pdf")
    ap.add_argument("--variant", choices=sorted(hb.SHOTS), default="bare")
    ap.add_argument("--theta-measured", type=float, default=16.58,
                    help="hinge opening the deployment photographs measure, in degrees")
    ap.add_argument("--view-r", type=float, default=None)
    ap.add_argument("--marker-frac", type=float, default=1.60)
    ap.add_argument("--width-frac", type=float, default=1.00,
                    help="page_bp width fraction; the \\includegraphics width must match it")
    ap.add_argument("--frame-on", choices=("sheet", "target"), default="target",
                    help="what the hero is framed on. ⚠ The 6.3%% of empty panel that made 'target' "
                         "unusable was measured in the DESIGN frame; once the panel is levelled on "
                         "the sheet the same framing costs 0.9%%, so the whole target rectangle "
                         "fits and the sheet still spans 88%% of the width")
    ap.add_argument("--margin", type=float, default=0.030,
                    help="white around the hero's content, as a fraction of its width. ⚠ It is "
                         "what separates the TARGET rectangle from the panel edge: at 0.010 the "
                         "two coincide and the dashed target reads as a frame around the "
                         "photograph rather than as the shape the sheet was asked to fill")
    ap.add_argument("--level", choices=("sheet", "target", "none"), default="target",
                    help="what the hero is squared to. ROT_DEG squares the panel to the CLOSED "
                         "sheet, so in a photograph of the DEPLOYED state nothing visible is "
                         "level: the sheet reads 7.9 deg off and the taped target 15.7. Default "
                         "'target' squares the TAPED RECTANGLE to the page, because it is the only "
                         "true rectangle in the picture and a rectangle is what the eye aligns to; "
                         "'sheet' levels the tessellation instead and leaves the target 7 deg off, "
                         "which reads as a crooked frame")
    ap.add_argument("--bar-mm", type=float, default=500.0,
                    help="length of the scale bar on panel (a), in millimetres")
    ap.add_argument("--bar-x", type=float, default=0.655,
                    help="left end of the scale bar, in axes fractions of panel (a)")
    ap.add_argument("--bar-y", type=float, default=0.185)
    ap.add_argument("--no-detail", action="store_true",
                    help="drop the magnified disagreement panel and run the key across six columns")
    ap.add_argument("--detail-state", choices=list(dv.STATES), default="Deployed")
    ap.add_argument("--detail-mm-per-bp", type=float, default=3.7,
                    help="millimetres per printed point in the magnified panel. ⚠ It is what "
                         "decides whether the disagreement is a length or a hairline: the state "
                         "row runs at about 29, where the widest sliver on the plate prints 1.5 pt")

    ap.add_argument("--no-boxes", action="store_true",
                    help="drop the two grouping frames, and the ~20 bp of height they cost")
    ap.add_argument("--hero-w", type=float, default=0.598,
                    help="photograph width as a fraction of the canvas; it pays for everything "
                         "else on the plate, so raise it only with the total height in view")
    ap.add_argument("--state-w", type=float, default=0.300)
    ap.add_argument("--chart-h", type=float, default=112.0)
    ap.add_argument("--no-retouch", action="store_true",
                    help="leave the empty corners, the glare and the lab objects in place")
    ap.add_argument("--sim-style", choices=sorted(dv.SIM_STYLES), default="ink")
    ap.add_argument("--align-sim", dest="align_sim", action="store_true", default=True)
    ap.add_argument("--raw-sim", dest="align_sim", action="store_false")
    ap.add_argument("--target-px", type=int, default=1200)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    fig, w_bp, h_bp, info = build(args)
    pdf, lo, hi = save_paper_figure(fig, args.out, w_bp, h_bp, dpi=args.dpi, strict_margins=True)
    plt.close(fig)
    with open(os.path.splitext(pdf)[0] + ".txt", "w") as fh:
        fh.write(caption_text(info, args, w_bp, h_bp))
    print("\n".join(prt.ledger_lines(info["ledger"])))
    print(f"wrote {pdf} ({w_bp:.1f} x {h_bp:.1f} bp, type {lo:.1f}-{hi:.1f} pt, "
          f"{os.path.getsize(pdf) / 1024.0:.0f} kB)")
    print(f"wrote {os.path.splitext(pdf)[0]}.txt")


if __name__ == "__main__":
    main()
