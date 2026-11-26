"""The design loop, and the two scales it couples. Replaces the old notation table.

A table lists symbols. It cannot show the one thing this work claims: that a design resolves
at TWO scales at once, that the two meet at the condensed hinge energy, and that the whole
chain is differentiable, so the loop closes. That is a picture, not a table, so it is drawn.

Eight boxes and one loop:

    design vector  ->  the cut sheet   |  one hinge          (the two scales, side by side)
                       hinge deformation |  condensed law
                            \_____________/
                                   ->  equilibrium  ->  objective  ->  gradient  -> back

A box is a quantity, an arrow is the operation that produces the next one. The plate carries
no legend and no reading rule: the caption states both, and everything here is the diagram.

Symbols that are side conditions rather than steps of the flow (the ideal deployment, the cut
frame, the linear reference law, the training domain) are named in OFF_PLATE and live only in the
glossary, which is the symbol reference. The plate is the relation reference.

    JAX_PLATFORMS=cpu conda run -n kgnn_mac \\
        python nff_paper/figures/plot_notation_map.py --audit
"""

import argparse
import os
import pickle
import re

import numpy as np

from nff.utils.figstyle import (FS_MICRO, GREEN, GREY, INK, ORANGE, PURPLE, RED, TEAL,
                                TEXT_W_BP, apply_charter, paper_rc, save_paper_figure, use_agg)

use_agg()
import matplotlib.pyplot as plt                                            # noqa: E402
from matplotlib.patches import (Circle, FancyArrowPatch, FancyBboxPatch,   # noqa: E402
                                Polygon as MplPolygon, Wedge)
from matplotlib.collections import LineCollection                          # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgb              # noqa: E402


# ══════════════════════════════════════════════════════════════════════════════
# Page geometry — authored at PRINTED size, so an authored point size IS printed points.
# Do not add bbox_inches to the save; it would break that guarantee.
# ══════════════════════════════════════════════════════════════════════════════

W_BP = TEXT_W_BP          # \includegraphics[width=\textwidth]
H_BP = 320.0              # + its caption sits well under \textheight 550.6

FS_LANE  = 6.6            # lane headers, set in caps
FS_TITLE = 6.8            # the name of a quantity
FS_SYM   = 6.8            # its symbol
FS_VERB  = 6.0            # the verb on an arrow. FS_MICRO is a HARD floor; this sits on it

# Band colours. Princeton charter, no blue.
C_DESIGN, C_SHEET, C_HINGE, C_EQ, C_GRAD = ORANGE, INK, TEAL, INK, PURPLE

STRESS = LinearSegmentedColormap.from_list("vm", ["#FDF3E7", "#F9B266", ORANGE, RED, "#7A1010"])

SPINE_CACHE = "data/outputs/notation_map_spine.pkl"
TEX = "paper/sn-article.tex"


def _tint(color, f):
    """``color`` mixed ``f`` of the way from white, for a card fill that keeps its family."""
    r, g, b = to_rgb(color)
    return (1 - f + f * r, 1 - f + f * g, 1 - f + f * b)


# ── Layout, all in points on the canvas ──────────────────────────────────────────────────────
# The left strip below X0 is reserved for the gradient riser, so no card may start left of it.
RISER_X = 10.0                              # the loop runs up this strip; keep it clear
X0, XE = 26.0, W_BP - 4.0
GUTTER = 14.0
COL_W = (XE - X0 - GUTTER) / 2              # the two scales, side by side
XA, XB = X0, X0 + COL_W + GUTTER
SPINE_W = 210.0                             # the boxes every design passes through
SPINE_X = (X0 + XE) / 2 - SPINE_W / 2
TH = 26.0                                   # thumbnail side

R0_Y, R0_H = 282.0, 30.0                    # design vector
R1_Y, R1_H = 224.0, 34.0                    # the cut sheet   | one hinge
R2_Y, R2_H = 168.0, 34.0                    # hinge deformation | condensed law
R3_Y, R3_H = 108.0, 34.0                    # equilibrium, where the two scales meet
R4_Y, R4_H = 58.0, 30.0                     # the objective
R5_Y, R5_H = 10.0, 28.0                     # the gradient, which closes the loop
BAND_TOP, BAND_BOT = 272.0, 158.0           # the tinted scale bands behind the branch

# ══════════════════════════════════════════════════════════════════════════════
# Primitives. Everything is placed in POINTS on a single full-canvas axes.
# ══════════════════════════════════════════════════════════════════════════════

def _wrap(text, fs, max_bp):
    """Break ``text`` on spaces so no line exceeds ``max_bp`` at ``fs``.

    A label that overruns its column is not shortened by the save; it is guillotined at the page
    edge. Wrapping is measured on the real glyph widths, never on a character count.
    """
    from nff.utils.figstyle import text_width_bp
    lines, cur = [], ""
    for w in text.split(" "):
        trial = f"{cur} {w}".strip()
        if cur and text_width_bp(trial, fs) > max_bp:
            lines.append(cur)
            cur = w
        else:
            cur = trial
    lines.append(cur)
    return "\n".join(lines)


def card(ax, x, y, w, h, title, symbol=None, color=INK, thumb_w=0.0):
    """A boxed quantity: optional thumbnail on the left, name and symbol stacked on the right.

    Returns the axes rect (in points) reserved for the thumbnail, or None.
    """
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0,rounding_size=2.2",
                                fc=_tint(color, 0.07), ec=color, lw=0.65, zorder=4))
    tx = x + 5.0
    if thumb_w > 0:
        tx = x + thumb_w + 8.0
    if symbol is None:
        ax.text(tx, y + h / 2, title, fontsize=FS_TITLE, color=INK, ha="left", va="center",
                zorder=6)
    else:
        ax.text(tx, y + h / 2 + 1.2, title, fontsize=FS_TITLE, color=INK, ha="left",
                va="bottom", zorder=6)
        ax.text(tx, y + h / 2 - 1.6, symbol, fontsize=FS_SYM, color=color, ha="left", va="top",
                zorder=6)
    if thumb_w > 0:
        return (x + 4.0, y + (h - thumb_w) / 2, thumb_w, thumb_w)
    return None


def verb(ax, x, y, text, color=GREY, ha="left", va="center", rotation=0, mask=True):
    """A verb on an arrow. It carries a white ground so it can sit on a rule or a box edge.

    ⚠ ``rotation_mode="anchor"`` is load-bearing for rotated labels. Without it matplotlib
    aligns the string flat and THEN rotates the aligned box, so a vertical label grows off the
    page instead of centring on the point it names.
    """
    ax.text(x, y, text, fontsize=FS_VERB, color=color, ha=ha, va=va, rotation=rotation,
            rotation_mode="anchor" if rotation else None,
            zorder=7, linespacing=1.2,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.7) if mask else None)


def arrow(ax, p0, p1, color=GREY, lw=0.8, dashed=False, rad=0.0, head=8.0):
    """A hairline arrow. ``head`` is the mutation scale, so the head stays ~3 bp long."""
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>",
                                 connectionstyle=f"arc3,rad={rad}", mutation_scale=head,
                                 lw=lw, color=color, zorder=3, shrinkA=0, shrinkB=0,
                                 linestyle=(0, (2.2, 1.6)) if dashed else "solid",
                                 joinstyle="round", capstyle="round"))


def elbow(ax, pts, color=GREY, lw=0.8, dashed=False):
    """A right-angled polyline with an arrow head on the last leg. Points are in points."""
    pts = np.asarray(pts, float)
    ax.add_collection(LineCollection([pts[:-1]], colors=color, linewidths=lw, zorder=3,
                                     linestyles=(0, (2.2, 1.6)) if dashed else "solid"))
    arrow(ax, pts[-2], pts[-1], color=color, lw=lw, dashed=dashed)


def _ax(fig, rect):
    """A thumbnail axes from a rect given in POINTS on the canvas."""
    x, y, w, h = rect
    a = fig.add_axes([x / W_BP, y / H_BP, w / W_BP, h / H_BP])
    a.set_xticks([]); a.set_yticks([]); a.set_facecolor("none")
    for s in a.spines.values():
        s.set_visible(False)
    return a


def _equal(a, pts, pad=0.10):
    pts = np.asarray(pts, float)
    mn, mx = pts.min(0), pts.max(0)
    c, r = (mn + mx) / 2, (mx - mn).max() / 2 * (1 + pad)
    a.set_xlim(c[0] - r, c[0] + r); a.set_ylim(c[1] - r, c[1] + r)
    a.set_aspect("equal"); a.axis("off")


# ══════════════════════════════════════════════════════════════════════════════
# Thumbnails. Each is a 20 x 20 bp glyph, drawn schematic rather than pictorial: at this
# size a faithful render is a smudge, and the plate's job is to name a quantity, not to
# show a result. The two that CAN carry real data (the cut pattern and the deployed sheet)
# take it from the same trained run Fig. fig:pipeline is built from.
# ══════════════════════════════════════════════════════════════════════════════

def thumb_cutpattern(a, S):
    from nff_paper.figures.plot_pipeline_overview import _shapely_to_patch
    g = S["geom_init"]
    a.add_patch(_shapely_to_patch(g["sheet"], fc=_tint(ORANGE, 0.16), ec="none", zorder=1))
    a.add_patch(_shapely_to_patch(g["cuts"], fc=INK, ec="none", zorder=2))
    b = np.asarray(g["sheet"].exterior.coords)
    _equal(a, b, pad=0.04)


def thumb_measures(a, _S):
    """One hinge opened: the axial gap a, the shear offset s and the relative rotation."""
    lo = np.array([[-1.6, -1.5], [-0.15, -1.5], [-0.15, 0.1], [-1.6, 0.1]])
    t = np.deg2rad(-16.0)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    hi = (np.array([[0.15, -1.5], [1.6, -1.5], [1.6, 0.1], [0.15, 0.1]])
          - np.array([0.15, 0.1])) @ R.T + np.array([0.42, 0.34])
    for p in (lo, hi):
        a.add_patch(MplPolygon(p, closed=True, fc=_tint(ORANGE, 0.20), ec=INK, lw=0.35, zorder=2))
    a.annotate("", xy=(0.40, -0.30), xytext=(-0.15, -0.30),
               arrowprops=dict(arrowstyle="<|-|>", color=RED, lw=0.5, mutation_scale=3.0))
    a.annotate("", xy=(0.28, 0.36), xytext=(0.28, -0.24),
               arrowprops=dict(arrowstyle="<|-|>", color=TEAL, lw=0.5, mutation_scale=3.0))
    _equal(a, np.array([[-1.7, -1.6], [1.8, 0.7]]), pad=0.02)


def thumb_fea(a, _S):
    """The same window, meshed, carrying a stress field that peaks at the cut tip."""
    n = 9
    xs = np.linspace(-1.5, 1.5, n)
    ys = np.linspace(-0.15, 1.35, n // 2 + 2)
    X, Y = np.meshgrid(xs, ys)
    keep = (X ** 2 + (Y + 0.15) ** 2) <= 1.5 ** 2
    r = np.hypot(X, Y + 0.15)
    a.pcolormesh(X, Y, np.where(keep, np.exp(-2.4 * r), np.nan), cmap=STRESS, shading="gouraud",
                 zorder=2, rasterized=True)
    segs = ([np.column_stack([xs, np.full_like(xs, y)]) for y in ys]
            + [np.column_stack([np.full_like(ys, x), ys]) for x in xs])
    a.add_collection(LineCollection(segs, colors=INK, linewidths=0.18, alpha=0.55, zorder=3))
    a.add_patch(Wedge((0.0, -0.15), 1.5, 0, 180, fc="none", ec=TEAL, lw=0.5, zorder=4))
    a.plot([0.0], [-0.15], "o", ms=1.1, color=INK, mew=0, zorder=5)
    _equal(a, np.array([[-1.8, -0.9], [1.8, 1.6]]), pad=0.02)


def thumb_network(a, _S):
    """The embedding: five inputs, hidden layers, one energy and one damage output."""
    cols = [(-1.35, 5), (-0.45, 6), (0.45, 6), (1.35, 2)]
    pos = [np.column_stack([np.full(k, x), np.linspace(-1.15, 1.15, k)]) for x, k in cols]
    segs = [[p, q] for A, B in zip(pos[:-1], pos[1:]) for p in A for q in B]
    a.add_collection(LineCollection(segs, colors=TEAL, linewidths=0.13, alpha=0.5, zorder=2))
    for k, P in enumerate(pos):
        a.plot(P[:, 0], P[:, 1], "o", ms=1.5, mew=0, zorder=3,
               color=ORANGE if k == 0 else (RED if k == len(pos) - 1 else TEAL))
    _equal(a, np.array([[-1.7, -1.4], [1.7, 1.4]]), pad=0.02)


def thumb_deployed(a, S):
    from nff_paper.figures.plot_pipeline_overview import _draw_panels
    _draw_panels(a, S["panel_half"], facecolor=_tint(ORANGE, 0.22), edgecolor=INK, lw=0.3,
                 zorder=3)
    a.add_patch(Circle(S["tcen"], S["trad"], fc="none", ec=GREEN, lw=0.55,
                       ls=(0, (2.2, 1.6)), zorder=4))
    _equal(a, S["panel_half"].reshape(-1, 2), pad=0.06)


def band(ax, x0, x1, label, color):
    """A tinted lane behind one scale, named once at its top left.

    Two bands say `two scales' with one visual element and four words, which is cheaper than
    saying it again on every box inside them.
    """
    ax.add_patch(FancyBboxPatch((x0, BAND_BOT), x1 - x0, BAND_TOP - BAND_BOT,
                                boxstyle="round,pad=0,rounding_size=3.0",
                                fc=_tint(color, 0.05), ec="none", zorder=1))
    ax.text(x0 + 6.0, BAND_TOP + 5.0, label.upper(), fontsize=FS_LANE, color=color,
            ha="left", va="center", fontweight="bold", zorder=2)


def build_figure(S, out_base, audit=False):
    apply_charter(**paper_rc(**{"axes.grid": False, "font.size": FS_TITLE}))
    fig = plt.figure(figsize=(W_BP / 72.0, H_BP / 72.0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W_BP); ax.set_ylim(0, H_BP); ax.axis("off")
    ax.set_facecolor("white")

    xca, xcb = XA + COL_W / 2, XB + COL_W / 2       # the two scale chains run down these
    xsp = SPINE_X + SPINE_W / 2                     # the spine runs down this
    rects = {}

    band(ax, XA - 4.0, XA + COL_W + 4.0, "the sheet", C_SHEET)
    band(ax, XB - 4.0, XB + COL_W + 4.0, "the hinge", C_HINGE)

    # ── the design, drawn through the charts so every iterate is buildable ───
    card(ax, SPINE_X, R0_Y, SPINE_W, R0_H, "design vector",
         r"$\boldsymbol{\theta}=(\boldsymbol{\kappa},\{\mathbf{g}_h\})\in\Theta$",
         C_DESIGN)

    # ── the two scales ──────────────────────────────────────────────────────
    rects["sheet"] = card(ax, XA, R1_Y, COL_W, R1_H, "the cut sheet",
                          r"$P_k$,  $h\in\mathcal{H}$", C_SHEET, thumb_w=TH)
    rects["eps"] = card(ax, XA, R2_Y, COL_W, R2_H, "hinge deformation",
                        r"$\boldsymbol{\varepsilon}_h=(a,s,\vartheta)$", C_SHEET, thumb_w=TH)
    rects["fea"] = card(ax, XB, R1_Y, COL_W, R1_H, "one hinge",
                        r"$\mathbf{g}_h=(w_{\mathrm{lig}},\alpha_h,\rho_f,w_c,r_{\mathrm{win}})$", C_HINGE,
                        thumb_w=TH)
    rects["law"] = card(ax, XB, R2_Y, COL_W, R2_H, "condensed hinge law",
                        r"$\boldsymbol{\chi},\chi_D;\ "
                        r"\mathcal{W}(\boldsymbol{\varepsilon};\mathbf{g}),\hat{D}$",
                        C_HINGE, thumb_w=TH)

    # Every branch verb is CENTRED on its column and straddles its own arrow. Set beside the
    # arrow instead, it has only half a column to live in and runs off the page.
    for xc, lab in [(xca, r"the cut layout $\mathbf{r},\boldsymbol{\tau}$"),
                    (xcb, r"one design $\mathbf{g}_h$ per hinge")]:
        arrow(ax, (xc, R0_Y), (xc, R1_Y + R1_H), color=C_DESIGN)
        verb(ax, xc, BAND_TOP - 7.0, lab, color=C_DESIGN, ha="center")
    ymid = (R1_Y + R2_Y + R2_H) / 2
    arrow(ax, (xca, R1_Y), (xca, R2_Y + R2_H), color=C_SHEET)
    verb(ax, xca, ymid,
         _wrap(r"the panels move rigidly, by $\mathbf{q}=(\mathbf{u}_k,\phi_k)$",
               FS_VERB, COL_W - 14.0), color=GREY, ha="center")
    arrow(ax, (xcb, R1_Y), (xcb, R2_Y + R2_H), color=C_HINGE)
    verb(ax, xcb, ymid,
         _wrap(r"resolved in three dimensions on a window $\Omega$ of radius "
               r"$r_{\mathrm{win}}$, then condensed once, offline", FS_VERB, COL_W - 14.0),
         color=GREY, ha="center")

    # ── where the two scales meet ───────────────────────────────────────────
    rects["eq"] = card(ax, SPINE_X, R3_Y, SPINE_W, R3_H, "equilibrium",
                       r"$\mathbf{q}^{\star}(\zeta_i;\boldsymbol{\theta})"
                       r"=\arg\min_{\mathcal{C}}\Pi$", C_EQ, thumb_w=TH)
    elbow(ax, [(xca, R2_Y), (xca, R3_Y + R3_H + 8.0), (SPINE_X + 56.0, R3_Y + R3_H + 8.0),
               (SPINE_X + 56.0, R3_Y + R3_H)], color=C_SHEET)
    elbow(ax, [(xcb, R2_Y), (xcb, R3_Y + R3_H + 8.0),
               (SPINE_X + SPINE_W - 56.0, R3_Y + R3_H + 8.0),
               (SPINE_X + SPINE_W - 56.0, R3_Y + R3_H)], color=C_HINGE)
    verb(ax, xca + 4.0, R2_Y - 10.0, "one deformation per hinge", color=C_SHEET)
    verb(ax, xcb - 4.0, R2_Y - 10.0, "one energy per hinge", color=C_HINGE, ha="right")

    # ── what a design is scored on, and the return path ─────────────────────
    card(ax, SPINE_X, R4_Y, SPINE_W, R4_H, "the objective",
         r"$\mathcal{L}=\mathcal{L}_{\mathrm{geom}}(\mathcal{P};\mathcal{T})"
         r"+\mathcal{L}_{\mathrm{phys}}[\boldsymbol{\varepsilon}_{ih}]$", INK)
    card(ax, SPINE_X, R5_Y, SPINE_W, R5_H, "the design gradient",
         r"$\partial\mathcal{L}/\partial\boldsymbol{\theta}$", C_GRAD)
    arrow(ax, (xsp, R3_Y), (xsp, R4_Y + R4_H), color=C_EQ)
    verb(ax, xsp, (R3_Y + R4_Y + R4_H) / 2,
         "scored on the shape it reaches, and on the path that reached it", color=GREY,
         ha="center")
    arrow(ax, (xsp, R4_Y), (xsp, R5_Y + R5_H), color=C_GRAD)
    verb(ax, xsp, (R4_Y + R5_Y + R5_H) / 2,
         "differentiated backwards through every step above", color=C_GRAD, ha="center")

    # the loop. Drawn heavier than the rest: that it CLOSES is the claim.
    elbow(ax, [(SPINE_X, R5_Y + R5_H / 2), (RISER_X, R5_Y + R5_H / 2),
               (RISER_X, R0_Y + R0_H / 2), (SPINE_X, R0_Y + R0_H / 2)], color=C_GRAD, lw=1.1)
    verb(ax, RISER_X, (R5_Y + R5_H / 2 + R0_Y + R0_H / 2) / 2,
         r"the optimiser steps $\mathbf{z},\boldsymbol{\ell},\boldsymbol{\omega}$, and the "
         r"charts return a cuttable, deployable sheet", color=C_GRAD, rotation=90,
         ha="center", va="bottom")

    for key, fn in [("sheet", thumb_cutpattern), ("eps", thumb_measures), ("fea", thumb_fea),
                    ("law", thumb_network), ("eq", thumb_deployed)]:
        a = _ax(fig, rects[key])
        fn(a, S)
        for art in a.get_children():
            art.set_clip_on(True)
            art.set_clip_box(a.bbox)

    if audit:
        _audit(fig, ax)
    save_paper_figure(fig, out_base, W_BP, H_BP, floor=FS_MICRO)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# The slide cut. Same graph, same symbols, re-authored for a 16:9 body.
#
# A figure authored for a page column is 370 bp wide and 320 tall. A slide body is 420 bp wide
# and about 200 tall, so including the page figure there scales it to 62 per cent and takes its
# 6.8 pt type down to 4.2. The fix is to author the graph again at the printed size the slide
# actually gives it, not to scale the page one: the columns widen from 163 bp to 196, the six
# rows become five by setting the objective beside the gradient, and every verb is cut to what
# a speaker does not say out loud anyway.
# ══════════════════════════════════════════════════════════════════════════════

W_SLIDE, H_SLIDE = 440.0, 206.0
COLW_S = 196.0
XA_S, XB_S = 30.0, 236.0                    # the sheet lane | the hinge lane
SPINE_XS, SPINE_WS = 110.0, 240.0           # what every design passes through
RISER_XS = 14.0                             # the loop runs up this strip
TH_S = 22.0
R1_S, R1_HS = 170.0, 26.0                   # design vector
R2_S, R2_HS = 128.0, 28.0                   # the cut sheet    | one hinge
R3_S, R3_HS = 86.0, 28.0                    # hinge deformation | condensed law
R4_S, R4_HS = 44.0, 28.0                    # equilibrium
R5_S, R5_HS = 6.0, 28.0                     # the objective    | the design gradient
BAND_TOP_S, BAND_BOT_S = 162.0, 82.0


def build_figure_slide(S, out_base):
    """The design loop at the size a 16:9 slide prints it."""
    global W_BP, H_BP, BAND_TOP, BAND_BOT
    W_BP, H_BP = W_SLIDE, H_SLIDE
    BAND_TOP, BAND_BOT = BAND_TOP_S, BAND_BOT_S

    apply_charter(**paper_rc(**{"axes.grid": False, "font.size": FS_TITLE}))
    fig = plt.figure(figsize=(W_BP / 72.0, H_BP / 72.0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W_BP); ax.set_ylim(0, H_BP); ax.axis("off")
    ax.set_facecolor("white")

    xca, xcb = XA_S + COLW_S / 2, XB_S + COLW_S / 2
    xsp = SPINE_XS + SPINE_WS / 2
    rects = {}

    band(ax, XA_S - 4.0, XA_S + COLW_S + 4.0, "the sheet", C_SHEET)
    band(ax, XB_S - 4.0, XB_S + COLW_S + 4.0, "the hinge", C_HINGE)

    card(ax, SPINE_XS, R1_S, SPINE_WS, R1_HS, "design vector",
         r"$\boldsymbol{\theta}=(\boldsymbol{\kappa},\{\mathbf{g}_h\})\in\Theta$", C_DESIGN)

    rects["sheet"] = card(ax, XA_S, R2_S, COLW_S, R2_HS, "the cut sheet",
                          r"$P_k$,  $h\in\mathcal{H}$", C_SHEET, thumb_w=TH_S)
    rects["eps"] = card(ax, XA_S, R3_S, COLW_S, R3_HS, "hinge deformation",
                        r"$\boldsymbol{\varepsilon}_h=(a,s,\vartheta)$", C_SHEET, thumb_w=TH_S)
    rects["fea"] = card(ax, XB_S, R2_S, COLW_S, R2_HS, "one hinge",
                        r"$\mathbf{g}_h=(w_{\mathrm{lig}},\alpha_h,\rho_f,w_c,r_{\mathrm{win}})$",
                        C_HINGE, thumb_w=TH_S)
    rects["law"] = card(ax, XB_S, R3_S, COLW_S, R3_HS, "condensed hinge law",
                        r"$\boldsymbol{\chi},\chi_D;\ "
                        r"\mathcal{W}(\boldsymbol{\varepsilon};\mathbf{g}),\hat{D}$",
                        C_HINGE, thumb_w=TH_S)

    # design vector into each lane
    for xc, lab in [(xca, r"the cut layout $\mathbf{r},\boldsymbol{\tau}$"),
                    (xcb, r"one design $\mathbf{g}_h$ per hinge")]:
        elbow(ax, [(xsp, R1_S), (xsp, R1_S - 7.0), (xc, R1_S - 7.0), (xc, R2_S + R2_HS)],
              color=C_DESIGN)
    verb(ax, xca, R1_S - 7.0, r"the cut layout $\mathbf{r},\boldsymbol{\tau}$",
         color=C_DESIGN, ha="center")
    verb(ax, xcb, R1_S - 7.0, r"one design $\mathbf{g}_h$ per hinge",
         color=C_DESIGN, ha="center")

    ymid = R3_S + R3_HS + 7.0
    arrow(ax, (xca, R2_S), (xca, R3_S + R3_HS), color=C_SHEET)
    verb(ax, xca, ymid, r"the panels move rigidly, by $\mathbf{q}=(\mathbf{u}_k,\phi_k)$",
         color=GREY, ha="center")
    arrow(ax, (xcb, R2_S), (xcb, R3_S + R3_HS), color=C_HINGE)
    verb(ax, xcb, ymid,
         _wrap(r"resolved in three dimensions on a window of radius $r_{\mathrm{win}}$, "
               r"then condensed offline", FS_VERB, COLW_S - 6.0), color=GREY, ha="center")

    rects["eq"] = card(ax, SPINE_XS, R4_S, SPINE_WS, R4_HS, "equilibrium",
                       r"$\mathbf{q}^{\star}(\zeta_i;\boldsymbol{\theta})"
                       r"=\arg\min_{\mathcal{C}}\Pi$", C_EQ, thumb_w=TH_S)
    yj = R4_S + R4_HS + 7.0
    elbow(ax, [(xca, R3_S), (xca, yj), (SPINE_XS + 52.0, yj),
               (SPINE_XS + 52.0, R4_S + R4_HS)], color=C_SHEET)
    elbow(ax, [(xcb, R3_S), (xcb, yj), (SPINE_XS + SPINE_WS - 52.0, yj),
               (SPINE_XS + SPINE_WS - 52.0, R4_S + R4_HS)], color=C_HINGE)
    verb(ax, xca - 26.0, yj, "one deformation per hinge", color=C_SHEET, ha="center")
    verb(ax, xcb + 12.0, yj, "one energy per hinge", color=C_HINGE, ha="center")

    # The objective beside the gradient, not under it: two 26 bp rows would not fit, and the
    # pair reads left to right as one backward pass.
    card(ax, XA_S, R5_S, COLW_S, R5_HS, "the objective",
         r"$\mathcal{L}=\mathcal{L}_{\mathrm{geom}}(\mathcal{P};\mathcal{T})"
         r"+\mathcal{L}_{\mathrm{phys}}[\boldsymbol{\varepsilon}_{ih}]$", INK)
    card(ax, XB_S, R5_S, COLW_S, R5_HS, "the design gradient",
         r"$\partial\mathcal{L}/\partial\boldsymbol{\theta}$", C_GRAD)
    elbow(ax, [(xsp, R4_S), (xsp, R5_S + R5_HS + 5.0), (xca, R5_S + R5_HS + 5.0),
               (xca, R5_S + R5_HS)], color=C_EQ)
    verb(ax, xca - 74.0, R5_S + R5_HS + 5.0, "scored on shape and on path",
         color=GREY, ha="left")
    arrow(ax, (XA_S + COLW_S, R5_S + R5_HS / 2), (XB_S, R5_S + R5_HS / 2), color=C_GRAD)
    verb(ax, (XA_S + COLW_S + XB_S) / 2, R5_S + R5_HS + 5.0, "backpropagation",
         color=C_GRAD, ha="center")

    # the loop. Drawn heavier than the rest: that it CLOSES is the claim.
    elbow(ax, [(XB_S, R5_S + R5_HS / 2), (RISER_XS, R5_S + R5_HS / 2),
               (RISER_XS, R1_S + R1_HS / 2), (SPINE_XS, R1_S + R1_HS / 2)],
          color=C_GRAD, lw=1.1)
    verb(ax, RISER_XS + 4.0, (R5_S + R1_S) / 2 + 14.0,
         r"the optimiser steps $\mathbf{z},\boldsymbol{\ell},\boldsymbol{\omega}$",
         color=C_GRAD, rotation=90, ha="center", va="bottom")

    for key, fn in [("sheet", thumb_cutpattern), ("eps", thumb_measures), ("fea", thumb_fea),
                    ("law", thumb_network), ("eq", thumb_deployed)]:
        a = _ax(fig, rects[key])
        fn(a, S)
        for art in a.get_children():
            art.set_clip_on(True)
            art.set_clip_box(a.bbox)

    save_paper_figure(fig, out_base, W_BP, H_BP, floor=FS_MICRO)


def _w(text, fs):
    from nff.utils.figstyle import text_width_bp
    return text_width_bp(text, fs)


# ══════════════════════════════════════════════════════════════════════════════
# Audit — the guard that keeps this plate and the glossary from drifting apart the
# way the table it replaces did.
# ══════════════════════════════════════════════════════════════════════════════

# Symbols on NEITHER the plate nor the glossary: each belongs to one argument and is defined
# where it is used. The user sanctioned this ("it's ok if some symbols are not presented").
DROPPED = {"Lambda", "sigma_y", "u_z", "u_0", "t", "Y", "B", "p", "ell_eg", "Gamma",
           "L_star", "theta_o", "ell_f", "ell_B", "varepsilon_f"}

# Glossary entries whose symbol is deliberately NOT drawn. Each is a side condition on the
# chain rather than a step of it, and drawing it was what made the first plate unreadable.
# ⚠ This set is the whole point of the audit: adding a glossary entry FAILS the build until
# its symbol is either put on the plate or listed here, on purpose.
OFF_PLATE = {"cut-frame", "linear-hinge-law", "training-domain", "main-cut", "secondary-cut"}

# The manuscript's own shorthands. Expanded before anything is compared, since the glossary
# writes \epsh where mathtext must write \varepsilon.
MACROS = {r"\epsh": r"\varepsilon", r"\wlig": r"w_{lig}", r"\rwin": r"r_{win}",
          r"\leg": r"\ell_{eg}"}
_BOLD = r"\\(?:bm|boldsymbol|mathbf|mathrm|mathsf|mathcal|mathit|hat|widehat|,|;|!|\s)"


def _canon(sym):
    """A symbol reduced to comparable letters, so \bm{\theta} and \boldsymbol{\theta} agree.

    The plate and the glossary cannot share markup: mathtext has no \bm and no user macros. What
    they must share is the SYMBOL, so both sides are stripped to it before comparison.
    """
    for k, v in MACROS.items():
        sym = sym.replace(k, v + " ")
    sym = re.sub(_BOLD, "", sym)
    return re.sub(r"[\\{}$^_\s]", "", sym)


def _glossary_symbols(tex=TEX):
    """The symbols each glossary entry carries in its label, keyed by entry."""
    src = open(tex, encoding="utf-8").read()
    src = "\n".join(l for l in src.split("\n") if not l.lstrip().startswith("%"))
    out = {}
    for m in re.finditer(r"\\item\[\\gentry\{([a-z-]+)\}\{[^}]*\}([^\]]*)\]", src):
        out[m.group(1)] = re.findall(r"\$([^$]+)\$", m.group(2))
    return out


def _audit(fig, ax):
    """What the plate prints, against what the glossary carries. The guard against re-drift.

    The table this plate replaced drifted from the body unnoticed for weeks. Nothing here is
    hand-maintained: the glossary is parsed out of the manuscript on every build.

    Raises:
        ValueError: if a glossary symbol is neither drawn nor listed in ``OFF_PLATE``.
    """
    gl = _glossary_symbols()
    carried = {k: v for k, v in gl.items() if v}
    printed = _canon(" ".join(t.get_text() for t in fig.findobj(plt.Text) if t.get_visible()))

    absent = []
    for key, syms in carried.items():
        head = re.split(r"[=,;(\\]", _canon(syms[0]))[0]
        if head and head not in printed and head.rstrip("h") not in printed:
            absent.append(key)
    stale = sorted(OFF_PLATE - set(absent))
    unlisted = sorted(set(absent) - OFF_PLATE)

    print(f"[glossary] {len(gl)} entries, {len(carried)} carrying a symbol")
    print(f"[plate]    {sum(1 for p in ax.patches if isinstance(p, FancyBboxPatch))} boxes, "
          f"{sum(1 for p in ax.patches if isinstance(p, FancyArrowPatch))} arrows, "
          f"{len(carried) - len(absent)} glossary symbols drawn")
    print(f"[off-plate] side conditions, glossary only: {sorted(OFF_PLATE)}")
    if stale:
        print(f"[off-plate] WARNING now drawn after all, drop from OFF_PLATE: {stale}")
    if unlisted:
        raise ValueError(f"glossary symbols neither drawn nor declared off-plate: {unlisted}. "
                         f"Put them on the plate, or add them to OFF_PLATE on purpose.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", default="overview")
    ap.add_argument("--out", default="paper/Images/notation_map")
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--slide", action="store_true",
                    help="the 16:9 cut, authored at the size a slide prints it")
    ap.add_argument("--refresh", action="store_true", help="re-solve instead of using the cache")
    args = ap.parse_args()

    if os.path.exists(SPINE_CACHE) and not args.refresh:
        with open(SPINE_CACHE, "rb") as f:
            S = pickle.load(f)
    else:
        from nff_paper.figures.plot_pipeline_overview import extract_spine
        S = extract_spine(args.config_name)
        os.makedirs(os.path.dirname(SPINE_CACHE), exist_ok=True)
        with open(SPINE_CACHE, "wb") as f:
            pickle.dump(S, f)

    if args.slide:
        build_figure_slide(S, args.out)
    else:
        build_figure(S, args.out, audit=args.audit)


if __name__ == "__main__":
    main()
