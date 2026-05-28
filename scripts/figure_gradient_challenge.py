#!/usr/bin/env python3
"""
Figure: gradient signal propagation challenge through the three-stage pipeline.

Produces: data/outputs/notebook_figures/gradient_challenge.png / .pdf
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.lines import Line2D

# ═══════════════════════════════════════════════════════════════════════════
# Style — Princeton palette, matches figure_experiment_types.py
# ═══════════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size":   10,
    "axes.titlesize":   12,
    "axes.titleweight": "bold",
})

P_ORANGE  = "#F58025"
P_DARK    = "#2B2B2B"
P_EDGE    = "#1A1A1A"
P_BG      = "#FFFFFF"
P_GREY    = "#CCCCCC"
P_LGREY   = "#F0F0F0"
P_RED     = "#C0392B"   # warning / vanishing signal
P_BLUE    = "#154360"   # implicit solver annotation

# ═══════════════════════════════════════════════════════════════════════════
# Layout constants
# ═══════════════════════════════════════════════════════════════════════════
FIG_W, FIG_H = 14, 5.4

# Five nodes along x: params | stage0 | stage1 | stage2 | loss
# Widths  (box half-widths in figure-fraction units)
XS = [0.08, 0.26, 0.46, 0.66, 0.88]   # box centres (figure fraction)
BOX_W = 0.14      # full width  (fig fraction)
BOX_H = 0.22      # full height (fig fraction)
Y_BOX  = 0.54     # vertical centre of all boxes
Y_FWD  = 0.82     # y of forward-pass arrows
Y_BWD  = 0.30     # y of backward gradient arrows
Y_ANNO = 0.10     # y of bottleneck annotations

# ═══════════════════════════════════════════════════════════════════════════
# Node definitions
# ═══════════════════════════════════════════════════════════════════════════
NODES = [
    dict(
        x=XS[0], label="Map\nParameters\n$\\theta$",
        color=P_ORANGE, textcolor="white",
        sublabel="trainable\nJAX PyTree",
    ),
    dict(
        x=XS[1], label="Stage 0\nInitial\nMapping",
        color=P_DARK, textcolor="white",
        sublabel="conformal / poly\nmap · Jacobian",
    ),
    dict(
        x=XS[2], label="Stage 1\nGeometric\nValidity",
        color=P_DARK, textcolor="white",
        sublabel="L-BFGS\ngeometric opt.",
    ),
    dict(
        x=XS[3], label="Stage 2\nPhysics\nSolver",
        color=P_DARK, textcolor="white",
        sublabel="L-BFGS\nenergy min.",
    ),
    dict(
        x=XS[4], label="Loss\n$\\mathcal{L}_{\\mathrm{Chamfer}}$",
        color=P_ORANGE, textcolor="white",
        sublabel="boundary\npoint cloud",
    ),
]

# ═══════════════════════════════════════════════════════════════════════════
# Bottleneck annotations between consecutive nodes
# ═══════════════════════════════════════════════════════════════════════════
BOTTLENECKS = [
    dict(
        x_mid=(XS[0]+XS[1])/2,
        title="Jacobian\npropagation",
        body="Dense $\\partial$-map:\nscales with\n# free params",
        color=P_BLUE,
        severity=0.25,
    ),
    dict(
        x_mid=(XS[1]+XS[2])/2,
        title="Implicit\ndifferentiation",
        body="KKT / fixed-point\n$\\nabla_{\\theta}\\,u^\\star$\nvia adj. method",
        color=P_RED,
        severity=0.40,
    ),
    dict(
        x_mid=(XS[2]+XS[3])/2,
        title="Nested\noptimizer",
        body="L-BFGS inside\nL-BFGS — gradient\ncheckpointing reqd.",
        color=P_RED,
        severity=0.18,
    ),
    dict(
        x_mid=(XS[3]+XS[4])/2,
        title="Long\ncomputational\ngraph",
        body="100s of load steps\nvia lax.scan —\ngradient accumulation",
        color=P_RED,
        severity=0.55,
    ),
]

# ═══════════════════════════════════════════════════════════════════════════
# Build figure
# ═══════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=P_BG)
ax  = fig.add_axes([0, 0, 1, 1])      # full-figure canvas
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect("auto")
ax.axis("off")
ax.set_facecolor(P_BG)

# ── Title ────────────────────────────────────────────────────────────────
ax.text(
    0.5, 0.97,
    "Gradient Propagation Challenge in the Three-Stage Differentiable Pipeline",
    ha="center", va="top",
    fontsize=13, fontweight="bold", color=P_EDGE,
    transform=ax.transAxes,
)

# ── Helper: draw a rounded box ────────────────────────────────────────────
def draw_box(cx, cy, w, h, facecolor, edgecolor=P_EDGE, lw=1.5, radius=0.018):
    box = FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=lw,
        transform=ax.transAxes, clip_on=False, zorder=10,
    )
    ax.add_patch(box)

# ── Helper: draw an arrow in axes-fraction coords ─────────────────────────
def draw_arrow(x0, y0, x1, y1, color, lw=2.0, alpha=1.0,
               arrowstyle="->, head_width=6, head_length=8",
               connectionstyle="arc3,rad=0.0"):
    arr = FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle=arrowstyle,
        color=color, linewidth=lw, alpha=alpha,
        mutation_scale=10,
        connectionstyle=connectionstyle,
        transform=ax.transAxes, clip_on=False, zorder=5,
    )
    ax.add_patch(arr)

# ── Draw nodes ────────────────────────────────────────────────────────────
for node in NODES:
    cx = node["x"]
    cy = Y_BOX
    draw_box(cx, cy, BOX_W, BOX_H, node["color"])
    ax.text(cx, cy + 0.025, node["label"],
            ha="center", va="center",
            fontsize=9.5, fontweight="bold", color=node["textcolor"],
            transform=ax.transAxes, zorder=15,
            multialignment="center", linespacing=1.35)
    ax.text(cx, cy - 0.072, node["sublabel"],
            ha="center", va="center",
            fontsize=7.5, color=node["textcolor"], alpha=0.82,
            transform=ax.transAxes, zorder=15,
            multialignment="center", linespacing=1.3,
            style="italic")

# ── Forward-pass arrows (top) ─────────────────────────────────────────────
fwd_y = Y_BOX + BOX_H/2 + 0.06
for i in range(len(NODES) - 1):
    x0 = NODES[i]["x"]   + BOX_W/2
    x1 = NODES[i+1]["x"] - BOX_W/2
    draw_arrow(x0, fwd_y, x1, fwd_y,
               color=P_EDGE, lw=2.0, alpha=1.0)

ax.text(0.5, fwd_y + 0.048, "Forward pass   ⟶",
        ha="center", va="bottom",
        fontsize=9, color=P_EDGE, fontweight="bold",
        transform=ax.transAxes)

# ── Backward gradient arrows (bottom) — thinning signal ───────────────────
# We draw from right to left; alpha decreases as signal weakens going left.
bwd_y  = Y_BOX - BOX_H/2 - 0.06

# cumulative severity: product of all bottleneck severity factors
cum_severity = [1.0]  # at loss node, full signal
sv = 1.0
for b in reversed(BOTTLENECKS):
    sv *= b["severity"]
    cum_severity.insert(0, sv)

# arrow widths proportional to signal strength (lw range 0.6 – 3.0)
def lw_from_alpha(a):
    return 0.7 + 2.5 * a

for i in range(len(NODES) - 1):
    # arrow goes from node i+1 to node i (right→left)
    x0 = NODES[i+1]["x"] - BOX_W/2
    x1 = NODES[i]["x"]   + BOX_W/2
    alpha_here = max(0.12, cum_severity[i+1])
    draw_arrow(x0, bwd_y, x1, bwd_y,
               color=P_ORANGE,
               lw=lw_from_alpha(alpha_here),
               alpha=min(1.0, alpha_here + 0.25),
               arrowstyle="->, head_width=5, head_length=7")

ax.text(0.5, bwd_y - 0.052, r"$\longleftarrow$   Backward gradient   $\nabla_\theta\,\mathcal{L}$",
        ha="center", va="top",
        fontsize=9, color=P_ORANGE, fontweight="bold",
        transform=ax.transAxes)

# gradient strength label at each gap
for i, sev in enumerate(cum_severity[:-1]):
    cx = NODES[i]["x"] + BOX_W/2 + (NODES[i+1]["x"] - BOX_W/2 - NODES[i]["x"] - BOX_W/2)/2
    pct = sev * 100
    label = f"<1%" if pct < 1 else f"~{int(round(pct))}%"
    ax.text(cx, bwd_y + 0.028, label,
            ha="center", va="bottom",
            fontsize=8, color=P_ORANGE, alpha=1.0,
            transform=ax.transAxes, fontweight="bold")

# ── Bottleneck annotations (below backward arrows) ─────────────────────────
for b in BOTTLENECKS:
    cx = b["x_mid"]
    cy_top = bwd_y - 0.07

    # Vertical tick line
    ax.plot([cx, cx], [bwd_y - 0.015, cy_top + 0.005],
            color=b["color"], lw=1.2, alpha=0.85,
            transform=ax.transAxes, zorder=6)

    # Annotation box
    box_h_anno = 0.19
    draw_box(cx, cy_top - box_h_anno/2, 0.155, box_h_anno,
             facecolor=P_LGREY, edgecolor=b["color"], lw=1.2, radius=0.012)

    ax.text(cx, cy_top - 0.022, b["title"],
            ha="center", va="top",
            fontsize=8.2, fontweight="bold", color=b["color"],
            transform=ax.transAxes, zorder=15,
            multialignment="center", linespacing=1.3)

    ax.text(cx, cy_top - 0.075, b["body"],
            ha="center", va="top",
            fontsize=7.5, color=P_DARK, alpha=0.9,
            transform=ax.transAxes, zorder=15,
            multialignment="center", linespacing=1.3)

# ── Legend ────────────────────────────────────────────────────────────────
legend_handles = [
    Line2D([0], [0], color=P_EDGE,   lw=2.0, label="Forward computation"),
    Line2D([0], [0], color=P_ORANGE, lw=2.5, label="Backward gradient  $\\nabla_\\theta\\mathcal{L}$"),
    mpatches.Patch(fc=P_LGREY, ec=P_RED,  lw=1.2, label="Gradient bottleneck (implicit solver / nested opt.)"),
    mpatches.Patch(fc=P_LGREY, ec=P_BLUE, lw=1.2, label="Dense Jacobian propagation"),
]
ax.legend(
    handles=legend_handles,
    loc="upper left",
    bbox_to_anchor=(0.01, 0.93),
    fontsize=8.2,
    frameon=True,
    framealpha=0.95,
    edgecolor=P_GREY,
    handlelength=2.0,
)

# ── Save ─────────────────────────────────────────────────────────────────
out_dir = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "data", "outputs", "notebook_figures")
)
os.makedirs(out_dir, exist_ok=True)

for ext in ("png", "pdf"):
    out_path = os.path.join(out_dir, f"gradient_challenge.{ext}")
    fig.savefig(out_path, dpi=250, bbox_inches="tight", facecolor=P_BG)
    print(f"Saved → {out_path}")
