#!/usr/bin/env python3
"""
Figure: canonical experiment types on the 2×2 RDQK_D kirigami tessellation.
Standalone — no project imports required.

Produces: data/outputs/notebook_figures/experiment_types.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPolygon, FancyArrowPatch

# ═══════════════════════════════════════════════════════════════════════════
# Style — matches Princeton palette used throughout the project
# ═══════════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
})

P_ORANGE  = "#F58025"   # Princeton orange — free face fill
P_CLAMP   = "#2B2B2B"   # near-black — clamped (Dirichlet) face
P_EDGE    = "#1A1A1A"   # polygon edge
P_HINGE   = "#FFFFFF"   # hinge-dot fill
P_CCW     = "#C0392B"   # deep crimson — CCW moment (+Mz)
P_CW      = "#154360"   # dark navy   — CW  moment (−Mz)
P_FORCE   = "#1C1C2E"   # near-black  — applied force (Fx / Fy)
P_BG      = "#FFFFFF"

# ═══════════════════════════════════════════════════════════════════════════
# Geometry — 2×2 RDQK_D tessellation (16 diamond faces)
#
# Layout:
#   col0  col1  col2  col3
#    11    10    15    14   row3  (y ≈ 4.95)
#     8     9    12    13   row2  (y ≈ 3.54)
#     3     2     7     6   row1  (y ≈ 2.12)
#     0     1     4     5   row0  (y ≈ 0.71)
#
# Hubs (degree-4): 2, 7, 9, 12
# Corners (degree-2): 0, 11, 13, 14
# ═══════════════════════════════════════════════════════════════════════════
D = np.sqrt(2.0) / 2.0          # half-diagonal of each diamond face ≈ 0.707

CENT = np.array([               # face centroid positions
    [1*D, 1*D],  # 0  col0 row0
    [3*D, 1*D],  # 1  col1 row0
    [3*D, 3*D],  # 2  col1 row1  (hub)
    [1*D, 3*D],  # 3  col0 row1
    [5*D, 1*D],  # 4  col2 row0
    [7*D, 1*D],  # 5  col3 row0
    [7*D, 3*D],  # 6  col3 row1
    [5*D, 3*D],  # 7  col2 row1  (hub)
    [1*D, 5*D],  # 8  col0 row2
    [3*D, 5*D],  # 9  col1 row2  (hub)
    [3*D, 7*D],  # 10 col1 row3
    [1*D, 7*D],  # 11 col0 row3
    [5*D, 5*D],  # 12 col2 row2  (hub)
    [7*D, 5*D],  # 13 col3 row2
    [7*D, 7*D],  # 14 col3 row3
    [5*D, 7*D],  # 15 col2 row3
])

HINGES = [
    # Internal rings (within each 2×2 unit tile)
    (0,1),(1,2),(2,3),(3,0),
    (4,5),(5,6),(6,7),(4,7),
    (8,9),(9,10),(10,11),(8,11),
    (12,13),(13,14),(14,15),(12,15),
    # Cross-tile connections
    (1,4),(3,8),(2,7),(7,12),(9,12),(10,15),
    (2,5),(6,9),   # diagonal cross-hinges
]


def _diamond(i):
    """Return the 4 vertices of face i (diamond = square rotated 45°)."""
    cx, cy = CENT[i]
    return [(cx, cy - D), (cx + D, cy), (cx, cy + D), (cx - D, cy)]


def draw_tessellation(ax, *, clamped, moment_loads, force_loads):
    """
    Draw the full 16-face tessellation with load indicators.

    Parameters
    ----------
    clamped       : iterable of face indices with Dirichlet BC
    moment_loads  : list of (face_idx, sign)   sign = +1 CCW, -1 CW
    force_loads   : list of (face_idx, fx, fy) unit-direction components
    """
    clamped = set(clamped)

    # ── Faces ────────────────────────────────────────────────────────────
    for i in range(16):
        fc = P_CLAMP if i in clamped else P_ORANGE
        lw = 1.8 if i in clamped else 1.1
        ax.add_patch(MplPolygon(
            _diamond(i), closed=True,
            facecolor=fc, edgecolor=P_EDGE, linewidth=lw, zorder=10,
        ))

    # ── Hinge mid-point dots ──────────────────────────────────────────────
    for f1, f2 in HINGES:
        mid = (CENT[f1] + CENT[f2]) / 2.0
        ax.plot(*mid, "o", ms=3.2, color=P_HINGE,
                markeredgecolor=P_EDGE, markeredgewidth=0.6, zorder=16)

    # ── Moment arcs ───────────────────────────────────────────────────────
    r = 0.28    # arc radius in data coordinates
    for fi, sign in moment_loads:
        cx, cy = CENT[fi]
        col = P_CCW if sign > 0 else P_CW
        if sign > 0:                        # CCW — arc sweeps counterclockwise
            start = (cx + r,       cy - 0.5 * r)
            end   = (cx - 0.5 * r, cy + r)
            rad   = 0.65
        else:                               # CW  — arc sweeps clockwise
            start = (cx - r,       cy - 0.5 * r)
            end   = (cx + 0.5 * r, cy + r)
            rad   = -0.65
        ax.add_patch(FancyArrowPatch(
            start, end,
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="->, head_width=0.15, head_length=0.20",
            color=col, linewidth=2.8, mutation_scale=11, zorder=30,
        ))

    # ── Force arrows ──────────────────────────────────────────────────────
    # Each component (Fx, Fy) gets its own arrow originating from the
    # corresponding face edge, pointing outward.
    alen  = 0.52   # arrow shaft length
    apad  = 0.90   # start fraction along half-diagonal toward edge

    for fi, fx, fy in force_loads:
        cx, cy = CENT[fi]
        if abs(fx) > 1e-10:
            sx = cx + np.sign(fx) * D * apad
            sy = cy
            ex = sx + np.sign(fx) * alen
            ey = sy
            ax.annotate("",
                xy=(ex, ey), xytext=(sx, sy),
                arrowprops=dict(
                    arrowstyle="->, head_width=0.14, head_length=0.18",
                    color=P_FORCE, lw=2.2,
                    mutation_scale=12,
                ),
                zorder=32,
            )
        if abs(fy) > 1e-10:
            sx = cx
            sy = cy + np.sign(fy) * D * apad
            ex = sx
            ey = sy + np.sign(fy) * alen
            ax.annotate("",
                xy=(ex, ey), xytext=(sx, sy),
                arrowprops=dict(
                    arrowstyle="->, head_width=0.14, head_length=0.18",
                    color=P_FORCE, lw=2.2,
                    mutation_scale=12,
                ),
                zorder=32,
            )

    # ── Axes ──────────────────────────────────────────────────────────────
    pad = 1.15
    ax.set_xlim(-pad, 8 * D + pad)
    ax.set_ylim(-pad, 8 * D + pad)
    ax.set_aspect("equal")
    ax.axis("off")


# ═══════════════════════════════════════════════════════════════════════════
# Experiment definitions
#   Each panel represents a canonical loading family from the benchmark suite.
# ═══════════════════════════════════════════════════════════════════════════
# One representative per main loading category (suite_2x2_rdqk, 100 problems)
# ═══════════════════════════════════════════════════════════════════════════
EXPERIMENTS = [
    # ── Row 1: moment-only loadings ────────────────────────────────────────
    dict(
        label="(a)",
        title="Single-face torque",
        desc="Hub clamp (9) · CCW moment on\none direct neighbor (6)  —  p011",
        clamped=(9,),
        moments=[(6, +1)],
        forces=[],
    ),
    dict(
        label="(b)",
        title="Multi-neighbor moments",
        desc="Hub clamp (9) · CCW/CW/CCW on\n3 non-adjacent faces (2, 7, 12)  —  p001",
        clamped=(9,),
        moments=[(2, +1), (7, -1), (12, +1)],
        forces=[],
    ),
    dict(
        label="(c)",
        title="Corner global twist",
        desc="Corner clamp (0) · alternating $M_z$\nat 3 remote corners (5, 11, 14)  —  p008",
        clamped=(0,),
        moments=[(5, +1), (11, -1), (14, +1)],
        forces=[],
    ),
    # ── Row 2: force & mixed loadings ─────────────────────────────────────
    dict(
        label="(d)",
        title="Uniaxial edge tension",
        desc="Corner clamp (0) · $F_x = +3$\non full right edge (5, 6, 13, 14)  —  p032",
        clamped=(0,),
        moments=[],
        forces=[(5, 1, 0), (6, 1, 0), (13, 1, 0), (14, 1, 0)],
    ),
    dict(
        label="(e)",
        title="Biaxial tension",
        desc="Hub clamp (9) · $F_x$ (right edge)\n+ $F_y$ (top edge) simultaneously  —  p009",
        clamped=(9,),
        moments=[],
        forces=[
            (5,  1, 0), (6,  1, 0), (13, 1, 0), (14, 1, 0),   # right edge Fx
            (10, 0, 1), (11, 0, 1), (14, 0, 1), (15, 0, 1),   # top  edge Fy
        ],
    ),
    dict(
        label="(f)",
        title="Mixed torque + tension",
        desc="Hub clamp (9) · CCW on neighbor (6)\n+ right-edge $F_x$  —  p051",
        clamped=(9,),
        moments=[(6, +1)],
        forces=[(5, 1, 0), (6, 1, 0), (13, 1, 0), (14, 1, 0)],
    ),
]

# ═══════════════════════════════════════════════════════════════════════════
# Build figure
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(13.5, 9.8), facecolor=P_BG)
fig.patch.set_facecolor(P_BG)

for ax, exp in zip(axes.flat, EXPERIMENTS):
    ax.set_facecolor(P_BG)
    draw_tessellation(
        ax,
        clamped=exp["clamped"],
        moment_loads=exp["moments"],
        force_loads=exp["forces"],
    )

    # Panel label  "(a)" — upper-left, large bold
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xspan = xlim[1] - xlim[0]
    yspan = ylim[1] - ylim[0]

    ax.text(
        xlim[0] + 0.03 * xspan,
        ylim[1] - 0.03 * yspan,
        exp["label"],
        fontsize=13, fontweight="bold", color="#1A1A1A",
        va="top", ha="left", zorder=50,
    )

    # Title (bold) + description (regular) — centered above tessellation
    ax.set_title(
        exp["title"] + "\n" + exp["desc"],
        fontsize=9.5, fontweight="normal", color="#1A1A1A",
        linespacing=1.4, pad=6,
    )
    # Make first line of title bold
    ax.title.set_fontweight("bold")


# ── Legend ────────────────────────────────────────────────────────────────
legend_handles = [
    mpatches.Patch(fc=P_ORANGE, ec=P_EDGE, linewidth=0.8, label="Free face"),
    mpatches.Patch(fc=P_CLAMP,  ec=P_EDGE, linewidth=0.8, label="Clamped face (Dirichlet BC)"),
    mpatches.Patch(fc=P_CCW,    ec="none",                label="CCW moment (+$M_z$)"),
    mpatches.Patch(fc=P_CW,     ec="none",                label="CW moment (−$M_z$)"),
    mpatches.Patch(fc=P_FORCE,  ec="none",                label="Applied force ($F_x$ / $F_y$)"),
]
fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=5,
    fontsize=8.8,
    frameon=True,
    framealpha=0.95,
    edgecolor="#CCCCCC",
    bbox_to_anchor=(0.5, 0.00),
    borderpad=0.8,
    handlelength=1.6,
    columnspacing=1.2,
)

fig.suptitle(
    "Representative loading types — 2×2 RDQK_D benchmark suite (100 problems)",
    fontsize=10.5, fontweight="bold", color="#1A1A1A", y=1.002,
)

plt.tight_layout(rect=[0, 0.055, 1, 1], h_pad=3.0, w_pad=1.5)

# ── Save ─────────────────────────────────────────────────────────────────
out_dir = os.path.join(
    os.path.dirname(__file__), "..", "data", "outputs", "notebook_figures"
)
out_dir = os.path.normpath(out_dir)
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "experiment_types.png")
fig.savefig(out_path, dpi=250, bbox_inches="tight", facecolor=P_BG)
print(f"Saved → {out_path}")
