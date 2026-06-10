"""
Corner-hinge visualization — single-Bézier-per-half design.

GEOMETRY (unit_2face_D, deployed state):
  Corner = (0,0), shared exactly by both faces (machine precision).
  Hinge axis = centroid-to-centroid direction = horizontal (1, 0).

  Face 0 (left): fold edges at 135° (upper) and 225° (lower) from corner.
  Face 1 (right): fold edges at 45° (upper) and 315° (lower) from corner.

HINGE PARAMETRISATION:
  Two cubic Bézier curves — one above the axis, one below:

    Upper curve:  bf1 → bc1 → bc2 → bf2
      bf1 on Face 0 upper fold edge at walk distance s0 from corner
      bf2 on Face 1 upper fold edge at walk distance s1 from corner
      bc1, bc2 in the upper void (y > 0), free

    Lower curve:  mirror of upper through the axis (or independently set)

  HINGE REGION = union of:
    • Face 0 triangular domain  (corner → bf1_upper → bf1_lower)
    • Void region between upper and lower Bézier curves
    • Face 1 triangular domain  (corner → bf2_upper → bf2_lower)

  bf1 / bf2 are always ON the fold edge (not interior) — necessary for a
  conforming hex mesh with shared boundary nodes.

Usage:
    python scripts/visualize_bezier_hinge.py [--show]
Output: scripts/bezier_hinge_gallery.png
"""

import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection

matplotlib.use("Agg")

# ── Princeton palette ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.size":        9,
    "axes.titlesize":   8.5,
    "axes.titleweight": "bold",
})
P_ORANGE  = "#F58025"   # loaded face
P_CLAMP   = "#2B2B2B"   # clamped face
P_EDGE    = "#1A1A1A"   # face outline
P_HINGE   = "#154360"   # hinge material fill
P_UPPER   = "#C0392B"   # upper Bézier curve
P_LOWER   = "#8E44AD"   # lower Bézier curve
P_AXIS    = "#7F8C8D"   # hinge axis
P_FOLD    = "#FDEBD0"   # fold zone on face
P_VALID   = "#27AE60"
P_INVALID = "#C0392B"
P_BG      = "#FFFFFF"

# ── Geometry constants ─────────────────────────────────────────────────────────
D = np.sqrt(2.0) / 2.0          # face half-diagonal (normalised)
FOLD_EDGE_LEN = D * np.sqrt(2)  # = 1.0 in normalised units

# Fold edge unit vectors from corner
F0_U = np.array([np.cos(np.radians(135)), np.sin(np.radians(135))])  # Face 0 upper
F0_L = np.array([np.cos(np.radians(225)), np.sin(np.radians(225))])  # Face 0 lower
F1_U = np.array([np.cos(np.radians(45)),  np.sin(np.radians(45))])   # Face 1 upper
F1_L = np.array([np.cos(np.radians(315)), np.sin(np.radians(315))])  # Face 1 lower


def _face0_diamond():
    c = np.array([-D, 0.0])
    return c + D * np.array([[-1,0],[0,-1],[1,0],[0,1]], dtype=float)

def _face1_diamond():
    c = np.array([+D, 0.0])
    return c + D * np.array([[-1,0],[0,-1],[1,0],[0,1]], dtype=float)


# ── Bézier helpers ─────────────────────────────────────────────────────────────

def _cubic2d(p0, p1, p2, p3, t):
    """Cubic Bézier in 2-D, vectorised over t ∈ [0,1]."""
    t = t[:, None]
    return ((1-t)**3*p0 + 3*(1-t)**2*t*p1 +
            3*(1-t)*t**2*p2 + t**3*p3)


def build_hinge(s0, s1,
                bc1_upper, bc2_upper,
                bc1_lower=None, bc2_lower=None,
                n=240):
    """
    Build the two Bézier boundary curves of the hinge.

    Parameters
    ----------
    s0, s1     : walk distance along Face 0 / Face 1 fold edges from corner.
                 Same value used for upper and lower fold edges.
    bc1_upper, bc2_upper : interior CPs of upper curve (above axis), as (x,y).
    bc1_lower, bc2_lower : interior CPs of lower curve (below axis).
                           If None, mirrored from upper through y = 0.

    Returns
    -------
    B_upper, B_lower : (n, 2) curve arrays
    bf1u, bf2u, bf1l, bf2l : anchor points (2,) each
    bc arrays as np (2,) vectors
    """
    bf1u = s0 * F0_U
    bf2u = s1 * F1_U
    bf1l = s0 * F0_L
    bf2l = s1 * F1_L

    bc1u = np.asarray(bc1_upper, float)
    bc2u = np.asarray(bc2_upper, float)

    if bc1_lower is None:
        bc1l = np.array([bc1u[0], -bc1u[1]])
        bc2l = np.array([bc2u[0], -bc2u[1]])
    else:
        bc1l = np.asarray(bc1_lower, float)
        bc2l = np.asarray(bc2_lower, float)

    t = np.linspace(0.0, 1.0, n)
    B_upper = _cubic2d(bf1u, bc1u, bc2u, bf2u, t)
    B_lower = _cubic2d(bf1l, bc1l, bc2l, bf2l, t)

    return B_upper, B_lower, bf1u, bf2u, bf1l, bf2l, bc1u, bc2u, bc1l, bc2l


def hinge_min_clearance(B_upper, B_lower):
    """
    Minimum signed clearance: min y of upper curve and max y of lower curve.
    Returns (min_y_upper, max_y_lower).  Both positive / negative = valid.
    Crossing: min_y_upper < max_y_lower → invalid.
    """
    return float(np.min(B_upper[:, 1])), float(np.max(B_lower[:, 1]))


# ── Drawing ────────────────────────────────────────────────────────────────────

_XLIM = (-0.26, 0.26)
_YLIM = (-0.26, 0.26)


def draw_case(ax, s0, s1,
              bc1_upper, bc2_upper,
              bc1_lower=None, bc2_lower=None,
              show_labels=False, title=""):
    """
    Draw one hinge case.

    The hinge region = two half-polygons:
      Upper half: corner → bf1_upper → B_upper → bf2_upper → corner
      Lower half: corner → bf2_lower → B_lower_reversed → bf1_lower → corner
    """
    if s0 is None:
        # Draw faces only (no hinge)
        _draw_faces(ax)
        ax.set_xlim(*_XLIM); ax.set_ylim(*_YLIM)
        ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.set_title(title, fontsize=8, pad=3)
        return

    result = build_hinge(s0, s1, bc1_upper, bc2_upper, bc1_lower, bc2_lower)
    B_upper, B_lower, bf1u, bf2u, bf1l, bf2l, bc1u, bc2u, bc1l, bc2l = result

    min_yu, max_yl = hinge_min_clearance(B_upper, B_lower)
    valid = (min_yu > 0) and (max_yl < 0)

    # ── Faces ──────────────────────────────────────────────────────────────────
    _draw_faces(ax)

    # ── Hinge axis ─────────────────────────────────────────────────────────────
    ax.axhline(0, color=P_AXIS, lw=0.7, ls='--', alpha=0.55, zorder=3)

    # ── Fold zone bands (near fold edges) ──────────────────────────────────────
    for fd, perp_sign in [(F0_U, -1), (F1_U, +1), (F0_L, +1), (F1_L, -1)]:
        bw = max(s0, s1) * 0.18
        pn = np.array([-fd[1], fd[0]]) * perp_sign
        band = np.array([[0,0], fd*max(s0,s1)*1.1,
                         fd*max(s0,s1)*1.1 + pn*bw, pn*bw])
        ax.add_patch(MplPolygon(band, closed=True, fc=P_FOLD,
                                ec='none', alpha=0.45, zorder=3))

    # ── Hinge fill ─────────────────────────────────────────────────────────────
    corner = np.array([0.0, 0.0])
    # upper half: corner → bf1u → B_upper → bf2u → corner
    upper_poly = np.vstack([[corner], [bf1u], B_upper, [bf2u], [corner]])
    # lower half: corner → bf2l → B_lower_rev → bf1l → corner
    lower_poly  = np.vstack([[corner], [bf2l], B_lower[::-1], [bf1l], [corner]])

    hcol = P_VALID if valid else P_INVALID
    for poly in [upper_poly, lower_poly]:
        ax.add_patch(MplPolygon(poly, closed=True,
                                fc=P_HINGE, ec=hcol, lw=1.5, alpha=0.82, zorder=4))

    # ── Bézier curves ──────────────────────────────────────────────────────────
    ax.plot(B_upper[:, 0], B_upper[:, 1], '-', color=P_UPPER, lw=1.8, zorder=5)
    ax.plot(B_lower[:, 0], B_lower[:, 1], '-', color=P_LOWER, lw=1.8, zorder=5)

    # ── Control polygons ────────────────────────────────────────────────────────
    for pts, col in [([bf1u, bc1u, bc2u, bf2u], P_UPPER),
                     ([bf1l, bc1l, bc2l, bf2l], P_LOWER)]:
        pts = np.array(pts)
        ax.plot(pts[:,0], pts[:,1], '--', color=col, lw=0.75, alpha=0.55, zorder=5)

    # ── Endpoint markers (bf) ──────────────────────────────────────────────────
    for pt in [bf1u, bf2u, bf1l, bf2l]:
        ax.plot(*pt, 's', color=P_EDGE, ms=4.5, zorder=6)

    # ── Interior CP markers (bc) ───────────────────────────────────────────────
    for pt, col in [(bc1u, P_UPPER), (bc2u, P_UPPER),
                    (bc1l, P_LOWER), (bc2l, P_LOWER)]:
        ax.plot(*pt, 'o', color=col, ms=4.5,
                markeredgecolor='white', markeredgewidth=0.5, zorder=6)

    # ── Corner vertex ──────────────────────────────────────────────────────────
    ax.plot(0, 0, 'o', color=P_EDGE, ms=5.0, zorder=7)

    if show_labels:
        _annotate(ax, s0, s1, bf1u, bf2u, bc1u, bc2u)

    # ── Axes ────────────────────────────────────────────────────────────────────
    ax.set_xlim(*_XLIM); ax.set_ylim(*_YLIM)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)

    tc = P_VALID if valid else P_INVALID
    ax.set_title(f"{title}\nmin_y↑={min_yu:+.3f}  max_y↓={max_yl:+.3f}",
                 fontsize=8, color=tc, pad=3)


def _draw_faces(ax):
    f0 = _face0_diamond(); f1 = _face1_diamond()
    ax.add_patch(MplPolygon(f0, closed=True, fc=P_CLAMP,  ec=P_EDGE, lw=1.0, zorder=2))
    ax.add_patch(MplPolygon(f1, closed=True, fc=P_ORANGE, ec=P_EDGE, lw=1.0, zorder=2))
    ax.plot(0, 0, 'o', color=P_EDGE, ms=4.5, zorder=7)


def _annotate(ax, s0, s1, bf1u, bf2u, bc1u, bc2u):
    """Parameter labels for the reference case."""
    # s0 arrow: corner → bf1u along fold edge
    ax.annotate("", xy=bf1u, xytext=(0,0),
                arrowprops=dict(arrowstyle='->', color=P_UPPER, lw=1.2), zorder=9)
    ax.text(bf1u[0]*0.48 - 0.018, bf1u[1]*0.48,
            r"$s_0$", fontsize=8, color=P_UPPER, ha='right', va='center', zorder=9)

    # s1 arrow: corner → bf2u
    ax.annotate("", xy=bf2u, xytext=(0,0),
                arrowprops=dict(arrowstyle='->', color=P_UPPER, lw=1.2), zorder=9)
    ax.text(bf2u[0]*0.48 + 0.018, bf2u[1]*0.48,
            r"$s_1$", fontsize=8, color=P_UPPER, ha='left', va='center', zorder=9)

    # bc1 label
    ax.text(bc1u[0] - 0.018, bc1u[1] + 0.010,
            r"$bc_1$", fontsize=7.5, color=P_UPPER, ha='right', va='bottom', zorder=9)

    # bc2 label
    ax.text(bc2u[0] + 0.018, bc2u[1] + 0.010,
            r"$bc_2$", fontsize=7.5, color=P_UPPER, ha='left', va='bottom', zorder=9)

    # bf1 label
    ax.text(bf1u[0] - 0.012, bf1u[1] + 0.012,
            r"$bf_1$", fontsize=7.5, color=P_EDGE, ha='right', va='bottom', zorder=9)

    # bf2 label
    ax.text(bf2u[0] + 0.012, bf2u[1] + 0.012,
            r"$bf_2$", fontsize=7.5, color=P_EDGE, ha='left', va='bottom', zorder=9)

    # axis label
    ax.text(_XLIM[1] - 0.005, 0.008, "axis", fontsize=7, color=P_AXIS,
            ha='right', va='bottom', alpha=0.8, zorder=9)

    # face labels
    ax.text(-D*0.58, -0.01, "F0", fontsize=7.5, color='white',
            ha='center', va='center', fontweight='bold', zorder=8)
    ax.text(+D*0.58, -0.01, "F1", fontsize=7.5, color=P_EDGE,
            ha='center', va='center', fontweight='bold', zorder=8)


# ── Gallery cases ──────────────────────────────────────────────────────────────
# Helper: default CP positions for given bf1/bf2 and a scale factor
def _cps(s0, s1, h_scale=0.5, lateral=0.0):
    """
    h_scale > 1: CPs above the bf endpoint height → bowing outward
    h_scale < 1: CPs below bf height  → converging/pinching toward axis
    lateral: lateral asymmetry (positive shifts bc2 right / bc1 left)
    """
    bf1u = s0 * F0_U
    bf2u = s1 * F1_U
    x1 = bf1u[0] + (bf2u[0] - bf1u[0]) / 3.0 - lateral
    x2 = bf1u[0] + 2.0 * (bf2u[0] - bf1u[0]) / 3.0 + lateral
    avg_y = (bf1u[1] + bf2u[1]) / 2.0
    h = h_scale * avg_y
    return (x1, h), (x2, h)

# s values
_S  = 0.26   # default walk distance
_SL = 0.42   # large walk
_SS = 0.10   # small walk

# Pre-compute CP sets for default
_bc1_def, _bc2_def = _cps(_S, _S, h_scale=0.50)  # default (converging)
_bc1_wide, _bc2_wide = _cps(_S, _S, h_scale=1.50) # bowing outward
_bc1_tight, _bc2_tight = _cps(_S, _S, h_scale=0.08) # tight (near axis)
_bc1_L, _bc2_L = _cps(_SL, _SL, h_scale=0.50)
_bc1_S, _bc2_S = _cps(_SS, _SS, h_scale=0.50)
_bc1_asym, _bc2_asym = _cps(0.15, 0.38, h_scale=0.50)

CASES = [
    # (s0, s1, bc1_upper, bc2_upper, bc1_lower, bc2_lower, title)
    # bc1/bc2_lower = None → symmetric mirror

    # Row 1: walk parameter (face domain size)
    (_S,  _S,  _bc1_def,  _bc2_def,  None, None,
     "reference\ns₀=s₁=0.26, h·scale=0.5"),

    (_SL, _SL, _bc1_L, _bc2_L, None, None,
     "large walk  s=0.42\nmore face domain"),

    (_SS, _SS, _bc1_S, _bc2_S, None, None,
     "small walk  s=0.10\nless face domain"),

    (0.0, 0.0, (0, 0.001), (0, 0.001), None, None,
     "s=0  (degenerate)\nbf at corner — stress conc."),

    # Row 2: curve shape (CP height)
    (_S, _S, _bc1_wide, _bc2_wide, None, None,
     "bowing outward  h·scale=1.5\nfat hinge, CPs above bf"),

    (_S, _S, _bc1_tight, _bc2_tight, None, None,
     "tight neck  h·scale=0.08\nCPs near axis"),

    (_S, _S, _cps(_S, _S, h_scale=0.5, lateral=0.08)[0],
              _cps(_S, _S, h_scale=0.5, lateral=0.08)[1], None, None,
     "lateral spread  (bc1 left, bc2 right)\nwidened void coverage"),

    (_S, _S,
     (_cps(_S,_S)[0][0], _cps(_S,_S)[0][1]*1.8),  # bc1 high
     (_cps(_S,_S)[1][0], _cps(_S,_S)[1][1]*0.2),  # bc2 low → S-curve
     None, None,
     "S-curve  (bc1 high, bc2 low)\nasymmetric along arc"),

    # Row 3: asymmetric and special cases
    (0.15, 0.38, _bc1_asym, _bc2_asym, None, None,
     "asymmetric walk  s₀≠s₁\n0.15 vs 0.38"),

    (_S, _S, _bc1_def, _bc2_def,
     # independent lower: flatter than upper
     (_cps(_S,_S,0.08)[0]), (_cps(_S,_S,0.08)[1]),
     "upper converging / lower flat\nindependent lower CPs"),

    (_S, _S, _bc1_def, _bc2_def,
     # lower: bowing outward
     (_cps(_S,_S,1.5)[0][0], -_cps(_S,_S,1.5)[0][1]),
     (_cps(_S,_S,1.5)[1][0], -_cps(_S,_S,1.5)[1][1]),
     "upper converging / lower bowing\nasymmetric stiffness"),

    # INVALID: CPs cross axis (upper dips below axis)
    (_S, _S, (_cps(_S,_S)[0][0], -0.04), (_cps(_S,_S)[1][0], -0.04),
     None, None,
     "INVALID: upper CPs below axis\ncurves cross → min_y < 0"),
]


# ── Main ───────────────────────────────────────────────────────────────────────

def main(show=False):
    nrows, ncols = 3, 4
    fig = plt.figure(figsize=(18, 14), facecolor=P_BG)

    fig.suptitle(
        "Corner hinge — single Bézier per half  (unit_2face_D, deployed state,  "
        "simulation closes mechanism CCW)\n"
        r"Upper curve (red): $bf_1 \to bc_1 \to bc_2 \to bf_2$  above axis   "
        r"·   Lower curve (purple): mirror below axis (or independent)   "
        r"·   Hinge = face domains + void between curves",
        fontsize=10, y=0.98,
    )

    axes = []
    for r in range(nrows):
        for c in range(ncols):
            ax = fig.add_subplot(nrows, ncols, r * ncols + c + 1)
            ax.set_facecolor(P_BG)
            axes.append(ax)

    for idx, ax in enumerate(axes):
        if idx >= len(CASES):
            ax.axis('off'); continue
        case = CASES[idx]
        s0, s1, bc1u, bc2u, bc1l, bc2l, title = case
        draw_case(ax, s0, s1, bc1u, bc2u, bc1l, bc2l,
                  show_labels=(idx == 0),
                  title=title)

    # ── Legend ─────────────────────────────────────────────────────────────────
    handles = [
        mpatches.Patch(fc=P_CLAMP,  ec=P_EDGE, label="F0 — clamped face"),
        mpatches.Patch(fc=P_ORANGE, ec=P_EDGE, label="F1 — loaded face"),
        mpatches.Patch(fc=P_FOLD,   ec='none', alpha=0.6, label="fold zone"),
        mpatches.Patch(fc=P_HINGE,  ec=P_VALID, lw=1.3, label="hinge region (valid)"),
        mpatches.Patch(fc=P_HINGE,  ec=P_INVALID, lw=1.3, label="hinge region (invalid)"),
        plt.Line2D([0],[0], color=P_UPPER, lw=1.6,
                   label=r"upper Bézier  $(bf_1 \to bc_1 \to bc_2 \to bf_2)$"),
        plt.Line2D([0],[0], color=P_LOWER, lw=1.6,
                   label="lower Bézier (below axis)"),
        plt.Line2D([0],[0], color=P_AXIS, lw=0.8, ls='--',
                   label="hinge axis (centroid → centroid)"),
        plt.Line2D([0],[0], marker='s', color='w', ms=4.5,
                   markerfacecolor=P_EDGE, label=r"$bf$ — fold-edge anchors"),
        plt.Line2D([0],[0], marker='o', color='w', ms=4.5,
                   markerfacecolor=P_UPPER, label=r"$bc$ — interior control points"),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=5,
               fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    out = "scripts/bezier_hinge_gallery.png"
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=P_BG)
    print(f"Saved: {out}")
    if show:
        matplotlib.use("TkAgg")
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    main(show=args.show)
