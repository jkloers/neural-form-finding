"""nff/sofa/hinge_viz.py — shared Princeton palette + hinge plotting helpers.

Client-side (kgnn_mac); matplotlib only, never imports ``Sofa``. Every hinge-run
figure script (`visualize_hinge_run`, `animate_closing`, `extract_hinge_stiffness`)
draws from this single source so the palette and the mesh/Bézier primitives are
defined exactly once.
"""
from __future__ import annotations

import numpy as np

# ── Princeton palette (mirrors nff/utils/visualization.py) ─────────────────────
P_ORANGE  = "#F58025"   # free / loaded panels AND hinge mesh fill
P_GRAY    = "#6C757D"   # clamped face
P_EDGE    = "#1A1A1A"   # mesh edges
P_DARK    = "#1A1A1A"   # text / lines
P_BG      = "#FFFFFF"   # background
ARROW_RED = "#D62828"   # positive-moment arrow
GREEN_UP  = "#1B6B3A"   # upper Bézier strip (dark Princeton green)
GREEN_LO  = "#5CB87F"   # lower Bézier strip (light Princeton green)
CP_COL    = "#16324A"   # Bézier control points / polygons


def quad_bezier(p0, c, p2, n: int = 200) -> np.ndarray:
    """Sample a quadratic Bézier arc ``(1-t)²p0 + 2(1-t)t·c + t²p2``."""
    t = np.linspace(0.0, 1.0, n)[:, None]
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * c + t ** 2 * p2


def bottom_tris(nodes: np.ndarray, tets: np.ndarray):
    """Boundary triangles on the bottom (z≈min) layer + their owning tet index."""
    faces: dict = {}
    for ti, t in enumerate(tets):
        for c in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)):
            f = tuple(sorted((int(t[c[0]]), int(t[c[1]]), int(t[c[2]]))))
            faces.setdefault(f, []).append(ti)
    zmin = nodes[:, 2].min()
    bot = nodes[:, 2] < zmin + 1e-6
    tri, owner = [], []
    for f, tis in faces.items():
        if len(tis) == 1 and bot[f[0]] and bot[f[1]] and bot[f[2]]:
            tri.append(f); owner.append(tis[0])
    return np.array(tri), np.array(owner)


def edges(tri: np.ndarray) -> np.ndarray:
    """Unique undirected edges of a triangle-index array."""
    e = np.sort(np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [0, 2]]]), axis=1)
    return np.unique(e, axis=0)


def draw_bezier_arcs(ax, geo: dict, scale: float = 1000.0, control_points: bool = False):
    """Draw each hinge's two quadratic Bézier strips (upper=dark, lower=light green).

    ``control_points=True`` also shows the endpoints + the single interior control
    point per arc with thin control polygons.
    """
    for hd in geo['hinge_data']:
        top_keys = ('p0_top', 'bc_up', 'p1_top')
        bot_keys = ('p0_bot', 'bc_lo', 'p1_bot')
        # Dark green = the visually-upper strip (the cell's internal up/lo axis can
        # sit either way round in world XY).
        top_y = 0.5 * (hd['p0_top'][1] + hd['p1_top'][1])
        bot_y = 0.5 * (hd['p0_bot'][1] + hd['p1_bot'][1])
        order = ([(top_keys, GREEN_UP), (bot_keys, GREEN_LO)] if top_y >= bot_y
                 else [(bot_keys, GREEN_UP), (top_keys, GREEN_LO)])
        for keys, col in order:
            p0, c, p2 = (hd[k] for k in keys)
            arc = quad_bezier(p0, c, p2) * scale
            ax.plot(arc[:, 0], arc[:, 1], '-', color=col, lw=2.8, zorder=9)
            if control_points:
                poly = np.array([p0, c, p2]) * scale
                ax.plot(poly[:, 0], poly[:, 1], '--', color=CP_COL, lw=0.9,
                        alpha=0.55, zorder=10)
                ax.plot(poly[[0, 2], 0], poly[[0, 2], 1], 's', color=CP_COL,
                        ms=7, zorder=12, markeredgecolor='white', markeredgewidth=1)
                ax.plot(poly[1, 0], poly[1, 1], 'o', color=CP_COL,
                        ms=7, zorder=12, markeredgecolor='white', markeredgewidth=1)


def hide_axes(ax):
    """Equal aspect, no axis decorations — for geometry/state plots."""
    ax.set_aspect('equal')
    ax.axis('off')
