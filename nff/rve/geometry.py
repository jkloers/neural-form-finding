"""Parametric single-hinge RVE geometry (sheet plane).

The RVE is the material around one hinge that we mesh and load: the two tiles it
joins, connected by the ligament left where the main cut is retracted by ``w_lig``.
It is bounded by

    - the SECONDARY cut (the top edge, y = 0)                  -> free (traction-free)
    - the MAIN cut (a kerf-wide slit with a rho-radius tip)    -> free
    - the far edges of the two tiles (Saint-Venant window)     -> rigid_A / rigid_B

The two rigid boundaries are the handles: we impose the relative in-plane motion
``(a, s, theta)`` between tile A (left of the main cut) and tile B (right of it),
and the ligament deforms. Lengths are in millimetres.

Frame: secondary cut along ``y = 0`` (material below), main cut pointing down from
its tip at ``(0, -w_lig)`` along ``u_m`` (tilted from vertical by ``alpha - 90``).
"""

from dataclasses import dataclass

import numpy as np
from shapely.geometry import Polygon, LineString, Point, box
from shapely.ops import unary_union


@dataclass(frozen=True)
class RVEParams:
    """Single-hinge RVE parameters [mm] (+ angle in degrees)."""
    w_lig: float = 8.0          # ligament gap (main-cut tip -> secondary cut)
    w_c: float = 0.2            # cut width (kerf) — standard laser value
    alpha_deg: float = 90.0     # angle between the two cuts
    rho: float = 0.5            # fillet radius at the main-cut tip
    thickness: float = 1.0      # sheet thickness (for the 3D extrusion)
    r_win: float = 20.0         # Saint-Venant window half-size

    def main_cut_dir(self) -> np.ndarray:
        a = np.radians(self.alpha_deg)
        u = np.array([np.cos(-a), np.sin(-a)])
        return u / np.linalg.norm(u)


def build_rve_domain(p: RVEParams) -> Polygon:
    """2D RVE domain: the window minus the main-cut slit (filleted, retracted w_lig)."""
    W = H = p.r_win
    um = p.main_cut_dir()
    tip = np.array([0.0, -p.w_lig])
    slit = LineString([tip, tip + (H + 2.0 * p.w_lig) * um]).buffer(p.w_c / 2.0, cap_style=2)
    slit = unary_union([slit, Point(tip[0], tip[1]).buffer(p.rho, quad_segs=32)])
    dom = box(-W, -H, W, 0.0).difference(slit)
    if dom.geom_type != "Polygon":
        dom = max(dom.geoms, key=lambda g: g.area)      # keep the main connected piece
    return dom


def boundary_tag(mx: float, my: float, p: RVEParams, tol: float = 1e-6) -> str:
    """Tag for a boundary point ``(mx, my)`` on the RVE exterior.

    rigid_A : left tile far boundary (x = -W, and the bottom for x < 0)
    rigid_B : right tile far boundary (x = +W, and the bottom for x > 0)
    free    : the secondary cut (top, y = 0) and the main-cut slit (incl. fillet)
    """
    W = H = p.r_win
    if abs(mx + W) < tol or (abs(my + H) < tol and mx < 0):
        return "rigid_A"
    if abs(mx - W) < tol or (abs(my + H) < tol and mx > 0):
        return "rigid_B"
    return "free"


def classify_boundary(dom: Polygon, p: RVEParams, tol: float = 1e-6) -> dict:
    """Tag each exterior edge as ``rigid_A``, ``rigid_B`` or ``free`` (see ``boundary_tag``).

    Returns a dict tag -> list of ``((x0, y0), (x1, y1))`` segments.
    """
    tags = {"rigid_A": [], "rigid_B": [], "free": []}
    coords = list(dom.exterior.coords)
    for (x0, y0), (x1, y1) in zip(coords[:-1], coords[1:]):
        tags[boundary_tag(0.5 * (x0 + x1), 0.5 * (y0 + y1), p, tol)].append(((x0, y0), (x1, y1)))
    return tags


def ligament_present(dom: Polygon, p: RVEParams) -> bool:
    """True iff the two tiles are still bridged (a point just below the secondary
    cut on the main-cut axis is inside the domain — the retracted cut left material)."""
    um = p.main_cut_dir()
    probe = np.array([0.0, 0.0]) + 0.5 * p.w_lig * um    # mid-ligament, on the main-cut axis
    return dom.contains(Point(probe[0], probe[1]))
