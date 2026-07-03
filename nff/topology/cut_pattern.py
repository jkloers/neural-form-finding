"""Precise flat-sheet kirigami cut geometry (Shapely, physical mm) — the laser-cuttable pattern.

Rebuilt from the schematic thin-line renderer to the ACTUAL per-hinge geometry, using the SAME
hinge parameters the surrogate/RVE assume (kerf ``w_c``, ligament ``w_lig``, fillet ``rho``), so
what we cut matches what we simulated.

Topology (from the closed-builder cut network ``(coords, T)``): each interior cut is a slit that
runs from one hinge to its opposite hinge across a void, retracted at each INTERIOR end by the
ligament ``w_lig`` and passing SOLID through any crossing in between. So building each cut as a
kerf-width slot retracted at its own two ends — with a ``rho`` fillet at each retracted tip — is
faithful: at a crossing, the terminating cut is retracted (leaving the ligament) while the crossing
cut passes through, exactly as in ``nff/rve/geometry.build_rve_domain``.

This module is pure geometry (Shapely only) — rendering lives in ``nff/utils/visualization`` and a
future DXF/SVG export consumes the same objects.
"""

import numpy as np
from shapely.geometry import LineString, Point, MultiPoint
from shapely.ops import unary_union


def build_cut_geometry(coords, T, cols, *, w_c, w_lig, rho, length_scale=1.0):
    """Precise cut geometry of a flat closed sheet, in mm.

    Args:
        coords: (P, 2) cut-vertex positions [sheet units]; endpoint ``s`` of cut ``(i, j)`` is
            index ``2*(i*cols + j) + s`` (closed-builder ordering).
        T: (rows, cols) topology matrix — sign = interior(+)/boundary(-), |.| = horizontal(1)/
            vertical(2).
        cols: T.shape[1].
        w_c: kerf width [mm]. w_lig: ligament (uncut bridge) width [mm]. rho: tip fillet radius [mm].
        length_scale: mm per sheet unit (the Gap-2 physical scale; picks the real size).

    Returns:
        dict: ``sheet`` (Polygon outline), ``cuts`` (removed-material geometry = kerf slots + fillets),
        ``hinges`` (list of (tip_xy[mm], axial_dir)), and the manufacturing params — all in mm.
    """
    rows = T.shape[0]
    C = np.asarray(coords, dtype=float) * length_scale

    def pid(i, j, s):
        return 2 * (i * cols + j) + s

    def in_grid(a, b):
        return 0 <= a < rows and 0 <= b < cols

    # Sheet outline (v1: convex hull of the boundary-cut vertices; ordered-boundary is a refinement).
    bpos = [C[pid(i, j, 0)] for i in range(rows) for j in range(cols) if T[i, j] < 0]
    sheet = MultiPoint(bpos).convex_hull

    slots, hinges = [], []
    for i in range(rows):
        for j in range(cols):
            t = int(T[i, j])
            if t == 0:
                continue
            if abs(t) == 1:                                     # horizontal cut
                (ai, aj, as_), (bi, bj, bs) = (i - 1, j, 0), (i + 1, j, 1)
            else:                                               # vertical cut
                (ai, aj, as_), (bi, bj, bs) = (i, j - 1, 1), (i, j + 1, 0)
            a_in, b_in = in_grid(ai, aj), in_grid(bi, bj)
            A = C[pid(ai, aj, as_)] if a_in else C[pid(i, j, 0)]
            B = C[pid(bi, bj, bs)] if b_in else C[pid(i, j, 0)]
            d = B - A
            L = float(np.linalg.norm(d))
            if L < 1e-9:
                continue
            u = d / L
            a_hinge = a_in and T[ai, aj] > 0                    # interior end -> retract + fillet
            b_hinge = b_in and T[bi, bj] > 0
            p0 = A + (w_lig if a_hinge else 0.0) * u
            p1 = B - (w_lig if b_hinge else 0.0) * u
            if float(np.linalg.norm(p1 - p0)) < 1e-9:
                continue
            slots.append(LineString([p0, p1]).buffer(w_c / 2.0, cap_style=2))   # flat-capped kerf slot
            if a_hinge:
                slots.append(Point(p0).buffer(rho))             # tip fillet
                hinges.append((p0, u))
            if b_hinge:
                slots.append(Point(p1).buffer(rho))
                hinges.append((p1, -u))

    cuts = unary_union(slots)
    return dict(sheet=sheet, cuts=cuts, hinges=hinges,
                w_c=w_c, w_lig=w_lig, rho=rho, length_scale=length_scale)


def cut_sheet(geom):
    """The laser-cut result: the sheet with the kerf slots removed (Polygon/MultiPolygon, mm)."""
    return geom["sheet"].difference(geom["cuts"])
