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


def build_cut_geometry(coords, T, cols, *, w_c, w_lig, rho, length_scale=1.0,
                       hinge_lookup=None, fillet_quad_segs=32):
    """Precise cut geometry of a flat closed sheet, in mm.

    Args:
        coords: (P, 2) cut-vertex positions [sheet units]; endpoint ``s`` of cut ``(i, j)`` is
            index ``2*(i*cols + j) + s`` (closed-builder ordering).
        T: (rows, cols) topology matrix — sign = interior(+)/boundary(-), |.| = horizontal(1)/
            vertical(2).
        cols: T.shape[1].
        w_c: kerf width [mm]. w_lig: ligament (uncut bridge) width [mm]. rho: tip fillet radius [mm].
            ``w_lig``/``rho`` are the SCALAR fallback used when a hinge has no per-hinge entry.
        length_scale: mm per sheet unit (the Gap-2 physical scale; picks the real size).
        hinge_lookup: optional (H, 4) array ``[x_mm, y_mm, w_lig_mm, rho_mm]`` — per-hinge
            manufacturing params keyed by pivot POSITION (mm). Each terminating cut end is matched to
            its nearest pivot so the printed hinge uses the SAME ``w_lig``/``rho`` the surrogate saw.
            ``None`` -> the scalar ``w_lig``/``rho`` for every hinge (legacy behaviour).
        fillet_quad_segs: segments/quadrant for the tip fillet disc (>=32 to match the RVE's smooth
            ``Point(tip).buffer(rho, quad_segs=32)``; the Shapely default of 8 prints an octagon).

    Returns:
        dict: ``sheet`` (Polygon outline), ``cuts`` (removed-material geometry = kerf slots + fillets),
        ``hinges`` (list of ``(tip_xy[mm], axial_dir)`` tuples — legacy), ``hinge_info`` (list of
        per-hinge dicts: ``tip, dir, w_lig, rho, vertex, cl``), ``centerlines`` (list of
        ``(p0, p1)`` kerf centrelines — used by the round-trip check + the print), and the params.
    """
    rows = T.shape[0]
    C = np.asarray(coords, dtype=float) * length_scale
    HL = np.asarray(hinge_lookup, dtype=float) if hinge_lookup is not None else None
    HLxy = HL[:, :2] if HL is not None else None

    def pid(i, j, s):
        return 2 * (i * cols + j) + s

    def in_grid(a, b):
        return 0 <= a < rows and 0 <= b < cols

    def params_at(xy):
        """Per-hinge (w_lig, rho) by nearest pivot [mm]; scalar fallback if no lookup."""
        if HLxy is None:
            return float(w_lig), float(rho)
        k = int(np.argmin(np.sum((HLxy - xy) ** 2, axis=1)))
        return float(HL[k, 2]), float(HL[k, 3])

    # Sheet outline (v1: convex hull of the boundary-cut vertices; ordered-boundary is a refinement).
    bpos = [C[pid(i, j, 0)] for i in range(rows) for j in range(cols) if T[i, j] < 0]
    sheet = MultiPoint(bpos).convex_hull

    slots, hinges, hinge_info, centerlines = [], [], [], []
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
            # A cut end is a HINGE (retract by w_lig + fillet) at a crossing that leaves a ligament:
            #   * neighbour is an interior cut (T>0), OR
            #   * neighbour is a BOUNDARY cut (T<0) and THIS cut is interior (t>0).
            # The second case is the boundary hinges: without it the interior cut runs straight
            # THROUGH the border, severing a ligament that must stay (the sheet falls apart when cut).
            # This exactly matches the physics hinge set (_build_hinges): 60 @ 6x6, 180 @ 10x10.
            a_hinge = a_in and (T[ai, aj] > 0 or (T[ai, aj] < 0 and t > 0))
            b_hinge = b_in and (T[bi, bj] > 0 or (T[bi, bj] < 0 and t > 0))
            wl_a, rho_a = params_at(A) if a_hinge else (0.0, 0.0)
            wl_b, rho_b = params_at(B) if b_hinge else (0.0, 0.0)
            p0 = A + (wl_a if a_hinge else 0.0) * u
            p1 = B - (wl_b if b_hinge else 0.0) * u
            if float(np.linalg.norm(p1 - p0)) < 1e-9:
                continue
            slots.append(LineString([p0, p1]).buffer(w_c / 2.0, cap_style=2))   # flat-capped kerf slot
            cl = len(centerlines)
            centerlines.append((p0.copy(), p1.copy()))
            if a_hinge:
                slots.append(Point(p0).buffer(rho_a, quad_segs=fillet_quad_segs))   # tip fillet disc
                hinges.append((p0, u))
                hinge_info.append(dict(tip=p0.copy(), dir=u.copy(), w_lig=wl_a, rho=rho_a,
                                       vertex=A.copy(), cl=cl))
            if b_hinge:
                slots.append(Point(p1).buffer(rho_b, quad_segs=fillet_quad_segs))
                hinges.append((p1, -u))
                hinge_info.append(dict(tip=p1.copy(), dir=(-u).copy(), w_lig=wl_b, rho=rho_b,
                                       vertex=B.copy(), cl=cl))

    cuts = unary_union(slots)
    return dict(sheet=sheet, cuts=cuts, hinges=hinges, hinge_info=hinge_info,
                centerlines=centerlines, w_c=float(w_c), w_lig=float(w_lig), rho=float(rho),
                length_scale=float(length_scale))


def measure_cut_geometry(geom, alpha_expected=None):
    """Round-trip check: recover per-hinge (w_lig, rho, alpha) from the BUILT geometry.

    Guarantees "what prints == what was simulated" by measuring back off the geometry rather than
    trusting the inputs:

    - ``w_lig`` = distance from the fillet tip to its cut vertex (the retracted ligament length).
    - ``rho``   = the fillet disc radius carried on the hinge.
    - ``alpha`` = angle [deg] between this (terminating) cut and the CROSSING cut through its vertex
      — the through-cut is the OTHER centreline passing nearest the vertex. This is the angle the
      surrogate's ``alpha`` denotes (angle between the two cuts at the hinge).

    Args:
        geom: output of :func:`build_cut_geometry`.
        alpha_expected: optional (H,) array [deg] of the surrogate's per-hinge alpha, aligned to
            ``geom['hinge_info']`` order, to report the drawn-vs-simulated angle error.

    Returns:
        dict with per-hinge arrays ``w_lig``, ``rho``, ``alpha_deg`` and, if ``alpha_expected`` given,
        ``alpha_err_deg`` (abs) plus scalar ``max_alpha_err_deg`` / ``max_w_lig_err``.
    """
    info = geom["hinge_info"]
    cls = [LineString([q0, q1]) for (q0, q1) in geom["centerlines"]]
    seg = [(np.asarray(q0), np.asarray(q1)) for (q0, q1) in geom["centerlines"]]
    # tolerance for "this cut's LINE passes through the vertex": a fraction of the ligament scale
    wl_typ = np.median([h["w_lig"] for h in info]) if info else 1.0
    tol = 0.5 * float(wl_typ)
    w_meas, rho_meas, a_meas = [], [], []
    for h in info:
        tip, v, own = np.asarray(h["tip"]), np.asarray(h["vertex"]), int(h["cl"])
        w_meas.append(float(np.linalg.norm(tip - v)))
        rho_meas.append(float(h["rho"]))
        u = np.asarray(h["dir"]); u = u / (np.linalg.norm(u) + 1e-12)
        vp = Point(v[0], v[1])
        # the CROSSING (secondary) cut passes SOLID through the vertex (dist~0) while the terminating
        # cut is retracted away; among cuts whose segment touches the vertex, take the MOST
        # perpendicular one — that is the cut whose angle to this one defines the hinge alpha.
        best_k, best_ang = None, -1.0
        for k in range(len(cls)):
            if k == own or cls[k].distance(vp) > tol:
                continue
            q0, q1 = seg[k]
            vdir = q1 - q0; vdir = vdir / (np.linalg.norm(vdir) + 1e-12)
            ang = np.degrees(np.arccos(np.clip(abs(float(np.dot(u, vdir))), 0.0, 1.0)))
            if ang > best_ang:
                best_ang, best_k = ang, k
        a_meas.append(best_ang if best_k is not None else np.nan)
    out = dict(w_lig=np.array(w_meas), rho=np.array(rho_meas), alpha_deg=np.array(a_meas))
    if alpha_expected is not None:
        exp = np.asarray(alpha_expected, float)
        err = np.abs(out["alpha_deg"] - exp)
        out["alpha_err_deg"] = err
        out["max_alpha_err_deg"] = float(np.nanmax(err)) if err.size else 0.0
    if info:
        wl_assigned = np.array([h["w_lig"] for h in info])
        out["max_w_lig_err"] = float(np.max(np.abs(out["w_lig"] - wl_assigned)))
    return out


def cut_sheet(geom):
    """The laser-cut result: the sheet with the kerf slots removed (Polygon/MultiPolygon, mm)."""
    return geom["sheet"].difference(geom["cuts"])
