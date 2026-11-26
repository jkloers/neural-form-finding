"""Workshop layout schedule for a flat cut pattern: datum, marking points, cut and hole tables.

Turns the Shapely cut geometry (``nff.fabrication.cut_pattern.build_cut_geometry``) into the numbers a
person needs to lay the pattern out on a real sheet with a tape measure, a straightedge and a knife.

What gets tabulated is the LATTICE VERTICES — the crossings where cuts meet — not the retracted cut
ends. Two reasons. The cut ends are set back along the cut by the ligament ``w_lig``, so tabulating
them doubles the point count; and marking the crossings makes the layout self-checking, because every
line drawn must pass through two independently measured marks. The ligament is then one short setback
measured along a line that already exists.

Each point carries both a rectangular ordinate ``(x, y)`` from the datum corner and the distances to
two datum corners. The distances let a large sheet be laid out by striking two tape arcs, which needs
no square -- squaring off a 2.4 m edge is where hand layout usually goes wrong.

Pure geometry: numpy + shapely only. Rendering lives in ``nff.fabrication.cut_layout_render``.
"""

import csv
import os

import numpy as np


# Preferred metric twist-drill sizes for the hinge-tip relief hole.
_DRILL_SERIES_MM = (2.0, 2.5, 3.0, 3.2, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 8.0)

_EDGE_NAMES = ("left", "right", "bottom", "top")


def _nearest_drill(dia_mm):
    """Nearest preferred drill size [mm] to an exact fillet diameter."""
    return float(min(_DRILL_SERIES_MM, key=lambda d: abs(d - dia_mm)))


def _classify_ends(geom, tol_mm):
    """Per centreline, the (vertex, setback) at each end.

    A cut end that terminates in a ligament has a ``hinge_info`` entry carrying the un-retracted
    lattice ``vertex`` and its own ``w_lig``. An end without one runs out to the sheet edge, so the
    drawn line ends exactly at the cut end and the setback is zero.

    Returns:
        list of ``((v0, s0), (v1, s1))`` aligned to ``geom['centerlines']``; ``v`` are (2,) mm.
    """
    ends = [[None, None] for _ in geom["centerlines"]]
    for h in geom["hinge_info"]:
        c = int(h["cl"])
        p0, p1 = geom["centerlines"][c]
        tip = np.asarray(h["tip"], float)
        k = 0 if np.linalg.norm(tip - np.asarray(p0, float)) <= np.linalg.norm(tip - np.asarray(p1, float)) else 1
        ends[c][k] = (np.asarray(h["vertex"], float), float(h["w_lig"]))
    out = []
    for c, (p0, p1) in enumerate(geom["centerlines"]):
        e0 = ends[c][0] if ends[c][0] is not None else (np.asarray(p0, float), 0.0)
        e1 = ends[c][1] if ends[c][1] is not None else (np.asarray(p1, float), 0.0)
        out.append((e0, e1))
    return out


def _edge_of(p, w, h, tol_mm):
    """``(edge_name, distance_along_from_datum_corner)`` if ``p`` lies on the stock edge, else None.

    The along-edge distance is measured from the datum corner of that edge, so a point on the bottom
    or top edge reports its x and a point on the left or right edge reports its y. That is the single
    number needed to mark it with a tape run along the factory edge.
    """
    x, y = float(p[0]), float(p[1])
    if abs(x) <= tol_mm:
        return "left", y
    if abs(x - w) <= tol_mm:
        return "right", y
    if abs(y) <= tol_mm:
        return "bottom", x
    if abs(y - h) <= tol_mm:
        return "top", x
    return None


def build_layout_schedule(geom, *, tol_mm=0.05, line_tol_mm=0.5, hole_dia_mm=None,
                          face_centroids_mm=None, hold_faces=(), pull_faces=()):
    """Marking points, cut schedule and hole schedule for a cut pattern, in datum coordinates.

    The datum is the lower-left corner of the stock rectangle, x to the right along the short edge
    and y up along the long edge. Only that corner is supported on purpose: reflecting the frame to
    another corner would mirror a chiral pattern, and a mirrored RDPQK sheet deploys to the mirrored
    shape.

    Args:
        geom: output of ``build_cut_geometry`` (mm), already at the scale that will be cut.
        tol_mm: how close to a stock edge a point must be to count as lying on it.
        line_tol_mm: how close to a cut's construction line a point must be to count as lying on it.
        hole_dia_mm: single nominal relief-hole diameter [mm] for every hinge; ``None`` -> the
            preferred drill size nearest the median exact fillet diameter.
        face_centroids_mm: (F, 2) tile centroids [mm] in the design frame, for tile numbering and
            the activation marks.
        hold_faces / pull_faces: face indices to mark HOLD and PULL. These describe how the sheet
            will be gripped by hand, which need not be the boundary condition the design was
            optimised under.

    Returns:
        dict with ``stock`` (w, h), ``origin`` (the design-frame point mapped to the datum),
        ``points``, ``cuts``, ``perimeter_cuts``, ``holes``, ``checks``, ``tiles``, ``activation``
        and ``summary``.
    """
    minx, miny, maxx, maxy = geom["sheet"].bounds
    w, h = maxx - minx, maxy - miny
    org = np.array([minx, miny])

    def to_datum(p):
        return np.asarray(p, float) - org

    ends = _classify_ends(geom, tol_mm)

    # ── unique marking points, keyed by rounded position so shared crossings collapse ──
    pts, key_of = {}, {}

    def point_key(p):
        p = to_datum(p)
        k = (round(float(p[0]) / max(tol_mm, 1e-9)), round(float(p[1]) / max(tol_mm, 1e-9)))
        if k not in pts:
            pts[k] = dict(xy=p, n_cuts=0)
        return k

    # A cut lying ALONG a stock edge is the sheet perimeter, not a cut: its midpoint sits on the
    # rectangle boundary. Those are excluded from the schedule (the factory edge already is the cut)
    # and their vertices are only kept if some real cut also lands there.
    raw = []
    for c, ((v0, s0), (v1, s1)) in enumerate(ends):
        a, b = to_datum(v0), to_datum(v1)
        mid = 0.5 * (a + b)
        if _edge_of(mid, w, h, tol_mm) is not None:
            raw.append(dict(c=c, perimeter=True))
            continue
        raw.append(dict(c=c, perimeter=False, v0=v0, v1=v1, s0=s0, s1=s1))

    for r in raw:
        if r["perimeter"]:
            continue
        r["k0"], r["k1"] = point_key(r["v0"]), point_key(r["v1"])

    # ── point table: numbered up the sheet, then left to right ──
    order = sorted(pts, key=lambda k: (pts[k]["xy"][1], pts[k]["xy"][0]))
    corner_a, corner_b = np.array([0.0, 0.0]), np.array([w, 0.0])
    points, id_of = [], {}
    for n, k in enumerate(order, start=1):
        xy = pts[k]["xy"]
        pid = f"P{n:02d}"
        id_of[k] = pid
        edge = _edge_of(xy, w, h, tol_mm)
        points.append(dict(
            id=pid, x=float(xy[0]), y=float(xy[1]),
            d_from_A=float(np.linalg.norm(xy - corner_a)),
            d_from_B=float(np.linalg.norm(xy - corner_b)),
            edge=edge[0] if edge else "", along=float(edge[1]) if edge else float("nan"),
            n_lines=0))
    pt_index = {p["id"]: i for i, p in enumerate(points)}

    # ── cut table, ordered by the lower end so the work climbs the sheet ──
    cuts = []
    for r in raw:
        if r["perimeter"]:
            continue
        a, b = to_datum(r["v0"]), to_datum(r["v1"])
        if (a[1], a[0]) > (b[1], b[0]):                       # draw from the lower/left end
            a, b = b, a
            r["k0"], r["k1"], r["s0"], r["s1"] = r["k1"], r["k0"], r["s1"], r["s0"]
        d = b - a
        line_len = float(np.linalg.norm(d))
        u = d / line_len
        p_start, p_end = a + r["s0"] * u, b - r["s1"] * u
        cuts.append(dict(
            from_id=id_of[r["k0"]], to_id=id_of[r["k1"]],
            x_from=float(a[0]), y_from=float(a[1]), x_to=float(b[0]), y_to=float(b[1]),
            line_len=line_len, angle_deg=float(np.degrees(np.arctan2(d[1], d[0])) % 180.0),
            setback_from=float(r["s0"]), setback_to=float(r["s1"]),
            cut_len=float(np.linalg.norm(p_end - p_start)),
            x_cut_from=float(p_start[0]), y_cut_from=float(p_start[1]),
            x_cut_to=float(p_end[0]), y_cut_to=float(p_end[1]),
            cl=r["c"]))
    cuts.sort(key=lambda c: (round(c["y_from"], 3), round(c["x_from"], 3)))
    for n, c in enumerate(cuts, start=1):
        c["id"] = f"C{n:02d}"

    # A cut runs SOLID through any crossing between its two ends, so its construction line passes
    # over marking points other than its own. Naming them turns the scribed line into a check: it
    # has to hit every one of them, and a line that misses one indicts a mark, not the line.
    max_dev = _link_lines_through_points(cuts, points, pt_index, line_tol_mm)

    # ── hole table: one relief hole per ligament tip, drilled BEFORE the cut reaches it ──
    # Each hole gets its OWN nearest drill. The fillet radius tracks that hinge's ligament width, so
    # a design with a wide w_lig spread has a wide hole spread, and one drill sized to the median
    # would over-cut the narrowest ligament -- exactly the hinge with the least material to spare.
    # An explicit ``hole_dia_mm`` overrides that and forces a single size.
    exact = np.array([2.0 * float(hg["rho"]) for hg in geom["hinge_info"]])
    holes = []
    for hg in geom["hinge_info"]:
        tip = to_datum(hg["tip"])
        d_exact = 2.0 * float(hg["rho"])
        holes.append(dict(x=float(tip[0]), y=float(tip[1]), dia_exact=d_exact,
                          dia_nominal=(float(hole_dia_mm) if hole_dia_mm is not None
                                       else _nearest_drill(d_exact)),
                          w_lig=float(hg["w_lig"])))
    drills = sorted({h["dia_nominal"] for h in holes})
    holes.sort(key=lambda hh: (round(hh["y"], 3), round(hh["x"], 3)))
    for n, hh in enumerate(holes, start=1):
        hh["id"] = f"H{n:02d}"

    _resolve_origins(cuts, points)

    # ── tiles and the hand-grip activation marks ──
    tiles, activation = [], []
    if face_centroids_mm is not None:
        fc = np.asarray(face_centroids_mm, float).reshape(-1, 2)
        for f, p in enumerate(fc):
            q = to_datum(p)
            tiles.append(dict(face=int(f), x=float(q[0]), y=float(q[1])))
        for role, faces in (("HOLD", hold_faces), ("PULL", pull_faces)):
            for f in faces:
                q = to_datum(fc[int(f)])
                activation.append(dict(role=role, face=int(f), x=float(q[0]), y=float(q[1])))

    checks = _build_checks(points, w, h)
    wl = np.array([hg["w_lig"] for hg in geom["hinge_info"]])
    summary = dict(
        stock_w=float(w), stock_h=float(h), n_points=len(points), n_cuts=len(cuts),
        n_perimeter=sum(1 for r in raw if r["perimeter"]), n_holes=len(holes),
        w_lig_min=float(wl.min()), w_lig_max=float(wl.max()), w_lig_med=float(np.median(wl)),
        hole_drills=drills,
        hole_drill_label=(f"Ø{drills[0]:.1f} mm" if len(drills) == 1 else
                          f"Ø{drills[0]:.1f}-{drills[-1]:.1f} mm, {len(drills)} sizes"),
        hole_dia_nominal=max(drills, key=lambda d: sum(h["dia_nominal"] == d for h in holes)),
        hole_dia_min=float(exact.min()), hole_dia_max=float(exact.max()),
        cut_len_total=float(sum(c["cut_len"] for c in cuts)),
        max_collinearity_dev=float(max_dev),
        n_measured=sum(1 for p in points if p["origin"] == "MEASURE"))
    return dict(stock=(float(w), float(h)), origin=org, points=points, cuts=cuts,
                holes=holes, checks=checks, tiles=tiles, activation=activation,
                summary=summary, tol_mm=float(tol_mm))


def _link_lines_through_points(cuts, points, pt_index, line_tol_mm):
    """Fill each cut's ``through`` list and each point's ``n_lines`` count.

    Returns the largest off-line deviation found, so a pattern whose crossings are not actually
    collinear cannot pass silently.
    """
    xy = np.array([[p["x"], p["y"]] for p in points])
    max_dev = 0.0
    for c in cuts:
        a = np.array([c["x_from"], c["y_from"]])
        b = np.array([c["x_to"], c["y_to"]])
        d = b - a
        L = float(np.linalg.norm(d))
        u = d / L
        t = (xy - a) @ u
        perp = np.abs((xy - a) @ np.array([-u[1], u[0]]))
        on = (perp <= line_tol_mm) & (t >= -line_tol_mm) & (t <= L + line_tol_mm)
        order = np.flatnonzero(on)[np.argsort(t[on])]
        ids = [points[i]["id"] for i in order]
        if on.any():
            max_dev = max(max_dev, float(perp[on].max()))
        for i in order:
            points[i]["n_lines"] += 1
        c["through"] = [i for i in ids if i not in (c["from_id"], c["to_id"])]
        # Stations: distance along the scribed line from its first point. Once two points fix the
        # line, every other point on it is a single measurement ALONG it -- no second ordinate, no
        # squaring off, and the marks that follow inherit the line's accuracy instead of adding
        # their own. This is why the line, not the point, is the unit of work.
        c["stations"] = [(points[i]["id"], float(t[i])) for i in order]
        c["cut_from_station"] = float(c["setback_from"])
        c["cut_to_station"] = float(L - c["setback_to"])
    return max_dev


def _resolve_origins(cuts, points):
    """Decide, per point, whether it must be MEASURED or can be stepped off a line already scribed.

    A line needs two known points before it can be scribed; every other point on it then follows
    from a single distance along it. Walking the lines in order and only measuring what is not yet
    known gives a small set of measured points and leaves the rest inheriting a scribed line's
    accuracy. Each measured point is an independent chance to be wrong, so fewer is better.
    """
    by_id = {p["id"]: p for p in points}
    for p in points:
        p["origin"] = ""
    known, order = set(), []
    for c in cuts:                                   # in table order: up the sheet
        for pid in (c["from_id"], c["to_id"]):       # the two anchors must exist to scribe
            if pid not in known:
                by_id[pid]["origin"] = "MEASURE"
                known.add(pid)
                order.append(pid)
        for pid, s in c["stations"]:
            if pid not in known:
                by_id[pid]["origin"] = f"{c['id']}@{s:.1f}"
                known.add(pid)
                order.append(pid)
        c["anchors"] = [c["from_id"], c["to_id"]]
    return sum(1 for p in points if p["origin"] == "MEASURE")


def _build_checks(points, w, h):
    """Long diagonals to measure BEFORE cutting, so a mis-marked point shows up while it is free.

    The stock diagonal comes first: if the sheet itself is out of square every ordinate is wrong.
    """
    checks = [dict(name="stock diagonal", frm="corner (0,0)", to=f"corner ({w:.1f},{h:.1f})",
                   dist=float(np.hypot(w, h)))]
    interior = [p for p in points if not p["edge"]]
    if len(interior) < 4:
        return checks
    # The interior point nearest each stock corner: guaranteed to be spread, so the diagonals
    # between them are long and a small marking error shows up as a large closure error.
    xs = np.array([[p["x"], p["y"]] for p in interior])
    picked = []
    for corner in ([0, 0], [w, 0], [0, h], [w, h]):
        i = int(np.argmin(np.linalg.norm(xs - np.array(corner, float), axis=1)))
        if interior[i] not in picked:
            picked.append(interior[i])
    for i in range(len(picked)):
        for j in range(i + 1, len(picked)):
            a, b = picked[i], picked[j]
            checks.append(dict(name="layout check", frm=a["id"], to=b["id"],
                               dist=float(np.hypot(a["x"] - b["x"], a["y"] - b["y"]))))
    return checks


def write_layout_csv(schedule, out_dir):
    """Write ``points.csv`` / ``cuts.csv`` / ``holes.csv`` (mm, 2 dp) into ``out_dir``."""
    os.makedirs(out_dir, exist_ok=True)
    specs = [
        ("points.csv", schedule["points"],
         ["id", "origin", "x", "y", "d_from_A", "d_from_B", "edge", "along", "n_lines"]),
        ("cuts.csv", schedule["cuts"],
         ["id", "from_id", "to_id", "through", "stations_str", "x_from", "y_from", "x_to", "y_to",
          "line_len", "angle_deg", "setback_from", "setback_to", "cut_len",
          "cut_from_station", "cut_to_station",
          "x_cut_from", "y_cut_from", "x_cut_to", "y_cut_to"]),
        ("holes.csv", schedule["holes"], ["id", "x", "y", "dia_nominal", "dia_exact", "w_lig"]),
    ]
    for c in schedule["cuts"]:                       # flatten the station list for the CSV
        c["stations_str"] = " ".join(f"{pid}@{s:.1f}" for pid, s in c["stations"])
    written = []
    for name, rows, cols in specs:
        path = os.path.join(out_dir, name)
        with open(path, "w", newline="") as f:
            wtr = csv.writer(f)
            wtr.writerow(cols)
            for r in rows:
                wtr.writerow([f"{r[c]:.2f}" if isinstance(r[c], float)
                              else " ".join(r[c]) if isinstance(r[c], list) else r[c]
                              for c in cols])
        written.append(path)
    return written


def verify_schedule(schedule, geom, *, tol_mm=1e-6):
    """Check the tables reproduce the geometry they were derived from.

    Every non-perimeter centreline must appear once in the cut table with both retracted endpoints
    matching the built geometry, and every on-edge point's ordinate must equal its along-edge
    distance. Raises ``AssertionError`` on any mismatch; returns the measured worst-case error.
    """
    org = schedule["origin"]
    worst = 0.0
    by_cl = {c["cl"]: c for c in schedule["cuts"]}
    assert len(by_cl) == len(schedule["cuts"]), "duplicate centreline in the cut table"
    for cl, c in by_cl.items():
        p0, p1 = [np.asarray(p, float) - org for p in geom["centerlines"][cl]]
        got = [np.array([c["x_cut_from"], c["y_cut_from"]]), np.array([c["x_cut_to"], c["y_cut_to"]])]
        err = min(max(np.linalg.norm(got[0] - a), np.linalg.norm(got[1] - b))
                  for a, b in ((p0, p1), (p1, p0)))
        worst = max(worst, float(err))
    assert worst <= 1e-2, f"cut table does not reproduce the geometry (worst {worst:.2e} mm)"

    w, h = schedule["stock"]
    for p in schedule["points"]:
        if not p["edge"]:
            continue
        expect = p["y"] if p["edge"] in ("left", "right") else p["x"]
        assert abs(expect - p["along"]) <= tol_mm, f"{p['id']}: along-edge distance disagrees"
        on = {"left": p["x"], "right": w - p["x"], "bottom": p["y"], "top": h - p["y"]}[p["edge"]]
        assert abs(on) <= schedule["tol_mm"], f"{p['id']}: flagged on {p['edge']} but is {on:.3f} mm off"
    return worst
