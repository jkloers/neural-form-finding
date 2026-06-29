"""
nff/sofa/hinge_geometry.py — geometry resolver for kirigami corner hinges.

Turns the design parameters (gap + four edge reaches + two Bézier control points)
and the face/hinge topology into the concrete 2D outline of each hinge: the two
faces pushed apart by ``gap``, four anchor points on the face edges, and the two
quadratic Bézier arcs that bound the hinge strip.

This is pure geometry — no meshing, no SOFA, numpy only — so it is shared across
both environments: the SOFA oracle (as the first step of build_mesh_gmsh) and the
JAX-side hinge optimizer (to compute the analytic hinge area for its material
gradient) and the visualization scripts. Self-contained so it imports cleanly in
both the package (nff.sofa.*) and the flat (Docker) contexts.
"""

from __future__ import annotations
import numpy as np

# Hinges whose two faces share a corner within this gap are "corner hinges".
CLOSED_GAP_TOL = 1e-6


# ── face-intersection guard ────────────────────────────────────────────────────

def _strictly_inside(pt, poly, tol):
    n = len(poly)
    cs = []
    for i in range(n):
        v0, v1 = poly[i], poly[(i + 1) % n]
        cs.append((v1[0]-v0[0])*(pt[1]-v0[1]) - (v1[1]-v0[1])*(pt[0]-v0[0]))
    return all(c > tol for c in cs) or all(c < -tol for c in cs)


def _seg_cross(a, b, c, d, tol):
    def x2(o, p, q):
        return (p[0]-o[0])*(q[1]-o[1]) - (p[1]-o[1])*(q[0]-o[0])
    d1, d2 = x2(c, d, a), x2(c, d, b)
    d3, d4 = x2(a, b, c), x2(a, b, d)
    return (((d1 > tol and d2 < -tol) or (d1 < -tol and d2 > tol)) and
            ((d3 > tol and d4 < -tol) or (d3 < -tol and d4 > tol)))


def _quads_overlap(p1, p2, tol=1e-8):
    for v in p1:
        if _strictly_inside(v, p2, tol): return True
    for v in p2:
        if _strictly_inside(v, p1, tol): return True
    n1, n2 = len(p1), len(p2)
    for i in range(n1):
        if _strictly_inside((p1[i] + p1[(i+1) % n1]) * 0.5, p2, tol): return True
    for j in range(n2):
        if _strictly_inside((p2[j] + p2[(j+1) % n2]) * 0.5, p1, tol): return True
    for i in range(n1):
        for j in range(n2):
            if _seg_cross(p1[i], p1[(i+1) % n1], p2[j], p2[(j+1) % n2], tol):
                return True
    return False


def check_face_intersections(face_verts: np.ndarray, tol: float = 1e-8) -> None:
    """Raise ValueError if any two faces have area overlap (not just corner-touching)."""
    n = len(face_verts)
    bad = [(i, j) for i in range(n) for j in range(i + 1, n)
           if _quads_overlap(face_verts[i], face_verts[j], tol)]
    if bad:
        raise ValueError(
            f"Face area intersections: {', '.join(f'({i},{j})' for i, j in bad)}. "
            "Fix Stage-1 output before meshing.")


# ── geometry helper ───────────────────────────────────────────────────────────

def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-14 else np.zeros_like(v)


# ── hinge geometry (shared by builder, optimizer, and visualizer) ─────────────

def compute_hinge_geometry(
    cs,
    gap: float = 0.003,
    bezier_params: dict = None,
) -> dict:
    """Resolve face polygons and per-hinge Bézier strip geometry.

    The two faces of each corner hinge are rigidly translated apart by ``gap``
    along the centroid-to-centroid axis ``ê_arm`` (face fi by ``-gap/2·ê_arm``,
    face fk by ``+gap/2·ê_arm``), opening a real void.  Each hinge then yields
    FOUR distinct anchor points — two on each (translated) face, on either side
    of its corner: ``p0_top``/``p0_bot`` slide along face fi's upper/lower edges
    and ``p1_top``/``p1_bot`` along face fk's.  Because they are measured from
    each face's own corner, the anchors lie exactly on the face boundaries.  The
    upper arc joins the two top anchors, the lower arc the two bottom anchors;
    they do NOT share endpoints.  Single source of truth consumed by both
    ``build_mesh_gmsh`` and the visualization script.

    Args:
        cs: CentroidalState (or SimpleNamespace) describing faces and hinges.
        gap: rigid face separation along ê_arm [m]. Must be > 0 for corner hinges.
        bezier_params: optional overrides (see ``build_mesh_gmsh``).

    Returns:
        dict with keys ``face_verts`` (n_faces, 4, 2, after translation),
        ``hinge_data`` (per-hinge anchor/CP dicts), ``clamped_faces``,
        ``loaded_faces``. Mesh element sizing is the caller's concern
        (``build_mesh_gmsh``), derived from this geometry.
    """
    bp = bezier_params or {}

    fc  = np.asarray(cs.face_centroids, dtype=np.float64)
    cnv = np.asarray(cs.centroid_node_vectors, dtype=np.float64)
    hnp = np.asarray(cs.hinge_node_pairs, dtype=np.int32)

    raw_verts = fc[:, None, :] + cnv[:, :4, :]   # (n_faces, 4, 2) rigid panels
    n_faces = raw_verts.shape[0]

    if gap is None or gap <= 0:
        raise ValueError("compute_hinge_geometry requires gap > 0 (corner hinges).")

    # ── Per-hinge axis + corner-coincidence (corner-hinge) detection ──────────
    hinges = []
    for h in range(hnp.shape[0]):
        fi, lj = int(hnp[h, 0, 0]), int(hnp[h, 0, 1])
        fk, ll = int(hnp[h, 1, 0]), int(hnp[h, 1, 1])
        corner = raw_verts[fi, lj]
        if float(np.linalg.norm(raw_verts[fk, ll] - corner)) >= CLOSED_GAP_TOL:
            raise NotImplementedError(
                "build_mesh_gmsh supports corner hinges only "
                f"(hinge {h}: faces {fi}↔{fk} do not share a corner).")
        earm = _unit(fc[fk] - fc[fi])
        hinges.append({'fi': fi, 'lj': lj, 'fk': fk, 'll': ll,
                       'corner': corner.copy(), 'earm': earm})

    # ── Rigid face translation: push each face away from its hinge partner ─────
    disp = np.zeros((n_faces, 2), dtype=np.float64)
    for hg in hinges:
        disp[hg['fi']] -= 0.5 * gap * hg['earm']
        disp[hg['fk']] += 0.5 * gap * hg['earm']
    face_verts = raw_verts + disp[:, None, :]
    check_face_intersections(face_verts)

    clamped_faces = sorted({int(r[0]) for r in np.asarray(cs.constrained_face_DOF_pairs)})
    loaded_faces  = sorted({int(r[0]) for r in np.asarray(cs.loaded_face_DOF_pairs)})

    hinge_data = []
    for hg in hinges:
        fi, lj, fk, ll = hg['fi'], hg['lj'], hg['fk'], hg['ll']
        earm = hg['earm']
        perp = np.array([-earm[1], earm[0]])      # unit perpendicular to hinge axis
        # Each face's own (translated) corner — anchors slide from here.
        corner_i = face_verts[fi, lj]
        corner_k = face_verts[fk, ll]

        adj_prev_fi = face_verts[fi, (lj - 1) % 4]
        adj_next_fi = face_verts[fi, (lj + 1) % 4]
        if np.dot(adj_prev_fi - corner_i, perp) > 0:
            adj_up_fi, adj_dn_fi = adj_prev_fi, adj_next_fi
            up_is_prev_fi = True
        else:
            adj_up_fi, adj_dn_fi = adj_next_fi, adj_prev_fi
            up_is_prev_fi = False

        adj_prev_fk = face_verts[fk, (ll - 1) % 4]
        adj_next_fk = face_verts[fk, (ll + 1) % 4]
        if np.dot(adj_prev_fk - corner_k, perp) > 0:
            adj_up_fk, adj_dn_fk = adj_prev_fk, adj_next_fk
            up_is_prev_fk = True
        else:
            adj_up_fk, adj_dn_fk = adj_next_fk, adj_prev_fk
            up_is_prev_fk = False

        # Reach distances (default = gap) — endpoints slide on the face edges.
        s0_top = float(bp.get('s0_top', gap))
        s0_bot = float(bp.get('s0_bot', gap))
        s1_top = float(bp.get('s1_top', gap))
        s1_bot = float(bp.get('s1_bot', gap))

        p0_top = corner_i + s0_top * _unit(adj_up_fi - corner_i)
        p0_bot = corner_i + s0_bot * _unit(adj_dn_fi - corner_i)
        p1_top = corner_k + s1_top * _unit(adj_up_fk - corner_k)
        p1_bot = corner_k + s1_bot * _unit(adj_dn_fk - corner_k)

        # Single interior CP per arc (quadratic Bézier) — default: chord midpoint
        # bowed outward by 2·gap (arc apex ≈ gap from the chord).
        def _default_cp(pa, pb, sign):
            return 0.5 * (pa + pb) + sign * perp * 2.0 * gap

        bc_up_def = _default_cp(p0_top, p1_top, +1.0)
        bc_lo_def = _default_cp(p0_bot, p1_bot, -1.0)

        bc_up = np.asarray(bp['bc_up_xy'], float) if 'bc_up_xy' in bp else bc_up_def
        bc_lo = np.asarray(bp['bc_lo_xy'], float) if 'bc_lo_xy' in bp else bc_lo_def

        hinge_data.append({
            'fi': fi, 'lj': lj,
            'fk': fk, 'll': ll,
            'corner': 0.5 * (corner_i + corner_k), 'earm': earm, 'perp': perp, 'gap': gap,
            'p0_top': p0_top, 'p0_bot': p0_bot,
            'p1_top': p1_top, 'p1_bot': p1_bot,
            'bc_up': bc_up, 'bc_lo': bc_lo,
            'up_is_prev_fi': up_is_prev_fi,
            'up_is_prev_fk': up_is_prev_fk,
        })

    return {
        'face_verts': face_verts,
        'hinge_data': hinge_data,
        'clamped_faces': clamped_faces,
        'loaded_faces': loaded_faces,
    }
