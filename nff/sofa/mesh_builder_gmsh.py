"""
nff/sofa/mesh_builder_gmsh.py — gmsh-based mesh builder for kirigami unit cells.

Each hinge is a curved strip connecting two face edges.  The strip is bounded by:
  - Two cubic Bézier arcs: upper (p0_top → p1_top) and lower (p0_bot → p1_bot)
  - Two straight closing edges along the face boundaries (shared with face surfaces)

The two faces of a corner hinge are rigidly translated apart by ``gap`` along the
centroid-to-centroid axis, opening a real void that the Bézier hinge spans.  The
four arc endpoints slide along each (translated) face's own edges via the reach
parameters, so they always lie exactly on the face boundaries.

Parametrisation per hinge
--------------------------
  gap             — rigid separation of the two faces along ê_arm [m]
  s0_top, s0_bot  — reach from face-i corner along its upper/lower edges [m]
  s1_top, s1_bot  — reach from face-k corner along its upper/lower edges [m]
  bc_up_xy — single interior CP for the upper (quadratic) arc, absolute XY [m]
  bc_lo_xy — single interior CP for the lower (quadratic) arc, absolute XY [m]

Workflow
--------
  1. Resolve face polygons + hinge anchors via compute_hinge_geometry
     (faces translated apart by gap; endpoints anchored on the face edges).
  2. Build conforming 2D domain in gmsh: faces + hinge share boundary edges.
  3. Extrude by sheet_thickness → prism elements → split each prism into 3 tetrahedra.
  4. Classify nodes into clamped / loaded via 2D point-in-polygon.

Returns
-------
  nodes    : (N, 3) float64
  tets     : (M, 4) int32   (0-indexed)
  bc_masks : {'clamped': bool (N,), 'loaded': bool (N,)}
"""

from __future__ import annotations
import numpy as np
from matplotlib.path import Path as _MplPath

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


# ── prism → tet splitting ─────────────────────────────────────────────────────

def _prisms_to_tets(prisms: np.ndarray) -> np.ndarray:
    """Split (N, 6) prism connectivity into (3N, 4) tetrahedra.

    Prism node layout (gmsh default after straight extrusion of a triangle):
      bottom: v0, v1, v2   top: v3, v4, v5  (v3 above v0, etc.)
    """
    v = prisms
    return np.vstack([
        np.column_stack([v[:, 0], v[:, 1], v[:, 2], v[:, 3]]),
        np.column_stack([v[:, 1], v[:, 3], v[:, 4], v[:, 2]]),
        np.column_stack([v[:, 2], v[:, 3], v[:, 4], v[:, 5]]),
    ]).astype(np.int32)


# ── BC classification ─────────────────────────────────────────────────────────

def _nodes_in_polygon(nodes_3d: np.ndarray, poly_xy: np.ndarray) -> np.ndarray:
    return _MplPath(poly_xy).contains_points(nodes_3d[:, :2], radius=1e-9)


# ── geometry helpers ──────────────────────────────────────────────────────────

def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-14 else np.zeros_like(v)


def _bezier_sample(p0: np.ndarray, c: np.ndarray, p2: np.ndarray,
                   n_segs: int) -> np.ndarray:
    """Return n_segs+1 uniformly-parameterised points along quadratic Bézier p0,c,p2.

    One interior control point → a single convex arc (no inflections), which gives
    more regular hinge boundaries than a cubic and fewer design parameters.
    """
    t = np.linspace(0.0, 1.0, n_segs + 1)[:, None]
    return (1-t)**2 * p0 + 2*(1-t)*t * c + t**2 * p2


# ── hinge geometry (shared by builder and visualizer) ─────────────────────────

def compute_hinge_geometry(
    cs,
    gap: float = 0.003,
    bezier_params: dict = None,
    mesh_size_face: float = None,
    mesh_size_hinge: float = None,
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
        mesh_size_face: target element size in face bodies [m].
        mesh_size_hinge: target element size inside the hinge [m].

    Returns:
        dict with keys ``face_verts`` (n_faces, 4, 2, after translation),
        ``hinge_data`` (per-hinge anchor/CP dicts), ``clamped_faces``,
        ``loaded_faces``, ``lc_f``, ``lc_h``.
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

    if mesh_size_face is None:
        span = float(np.ptp(face_verts.reshape(-1, 2), axis=0).max())
        mesh_size_face = span / 8.0
    if mesh_size_hinge is None:
        mesh_size_hinge = max(gap * 1.5, 1e-5)
    lc_f, lc_h = mesh_size_face, mesh_size_hinge

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
        'lc_f': lc_f, 'lc_h': lc_h,
    }


# ── main builder ──────────────────────────────────────────────────────────────

def build_mesh_gmsh(
    cs,
    gap: float = 0.003,
    sheet_thickness: float = 0.001,
    bezier_params: dict = None,
    mesh_size_face: float = None,
    mesh_size_hinge: float = None,
    n_z_layers: int = 2,
    verbose: bool = False,
) -> tuple:
    """
    Build a tetrahedral FEM mesh for a kirigami unit cell using gmsh.

    Parameters
    ----------
    cs                  : CentroidalState (or SimpleNamespace from build_physical_cs)
    gap                 : rigid separation of the two faces along ê_arm [m].
                          Translates whole faces apart, opening the hinge void.
    sheet_thickness     : extrusion depth in z [m]
    bezier_params       : dict controlling hinge shape (all keys optional):
                            's0_top', 's0_bot' — reach along face-i edges [m]
                            's1_top', 's1_bot' — reach along face-k edges [m]
                            'bc_up_xy' — single upper-arc CP, absolute XY [m]
                            'bc_lo_xy' — single lower-arc CP, absolute XY [m]
                          Missing reach keys default to gap; missing CPs default to
                          symmetrically bowed arcs (gap away from the chord axis).
    mesh_size_face      : target element size in face bodies [m]
    mesh_size_hinge     : target element size inside the hinge [m]
    n_z_layers          : prism layers through thickness
    verbose             : show gmsh output

    Returns
    -------
    nodes    : (N, 3) float64
    tets     : (M, 4) int32
    bc_masks : {'clamped': bool (N,), 'loaded': bool (N,)}
    """
    import gmsh

    # ── 1-3. Face geometry + hinge strip endpoints ───────────────────────────
    geo = compute_hinge_geometry(
        cs, gap=gap, bezier_params=bezier_params,
        mesh_size_face=mesh_size_face, mesh_size_hinge=mesh_size_hinge)
    face_verts    = geo['face_verts']
    hinge_data    = geo['hinge_data']
    clamped_faces = geo['clamped_faces']
    loaded_faces  = geo['loaded_faces']
    lc_f, lc_h    = geo['lc_f'], geo['lc_h']
    n_faces       = face_verts.shape[0]

    # ── 4. gmsh session ───────────────────────────────────────────────────────
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 3 if verbose else 0)
    gmsh.model.add("kirigami")

    # Point registry: rounded XY key → gmsh tag
    _pt_cache: dict[tuple, int] = {}

    def _add_point(xy: np.ndarray, lc: float) -> int:
        key = (round(float(xy[0]), 12), round(float(xy[1]), 12))
        if key not in _pt_cache:
            _pt_cache[key] = gmsh.model.geo.addPoint(float(xy[0]), float(xy[1]), 0.0, lc)
        return _pt_cache[key]

    # Shared-edge registry: frozenset{pt_a, pt_b} → (canonical_tag, from_pt)
    _edge_cache: dict[frozenset, tuple] = {}

    def _get_line(pt_a: int, pt_b: int) -> int:
        """Return signed curve tag for pt_a → pt_b, creating if needed."""
        key = frozenset([pt_a, pt_b])
        if key in _edge_cache:
            ctag, from_pt = _edge_cache[key]
            return ctag if from_pt == pt_a else -ctag
        ctag = gmsh.model.geo.addLine(pt_a, pt_b)
        _edge_cache[key] = (ctag, pt_a)
        return ctag

    # ── 5. Build hinge surfaces ───────────────────────────────────────────────
    # CCW loop: p0_top → p0_bot (face fi edge)
    #         → p1_bot (lower arc forward: p0_bot → p1_bot)
    #         → p1_top (face fk edge)
    #         → p0_top (upper arc reversed: p1_top → p0_top)
    hinge_surf_tags = []
    hinge_curve_data = []

    for hd in hinge_data:
        pt_p0_top = _add_point(hd['p0_top'], lc_h)
        pt_p0_bot = _add_point(hd['p0_bot'], lc_h)
        pt_p1_top = _add_point(hd['p1_top'], lc_h)
        pt_p1_bot = _add_point(hd['p1_bot'], lc_h)

        # Segment counts proportional to arc length
        def _n_segs(pa, c, pb):
            bow = float(np.linalg.norm(c - 0.5*(pa+pb)))
            arc = float(np.linalg.norm(pb - pa)) + 2.0 * bow
            return max(8, int(np.ceil(arc / lc_h)))

        n_up = _n_segs(hd['p0_top'], hd['bc_up'], hd['p1_top'])
        n_lo = _n_segs(hd['p0_bot'], hd['bc_lo'], hd['p1_bot'])

        # Upper arc: p0_top → p1_top
        pts_up = _bezier_sample(hd['p0_top'], hd['bc_up'], hd['p1_top'], n_up)
        tags_up = ([pt_p0_top]
                   + [_add_point(pts_up[k], lc_h) for k in range(1, n_up)]
                   + [pt_p1_top])
        bez_up = [gmsh.model.geo.addLine(tags_up[k], tags_up[k+1]) for k in range(n_up)]

        # Lower arc: p0_bot → p1_bot
        pts_lo = _bezier_sample(hd['p0_bot'], hd['bc_lo'], hd['p1_bot'], n_lo)
        tags_lo = ([pt_p0_bot]
                   + [_add_point(pts_lo[k], lc_h) for k in range(1, n_lo)]
                   + [pt_p1_bot])
        bez_lo = [gmsh.model.geo.addLine(tags_lo[k], tags_lo[k+1]) for k in range(n_lo)]

        # Closing edges along face boundaries (shared with face outlines via _get_line)
        edge_fi = _get_line(pt_p0_top, pt_p0_bot)   # face fi side: p0_top → p0_bot
        edge_fk = _get_line(pt_p1_bot, pt_p1_top)   # face fk side: p1_bot → p1_top

        # CCW loop: edge_fi + bez_lo + edge_fk + (bez_up reversed)
        hinge_loop = gmsh.model.geo.addCurveLoop(
            [edge_fi]
            + bez_lo
            + [edge_fk]
            + [-l for l in reversed(bez_up)]
        )
        hinge_surf_tags.append(gmsh.model.geo.addPlaneSurface([hinge_loop]))

        hinge_curve_data.append({
            'fi': hd['fi'], 'lj': hd['lj'],
            'fk': hd['fk'], 'll': hd['ll'],
            'pt_p0_top': pt_p0_top, 'pt_p0_bot': pt_p0_bot,
            'pt_p1_top': pt_p1_top, 'pt_p1_bot': pt_p1_bot,
            'p0_top': hd['p0_top'], 'p0_bot': hd['p0_bot'],
            'p1_top': hd['p1_top'], 'p1_bot': hd['p1_bot'],
            'up_is_prev_fi': hd['up_is_prev_fi'],
            'up_is_prev_fk': hd['up_is_prev_fk'],
        })

    # Lookup: (face_idx, corner_lj) → (role, hcd)
    corner_hinge: dict[tuple, tuple] = {}
    for hcd in hinge_curve_data:
        corner_hinge[(hcd['fi'], hcd['lj'])] = ('fi', hcd)
        corner_hinge[(hcd['fk'], hcd['ll'])] = ('fk', hcd)

    # ── 6. Build face surfaces ────────────────────────────────────────────────
    # Each hinge corner lj on a face is replaced by TWO outline points (p_top, p_bot
    # in CCW order).  The closing edge between them is shared with the hinge surface.
    face_surf_tags = []
    face_polys_2d  = []

    for fi in range(n_faces):
        fv = face_verts[fi]   # (4, 2) CCW

        outline: list[tuple[int, np.ndarray]] = []

        for lj in range(4):
            if (fi, lj) in corner_hinge:
                role, hcd = corner_hinge[(fi, lj)]
                # Insert the two strip endpoints in CCW face order.
                # "first" = endpoint on the incoming edge (from vertex (lj-1)%4).
                if role == 'fi':
                    if hcd['up_is_prev_fi']:
                        # adj_up is at (lj-1)%4 → p0_top is on the incoming edge
                        outline.append((hcd['pt_p0_top'], hcd['p0_top']))
                        outline.append((hcd['pt_p0_bot'], hcd['p0_bot']))
                    else:
                        # adj_dn is at (lj-1)%4 → p0_bot is on the incoming edge
                        outline.append((hcd['pt_p0_bot'], hcd['p0_bot']))
                        outline.append((hcd['pt_p0_top'], hcd['p0_top']))
                else:  # 'fk'
                    if hcd['up_is_prev_fk']:
                        outline.append((hcd['pt_p1_top'], hcd['p1_top']))
                        outline.append((hcd['pt_p1_bot'], hcd['p1_bot']))
                    else:
                        outline.append((hcd['pt_p1_bot'], hcd['p1_bot']))
                        outline.append((hcd['pt_p1_top'], hcd['p1_top']))
            else:
                tag = _add_point(fv[lj], lc_f)
                outline.append((tag, fv[lj]))

        outline_pt_tags = [t for t, _ in outline]
        face_polys_2d.append(np.array([xy for _, xy in outline]))

        n_out = len(outline_pt_tags)
        curves = [_get_line(outline_pt_tags[k], outline_pt_tags[(k + 1) % n_out])
                  for k in range(n_out)]

        face_loop = gmsh.model.geo.addCurveLoop(curves)
        face_surf_tags.append(gmsh.model.geo.addPlaneSurface([face_loop]))

    # ── 7. Mesh 2D ────────────────────────────────────────────────────────────
    gmsh.model.geo.synchronize()
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc_h * 0.5)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc_f * 2.0)
    gmsh.model.mesh.generate(2)

    # ── 8. Extrude ────────────────────────────────────────────────────────────
    all_surfs = face_surf_tags + hinge_surf_tags
    gmsh.model.geo.extrude(
        [(2, tag) for tag in all_surfs],
        0.0, 0.0, float(sheet_thickness),
        numElements=[n_z_layers],
        recombine=False,
    )
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(3)

    # ── 9. Extract nodes ──────────────────────────────────────────────────────
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    coords = np.array(coords, dtype=np.float64).reshape(-1, 3)
    tag_to_idx = {int(t): i for i, t in enumerate(node_tags)}
    n_nodes = len(node_tags)

    # ── 10. Extract 3D elements → tets ───────────────────────────────────────
    etypes, _, enode_tags_list = gmsh.model.mesh.getElements(dim=3)
    tet_blocks = []
    el_nodes_per_type = {4: 4, 6: 6}
    for etype, enodes in zip(etypes, enode_tags_list):
        npe = el_nodes_per_type.get(int(etype))
        if npe is None:
            continue
        rows = np.array(enodes, dtype=np.int64).reshape(-1, npe)
        idx  = np.array([[tag_to_idx[int(t)] for t in r] for r in rows], np.int32)
        tet_blocks.append(_prisms_to_tets(idx) if npe == 6 else idx)

    if not tet_blocks:
        raise RuntimeError("gmsh produced no 3D elements.")
    tets = np.vstack(tet_blocks).astype(np.int32)

    gmsh.finalize()

    # ── 11. BC masks ──────────────────────────────────────────────────────────
    bc_masks: dict[str, np.ndarray] = {
        'clamped': np.zeros(n_nodes, bool),
        'loaded':  np.zeros(n_nodes, bool),
    }
    for fi_idx, poly_xy in enumerate(face_polys_2d):
        mask = _nodes_in_polygon(coords, poly_xy)
        if fi_idx in clamped_faces:
            bc_masks['clamped'] |= mask
        if fi_idx in loaded_faces:
            bc_masks['loaded'] |= mask

    return coords, tets, bc_masks
