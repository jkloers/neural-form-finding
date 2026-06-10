"""
nff/sofa/mesh_builder.py — Per-face bilinear hex mesh from CentroidalState.

Each face is meshed independently in its own parametric (s,t) space using
bilinear interpolation of the 4 physical corners.  Hinge strips are
parallelogram slabs sharing fold-boundary nodes with both adjacent face
meshes — fully conforming, no staircase, no disconnected components.

Public API
----------
  CLOSED_GAP_TOL  : float constant
  check_face_intersections(face_verts)
  build_mesh_from_centroidal_state(cs, fold_top, sheet_thickness,
                                   n_face=4, n_hinge=2, n_z=2,
                                   arm_width_physical=None,
                                   fold_bot=0.0, bezier_params=None)
  -> (nodes (N,3), hexes (H,8), bc_masks dict)
  fold_length kwarg retained as backward-compat alias for fold_top.

Corner-hinge Bézier parametrisation (bezier_params keys)
---------------------------------------------------------
When bezier_params is supplied and the hinge is a CORNER hinge (gap ≈ 0),
two wing strips are built — one above, one below the hinge axis:

  'bc1_upper_xy' : [x, y]  — 1st interior CP of upper-wing far-boundary Bézier
  'bc2_upper_xy' : [x, y]  — 2nd interior CP of upper-wing far-boundary Bézier
  'bc1_lower_xy' : [x, y]  — 1st interior CP of lower-wing Bézier
                              (None → mirrored from upper through hinge axis)
  'bc2_lower_xy' : [x, y]  — 2nd interior CP of lower-wing Bézier (same default)

All coordinates are in PHYSICAL metres, matching the face_verts coordinate frame.

For non-corner hinges, existing keys still apply:
  'waist_top', 'waist_bot', 'n_ctrl'
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple

CLOSED_GAP_TOL = 1e-6

# Parametric corners: _PC[lj] = (s, t)
_PC = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])


# ─── bilinear helpers ────────────────────────────────────────────────────────

def _blerp(c, s, t):
    """P(0,0)=c[0], P(1,0)=c[1], P(1,1)=c[2], P(0,1)=c[3]."""
    return (1-s)*(1-t)*c[0] + s*(1-t)*c[1] + s*t*c[2] + (1-s)*t*c[3]


def _stitch(breaks, nelems):
    pts = [float(breaks[0])]
    for (a, b), n in zip(zip(breaks[:-1], breaks[1:]), nelems):
        pts.extend(np.linspace(a, b, n + 1)[1:].tolist())
    return np.array(pts, dtype=np.float64)


def _bezier_eval(ctrl_pts: np.ndarray, t: float) -> np.ndarray:
    """Evaluate a Bézier curve at t ∈ [0,1] using De Casteljau's algorithm.

    Works for any degree (2 control pts = linear, 3 = quadratic, …).
    ctrl_pts: (n_ctrl, 2) array of 2-D control points.
    """
    pts = np.asarray(ctrl_pts, dtype=np.float64).copy()
    n = len(pts)
    for _ in range(1, n):
        pts[:n-1] = (1.0 - t) * pts[:n-1] + t * pts[1:n]
        n -= 1
    return pts[0]


# ─── face intersection detection (public) ───────────────────────────────────

def _strictly_inside(pt, poly, tol):
    n = len(poly)
    cs = []
    for i in range(n):
        v0, v1 = poly[i], poly[(i+1) % n]
        cs.append((v1[0]-v0[0])*(pt[1]-v0[1]) - (v1[1]-v0[1])*(pt[0]-v0[0]))
    return all(c > tol for c in cs) or all(c < -tol for c in cs)


def _seg_cross(a, b, c, d, tol):
    def x2(o, p, q):
        return (p[0]-o[0])*(q[1]-o[1]) - (p[1]-o[1])*(q[0]-o[0])
    d1, d2 = x2(c,d,a), x2(c,d,b)
    d3, d4 = x2(a,b,c), x2(a,b,d)
    return (((d1>tol and d2<-tol) or (d1<-tol and d2>tol)) and
            ((d3>tol and d4<-tol) or (d3<-tol and d4>tol)))


def _quads_overlap(p1, p2, tol=1e-8):
    for v in p1:
        if _strictly_inside(v, p2, tol): return True
    for v in p2:
        if _strictly_inside(v, p1, tol): return True
    n1, n2 = len(p1), len(p2)
    for i in range(n1):
        mid = (p1[i] + p1[(i+1)%n1]) * 0.5
        if _strictly_inside(mid, p2, tol): return True
    for j in range(n2):
        mid = (p2[j] + p2[(j+1)%n2]) * 0.5
        if _strictly_inside(mid, p1, tol): return True
    for i in range(n1):
        for j in range(n2):
            if _seg_cross(p1[i], p1[(i+1)%n1], p2[j], p2[(j+1)%n2], tol):
                return True
    return False


def check_face_intersections(face_verts: np.ndarray, tol: float = 1e-8) -> None:
    """Raise ValueError if any two faces have area overlap."""
    n = len(face_verts)
    bad = [(i, j) for i in range(n) for j in range(i+1, n)
           if _quads_overlap(face_verts[i], face_verts[j], tol)]
    if bad:
        raise ValueError(
            f"Face area intersections: {', '.join(f'({i},{j})' for i,j in bad)}. "
            "Fix Stage-1 output before meshing.")


# ─── geometry extraction ─────────────────────────────────────────────────────

def _extract_geometry(cs, fold_top, fold_bot=0.0, arm_width_physical=None):
    """
    Build adjusted face_verts and hinge strip descriptors from CentroidalState.

    Parameters
    ----------
    fold_top : float — fold zone depth for the far anchor on each face [m]
    fold_bot : float — fold zone depth for the near anchor (default 0 = corner)

    Returns
    -------
    face_verts : (n_faces, 4, 2) — corner positions after arm_width override
    hinge_strips : list of dicts per hinge (each has fold_top, fold_bot fields)
    """
    fc  = np.array(cs.face_centroids)
    cnv = np.array(cs.centroid_node_vectors)
    hnp = np.array(cs.hinge_node_pairs)
    adj = np.array(cs.hinge_adj_info)

    face_verts = (fc[:, None, :] + cnv[:, :4, :]).copy()
    n_hinges = adj.shape[0]

    check_face_intersections(face_verts)

    updates = []
    arm_widths = []
    for h in range(n_hinges):
        fi, lj = int(hnp[h,0,0]), int(hnp[h,0,1])
        fk, ll = int(hnp[h,1,0]), int(hnp[h,1,1])
        p1o = face_verts[fi, lj].copy()
        p2o = face_verts[fk, ll].copy()
        gap = float(np.linalg.norm(p2o - p1o))
        if gap < CLOSED_GAP_TOL:
            if not arm_width_physical or arm_width_physical <= 0:
                raise ValueError(
                    f"Hinge {h} (faces {fi}↔{fk}) gap≈0 but arm_width_physical unset.")
            p_mid = (p1o + p2o) * 0.5
            d = fc[fk] - fc[fi]
            nd = float(np.linalg.norm(d))
            hdir = d / nd if nd > 1e-12 else np.array([1.0, 0.0])
            hw = arm_width_physical * 0.5
            updates.append((fi, lj, p_mid - hw*hdir, fk, ll, p_mid + hw*hdir))
            arm_widths.append(arm_width_physical)
        else:
            updates.append((fi, lj, p1o, fk, ll, p2o))
            arm_widths.append(gap)

    for fi, lj, p1e, fk, ll, p2e in updates:
        face_verts[fi, lj] = p1e
        face_verts[fk, ll] = p2e

    hinge_strips = []
    for h in range(n_hinges):
        fi, lj = int(hnp[h,0,0]), int(hnp[h,0,1])
        fk, ll = int(hnp[h,1,0]), int(hnp[h,1,1])
        adj_li = int(adj[h, 3])
        adj_lk = int(adj[h, 4])
        p1 = face_verts[fi, lj].copy()
        p2 = face_verts[fk, ll].copy()
        # Use post-adjustment corners for exact conformity
        raw_i = face_verts[fi, adj_li] - p1
        ni = float(np.linalg.norm(raw_i))
        fold_dir_i = raw_i / ni if ni > 1e-12 else np.array([1.0, 0.0])
        raw_k = face_verts[fk, adj_lk] - p2
        nk = float(np.linalg.norm(raw_k))
        fold_dir_k = raw_k / nk if nk > 1e-12 else np.array([1.0, 0.0])
        hinge_strips.append({
            'h': h, 'fi': fi, 'lj': lj, 'fk': fk, 'll': ll,
            'adj_li': adj_li, 'adj_lk': adj_lk,
            'p1': p1, 'p2': p2,
            'fold_dir_i': fold_dir_i, 'fold_dir_k': fold_dir_k,
            'arm_width': arm_widths[h],
            'fold_top': float(fold_top),
            'fold_bot': float(fold_bot),
        })

    return face_verts, hinge_strips


# ─── parametric grid helpers ─────────────────────────────────────────────────

def _fold_axis_info(lj, adj_li):
    """
    Returns (fold_axis, fold_sign, arm_axis, arm_val_sc_or_tc).

    fold_axis : 's' or 't'  — parametric direction of fold
    fold_sign : +1 or -1
    arm_axis  : 's' or 't'  — perpendicular axis (where arm edge lies)
    arm_val   : 0.0 or 1.0  — parametric value of arm edge on this face
    """
    sc, tc = _PC[lj]
    sa, ta = _PC[adj_li]
    ds, dt = sa - sc, ta - tc
    if abs(ds) >= abs(dt):
        return 's', int(np.sign(ds)), 't', float(tc)
    else:
        return 't', int(np.sign(dt)), 's', float(sc)


def _face_grid_breaks(face_verts_i, hinge_corners, n_face, n_hinge):
    """
    Build parametric break points and element counts for one face.

    hinge_corners : list of (lj, adj_li, fold_top, fold_bot)
      fold_top — far anchor depth [m]; fold_bot — near anchor depth [m] (0 = corner)

    Returns s_breaks, t_breaks, s_nelems, t_nelems
    """
    s_set = {0.0, 1.0}
    t_set = {0.0, 1.0}

    for lj, adj_li, fold_top, fold_bot in hinge_corners:
        fold_axis, fold_sign, _, _ = _fold_axis_info(lj, adj_li)
        c_lj  = face_verts_i[lj]
        c_adj = face_verts_i[adj_li]
        phys  = float(np.linalg.norm(c_adj - c_lj))
        ff_top = min(fold_top / phys if phys > 1e-12 else 0.1, 0.49)
        sc, tc = _PC[lj]
        if fold_axis == 's':
            s_set.add(float(np.clip(sc + fold_sign * ff_top, 0.0, 1.0)))
            if fold_bot > 1e-9:
                ff_bot = min(fold_bot / phys if phys > 1e-12 else 0.0, ff_top - 1e-6)
                s_set.add(float(np.clip(sc + fold_sign * ff_bot, 0.0, 1.0)))
        else:
            t_set.add(float(np.clip(tc + fold_sign * ff_top, 0.0, 1.0)))
            if fold_bot > 1e-9:
                ff_bot = min(fold_bot / phys if phys > 1e-12 else 0.0, ff_top - 1e-6)
                t_set.add(float(np.clip(tc + fold_sign * ff_bot, 0.0, 1.0)))

    def _ne(vals):
        if len(vals) < 2:
            return []
        lengths = [vals[i+1] - vals[i] for i in range(len(vals)-1)]
        mx = max(lengths)
        return [n_face if l >= 0.5*mx else n_hinge for l in lengths]

    s_breaks = sorted(s_set)
    t_breaks = sorted(t_set)
    return s_breaks, t_breaks, _ne(s_breaks), _ne(t_breaks)


def _find_closest_idx(arr, val, tol=1e-9):
    idx = int(np.argmin(np.abs(arr - val)))
    if abs(arr[idx] - val) > tol:
        raise ValueError(f"Grid value {val} not found in {arr}")
    return idx


def _interface_slice(node_grid, s_nodes, t_nodes, lj, adj_li,
                     fold_top, fold_bot, face_verts_i):
    """
    Return (n_fold+1, nz) array of global node indices at the hinge interface.

    The slice spans from the near anchor (fold_bot from corner) to the far
    anchor (fold_top from corner).  Result is always ordered so that index 0
    corresponds to the near anchor (fold_bot side) and index -1 to the far
    anchor (fold_top side), regardless of fold_sign.

    fold_top : far anchor depth [m] (was fold_length)
    fold_bot : near anchor depth [m] (0 = hinge corner)
    """
    fold_axis, fold_sign, arm_axis, arm_val = _fold_axis_info(lj, adj_li)
    c_lj  = face_verts_i[lj]
    c_adj = face_verts_i[adj_li]
    phys  = float(np.linalg.norm(c_adj - c_lj))
    ff_top = min(fold_top / phys if phys > 1e-12 else 0.1, 0.49)
    ff_bot = (min(fold_bot / phys if phys > 1e-12 else 0.0, ff_top - 1e-6)
              if fold_bot > 1e-9 else 0.0)
    sc, tc = _PC[lj]

    if arm_axis == 's':
        arm_idx = _find_closest_idx(s_nodes, arm_val)
    else:
        arm_idx = _find_closest_idx(t_nodes, arm_val)

    if fold_axis == 't':
        near_param = tc + fold_sign * ff_bot   # parametric near-anchor position
        far_param  = tc + fold_sign * ff_top   # parametric far-anchor position
        lo = min(near_param, far_param)
        hi = max(near_param, far_param)
        fi_lo = _find_closest_idx(t_nodes, lo)
        fi_hi = _find_closest_idx(t_nodes, hi)
        result = node_grid[arm_idx, fi_lo:fi_hi+1, :]   # (n_fold+1, nz)
        # Ensure index-0 = near anchor.  When fold_sign > 0: lo = near, already correct.
        # When fold_sign < 0: lo = far, so reverse.
        if fold_sign < 0:
            result = result[::-1, :]
        return result
    else:
        near_param = sc + fold_sign * ff_bot
        far_param  = sc + fold_sign * ff_top
        lo = min(near_param, far_param)
        hi = max(near_param, far_param)
        fi_lo = _find_closest_idx(s_nodes, lo)
        fi_hi = _find_closest_idx(s_nodes, hi)
        result = node_grid[fi_lo:fi_hi+1, arm_idx, :]   # (n_fold+1, nz)
        if fold_sign < 0:
            result = result[::-1, :]
        return result


# ─── face mesh builder ───────────────────────────────────────────────────────

def _mesh_face(face_verts_i, hinge_corners, sheet_thickness,
               n_face, n_hinge, n_z, node_list):
    """
    Build bilinear hex mesh for one face.

    hinge_corners : list of (lj, adj_li, fold_top, fold_bot)

    Returns
    -------
    node_grid : (ns, nt, nz) int32 — global node indices
    s_nodes   : (ns,) parametric s values
    t_nodes   : (nt,) parametric t values
    z_vals    : (nz,) z coordinates
    hexes     : list of [8] int connectivity
    """
    s_breaks, t_breaks, s_ne, t_ne = _face_grid_breaks(
        face_verts_i, hinge_corners, n_face, n_hinge)

    s_nodes = _stitch(s_breaks, s_ne)
    t_nodes = _stitch(t_breaks, t_ne)
    z_vals  = np.linspace(-sheet_thickness/2, sheet_thickness/2, n_z + 1)

    ns, nt, nz = len(s_nodes), len(t_nodes), len(z_vals)
    c = face_verts_i

    node_grid = np.empty((ns, nt, nz), dtype=np.int32)
    base = len(node_list)
    for iz in range(nz):
        for it in range(nt):
            for is_ in range(ns):
                xy = _blerp(c, s_nodes[is_], t_nodes[it])
                node_list.append([xy[0], xy[1], z_vals[iz]])
                node_grid[is_, it, iz] = base
                base += 1

    hexes = []
    for iz in range(nz - 1):
        for it in range(nt - 1):
            for is_ in range(ns - 1):
                a = node_grid[is_,   it,   iz  ]
                b = node_grid[is_+1, it,   iz  ]
                c_ = node_grid[is_+1, it+1, iz  ]
                d = node_grid[is_,   it+1, iz  ]
                e = node_grid[is_,   it,   iz+1]
                f = node_grid[is_+1, it,   iz+1]
                g = node_grid[is_+1, it+1, iz+1]
                h = node_grid[is_,   it+1, iz+1]
                hexes.append([a, b, c_, d, e, f, g, h])

    return node_grid, s_nodes, t_nodes, z_vals, hexes


# ─── hinge strip builder ─────────────────────────────────────────────────────

def _mesh_hinge(hs, n_hinge, face_verts,
                face_node_grids, face_s_nodes, face_t_nodes, face_z_vals,
                node_list, bezier_params=None):
    """
    Build hinge strip hexes.

    Shares fold-boundary nodes with both adjacent face meshes (via
    _interface_slice).  Adds new interior columns (u = 1..n_arm-1) only.

    Parameters
    ----------
    bezier_params : None | dict
        If None: bilinear transfinite interpolation.
        If dict:  Bézier transfinite interpolation with keys
          'waist_top' — control-point offset (fold direction) for far Bézier [m]
          'waist_bot' — control-point offset (fold direction) for near Bézier [m]
          'n_ctrl'    — number of Bézier control points (default 3 = quadratic)

    Returns list of [8] hex connectivity.
    """
    fi, lj, adj_li = hs['fi'], hs['lj'], hs['adj_li']
    fk, ll, adj_lk = hs['fk'], hs['ll'], hs['adj_lk']
    p1, p2     = hs['p1'], hs['p2']
    fdi, fdk   = hs['fold_dir_i'], hs['fold_dir_k']
    fold_top   = hs['fold_top']
    fold_bot   = hs['fold_bot']
    n_arm  = n_hinge
    n_fold = n_hinge

    z_vals = face_z_vals[fi]
    nz = len(z_vals)

    # _interface_slice already orders index-0 = near anchor, index-1 = far anchor.
    i_nodes = _interface_slice(
        face_node_grids[fi], face_s_nodes[fi], face_t_nodes[fi],
        lj, adj_li, fold_top, fold_bot, face_verts[fi])
    k_nodes = _interface_slice(
        face_node_grids[fk], face_s_nodes[fk], face_t_nodes[fk],
        ll, adj_lk, fold_top, fold_bot, face_verts[fk])

    nv = i_nodes.shape[0]
    assert nv == n_fold + 1, f"Interface mismatch: got {nv}, expected {n_fold+1}"

    # Strip anchor corners (physical XY)
    P0_bot = p1 + fdi * fold_bot   # face i, near anchor
    P2_bot = p2 + fdk * fold_bot   # face k, near anchor
    P0_top = p1 + fdi * fold_top   # face i, far anchor
    P2_top = p2 + fdk * fold_top   # face k, far anchor

    # Full strip grid: (n_arm+1, n_fold+1, nz) — index-0 in v = near side
    strip = np.full((n_arm+1, n_fold+1, nz), -1, dtype=np.int32)
    strip[0, :, :]     = i_nodes
    strip[n_arm, :, :] = k_nodes

    if bezier_params is not None:
        # Bézier transfinite interpolation:
        #   xy(u,v) = (1-v)*B_bot(u) + v*B_top(u)
        # where B_bot / B_top are quadratic (n_ctrl=3) Bézier curves.
        # The waist control point is offset along the average fold direction.
        waist_top = float(bezier_params.get('waist_top', 0.0))
        waist_bot = float(bezier_params.get('waist_bot', 0.0))
        n_ctrl    = int(bezier_params.get('n_ctrl', 3))
        fold_mid  = (fdi + fdk) * 0.5
        nm = float(np.linalg.norm(fold_mid))
        fold_mid_hat = fold_mid / nm if nm > 1e-12 else fdi

        def _build_ctrl(P0, P2, waist, n):
            """Build n control points between P0 and P2 with waist offset."""
            if n <= 2:
                return np.stack([P0, P2])
            mid = (P0 + P2) * 0.5 + waist * fold_mid_hat
            if n == 3:
                return np.stack([P0, mid, P2])
            # For n > 3: distribute extra interior points linearly,
            # with the midpoint kept at the waist position.
            pts = [P0]
            for k in range(1, n - 1):
                t = k / (n - 1)
                base_pt = (1 - t) * P0 + t * P2
                # gaussian bump centred at t=0.5
                bump = waist * np.exp(-8.0 * (t - 0.5)**2) * fold_mid_hat
                pts.append(base_pt + bump)
            pts.append(P2)
            return np.stack(pts)

        ctrl_bot = _build_ctrl(P0_bot, P2_bot, waist_bot, n_ctrl)
        ctrl_top = _build_ctrl(P0_top, P2_top, waist_top, n_ctrl)

        base = len(node_list)
        for iu in range(1, n_arm):
            u = iu / n_arm
            b_bot = _bezier_eval(ctrl_bot, u)
            b_top = _bezier_eval(ctrl_top, u)
            for iv in range(n_fold + 1):
                v = iv / n_fold
                xy = (1.0 - v) * b_bot + v * b_top
                for iw in range(nz):
                    node_list.append([xy[0], xy[1], z_vals[iw]])
                    strip[iu, iv, iw] = base
                    base += 1
    else:
        # Bilinear transfinite interpolation (original behaviour).
        hs0, hs1, hs2, hs3 = P0_bot, P2_bot, P2_top, P0_top
        base = len(node_list)
        for iu in range(1, n_arm):
            u = iu / n_arm
            for iv in range(n_fold + 1):
                v = iv / n_fold
                xy = _blerp([hs0, hs1, hs2, hs3], u, v)
                for iw in range(nz):
                    node_list.append([xy[0], xy[1], z_vals[iw]])
                    strip[iu, iv, iw] = base
                    base += 1

    hexes = []
    for iu in range(n_arm):
        for iv in range(n_fold):
            for iw in range(nz - 1):
                a  = strip[iu,   iv,   iw  ]
                b  = strip[iu+1, iv,   iw  ]
                c_ = strip[iu+1, iv+1, iw  ]
                d  = strip[iu,   iv+1, iw  ]
                e  = strip[iu,   iv,   iw+1]
                f  = strip[iu+1, iv,   iw+1]
                g  = strip[iu+1, iv+1, iw+1]
                h  = strip[iu,   iv+1, iw+1]
                hexes.append([a, b, c_, d, e, f, g, h])

    return hexes


# ─── node merging ────────────────────────────────────────────────────────────

def _merge_nodes(node_list, all_hexes, face_node_grids, tol):
    """
    Merge coincident nodes (same physical position within tol).

    Returns updated node array, hex array, and updated face_node_grids.
    Necessary because face corners that are physically connected (not at a
    hinge or cut) are generated as separate indices by the per-face builder.
    """
    from scipy.spatial import cKDTree
    nodes = np.array(node_list, dtype=np.float64)
    tree  = cKDTree(nodes)
    pairs = tree.query_pairs(tol)

    if not pairs:
        return nodes, np.array(all_hexes, dtype=np.int32), face_node_grids

    # Build canonical representative: smaller index wins
    parent = np.arange(len(nodes), dtype=np.int32)
    for a, b in sorted(pairs):
        lo, hi = min(a, b), max(a, b)
        parent[hi] = lo  # direct assign (no cycles for point-set merges at once)

    # Flatten parent to root (single pass is sufficient for direct assignments)
    for i in range(len(parent)):
        while parent[parent[i]] != parent[i]:
            parent[i] = parent[parent[i]]

    # Compact: unique canonical nodes
    canon = np.unique(parent)
    remap = np.full(len(nodes), -1, dtype=np.int32)
    remap[canon] = np.arange(len(canon), dtype=np.int32)
    remap_all = remap[parent]  # maps old index → new compact index

    merged_nodes = nodes[canon]
    merged_hexes = remap_all[np.array(all_hexes, dtype=np.int32)]

    # Update face_node_grids to use new indices
    merged_grids = [remap_all[ng] for ng in face_node_grids]

    return merged_nodes, merged_hexes, merged_grids


# ─── public entry point ──────────────────────────────────────────────────────

def build_mesh_from_centroidal_state(
    cs,
    fold_length: float = None,
    sheet_thickness: float = 0.001,
    n_face: int = 4,
    n_hinge: int = 2,
    n_z: int = 2,
    arm_width_physical: float = None,
    arm_width_override: float = None,
    fold_top: float = None,
    fold_bot: float = 0.0,
    bezier_params: dict = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Build a conforming SOFA hex mesh from a CentroidalState.

    Per-face bilinear meshing with conforming hinge strips.
    No global Cartesian grid — works for any face rotation.

    Parameters
    ----------
    cs                 : CentroidalState (Stage-1 output, physical units)
    fold_length        : legacy alias for fold_top (backward compat)
    sheet_thickness    : full sheet thickness in z [m]
    n_face             : hex elements per face-panel segment (default 4)
    n_hinge            : hex elements per hinge/fold segment (default 2)
    n_z                : hex layers through thickness (default 2)
    arm_width_physical : required for closed hinges (gap < CLOSED_GAP_TOL)
    fold_top           : far anchor depth along face edge [m] (replaces fold_length)
    fold_bot           : near anchor depth along face edge [m] (default 0 = corner)
    bezier_params      : None for bilinear strip interpolation; or dict with keys:
                           'waist_top' — fold-dir offset for far Bézier control pt [m]
                           'waist_bot' — fold-dir offset for near Bézier control pt [m]
                           'n_ctrl'    — Bézier degree + 1 (default 3 = quadratic)

    Returns
    -------
    nodes    : (N, 3) float64
    hexes    : (H, 8) int32
    bc_masks : dict  'f{i}' / 'face_{i}' / 'clamped' / 'loaded'
    """
    # Backward-compat: fold_length is an alias for fold_top.
    if fold_top is None:
        if fold_length is not None:
            fold_top = fold_length
        else:
            raise ValueError("Provide fold_top (or legacy fold_length).")

    if arm_width_override is not None and arm_width_physical is None:
        arm_width_physical = arm_width_override

    face_verts, hinge_strips = _extract_geometry(
        cs, fold_top, fold_bot, arm_width_physical)
    n_faces = len(face_verts)

    # ── collect hinge corners per face ───────────────────────────────────────
    face_hinge_corners: List[List] = [[] for _ in range(n_faces)]
    for hs in hinge_strips:
        face_hinge_corners[hs['fi']].append(
            (hs['lj'], hs['adj_li'], hs['fold_top'], hs['fold_bot']))
        face_hinge_corners[hs['fk']].append(
            (hs['ll'], hs['adj_lk'], hs['fold_top'], hs['fold_bot']))

    # ── mesh each face ───────────────────────────────────────────────────────
    node_list: List = []
    face_node_grids = []
    face_s_nodes    = []
    face_t_nodes    = []
    face_z_vals_list = []
    all_hexes: List = []

    for i in range(n_faces):
        ng, sn, tn, zv, fh = _mesh_face(
            face_verts[i], face_hinge_corners[i], sheet_thickness,
            n_face, n_hinge, n_z, node_list)
        face_node_grids.append(ng)
        face_s_nodes.append(sn)
        face_t_nodes.append(tn)
        face_z_vals_list.append(zv)
        all_hexes.extend(fh)

    # ── mesh each hinge strip ────────────────────────────────────────────────
    for hs in hinge_strips:
        hh = _mesh_hinge(
            hs, n_hinge,
            face_verts, face_node_grids, face_s_nodes, face_t_nodes, face_z_vals_list,
            node_list, bezier_params=bezier_params)
        all_hexes.extend(hh)

    # ── merge coincident nodes (free corners shared across faces) ────────────
    # Tolerance: 1 ppm of the hinge arm width (much smaller than any intended gap)
    merge_tol = (arm_width_physical or fold_top) * 1e-6
    nodes, hexes, face_node_grids = _merge_nodes(
        node_list, all_hexes, face_node_grids, tol=merge_tol)
    n_nodes = len(nodes)

    # ── BC masks: each face owns exactly the nodes in its node_grid ──────────
    bc_masks: Dict[str, np.ndarray] = {}
    for i in range(n_faces):
        mask = np.zeros(n_nodes, dtype=bool)
        mask[face_node_grids[i].ravel()] = True
        bc_masks[f'face_{i}'] = mask
        bc_masks[f'f{i}'] = mask

    # ── clamped / loaded from CS BCs ─────────────────────────────────────────
    constrained = np.array(cs.constrained_face_DOF_pairs)
    loaded_dofs  = np.array(cs.loaded_face_DOF_pairs)

    clamped = np.zeros(n_nodes, dtype=bool)
    if constrained.size > 0:
        for f in np.unique(constrained[:, 0]):
            clamped |= bc_masks[f'face_{int(f)}']
    bc_masks['clamped'] = clamped

    loaded = np.zeros(n_nodes, dtype=bool)
    if loaded_dofs.size > 0:
        for f in np.unique(loaded_dofs[:, 0]):
            loaded |= bc_masks[f'face_{int(f)}']
    bc_masks['loaded'] = loaded

    return nodes, hexes, bc_masks
