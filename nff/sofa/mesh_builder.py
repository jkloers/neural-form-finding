"""
nff/sofa/mesh_builder.py — Per-face bilinear hex mesh from CentroidalState.

Each face is meshed independently in its own parametric (s,t) space using
bilinear interpolation of the 4 physical corners.  Hinge strips are
parallelogram slabs sharing fold-boundary nodes with both adjacent face
meshes — fully conforming, no staircase, no disconnected components.

Public API (unchanged)
----------------------
  CLOSED_GAP_TOL  : float constant
  check_face_intersections(face_verts)
  build_mesh_from_centroidal_state(cs, fold_length, sheet_thickness,
                                   n_face=4, n_hinge=2, n_z=2,
                                   arm_width_physical=None)
  -> (nodes (N,3), hexes (H,8), bc_masks dict)
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

def _extract_geometry(cs, fold_length, arm_width_physical=None):
    """
    Build adjusted face_verts and hinge strip descriptors from CentroidalState.

    Returns
    -------
    face_verts : (n_faces, 4, 2) — corner positions after arm_width override
    hinge_strips : list of dicts per hinge
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


def _face_grid_breaks(face_verts_i, hinge_corners, fold_length, n_face, n_hinge):
    """
    Build parametric break points and element counts for one face.

    hinge_corners : list of (lj, adj_li)

    Returns s_breaks, t_breaks, s_nelems, t_nelems
    """
    s_set = {0.0, 1.0}
    t_set = {0.0, 1.0}

    for lj, adj_li in hinge_corners:
        fold_axis, fold_sign, _, _ = _fold_axis_info(lj, adj_li)
        c_lj  = face_verts_i[lj]
        c_adj = face_verts_i[adj_li]
        phys  = float(np.linalg.norm(c_adj - c_lj))
        ff    = min(fold_length / phys if phys > 1e-12 else 0.1, 0.49)
        sc, tc = _PC[lj]
        if fold_axis == 's':
            s_set.add(float(np.clip(sc + fold_sign * ff, 0.0, 1.0)))
        else:
            t_set.add(float(np.clip(tc + fold_sign * ff, 0.0, 1.0)))

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


def _interface_slice(node_grid, s_nodes, t_nodes, lj, adj_li, fold_length, face_verts_i):
    """
    Return (n_fold+1, nz) array of global node indices at the
    hinge interface on one face (at the arm edge, fold zone).
    """
    fold_axis, fold_sign, arm_axis, arm_val = _fold_axis_info(lj, adj_li)
    c_lj  = face_verts_i[lj]
    c_adj = face_verts_i[adj_li]
    phys  = float(np.linalg.norm(c_adj - c_lj))
    ff    = min(fold_length / phys if phys > 1e-12 else 0.1, 0.49)
    sc, tc = _PC[lj]

    if arm_axis == 's':
        arm_idx = _find_closest_idx(s_nodes, arm_val)
    else:
        arm_idx = _find_closest_idx(t_nodes, arm_val)

    if fold_axis == 't':
        fold_start = tc
        fold_end   = tc + fold_sign * ff
        if fold_sign < 0:
            fold_start, fold_end = fold_end, fold_start
        fi_start = _find_closest_idx(t_nodes, fold_start)
        fi_end   = _find_closest_idx(t_nodes, fold_end)
        # arm_axis == 's': slice [arm_idx, fi_start:fi_end+1, :]
        return node_grid[arm_idx, fi_start:fi_end+1, :]   # (n_fold+1, nz)
    else:
        fold_start = sc
        fold_end   = sc + fold_sign * ff
        if fold_sign < 0:
            fold_start, fold_end = fold_end, fold_start
        fi_start = _find_closest_idx(s_nodes, fold_start)
        fi_end   = _find_closest_idx(s_nodes, fold_end)
        # arm_axis == 't': slice [fi_start:fi_end+1, arm_idx, :]
        return node_grid[fi_start:fi_end+1, arm_idx, :]   # (n_fold+1, nz)


# ─── face mesh builder ───────────────────────────────────────────────────────

def _mesh_face(face_verts_i, hinge_corners, fold_length, sheet_thickness,
               n_face, n_hinge, n_z, node_list):
    """
    Build bilinear hex mesh for one face.

    Returns
    -------
    node_grid : (ns, nt, nz) int32 — global node indices
    s_nodes   : (ns,) parametric s values
    t_nodes   : (nt,) parametric t values
    z_vals    : (nz,) z coordinates
    hexes     : list of [8] int connectivity
    """
    s_breaks, t_breaks, s_ne, t_ne = _face_grid_breaks(
        face_verts_i, hinge_corners, fold_length, n_face, n_hinge)

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

def _mesh_hinge(hs, fold_length, n_hinge, face_verts,
                face_node_grids, face_s_nodes, face_t_nodes, face_z_vals,
                node_list):
    """
    Build hinge strip hexes.

    Shares fold-boundary nodes with both adjacent face meshes.
    Adds new interior columns (u = 1 .. n_arm-1) only.

    Returns list of [8] hex connectivity.
    """
    fi, lj, adj_li = hs['fi'], hs['lj'], hs['adj_li']
    fk, ll, adj_lk = hs['fk'], hs['ll'], hs['adj_lk']
    p1, p2 = hs['p1'], hs['p2']
    fdi, fdk = hs['fold_dir_i'], hs['fold_dir_k']
    n_arm  = n_hinge
    n_fold = n_hinge

    z_vals = face_z_vals[fi]
    nz = len(z_vals)

    # Interface node arrays: shape (n_fold+1, nz)
    # _interface_slice orders nodes by increasing parametric value.
    # When fold_sign < 0 the corner node (v=0 in the strip) is at the HIGH end
    # of the parametric range, so the slice is reversed — fix it here so that
    # i_nodes[0] / k_nodes[0] always corresponds to the hinge corner (v=0).
    _, fold_sign_i, _, _ = _fold_axis_info(lj, adj_li)
    _, fold_sign_k, _, _ = _fold_axis_info(ll, adj_lk)

    i_nodes = _interface_slice(
        face_node_grids[fi], face_s_nodes[fi], face_t_nodes[fi],
        lj, adj_li, fold_length, face_verts[fi])
    k_nodes = _interface_slice(
        face_node_grids[fk], face_s_nodes[fk], face_t_nodes[fk],
        ll, adj_lk, fold_length, face_verts[fk])

    if fold_sign_i < 0:
        i_nodes = i_nodes[::-1, :]
    if fold_sign_k < 0:
        k_nodes = k_nodes[::-1, :]

    nv = i_nodes.shape[0]
    assert nv == n_fold + 1, f"Interface mismatch: got {nv}, expected {n_fold+1}"

    # Strip corners (bilinear in u-v plane)
    hs0 = p1
    hs1 = p2
    hs2 = p2 + fdk * fold_length
    hs3 = p1 + fdi * fold_length

    # Full strip grid: (n_arm+1, n_fold+1, nz)
    strip = np.full((n_arm+1, n_fold+1, nz), -1, dtype=np.int32)
    strip[0, :, :]      = i_nodes
    strip[n_arm, :, :]  = k_nodes

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
    fold_length: float,
    sheet_thickness: float,
    n_face: int = 4,
    n_hinge: int = 2,
    n_z: int = 2,
    arm_width_physical: float = None,
    arm_width_override: float = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Build a conforming SOFA hex mesh from a CentroidalState.

    Per-face bilinear meshing with conforming hinge strips.
    No global Cartesian grid — works for any face rotation.

    Parameters
    ----------
    cs                 : CentroidalState (Stage-1 output, physical units)
    fold_length        : hinge strip depth along face edge [m]
    sheet_thickness    : full sheet thickness in z [m]
    n_face             : hex elements per face-panel segment (default 4)
    n_hinge            : hex elements per hinge/fold segment (default 2)
    n_z                : hex layers through thickness (default 2)
    arm_width_physical : required for closed hinges (gap < CLOSED_GAP_TOL)

    Returns
    -------
    nodes    : (N, 3) float64
    hexes    : (H, 8) int32
    bc_masks : dict  'f{i}' / 'face_{i}' / 'clamped' / 'loaded'
    """
    if arm_width_override is not None and arm_width_physical is None:
        arm_width_physical = arm_width_override

    face_verts, hinge_strips = _extract_geometry(cs, fold_length, arm_width_physical)
    n_faces = len(face_verts)

    # ── collect hinge corners per face ───────────────────────────────────────
    face_hinge_corners: List[List] = [[] for _ in range(n_faces)]
    for hs in hinge_strips:
        face_hinge_corners[hs['fi']].append((hs['lj'], hs['adj_li']))
        face_hinge_corners[hs['fk']].append((hs['ll'], hs['adj_lk']))

    # ── mesh each face ───────────────────────────────────────────────────────
    node_list: List = []
    face_node_grids = []
    face_s_nodes    = []
    face_t_nodes    = []
    face_z_vals_list = []
    all_hexes: List = []

    for i in range(n_faces):
        ng, sn, tn, zv, fh = _mesh_face(
            face_verts[i], face_hinge_corners[i], fold_length, sheet_thickness,
            n_face, n_hinge, n_z, node_list)
        face_node_grids.append(ng)
        face_s_nodes.append(sn)
        face_t_nodes.append(tn)
        face_z_vals_list.append(zv)
        all_hexes.extend(fh)

    # ── mesh each hinge strip ────────────────────────────────────────────────
    for hs in hinge_strips:
        hh = _mesh_hinge(
            hs, fold_length, n_hinge,
            face_verts, face_node_grids, face_s_nodes, face_t_nodes, face_z_vals_list,
            node_list)
        all_hexes.extend(hh)

    # ── merge coincident nodes (free corners shared across faces) ────────────
    # Tolerance: 1 ppm of the hinge arm width (much smaller than any intended gap)
    merge_tol = (arm_width_physical or fold_length) * 1e-6
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
