"""
nff/sofa/mesh_builder.py
========================
Build a conforming SOFA hex mesh directly from a CentroidalState.

All face panel and hinge strip positions are derived from the CentroidalState's
face_centroids, centroid_node_vectors, hinge_node_pairs, and hinge_adj_info.
No separate face_size or arm_width parameter is needed:
  - face corners   = fc[i] + cnv[i, 0:4]
  - hinge p1, p2   = face_verts[fi, lj] and face_verts[fk, ll]  (from hinge_node_pairs)
  - arm_width      = |p2 - p1|  (derived, not a parameter)
  - fold_dir       = normalize(adj_vertex - p1)  (from hinge_adj_info)

The only external physical parameters are:
  fold_length     — hinge strip extent along the face edge [same units as the CS]
  sheet_thickness — full sheet depth in z [same units as the CS]

Scope
-----
Valid for axis-aligned rectangular faces (standard RDQK after deployment).
For general bilinear quad faces (after non-trivial Stage-0 mapping), a
general bilinear mesher is needed.

Public API
----------
  build_mesh_from_centroidal_state(cs, fold_length, sheet_thickness,
                                   n_face=4, n_hinge=2, n_z=2)
  -> (nodes (N,3), hexes (H,8), bc_masks dict)
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Grid helpers
# ─────────────────────────────────────────────────────────────────────────────

def _stitch(breaks: list, n_elems: list) -> np.ndarray:
    """Build a 1-D coordinate array by stitching segments (shared endpoints)."""
    pts = [float(breaks[0])]
    for (a, b), n in zip(zip(breaks[:-1], breaks[1:]), n_elems):
        pts.extend(np.linspace(a, b, n + 1, dtype=np.float64)[1:].tolist())
    return np.array(pts, dtype=np.float64)


def _assign_n_elems_1d(breaks: list, n_face: int, n_hinge: int) -> list:
    """
    Assign element count per 1-D segment.

    Large segments (≥ 50 % of the longest segment) → n_face.
    Short segments (face edge → hinge attachment zones, gap widths) → n_hinge.
    """
    if len(breaks) < 2:
        return []
    lengths = [breaks[i + 1] - breaks[i] for i in range(len(breaks) - 1)]
    threshold = max(lengths) * 0.5
    return [n_face if ln >= threshold else n_hinge for ln in lengths]


# ─────────────────────────────────────────────────────────────────────────────
# Geometry extraction from CentroidalState
# ─────────────────────────────────────────────────────────────────────────────

def _extract_geometry(cs, fold_length: float):
    """
    Extract face corner positions and hinge strip geometry from a CentroidalState.

    Returns
    -------
    face_verts : (n_faces, 4, 2) — absolute corner positions per face
    hinge_strips : list of dicts, one per hinge:
        p1, p2   : hinge vertex positions on each face
        fold_dir : unit vector into the hinge strip (from p1 side)
        arm_width: |p2 - p1|, derived from the CS
    """
    fc  = np.array(cs.face_centroids)           # (n_faces, 2)
    cnv = np.array(cs.centroid_node_vectors)     # (n_faces, max_nodes, 2)
    hnp = np.array(cs.hinge_node_pairs)          # (n_hinge_verts, 2, 2)
    adj = np.array(cs.hinge_adj_info)            # (n_hinges, 5)

    face_verts = fc[:, None, :] + cnv[:, :4, :]  # (n_faces, 4, 2)

    n_hinges = adj.shape[0]
    hinge_strips = []
    for h in range(n_hinges):
        fi, lj = int(hnp[h, 0, 0]), int(hnp[h, 0, 1])
        fk, ll = int(hnp[h, 1, 0]), int(hnp[h, 1, 1])

        p1 = face_verts[fi, lj].copy()  # hinge vertex on face i
        p2 = face_verts[fk, ll].copy()  # hinge vertex on face k

        # Fold direction: from hinge vertex toward adjacent vertex on face i.
        # adj_info columns: [face_i, face_k, pivot_local_i, adj_local_i, adj_local_k]
        adj_li = int(adj[h, 3])
        adj_pt = face_verts[fi, adj_li]
        fold_raw = adj_pt - p1
        fold_norm = float(np.linalg.norm(fold_raw))
        fold_dir = fold_raw / fold_norm if fold_norm > 1e-12 else np.zeros(2, dtype=np.float64)

        hinge_strips.append({
            'h': h, 'fi': fi, 'lj': lj, 'fk': fk, 'll': ll,
            'p1': p1, 'p2': p2,
            'fold_dir': fold_dir,
            'arm_width': float(np.linalg.norm(p2 - p1)),
        })

    return face_verts, hinge_strips


# ─────────────────────────────────────────────────────────────────────────────
# Breakpoint computation
# ─────────────────────────────────────────────────────────────────────────────

def _compute_breakpoints(
    face_verts: np.ndarray,
    hinge_strips: list,
    fold_length: float,
) -> Tuple[list, list]:
    """
    Compute global x and y breakpoints for the structured mesh grid.

    Sources:
      - Face corner x and y coordinates
      - Hinge fold endpoints: p + fold_dir * fold_length  (both p1 and p2 ends)
    """
    x_vals: set = set()
    y_vals: set = set()

    for i in range(len(face_verts)):
        for j in range(4):
            x_vals.add(round(float(face_verts[i, j, 0]), 10))
            y_vals.add(round(float(face_verts[i, j, 1]), 10))

    for hs in hinge_strips:
        for p in (hs['p1'], hs['p2']):
            end = p + hs['fold_dir'] * fold_length
            x_vals.add(round(float(end[0]), 10))
            y_vals.add(round(float(end[1]), 10))

    return sorted(x_vals), sorted(y_vals)


# ─────────────────────────────────────────────────────────────────────────────
# Element classification
# ─────────────────────────────────────────────────────────────────────────────

def _point_in_face_bbox(
    xc: float, yc: float, face_verts: np.ndarray, tol: float = 1e-9
) -> int:
    """
    Return the face index whose bounding box contains (xc, yc), or -1.

    For axis-aligned rectangles the bounding box IS the polygon, so this is
    exact.  Hinge strip interior nodes (outside all face bounding boxes) are
    correctly excluded.
    """
    for i, verts in enumerate(face_verts):
        if (verts[:, 0].min() - tol <= xc <= verts[:, 0].max() + tol and
                verts[:, 1].min() - tol <= yc <= verts[:, 1].max() + tol):
            return i
    return -1


def _point_in_hinge_strip(
    xc: float, yc: float, hinge_strips: list, fold_length: float, tol: float = 1e-9
) -> int:
    """
    Return the hinge index whose strip parallelogram contains (xc, yc), or -1.

    Each strip is parameterised as:
        p(s, t) = p1 + s * arm_hat * arm_width + t * fold_dir * fold_length
    with s ∈ [0, 1], t ∈ [0, 1].

    Works for any parallelogram strip — not restricted to axis-aligned geometry.
    """
    for hs in hinge_strips:
        p1      = hs['p1']
        arm_len = hs['arm_width']
        arm_hat = (hs['p2'] - p1) / arm_len if arm_len > 1e-12 else np.zeros(2)
        fold_hat = hs['fold_dir']

        dp = np.array([xc, yc]) - p1
        s  = float(np.dot(dp, arm_hat))
        t  = float(np.dot(dp, fold_hat))

        if -tol <= s <= arm_len + tol and -tol <= t <= fold_length + tol:
            return hs['h']
    return -1


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def build_mesh_from_centroidal_state(
    cs,
    fold_length: float,
    sheet_thickness: float,
    n_face: int = 4,
    n_hinge: int = 2,
    n_z: int = 2,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Build a conforming SOFA hex mesh directly from a CentroidalState.

    Parameters
    ----------
    cs : CentroidalState
        Stage-1 output in physical units.  Must be the DEPLOYED state
        (non-zero hinge bond vectors) so that face panels are separated by
        visible gaps and the hinge strips fall in those gaps.
    fold_length : float
        Hinge strip extent along the face edge [same units as the CS].
        The one physical design parameter not stored in the CentroidalState.
    sheet_thickness : float
        Full sheet depth in z [same units as the CS].
    n_face : int
        Hex elements per face-panel segment (default 4).
    n_hinge : int
        Hex elements per hinge/gap segment (default 2).
    n_z : int
        Hex layers through thickness — must be ≥ 1 (default 2).

    Returns
    -------
    nodes : (N, 3) float64
        Stress-free node positions.
    hexes : (H, 8) int32
        Hex connectivity, VTK bottom-CCW then top-CCW ordering.
    bc_masks : dict
        ``'face_i'`` and ``'fi'`` (alias) → (N,) bool, nodes in face panel i.
        ``'clamped'`` → union of all constrained faces (from CS BCs).
        ``'loaded'``  → union of all loaded faces (from CS BCs).

    Notes
    -----
    Valid for axis-aligned rectangular faces (standard RDQK in the deployed
    state).  For general bilinear quad faces after Stage-0 mapping, a
    general bilinear mesher is needed.

    The face-bounding-box BC masks correctly exclude hinge strip interior
    nodes: a node at p1 + fold_dir * fold_length / 2 (inside the gap) is
    NOT in any face mask.  Only the shared interface nodes at the face
    boundary are included in the adjacent face's mask.
    """
    # ── 1. Extract geometry from CS ──────────────────────────────────────────
    face_verts, hinge_strips = _extract_geometry(cs, fold_length)
    n_faces = len(face_verts)

    # ── 2. Compute global 1-D breakpoints from actual CS vertex positions ────
    x_breaks, y_breaks = _compute_breakpoints(face_verts, hinge_strips, fold_length)

    # ── 3. Assign element counts per segment ─────────────────────────────────
    nx_elems = _assign_n_elems_1d(x_breaks, n_face, n_hinge)
    ny_elems = _assign_n_elems_1d(y_breaks, n_face, n_hinge)

    # ── 4. Build 1-D coordinate arrays ───────────────────────────────────────
    xs = _stitch(x_breaks, nx_elems)
    ys = _stitch(y_breaks, ny_elems)
    zs = np.linspace(-sheet_thickness / 2.0, sheet_thickness / 2.0, n_z + 1,
                     dtype=np.float64)

    nx, ny, nz = len(xs), len(ys), len(zs)

    def nidx(ix: int, iy: int, iz: int) -> int:
        return iz * nx * ny + iy * nx + ix

    # ── 5. Build dense node array (full grid, before compaction) ─────────────
    all_pos = np.empty((nx * ny * nz, 3), dtype=np.float64)
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                all_pos[nidx(ix, iy, iz)] = [xs[ix], ys[iy], zs[iz]]

    # ── 6. Collect active hexes (face panels + hinge strips, skip voids) ─────
    hexes_raw: list = []
    for iz in range(nz - 1):
        for iy in range(ny - 1):
            for ix in range(nx - 1):
                xc = (xs[ix] + xs[ix + 1]) * 0.5
                yc = (ys[iy] + ys[iy + 1]) * 0.5
                fi = _point_in_face_bbox(xc, yc, face_verts)
                active = fi >= 0
                if not active:
                    active = _point_in_hinge_strip(xc, yc, hinge_strips, fold_length) >= 0
                if active:
                    hexes_raw.append([
                        nidx(ix,   iy,   iz  ), nidx(ix+1, iy,   iz  ),
                        nidx(ix+1, iy+1, iz  ), nidx(ix,   iy+1, iz  ),
                        nidx(ix,   iy,   iz+1), nidx(ix+1, iy,   iz+1),
                        nidx(ix+1, iy+1, iz+1), nidx(ix,   iy+1, iz+1),
                    ])

    hexes_raw_arr = np.array(hexes_raw, dtype=np.int32)

    # ── 7. Compact: keep only nodes referenced by active hexes ───────────────
    used  = np.unique(hexes_raw_arr.ravel())
    remap = np.full(nx * ny * nz, -1, dtype=np.int32)
    remap[used] = np.arange(len(used), dtype=np.int32)

    nodes = all_pos[used]
    hexes = remap[hexes_raw_arr]

    # ── 8. BC masks from actual face polygon bounding boxes ──────────────────
    nodes_xy = nodes[:, :2]
    tol = 1e-9

    bc_masks: Dict[str, np.ndarray] = {}
    for i in range(n_faces):
        verts = face_verts[i]
        x_min = float(verts[:, 0].min()) - tol
        x_max = float(verts[:, 0].max()) + tol
        y_min = float(verts[:, 1].min()) - tol
        y_max = float(verts[:, 1].max()) + tol
        mask = ((nodes_xy[:, 0] >= x_min) & (nodes_xy[:, 0] <= x_max) &
                (nodes_xy[:, 1] >= y_min) & (nodes_xy[:, 1] <= y_max))
        bc_masks[f'face_{i}'] = mask
        bc_masks[f'f{i}'] = mask  # backward-compat alias for scene_builder.py

    # ── 9. Clamped / loaded masks from CentroidalState BCs ───────────────────
    constrained = np.array(cs.constrained_face_DOF_pairs)
    loaded_dofs  = np.array(cs.loaded_face_DOF_pairs)
    n_nodes = len(nodes)

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
