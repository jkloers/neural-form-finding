"""
geometry.py — Unified hex mesh for the 1×1 unit_RDQK_0 kirigami unit cell.

Physical layout (face_size=a, arm_width=w, fold_length=L):

    F3 ─ H3 ─ (gap) ── F0
    │              │
    H2   [void]   H0
    │              │
    F2 ─ H1 ─ (gap) ── F1

Hinges are short strips at specific corners (from patterns.yaml vertex pairs):
  H0 (F0↔F1): lower vertical gap, bottom corner  → x∈[a,a+w],   y∈[0,L]
  H1 (F1↔F2): right horizontal gap, left corner  → x∈[a+w,a+w+L], y∈[a,a+w]
  H2 (F2↔F3): upper vertical gap, top corner     → x∈[a,a+w],   y∈[2a+w-L,2a+w]
  H3 (F3↔F0): left horizontal gap, right corner  → x∈[a-L,a],   y∈[a,a+w]

All regions share the same material.  Face panels are large (a×a) so they behave
quasi-rigidly; hinges are thin (fold_length L << a) so deformation concentrates there.
"""

import numpy as np
from typing import Dict, Tuple


def _stitch(breaks: list, n_elems: list) -> np.ndarray:
    """Build 1D coordinate array by stitching segments (shared endpoints)."""
    pts = [breaks[0]]
    for (a, b), n in zip(zip(breaks[:-1], breaks[1:]), n_elems):
        pts.extend(np.linspace(a, b, n + 1)[1:])
    return np.array(pts, dtype=np.float64)


def _classify(xc: float, yc: float, a: float, w: float, L: float) -> str:
    """Return region label for an element centered at (xc, yc), or 'void'."""
    # Face panels (large blocks)
    if 0 <= xc <= a     and 0 <= yc <= a:      return 'F0'
    if a+w <= xc        and 0 <= yc <= a:      return 'F1'
    if a+w <= xc        and a+w <= yc:         return 'F2'
    if 0 <= xc <= a     and a+w <= yc:         return 'F3'
    # Hinge strips (small, same material)
    if a <= xc <= a+w   and 0 <= yc <= L:      return 'H0'
    if a+w <= xc <= a+w+L and a <= yc <= a+w: return 'H1'
    if a <= xc <= a+w   and 2*a+w-L <= yc:    return 'H2'
    if a-L <= xc <= a   and a <= yc <= a+w:   return 'H3'
    return 'void'


def build_unified_mesh(
    face_size:       float,
    arm_width:       float,
    fold_length:     float,
    sheet_thickness: float,
    n_face:  int = 4,
    n_hinge: int = 2,
    n_z:     int = 2,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Build the full unified hexahedral mesh for unit_RDQK_0, 1×1 cell.

    The mesh covers 4 face panels + 4 hinge strips in one MechanicalObject.
    Kirigami cuts are voids (no elements). Face/hinge boundary nodes are shared.

    Parameters
    ----------
    face_size, arm_width, fold_length, sheet_thickness : floats [m]
    n_face  : hex elements per face segment in x or y (coarse: 4)
    n_hinge : hex elements per hinge/boundary segment (fine: 2)
    n_z     : hex layers through thickness (2 minimum for bending)

    Returns
    -------
    nodes    : (N, 3) float64 — stress-free node positions [m]
    hexes    : (H, 8) int32  — VTK/SOFA hex connectivity
    bc_masks : dict 'f0'..'f3' → (N,) bool — face-panel node masks
    """
    a, w, L, t = face_size, arm_width, fold_length, sheet_thickness

    xs = _stitch([0, a-L, a, a+w, a+w+L, 2*a+w], [n_face, n_hinge, n_hinge, n_hinge, n_face])
    ys = _stitch([0, L,   a, a+w, 2*a+w-L, 2*a+w], [n_hinge, n_face, n_hinge, n_face, n_hinge])
    zs = _stitch([-t/2, t/2], [n_z])

    nx, ny, nz = len(xs), len(ys), len(zs)

    def nidx(ix, iy, iz):
        return iz * nx * ny + iy * nx + ix

    # Build all potential nodes in one flat array
    all_pos = np.empty((nx * ny * nz, 3), dtype=np.float64)
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                all_pos[nidx(ix, iy, iz)] = [xs[ix], ys[iy], zs[iz]]

    # Collect active elements (non-void)
    hexes_raw = []
    for iz in range(nz - 1):
        for iy in range(ny - 1):
            for ix in range(nx - 1):
                xc = (xs[ix] + xs[ix+1]) * 0.5
                yc = (ys[iy] + ys[iy+1]) * 0.5
                if _classify(xc, yc, a, w, L) != 'void':
                    hexes_raw.append([
                        nidx(ix,   iy,   iz  ), nidx(ix+1, iy,   iz  ),
                        nidx(ix+1, iy+1, iz  ), nidx(ix,   iy+1, iz  ),
                        nidx(ix,   iy,   iz+1), nidx(ix+1, iy,   iz+1),
                        nidx(ix+1, iy+1, iz+1), nidx(ix,   iy+1, iz+1),
                    ])

    hexes_raw = np.array(hexes_raw, dtype=np.int32)

    # Compact: keep only referenced nodes
    used = np.unique(hexes_raw.ravel())
    remap = np.full(nx * ny * nz, -1, dtype=np.int32)
    remap[used] = np.arange(len(used), dtype=np.int32)

    nodes = all_pos[used]
    hexes = remap[hexes_raw]

    # BC masks: closed-interval check in (x, y)
    eps = 1e-9
    x, y = nodes[:, 0], nodes[:, 1]
    bc_masks = {
        'f0': (x >= -eps) & (x <= a+eps)     & (y >= -eps) & (y <= a+eps),
        'f1': (x >= a+w-eps)                 & (y >= -eps) & (y <= a+eps),
        'f2': (x >= a+w-eps)                 & (y >= a+w-eps),
        'f3': (x >= -eps) & (x <= a+eps)     & (y >= a+w-eps),
    }

    return nodes, hexes, bc_masks
