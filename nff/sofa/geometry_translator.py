"""
nff/sofa/geometry_translator.py
================================
Convert a CentroidalState (Stage-1 output) to a SOFA hex mesh + BC masks.

The mesh is built directly from the CentroidalState's vertex positions and
hinge connectivity — NOT from a parametric face_size / arm_width formula.
See nff/sofa/mesh_builder.py for the implementation.

Usage
-----
>>> from nff.sofa.geometry_translator import build_sofa_scene_from_centroidal_state
>>> nodes, hexes, bc = build_sofa_scene_from_centroidal_state(
...     cs,
...     fold_length    = 0.003,    # m — only external physical parameter
...     sheet_thickness= 0.001,    # m
... )
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from nff.sofa.mesh_builder import build_mesh_from_centroidal_state


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_sofa_scene_from_centroidal_state(
    cs,
    fold_length: float     = 0.003,
    sheet_thickness: float = 0.001,
    n_face: int            = 4,
    n_hinge: int           = 2,
    n_z: int               = 2,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Convert a deployed CentroidalState to a conforming SOFA hex mesh.

    All face panel geometry (corners) and hinge strip geometry (p1, p2,
    arm_width, fold_dir) are derived directly from the CentroidalState's
    face_centroids, centroid_node_vectors, hinge_node_pairs, and
    hinge_adj_info.

    The CentroidalState MUST be the DEPLOYED state (non-zero hinge bond
    vectors, face panels separated by arm_width gaps).  A flat/closed
    tessellation has zero bond vectors and produces a degenerate mesh.

    Parameters
    ----------
    cs : CentroidalState
        Stage-1 output in physical units (metres).
    fold_length : float
        Hinge strip extent along the face edge [m].  The one physical
        design parameter not stored in the CentroidalState.
    sheet_thickness : float
        Full sheet depth in z [m].
    n_face, n_hinge, n_z : mesh resolution.

    Returns
    -------
    nodes : (N, 3) float64
    hexes : (H, 8) int32
    bc_masks : dict
        ``'face_i'`` and ``'fi'`` : nodes in face panel i.
        ``'clamped'`` : constrained faces (from CS BCs).
        ``'loaded'``  : loaded faces (from CS BCs).
    """
    return build_mesh_from_centroidal_state(
        cs,
        fold_length     = fold_length,
        sheet_thickness = sheet_thickness,
        n_face          = n_face,
        n_hinge         = n_hinge,
        n_z             = n_z,
    )


# ---------------------------------------------------------------------------
# Convenience: build a deployed RDQK CentroidalState and translate
# ---------------------------------------------------------------------------

def translate_rdqk_unit_cell(
    face_size: float       = 0.100,
    arm_width: float       = 0.010,
    fold_length: float     = 0.003,
    sheet_thickness: float = 0.001,
    clamp_face: int        = 0,
    load_face: Optional[int] = 1,
    load_dof: int          = 2,
    load_value: float      = 1.0,
    n_face: int            = 4,
    n_hinge: int           = 2,
    n_z: int               = 2,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Build a DEPLOYED 1×1 unit_RDQK_0 CentroidalState and translate it to a
    SOFA hex mesh.

    Deployed state: each face is shifted by arm_width in the appropriate
    direction so that hinge bond vectors are non-zero (≈ arm_width in
    length).  This matches the deployed starting state used in JAX Stage-2.

    Parameters
    ----------
    face_size : float
        Square face panel side [m].
    arm_width : float
        Gap between adjacent face panels [m].  Applied as per-face shifts.
    fold_length : float
        Hinge strip corner extent [m].
    sheet_thickness : float
        Sheet depth in z [m].
    clamp_face : int
        Face index to clamp (all DOFs fixed).
    load_face : int or None
        Face index to apply a load.
    load_dof : int
        DOF to load (0=x, 1=y, 2=θ).
    load_value : float
        Load magnitude.

    Returns
    -------
    nodes, hexes, bc_masks — same as build_sofa_scene_from_centroidal_state.
    """
    import pathlib
    import yaml

    from nff.topology.core import UnitPattern
    from nff.topology.builder import build_tessellation
    from nff.stages.state import CentroidalState

    patterns_path = (pathlib.Path(__file__).parent.parent.parent
                     / 'data' / 'library' / 'patterns.yaml')
    with open(patterns_path) as fh:
        lib = yaml.safe_load(fh)
    p_cfg = lib['unit_RDQK_0']

    # Scale to physical face_size and apply per-face ARM_WIDTH deployment shifts.
    # F0 → no shift, F1 → (+arm, 0), F2 → (+arm, +arm), F3 → (0, +arm).
    verts = np.array(p_cfg['vertices'], dtype=np.float64) * face_size
    face_vertex_ranges = {0: range(0, 4), 1: range(4, 8),
                          2: range(8, 12),  3: range(12, 16)}
    face_shifts = {
        0: np.array([0.0,      0.0      ]),
        1: np.array([arm_width, 0.0     ]),
        2: np.array([arm_width, arm_width]),
        3: np.array([0.0,      arm_width]),
    }
    for fi, vi_range in face_vertex_ranges.items():
        verts[list(vi_range)] += face_shifts[fi]

    pattern = UnitPattern(
        vertices        = verts,
        faces           = p_cfg['faces'],
        internal_hinges = p_cfg['internal_hinges'],
        external_hinges = p_cfg.get('external_hinges', []),
    )

    tess = build_tessellation(pattern, nx=1, ny=1)
    tess.set_face_dofs(clamp_face, [0, 1, 2])
    if load_face is not None:
        tess.set_face_load(load_face, load_dof, load_value)

    cs = CentroidalState.from_tessellation(tess)

    return build_sofa_scene_from_centroidal_state(
        cs,
        fold_length     = fold_length,
        sheet_thickness = sheet_thickness,
        n_face          = n_face,
        n_hinge         = n_hinge,
        n_z             = n_z,
    )
