"""Deployed-geometry helpers for the closed-state pipeline.

Small pure-numpy reductions from a (valid_state, displacement) pair to the
geometric quantities the closed driver and its diagnostics both need: the
deployed boundary cloud, an algebraic best-fit circle, the flattened global
vertex array, and the deployed hinge midpoints. Extracted from ``run_closed`` so
diagnostics can import them without reaching into a CLI script.
"""

import numpy as np


def _fit_circle(cloud):
    """Algebraic (Kåsa) best-fit circle of a point cloud -> (center, radius)."""
    n = cloud.shape[0]
    A = np.concatenate([2.0 * cloud, np.ones((n, 1))], axis=1)
    cx, cy, c = np.linalg.solve(A.T @ A + 1e-8 * np.eye(3), A.T @ np.sum(cloud ** 2, axis=1))
    return np.array([cx, cy]), float(np.sqrt(max(c + cx * cx + cy * cy, 1e-12)))


def _boundary_cloud(vs, disp):
    """Deployed boundary-vertex positions from a valid state + displacement field."""
    b = np.asarray(vs.boundary_face_node_ids)
    fc = np.asarray(vs.face_centroids) + np.asarray(disp[:, :2])
    th = np.asarray(disp[:, 2])
    cnv = np.asarray(vs.centroid_node_vectors)
    bf, bl = b[:, 0], b[:, 1]
    vec = cnv[bf, bl]
    ct, st = np.cos(th[bf]), np.sin(th[bf])
    return fc[bf] + np.stack([ct * vec[:, 0] - st * vec[:, 1], st * vec[:, 0] + ct * vec[:, 1]], axis=-1)


def _global_verts(tess, node_positions):
    """Flatten per-face node positions into the global vertex array of ``tess``."""
    v = np.array(tess.vertices, dtype=float)
    for f_id, face in enumerate(tess.faces):
        for local, gv in enumerate(face.vertex_indices):
            v[gv] = node_positions[f_id, local]
    return v


def _deployed_hinge_xy(vs, disp):
    """(n_hinges, 2) deployed hinge midpoints, in bond_connectivity order (= per-hinge D order).

    Each bond connects two NODES; the deployed node positions come from the rigid-tile kinematics
    (face centroid + rotated centroid->node vector + displacement), reshaped to the flat node index
    space that ``bond_connectivity`` indexes.
    """
    fc = np.asarray(vs.face_centroids) + np.asarray(disp[:, :2])
    th = np.asarray(disp[:, 2]); ct, st = np.cos(th), np.sin(th)
    cnv = np.asarray(vs.centroid_node_vectors)                      # (nf, nn, 2)
    rx = ct[:, None] * cnv[:, :, 0] - st[:, None] * cnv[:, :, 1]
    ry = st[:, None] * cnv[:, :, 0] + ct[:, None] * cnv[:, :, 1]
    node_xy = (fc[:, None, :] + np.stack([rx, ry], -1)).reshape(-1, 2)   # (nf*nn, 2)
    bc = np.asarray(vs.bond_connectivity)
    return 0.5 * (node_xy[bc[:, 0]] + node_xy[bc[:, 1]])
