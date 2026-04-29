"""
Initial mapping in centroidal coordinates.

Maps a flat tessellation (c, s) into a target shape by deforming the
reconstructed vertices and then decomposing back into (c_new, s_new).

This module is designed to be replaced by a GNN in the future.
The interface is: CentroidalState → CentroidalState, pure JAX, differentiable.
"""

import jax.numpy as jnp

from jax_backend.centroidal.state import CentroidalState
from jax_backend.centroidal.geometry import reconstruct_vertices

# Need boundary points — import here to avoid circular deps
from problem.targets import get_target_points


# ─────────────────────────────────────────────────────────────────────────────
# Low-level mapping functions (operate on raw vertex arrays)
# ─────────────────────────────────────────────────────────────────────────────

def _elliptical_grip_mapping(vertices_flat, center, radius):
    """Rectangle-to-ellipse conformal-like mapping.

    Args:
        vertices_flat: (N, 2) — flat vertex positions.
        center: (2,) — target center.
        radius: float — target radius.

    Returns:
        (N, 2) — mapped vertex positions.
    """
    center = jnp.asarray(center, dtype=float)
    min_xy = jnp.min(vertices_flat, axis=0)
    max_xy = jnp.max(vertices_flat, axis=0)
    box_center = (min_xy + max_xy) / 2.0
    half_sizes = (max_xy - min_xy) / 2.0
    half_sizes = jnp.where(half_sizes == 0.0, 1.0, half_sizes)

    normalized = (vertices_flat - box_center) / half_sizes
    u = normalized[:, 0]
    v = normalized[:, 1]

    x_disk = u * jnp.sqrt(jnp.maximum(0.0, 1.0 - (v ** 2) / 2.0))
    y_disk = v * jnp.sqrt(jnp.maximum(0.0, 1.0 - (u ** 2) / 2.0))

    return jnp.column_stack((x_disk, y_disk)) * radius + center


def _boundary_projection_mapping(vertices_flat, boundary_points):
    """Proportional warping to fill an arbitrary target shape.

    Args:
        vertices_flat: (N, 2) — flat vertex positions.
        boundary_points: (M, 2) — target boundary point cloud.

    Returns:
        (N, 2) — mapped vertex positions.
    """
    boundary = jnp.asarray(boundary_points, dtype=float)
    shape_center = jnp.mean(boundary, axis=0)
    boundary_vec = boundary - shape_center
    boundary_angles = jnp.arctan2(boundary_vec[:, 1], boundary_vec[:, 0])
    boundary_radii = jnp.linalg.norm(boundary_vec, axis=1)

    order = jnp.argsort(boundary_angles)
    b_angles = boundary_angles[order]
    b_radii = boundary_radii[order]
    b_angles = jnp.concatenate([b_angles - 2 * jnp.pi, b_angles, b_angles + 2 * jnp.pi])
    b_radii = jnp.tile(b_radii, 3)

    points = jnp.asarray(vertices_flat, dtype=float)
    p_min, p_max = jnp.min(points, axis=0), jnp.max(points, axis=0)
    p_center = (p_min + p_max) / 2.0

    offsets = points - p_center
    p_angles = jnp.arctan2(offsets[:, 1], offsets[:, 0])
    p_norms = jnp.linalg.norm(offsets, axis=1)

    w = (p_max[0] - p_min[0]) / 2.0
    h = (p_max[1] - p_min[1]) / 2.0
    norm_max_tess = 1.0 / jnp.maximum(
        jnp.abs(jnp.cos(p_angles)) / w,
        jnp.abs(jnp.sin(p_angles)) / h)

    target_boundary_radii = jnp.interp(p_angles, b_angles, b_radii)
    scale = jnp.where(p_norms > 0, target_boundary_radii / norm_max_tess, 0.0)

    return shape_center + offsets * scale[:, None]


# ─────────────────────────────────────────────────────────────────────────────
# Decomposition: mapped vertices → (c_new, s_new)
# ─────────────────────────────────────────────────────────────────────────────

def _vertices_to_centroidal(mapped_vertices, n_faces, max_nodes):
    """Decompose mapped vertex positions back into (face_centroids, cnv).

    Args:
        mapped_vertices: (n_faces, max_nodes, 2) — mapped absolute vertex positions.
        n_faces: int
        max_nodes: int

    Returns:
        face_centroids: (n_faces, 2)
        centroid_node_vectors: (n_faces, max_nodes, 2)
    """
    face_centroids = jnp.mean(mapped_vertices, axis=1)           # (n_faces, 2)
    cnv = mapped_vertices - face_centroids[:, None, :]            # (n_faces, max_nodes, 2)
    return face_centroids, cnv


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def apply_initial_map(
        state: CentroidalState,
        target_params: dict,
        map_type: str = 'elliptical_grip',
        scale_factor: float = 1.0) -> CentroidalState:
    """Apply an initial mapping to a CentroidalState.

    Deforms the flat tessellation into the target shape by:
    1. Reconstructing vertices from (c, s)
    2. Applying a spatial mapping to all vertices
    3. Decomposing mapped vertices back into (c_new, s_new)

    This function is designed to be a drop-in replacement for a GNN:
    both take a CentroidalState and return a CentroidalState.

    Args:
        state: CentroidalState with flat tessellation geometry.
        target_params: dict with 'type', 'center', 'radius' (and optionally
                       a precomputed 'boundary_points' array).
        map_type: 'elliptical_grip' or 'boundary_projection'.
        scale_factor: scaling applied after mapping.

    Returns:
        CentroidalState with updated (face_centroids, centroid_node_vectors).
    """
    c = state.face_centroids
    s = state.centroid_node_vectors
    n_faces, max_nodes, dim = s.shape

    # 1. Reconstruct all vertices: (n_faces, max_nodes, 2)
    all_vertices = reconstruct_vertices(c, s)

    # 2. Flatten to (n_faces * max_nodes, 2) for the mapping
    vertices_flat = all_vertices.reshape(-1, dim)

    # 3. Apply spatial mapping
    shape_type = target_params.get('type', 'circle')
    center = jnp.asarray(target_params.get('center', [0.0, 0.0]), dtype=float)
    radius = float(target_params.get('radius', 1.0))

    if map_type == 'elliptical_grip' and shape_type == 'circle':
        mapped_flat = _elliptical_grip_mapping(vertices_flat, center, radius)
        mapped_flat = center + (mapped_flat - center) * scale_factor
    elif map_type == 'boundary_projection' or shape_type != 'circle':
        boundary_pts = get_target_points(target_params, n_points=500)
        mapped_flat = _boundary_projection_mapping(vertices_flat, boundary_pts)
        center_bp = jnp.mean(jnp.asarray(boundary_pts), axis=0)
        mapped_flat = center_bp + (mapped_flat - center_bp) * scale_factor
    else:
        # Fallback: simple scaling
        mapped_flat = vertices_flat * scale_factor

    # 4. Reshape back to (n_faces, max_nodes, 2)
    mapped_vertices = mapped_flat.reshape(n_faces, max_nodes, dim)

    # 5. Decompose into (c_new, s_new)
    c_new, s_new = _vertices_to_centroidal(mapped_vertices, n_faces, max_nodes)

    return state._replace(
        face_centroids=c_new,
        centroid_node_vectors=s_new,
    )
