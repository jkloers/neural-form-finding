"""
Initial mapping in centroidal coordinates.

Maps a flat tessellation (c, s) into a target shape by defining a continuous
mapping function f: R^2 -> R^2 and applying it strictly to face centroids.
The face shapes (centroid_node_vectors) are then transformed using the Jacobian
matrix J_f evaluated at each centroid, resulting in a continuous, differentiable
Rigid-Face mapping that prevents internal criss-crossing for ANY arbitrary mapping.

This module is designed to be replaced by a GNN in the future.
The interface is: CentroidalState → CentroidalState, pure JAX, differentiable.
"""
import jax
import jax.numpy as jnp
from jax_backend.state import CentroidalState
from jax_backend.geometry import reconstruct_vertices
from problem.targets import get_target_points

def map_elliptical_grip(p, box_center, half_sizes, center, radius, scale_factor):
    h_sizes = jnp.where(half_sizes == 0.0, 1.0, half_sizes)
    normalized = (p - box_center) / h_sizes
    u, v = normalized[0], normalized[1]
    x_disk = u * jnp.sqrt(jnp.maximum(0.0, 1.0 - (v ** 2) / 2.0))
    y_disk = v * jnp.sqrt(jnp.maximum(0.0, 1.0 - (u ** 2) / 2.0))
    mapped = jnp.array([x_disk, y_disk]) * radius + center
    return center + (mapped - center) * scale_factor

def map_conformal_polynomial(p, box_center, half_size, domain_restriction, map_params, center, radius, scale_factor):
    normalized_p = (p - box_center) / half_size
    z = (normalized_p[0] + 1j * normalized_p[1]) * domain_restriction
    
    if map_params is None or map_params.shape[0] < 4:
        prms = jnp.concatenate([jnp.array([0.0, 0.0, 0.0, 1.0]), jnp.zeros(1)])
    else:
        prms = map_params
    
    tx, ty, theta, s_val = prms[0], prms[1], prms[2], prms[3]
    c_val = prms[4:]
    
    w = z
    for k in range(c_val.shape[0]):
        power = 4 * (k + 1) + 1
        w = w + c_val[k] * (z ** power)
        
    w = s_val * w * jnp.exp(1j * theta)
    x_new = jnp.real(w) * radius + center[0] + tx
    y_new = jnp.imag(w) * radius + center[1] + ty
    mapped = jnp.array([x_new, y_new])
    return center + (mapped - center) * scale_factor

def map_boundary_projection(p, box_center, half_sizes, shape_center, b_angles, b_radii, scale_factor):
    offset = p - box_center
    p_angle = jnp.arctan2(offset[1], offset[0])
    p_norm = jnp.linalg.norm(offset)
    
    w_box = half_sizes[0]
    h_box = half_sizes[1]
    norm_max_tess = 1.0 / jnp.maximum(
        jnp.abs(jnp.cos(p_angle)) / w_box,
        jnp.abs(jnp.sin(p_angle)) / h_box)
        
    target_boundary_radius = jnp.interp(p_angle, b_angles, b_radii)
    scale_rad = jnp.where(p_norm > 0, target_boundary_radius / norm_max_tess, 0.0)
    
    mapped = shape_center + offset * scale_rad
    return shape_center + (mapped - shape_center) * scale_factor

def apply_initial_map(
        state: CentroidalState,
        target_params: dict,
        map_type: str = 'elliptical_grip',
        scale_factor: float = 1.0,
        map_params: jnp.ndarray = None,
        domain_restriction: float = 0.8) -> CentroidalState:
    """Apply an initial mapping to a CentroidalState using Rigid-Face generalized mapping.

    Args:
        state: CentroidalState with flat tessellation geometry.
        target_params: dict with 'type', 'center', 'radius'
        map_type: 'elliptical_grip', 'boundary_projection' or 'conformal_polynomial'.
        scale_factor: scaling applied after mapping.
        map_params: parameters for the parameterized map (e.g. polynomial coefficients).
        domain_restriction: limits evaluation to a fraction of the domain to avoid singularities.

    Returns:
        CentroidalState with updated (face_centroids, centroid_node_vectors).
    """
    c = state.face_centroids
    s = state.centroid_node_vectors
    n_faces, max_nodes, dim = s.shape

    # 1. Gather global context for the mapping (bounding box)
    all_vertices = reconstruct_vertices(c, s)
    vertices_flat = all_vertices.reshape(-1, dim)
    
    min_xy = jnp.min(vertices_flat, axis=0)
    max_xy = jnp.max(vertices_flat, axis=0)
    box_center = (min_xy + max_xy) / 2.0
    half_sizes = (max_xy - min_xy) / 2.0
    half_size = jnp.maximum(jnp.max(half_sizes), 1e-6)

    shape_type = target_params.get('type', 'circle')
    center = jnp.asarray(target_params.get('center', [0.0, 0.0]), dtype=float)
    radius = float(target_params.get('radius', 1.0))

    # Pre-computation for boundary_projection
    if map_type == 'boundary_projection' or shape_type != 'circle':
        boundary_pts = jnp.asarray(get_target_points(target_params, n_points=500), dtype=float)
        shape_center = jnp.mean(boundary_pts, axis=0)
        boundary_vec = boundary_pts - shape_center
        boundary_angles = jnp.arctan2(boundary_vec[:, 1], boundary_vec[:, 0])
        boundary_radii = jnp.linalg.norm(boundary_vec, axis=1)

        order = jnp.argsort(boundary_angles)
        b_angles = boundary_angles[order]
        b_radii = boundary_radii[order]
        b_angles = jnp.concatenate([b_angles - 2 * jnp.pi, b_angles, b_angles + 2 * jnp.pi])
        b_radii = jnp.tile(b_radii, 3)
    else:
        shape_center = center
        b_angles = jnp.zeros(3)
        b_radii = jnp.zeros(3)

    # 2. Define the generic point mapping function f: R^2 -> R^2
    def f_point(p):
        if map_type == 'elliptical_grip' and shape_type == 'circle':
            return map_elliptical_grip(p, box_center, half_sizes, center, radius, scale_factor)
            
        elif map_type == 'conformal_polynomial' and shape_type == 'circle':
            return map_conformal_polynomial(p, box_center, half_size, domain_restriction, map_params, center, radius, scale_factor)
            
        elif map_type == 'boundary_projection' or shape_type != 'circle':
            return map_boundary_projection(p, box_center, half_sizes, shape_center, b_angles, b_radii, scale_factor)
            
        else:
            return p * scale_factor

    # 3. Compute Jacobian matrix function using JAX
    jac_f = jax.jacfwd(f_point)
    
    # 4. Vectorize across all centroids
    f_vmap = jax.vmap(f_point)
    jac_vmap = jax.vmap(jac_f)
    
    # 5. Map centroids
    c_new = f_vmap(c)
    
    # 6. Transform CNVs using the Jacobian
    # jac_matrices shape: (n_faces, 2, 2)
    # s shape: (n_faces, max_nodes, 2)
    jac_matrices = jac_vmap(c)
    
    # s_new[i, j, :] = jac_matrices[i, :, :] @ s[i, j, :]
    s_new = jnp.einsum('fab,fnb->fna', jac_matrices, s)

    return state._replace(
        face_centroids=c_new,
        centroid_node_vectors=s_new,
    )
