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
from typing import Callable, Any
from jax_backend.state import CentroidalState
from jax_backend.geometry import reconstruct_vertices
from problem.targets import get_target_points

def map_elliptical_grip(p_restricted, params, context):
    h_sizes = jnp.where(context['half_sizes'] == 0.0, 1.0, context['half_sizes'])
    normalized = (p_restricted - context['box_center']) / h_sizes
    u, v = normalized[0], normalized[1]
    x_disk = u * jnp.sqrt(jnp.maximum(0.0, 1.0 - (v ** 2) / 2.0))
    y_disk = v * jnp.sqrt(jnp.maximum(0.0, 1.0 - (u ** 2) / 2.0))
    mapped = jnp.array([x_disk, y_disk]) * context['radius'] + context['center']
    return mapped

def map_homothetic(p_restricted, params, context):
    offset = p_restricted - context['box_center']
    return context['center'] + offset

def map_conformal_polynomial(p_restricted, params, context):
    # 1. Map square to unit disk (Shirley-Chiu mapping)
    # This ensures the domain is contained within the target circle.
    h_sizes = jnp.where(context['half_sizes'] == 0.0, 1.0, context['half_sizes'])
    normalized = (p_restricted - context['box_center']) / h_sizes
    u, v = normalized[0], normalized[1]
    
    x_disk = u * jnp.sqrt(jnp.maximum(0.0, 1.0 - (v ** 2) / 2.0))
    y_disk = v * jnp.sqrt(jnp.maximum(0.0, 1.0 - (u ** 2) / 2.0))
    
    z = x_disk + 1j * y_disk
    # Support both dictionary (new) and array (legacy) map_params
    if isinstance(params, dict):
        tx = params.get('tx', 0.0)
        ty = params.get('ty', 0.0)
        theta = params.get('theta', 0.0)
        s_val = params.get('s_val', 1.0)
        c_val = params.get('c_val', jnp.zeros(1))
    else:
        # Fallback to legacy array format
        if params is None or params.shape[0] < 4:
            prms = jnp.concatenate([jnp.array([0.0, 0.0, 0.0, 1.0]), jnp.zeros(1)])
        else:
            prms = params
        tx, ty, theta, s_val = prms[0], prms[1], prms[2], prms[3]
        c_val = prms[4:]
    
    w = z
    for k in range(c_val.shape[0]):
        power = 4 * (k + 1) + 1
        w = w + c_val[k] * (z ** power)
        
    # Temporarily disable rotation to prevent the optimizer from 'cheating'
    # by aligning forces with the strongest axis.
    theta = 0.0
    w = s_val * w * jnp.exp(1j * theta)
    x_new = jnp.real(w) * context['radius'] + context['center'][0] + tx
    y_new = jnp.imag(w) * context['radius'] + context['center'][1] + ty
    return jnp.array([x_new, y_new])

def map_boundary_projection(p_restricted, params, context):
    offset = p_restricted - context['box_center']
    p_angle = jnp.arctan2(offset[1], offset[0])
    p_norm = jnp.linalg.norm(offset)
    
    w_box = context['half_sizes'][0]
    h_box = context['half_sizes'][1]
    norm_max_tess = 1.0 / jnp.maximum(
        jnp.abs(jnp.cos(p_angle)) / w_box,
        jnp.abs(jnp.sin(p_angle)) / h_box)
        
    target_boundary_radius = jnp.interp(p_angle, context['b_angles'], context['b_radii'])
    scale_rad = jnp.where(p_norm > 0, target_boundary_radius / norm_max_tess, 0.0)
    
    mapped = context['shape_center'] + offset * scale_rad
    return mapped

def build_mapping_fn(
        state: CentroidalState,
        target_params: dict,
        map_type: str = 'elliptical_grip',
        scale_factor: float = 1.0,
        domain_restriction: float = 0.8) -> Callable:
    """Factory function to build a pure JAX mapping function.

    Args:
        state: CentroidalState with flat tessellation geometry.
        target_params: dict with 'type', 'center', 'radius'
        map_type: 'elliptical_grip', 'boundary_projection' or 'conformal_polynomial'.
        scale_factor: scaling applied after mapping.
        domain_restriction: limits evaluation to a fraction of the domain to avoid singularities.

    Returns:
        mapping_fn: A callable `f(p, map_params)` that maps R^2 -> R^2.
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
    max_half_size = jnp.maximum(jnp.max(half_sizes), 1e-6)

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

    # 2. Context dictionary passed to pure maps
    context = {
        'box_center': box_center,
        'half_sizes': half_sizes,
        'max_half_size': max_half_size,
        'center': center,
        'radius': radius,
        'b_angles': b_angles,
        'b_radii': b_radii,
        'shape_center': shape_center
    }

    # 3. Select the core mapping function
    if map_type == 'elliptical_grip':
        core_map = map_elliptical_grip
    elif map_type == 'conformal_polynomial':
        core_map = map_conformal_polynomial
    elif map_type == 'boundary_projection':
        core_map = map_boundary_projection
    elif map_type == 'homothetic':
        core_map = map_homothetic
    else:
        print(f"WARNING: Unknown map_type '{map_type}'. Falling back to Identity mapping.")
        core_map = lambda p, params, ctx: p

    # 4. Create the generic wrapper
    def mapping_fn(p, map_params=None):
        if map_params is None:
            map_params = {}
            
        # Universal Pre-processing: Domain restriction
        p_restricted = context['box_center'] + (p - context['box_center']) * domain_restriction
        
        # Apply the chosen mathematical core
        mapped_p = core_map(p_restricted, map_params, context)
        
        # Universal Post-processing: Rescale global around target center
        return context['center'] + (mapped_p - context['center']) * scale_factor

    return mapping_fn

def apply_mapping(
        state: CentroidalState, 
        mapping_fn: Callable, 
        map_params: Any = None) -> CentroidalState:
    """Apply a generic mapping function to a CentroidalState using Rigid-Face generalized mapping.
    
    Args:
        state: CentroidalState with flat tessellation geometry.
        mapping_fn: callable f(p, params) that maps a point R^2 -> R^2.
        map_params: parameters for the parameterized map (e.g. polynomial coefficients).
        
    Returns:
        CentroidalState with updated (face_centroids, centroid_node_vectors).
    """
    c = state.face_centroids
    s = state.centroid_node_vectors
    
    # 1. Bind parameters to create a function purely of p
    f_point = lambda p: mapping_fn(p, map_params)

    # 2. Compute Jacobian matrix function using JAX
    jac_f = jax.jacfwd(f_point)
    
    # 3. Vectorize across all centroids
    f_vmap = jax.vmap(f_point)
    jac_vmap = jax.vmap(jac_f)
    
    # 4. Map centroids
    c_new = f_vmap(c)
    
    # 5. Transform CNVs using the Jacobian
    jac_matrices = jac_vmap(c)
    s_new = jnp.einsum('fab,fnb->fna', jac_matrices, s)

    return state._replace(
        face_centroids=c_new,
        centroid_node_vectors=s_new,
    )
