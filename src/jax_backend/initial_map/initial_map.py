import jax.numpy as jnp
from jax_backend.pytrees import TessellationState, compute_face_lengths_sq
from geometry.target_shape import get_target_points

def elliptical_grip_mapping(vertices, center, radius): 
    # rectangle to ellipse mapping
    min_xy = jnp.min(vertices, axis=0)
    max_xy = jnp.max(vertices, axis=0)
    box_center = (min_xy + max_xy) / 2.0
    half_sizes = (max_xy - min_xy) / 2.0
    half_sizes = jnp.where(half_sizes == 0.0, 1.0, half_sizes)

    normalized = (vertices - box_center) / half_sizes
    u = normalized[:, 0]
    v = normalized[:, 1]

    x_disk = u * jnp.sqrt(jnp.maximum(0.0, 1.0 - (v ** 2) / 2.0))
    y_disk = v * jnp.sqrt(jnp.maximum(0.0, 1.0 - (u ** 2) / 2.0))

    points = jnp.column_stack((x_disk, y_disk)) * radius
    return points + jnp.asarray(center, dtype=float)

def _project_to_boundary(vertices, boundary_points):
    """
    Proportional warping to fill the target shape.
    """
    boundary = jnp.asarray(boundary_points, dtype=float)
    shape_center = jnp.mean(boundary, axis=0)
    boundary_vec = boundary - shape_center
    boundary_angles = jnp.arctan2(boundary_vec[:, 1], boundary_vec[:, 0])
    boundary_radii = jnp.linalg.norm(boundary_vec, axis=1)

    # Sorting and periodic extension for interpolation
    order = jnp.argsort(boundary_angles)
    b_angles = boundary_angles[order]
    b_radii = boundary_radii[order]
    b_angles = jnp.concatenate([b_angles - 2*jnp.pi, b_angles, b_angles + 2*jnp.pi])
    b_radii = jnp.tile(b_radii, 3)

    # Calculations on the tessellation
    points = jnp.asarray(vertices, dtype=float)
    p_min, p_max = jnp.min(points, axis=0), jnp.max(points, axis=0)
    p_center = (p_min + p_max) / 2.0
    
    offsets = points - p_center
    p_angles = jnp.arctan2(offsets[:, 1], offsets[:, 0])
    p_norms = jnp.linalg.norm(offsets, axis=1)

    # Maximum distance to the edge of the tessellation bounding box
    w, h = (p_max[0] - p_min[0])/2.0, (p_max[1] - p_min[1])/2.0
    norm_max_tess = 1.0 / jnp.maximum(jnp.abs(jnp.cos(p_angles))/w, jnp.abs(jnp.sin(p_angles))/h)

    # Target radius corresponding to each angle
    target_boundary_radii = jnp.interp(p_angles, b_angles, b_radii)
    
    # Proportional warping: r_final = (r_initial / r_max_initial) * r_target
    scale = jnp.where(p_norms > 0, target_boundary_radii / norm_max_tess, 0.0)

    return shape_center + offsets * scale[:, None]

def _resolve_circle_boundary(target_boundary):
    if isinstance(target_boundary, dict) and target_boundary.get("type") == "circle":
        return jnp.asarray(target_boundary["center"], dtype=float), float(target_boundary["radius"])
    if isinstance(target_boundary, tuple) and len(target_boundary) == 3 and target_boundary[0] == "circle":
        return jnp.asarray(target_boundary[1], dtype=float), float(target_boundary[2])
    return None

def make_initial_map(
    state: TessellationState,
    target_boundary,
    map_type: str = "conformal",
    scale_factor: float = 1.0,
) -> TessellationState:
    """Remaps the vertices to the target shape and updates the rest lengths."""
    original_vertices = state.X
    if original_vertices.size == 0:
        return state

    if map_type == "standard" or map_type == "rescaled":
        mapped_vertices = original_vertices * scale_factor
    elif map_type == "elliptical_grip":
        circle_params = _resolve_circle_boundary(target_boundary)
        
        if circle_params is not None:
            # Classic Circle Case
            center, radius = circle_params
            mapped_vertices = elliptical_grip_mapping(original_vertices, center, radius)
            mapped_vertices = center + (mapped_vertices - center) * scale_factor
        else:
            # For any other shape (Heart, etc.), we use boundary projection
            target_pts = get_target_points(n_points=500)
            mapped_vertices = _project_to_boundary(original_vertices, target_pts)
            center = jnp.mean(jnp.asarray(target_pts), axis=0)
            mapped_vertices = center + (mapped_vertices - center) * scale_factor
    else:
        raise ValueError(f"Invalid map_type: {map_type}")

    new_F_rest = compute_face_lengths_sq(mapped_vertices, state.F_idx)

    return state._replace(X=mapped_vertices, F_rest_lengths_sq=new_F_rest)
