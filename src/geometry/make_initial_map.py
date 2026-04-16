import numpy as np
from topology.core import Tessellation

def elliptical_grip_mapping(vertices, center, radius): # rectangle to ellipse mapping
    min_xy = np.min(vertices, axis=0)
    max_xy = np.max(vertices, axis=0)
    box_center = (min_xy + max_xy) / 2.0
    half_sizes = (max_xy - min_xy) / 2.0
    half_sizes[half_sizes == 0.0] = 1.0

    normalized = (vertices - box_center) / half_sizes
    u = normalized[:, 0]
    v = normalized[:, 1]

    x_disk = u * np.sqrt(np.maximum(0.0, 1.0 - (v ** 2) / 2.0))
    y_disk = v * np.sqrt(np.maximum(0.0, 1.0 - (u ** 2) / 2.0))

    points = np.column_stack((x_disk, y_disk)) * radius
    return points + np.asarray(center, dtype=float)


def _project_to_boundary(vertices, boundary_points):
    """
    Project vertices onto the boundary of the target shape.
    """
    boundary = np.asarray(boundary_points, dtype=float)
    if boundary.ndim != 2 or boundary.shape[1] != 2:
        raise ValueError("target_boundary must be a Nx2 array of boundary points")

    shape_center = np.mean(boundary, axis=0)
    boundary_vec = boundary - shape_center
    boundary_angles = np.arctan2(boundary_vec[:, 1], boundary_vec[:, 0])
    boundary_radii = np.linalg.norm(boundary_vec, axis=1)

    order = np.argsort(boundary_angles)
    boundary_angles = boundary_angles[order]
    boundary_radii = boundary_radii[order]

    # Extend angles periodically so interpolation is stable across the branch cut.
    boundary_angles = np.concatenate([boundary_angles - 2 * np.pi, boundary_angles, boundary_angles + 2 * np.pi])
    boundary_radii = np.tile(boundary_radii, 3)

    points = np.asarray(vertices, dtype=float)
    points_center = np.mean(points, axis=0)
    offsets = points - points_center
    angles = np.arctan2(offsets[:, 1], offsets[:, 0])
    norms = np.linalg.norm(offsets, axis=1)

    target_radii = np.interp(np.unwrap(angles), boundary_angles, boundary_radii)
    scale = np.zeros_like(norms)
    nonzero = norms > 0
    scale[nonzero] = np.minimum(1.0, target_radii[nonzero] / norms[nonzero])

    return points_center + offsets * scale[:, None]


def _resolve_circle_boundary(target_boundary):
    if isinstance(target_boundary, dict) and target_boundary.get("type") == "circle":
        return np.asarray(target_boundary["center"], dtype=float), float(target_boundary["radius"])
    if isinstance(target_boundary, tuple) and len(target_boundary) == 3 and target_boundary[0] == "circle":
        return np.asarray(target_boundary[1], dtype=float), float(target_boundary[2])
    return None

def compute_initial_map(
    tessellation: Tessellation,
    target_boundary,
    map_type: str = "conformal",
    scale_factor: float = 1.0,
) -> Tessellation:
    """
    Map the tessellation vertices into a target shape.

    Parameters
    ----------
    tessellation:
        Input tessellation with vertices, faces and hinges.
    target_boundary:
        Either a circle descriptor like ('circle', center, radius),
        a dict {'type': 'circle', 'center': center, 'radius': radius},
        or an Nx2 array of boundary points describing the target shape.
    map_type:
        'conformal' uses a square-to-disk mapping for circle targets and
        boundary projection for arbitrary shapes.
        'standard' and 'rescaled' simply scale the original pattern.
    scale_factor:
        Uniform scale applied after mapping.

    Returns
    -------
    Tessellation
        New tessellation with remapped vertices.
    """
    original_vertices = np.asarray(tessellation.vertices, dtype=float)
    if original_vertices.size == 0:
        return Tessellation(vertices=original_vertices, faces=[face.vertex_indices for face in tessellation.faces], hinges=tessellation.hinges)

    if map_type == "standard":
        mapped_vertices = original_vertices * scale_factor
    elif map_type == "rescaled":
        mapped_vertices = original_vertices * scale_factor
    elif map_type == "elliptical_grip":
        circle_params = _resolve_circle_boundary(target_boundary)
        if circle_params is not None:
            center, radius = circle_params
            mapped_vertices = elliptical_grip_mapping(original_vertices, center, radius)
            mapped_vertices = center + (mapped_vertices - center) * scale_factor
        elif isinstance(target_boundary, np.ndarray):
            mapped_vertices = _project_to_boundary(original_vertices, target_boundary)
            mapped_vertices = target_boundary.mean(axis=0) + (mapped_vertices - target_boundary.mean(axis=0)) * scale_factor
        else:
            raise ValueError("Unsupported target_boundary type for elliptical grip mapping")
    else:
        raise ValueError(f"Invalid map_type: {map_type}")

    return Tessellation(
        vertices=mapped_vertices,
        faces=[face.vertex_indices for face in tessellation.faces],
        hinges=tessellation.hinges,
    )

