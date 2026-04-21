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
    Déformation (Warping) proportionnelle pour remplir la forme cible.
    """
    boundary = np.asarray(boundary_points, dtype=float)
    shape_center = np.mean(boundary, axis=0)
    boundary_vec = boundary - shape_center
    boundary_angles = np.arctan2(boundary_vec[:, 1], boundary_vec[:, 0])
    boundary_radii = np.linalg.norm(boundary_vec, axis=1)

    # Tri et extension périodique pour l'interpolation
    order = np.argsort(boundary_angles)
    b_angles = boundary_angles[order]
    b_radii = boundary_radii[order]
    b_angles = np.concatenate([b_angles - 2*np.pi, b_angles, b_angles + 2*np.pi])
    b_radii = np.tile(b_radii, 3)

    # Calcul sur la tessellation
    points = np.asarray(vertices, dtype=float)
    p_min, p_max = np.min(points, axis=0), np.max(points, axis=0)
    p_center = (p_min + p_max) / 2.0
    
    offsets = points - p_center
    p_angles = np.arctan2(offsets[:, 1], offsets[:, 0])
    p_norms = np.linalg.norm(offsets, axis=1)

    # Distance maximale vers le bord de la boîte englobante de la tessellation (pour normaliser)
    # On suppose une boîte rectangulaire [w, h]
    w, h = (p_max[0] - p_min[0])/2.0, (p_max[1] - p_min[1])/2.0
    # Distance au bord d'un rectangle de demi-côtés w, h à l'angle theta
    # r = 1 / max(|cos/w|, |sin/h|)
    norm_max_tess = 1.0 / np.maximum(np.abs(np.cos(p_angles))/w, np.abs(np.sin(p_angles))/h)

    # Rayon cible correspondant à chaque angle
    target_boundary_radii = np.interp(p_angles, b_angles, b_radii)
    
    # Warping proportionnel : r_final = (r_initial / r_max_initial) * r_target
    scale = np.zeros_like(p_norms)
    nonzero = p_norms > 0
    scale[nonzero] = target_boundary_radii[nonzero] / norm_max_tess[nonzero]

    return shape_center + offsets * scale[:, None]


def _resolve_circle_boundary(target_boundary):
    if isinstance(target_boundary, dict) and target_boundary.get("type") == "circle":
        return np.asarray(target_boundary["center"], dtype=float), float(target_boundary["radius"])
    if isinstance(target_boundary, tuple) and len(target_boundary) == 3 and target_boundary[0] == "circle":
        return np.asarray(target_boundary[1], dtype=float), float(target_boundary[2])
    return None

from geometry.target_shape import get_target_points

def compute_initial_map(
    tessellation: Tessellation,
    target_boundary,
    map_type: str = "conformal",
    scale_factor: float = 1.0,
) -> Tessellation:
    """Remappe les sommets vers la forme cible."""
    original_vertices = np.asarray(tessellation.vertices, dtype=float)
    if original_vertices.size == 0:
        return tessellation

    if map_type == "standard" or map_type == "rescaled":
        mapped_vertices = original_vertices * scale_factor
    elif map_type == "elliptical_grip":
        circle_params = _resolve_circle_boundary(target_boundary)
        
        if circle_params is not None:
            # Cas Cercle Classique
            center, radius = circle_params
            mapped_vertices = elliptical_grip_mapping(original_vertices, center, radius)
            mapped_vertices = center + (mapped_vertices - center) * scale_factor
        else:
            # Pour toute autre forme (Coeur, etc.), on utilise la projection de bordure
            target_pts = get_target_points(n_points=500)
            mapped_vertices = _project_to_boundary(original_vertices, target_pts)
            center = np.mean(target_pts, axis=0)
            mapped_vertices = center + (mapped_vertices - center) * scale_factor
    else:
        raise ValueError(f"Invalid map_type: {map_type}")

    return Tessellation(
        vertices=mapped_vertices,
        faces=[f for f in tessellation.faces],
        hinges=tessellation.hinges,
        voids=tessellation.voids,
    )

