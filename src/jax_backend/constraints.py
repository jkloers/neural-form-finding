"""
Geometric validity constraints in centroidal coordinates.

Each constraint is a pure JAX function operating on (face_centroids, cnv)
and returning a scalar penalty. All are differentiable.

These replace the vertex-based constraints in ff_optimizer/objective.py.
"""

import jax
import jax.numpy as jnp

from jax_backend.geometry import (
    hinge_vertex_positions,
    hinge_adj_vertex_positions,
    boundary_vertex_positions,
)


def hinge_connectivity(face_centroids, cnv, hinge_node_pairs):
    """Penalizes gap between vertices that should coincide at hinges.

    For each hinge vertex pair: c[i] + s[i][j] should equal c[k] + s[k][l].

    Args:
        face_centroids: (n_faces, 2)
        cnv: (n_faces, max_nodes, 2)
        hinge_node_pairs: (n_pairs, 2, 2)

    Returns:
        scalar — sum of squared distances between paired vertices.
    """
    p1, p2 = hinge_vertex_positions(face_centroids, cnv, hinge_node_pairs)
    return jnp.sum((p1 - p2)**2)


def hinge_non_intersection(face_centroids, cnv, hinge_adj_info, margin=1e-3):
    """Prevents local inversion of faces at hinges (negative angles).

    Checks that the cross product of the two hinge arms (pivot->adj1, pivot->adj2)
    stays positive, meaning the faces don't cross each other.

    Args:
        face_centroids: (n_faces, 2)
        cnv: (n_faces, max_nodes, 2)
        hinge_adj_info: (n_hinges, 5)
        margin: minimum normalized determinant before penalty kicks in.

    Returns:
        scalar — penalty on negative/small angles.
    """
    pivot, p_adj1, p_adj2 = hinge_adj_vertex_positions(
        face_centroids, cnv, hinge_adj_info)

    v1 = p_adj1 - pivot
    v2 = p_adj2 - pivot

    det = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    norm_v1 = jnp.linalg.norm(v1, axis=-1)
    norm_v2 = jnp.linalg.norm(v2, axis=-1)

    normalized_det = det / (norm_v1 * norm_v2 + 1e-8)
    violations = jax.nn.relu(margin - normalized_det)

    return jnp.sum(violations**2)


def target_fitting(face_centroids, cnv, boundary_face_node_ids, target_cloud):
    """Chamfer distance between boundary vertices and target point cloud.

    Args:
        face_centroids: (n_faces, 2)
        cnv: (n_faces, max_nodes, 2)
        boundary_face_node_ids: (n_boundary, 2)
        target_cloud: (n_target, 2)

    Returns:
        scalar — bidirectional Chamfer distance.
    """
    boundary_verts = boundary_vertex_positions(
        face_centroids, cnv, boundary_face_node_ids)

    dist_matrix = jnp.sum(
        (boundary_verts[:, None, :] - target_cloud[None, :, :])**2, axis=-1)

    min_to_target = jnp.min(dist_matrix, axis=1)
    min_to_boundary = jnp.min(dist_matrix, axis=0)

    return jnp.mean(min_to_target) + jnp.mean(min_to_boundary)


def hinge_arm_symmetry(face_centroids, cnv, hinge_adj_info, boundary_face_node_ids):
    """Forces hinge arms on the boundary to have equal lengths.

    Args:
        face_centroids: (n_faces, 2)
        cnv: (n_faces, max_nodes, 2)
        hinge_adj_info: (n_hinges, 5)
        boundary_face_node_ids: (n_boundary, 2)

    Returns:
        scalar — penalty on arm length asymmetry for boundary hinges.
    """
    pivot, p_adj1, p_adj2 = hinge_adj_vertex_positions(
        face_centroids, cnv, hinge_adj_info)

    fi = hinge_adj_info[:, 0]
    adj_li = hinge_adj_info[:, 3]
    adj_lk = hinge_adj_info[:, 4]
    fk = hinge_adj_info[:, 1]

    # Check if adjacent vertices are on the boundary
    boundary_pairs = boundary_face_node_ids  # (n_boundary, 2)
    adj1_pairs = jnp.stack([fi, adj_li], axis=-1)
    adj2_pairs = jnp.stack([fk, adj_lk], axis=-1)

    is_adj1_boundary = jnp.any(
        jnp.all(adj1_pairs[:, None, :] == boundary_pairs[None, :, :], axis=-1),
        axis=1)
    is_adj2_boundary = jnp.any(
        jnp.all(adj2_pairs[:, None, :] == boundary_pairs[None, :, :], axis=-1),
        axis=1)

    mask = (is_adj1_boundary | is_adj2_boundary).astype(jnp.float32)

    l1_sq = jnp.sum((pivot - p_adj1)**2, axis=-1)
    l2_sq = jnp.sum((pivot - p_adj2)**2, axis=-1)

    return jnp.sum(mask * (l1_sq - l2_sq)**2)


def void_opposite_edges_validity(cnv, void_opposite_node_pairs):
    """Maintains symmetry and collinearity of opposite edges in voids.

    Args:
        cnv: (n_faces, max_nodes, 2)
        void_opposite_node_pairs: (n_void_edges, 2, 3) -> [[f1, na1, nb1], [f2, na2, nb2]]

    Returns:
        tuple (length_penalty, collinearity_penalty)
    """
    if void_opposite_node_pairs.shape[0] == 0:
        return 0.0, 0.0

    # Extract node vectors for each edge vertex
    # cnv has shape (n_faces, max_nodes, 2)
    f1 = void_opposite_node_pairs[:, 0, 0]
    na1 = void_opposite_node_pairs[:, 0, 1]
    nb1 = void_opposite_node_pairs[:, 0, 2]
    
    f2 = void_opposite_node_pairs[:, 1, 0]
    na2 = void_opposite_node_pairs[:, 1, 1]
    nb2 = void_opposite_node_pairs[:, 1, 2]

    # vectors s_b - s_a
    v1 = cnv[f1, nb1] - cnv[f1, na1]
    v2 = cnv[f2, nb2] - cnv[f2, na2]

    # 1. Length symmetry: |v1|^2 == |v2|^2
    l1_sq = jnp.sum(v1**2, axis=-1)
    l2_sq = jnp.sum(v2**2, axis=-1)
    length_penalty = jnp.sum((l1_sq - l2_sq)**2)

    # 2. Collinearity: v1 x v2 == 0
    cross_prod = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    collinearity_penalty = jnp.sum(cross_prod**2)

    return length_penalty, collinearity_penalty


def initial_map_anchoring(face_centroids, cnv, state):
    """L2 penalty on distance to the initial mapped geometry (the reference state).

    Args:
        face_centroids: (n_faces, 2) — current coordinates
        cnv: (n_faces, max_nodes, 2) — current coordinates
        state: CentroidalState — initial reference state

    Returns:
        scalar — squared distance to reference.
    """
    c_diff = face_centroids - state.face_centroids
    s_diff = cnv - state.centroid_node_vectors
    return jnp.sum(c_diff**2) + jnp.sum(s_diff**2)


def boundary_face_rigidity(cnv, boundary_face_node_ids):
    """Penalizes deformation (non-squareness) of boundary faces.
    Forces all 4 edges to have equal length and the 2 diagonals to have equal length.

    Args:
        cnv: (n_faces, max_nodes, 2)
        boundary_face_node_ids: (n_boundary, 2)

    Returns:
        scalar — penalty on shape distortion for boundary faces.
    """
    if boundary_face_node_ids.shape[0] == 0:
        return 0.0

    # Use raw face IDs without jnp.unique. 
    # JAX requires static shapes, so jnp.unique crashes the compiler here.
    # Duplicates just mean the penalty is applied multiple times for corner faces.
    b_faces = boundary_face_node_ids[:, 0]
    b_cnv = cnv[b_faces]  # (n_boundary_nodes, max_nodes, 2)
    
    # Assuming faces are quads (first 4 nodes are valid)
    e1 = b_cnv[:, 1] - b_cnv[:, 0]
    e2 = b_cnv[:, 2] - b_cnv[:, 1]
    e3 = b_cnv[:, 3] - b_cnv[:, 2]
    e4 = b_cnv[:, 0] - b_cnv[:, 3]
    
    l1 = jnp.sum(e1**2, axis=1)
    l2 = jnp.sum(e2**2, axis=1)
    l3 = jnp.sum(e3**2, axis=1)
    l4 = jnp.sum(e4**2, axis=1)
    
    length_penalty = jnp.sum((l1 - l2)**2 + (l2 - l3)**2 + (l3 - l4)**2 + (l4 - l1)**2)
    
    d1 = b_cnv[:, 2] - b_cnv[:, 0]
    d2 = b_cnv[:, 3] - b_cnv[:, 1]
    diag_penalty = jnp.sum((jnp.sum(d1**2, axis=1) - jnp.sum(d2**2, axis=1))**2)
    
    return length_penalty + diag_penalty


def face_non_inversion(cnv):
    """Penalizes faces that have flipped inside out (negative signed area).
    Uses the shoelace formula on the centroid-node vectors.

    Args:
        cnv: (n_faces, max_nodes, 2)

    Returns:
        scalar — penalty for faces with negative oriented area.
    """
    # Assuming quad faces (first 4 nodes are ordered CCW originally)
    # Area = 0.5 * sum(x_i * y_i+1 - x_i+1 * y_i)
    area = 0.5 * (
        cnv[:, 0, 0] * cnv[:, 1, 1] - cnv[:, 0, 1] * cnv[:, 1, 0] +
        cnv[:, 1, 0] * cnv[:, 2, 1] - cnv[:, 1, 1] * cnv[:, 2, 0] +
        cnv[:, 2, 0] * cnv[:, 3, 1] - cnv[:, 2, 1] * cnv[:, 3, 0] +
        cnv[:, 3, 0] * cnv[:, 0, 1] - cnv[:, 3, 1] * cnv[:, 0, 0]
    )
    
    # We penalize if area drops below a small positive margin
    margin = 1e-4
    violations = jax.nn.relu(margin - area)
    
    return jnp.sum(violations**2)


def compute_geometric_objective(face_centroids, cnv, state, target_cloud, weights):
    """Total geometric validity objective.

    Args:
        face_centroids: (n_faces, 2) — optimizable
        cnv: (n_faces, max_nodes, 2) — optimizable
        state: CentroidalState — fixed topology
        target_cloud: (n_target, 2) — target shape
        weights: dict with keys:
            'connectivity', 'non_intersection', 'target', 'arm_symmetry', 
            'void_length', 'void_collinear'

    Returns:
        scalar — weighted sum of all geometric penalties.
    """
    e_connect = hinge_connectivity(
        face_centroids, cnv, state.hinge_node_pairs)

    e_non_inv = hinge_non_intersection(
        face_centroids, cnv, state.hinge_adj_info)

    e_target = target_fitting(
        face_centroids, cnv, state.boundary_face_node_ids, target_cloud)

    e_symmetry = hinge_arm_symmetry(
        face_centroids, cnv, state.hinge_adj_info, state.boundary_face_node_ids)

    e_void_l, e_void_c = void_opposite_edges_validity(
        cnv, state.void_opposite_node_pairs)

    e_anchoring = initial_map_anchoring(
        face_centroids, cnv, state)

    e_bound_rigid = boundary_face_rigidity(
        cnv, state.boundary_face_node_ids)

    e_inversion = face_non_inversion(cnv)

    return (weights.get('connectivity', 700.) * e_connect +
            weights.get('non_intersection', 1000.) * e_non_inv +
            weights.get('target', 1.) * e_target +
            weights.get('arm_symmetry', 1.) * e_symmetry +
            weights.get('void_length', 1.) * e_void_l +
            weights.get('void_collinear', 1.) * e_void_c +
            weights.get('anchoring', 100.) * e_anchoring +
            weights.get('boundary_rigidity', 10.) * e_bound_rigid +
            weights.get('face_inversion', 1000.) * e_inversion)
