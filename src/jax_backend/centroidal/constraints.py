"""
Geometric validity constraints in centroidal coordinates.

Each constraint is a pure JAX function operating on (face_centroids, cnv)
and returning a scalar penalty. All are differentiable.

These replace the vertex-based constraints in ff_optimizer/objective.py.
"""

import jax
import jax.numpy as jnp

from jax_backend.centroidal.geometry import (
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

    return (weights.get('connectivity', 700.) * e_connect +
            weights.get('non_intersection', 1000.) * e_non_inv +
            weights.get('target', 1.) * e_target +
            weights.get('arm_symmetry', 1.) * e_symmetry +
            weights.get('void_length', 1.) * e_void_l +
            weights.get('void_collinear', 1.) * e_void_c)
