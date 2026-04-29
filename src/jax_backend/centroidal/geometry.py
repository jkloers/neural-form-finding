"""
Centroidal geometry primitives.

Pure JAX functions to reconstruct vertex positions, bond vectors, and
other geometric quantities from the centroidal representation (c, s).

All functions are differentiable and vmappable.
"""

import jax.numpy as jnp


def reconstruct_vertices(face_centroids, centroid_node_vectors):
    """Reconstruct absolute vertex positions from centroidal data.

    Args:
        face_centroids: (n_faces, 2)
        centroid_node_vectors: (n_faces, max_nodes, 2)

    Returns:
        (n_faces, max_nodes, 2) — absolute vertex positions
    """
    return face_centroids[:, None, :] + centroid_node_vectors


def hinge_vertex_positions(face_centroids, cnv, hinge_node_pairs):
    """Compute positions of shared vertices at hinges for both faces.

    For each entry in hinge_node_pairs, computes:
        p1 = c[face_i] + s[face_i, local_j]
        p2 = c[face_k] + s[face_k, local_l]

    If the hinge is properly connected, p1 ≈ p2.

    Args:
        face_centroids: (n_faces, 2)
        cnv: (n_faces, max_nodes, 2)
        hinge_node_pairs: (n_pairs, 2, 2) — [[face_i, local_j], [face_k, local_l]]

    Returns:
        p1: (n_pairs, 2) — vertex position from face_i side
        p2: (n_pairs, 2) — vertex position from face_k side
    """
    fi = hinge_node_pairs[:, 0, 0]   # face indices, side 1
    lj = hinge_node_pairs[:, 0, 1]   # local node indices, side 1
    fk = hinge_node_pairs[:, 1, 0]   # face indices, side 2
    ll = hinge_node_pairs[:, 1, 1]   # local node indices, side 2

    p1 = face_centroids[fi] + cnv[fi, lj]
    p2 = face_centroids[fk] + cnv[fk, ll]
    return p1, p2


def hinge_adj_vertex_positions(face_centroids, cnv, hinge_adj_info):
    """Compute pivot and adjacent vertex positions for non-intersection checks.

    Args:
        face_centroids: (n_faces, 2)
        cnv: (n_faces, max_nodes, 2)
        hinge_adj_info: (n_hinges, 5) — [face_i, face_k, pivot_local_i, adj_local_i, adj_local_k]

    Returns:
        pivot: (n_hinges, 2) — pivot vertex position (from face_i)
        p_adj1: (n_hinges, 2) — adjacent vertex in face_i
        p_adj2: (n_hinges, 2) — adjacent vertex in face_k
    """
    fi = hinge_adj_info[:, 0]
    fk = hinge_adj_info[:, 1]
    pivot_li = hinge_adj_info[:, 2]
    adj_li = hinge_adj_info[:, 3]
    adj_lk = hinge_adj_info[:, 4]

    pivot = face_centroids[fi] + cnv[fi, pivot_li]
    p_adj1 = face_centroids[fi] + cnv[fi, adj_li]
    p_adj2 = face_centroids[fk] + cnv[fk, adj_lk]

    return pivot, p_adj1, p_adj2


def boundary_vertex_positions(face_centroids, cnv, boundary_face_node_ids):
    """Compute positions of boundary vertices.

    Args:
        face_centroids: (n_faces, 2)
        cnv: (n_faces, max_nodes, 2)
        boundary_face_node_ids: (n_boundary, 2) — [face_id, local_node_id]

    Returns:
        (n_boundary, 2) — absolute positions of boundary vertices
    """
    fi = boundary_face_node_ids[:, 0]
    lj = boundary_face_node_ids[:, 1]
    return face_centroids[fi] + cnv[fi, lj]
