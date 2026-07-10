"""
Centroidal geometry primitives.

Pure JAX functions to reconstruct vertex positions, bond vectors, and
other geometric quantities from the centroidal representation (c, s).

All functions are differentiable and vmappable.
"""

import jax
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


def deformed_vertices(state, displacement):
    """Apply a per-face rigid-body displacement to the panel vertices.

    Used to reconstruct deployed geometry from a Stage-2 solution (each face
    translates by [dx, dy] and rotates by dtheta about its centroid).

    Args:
        state: reference CentroidalState.
        displacement: (n_faces, 3) = [dx, dy, dtheta] per face.

    Returns:
        (n_faces, max_nodes, 2) deformed node positions.
    """
    centroids = state.face_centroids + displacement[:, :2]
    c, s = jnp.cos(displacement[:, 2]), jnp.sin(displacement[:, 2])
    rot = jnp.stack([jnp.stack([c, -s], axis=-1),
                     jnp.stack([s, c], axis=-1)], axis=-2)        # (n_faces, 2, 2)
    rotated = jnp.einsum("fij,fnj->fni", rot, state.centroid_node_vectors)
    return centroids[:, None, :] + rotated


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


def build_reference_bond_vectors(valid_state, hinge_axial_dirs=None, l0_nom=1e-4):
    """Compute bond reference vectors from the validated centroidal state.

    Each hinge connects vertex1 (in face1) to vertex2 (in face2).
    hinge_node_pairs has exactly 1 entry per hinge.

    Point-ROM ↔ physical-hinge frame bridge (Phase 3, Gap 1): for closed hinges the two hinge
    vertices coincide, so the natural bond ``p2 - p1`` is degenerate and its direction is
    meaningless. ``ligament_strains`` resolves axial vs shear relative to this vector, so a
    degenerate/arbitrary direction gives an arbitrary (a, s) split. When ``hinge_axial_dirs``
    (the per-hinge secondary-cut direction ``sec_dir`` from ``compute_hinge_descriptors``) is
    supplied, each degenerate hinge is given a nominal-length reference vector ALONG that cut
    axis — the same frame the RVE imposed (its axial DOF ``a`` is translation along the secondary
    cut). The axis is oriented face_i→face_k so opening is a > 0.

    Backward-compatible: with ``hinge_axial_dirs=None`` the old unit-x fallback is used, and for
    the canonical closed configs (k_stretch = k_shear, isotropic) the axis choice does not change
    the spring energy at all — it only matters for anisotropic springs and the learned surrogate.

    Args:
        valid_state:      CentroidalState after geometric validation.
        hinge_axial_dirs: optional (n_hinges, 2) unit axial (secondary-cut) directions.
        l0_nom:           nominal reference length for degenerate (closed) hinges.

    Returns:
        (n_hinges, 2) — reference bond vectors.
    """
    p1, p2 = hinge_vertex_positions(
        valid_state.face_centroids,
        valid_state.centroid_node_vectors,
        valid_state.hinge_node_pairs,
    )
    bond = p2 - p1

    # Use squared norm (no sqrt) for the condition so the backward pass never computes
    # bond/|bond| at bond=0 (which would give NaN via 0*NaN in XLA).
    min_len = 1e-4
    sq_norms = jnp.sum(bond ** 2, axis=-1, keepdims=True)   # gradient = 2*bond, zero at 0

    if hinge_axial_dirs is None:
        fallback = jnp.zeros_like(bond).at[:, 0].set(min_len)          # canonical unit-x
    else:
        u = hinge_axial_dirs / (jnp.linalg.norm(hinge_axial_dirs, axis=-1, keepdims=True) + 1e-12)
        # orient face_i → face_k so a > 0 is the hinge opening
        fi = valid_state.hinge_node_pairs[:, 0, 0]
        fk = valid_state.hinge_node_pairs[:, 1, 0]
        ab = valid_state.face_centroids[fk] - valid_state.face_centroids[fi]
        sgn = jnp.sign(jnp.sum(u * ab, axis=-1, keepdims=True) + 1e-30)
        fallback = l0_nom * u * sgn
    return jnp.where(sq_norms < min_len ** 2, fallback, bond)


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


def compute_polygon_area(vertices):
    """Compute the area of a 2D polygon using the Shoelace formula.
    
    Args:
        vertices: (N, 2) array of ordered coordinates.
        
    Returns:
        Scalar area.
    """
    x = vertices[:, 0]
    y = vertices[:, 1]
    # Circular shift to multiply x_i * y_{i+1} and y_i * x_{i+1}
    # Signed shoelace (positive for CCW-oriented faces, which is the convention throughout).
    # Avoids jnp.abs whose gradient is 0/0 at S=0 (NaN when a face becomes degenerate).
    return 0.5 * (jnp.dot(x, jnp.roll(y, -1)) - jnp.dot(y, jnp.roll(x, -1)))


def compute_face_areas(centroid_node_vectors):
    """Compute the area of each face using the Shoelace formula.
    
    Args:
        centroid_node_vectors: (n_faces, max_nodes, 2)
        
    Returns:
        (n_faces,) — area of each face
    """
    # Vectorize compute_polygon_area across all faces
    return jax.vmap(compute_polygon_area)(centroid_node_vectors)


def compute_total_area(centroid_node_vectors):
    """Compute the total area of the tessellation (sum of all face areas).

    Args:
        centroid_node_vectors: (n_faces, max_nodes, 2)

    Returns:
        Scalar total area.
    """
    return jnp.sum(compute_face_areas(centroid_node_vectors))


def compute_void_area(face_centroids, centroid_node_vectors, boundary_face_node_ids):
    """Compute total void (aperture) area: boundary polygon area minus total face area.

    boundary_face_node_ids must be pre-sorted CCW (done in CentroidalState.from_tessellation).
    Differentiable — no argsort at runtime.

    Args:
        face_centroids:         (n_faces, 2)
        centroid_node_vectors:  (n_faces, max_nodes, 2)
        boundary_face_node_ids: (n_boundary, 2) — [face_id, local_node_id], CCW-sorted

    Returns:
        Scalar void area (boundary polygon area minus sum of face areas).
    """
    fi = boundary_face_node_ids[:, 0]
    li = boundary_face_node_ids[:, 1]
    b_verts = face_centroids[fi] + centroid_node_vectors[fi, li]   # (n_bnd, 2)
    boundary_area = compute_polygon_area(b_verts)
    face_area     = compute_total_area(centroid_node_vectors)
    return boundary_area - face_area
