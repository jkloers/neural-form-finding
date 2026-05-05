"""
Centroidal state representation for rigid-face assemblies.

CentroidalState is the central data structure for the unified pipeline.
It carries both the optimizable variables (face_centroids, centroid_node_vectors)
and the fixed topology (hinge connectivity, BCs, mechanical properties).
"""

from typing import NamedTuple

import jax.numpy as jnp


class CentroidalState(NamedTuple):
    """Full centroidal state of a tessellation.

    Optimizable variables (modified by the geometric optimizer):
        face_centroids:         (n_faces, 2)              — centroid positions
        centroid_node_vectors:  (n_faces, max_nodes, 2)    — face shapes (centroid→node vectors)

    Fixed topology (never modified during optimization):
        hinge_face_pairs:       (n_hinge_vertices, 2)      — [face_i, face_k] sharing a vertex
        hinge_node_pairs:       (n_hinge_vertices, 2, 2)   — [[face_i, local_j], [face_k, local_l]]
                                                             for each shared vertex at a hinge
        bond_connectivity:      (n_hinges, 2)              — [node_idx_i, node_idx_k] 
                                                             pre-computed global indices for ligaments
        hinge_adj_info:         (n_hinges, 5)              — [face_i, face_k, pivot_local_i,
                                                               adj_local_i, adj_local_k]
                                                             for non-intersection checks
        boundary_face_node_ids: (n_boundary_nodes, 2)      — [face_id, local_node_id]
                                                             for target fitting

    Boundary conditions:
        constrained_face_DOF_pairs: (n_constraints, 2)     — [face_id, DOF_id] (Dirichlet)
        loaded_face_DOF_pairs:  (n_loaded, 2)              — [face_id, DOF_id] (Neumann)
        load_values:            (n_loaded,)                — force magnitudes

    Mechanical properties:
        k_stretch:              (n_hinges,)
        k_shear:                (n_hinges,)
        k_rot:              (n_hinges,)
        density:                (n_faces,)
    """

    # ── Optimizable variables ─────────────────────────────────────────────────
    face_centroids: jnp.ndarray
    centroid_node_vectors: jnp.ndarray

    # ── Fixed topology ────────────────────────────────────────────────────────
    hinge_face_pairs: jnp.ndarray
    hinge_node_pairs: jnp.ndarray
    bond_connectivity: jnp.ndarray
    hinge_adj_info: jnp.ndarray
    boundary_face_node_ids: jnp.ndarray
    void_opposite_node_pairs: jnp.ndarray  # (n_void_edges, 2, 3) -> [[f1, na1, nb1], [f2, na2, nb2]]

    # ── Boundary conditions ───────────────────────────────────────────────────
    constrained_face_DOF_pairs: jnp.ndarray
    loaded_face_DOF_pairs: jnp.ndarray
    load_values: jnp.ndarray

    # ── Mechanical properties ─────────────────────────────────────────────────
    k_stretch: jnp.ndarray
    k_shear: jnp.ndarray
    k_rot: jnp.ndarray
    density: jnp.ndarray

    # ── Methods ──────────────────────────────────────────────────────────────
    def get_loading_function(self):
        """Returns the loading function or None if no loads are defined."""
        if len(self.loaded_face_DOF_pairs) > 0:
            force_values = self.load_values
            return lambda state, t, **kwargs: t * force_values
        return None
