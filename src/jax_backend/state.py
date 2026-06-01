"""
Centroidal state representation for rigid-face assemblies.

CentroidalState is the central data structure for the unified pipeline.
It carries both the optimizable variables (face_centroids, centroid_node_vectors)
and the fixed topology (hinge connectivity, BCs, mechanical properties).
"""

from typing import NamedTuple

import numpy as np
import jax.numpy as jnp
from jaxtyping import Array, Float, Int


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

    # ── Optimizable variables (JAX arrays — may receive gradients) ───────────
    face_centroids:         Float[Array, "n_faces 2"]
    centroid_node_vectors:  Float[Array, "n_faces max_nodes 2"]

    # ── Fixed topology (NumPy arrays — never traced by JAX) ──────────────────
    hinge_face_pairs:         Int[np.ndarray, "n_hinge_vertices 2"]
    hinge_node_pairs:         Int[np.ndarray, "n_hinge_vertices 2 2"]
    bond_connectivity:        Int[np.ndarray, "n_hinges 2"]
    hinge_adj_info:           Int[np.ndarray, "n_hinges 5"]
    boundary_face_node_ids:   Int[np.ndarray, "n_boundary_nodes 2"]
    void_opposite_node_pairs: Int[np.ndarray, "n_void_edges 2 3"]

    # ── Boundary conditions ───────────────────────────────────────────────────
    constrained_face_DOF_pairs: Int[np.ndarray, "n_constraints 2"]
    loaded_face_DOF_pairs:      Int[np.ndarray, "n_loaded 2"]
    load_values:                Float[Array, "n_loaded"]

    # ── Mechanical properties (JAX arrays — broadcast-friendly scalars) ───────
    k_stretch: Float[Array, "n_hinges"]
    k_shear:   Float[Array, "n_hinges"]
    k_rot:     Float[Array, "n_hinges"]
    density:   Float[Array, "n_faces"]

    # ── Area constraints ──────────────────────────────────────────────────────
    initial_face_areas: Float[Array, "n_faces"]

    # ── Methods ──────────────────────────────────────────────────────────────

    def get_loading_function(self):
        """Returns the loading function or None if no loads are defined."""
        if len(self.loaded_face_DOF_pairs) > 0:
            force_values = self.load_values
            return lambda state, t, **kwargs: t * force_values
        return None

    @classmethod
    def from_tessellation(cls, tessellation, target_cfg=None) -> 'CentroidalState':
        """Builds a CentroidalState from a configured Tessellation object.

        Handles the NumPy→JAX conversion and initial area calculations.
        """
        import numpy as np
        from jax_backend.geometry import compute_face_areas

        cs_dict = tessellation._to_dict()

        # Fields that are dynamic (JAX arrays, may be differentiated)
        dynamic_fields = {'face_centroids', 'centroid_node_vectors', 'load_values'}

        state_kwargs = {
            k: jnp.array(v) if k in dynamic_fields else np.array(v)
            for k, v in cs_dict.items()
        }

        # Étape 3 : Calcul des aires initiales
        cnv = state_kwargs['centroid_node_vectors']
        initial_face_areas = compute_face_areas(cnv)
        state_kwargs['initial_face_areas'] = initial_face_areas

        # Sort boundary_face_node_ids CCW by angle around the tessellation centroid.
        # This fixed ordering (computed from flat positions) lets compute_void_area
        # apply the shoelace formula directly during training without argsort.
        bids = state_kwargs['boundary_face_node_ids']          # (n_bnd, 2)
        fc_np   = np.array(state_kwargs['face_centroids'])
        cnv_np  = np.array(cnv)
        b_pos   = fc_np[bids[:, 0]] + cnv_np[bids[:, 0], bids[:, 1]]  # (n_bnd, 2)
        centroid_xy = b_pos.mean(axis=0)
        angles  = np.arctan2(b_pos[:, 1] - centroid_xy[1],
                             b_pos[:, 0] - centroid_xy[0])
        state_kwargs['boundary_face_node_ids'] = bids[np.argsort(angles)]

        return cls(**state_kwargs)
