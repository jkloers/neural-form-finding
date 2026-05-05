"""
Geometric validity solver in centroidal coordinates.

Optimizes (face_centroids, centroid_node_vectors) to satisfy geometric
constraints (hinge connectivity, non-intersection, target fitting, etc.)
while staying close to the initially mapped configuration.

The two optimizable variables are:
  - face_centroids         (n_faces, 2)            abbreviated `c` or `centroids`
  - centroid_node_vectors  (n_faces, max_nodes, 2)  abbreviated `cnv` or `shapes`

They are packed into a single flat vector for the L-BFGS optimizer and
unpacked after convergence.
"""

import jax.numpy as jnp
from jaxopt import LBFGS

from jax_backend.state import CentroidalState
from jax_backend.constraints import compute_geometric_objective


# Default weights for the geometric objective.
# Exported so callers can selectively override individual keys.
DEFAULT_GEOMETRIC_WEIGHTS = {
    'connectivity':      700.,
    'non_intersection': 1000.,
    'target':              1.,
    'arm_symmetry':        1.,
    'void_length':         1.,
    'void_collinear':      1.,
    'anchoring':         100.,
    'boundary_rigidity':  10.,
    'face_inversion':   1000.,
}


def _pack(face_centroids: jnp.ndarray,
          centroid_node_vectors: jnp.ndarray) -> tuple[jnp.ndarray, int]:
    """Packs (face_centroids, centroid_node_vectors) into a flat 1D vector.

    Returns:
        x:         flat array ready for the optimizer
        split_idx: index separating the two variables in x
    """
    split_idx = face_centroids.shape[0] * 2
    x = jnp.concatenate([face_centroids.reshape(-1),
                          centroid_node_vectors.reshape(-1)])
    return x, split_idx


def _unpack(x: jnp.ndarray,
            split_idx: int,
            n_faces: int,
            max_nodes: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Unpacks the flat optimizer vector back to (face_centroids, centroid_node_vectors)."""
    face_centroids = x[:split_idx].reshape(n_faces, 2)
    centroid_node_vectors = x[split_idx:].reshape(n_faces, max_nodes, 2)
    return face_centroids, centroid_node_vectors


def solve_geometric_validity(
        initial_state: CentroidalState,
        target_cloud: jnp.ndarray,
        weights: dict = None) -> CentroidalState:
    """Optimize (face_centroids, centroid_node_vectors) for geometric validity.

    Minimizes the geometric objective — a weighted sum of hinge connectivity,
    non-intersection, target fitting, arm symmetry, and anchoring penalties —
    over the centroidal variables, keeping topology fixed.

    Args:
        initial_state: CentroidalState with initial geometry and fixed topology.
        target_cloud:  (n_target, 2) — target boundary point cloud.
        weights:       dict of penalty weights; missing keys use DEFAULT_GEOMETRIC_WEIGHTS.

    Returns:
        CentroidalState with optimized (face_centroids, centroid_node_vectors)
        and unchanged topology.
    """
    w = {**DEFAULT_GEOMETRIC_WEIGHTS, **(weights or {})}

    n_faces = initial_state.face_centroids.shape[0]
    max_nodes = initial_state.centroid_node_vectors.shape[1]

    x0, split_idx = _pack(initial_state.face_centroids,
                           initial_state.centroid_node_vectors)

    def objective(x, state_param):
        centroids, cnv = _unpack(x, split_idx, n_faces, max_nodes)
        return compute_geometric_objective(
            centroids, cnv, state_param, target_cloud, w)

    result = LBFGS(fun=objective).run(x0, state_param=initial_state)

    centroids_opt, cnv_opt = _unpack(result.params, split_idx, n_faces, max_nodes)

    return initial_state._replace(
        face_centroids=centroids_opt,
        centroid_node_vectors=cnv_opt,
    )
