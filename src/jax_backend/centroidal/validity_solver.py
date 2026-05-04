"""
Geometric validity solver in centroidal coordinates.

Optimizes (face_centroids, centroid_node_vectors) to satisfy geometric
constraints while fitting a target shape. Topology is fixed.
"""

import jax.numpy as jnp
from jaxopt import ScipyMinimize

from jax_backend.centroidal.state import CentroidalState
from jax_backend.centroidal.constraints import compute_geometric_objective


def solve_geometric_validity(
        initial_state: CentroidalState,
        target_cloud: jnp.ndarray,
        weights: dict = None,
        method: str = 'BFGS') -> CentroidalState:
    """Optimize (face_centroids, centroid_node_vectors) for geometric validity.

    Minimizes the geometric objective (hinge connectivity, non-intersection,
    target fitting, arm symmetry) over the centroidal variables.

    Args:
        initial_state: CentroidalState with initial (c, s) and fixed topology.
        target_cloud: (n_target, 2) — target boundary points.
        weights: dict of constraint weights. See compute_geometric_objective.
        method: optimization method (default 'BFGS').

    Returns:
        CentroidalState with optimized (c*, s*) and unchanged topology.
    """
    if weights is None:
        weights = {
            'connectivity': 700.,
            'non_intersection': 1000.,
            'target': 1.,
            'arm_symmetry': 1.,
        }

    c0 = initial_state.face_centroids
    s0 = initial_state.centroid_node_vectors
    n_faces = c0.shape[0]
    max_nodes = s0.shape[1]

    # Pack (c, s) into a flat vector for the optimizer
    x0 = jnp.concatenate([c0.reshape(-1), s0.reshape(-1)])
    split_idx = n_faces * 2  # boundary between c and s in the flat vector

    def objective(x):
        c = x[:split_idx].reshape(n_faces, 2)
        s = x[split_idx:].reshape(n_faces, max_nodes, 2)
        return compute_geometric_objective(c, s, initial_state, target_cloud, weights)

    solver = ScipyMinimize(fun=objective, method=method, implicit_diff=True)
    result = solver.run(x0)

    c_opt = result.params[:split_idx].reshape(n_faces, 2)
    s_opt = result.params[split_idx:].reshape(n_faces, max_nodes, 2)

    return initial_state._replace(
        face_centroids=c_opt,
        centroid_node_vectors=s_opt,
    )
