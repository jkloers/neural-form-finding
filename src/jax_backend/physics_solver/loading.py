"""
The `loading` module implements loading functions for static and dynamic problems.
"""

from typing import Callable, Dict

import jax.numpy as jnp

from jax_backend.physics_solver.kinematics import DOFsInfo


def build_loading(
        geometry,
        loaded_face_DOF_pairs: jnp.ndarray,
        loading_fn: Callable,
        constrained_face_DOF_pairs: jnp.ndarray = jnp.array([])):
    """Defines the loading function.

    Args:
        geometry: Geometry object (must have `n_faces` attribute).
        loaded_face_DOF_pairs (jnp.ndarray): array of shape (Any, 2) where each row
            defines a pair of [face_id, DOF_id] where DOF_id is either 0, 1, or 2.
        loading_fn (Callable): Loading function. Output shape should either be scalar
            or match (len(loaded_face_DOF_pairs),).
        constrained_face_DOF_pairs (jnp.ndarray, optional): Array of shape (n_constraints, 2)
            where each row is of the form [face_id, DOF_id]. Defaults to jnp.array([]).

    Returns:
        Callable: vector loading function evaluating to `loading_fn` for the DOFs
            defined by `loaded_face_DOF_pairs` and 0 otherwise.
    """

    # loaded DOF ids based on global numeration
    loaded_DOF_ids = jnp.array(
        [face_id * 3 + DOF_id for face_id, DOF_id in loaded_face_DOF_pairs])
    # Retrieve free DOFs from constraints info (this information is assumed to be static)
    free_DOF_ids, _, all_DOF_ids = DOFsInfo(
        geometry.n_faces, constrained_face_DOF_pairs)

    def global_loading_fn(state, t, loading_params: Dict):

        loading_vector = jnp.zeros((len(all_DOF_ids),))
        loading_vector = loading_vector.at[loaded_DOF_ids].set(
            loading_fn(state, t, **loading_params)
        )
        # Reduce loading vector to the free DOFs
        loading_vector = loading_vector[free_DOF_ids]

        return loading_vector

    return global_loading_fn


def build_static_loading(
        geometry,
        loaded_face_DOF_pairs: jnp.ndarray,
        force_values: jnp.ndarray,
        constrained_face_DOF_pairs: jnp.ndarray = jnp.array([])):
    """Builds a static (constant) loading vector reduced to the free DOFs.

    Args:
        geometry: Geometry object (must have `n_faces` attribute).
        loaded_face_DOF_pairs (jnp.ndarray): shape (n_loaded, 2) — [face_id, DOF_id].
        force_values (jnp.ndarray): shape (n_loaded,) — force magnitude on each loaded DOF.
        constrained_face_DOF_pairs (jnp.ndarray, optional): shape (n_constraints, 2).

    Returns:
        jnp.ndarray: shape (n_free_DOFs,) — static force vector reduced to free DOFs.
    """

    loaded_DOF_ids = jnp.array(
        [face_id * 3 + dof_id for face_id, dof_id in loaded_face_DOF_pairs])
    free_DOF_ids, _, all_DOF_ids = DOFsInfo(
        geometry.n_faces, constrained_face_DOF_pairs)

    full_loading = jnp.zeros(len(all_DOF_ids))
    full_loading = full_loading.at[loaded_DOF_ids].set(force_values)

    return full_loading[free_DOF_ids]