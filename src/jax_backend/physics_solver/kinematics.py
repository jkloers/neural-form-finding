"""
The `kinematics` module implements the face-to-node rigid body kinematics.
State is displacement-only: (n_faces, 3) = [dx, dy, d_theta]. No velocities.
"""

from typing import Callable, Dict, Tuple, Union
import jax.numpy as jnp
from jax import vmap

from jax_backend.utils.linalg import rotation_matrix


def _face_to_node_displacement(face_displacement: jnp.ndarray, centroid_node_vectors: jnp.ndarray):
    """Computes displacement of a node belonging to a rigidly displaced face.

    Args:
        face_displacement (ndarray): shape (3,) = [dx, dy, d_theta].
        centroid_node_vectors (ndarray): shape (2,) vector from centroid to node.

    Returns:
        ndarray: shape (3,) = [node_dx, node_dy, d_theta].
    """
    face_centroid_displacement = face_displacement[:2]
    face_rotation = face_displacement[2]

    node_displacement = face_centroid_displacement + \
        jnp.dot(rotation_matrix(face_rotation) - jnp.eye(2), centroid_node_vectors)

    return jnp.concatenate([node_displacement, jnp.array([face_rotation]).flatten()])


# Vectorize over nodes per face (inner) and then over faces (outer)
face_to_node_kinematics = vmap(
    vmap(_face_to_node_displacement, in_axes=(None, 0)), in_axes=(0, 0)
)


def DOFsInfo(n_faces: int, constrained_face_DOF_pairs: jnp.ndarray) -> Tuple:
    """Returns free, constrained, and all DOF ids.

    Args:
        n_faces (int): Number of faces.
        constrained_face_DOF_pairs: shape (n_constraints, 2), each row [face_id, DOF_id].

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]: free_DOF_ids, constrained_DOF_ids, all_DOF_ids.
    """
    constrained_DOF_ids = jnp.array(
        [face_id * 3 + dof_id for face_id, dof_id in constrained_face_DOF_pairs]
    )
    all_DOF_ids = jnp.arange(n_faces * 3)
    free_DOF_ids = jnp.array(
        [dof for dof in all_DOF_ids if dof not in constrained_DOF_ids]
    )
    return free_DOF_ids, constrained_DOF_ids, all_DOF_ids


def build_constrained_kinematics(
        geometry,
        constrained_face_DOF_pairs: jnp.ndarray,
        constrained_DOFs_fn: Callable = lambda t, **kwargs: 0.):
    """Builds a constrained kinematics mapping free DOFs to full face displacements.

    Args:
        geometry: Geometry object (must have `n_faces` attribute).
        constrained_face_DOF_pairs (jnp.ndarray): shape (n_constraints, 2),
            each row [face_id, DOF_id] where DOF_id in {0: x, 1: y, 2: theta}.
        constrained_DOFs_fn (Callable, optional): Returns imposed values for constrained DOFs.
            Signature: (t, **kwargs) -> scalar or (n_constraints,).
            Defaults to 0 (zero displacement = clamped).

    Returns:
        Callable: ``constrained_kinematics(free_DOFs, t, constraint_params)``
            returns jnp.ndarray of shape (n_faces, 3).
    """
    n_faces = geometry.n_faces

    free_DOF_ids, constrained_DOF_ids, all_DOF_ids = DOFsInfo(
        n_faces, constrained_face_DOF_pairs)

    def constrained_kinematics(
            free_DOFs: jnp.ndarray,
            t: float = 0.,
            constraint_params: Dict = dict()) -> jnp.ndarray:
        """Maps free DOFs (and imposed values) to the full displacement field.

        Args:
            free_DOFs (jnp.ndarray): shape (n_free_DOFs,).
            t (float): time (for time-dependent constraints). Defaults to 0.
            constraint_params (Dict): kwargs forwarded to `constrained_DOFs_fn`.

        Returns:
            jnp.ndarray: shape (n_faces, 3) = [dx, dy, d_theta] per face.
        """
        all_DOFs = jnp.zeros((len(all_DOF_ids),))
        if len(constrained_DOF_ids) != 0:
            all_DOFs = all_DOFs.at[constrained_DOF_ids].set(
                constrained_DOFs_fn(t, **constraint_params)
            )
        all_DOFs = all_DOFs.at[free_DOF_ids].set(free_DOFs)
        return all_DOFs.reshape((n_faces, 3))

    return constrained_kinematics
