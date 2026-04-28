"""
The `statics` module implements the static equilibrium solver for rigid face assemblies.
State is displacement-only: (n_faces, 3) = [dx, dy, d_theta]. No velocities.
"""

from typing import Callable, Optional

import jax.numpy as jnp

from jax_backend.physics_solver.energy import constrain_energy
from jax_backend.physics_solver.loading import build_loading, build_static_loading
from jax_backend.physics_solver.kinematics import build_constrained_kinematics, DOFsInfo

from jax_backend.utils.utils import ControlParams, SolutionData

from jax.scipy.optimize import minimize


def setup_static_solver(
        geometry,
        energy_fn: Callable,
        loaded_face_DOF_pairs: Optional[jnp.ndarray] = None,
        loading_fn: Optional[Callable] = None,
        constrained_face_DOF_pairs: jnp.ndarray = jnp.array([]),
        constrained_DOFs_fn: Callable = lambda t, **kwargs: 0.) -> Callable:
    """Setup a static equilibrium solver by minimizing the total potential energy.

    Args:
        geometry: Geometry object (must have `n_faces` attribute).
        energy_fn (Callable): Internal potential energy with signature
            (face_displacement: (n_faces, 3), control_params) -> scalar.
        loaded_face_DOF_pairs (jnp.ndarray, optional): shape (n_loaded, 2),
            pairs [face_id, DOF_id] where external forces are applied.
        loading_fn (Callable, optional): External force function.
            Signature: (state, t, **loading_params) -> (n_loaded,).
        constrained_face_DOF_pairs (jnp.ndarray): shape (n_constraints, 2),
            pairs [face_id, DOF_id] for Dirichlet BCs (clamped DOFs).
        constrained_DOFs_fn (Callable): Returns imposed displacement values.
            Signature: (t, **kwargs) -> scalar or (n_constraints,).
            Defaults to 0 (zero displacement = clamped).

    Returns:
        Callable: solve_statics(state0, control_params) -> SolutionData.
    """

    # Build kinematics: free_DOFs -> full displacement (n_faces, 3)
    kinematics = build_constrained_kinematics(
        geometry=geometry,
        constrained_face_DOF_pairs=constrained_face_DOF_pairs,
        constrained_DOFs_fn=constrained_DOFs_fn
    )

    # Constrained energy: only depends on free DOFs
    constrained_energy = constrain_energy(energy_fn, kinematics)

    # Free DOF ids
    free_DOF_ids, _, _ = DOFsInfo(geometry.n_faces, constrained_face_DOF_pairs)

    # External loading
    if loaded_face_DOF_pairs is not None and loading_fn is not None:
        _loading_fn = build_loading(
            geometry=geometry,
            loaded_face_DOF_pairs=loaded_face_DOF_pairs,
            loading_fn=loading_fn,
            constrained_face_DOF_pairs=constrained_face_DOF_pairs
        )
    else:
        _loading_fn = None

    # Total potential energy = U_internal - W_external
    def total_potential_energy(free_DOFs: jnp.ndarray, t: float, control_params: ControlParams) -> float:
        U_int = constrained_energy(free_DOFs, t, control_params)
        if _loading_fn is not None:
            F_ext = _loading_fn(None, t, control_params.loading_params)
            W_ext = jnp.dot(F_ext, free_DOFs)
            return U_int - W_ext
        return U_int

    def solve_statics(state0: jnp.ndarray, control_params: ControlParams) -> SolutionData:
        """Solve for the static equilibrium.

        Args:
            state0 (jnp.ndarray): Initial displacement guess, shape (n_faces, 3).
            control_params (ControlParams): Geometrical + mechanical parameters.

        Returns:
            SolutionData: Equilibrium solution.
        """
        initial_free = state0.reshape(-1)[free_DOF_ids]
        result = minimize(total_potential_energy, initial_free,
                          args=(0., control_params), method='BFGS')
        displacement = kinematics(result.x, 0., control_params.constraint_params)
        return SolutionData(
            face_centroids=control_params.geometrical_params.face_centroids,
            centroid_node_vectors=control_params.geometrical_params.centroid_node_vectors,
            bond_connectivity=control_params.geometrical_params.bond_connectivity,
            fields=displacement
        )

    return solve_statics
