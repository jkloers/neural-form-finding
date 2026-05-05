"""
The `statics` module implements the static equilibrium solver for rigid face assemblies.
State is displacement-only: (n_faces, 3) = [dx, dy, d_theta]. No velocities.
"""

from typing import Callable, Optional

import jax
import jax.numpy as jnp

from jax_backend.physics_solver.energy import constrain_energy
from jax_backend.physics_solver.loading import build_loading
from jax_backend.physics_solver.kinematics import build_constrained_kinematics, DOFsInfo

from jax_backend.physics_solver.params import ControlParams, SolutionData

from jaxopt import LBFGS


def setup_static_solver(
        geometry,
        energy_fn: Callable,
        loaded_face_DOF_pairs: Optional[jnp.ndarray] = None,
        loading_fn: Optional[Callable] = None,
        constrained_face_DOF_pairs: jnp.ndarray = jnp.array([]),
        constrained_DOFs_fn: Callable = lambda t, **kwargs: 0.,
        incremental: bool = False,
        num_steps: int = 10) -> Callable:
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
    free_DOF_ids, _ = DOFsInfo(geometry.n_faces, constrained_face_DOF_pairs)

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

        solver = LBFGS(fun=total_potential_energy)

        def solve_single_step(current_free_DOFs, t):
            result = solver.run(current_free_DOFs, t=t, control_params=control_params)
            fun_val = total_potential_energy(result.params, t, control_params)
            return result.params, (result.params, fun_val)

        if not incremental:
            final_free, (history_free, history_energy) = solve_single_step(initial_free, 1.0)
            history_free = history_free[None, :]
            history_energy = history_energy[None]
            t_array = jnp.array([1.0])
        else:
            t_array = jnp.linspace(1.0 / num_steps, 1.0, num_steps)
            final_free, (history_free, history_energy) = jax.lax.scan(
                solve_single_step,
                init=initial_free,
                xs=t_array
            )

        # Reconstruct displacements for the entire history
        mapped_kinematics = jax.vmap(kinematics, in_axes=(0, 0, None))
        history_displacement = mapped_kinematics(history_free, t_array, control_params.constraint_params)

        return SolutionData(
            fields=history_displacement,
            energies=history_energy,
        )

    return solve_statics
