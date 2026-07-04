"""
The `statics` module implements the static equilibrium solver for rigid face assemblies.
State is displacement-only: (n_faces, 3) = [dx, dy, d_theta]. No velocities.
"""

from typing import Callable, Optional

import jax
import jax.numpy as jnp

from nff.stages.physics.energy import constrain_energy
from nff.stages.physics.loading import build_loading
from nff.stages.physics.kinematics import build_constrained_kinematics, DOFsInfo

from nff.stages.physics.params import ControlParams, SolutionData

import functools
from jaxopt import LBFGS
from jaxopt.linear_solve import solve_cg


def setup_static_solver(
        geometry,
        energy_fn: Callable,
        loaded_face_DOF_pairs: Optional[jnp.ndarray] = None,
        loading_fn: Optional[Callable] = None,
        constrained_face_DOF_pairs: jnp.ndarray = jnp.array([]),
        constrained_DOFs_fn: Callable = lambda t, **kwargs: 0.,
        incremental: bool = False,
        num_steps: int = 10,
        solver_maxiter: int = 1000,
        solver_tol: float = 1e-5,
        updated_lagrangian: bool = False,
        backward_reg: float = 0.0) -> Callable:
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
        Callable: solve_statics_fn(initial_displacements, control_params) -> SolutionData.
    """

    # Build kinematics: free_DOFs -> full displacement (n_faces, 3)
    kinematics_fn = build_constrained_kinematics(
        geometry=geometry,
        constrained_face_DOF_pairs=constrained_face_DOF_pairs,
        constrained_DOFs_fn=constrained_DOFs_fn
    )

    # Constrained energy: only depends on free DOFs
    constrained_energy_fn = constrain_energy(energy_fn, kinematics_fn)

    # Free DOF ids
    free_DOF_ids, _ = DOFsInfo(geometry.n_faces, constrained_face_DOF_pairs)

    # External loading: defaults to zero if not provided
    if loaded_face_DOF_pairs is not None and loading_fn is not None:
        _loading_fn = build_loading(
            geometry=geometry,
            loaded_face_DOF_pairs=loaded_face_DOF_pairs,
            loading_fn=loading_fn,
            constrained_face_DOF_pairs=constrained_face_DOF_pairs
        )
    else:
        _loading_fn = lambda state, t, loading_params: jnp.zeros_like(free_DOF_ids, dtype=float)

    # Total potential energy = U_internal - W_external
    def total_potential_energy(free_DOFs: jnp.ndarray, t: float, control_params: ControlParams) -> float:
        U_int = constrained_energy_fn(free_DOFs, t, control_params)
        F_ext = _loading_fn(None, t, control_params.loading_params)
        W_ext = jnp.dot(F_ext, free_DOFs)
        return U_int - W_ext

    # Capture original geometry for UL reference updates (static closures, never traced)
    _original_centroids = geometry.face_centroids
    _original_cnv = geometry.centroid_node_vectors
    _bond_connectivity = geometry.bond_connectivity  # static NumPy array

    def solve_statics_fn(initial_displacements: jnp.ndarray, control_params: ControlParams) -> SolutionData:
        """Solve for the static equilibrium.

        Args:
            initial_displacements (jnp.ndarray): Initial displacement guess, shape (n_faces, 3).
            control_params (ControlParams): Geometrical + mechanical parameters.

        Returns:
            SolutionData: Equilibrium solution.
        """
        initial_free = initial_displacements.reshape(-1)[free_DOF_ids]
        # backward_reg > 0: Tikhonov-regularize the IMPLICIT-DIFF (adjoint) linear solve so the
        # near-singular / indefinite tangent stiffness (plastic softening -> negative eigenvalues)
        # is lifted to PD before inversion. Forward equilibrium is untouched; only the gradient path.
        _ids = (functools.partial(solve_cg, ridge=backward_reg) if backward_reg > 0 else None)
        solver = LBFGS(fun=total_potential_energy, maxiter=solver_maxiter, tol=solver_tol,
                       implicit_diff_solve=_ids)
        t_array = jnp.linspace(1.0 / num_steps, 1.0, num_steps) if incremental else jnp.array([1.0])

        if not updated_lagrangian:
            def solve_single_step(current_free_DOFs, t):
                result = solver.run(current_free_DOFs, t=t, control_params=control_params)
                fun_val = total_potential_energy(result.params, t, control_params)
                return result.params, (result.params, fun_val)

            _, (history_free, history_energy) = jax.lax.scan(
                solve_single_step, init=initial_free, xs=t_array)

            mapped_kinematics_fn = jax.vmap(kinematics_fn, in_axes=(0, 0, None))
            history_displacement = mapped_kinematics_fn(
                history_free, t_array, control_params.constraint_params)

        else:
            # Updated Lagrangian: after each step, update the reference geometry
            # to the current deformed configuration. Strains are linearized
            # within each increment but the global deformation is captured via
            # the reference update.
            from nff.stages.physics.params import ReferenceGeometry

            # Capture original reference bond vectors — rest lengths must not change.
            _original_ref_bonds = control_params.mechanical_params.bond_params.reference_vector
            _max_nodes = _original_cnv.shape[1]
            # Face indices for each bond endpoint (static, used to look up rotations)
            _bond_face1 = _bond_connectivity[:, 0] // _max_nodes
            _bond_face2 = _bond_connectivity[:, 1] // _max_nodes

            def _update_control_params(accumulated_disp, ctrl):
                """Update reference geometry for the next UL increment.

                Centroids and CNVs follow the full accumulated rigid-body motion.
                Reference bond vectors are ROTATED by the mean face rotation at
                each hinge but keep their original LENGTH, so elastic restoring
                forces accumulate correctly across all increments.
                """
                thetas = accumulated_disp[:, 2]
                cos_t = jnp.cos(thetas)[:, None]
                sin_t = jnp.sin(thetas)[:, None]

                new_centroids = _original_centroids + accumulated_disp[:, :2]
                new_cnv = jnp.stack([
                    cos_t * _original_cnv[:, :, 0] - sin_t * _original_cnv[:, :, 1],
                    sin_t * _original_cnv[:, :, 0] + cos_t * _original_cnv[:, :, 1],
                ], axis=-1)

                # Rotate original rest-length bond vectors by the mean face rotation
                # at each hinge. This tracks the changing hinge orientation (geometric
                # nonlinearity) without resetting the rest length (material linearity).
                mean_theta = (thetas[_bond_face1] + thetas[_bond_face2]) / 2.0
                cos_b = jnp.cos(mean_theta)
                sin_b = jnp.sin(mean_theta)
                new_ref_bonds = jnp.stack([
                    cos_b * _original_ref_bonds[:, 0] - sin_b * _original_ref_bonds[:, 1],
                    sin_b * _original_ref_bonds[:, 0] + cos_b * _original_ref_bonds[:, 1],
                ], axis=-1)

                new_ref_geom = ReferenceGeometry(
                    new_centroids, new_cnv, _bond_connectivity, new_ref_bonds)
                new_bond_params = ctrl.mechanical_params.bond_params._replace(
                    reference_vector=new_ref_bonds)
                new_mech = ctrl.mechanical_params._replace(bond_params=new_bond_params)
                return ctrl._replace(reference_geometry=new_ref_geom, mechanical_params=new_mech)

            delta_t = 1.0 / num_steps  # incremental load fraction per step

            def ul_solve_single_step(carry, t):
                delta_free, accumulated_disp, ctrl = carry

                # Apply only the incremental load delta_t*F_max, not the total t*F_max.
                # The reference is already at equilibrium under F(t-delta_t), so only
                # the load increment needs to be absorbed by the incremental displacement.
                result = solver.run(delta_free, t=delta_t, control_params=ctrl)
                new_delta_free = result.params

                # Incremental face displacement (n_faces, 3)
                delta_face_disp = kinematics_fn(new_delta_free, t, ctrl.constraint_params)

                # Accumulate total displacement from original reference
                new_accumulated = accumulated_disp + delta_face_disp

                new_ctrl = _update_control_params(new_accumulated, ctrl)
                fun_val = total_potential_energy(new_delta_free, delta_t, ctrl)

                return (new_delta_free, new_accumulated, new_ctrl), (new_accumulated, fun_val)

            init_accumulated = jnp.zeros((geometry.n_faces, 3), dtype=float)
            _, (history_displacement, history_energy) = jax.lax.scan(
                ul_solve_single_step,
                init=(initial_free, init_accumulated, control_params),
                xs=t_array,
            )

        return SolutionData(fields=history_displacement, energies=history_energy)

    return solve_statics_fn
