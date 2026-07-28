"""Displacement-controlled actuation for the Stage-2 static solver.

Force control prescribes the load and solves for the motion; displacement control prescribes the
motion and reads back the force. Both are Stage-2 Dirichlet/Neumann boundary conditions, and the
solver already supports both: ``build_constrained_kinematics`` accepts a
``constrained_DOFs_fn(t, **params)`` supplying imposed values for the constrained DOFs, of which a
clamp is simply the case value = 0.

This module turns a ``displacement_control:`` config block into

  1. the augmented Dirichlet set — the clamped rows the tessellation already carries, with one row
     appended per prescribed DOF, and
  2. the ramp ``t -> t * u_target``, so imposed motion is applied in step with the incremental load
     stepping and reaches its full value at t = 1.

The reaction force at a prescribed DOF is ``dU_int/du`` at equilibrium: the external force that must
be supplied to hold the imposed motion. That readout is the point of displacement control — the
force becomes an output instead of an input, which is what a testing machine measures.
"""

from typing import Callable, NamedTuple, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int


DOF_NAMES = ('dx', 'dy', 'dtheta')


class PrescribedDisplacement(NamedTuple):
    """One imposed rigid-body DOF value on one face, reached at t = 1.

    Attrs:
        face:  face index.
        dof:   0 = dx, 1 = dy, 2 = dtheta.
        value: pipeline length units (dof 0, 1) or radians (dof 2).
    """
    face: int
    dof: int
    value: float


class DisplacementControl(NamedTuple):
    """Everything Stage 2 needs to run under (partial) displacement control.

    Attrs:
        constrained_face_DOF_pairs: (n_constraints, 2) clamped rows followed by prescribed rows.
        constrained_DOFs_fn:        (t, **params) -> (n_constraints,) imposed values at load step t.
        prescribed_face_DOF_pairs:  (n_prescribed, 2) the prescribed rows alone, for reaction readout.
        prescribed_values:          (n_prescribed,) imposed values at full deployment (t = 1).
    """
    constrained_face_DOF_pairs: Int[np.ndarray, "n_constraints 2"]
    constrained_DOFs_fn: Callable
    prescribed_face_DOF_pairs: Int[np.ndarray, "n_prescribed 2"]
    prescribed_values: Float[np.ndarray, "n_prescribed"]


class ReactionReport(NamedTuple):
    """Support forces sustaining an imposed motion, as returned by the pipeline.

    Attrs:
        values:            (n_steps, n_prescribed) reaction at each load step, or (n_prescribed,).
        face_DOF_pairs:    (n_prescribed, 2) rows [face_id, DOF_id] the reactions belong to.
        prescribed_values: (n_prescribed,) imposed values at full deployment (t = 1).
    """
    values: Float[Array, "..."]
    face_DOF_pairs: Int[np.ndarray, "n_prescribed 2"]
    prescribed_values: Float[np.ndarray, "n_prescribed"]


def parse_displacement_specs(specs: Optional[Sequence[dict]]) -> tuple[PrescribedDisplacement, ...]:
    """Expand raw config specs into one `PrescribedDisplacement` per (face, DOF).

    Spec format — ``face`` may be a list, so one entry describes a whole edge::

        - {face: [2, 5, 8], dof: 1, value: 0.46}     # imposed dy, pipeline length units
        - {face: 4, dof: 2, value_deg: 30.0}         # imposed rotation, degrees

    Args:
        specs: raw spec dicts, or None.

    Returns:
        One entry per (face, DOF); empty when nothing is prescribed.
    """
    prescribed: list[PrescribedDisplacement] = []
    seen: set[tuple[int, int]] = set()

    for spec in (specs or ()):
        dof = int(spec['dof'])
        if dof not in (0, 1, 2):
            raise ValueError(f"displacement_control: dof must be 0, 1 or 2 — got {dof}.")

        has_rad, has_deg = 'value' in spec, 'value_deg' in spec
        if has_rad == has_deg:
            raise ValueError(
                "displacement_control: give exactly one of 'value' or 'value_deg' per entry.")
        if has_deg and dof != 2:
            raise ValueError(
                f"displacement_control: 'value_deg' applies to dof 2 (rotation) only — "
                f"use 'value' for the translational dof {dof} ({DOF_NAMES[dof]}).")
        value = float(np.deg2rad(float(spec['value_deg']))) if has_deg else float(spec['value'])

        faces = spec['face']
        faces = faces if isinstance(faces, (list, tuple, np.ndarray)) else [faces]
        for face in faces:
            key = (int(face), dof)
            if key in seen:
                raise ValueError(
                    f"displacement_control: face {key[0]} {DOF_NAMES[dof]} is prescribed twice.")
            seen.add(key)
            prescribed.append(PrescribedDisplacement(int(face), dof, value))

    return tuple(prescribed)


def build_displacement_control(
        specs: Optional[Sequence[dict]],
        constrained_face_DOF_pairs: Int[np.ndarray, "n_clamped 2"],
        loaded_face_DOF_pairs: Optional[Int[np.ndarray, "n_loaded 2"]] = None,
) -> Optional[DisplacementControl]:
    """Build the augmented Dirichlet set and the imposed-value ramp.

    A prescribed DOF may be neither already clamped (the clamp would win and silently pin it at 0)
    nor externally loaded (a force on a constrained DOF does no work and is dropped by
    ``build_loading``). Both cases raise rather than fail quietly.

    Args:
        specs: raw `displacement_control` spec dicts, or None.
        constrained_face_DOF_pairs: (n_clamped, 2) the tessellation's existing Dirichlet rows.
        loaded_face_DOF_pairs: (n_loaded, 2) externally loaded rows, checked for conflicts.

    Returns:
        `DisplacementControl`, or None when nothing is prescribed (force control — the caller
        keeps its existing clamped-only constraint set).
    """
    prescribed = parse_displacement_specs(specs)
    if not prescribed:
        return None

    clamped = np.asarray(constrained_face_DOF_pairs, dtype=np.int32).reshape(-1, 2)
    loaded = (np.zeros((0, 2), dtype=np.int32) if loaded_face_DOF_pairs is None
              else np.asarray(loaded_face_DOF_pairs, dtype=np.int32).reshape(-1, 2))
    clamped_set = {(int(f), int(d)) for f, d in clamped}
    loaded_set = {(int(f), int(d)) for f, d in loaded}

    for p in prescribed:
        if (p.face, p.dof) in clamped_set:
            raise ValueError(
                f"displacement_control: face {p.face} {DOF_NAMES[p.dof]} is both clamped and "
                f"prescribed. Release it in boundary_conditions (clamped_faces / clamped_dofs) "
                f"to put it under displacement control.")
        if (p.face, p.dof) in loaded_set:
            raise ValueError(
                f"displacement_control: face {p.face} {DOF_NAMES[p.dof]} carries both an external "
                f"load and a prescribed displacement. A force on a constrained DOF does no work — "
                f"prescribe the motion or apply the force, not both.")

    prescribed_pairs = np.array([[p.face, p.dof] for p in prescribed], dtype=np.int32)
    prescribed_values = np.array([p.value for p in prescribed], dtype=float)

    # Row order is load-bearing: `build_constrained_kinematics` scatters the value vector onto
    # `constrained_face_DOF_pairs` in exactly this order. Clamps stay 0 for all t; only the
    # appended prescribed rows ramp.
    pairs = np.vstack([clamped, prescribed_pairs]).astype(np.int32)
    # NUMPY, deliberately. This function is closed over by the Stage-2 objective, and jaxopt's
    # custom_vjp cannot capture a closed-over JAX tracer — and `jnp.asarray` called inside a trace
    # produces exactly that (same reason `force_vals_jax` travels through control_params). A NumPy
    # array closes over as a compile-time XLA constant instead. Imposed values come from config and
    # never need gradients; making the actuation itself learnable would mean threading it through
    # `control_params.constraint_params` as an explicit solver argument.
    values = np.concatenate([np.zeros(len(clamped), dtype=float), prescribed_values])

    def constrained_DOFs_fn(t: float, **kwargs) -> Float[Array, "n_constraints"]:
        """Imposed Dirichlet values at load step t: clamps held at 0, prescribed DOFs ramped."""
        return t * values

    return DisplacementControl(
        constrained_face_DOF_pairs=pairs,
        constrained_DOFs_fn=constrained_DOFs_fn,
        prescribed_face_DOF_pairs=prescribed_pairs,
        prescribed_values=prescribed_values,
    )


def compute_reaction_forces(
        energy_fn: Callable,
        face_displacements: Float[Array, "..."],
        control_params,
        face_DOF_pairs: Int[np.ndarray, "n_pairs 2"],
) -> Float[Array, "..."]:
    """Reaction forces at the given DOFs — the readout displacement control exists for.

    At equilibrium the free DOFs satisfy ``dU_int/du = F_ext``; the same derivative taken on a
    CONSTRAINED DOF is the force the support must supply, positive along the +DOF direction
    (a moment for dof 2).

    Args:
        energy_fn: internal potential energy, (face_displacement (n_faces, 3), control_params) -> scalar.
        face_displacements: (n_faces, 3) equilibrium field, or (n_steps, n_faces, 3) history.
        control_params: the `ControlParams` the solve used.
        face_DOF_pairs: (n_pairs, 2) rows [face_id, DOF_id] to read the reaction at.

    Returns:
        (n_pairs,) for a single field, (n_steps, n_pairs) for a history.
    """
    pairs = np.asarray(face_DOF_pairs, dtype=np.int32).reshape(-1, 2)
    face_ids, dof_ids = pairs[:, 0], pairs[:, 1]
    grad_energy_fn = jax.grad(energy_fn)

    def reaction_at(displacement: Float[Array, "n_faces 3"]) -> Float[Array, "n_pairs"]:
        return grad_energy_fn(displacement, control_params)[face_ids, dof_ids]

    displacements = jnp.asarray(face_displacements)
    return (jax.vmap(reaction_at)(displacements) if displacements.ndim == 3
            else reaction_at(displacements))
