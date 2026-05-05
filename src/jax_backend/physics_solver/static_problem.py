"""
StaticForwardProblem — static version of ForwardProblem.

Key differences vs. the dynamic ForwardProblem:
  - No damping, simulation_time, n_timepoints, loading_rate, input_delay, etc.
  - State = (n_faces, 3) = [dx, dy, d_theta], no velocities
  - Solver: energy minimization (jax.scipy BFGS) instead of odeint
  - SolutionData.fields: shape (n_faces, 3)

Input: dict from Tessellation.to_jax_state_centroidal()
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import jax.numpy as jnp

from jax_backend.physics_solver.params import (
    GeometricalParams,
    ControlParams,
    MechanicalParams,
    LigamentParams,
    ContactParams,
    SolutionData,
    build_control_params,
)


@dataclass
class StaticForwardProblem:
    """
    Static forward problem for a rigid-face assembly.

    Built directly from the dict returned by ``Tessellation.to_jax_state_centroidal()``.
    All topology, geometry, stiffness, density, and constraint data come from that dict.

    Optionally, mechanical params and loading can be overridden at construction.

    State:  (n_faces, 3) = [dx, dy, d_theta] — positions only, no velocities.
    """

    # ── The tessellation export dict (the single source of truth) ─────────────
    tess_dict: dict = field(repr=False)

    # ── Overrides (optional — default to values from tess_dict) ───────────────
    k_stretch: Optional[Any] = None
    k_shear: Optional[Any] = None
    k_rot: Optional[Any] = None
    density: Optional[Any] = None

    # ── Static loading (optional) ─────────────────────────────────────────────
    loaded_face_DOF_pairs: Optional[Any] = None  # (n_loaded, 2)
    loading_fn: Optional[Callable] = None         # (state, t, **params) -> forces

    # ── Contact ───────────────────────────────────────────────────────────────
    use_contact: bool = True
    k_contact: Any = 1.
    min_angle: Any = 0.
    cutoff_angle: Any = 5. * jnp.pi / 180

    # ── Solver options ────────────────────────────────────────────────────────
    linearized_strains: bool = True
    incremental: bool = False
    num_load_steps: int = 10

    # ── Runtime state ─────────────────────────────────────────────────────────
    solution_data: Optional[SolutionData] = None
    is_setup: bool = False
    name: str = "static_forward_problem"

    def setup(self) -> None:
        """Compile the static solver. Must be called before ``self.solve()``."""

        td = self.tess_dict

        # Build geometry from the tessellation export dict.
        # bond_connectivity is kept as a static NumPy array in GeometricalParams.
        geometry = GeometricalParams.from_dict(td)

        # Resolve mechanical params (override or from tess_dict)
        _k_stretch = jnp.array(self.k_stretch if self.k_stretch is not None else td['k_stretch'])
        _k_shear   = jnp.array(self.k_shear   if self.k_shear   is not None else td['k_shear'])
        _k_rot     = jnp.array(self.k_rot     if self.k_rot     is not None else td['k_rot'])
        _density   = jnp.array(self.density   if self.density   is not None else td['density'])

        constrained_face_DOF_pairs = jnp.array(td['constrained_face_DOF_pairs'])

        # Loading from tessellation (Neumann BCs)
        _loaded_pairs = td.get('loaded_face_DOF_pairs')
        _load_values  = td.get('load_values')
        
        if self.loaded_face_DOF_pairs is not None:
            loaded_pairs = jnp.array(self.loaded_face_DOF_pairs)
            loading_fn = self.loading_fn
        elif _loaded_pairs is not None and len(_loaded_pairs) > 0:
            loaded_pairs = jnp.array(_loaded_pairs)
            _force_values = jnp.array(_load_values)
            loading_fn = lambda state, t, **kwargs: t * _force_values
        else:
            loaded_pairs = None
            loading_fn = None

        # Energy functional
        from jax_backend.physics_solver.energy import (
            build_strain_energy, build_contact_energy,
            combine_face_energies, ligament_energy, ligament_energy_linearized,
        )
        from jax_backend.physics_solver.statics import setup_static_solver

        strain_energy = build_strain_energy(
            bond_connectivity=geometry.bond_connectivity,
            bond_energy_fn=(ligament_energy_linearized
                            if self.linearized_strains else ligament_energy),
        )

        if self.use_contact:
            contact_energy = build_contact_energy(bond_connectivity=geometry.bond_connectivity)
            potential_energy = combine_face_energies(strain_energy, contact_energy)
        else:
            potential_energy = strain_energy

        # Static solver
        solve_statics = setup_static_solver(
            geometry=geometry,
            energy_fn=potential_energy,
            loaded_face_DOF_pairs=loaded_pairs,
            loading_fn=loading_fn,
            constrained_face_DOF_pairs=constrained_face_DOF_pairs,
            incremental=self.incremental,
            num_steps=self.num_load_steps
        )

        state0 = jnp.zeros((geometry.n_faces, 3), dtype=float)

        # Build a mock state for build_control_params using td values
        class _StateProxy:
            k_stretch = _k_stretch
            k_shear   = _k_shear
            k_rot     = _k_rot
            density   = _density

        def forward() -> SolutionData:
            control_params = build_control_params(
                geometry=geometry,
                state=_StateProxy(),
                k_contact=self.k_contact,
                min_angle=self.min_angle,
                cutoff_angle=self.cutoff_angle,
                use_contact=self.use_contact,
            )
            return solve_statics(state0=state0, control_params=control_params)

        self.solve = forward
        self.geometry = geometry
        self.is_setup = True
