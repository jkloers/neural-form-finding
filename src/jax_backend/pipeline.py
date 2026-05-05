"""
Forward pipeline for neural form-finding.

The pipeline is composed of three sequential stages:

    Stage 0 — Initial Mapping
        Maps the flat Kirigami tessellation into a target shape using a
        conformal-like polynomial mapping. This stage will eventually be
        replaced by a Graph Neural Network (GNN) that learns the mapping
        directly from data.

    Stage 1 — Geometric Validity
        Optimizes the centroidal coordinates (face positions and shapes) to
        satisfy geometric constraints: connectivity, non-intersection, target
        boundary fitting, and arm symmetry. This is a differentiable
        optimization via gradient descent.

    Stage 2 — Static Physics Solver
        Given the geometrically valid configuration, sets up and solves the
        static equilibrium problem by minimizing the total potential energy
        (elastic strain energy + contact energy − external work) using L-BFGS.

The pipeline is fully differentiable end-to-end via JAX, enabling
gradient-based optimization of the mapping parameters.
"""

import jax.numpy as jnp
from typing import Optional

from jax_backend.state import CentroidalState
from jax_backend.geometry import reconstruct_vertices
from jax_backend.initial_map import apply_initial_map
from jax_backend.validity_solver import solve_geometric_validity
from jax_backend.pytrees import TessellationGeometry
from jax_backend.physics_solver.energy import (
    ligament_energy, ligament_energy_linearized,
    build_strain_energy, build_contact_energy, combine_face_energies,
    build_energy_history,
)
from jax_backend.physics_solver.statics import setup_static_solver


def get_target_points(target_params: dict, n_points: int = 200) -> jnp.ndarray:
    """Samples a point cloud on the boundary of the target shape.

    Args:
        target_params: dict with keys 'type', 'center', 'radius'.
        n_points: number of boundary sample points.

    Returns:
        jnp.ndarray of shape (n_points, 2).
    """
    t = jnp.linspace(0, 2 * jnp.pi, n_points)
    if target_params['type'] == 'circle':
        r = target_params['radius']
        c = jnp.array(target_params['center'])
        return c + r * jnp.stack([jnp.cos(t), jnp.sin(t)], axis=1)
    # Additional shapes (heart, ellipse, …) can be added here.
    return jnp.zeros((n_points, 2))


def forward_pipeline(
        initial_state: CentroidalState,
        target_params: dict,
        map_type: str = 'elliptical_grip',
        scale_factor: float = 1.0,
        map_params: Optional[dict] = None,
        geom_weights: Optional[dict] = None,
        use_contact: bool = True,
        k_contact: float = 1.0,
        min_angle: float = 0.1 * jnp.pi / 180,
        cutoff_angle: float = 5. * jnp.pi / 180,
        linearized_strains: bool = True,
        incremental: bool = False,
        num_load_steps: int = 10) -> dict:
    """Full differentiable pipeline: initial map → geometric validity → static equilibrium.

    Args:
        initial_state: Flat CentroidalState exported from the Tessellation.
        target_params: Target shape config dict ('type', 'center', 'radius').
        map_type: Initial mapping type ('elliptical_grip' or 'boundary_projection').
        scale_factor: Scaling applied after initial mapping.
        map_params: Optional polynomial coefficients for the conformal mapping.
        geom_weights: Constraint weights for the geometric validity optimizer.
        use_contact: Whether to include contact/non-intersection energy.
        k_contact: Contact stiffness coefficient.
        min_angle: Minimum void angle before contact activates.
        cutoff_angle: Void angle at which contact energy saturates.
        linearized_strains: Use linearized (vs. nonlinear) strain measures.
        incremental: Use incremental load stepping (for large deformations).
        num_load_steps: Number of load steps for incremental solving.

    Returns:
        dict with keys:
            'mapped_state'          — CentroidalState after Stage 0
            'valid_state'           — CentroidalState after Stage 1
            'solution'              — SolutionData with equilibrium fields + energy history
            'vertices_reference'    — (n_faces, n_nodes, 2) reference vertex positions
            'reference_bond_vectors'— (n_hinges, 2) reference ligament vectors
    """

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 0 — Initial Mapping
    # Maps the flat tessellation into the target shape using a conformal
    # polynomial (or other parametric) mapping. `map_params` are the
    # differentiable variables being optimized during training.
    # ══════════════════════════════════════════════════════════════════════════
    mapped_state = apply_initial_map(
        initial_state, target_params,
        map_type=map_type,
        scale_factor=scale_factor,
        map_params=map_params,
    )

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 1 — Geometric Validity
    # Optimizes face centroids and shapes to satisfy: connectivity (hinges
    # must share a point), non-intersection, target fitting, and symmetry.
    # ══════════════════════════════════════════════════════════════════════════
    target_cloud = jnp.array(get_target_points(target_params, n_points=200))
    valid_state = solve_geometric_validity(
        mapped_state, target_cloud, weights=geom_weights)

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 2 — Static Physics Solver
    # ══════════════════════════════════════════════════════════════════════════

    # 2.1 — Geometry instantiation (Factory Pattern)
    # TessellationGeometry wraps the physical reference configuration.
    # bond_connectivity is read from the state where it was pre-computed
    # as a static NumPy array, ensuring JAX never sees it as a Tracer.
    geometry = TessellationGeometry.from_centroidal_state(valid_state)

    # 2.2 — Energy functional construction
    # The potential energy is the sum of:
    #   - Elastic strain energy (axial + shear + rotational, per ligament)
    #   - Contact energy (penalizes face interpenetration via void angles)
    bond_energy_fn = ligament_energy_linearized if linearized_strains else ligament_energy
    strain_energy = build_strain_energy(
        bond_connectivity=geometry.bond_connectivity(),
        bond_energy_fn=bond_energy_fn,
    )
    if use_contact:
        contact_energy = build_contact_energy(bond_connectivity=geometry.bond_connectivity())
        potential_energy = combine_face_energies(strain_energy, contact_energy)
    else:
        potential_energy = strain_energy

    # 2.3 — Loading (Neumann BCs)
    # Returns a callable t → force values, or None if no loads are defined.
    loading_fn = valid_state.get_loading_function()

    # 2.4 — Static solver setup
    # Wraps the L-BFGS minimizer with the constrained kinematics (Dirichlet BCs)
    # and optional incremental load stepping.
    solve_statics = setup_static_solver(
        geometry=geometry,
        energy_fn=potential_energy,
        loaded_face_DOF_pairs=valid_state.loaded_face_DOF_pairs if loading_fn else None,
        loading_fn=loading_fn,
        constrained_face_DOF_pairs=valid_state.constrained_face_DOF_pairs,
        incremental=incremental,
        num_steps=num_load_steps,
    )

    # 2.5 — ControlParams assembly (delegated to TessellationGeometry)
    # Encapsulates the mapping (geometry, state) → ControlParams so that
    # this pipeline does not need to know the internal structure of either.
    control_params = geometry.build_control_params(
        state=valid_state,
        k_contact=k_contact,
        min_angle=min_angle,
        cutoff_angle=cutoff_angle,
        use_contact=use_contact,
    )

    # 2.6 — Solve for static equilibrium
    # Initial displacement guess is zero (undeformed configuration).
    state0 = jnp.zeros((geometry.n_faces, 3), dtype=float)
    solution = solve_statics(state0=state0, control_params=control_params)

    # 2.7 — Energy history decomposition (delegated to energy module)
    # Decomposes the total energy into components (stretch, shear, rot, contact)
    # across all load steps and packages them into a dictionary.
    energies_dict = build_energy_history(
        solution=solution,
        control_params=control_params,
        linearized_strains=linearized_strains,
        use_contact=use_contact,
    )
    solution = solution._replace(energies=energies_dict)

    # 2.8 — Reconstruct vertex positions in the reference configuration
    vertices_ref = reconstruct_vertices(
        geometry.face_centroids(), geometry.centroid_node_vectors())

    return {
        'mapped_state':           mapped_state,
        'valid_state':            valid_state,
        'solution':               solution,
        'vertices_reference':     vertices_ref,
        'reference_bond_vectors': geometry.reference_bond_vectors(),
    }
