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
from jax_backend.initial_map import build_mapping_fn, apply_mapping, apply_gnn_mapping
from jax_backend.validity_solver import solve_geometric_validity
from jax_backend.physics_solver.energy import (
    build_potential_energy,
    build_energy_history,
)
from jax_backend.physics_solver.statics import setup_static_solver
from jax_backend.physics_solver.params import ReferenceGeometry, build_control_params
from jax_backend.physics_solver.force_types import (
    has_geometry_dependent_loads,
    build_geometry_dependent_loading,
)
from problem.targets import get_target_points
from problem.config import TargetConfig, PhysicsConfig, ValidityConfig


def forward_pipeline(
        initial_state: CentroidalState,
        target_cfg: TargetConfig,
        validity_cfg: ValidityConfig,
        physics_cfg: PhysicsConfig,
        map_type: str = 'conformal_polynomial',
        map_params: Optional[jnp.ndarray] = None,
        use_shirley_chiu: bool = True,
        strict_boundary_fit: bool = True,
        static_features=None,
        load_specs: Optional[list] = None) -> dict:
    """Full differentiable pipeline: initial map → geometric validity → static equilibrium.

    Args:
        initial_state: Flat CentroidalState exported from the Tessellation.
        target_cfg:    Structured TargetConfig (type, center, radius).
        physics_cfg:   Structured PhysicsConfig (material, contact, weights, solver).
        map_type:      Initial mapping type.
        map_params:    Differentiable parameters for the mapping.
        use_shirley_chiu: Whether to apply Shirley-Chiu projection.

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
    # Deux branches selon le type de mapping :
    #   • GNN (map_type starts with 'gnn_') : apply_gnn_mapping reçoit le
    #     graphe topologique et prédit de nouvelles positions de centroïdes.
    #     static_features est calculé ici — initial_state est une constante
    #     de closure dans JIT, donc tous ses champs sont concrets.
    #   • Paramétrique classique : pipeline conformal/asymmetric_roots existant.
    # ══════════════════════════════════════════════════════════════════════════
    target_params = {
        'type': target_cfg.type,
        'center': target_cfg.center,
        'radius': target_cfg.radius
    }

    if map_type.startswith('gnn_'):
        # static_features est normalement précomputé AVANT le JIT dans create_train_step.
        # Pour les appels hors-JIT (visualisation finale), on le calcule ici.
        if static_features is None:
            from jax_backend.gnn.graph_builder import build_static_graph_features
            static_features = build_static_graph_features(initial_state)
        mapped_state = apply_gnn_mapping(initial_state, map_params, static_features, map_type=map_type)
        mapping_fn = None
    else:
        mapping_fn = build_mapping_fn(
            initial_state, target_params,
            map_type=map_type,
            domain_restriction=physics_cfg.domain_restriction,
            use_shirley_chiu=use_shirley_chiu,
            strict_boundary_fit=strict_boundary_fit
        )

        mapped_state = apply_mapping(
            initial_state, mapping_fn,
            map_params=map_params,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 1 — Geometric Validity
    # Two implementations, selected by validity_cfg.validity_method:
    #
    #   'lbfgs' (default)              — weighted L-BFGS penalty minimisation.
    #                                    Optimises both centroids and CNVs.
    #
    #   'alternating_projection'       — closed-form alternating projections.
    #                                    Projects CNVs only (centroids fixed).
    #                                    Enforces hinge gaps = 0 and voids
    #                                    are parallelograms exactly, in O(n_iters)
    #                                    instead of O(maxiter × LBFGS).
    # ══════════════════════════════════════════════════════════════════════════
    target_cloud = jnp.array(get_target_points(target_params, n_points=200))
    if getattr(validity_cfg, 'validity_method', 'lbfgs') == 'alternating_projection':
        from jax_backend.projection_solver import solve_alternating_projections
        valid_state = solve_alternating_projections(
            mapped_state, n_iters=validity_cfg.n_proj_iters)
    else:
        valid_state = solve_geometric_validity(
            mapped_state, target_cloud, validity_cfg=validity_cfg)

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 2 — Static Physics Solver
    # ══════════════════════════════════════════════════════════════════════════

    # 2.1 — Geometry instantiation (Factory Pattern)
    # ReferenceGeometry is the single geometry container. bond_connectivity
    # is read from the state as a static NumPy array — never a JAX Tracer.
    geometry = ReferenceGeometry.from_centroidal_state(valid_state)

    # 2.2 — Energy functional construction
    # build_potential_energy composes elastic strain energy and contact energy
    # into a single callable, choosing linearized or nonlinear strains.
    potential_energy_fn = build_potential_energy(
        bond_connectivity=geometry.bond_connectivity,
        linearized_strains=physics_cfg.linearized_strains,
        use_contact=physics_cfg.use_contact,
    )

    # 2.3 — Loading (Neumann BCs)
    # Two paths:
    #   • Typed load specs (tile_to_tile / tess_frame / explicit global_frame):
    #     force directions computed from live centroids → differentiable.
    #     force_vals_jax is threaded through control_params.loading_params so
    #     that jaxopt's custom_vjp solver can differentiate through it as an
    #     explicit argument (closed-over JAX tracers are not supported there).
    #   • Legacy (no 'type' key in load specs): fixed values from valid_state.
    if has_geometry_dependent_loads(load_specs):
        loaded_face_DOF_pairs, loading_fn, force_vals_jax = build_geometry_dependent_loading(
            load_specs, valid_state.face_centroids)
    else:
        loading_fn = valid_state.get_loading_function()
        loaded_face_DOF_pairs = valid_state.loaded_face_DOF_pairs if loading_fn else None
        force_vals_jax = None

    # 2.4 — Static solver setup
    # Wraps the L-BFGS minimizer with the constrained kinematics (Dirichlet BCs)
    # and optional incremental load stepping.
    solve_statics_fn = setup_static_solver(
        geometry=geometry,
        energy_fn=potential_energy_fn,
        loaded_face_DOF_pairs=loaded_face_DOF_pairs,
        loading_fn=loading_fn,
        constrained_face_DOF_pairs=valid_state.constrained_face_DOF_pairs,
        incremental=physics_cfg.incremental,
        num_steps=physics_cfg.num_load_steps,
        solver_maxiter=physics_cfg.solver_maxiter,
        solver_tol=physics_cfg.solver_tol,
        updated_lagrangian=physics_cfg.updated_lagrangian,
    )

    # 2.5 — ControlParams assembly
    # build_control_params takes all mechanical parameters explicitly —
    # no proxy objects or implicit duck-typing.
    control_params = build_control_params(
        geometry=geometry,
        k_stretch=valid_state.k_stretch,
        k_shear=valid_state.k_shear,
        k_rot=valid_state.k_rot,
        density=valid_state.density,
        k_contact=physics_cfg.k_contact,
        min_angle=physics_cfg.min_angle,
        cutoff_angle=physics_cfg.cutoff_angle,
        use_contact=physics_cfg.use_contact,
    )
    # For geometry-dependent loads, force_values is a live JAX array that must
    # be an explicit solver argument (not a closure) so jaxopt's custom_vjp
    # can differentiate through it.  Inject it via loading_params here.
    if force_vals_jax is not None:
        control_params = control_params._replace(loading_params={'force_values': force_vals_jax})

    # 2.6 — Solve for static equilibrium
    # Initial displacement guess is zero (undeformed configuration).
    initial_displacements = jnp.zeros((geometry.n_faces, 3), dtype=float)
    solution = solve_statics_fn(initial_displacements=initial_displacements, control_params=control_params)

    # 2.7 — Energy history decomposition (delegated to energy module)
    # Decomposes the total energy into components (stretch, shear, rot, contact)
    # across all load steps and packages them into a dictionary.
    energies_dict = build_energy_history(
        solution=solution,
        control_params=control_params,
        linearized_strains=physics_cfg.linearized_strains,
        use_contact=physics_cfg.use_contact,
    )
    solution = solution._replace(energies=energies_dict)

    # 2.8 — Reconstruct vertex positions in the reference configuration
    vertices_ref = reconstruct_vertices(
        geometry.face_centroids, geometry.centroid_node_vectors)

    return {
        'mapped_state':           mapped_state,
        'valid_state':            valid_state,
        'solution':               solution,
        'vertices_reference':     vertices_ref,
        'reference_bond_vectors': geometry.reference_bond_vectors,
        'mapping_fn':             mapping_fn,
        'map_params':             map_params,
    }
