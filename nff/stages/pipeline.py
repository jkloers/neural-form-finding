"""
Forward pipeline for neural form-finding.

The pipeline is composed of three sequential stages:

    Stage 0 — Initial Mapping
        Maps the flat Kirigami tessellation into a target shape.

    Stage 1 — Geometric Validity
        Optimizes face positions to satisfy geometric constraints (no intersections, closed hinges).

    Stage 2 — Static Physics Solver
        Solves static equilibrium by minimizing total potential energy.

The pipeline is fully differentiable end-to-end via JAX.
"""

import jax.numpy as jnp
from typing import Any, Optional

from nff.stages.state import CentroidalState
from nff.stages.geometry import reconstruct_vertices
from nff.stages.mapping import (
    build_mapping_fn, apply_mapping,
    apply_gnn_mapping, apply_direct_mapping, apply_direct_transform_mapping,
    apply_closed_les_mapping,
)
from nff.stages.validity import solve_geometric_validity
from nff.stages.physics.energy import (
    build_potential_energy,
    build_energy_history,
)
from nff.stages.physics.statics import setup_static_solver
from nff.stages.physics.params import ReferenceGeometry, build_control_params, SolutionData
from nff.stages.physics.force_types import (
    has_geometry_dependent_loads,
    build_geometry_dependent_loading,
)
from nff.config.targets import get_target_points
from nff.config.experiment import TargetConfig, PhysicsConfig, ValidityConfig


# ══════════════════════════════════════════════════════════════════════════════
# PRESENTATION-READY MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def forward_pipeline(
        initial_state: CentroidalState,
        target_cfg: TargetConfig,
        validity_cfg: ValidityConfig,
        physics_cfg: PhysicsConfig,
        map_type: str = 'conformal_polynomial',
        map_params: Optional[dict] = None,
        use_shirley_chiu: bool = True,
        strict_boundary_fit: bool = True,
        static_features: Optional[dict] = None,
        load_specs: Optional[list] = None,
        bond_energy_fn=None,
        hinge_w_lig=None) -> dict:
    """Full differentiable pipeline: Initial Mapping → Geometric Validity → Static Equilibrium.

    GRADIENT PATH (Reverse-mode AD):
    Loss → Stage 2 (Physics VJP) → Stage 1 (Validity VJP) → Stage 0 (Mapping VJP) → map_params gradients

    ``bond_energy_fn``: optional Stage-2 single-bond energy override (e.g. the learned hinge-energy
    surrogate). Default ``None`` = the linear-spring ROM, so existing behavior is unchanged.
    """

    # ── Stage 0: Initial Mapping ──
    # Forward: map_params (GNN weights or analytic coeffs) → mapped_state
    # Backward: VJP computes gradients w.r.t map_params
    mapped_state, mapping_fn = _execute_stage0_mapping(
        initial_state, target_cfg, map_type, map_params,
        use_shirley_chiu, strict_boundary_fit, static_features, physics_cfg.domain_restriction
    )

    # ── Stage 1: Geometric Validity ──
    # Forward: mapped_state → valid_state (closes hinges, fixes overlaps, fits boundary)
    # Backward: VJP flows automatically through the L-BFGS solver via jaxopt.custom_vjp
    valid_state = _execute_stage1_validity(
        mapped_state, target_cfg, validity_cfg
    )

    # ── Stage 2: Static Physics Solver ──
    # Forward: valid_state → solution (equilibrium displacements, strain energy)
    # Backward: VJP flows automatically through the physics minimizer
    solution, geometry = _execute_stage2_physics(
        valid_state, physics_cfg, load_specs, bond_energy_fn=bond_energy_fn, hinge_w_lig=hinge_w_lig
    )

    vertices_ref = reconstruct_vertices(
        valid_state.face_centroids, valid_state.centroid_node_vectors)

    return {
        'mapped_state':           mapped_state,
        'valid_state':            valid_state,
        'solution':               solution,
        'vertices_reference':     vertices_ref,
        'reference_bond_vectors': geometry.reference_bond_vectors if geometry else jnp.zeros((0, 2), dtype=float),
        'mapping_fn':             mapping_fn,
        'map_params':             map_params,
    }


# ══════════════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS (The "Dirty" Work)
# ══════════════════════════════════════════════════════════════════════════════

def _execute_stage0_mapping(
        initial_state: CentroidalState,
        target_cfg: TargetConfig,
        map_type: str,
        map_params: Optional[dict],
        use_shirley_chiu: bool,
        strict_boundary_fit: bool,
        static_features: Optional[dict],
        domain_restriction: str):
    
    target_params = {
        'type': target_cfg.type,
        'center': target_cfg.center,
        'radius': target_cfg.radius
    }

    if map_type.startswith('gnn_'):
        if static_features is None:
            from nff.models.graph_builder import build_static_graph_features
            static_features = build_static_graph_features(initial_state)
        mapped_state = apply_gnn_mapping(initial_state, map_params, static_features, map_type=map_type)
        mapping_fn = None
    elif map_type == 'direct_vertices':
        mapped_state = apply_direct_mapping(initial_state, map_params)
        mapping_fn = None
    elif map_type == 'direct_transform':
        mapped_state = apply_direct_transform_mapping(initial_state, map_params)
        mapping_fn = None
    elif map_type == 'closed_les':
        # Closed-state RDPQK builder: design params -> flat sheet via LES.
        # Stage 1 is bypassed (validity_method: none); Stage 2 deploys via physics.
        mapped_state = apply_closed_les_mapping(initial_state, map_params, static_features)
        mapping_fn = None
    else:
        mapping_fn = build_mapping_fn(
            initial_state, target_params,
            map_type=map_type,
            domain_restriction=domain_restriction,
            use_shirley_chiu=use_shirley_chiu,
            strict_boundary_fit=strict_boundary_fit
        )
        mapped_state = apply_mapping(
            initial_state, mapping_fn,
            map_params=map_params,
        )
    return mapped_state, mapping_fn


def _execute_stage1_validity(mapped_state, target_cfg, validity_cfg):
    target_params = {
        'type': target_cfg.type,
        'center': target_cfg.center,
        'radius': target_cfg.radius
    }
    target_cloud = jnp.array(get_target_points(target_params, n_points=200))
    
    if validity_cfg.validity_method == 'none':
        valid_state = mapped_state
    elif validity_cfg.validity_method == 'alternating_projection':
        from nff.stages.projection import solve_alternating_projections
        valid_state = solve_alternating_projections(
            mapped_state, n_iters=validity_cfg.n_proj_iters)
    else:  # 'lbfgs' (default)
        valid_state = solve_geometric_validity(
            mapped_state, target_cloud, validity_cfg=validity_cfg)
        
    return valid_state


def _execute_stage2_physics(valid_state, physics_cfg, load_specs, bond_energy_fn=None, hinge_w_lig=None):
    if not getattr(physics_cfg, 'use_stage2', True):
        n_faces = valid_state.face_centroids.shape[0]
        zero_fields = jnp.zeros((1, n_faces, 3), dtype=float)
        dummy_energies = {k: jnp.zeros(1, dtype=float)
                         for k in ('stretch', 'shear', 'rot', 'contact')}
        solution = SolutionData(fields=zero_fields, energies=dummy_energies)
        return solution, None

    geometry = ReferenceGeometry.from_centroidal_state(valid_state)

    potential_energy_fn = build_potential_energy(
        bond_connectivity=geometry.bond_connectivity,
        linearized_strains=physics_cfg.linearized_strains,
        use_contact=physics_cfg.use_contact,
        bond_energy_fn=bond_energy_fn,
    )

    if has_geometry_dependent_loads(load_specs):
        loaded_face_DOF_pairs, loading_fn, force_vals_jax = build_geometry_dependent_loading(
            load_specs, valid_state.face_centroids)
    else:
        loading_fn = valid_state.get_loading_function()
        loaded_face_DOF_pairs = valid_state.loaded_face_DOF_pairs if loading_fn else None
        force_vals_jax = None

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
        w_lig=hinge_w_lig,          # None -> surrogate uses its fixed closure width (unchanged)
    )
    
    if force_vals_jax is not None:
        control_params = control_params._replace(loading_params={'force_values': force_vals_jax})

    initial_displacements = jnp.zeros((geometry.n_faces, 3), dtype=float)
    solution = solve_statics_fn(initial_displacements=initial_displacements, control_params=control_params)

    energies_dict = build_energy_history(
        solution=solution,
        control_params=control_params,
        linearized_strains=physics_cfg.linearized_strains,
        use_contact=physics_cfg.use_contact,
    )
    solution = solution._replace(energies=energies_dict)

    return solution, geometry
