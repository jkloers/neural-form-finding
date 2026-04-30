"""
Unified centroidal pipeline: initial map → geometric validity → physics solver.

This module orchestrates the three-stage sequential pipeline:
0. Initial mapping (c, s) → (c_mapped, s_mapped) — spatial deformation to target shape
1. Geometric validity optimization on (c, s) — face shapes & positions
2. Static physics solve for q = [dx, dy, dθ] — face displacements at equilibrium
"""

import jax.numpy as jnp

from jax_backend.centroidal.state import CentroidalState
from jax_backend.centroidal.geometry import reconstruct_vertices, hinge_vertex_positions
from jax_backend.centroidal.initial_map import apply_initial_map
from jax_backend.centroidal.validity_solver import solve_geometric_validity
from jax_backend.pytrees import TessellationGeometry
from jax_backend.physics_solver.statics import setup_static_solver
from jax_backend.physics_solver.kinematics import face_to_node_kinematics, rotation_matrix
from jax_backend.utils.utils import (
    ControlParams, GeometricalParams, MechanicalParams,
    LigamentParams, ContactParams, SolutionData,
)

from jax_backend.physics_solver.energy import (
    build_strain_energy, build_contact_energy, combine_face_energies,
    ligament_energy, ligament_energy_linearized,
)
from problem.targets import get_target_points

def _build_reference_bond_vectors(valid_state: CentroidalState):
    """Compute bond reference vectors from the validated centroidal state.

    Each hinge connects vertex1 (in face1) to vertex2 (in face2).
    hinge_node_pairs has exactly 1 entry per hinge.

    Args:
        valid_state: CentroidalState after geometric validation.

    Returns:
        (n_hinges, 2) — reference bond vectors (from vertex1 to vertex2).
    """
    p1, p2 = hinge_vertex_positions(
        valid_state.face_centroids,
        valid_state.centroid_node_vectors,
        valid_state.hinge_node_pairs,
    )
    # p1 = vertex1 position (from face1), p2 = vertex2 position (from face2)
    # After validity optimization, these should be nearly coincident
    return p2 - p1


def forward_pipeline(
        initial_state: CentroidalState,
        target_params: dict,
        map_type: str = 'elliptical_grip',
        scale_factor: float = 1.0,
        geom_weights: dict = None,
        use_contact: bool = True,
        k_contact: float = 1.,
        min_angle: float = 0.,
        cutoff_angle: float = 5. * jnp.pi / 180,
        linearized_strains: bool = True,
        incremental: bool = False,
        num_load_steps: int = 10) -> dict:
    """Full pipeline: initial map → geometric validity → static physics solver.

    Stage 0: Map flat tessellation into target shape (→ replaced by GNN later)
    Stage 1: Optimize (c, s) for geometric validity
    Stage 2: Build physics reference from (c*, s*), solve for q*

    Args:
        initial_state: CentroidalState from tessellation export (flat geometry).
        target_params: dict with 'type', 'center', 'radius' for the target shape.
        map_type: initial mapping type ('elliptical_grip' or 'boundary_projection').
        scale_factor: scaling applied after initial mapping.
        geom_weights: dict of constraint weights for the geometric optimizer.
        use_contact: whether to include contact energy.
        k_contact, min_angle, cutoff_angle: contact parameters.
        linearized_strains: use linearized strain energy.
        incremental: use incremental loading.
        num_load_steps: number of steps for incremental loading.

    Returns:
        dict with:
            'mapped_state': CentroidalState after initial mapping (Stage 0)
            'valid_state': CentroidalState after geometric optimization (Stage 1)
            'solution': SolutionData with equilibrium fields (n_faces, 3) (Stage 2)
            'vertices_reference': (n_faces, max_nodes, 2) reference vertex positions
            'reference_bond_vectors': (n_hinges, 2)
    """

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 0 — Initial Mapping (to be replaced by GNN)
    # ══════════════════════════════════════════════════════════════════════════
    mapped_state = apply_initial_map(
        initial_state, target_params,
        map_type=map_type, scale_factor=scale_factor)

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 1 — Geometric Validity
    # ══════════════════════════════════════════════════════════════════════════
    target_cloud = jnp.array(get_target_points(target_params, n_points=200))
    valid_state = solve_geometric_validity(
        mapped_state, target_cloud, weights=geom_weights)

    # # ══════════════════════════════════════════════════════════════════════════
    # # Stage 2 — Physics Solver
    # # ══════════════════════════════════════════════════════════════════════════

    # Build reference geometry from validated (c*, s*)
    _face_centroids = valid_state.face_centroids
    _cnv = valid_state.centroid_node_vectors
    _reference_bond_vectors = _build_reference_bond_vectors(valid_state)
    
    # Correct node indexing for smap.bond: face_id * n_nodes + local_node_id
    hnp = valid_state.hinge_node_pairs
    n_faces, n_nodes, _ = _cnv.shape
    _bond_connectivity = jnp.stack([
        hnp[:, 0, 0] * n_nodes + hnp[:, 0, 1],
        hnp[:, 1, 0] * n_nodes + hnp[:, 1, 1]
    ], axis=1)

    # Build Geometry object for the physics solver
    tess_dict = {
        'face_centroids': _face_centroids,
        'centroid_node_vectors': _cnv,
        'bond_connectivity': _bond_connectivity,
        'reference_bond_vectors': _reference_bond_vectors,
    }
    geometry = TessellationGeometry.from_dict(tess_dict)

    # Energy
    bond_energy_fn = ligament_energy_linearized if linearized_strains else ligament_energy
    strain_energy = build_strain_energy(
        bond_connectivity=_bond_connectivity,
        bond_energy_fn=bond_energy_fn,
    )
    if use_contact:
        contact_energy = build_contact_energy(bond_connectivity=_bond_connectivity)
        potential_energy = combine_face_energies(strain_energy, contact_energy)
    else:
        potential_energy = strain_energy

    # Loading
    loaded_pairs = valid_state.loaded_face_DOF_pairs
    if len(loaded_pairs) > 0:
        _force_values = valid_state.load_values
        loading_fn = lambda state, t, **kwargs: t * _force_values
    else:
        loaded_pairs = None
        loading_fn = None

    # Solver
    solve_statics = setup_static_solver(
        geometry=geometry,
        energy_fn=potential_energy,
        loaded_face_DOF_pairs=loaded_pairs,
        loading_fn=loading_fn,
        constrained_face_DOF_pairs=valid_state.constrained_face_DOF_pairs,
        incremental=incremental,
        num_steps=num_load_steps
    )

    # Control params
    control_params = ControlParams(
        geometrical_params=GeometricalParams(
            face_centroids=_face_centroids,
            centroid_node_vectors=_cnv,
            bond_connectivity=_bond_connectivity,
            reference_bond_vectors=_reference_bond_vectors,
        ),
        mechanical_params=MechanicalParams(
            bond_params=LigamentParams(
                k_stretch=valid_state.k_stretch,
                k_shear=valid_state.k_shear,
                k_rot=valid_state.k_rot,
                reference_vector=_reference_bond_vectors,
            ),
            density=valid_state.density,
            contact_params=ContactParams(
                k_contact=k_contact,
                min_angle=min_angle,
                cutoff_angle=cutoff_angle,
            ) if use_contact else None,
        ),
        constraint_params=dict(),
    )

    state0 = jnp.zeros((n_faces, 3), dtype=float)
    solution = solve_statics(state0=state0, control_params=control_params)

    vertices_ref = reconstruct_vertices(_face_centroids, _cnv)

    return {
        'mapped_state': mapped_state,
        'valid_state': valid_state,
        'solution': solution,
        'vertices_reference': vertices_ref,
        'reference_bond_vectors': _reference_bond_vectors,
    }
