"""
main_centroidal.py — Unified centroidal form-finding pipeline.

Pipeline:
    Tessellation (numpy)
        → to_centroidal_state()
        → CentroidalState (JAX)
        → Stage 0: Initial Mapping (→ GNN later)
        → Stage 1: Geometric Validity Optimization
        → Stage 2: Static Physics Solver
        → SolutionData
"""

import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax
jax.config.update("jax_enable_x64", True)

import sys
sys.path.append(os.path.abspath('.'))

import jax.numpy as jnp
import matplotlib.pyplot as plt
from dataclasses import dataclass

# Topology
from topology.unit_patterns import unit_RDQK_D, unit_RDQK_0
from topology.builder import build_tessellation

# Target shape
from geometry.target_shape import DEFAULT_TARGET, get_target_points

# Centroidal pipeline
from jax_backend.centroidal.state import CentroidalState
from jax_backend.centroidal.geometry import reconstruct_vertices
from jax_backend.centroidal.pipeline import forward_pipeline


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CentroidalConfig:
    # Tessellation
    width: int = 2
    height: int = 2
    pattern: callable = unit_RDQK_D

    # Target shape
    target_type: str = DEFAULT_TARGET['type']
    target_center: tuple = (0.0, 0.0)
    target_radius: float = DEFAULT_TARGET['radius']

    # Material properties
    k_stretch: float = 10.0
    k_shear: float = 5.0
    k_rot: float = 1.0
    density: float = 1.0

    # Initial mapping
    map_type: str = 'elliptical_grip'
    scale_factor: float = 1.0

    # Geometric validity weights
    w_connectivity: float = 700.0
    w_non_intersection: float = 1000.0
    w_target: float = 1.0
    w_arm_symmetry: float = 1.0

    # Physics
    use_contact: bool = True
    linearized_strains: bool = True


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    config = CentroidalConfig()

    target_params = {
        'type': config.target_type,
        'center': list(config.target_center),
        'radius': config.target_radius,
    }

    # ══════════════════════════════════════════════════════════════════════════
    # 1. Build tessellation (numpy)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"Building tessellation ({config.width}x{config.height})...")
    tessellation = build_tessellation(config.pattern, config.width, config.height)
    print(f"  {len(tessellation.vertices)} vertices, "
          f"{len(tessellation.faces)} faces, "
          f"{len(tessellation.hinges)} hinges, "
          f"{len(tessellation.voids)} voids.")

    # printing tessalation
    print(tessellation.faces)
    print(tessellation.hinges)
    print(tessellation.vertices)

    # ══════════════════════════════════════════════════════════════════════════
    # 2. Configure boundary conditions & material properties
    # ══════════════════════════════════════════════════════════════════════════
    # Material
    tessellation.set_hinge_properties(
        k_stretch=config.k_stretch,
        k_shear=config.k_shear,
        k_rot=config.k_rot)
    tessellation.set_all_faces_properties(density=config.density)

    # Dirichlet BCs: clamp boundary faces
    clamped_ids = tessellation.clamp_boundary_faces()
    print(f"  Clamped {len(clamped_ids)} boundary faces.")

    # Neumann BCs: example — downward force on a central interior face
    all_ids = set(range(len(tessellation.faces)))
    interior_ids = sorted(all_ids - set(clamped_ids))
    if interior_ids:
        loaded_face = interior_ids[len(interior_ids) // 2]
        tessellation.set_face_load(loaded_face, dof_id=1, value=-1.0)
        print(f"  Applied F_y = -1.0 on face {loaded_face}.")

    # ══════════════════════════════════════════════════════════════════════════
    # 3. Export to CentroidalState
    # ══════════════════════════════════════════════════════════════════════════
    print("\nExporting to CentroidalState...")
    cs_dict = tessellation.to_centroidal_state()
    initial_state = CentroidalState(**{k: jnp.array(v) for k, v in cs_dict.items()})

    n_faces = initial_state.face_centroids.shape[0]
    n_hinges = initial_state.hinge_face_pairs.shape[0] // 2  # 2 entries per hinge
    print(f"  n_faces = {n_faces}")
    print(f"  n_hinges = {n_hinges}")
    print(f"  centroid_node_vectors: {initial_state.centroid_node_vectors.shape}")
    print(f"  hinge_node_pairs:      {initial_state.hinge_node_pairs.shape}")
    print(f"  hinge_adj_info:        {initial_state.hinge_adj_info.shape}")
    print(f"  boundary_face_nodes:   {initial_state.boundary_face_node_ids.shape}")
    print(f"  constrained DOFs:      {initial_state.constrained_face_DOF_pairs.shape[0]}")
    print(f"  loaded DOFs:           {initial_state.loaded_face_DOF_pairs.shape[0]}")

    # ══════════════════════════════════════════════════════════════════════════
    # 4. Run the full pipeline
    # ══════════════════════════════════════════════════════════════════════════
    geom_weights = {
        'connectivity': config.w_connectivity,
        'non_intersection': config.w_non_intersection,
        'target': config.w_target,
        'arm_symmetry': config.w_arm_symmetry,
    }

    print("\n" + "=" * 60)
    print("CENTROIDAL PIPELINE")
    print("  Stage 0: Initial Mapping")
    print("  Stage 1: Geometric Validity")
    print("  Stage 2: Static Physics Solver")
    print("=" * 60)

    result = forward_pipeline(
        initial_state=initial_state,
        target_params=target_params,
        map_type=config.map_type,
        scale_factor=config.scale_factor,
        geom_weights=geom_weights,
        use_contact=config.use_contact,
        linearized_strains=config.linearized_strains,
    )

    # ══════════════════════════════════════════════════════════════════════════
    # 5. Results
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "-" * 60)
    print("RESULTS")
    print("-" * 60)

    mapped_state = result['mapped_state']
    valid_state = result['valid_state']
    solution = result['solution']

    print(f"\nStage 0 — Initial Mapping:")
    verts_mapped = reconstruct_vertices(
        mapped_state.face_centroids, mapped_state.centroid_node_vectors)
    print(f"  Mapped vertex range X: [{float(verts_mapped[:,:,0].min()):.4f}, "
          f"{float(verts_mapped[:,:,0].max()):.4f}]")
    print(f"  Mapped vertex range Y: [{float(verts_mapped[:,:,1].min()):.4f}, "
          f"{float(verts_mapped[:,:,1].max()):.4f}]")

    print(f"\nStage 1 — Geometric Validity:")
    verts_valid = reconstruct_vertices(
        valid_state.face_centroids, valid_state.centroid_node_vectors)
    print(f"  Valid vertex range X:  [{float(verts_valid[:,:,0].min()):.4f}, "
          f"{float(verts_valid[:,:,0].max()):.4f}]")
    print(f"  Valid vertex range Y:  [{float(verts_valid[:,:,1].min()):.4f}, "
          f"{float(verts_valid[:,:,1].max()):.4f}]")

    print(f"\nStage 2 — Static Equilibrium:")
    print(f"  Solution fields shape: {solution.fields.shape}")
    print(f"  Max displacement:      {float(jnp.max(jnp.abs(solution.fields[:, :2]))):.6f}")
    print(f"  Max rotation:          {float(jnp.max(jnp.abs(solution.fields[:, 2]))):.6f}")

    print("\nPipeline complete.")
