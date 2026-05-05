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
import jax.numpy as jnp
from jax import vmap
import matplotlib.pyplot as plt
import numpy as np
import copy

# Add src to path for imports
sys.path.append(os.path.abspath('src'))

# Visualization
from src.utils.visualization import plot_tessellation

# Builder
from src.topology.builder import build_tessellation

# Problem definition (Targets, Conditions, Config)
from src.problem.conditions import (
    apply_boundary_conditions, 
    apply_loads, 
    set_material_properties
)
from src.problem.config import load_config

# Centroidal pipeline
from src.jax_backend.state import CentroidalState
from src.jax_backend.geometry import reconstruct_vertices
from src.jax_backend.pipeline import forward_pipeline
from src.jax_backend.physics_solver.kinematics import rotation_matrix

from src.utils.pipeline_viz import visualize_pipeline_results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Load configuration from YAML
    config_dir = "complex_mapping"
    config_name = "5_cs_asy_complex"


    # Paths are now simple as we run from root
    config_path = f"data/configs/{config_dir}/{config_name}.yaml"
    config = load_config(config_path)
    print(f"Loaded configuration from {config_path}")

    # # Create output directory for this run
    # import datetime
    # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # output_dir = os.path.join("data/outputs", "runs", f"{timestamp}_{config_name}")
    
    # os.makedirs(output_dir, exist_ok=True)
    # os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
    
    # # Save a copy of the config used
    # import shutil
    # shutil.copy2(config_path, os.path.join(output_dir, "config_used.yaml"))
    # print(f"Results will be saved to: {output_dir}")

    target_params = {
        'type': config.target_type,
        'center': list(config.target_center),
        'radius': config.target_radius,
    }

    # # ══════════════════════════════════════════════════════════════════════════
    # 1. Build tessellation (numpy)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("TESS / PHYSICS PIPELINE")
    print("=" * 60)
    print(f"Building tessellation ({config.width}x{config.height})...")
    tessellation = build_tessellation(config.pattern, config.width, config.height)
    print(f"  {len(tessellation.vertices)} vertices, "
          f"{len(tessellation.faces)} faces, "
          f"{len(tessellation.hinges)} hinges, "
          f"{len(tessellation.voids)} voids.")

    # ══════════════════════════════════════════════════════════════════════════
    # 2. Configure boundary conditions & material properties
    # ══════════════════════════════════════════════════════════════════════════
    print("\nSetting material properties and boundary conditions...")
    set_material_properties(tessellation, config)

    clamped_ids = apply_boundary_conditions(tessellation, config)
    print(f"  Clamped {len(clamped_ids)} faces ({config.bc_clamped}).")

    applied_loads = apply_loads(tessellation, config)
    for fid, dof, val in applied_loads:
        dof_name = ["Fx", "Fy", "Mz"][dof]
        print(f"  Applied {dof_name} = {val} on face {fid}.")

    # ══════════════════════════════════════════════════════════════════════════
    # 3. Export to CentroidalState
    # ══════════════════════════════════════════════════════════════════════════
    print("\nExporting to CentroidalState...")
    initial_state = CentroidalState.from_tessellation(tessellation)

    n_faces = initial_state.face_centroids.shape[0]
    n_hinges = initial_state.hinge_face_pairs.shape[0]
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
        'void_length': config.w_void_length,
        'void_collinear': config.w_void_collinear,
        'boundary_rigidity': config.w_boundary_rigidity,
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
        map_params=jnp.array(config.map_params),
        scale_factor=config.scale_factor,
        geom_weights=geom_weights,
        use_contact=config.use_contact,
        k_contact=config.k_contact,
        min_angle=config.min_angle * jnp.pi / 180.0,
        cutoff_angle=config.cutoff_angle * jnp.pi / 180.0,
        linearized_strains=config.linearized_strains,
        incremental=config.incremental,
        num_load_steps=config.num_load_steps,
    )

    # ══════════════════════════════════════════════════════════════════════════
    # 5. Results & Visualization
    # ══════════════════════════════════════════════════════════════════════════
    print("\nVisualizing results...")
    visualize_pipeline_results(result, tessellation, config, target_params, config_name)
    print("\n" + "=" * 60)
    print("Pipeline complete. Let's all pat ourselves on the back! 💕")
    print("=" * 60)
