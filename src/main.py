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
# Ensure src is in path if running from root
if os.path.exists('src'):
    sys.path.append(os.path.abspath('src'))
    
# Add data/library to path (handle both running from root or src)
if os.path.exists('../data/library'):
    sys.path.append(os.path.abspath('../data/library'))
elif os.path.exists('data/library'):
    sys.path.append(os.path.abspath('data/library'))

import jax.numpy as jnp
from jax import vmap
import matplotlib.pyplot as plt
import numpy as np
import copy

# Visualization
from utils.visualization import plot_tessellation

# Builder
from topology.builder import build_tessellation

# Problem definition (Targets, Conditions, Config)
from problem.conditions import (
    apply_boundary_conditions, 
    apply_loads, 
    set_material_properties
)
from problem.loader import load_config

# Centroidal pipeline
from jax_backend.centroidal.state import CentroidalState
from jax_backend.centroidal.geometry import reconstruct_vertices
from jax_backend.centroidal.pipeline import forward_pipeline
from jax_backend.physics_solver.kinematics import rotation_matrix


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Load configuration from YAML
    config_name = "turn3"

    # Handle both running from root or src
    if os.path.exists(f"../data/configs/{config_name}.yaml"):
        config_path = f"../data/configs/{config_name}.yaml"
    else:
        config_path = f"data/configs/{config_name}.yaml"
        
    config = load_config(config_path)
    print(f"Loaded configuration from {config_path}")

    # Create output directory for this run
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    base_output = "../data/outputs" if os.path.exists("../data") else "data/outputs"
    output_dir = os.path.join(base_output, "runs", f"{timestamp}_{config_name}")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
    
    # Save a copy of the config used
    import shutil
    shutil.copy2(config_path, os.path.join(output_dir, "config_used.yaml"))
    print(f"Results will be saved to: {output_dir}")

    target_params = {
        'type': config.target_type,
        'center': list(config.target_center),
        'radius': config.target_radius,
    }

    # ══════════════════════════════════════════════════════════════════════════
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
    cs_dict = tessellation.to_centroidal_state()
    initial_state = CentroidalState(**{k: jnp.array(v) for k, v in cs_dict.items()})

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
        k_contact=config.k_contact,
        min_angle=config.min_angle * jnp.pi / 180.0,
        cutoff_angle=config.cutoff_angle * jnp.pi / 180.0,
        linearized_strains=config.linearized_strains,
    )

    # ══════════════════════════════════════════════════════════════════════════
    # 5. Results & Visualization
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "-" * 60)
    print("RESULTS VISUALIZATION")
    print("-" * 60)

    def plot_stage(state, title, save_figure=False):
        # 1. Reconstruct vertices from centroidal state
        c = state.face_centroids
        s = state.centroid_node_vectors
        verts_rec = reconstruct_vertices(c, s) # (n_faces, max_nodes, 2)
        
        # 2. Update a copy of the tessellation
        tess_copy = copy.deepcopy(tessellation)
        new_verts = np.zeros_like(tess_copy.vertices)
        for i, face in enumerate(tess_copy.faces):
            for j, v_idx in enumerate(face.vertex_indices):
                new_verts[v_idx] = verts_rec[i, j]
        tess_copy.update_vertices(new_verts)
        
        # 3. Plot
        fig, ax = plt.subplots(figsize=(8, 8))
        plot_tessellation(tess_copy, ax=ax, title=title, 
                          show_target=True, target_params=target_params)
        
        if save_figure:
            # Save figure
            filename = title.lower().replace(" ", "_").replace(":", "") + ".png"
            save_path = os.path.join(output_dir, "plots", filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Saved plot to {save_path}")
        plt.show()

    # Stage 0
    print("Displaying Stage 0: Initial Mapping...")
    plot_stage(result['mapped_state'], "Stage 0: Initial Mapping")

    # Stage 1
    print("Displaying Stage 1: Geometric Validity...")
    plot_stage(result['valid_state'], "Stage 1: Geometric Validity")

    # Stage 2
    print("Displaying Stage 2: Static Equilibrium...")
    # Reconstruct equilibrium state from solution
    sol = result['solution']
    valid_state = result['valid_state']
    
    # centroids_eq = centroids_valid + displacement
    c_eq = valid_state.face_centroids + sol.fields[:, :2]
    # s_eq = rotate(s_valid, theta)
    R = vmap(rotation_matrix)(sol.fields[:, 2])
    s_eq = jnp.einsum('nij, nkj -> nki', R, valid_state.centroid_node_vectors)
    
    equilibrium_state = valid_state._replace(face_centroids=c_eq, centroid_node_vectors=s_eq)
    plot_stage(equilibrium_state, "Stage 2: Static Equilibrium")

    print("\nPipeline complete. Let's pat ourselves on the back!")
