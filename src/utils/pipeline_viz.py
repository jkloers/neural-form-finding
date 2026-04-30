import os
import copy
import numpy as np
import matplotlib.pyplot as plt

import jax.numpy as jnp
from jax import vmap

from jax_backend.physics_solver.kinematics import rotation_matrix
from jax_backend.centroidal.geometry import reconstruct_vertices
from src.utils.visualization import plot_tessellation, animate_tessellation

def visualize_pipeline_results(result, tessellation, config, target_params, config_name):
    """
    Orchestrates the visualization of the entire pipeline, including static plots and animations.
    Controlled by the visualization settings in the config.
    """
    output_dir = "data/outputs/runs"
    plots_dir = os.path.join(output_dir, "plots")
    if config.save_plots:
        os.makedirs(plots_dir, exist_ok=True)

    def plot_stage(state, title, show=True, save=False):
        c = state.face_centroids
        s = state.centroid_node_vectors
        verts_rec = reconstruct_vertices(c, s)
        
        tess_copy = copy.deepcopy(tessellation)
        new_verts = np.zeros_like(tess_copy.vertices)
        for i, face in enumerate(tess_copy.faces):
            for j, v_idx in enumerate(face.vertex_indices):
                new_verts[v_idx] = verts_rec[i, j]
        tess_copy.update_vertices(new_verts)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        plot_tessellation(tess_copy, ax=ax, title=title, 
                          show_target=True, target_params=target_params)
        
        if save:
            filename = title.lower().replace(" ", "_").replace(":", "") + ".png"
            save_path = os.path.join(plots_dir, filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Saved plot to {save_path}")
        if show:
            plt.show()
        else:
            plt.close(fig)

    # Stage 0
    if config.show_stage0 or config.save_plots:
        print("Displaying Stage 0: Initial Mapping...")
        plot_stage(result['mapped_state'], "Stage 0: Initial Mapping", 
                   show=config.show_stage0, save=config.save_plots)

    # Stage 1
    if config.show_stage1 or config.save_plots:
        print("Displaying Stage 1: Geometric Validity...")
        plot_stage(result['valid_state'], "Stage 1: Geometric Validity", 
                   show=config.show_stage1, save=config.save_plots)

    # Stage 2
    if config.show_stage2 or config.save_plots:
        print("Displaying Stage 2: Static Equilibrium...")
        sol = result['solution']
        valid_state = result['valid_state']
        final_fields = sol.fields[-1]

        c_eq = valid_state.face_centroids + final_fields[:, :2]
        R = vmap(rotation_matrix)(final_fields[:, 2])
        s_eq = jnp.einsum('nij, nkj -> nki', R, valid_state.centroid_node_vectors)
        
        equilibrium_state = valid_state._replace(face_centroids=c_eq, centroid_node_vectors=s_eq)
        plot_stage(equilibrium_state, "Stage 2: Static Equilibrium", 
                   show=config.show_stage2, save=config.save_plots)

    # Animation
    if config.incremental and config.save_animation:
        print(f"\nGenerating animation from history ({config.num_load_steps} frames)...")
        sol = result['solution']
        valid_state = result['valid_state']
        state_history = []
        for i in range(sol.fields.shape[0]):
            fields = sol.fields[i]
            c_i = valid_state.face_centroids + fields[:, :2]
            R_i = vmap(rotation_matrix)(fields[:, 2])
            s_i = jnp.einsum('nij, nkj -> nki', R_i, valid_state.centroid_node_vectors)
            
            verts_rec = reconstruct_vertices(c_i, s_i)
            new_verts = np.zeros_like(tessellation.vertices)
            for j, face in enumerate(tessellation.faces):
                for k, v_idx in enumerate(face.vertex_indices):
                    new_verts[v_idx] = verts_rec[j, k]
            state_history.append(new_verts)
            
        ani_dir = "data/outputs/animations"
        os.makedirs(ani_dir, exist_ok=True)
        ani_path = os.path.join(ani_dir, f"{config_name}_incremental.gif")
        
        fps = max(5, config.num_load_steps // 3)
        animate_tessellation(tessellation, state_history, filepath=ani_path, fps=fps, target_params=target_params)
