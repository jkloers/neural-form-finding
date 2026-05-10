import os
import copy
import numpy as np
import matplotlib.pyplot as plt

import jax.numpy as jnp
from jax import vmap

from jax_backend.physics_solver.kinematics import rotation_matrix
from jax_backend.geometry import reconstruct_vertices, compute_face_areas
from src.utils.visualization import plot_tessellation, animate_tessellation, plot_tessellation_differences

def visualize_pipeline_results(result, tessellation, config, target_params, config_name, run_dir=None):
    """
    Orchestrates the visualization of the entire pipeline, including static plots and animations.
    Controlled by the visualization settings in the config.
    """
    def update_tessellation_from_state(state):
        c = state.face_centroids
        s = state.centroid_node_vectors
        verts_rec = reconstruct_vertices(c, s)
        
        tess_copy = copy.deepcopy(tessellation)
        new_verts = np.zeros_like(tess_copy.vertices)
        for i, face in enumerate(tess_copy.faces):
            for j, v_idx in enumerate(face.vertex_indices):
                new_verts[v_idx] = verts_rec[i, j]
        tess_copy.update_vertices(new_verts)
        return tess_copy

    def plot_stage(state, title, mapping_fn=None, map_params=None):
        tess_copy = update_tessellation_from_state(state)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        plot_kwargs = {
            'show_target': True,
            'target_params': target_params,
            'show_hinges': config.visualization.show_hinges,
            'show_hinge_indices': config.visualization.show_hinge_indices,
            'show_face_indices': config.visualization.show_face_indices,
            'show_external_forces': config.visualization.show_external_forces,
            'show_kinematic_blocks': config.visualization.show_kinematic_blocks,
        }
        
        plot_tessellation(tess_copy, ax=ax, title=title, 
                          mapping_fn=mapping_fn, map_params=map_params, 
                          original_vertices=tessellation.vertices, **plot_kwargs)
        
        if config.visualization.save_outputs and run_dir:
            filename = title.lower().replace(" ", "_").replace(":", "") + ".png"
            save_path = os.path.join(run_dir, filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Saved plot to {save_path}")
        if config.visualization.show_plots:
            plt.show()
        else:
            plt.close(fig)

    # Stage 0
    if config.visualization.stage0:
        print("Displaying Stage 0: Initial Mapping...")
        mapping_fn = result.get('mapping_fn', None)
        map_params = result.get('map_params', None)
        plot_stage(result['mapped_state'], "Stage 0: Initial Mapping", mapping_fn=mapping_fn, map_params=map_params)

    # Stage 1 — Geometric Validity (with Deformation Heatmap)
    if config.visualization.stage1:
        print("Displaying Stage 1: Geometric Validity (Deformation Map)...")
        
        # Calculate area deformation relative to Stage 0
        area0 = compute_face_areas(result['mapped_state'].centroid_node_vectors)
        area1 = compute_face_areas(result['valid_state'].centroid_node_vectors)
        
        # Percentage area change
        area_diff = ((area1 - area0) / area0) * 100.0
        
        # Update tessellation vertices for plotting
        tess_copy = update_tessellation_from_state(result['valid_state'])
        
        fig, ax = plt.subplots(figsize=(8, 8))
        plot_kwargs = {
            'show_target': True,
            'target_params': target_params,
            'show_hinges': config.visualization.show_hinges,
            'show_hinge_indices': config.visualization.show_hinge_indices,
            'show_face_indices': config.visualization.show_face_indices,
            'show_external_forces': False, # Hide forces for this view
            'show_kinematic_blocks': config.visualization.show_kinematic_blocks,
        }
        
        plot_tessellation_differences(tess_copy, area_diff, ax=ax, 
                                      title="Stage 1: Relative Area Deformation (%)",
                                      cmap_name='YlOrRd',
                                      **plot_kwargs)
        
        if config.visualization.save_outputs and run_dir:
            save_path = os.path.join(run_dir, "stage_1_deformation.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Saved deformation plot to {save_path}")
            
        if config.visualization.show_plots:
            plt.show()
        else:
            plt.close(fig)

    # Stage 2
    if config.visualization.stage2:
        print("Displaying Stage 2: Static Equilibrium...")
        sol = result['solution']
        valid_state = result['valid_state']
        final_fields = sol.fields[-1]

        c_eq = valid_state.face_centroids + final_fields[:, :2]
        R = vmap(rotation_matrix)(final_fields[:, 2])
        s_eq = jnp.einsum('nij, nkj -> nki', R, valid_state.centroid_node_vectors)
        
        equilibrium_state = valid_state._replace(face_centroids=c_eq, centroid_node_vectors=s_eq)
        plot_stage(equilibrium_state, "Stage 2: Static Equilibrium")

    # Animation
    if config.physics.incremental and config.visualization.animation:
        print(f"\nGenerating animation from history...")
        sol = result['solution']
        valid_state = result['valid_state']
        def interpolate_states(v1, v2, n_frames):
            return [v1 + (v2 - v1) * (i / (n_frames - 1)) for i in range(n_frames)]

        state_history = []
        
        # Extrait les vertices de l'état valide (point de départ de la physique)
        v_valid = update_tessellation_from_state(result['valid_state']).vertices.copy()
        
        # Le premier frame de la vidéo est l'état valide
        state_history.append(v_valid)
        
        # Animation : Incremental Physics (Stage 2)
        v_prev = v_valid
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
            
            interp_frames = interpolate_states(v_prev, new_verts, 5)
            state_history.extend(interp_frames[1:])
            v_prev = new_verts
            
        if config.visualization.save_outputs and run_dir:
            ani_path = os.path.join(run_dir, "animation.gif")
        else:
            ani_path = None
        
        plot_kwargs = {
            'show_target': True,
            'target_params': target_params,
            'show_hinges': config.visualization.show_hinges,
            'show_hinge_indices': config.visualization.show_hinge_indices,
            'show_face_indices': config.visualization.show_face_indices,
            'show_external_forces': config.visualization.show_external_forces,
            'show_kinematic_blocks': config.visualization.show_kinematic_blocks,
        }
        fps = max(5, config.physics.num_load_steps // 3)
        animate_tessellation(tessellation, state_history, filepath=ani_path, fps=fps, **plot_kwargs)
        if ani_path:
            print(f"  Saved animation to {ani_path}")

    # 5. Energy Plot (Stage 2 Analysis)
    if config.visualization.energy_plot and result.get('solution') and getattr(result['solution'], 'energies', None) is not None:
        print("Displaying Stage 2: Thermodynamic Balance Analysis...")
        energies_dict = result['solution'].energies
        
        if not isinstance(energies_dict, dict):
            print("  Warning: Energy history is not a dictionary. Skipping scientific plot.")
        else:
            # 1. Configuration esthétique stricte (Standard Scientifique)
            plt.rcParams.update({
                'font.family': 'serif',       # Police classique type article
                'axes.linewidth': 1.0,        # Cadre fin
                'lines.linewidth': 1.5,       # Lignes des totaux
                'font.size': 11,
                'legend.frameon': True,
                'legend.edgecolor': 'black',
                'legend.fancybox': False,
            })

            fig, ax = plt.subplots(figsize=(9, 5))

            # 2. Extraction des données
            steps = np.arange(len(energies_dict['work']))
            w_ext = energies_dict['work']
            
            e_stretch = energies_dict['stretch']
            e_shear = energies_dict['shear']
            e_rot = energies_dict['rot']
            e_contact = energies_dict['contact']
            
            # Recalcul de U_int total pour le bilan
            u_int = e_stretch + e_shear + e_rot + e_contact
            
            # Le Bilan (Théoriquement = 0)
            thermo_balance = u_int - w_ext

            # 3. Plot Balance Curves (Symmetry and Zero-check)
            ax.plot(steps, u_int, color='#000000', linewidth=2, label=r'Total Internal Energy ($U_{int}$)', zorder=5)
            ax.plot(steps, -w_ext, color='#D62828', linewidth=2, label=r'External Work ($-W_{ext}$)', zorder=5)
            ax.plot(steps, thermo_balance, color='#2D6A4F', linestyle='-', linewidth=2.5, 
                    label=r'Energy Balance ($\Sigma \approx 0$)', zorder=6)

            # 4. Stacked Area Plot (The "economist" breakdown)
            stack_labels = [r'Rotation ($E_{rot}$)', r'Stretch ($E_{stretch}$)', r'Shear ($E_{shear}$)', r'Contact ($E_{contact}$)' ]
            stack_colors = ['#F58025', '#FFB380', '#D9D9D9', '#808080']
            
            show_contact = np.max(e_contact) > 1e-6
            
            y_stack = [e_rot, e_stretch, e_shear]
            colors = stack_colors[:3]
            labels = stack_labels[:3]
            
            if show_contact:
                y_stack.append(e_contact)
                colors.append(stack_colors[3])
                labels.append(stack_labels[3])

            ax.stackplot(steps, *y_stack, labels=labels, colors=colors, alpha=0.7, zorder=2)

            # 5. Reference zero line
            ax.axhline(0, color='#000000', linewidth=0.8, linestyle='-', alpha=0.5)

            # 6. Final Touches
            ax.set_xlabel('Load Step')
            ax.set_ylabel('Energy (Joules)')
            ax.set_title("Energy Balance Analysis", pad=15, fontweight='bold')
            
            ax.grid(True, linestyle=':', alpha=0.3, zorder=1)
            
            # Place legend inside the plot to ensure it stays within the visible area
            ax.legend(loc='upper left', fontsize=9, ncol=2, frameon=True, framealpha=0.9)
            
            plt.tight_layout()
            
            if config.visualization.save_outputs and run_dir:
                save_path = os.path.join(run_dir, "energy_balance.png")
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"  Saved energy balance plot to {save_path}")
                
            if config.visualization.show_plots: 
                plt.show()
            else:
                plt.close(fig)


def plot_loss_history(history, config, run_dir=None):
    """Plots the training loss history with component breakdown (Étape 7)."""
    if not history:
        return
        
    plt.rcParams.update({
        'font.family': 'serif',
        'axes.linewidth': 1.0,
        'font.size': 11,
    })

    epochs = np.arange(len(history))
    
    # Extraction et calculs hiérarchiques
    total = [float(h['total']) for h in history]
    chamfer = [float(h['chamfer_total']) for h in history]
    energy = [float(h['energy']) for h in history]
    material = [float(h.get('global_material_area', 0.0)) for h in history]
    
    # Étape 8 : Calcul de la perte géométrique totale (Somme pondérée si nécessaire, 
    # mais ici on suit la logique de l'utilisateur : Chamfer + Conservation)
    geom_total = [c + m for c, m in zip(chamfer, material)]

    fig, ax = plt.subplots(figsize=(9, 5))

    # Totaux (Traits épais)
    ax.plot(epochs, total, color='#000000', linewidth=2.5, label='Total Loss', zorder=10)
    ax.plot(epochs, geom_total, color='#F58025', linewidth=2.0, label='Geometric Objective (Total)', zorder=8)
    ax.plot(epochs, energy, color='#2D6A4F', linewidth=2.0, label='Physical Objective (Energy)', zorder=8)
    
    # Sous-composantes géométriques (Traits fins pointillés)
    ax.plot(epochs, chamfer, color='#F58025', linestyle='--', linewidth=1.2, label='  └─ Target Fitting (Chamfer)', alpha=0.7)
    
    mat_array = np.array(material)
    if np.max(mat_array) > 1e-9:
        ax.plot(epochs, material, color='#D62828', linestyle=':', linewidth=1.2, label='  └─ Material Area Conservation', alpha=0.7)

    ax.set_yscale('log')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss Value (Log Scale)')
    ax.set_title("Neural Form-Finding Convergence Analysis", pad=15, fontweight='bold')
    
    ax.grid(True, which="both", linestyle=':', alpha=0.3)
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9, ncol=1)

    plt.tight_layout()

    if config.visualization.save_outputs and run_dir:
        save_path = os.path.join(run_dir, "training_loss.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved loss history plot to {save_path}")
        
    if config.visualization.show_plots:
        plt.show()
    else:
        plt.close(fig)

