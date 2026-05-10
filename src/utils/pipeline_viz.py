import os
import copy
import numpy as np
import matplotlib.pyplot as plt

import jax.numpy as jnp
from jax import vmap

from jax_backend.utils.linalg import rotation_matrix
from jax_backend.geometry import reconstruct_vertices, compute_face_areas
from utils.visualization import plot_tessellation, animate_tessellation, plot_tessellation_differences

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

    def _plot_kwargs(show_external_forces=None):
        return {
            'show_target': True,
            'target_params': target_params,
            'show_hinges': config.visualization.show_hinges,
            'show_hinge_indices': config.visualization.show_hinge_indices,
            'show_face_indices': config.visualization.show_face_indices,
            'show_external_forces': show_external_forces if show_external_forces is not None else config.visualization.show_external_forces,
            'show_kinematic_blocks': config.visualization.show_kinematic_blocks,
        }

    def plot_stage(state, title, mapping_fn=None, map_params=None):
        tess_copy = update_tessellation_from_state(state)

        fig, ax = plt.subplots(figsize=(8, 8))
        plot_tessellation(tess_copy, ax=ax, title=title,
                          mapping_fn=mapping_fn, map_params=map_params,
                          original_vertices=tessellation.vertices, **_plot_kwargs())

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

        area0 = compute_face_areas(result['mapped_state'].centroid_node_vectors)
        area1 = compute_face_areas(result['valid_state'].centroid_node_vectors)
        area_diff = ((area1 - area0) / area0) * 100.0

        tess_copy = update_tessellation_from_state(result['valid_state'])

        fig, ax = plt.subplots(figsize=(8, 8))
        plot_tessellation_differences(tess_copy, area_diff, ax=ax,
                                      title="Stage 1: Relative Area Deformation (%)",
                                      cmap_name='YlOrRd',
                                      **_plot_kwargs(show_external_forces=False))

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

        # Starting vertices for the physics animation (valid state)
        v_valid = update_tessellation_from_state(result['valid_state']).vertices.copy()
        state_history.append(v_valid)

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

        fps = max(5, config.physics.num_load_steps // 3)
        animate_tessellation(tessellation, state_history, filepath=ani_path, fps=fps, **_plot_kwargs())
        if ani_path:
            print(f"  Saved animation to {ani_path}")

    # Energy Plot (Stage 2 Analysis)
    if config.visualization.energy_plot and result.get('solution') and getattr(result['solution'], 'energies', None) is not None:
        print("Displaying Stage 2: Thermodynamic Balance Analysis...")
        energies_dict = result['solution'].energies

        if not isinstance(energies_dict, dict):
            print("  Warning: Energy history is not a dictionary. Skipping scientific plot.")
        else:
            plt.rcParams.update({
                'font.family': 'serif',
                'axes.linewidth': 1.0,
                'lines.linewidth': 1.5,
                'font.size': 11,
                'legend.frameon': True,
                'legend.edgecolor': 'black',
                'legend.fancybox': False,
            })

            fig, ax = plt.subplots(figsize=(9, 5))

            steps = np.arange(len(energies_dict['work']))
            w_ext = energies_dict['work']

            e_stretch = energies_dict['stretch']
            e_shear = energies_dict['shear']
            e_rot = energies_dict['rot']
            e_contact = energies_dict['contact']

            u_int = e_stretch + e_shear + e_rot + e_contact
            thermo_balance = u_int - w_ext

            ax.plot(steps, u_int, color='#000000', linewidth=2, label=r'Total Internal Energy ($U_{int}$)', zorder=5)
            ax.plot(steps, -w_ext, color='#D62828', linewidth=2, label=r'External Work ($-W_{ext}$)', zorder=5)
            ax.plot(steps, thermo_balance, color='#2D6A4F', linestyle='-', linewidth=2.5,
                    label=r'Energy Balance ($\Sigma \approx 0$)', zorder=6)

            stack_labels = [r'Rotation ($E_{rot}$)', r'Stretch ($E_{stretch}$)', r'Shear ($E_{shear}$)', r'Contact ($E_{contact}$)']
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
            ax.axhline(0, color='#000000', linewidth=0.8, linestyle='-', alpha=0.5)

            ax.set_xlabel('Load Step')
            ax.set_ylabel('Energy (Joules)')
            ax.set_title("Energy Balance Analysis", pad=15, fontweight='bold')
            ax.grid(True, linestyle=':', alpha=0.3, zorder=1)
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
    """Plots the training loss history with component breakdown."""
    if not history:
        return

    plt.rcParams.update({
        'font.family': 'serif',
        'axes.linewidth': 1.0,
        'font.size': 10,
        'grid.alpha': 0.3
    })

    epochs = np.arange(len(history))

    total = [float(h['loss_total']) for h in history]
    geom_total = [float(h['loss_geometric']) for h in history]
    phys_total = [float(h['loss_physical']) for h in history]

    chamfer = [float(h['comp_geom_chamfer']) for h in history]
    material = [float(h['comp_geom_material_area']) for h in history]

    stretch = [float(h['comp_phys_stretching']) for h in history]
    shear = [float(h['comp_phys_shearing']) for h in history]
    bend = [float(h['comp_phys_bending']) for h in history]
    contact = [float(h['comp_phys_contact']) for h in history]
    reg = [float(h['comp_regularization']) for h in history]

    fig, ax = plt.subplots(figsize=(11, 7))

    ax.plot(epochs, total, color='#111111', linewidth=2.0, label='Grand Total Loss', alpha=0.9, zorder=10)
    ax.plot(epochs, geom_total, color='#023E8A', linewidth=3.5, label='Total Geometric Loss', zorder=5)
    ax.plot(epochs, chamfer, color='#0077B6', linewidth=1.2, linestyle='--', label='  ↳ Chamfer', alpha=0.8)
    ax.plot(epochs, material, color='#48CAE4', linewidth=1.2, linestyle='--', label='  ↳ Material Area', alpha=0.8)

    ax.plot(epochs, phys_total, color='#9D0208', linewidth=3.5, label='Total Physical Loss', zorder=5)
    ax.plot(epochs, stretch, color='#D00000', linewidth=1.0, linestyle='-.', label='  ↳ Stretching', alpha=0.7)
    ax.plot(epochs, shear, color='#E85D04', linewidth=1.0, linestyle='-.', label='  ↳ Shearing', alpha=0.7)
    ax.plot(epochs, bend, color='#F48C06', linewidth=1.0, linestyle='-.', label='  ↳ Bending', alpha=0.7)
    ax.plot(epochs, contact, color='#FFBA08', linewidth=1.0, linestyle=':', label='  ↳ Contact', alpha=0.7)
    ax.plot(epochs, reg, color='#666666', linewidth=1.0, linestyle=':', label='Regularization', alpha=0.5)

    ax.set_yscale('log')
    ax.set_xlabel('Training Epochs', fontweight='bold')
    ax.set_ylabel('Weighted Loss Value (Log Scale)', fontweight='bold')
    ax.set_title('Neural Form-Finding Convergence: Hierarchical Decomposition', fontsize=14, pad=20, fontweight='bold')

    ax.grid(True, which="both", linestyle='--', alpha=0.4)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", frameon=True, fontsize=9)

    plt.tight_layout()

    if config.visualization.save_outputs and run_dir:
        save_path = os.path.join(run_dir, "training_loss.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved loss history plot to {save_path}")

    if config.visualization.show_plots:
        plt.show()
    else:
        plt.close(fig)
