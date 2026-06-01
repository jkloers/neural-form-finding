import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

import jax.numpy as jnp
from jax import vmap

from nff.utils.linalg import rotation_matrix
from nff.stages.geometry import reconstruct_vertices, compute_face_areas
from nff.utils.visualization import plot_tessellation, animate_tessellation, plot_tessellation_differences


def _draw_typed_loads(ax, load_specs, valid_centroids, eq_centroids):
    """Overlay force arrows and moment indicators for typed load specs.

    Draws at the equilibrium centroid positions; force directions are computed
    from the reference (valid_state) centroid positions, matching what the
    physics solver actually used.

    Args:
        ax:              matplotlib Axes to draw on.
        load_specs:      List of load dicts from the config (typed format).
        valid_centroids: (n_faces, 2) numpy — reference centroids pre-deformation.
        eq_centroids:    (n_faces, 2) numpy — equilibrium centroid positions.
    """
    if not load_specs:
        return

    valid_centroids = np.array(valid_centroids, dtype=float)
    eq_centroids    = np.array(eq_centroids,    dtype=float)

    # Pre-compute tess_frame principal axes once (same for all tess_frame specs)
    _tess_axes = None
    if any(s.get('type') == 'tess_frame' for s in load_specs):
        c = valid_centroids - valid_centroids.mean(axis=0)
        cov = c.T @ c + np.array([[2e-6, 0.0], [0.0, 1e-6]])
        _, eigvecs = np.linalg.eigh(cov)   # ascending eigenvalues; col1=major
        _tess_axes = eigvecs               # shape (2, 2)

    # Accumulate per-face force vectors and moments
    face_fx  = {}   # translational x
    face_fy  = {}   # translational y
    face_mz  = {}   # moments

    for spec in load_specs:
        load_type = spec.get('type', 'global_frame')

        if load_type == 'tile_to_tile':
            src = int(spec['source_face'])
            tgt = int(spec['target_face'])
            mag = float(spec['magnitude'])
            # Convention: diff = source - target (outward), magnitude < 0 → compression
            diff = eq_centroids[src] - eq_centroids[tgt]
            norm = np.linalg.norm(diff)
            d = diff / norm if norm > 1e-8 else np.array([1.0, 0.0])
            face_fx[src] = face_fx.get(src, 0.0) + mag * d[0]
            face_fy[src] = face_fy.get(src, 0.0) + mag * d[1]

        elif load_type == 'tess_frame':
            face     = int(spec['face'])
            tess_dof = int(spec['tess_dof'])   # 0=major, 1=minor
            value    = float(spec['value'])
            d = _tess_axes[:, 1 - tess_dof]    # col1=major, col0=minor
            face_fx[face] = face_fx.get(face, 0.0) + value * d[0]
            face_fy[face] = face_fy.get(face, 0.0) + value * d[1]

        elif load_type == 'global_frame':
            face  = int(spec['face'])
            dof   = int(spec['dof'])
            value = float(spec['value'])
            if dof == 0:
                face_fx[face] = face_fx.get(face, 0.0) + value
            elif dof == 1:
                face_fy[face] = face_fy.get(face, 0.0) + value
            elif dof == 2:
                face_mz[face] = face_mz.get(face, 0.0) + value

    FORCE_COLOR  = '#D62828'
    ARROW_LENGTH = 0.18   # fixed visual length (tessellation units)

    all_faces = set(face_fx) | set(face_fy) | set(face_mz)
    for fi in all_faces:
        cx, cy = eq_centroids[fi]
        fx = face_fx.get(fi, 0.0)
        fy = face_fy.get(fi, 0.0)
        mz = face_mz.get(fi, 0.0)

        # Translational force arrow (fixed length, direction only)
        force_len = np.sqrt(fx**2 + fy**2)
        if force_len > 1e-8:
            ux, uy = fx / force_len, fy / force_len
            ax.annotate(
                '', xy=(cx + ux * ARROW_LENGTH, cy + uy * ARROW_LENGTH),
                xytext=(cx, cy),
                arrowprops=dict(arrowstyle='->', color=FORCE_COLOR,
                                lw=2.0, mutation_scale=18),
                zorder=35)

        # Moment indicator (circular arrow)
        if abs(mz) > 1e-8:
            r = 0.10
            if mz > 0:   # CCW
                start = (cx + r, cy - r / 2)
                end   = (cx - r / 2, cy + r)
                rad   = 0.6
            else:         # CW
                start = (cx - r, cy - r / 2)
                end   = (cx + r / 2, cy + r)
                rad   = -0.6
            patch = FancyArrowPatch(
                start, end,
                connectionstyle=f"arc3,rad={rad}",
                color=FORCE_COLOR,
                arrowstyle="Simple, tail_width=1.5, head_width=7, head_length=9",
                zorder=35)
            ax.add_patch(patch)


def visualize_pipeline_results(result, tessellation, config, target_params, config_name,
                                run_dir=None, load_specs=None):
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

    def plot_stage(state, title, mapping_fn=None, map_params=None, on_ax=None):
        tess_copy = update_tessellation_from_state(state)

        fig, ax = plt.subplots(figsize=(8, 8))
        plot_tessellation(tess_copy, ax=ax, title=title,
                          mapping_fn=mapping_fn, map_params=map_params,
                          original_vertices=tessellation.vertices, **_plot_kwargs())

        if on_ax is not None:
            on_ax(ax)

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

        # Draw typed loads (tile_to_tile, tess_frame, global_frame moments) as
        # force arrows / moment indicators at the equilibrium face positions.
        _typed_specs = [s for s in (load_specs or []) if 'type' in s]
        def _on_ax_stage2(ax):
            _draw_typed_loads(ax, _typed_specs,
                              np.array(valid_state.face_centroids),
                              np.array(c_eq))

        plot_stage(equilibrium_state, "Stage 2: Static Equilibrium",
                   on_ax=_on_ax_stage2 if _typed_specs else None)

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
    """Plots the training loss history.

    Panel 1 (always): standard penalty terms on a log scale.
    Panel 2 (when active): closing reward terms on a linear scale — only
      rendered when openness or deformation weights are non-zero.
    """
    if not history:
        return

    plt.rcParams.update({
        'font.family': 'serif',
        'axes.linewidth': 1.0,
        'font.size': 10,
        'grid.alpha': 0.3
    })

    epochs = np.arange(len(history))

    def _series(key, default=0.0):
        return [float(h.get(key, default)) for h in history]

    total       = _series('loss_total')
    geom_total  = _series('loss_geometric')
    phys_total  = _series('loss_physical')
    chamfer     = _series('comp_geom_chamfer')
    hinge_gap   = _series('hinge_gap')
    reg         = _series('comp_regularization')
    mat_area    = _series('comp_geom_material_area')
    openness    = _series('openness')
    deformation = _series('deformation')

    has_openness    = any(v != 0.0 for v in openness)
    has_deformation = any(v != 0.0 for v in deformation)
    has_hinge_gap   = any(v != 0.0 for v in hinge_gap)
    has_closing     = has_openness or has_deformation

    n_rows = 2 if has_closing else 1
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 5 * n_rows), squeeze=False)
    ax = axes[0, 0]

    # ── Panel 1: standard penalty terms (log scale) ───────────────────────────
    ax.plot(epochs, total,      color='#111111', linewidth=3.0, label='Total',     zorder=10)
    ax.plot(epochs, geom_total, color='#F58025', linewidth=2.0, label='Geometric', zorder=5)
    ax.plot(epochs, phys_total, color='#2D6A4F', linewidth=2.0, label='Physical',  zorder=5)
    ax.plot(epochs, chamfer,    color='#E07B39', linewidth=1.5, linestyle='--',
            label='Chamfer', zorder=4, alpha=0.8)
    if has_hinge_gap:
        ax.plot(epochs, hinge_gap, color='#9B59B6', linewidth=1.5, linestyle='--',
                label='Hinge Gap', zorder=4, alpha=0.8)
    if any(v != 0.0 for v in reg):
        ax.plot(epochs, reg, color='#888888', linewidth=1.2, linestyle=':',
                label='Regularization', zorder=3, alpha=0.7)
    if any(v != 0.0 for v in mat_area):
        ax.plot(epochs, mat_area, color='#BDC3C7', linewidth=1.2, linestyle=':',
                label='Material Area', zorder=3, alpha=0.7)

    pos_vals = [v for v in total + geom_total + phys_total + chamfer if v > 0]
    if pos_vals:
        ax.set_yscale('log')
    ax.set_xlabel('Training Epoch', fontweight='bold')
    ax.set_ylabel('Weighted Loss (log scale)', fontweight='bold')
    ax.set_title('Training Loss — Standard Terms', fontsize=13, pad=12, fontweight='bold')
    ax.grid(True, which="both", linestyle='--', alpha=0.3)
    ax.legend(loc='upper right', frameon=True, fontsize=10, ncol=2)

    # ── Panel 2: closing reward terms (linear scale, values are negative) ─────
    if has_closing:
        ax2 = axes[1, 0]
        if has_openness:
            ax2.plot(epochs, openness, color='#1A73E8', linewidth=2.5,
                     label='Openness (−w·log1p(void))', zorder=5)
        if has_deformation:
            ax2.plot(epochs, deformation, color='#CC0000', linewidth=2.5,
                     label='Deformation (−w·log1p(U_bend))', zorder=5)
        ax2.axhline(0, color='#888888', linewidth=0.8, linestyle='-')

        ax2.set_xlabel('Training Epoch', fontweight='bold')
        ax2.set_ylabel('Reward value  (↓ = more active)', fontweight='bold')
        ax2.set_title('Training Loss — Closing Reward Terms', fontsize=13, pad=12, fontweight='bold')
        ax2.grid(True, linestyle='--', alpha=0.3)
        ax2.legend(loc='lower right', frameon=True, fontsize=10)

    plt.tight_layout()

    if config.visualization.save_outputs and run_dir:
        save_path = os.path.join(run_dir, "training_loss.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved loss history plot to {save_path}")

    if config.visualization.show_plots:
        plt.show()
    else:
        plt.close(fig)
