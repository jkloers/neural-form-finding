"""
Geometry history reconstruction and animation for training evolution.

Given a list of parameter snapshots collected during a training run,
reconstructs the tessellation geometry at a chosen pipeline stage for each
snapshot and animates the evolution across epochs.
"""

import math

import numpy as np
import jax.numpy as jnp
from jax import vmap
import matplotlib.pyplot as plt
import matplotlib.animation as mpl_animation
from typing import Any, Optional

from nff.stages.state import CentroidalState
from nff.stages.pipeline import forward_pipeline
from nff.stages.geometry import reconstruct_vertices
from nff.utils.linalg import rotation_matrix
from nff.utils.visualization import animate_tessellation, plot_tessellation
from nff.config.experiment import TargetConfig, ValidityConfig, PhysicsConfig


# ── Vertex extraction helpers ──────────────────────────────────────────────────

def _state_to_flat_vertices(state: CentroidalState, tessellation) -> np.ndarray:
    """CentroidalState → (n_vertices, 2) tessellation vertex array.

    Scatters per-face vertex copies (n_faces, max_nodes, 2) back into the
    shared-vertex layout expected by tessellation.update_vertices.
    """
    verts_rec = np.array(reconstruct_vertices(
        state.face_centroids, state.centroid_node_vectors))
    flat = np.zeros_like(tessellation.vertices)
    for i, face in enumerate(tessellation.faces):
        for j, v_idx in enumerate(face.vertex_indices):
            flat[v_idx] = verts_rec[i, j]
    return flat


def _equilibrium_state(valid_state: CentroidalState, solution) -> CentroidalState:
    """Apply final-step displacement fields to valid_state to get equilibrium geometry.

    Mirrors the Stage 2 reconstruction in pipeline_viz.visualize_pipeline_results.
    """
    fields = solution.fields[-1]                                  # (n_faces, 3): [dx, dy, dθ]
    c_eq   = valid_state.face_centroids + fields[:, :2]
    R      = vmap(rotation_matrix)(fields[:, 2])
    s_eq   = jnp.einsum('nij,nkj->nki', R, valid_state.centroid_node_vectors)
    return valid_state._replace(face_centroids=c_eq, centroid_node_vectors=s_eq)


def _result_to_vertices(result: dict, tessellation, target_stage: int) -> np.ndarray:
    """Extract flat (n_vertices, 2) vertex positions from a forward_pipeline result."""
    if target_stage == 0:
        state = result['mapped_state']
    elif target_stage == 1:
        state = result['valid_state']
    else:
        state = _equilibrium_state(result['valid_state'], result['solution'])
    return _state_to_flat_vertices(state, tessellation)


# ── 2-panel animation ─────────────────────────────────────────────────────────

def _animate_with_loss(
        tessellation,
        vertex_history: list[np.ndarray],
        snapshot_epochs: list[int],
        history_loss: list[dict],
        filepath: Optional[str],
        fps: int,
        target_params: dict,
        **plot_kwargs,
) -> None:
    """Render geometry (left) + animated stacked loss chart (right), synchronized.

    The loss panel mirrors plot_loss_history: stacked positive penalty terms,
    negative reward terms below zero, bold total loss line on top.
    """
    # ── Fixed geometry bounds ──────────────────────────────────────────────
    all_X     = np.concatenate(vertex_history)
    x_min, y_min = all_X.min(axis=0)
    x_max, y_max = all_X.max(axis=0)
    cx        = (x_max + x_min) / 2
    cy        = (y_max + y_min) / 2
    geo_range = max(x_max - x_min, y_max - y_min) * 1.1

    # ── Pre-compute loss component arrays (all epochs) ─────────────────────
    def _s(key):
        return np.array([float(d.get(key, 0.0)) for d in history_loss])

    def _active(arr):
        return bool(np.any(np.abs(arr) > 1e-10))

    chamfer    = _s('comp_geom_chamfer')
    void_cl    = _s('void_closure')
    hinge_gap  = _s('hinge_gap')
    stretching = _s('comp_phys_stretching')
    shearing   = _s('comp_phys_shearing')
    bending    = _s('comp_phys_bending')
    contact    = _s('comp_phys_contact')
    mat_area   = _s('comp_geom_material_area')
    cl_delta   = _s('closure_delta')    # ≤ 0
    openness   = _s('openness')         # ≤ 0
    deform     = _s('deformation')      # ≤ 0
    total      = _s('total')
    epochs_all = np.arange(len(history_loss))

    pos_terms = [
        (chamfer,    'Chamfer',       '#E07B39'),
        (void_cl,    'Void closure',  '#1565C0'),
        (hinge_gap,  'Hinge gap',     '#8E24AA'),
        (stretching, 'Stretching',    '#2D6A4F'),
        (shearing,   'Shearing',      '#4CAF50'),
        (bending,    'Bending',       '#80CBC4'),
        (contact,    'Contact',       '#9E9E9E'),
        (mat_area,   'Material area', '#BDBDBD'),
    ]
    neg_terms = [
        (cl_delta, 'Closure delta  (reward ↓)', '#D32F2F'),
        (openness, 'Openness  (reward ↓)',       '#1976D2'),
        (deform,   'Deformation  (reward ↓)',    '#F57C00'),
    ]
    active_pos = [(a, l, c) for a, l, c in pos_terms if _active(a)]
    active_neg = [(a, l, c) for a, l, c in neg_terms if _active(a)]

    # ── Fixed y-limits from full history (skip JIT warmup spike) ──────────
    skip = min(10, max(1, len(epochs_all) // 20))
    sl   = slice(skip, None)

    pos_sum = np.zeros(len(epochs_all))
    for a, _, _ in active_pos:
        pos_sum += a
    neg_bottom = np.zeros(len(epochs_all))
    for a, _, _ in active_neg:
        neg_bottom += a

    total_c   = np.where(np.isfinite(total),      total,      0.0)
    pos_c     = np.where(np.isfinite(pos_sum),    pos_sum,    0.0)
    neg_c     = np.where(np.isfinite(neg_bottom), neg_bottom, 0.0)

    above = np.concatenate([pos_c[sl], total_c[sl][total_c[sl] > 0]])
    below = np.concatenate([neg_c[sl][neg_c[sl] < 0], total_c[sl][total_c[sl] < 0]])

    y_max = float(np.percentile(above, 99)) * 1.35 if above.size > 0 else 1.0
    y_min = float(np.percentile(below, 1))  * 1.25 if below.size > 0 else -0.04 * y_max
    y_min = min(y_min, -0.03 * y_max)

    # ── Figure ─────────────────────────────────────────────────────────────
    plt.rcParams.update({
        'font.family': 'serif',
        'axes.linewidth': 1.0,
        'lines.linewidth': 1.5,
        'font.size': 11,
        'legend.frameon': True,
        'legend.edgecolor': 'black',
        'legend.fancybox': False,
    })
    plt.style.use('default')
    fig, (ax_geo, ax_loss) = plt.subplots(
        1, 2, figsize=(18, 8),
        gridspec_kw={'width_ratios': [1, 1]},
        facecolor='#FFFFFF',
    )

    def update(frame):
        ax_geo.clear()
        ax_loss.clear()
        cur_ep = snapshot_epochs[frame]

        # ── Geometry panel ──────────────────────────────────────────────
        tessellation.update_vertices(vertex_history[frame])
        plot_tessellation(
            tessellation, ax=ax_geo,
            title="Target Matching at Physical Equilibrium",
            target_params=target_params,
            **plot_kwargs,
        )
        ax_geo.set_xlim(cx - geo_range / 2, cx + geo_range / 2)
        ax_geo.set_ylim(cy - geo_range / 2, cy + geo_range / 2)
        # Subtle epoch counter in the top-left corner of the geometry panel
        ax_geo.text(
            0.03, 0.97, f"epoch  {cur_ep:04d}",
            transform=ax_geo.transAxes,
            fontsize=10, va='top', ha='left', color='#555555',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      alpha=0.85, edgecolor='#DDDDDD'),
        )

        # ── Loss panel — stacked component chart up to cur_ep ───────────
        ep  = epochs_all[:cur_ep + 1]
        sl2 = slice(0, cur_ep + 1)

        ax_loss.set_ylim(y_min, y_max)

        if active_pos:
            arrays = [a[sl2] for a, _, _ in active_pos]
            labels = [l for _, l, _ in active_pos]
            colors = [c for _, _, c in active_pos]
            ax_loss.stackplot(ep, *arrays, labels=labels,
                              colors=colors, alpha=0.70, zorder=2)

        cum = np.zeros(len(ep))
        for arr, label, color in active_neg:
            ax_loss.fill_between(ep, cum, cum + arr[sl2],
                                 color=color, alpha=0.55, label=label, zorder=2)
            cum = cum + arr[sl2]

        ax_loss.plot(ep, total[sl2], color='#111111', linewidth=2.5,
                     label='Total loss', zorder=10)
        ax_loss.axhline(0, color='#333333', linewidth=0.8, linestyle='-', alpha=0.4)

        ax_loss.set_xlim(0, len(history_loss) - 1)
        ax_loss.set_xlabel('Training Epoch', fontweight='bold')
        ax_loss.set_ylabel('Weighted loss contribution', fontweight='bold')
        ax_loss.set_title('Training Loss — Component Contributions',
                          fontsize=13, pad=8, fontweight='bold')
        ax_loss.grid(True, linestyle=':', alpha=0.3, zorder=1)
        ax_loss.legend(loc='upper right', fontsize=9, ncol=2,
                       frameon=True, framealpha=0.92)

    print(f"Generating animation with {len(vertex_history)} frames...")
    ani = mpl_animation.FuncAnimation(
        fig, update, frames=len(vertex_history), blit=False)

    if filepath is not None:
        writer = 'pillow' if filepath.endswith('.gif') else None
        ani.save(filepath, writer=writer, fps=fps)
        print(f"Animation successfully saved to {filepath}")

    plt.close(fig)


# ── Public API ─────────────────────────────────────────────────────────────────

def build_geometry_history(
        params_history: list[Any],
        initial_state: CentroidalState,
        tessellation,
        target_cfg: TargetConfig,
        validity_cfg: ValidityConfig,
        physics_cfg: PhysicsConfig,
        map_type: str,
        target_stage: int = 1,
        use_shirley_chiu: bool = True,
        strict_boundary_fit: bool = True,
        load_specs: Optional[list] = None,
        static_features: Optional[dict] = None,
) -> list[np.ndarray]:
    """Reconstruct tessellation vertex arrays for each parameter snapshot.

    Calls forward_pipeline once per snapshot and extracts (n_vertices, 2)
    positions at the requested pipeline stage.

    Args:
        params_history:  CPU numpy param snapshots, one per recorded epoch.
        initial_state:   Flat CentroidalState used during training.
        tessellation:    Reference tessellation for vertex index mapping.
        target_stage:    0 → mapping output; 1 → validity; 2 → equilibrium.
        static_features: Required for GNN map types (carries inner_depth, num_layers).

    Returns:
        List of (n_vertices, 2) numpy arrays, one per snapshot.
    """
    if not params_history:
        return []

    if target_stage == 2 and not getattr(physics_cfg, 'use_stage2', True):
        print("Warning: target_stage=2 but use_stage2=False — falling back to Stage 1.")
        target_stage = 1

    vertex_history = []
    n = len(params_history)
    for idx, params in enumerate(params_history):
        print(f"  Snapshot {idx + 1}/{n} ...", end='\r', flush=True)
        result = forward_pipeline(
            initial_state=initial_state,
            target_cfg=target_cfg,
            validity_cfg=validity_cfg,
            physics_cfg=physics_cfg,
            map_type=map_type,
            map_params=params,
            use_shirley_chiu=use_shirley_chiu,
            strict_boundary_fit=strict_boundary_fit,
            static_features=static_features,
            load_specs=load_specs,
        )
        vertex_history.append(_result_to_vertices(result, tessellation, target_stage))

    print(f"  Done — {n} snapshots reconstructed.             ")
    return vertex_history


def animate_training_evolution(
        params_history: list[Any],
        initial_state: CentroidalState,
        tessellation,
        target_cfg: TargetConfig,
        validity_cfg: ValidityConfig,
        physics_cfg: PhysicsConfig,
        target_params: dict,
        map_type: str = 'conformal_polynomial',
        target_stage: int = 1,
        filepath: Optional[str] = 'training_evolution.gif',
        fps: int = 10,
        use_shirley_chiu: bool = True,
        strict_boundary_fit: bool = True,
        load_specs: Optional[list] = None,
        static_features: Optional[dict] = None,
        snapshot_epochs: Optional[list[int]] = None,
        history_loss: Optional[list[dict]] = None,
        **plot_kwargs,
) -> None:
    """Animate geometry evolution across training snapshots and save to file.

    When snapshot_epochs and history_loss are both provided, renders a 2-panel
    figure with the geometry on the left and an animated chamfer loss curve on
    the right, synchronized to the current epoch.  Otherwise falls back to the
    single-panel geometry animation.

    Args:
        params_history:   CPU numpy param snapshots from a training run.
        target_stage:     0 / 1 / 2 — which pipeline stage to visualise per frame.
        filepath:         Output path (.gif). None skips saving.
        fps:              Frames per second.
        snapshot_epochs:  Epoch index for each entry in params_history.
        history_loss:     Per-epoch metric dicts from the training loop.
        **plot_kwargs:    Forwarded to plot_tessellation.
    """
    if not params_history:
        print("params_history is empty — skipping animation.")
        return

    stage_label = {0: 'Stage 0 (mapping)', 1: 'Stage 1 (validity)', 2: 'Stage 2 (equilibrium)'}
    print(f"Building geometry history — {len(params_history)} snapshots, "
          f"{stage_label.get(target_stage, target_stage)} ...")

    vertex_history = build_geometry_history(
        params_history=params_history,
        initial_state=initial_state,
        tessellation=tessellation,
        target_cfg=target_cfg,
        validity_cfg=validity_cfg,
        physics_cfg=physics_cfg,
        map_type=map_type,
        target_stage=target_stage,
        use_shirley_chiu=use_shirley_chiu,
        strict_boundary_fit=strict_boundary_fit,
        load_specs=load_specs,
        static_features=static_features,
    )

    if not vertex_history:
        return

    if history_loss is not None and snapshot_epochs is not None:
        _animate_with_loss(
            tessellation, vertex_history, snapshot_epochs, history_loss,
            filepath, fps, target_params, **plot_kwargs,
        )
    else:
        animate_tessellation(
            tessellation, vertex_history,
            filepath=filepath,
            fps=fps,
            target_params=target_params,
            **plot_kwargs,
        )
