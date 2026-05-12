"""
train.py — Neural Form-Finding Training Pipeline.

Optimizes the initial mapping parameters (polynomial coefficients)
using gradient descent through the fully differentiable physics pipeline.
"""

import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import sys
sys.path.append(os.path.abspath('src'))

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import argparse
from types import SimpleNamespace
import datetime
import shutil

from src.topology.builder import build_tessellation
from src.problem.conditions import configure_tessellation
from src.problem.config import load_and_parse_config
from src.jax_backend.state import CentroidalState
from src.jax_backend.pipeline import forward_pipeline
from src.jax_backend.training.trainer import train_pipeline
from src.utils.pipeline_viz import visualize_pipeline_results, plot_loss_history

if __name__ == "__main__":

    # ── Configuration ─────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="Neural Form-Finding Training.")
    parser.add_argument("--config-dir", type=str, default="poster")
    parser.add_argument("--config-name", type=str, required=True)
    args = parser.parse_args()

    config_path = f"data/configs/{args.config_dir}/{args.config_name}.yaml"
    config = load_and_parse_config(config_path)
    print(f"Loaded config: {config_path}")

    # ── Tessellation setup ────────────────────────────────────────────────────
    topo = config.topology
    topo_obj = SimpleNamespace(**topo)
    
    tessellation = build_tessellation(topo.get('pattern'), 
                                      topo.get('width', 5), 
                                      topo.get('height', 5))
    
    # ── Initial Area Scaling (Étape 9) ────────────────────────────────────────
    # Scale the initial tessellation to match a requested total material area
    requested_area = topo.get('total_area')
    if requested_area:
        current_area = tessellation.compute_total_area()
        # Scale factor is the square root of the area ratio (L2 scaling for area)
        scale = np.sqrt(requested_area / current_area)
        print(f"Scaling initial tessellation by factor {scale:.4f} to reach area {requested_area:.2f}")
        tessellation.update_vertices(tessellation.vertices * scale)

    configure_tessellation(tessellation, topo_obj)
    
    # ── MODE & BASELINE SUMMARY ───────────────────────────────────────────────
    mode_str = "LEARN SCALE (Variable Geometry)" if config.mapping.learn_global_scale else "FIXED MATERIAL (Conservation of Matter)"
    print("\n" + "═" * 60)
    print(f" PIPELINE MODE: {mode_str}")
    print("═" * 60)

    initial_area = tessellation.compute_total_area()
    print(f" [PHYSICAL BASELINE] Initial Total Material: {initial_area:.6f}")
    print("═" * 60 + "\n")
    
    initial_state = CentroidalState.from_tessellation(tessellation, target_cfg=config.target)

    # ── Training ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STARTING END-TO-END TRAINING")
    print("=" * 60)

    initial_map_params = config.mapping.params

    # Option 2: if learning global scale, ensure log_scale = 0 is in the param tree
    # (so the optimizer sees it as a leaf). log_scale can also be set explicitly in the yaml.
    if config.mapping.learn_global_scale and 'log_scale' not in initial_map_params:
        initial_map_params = {**initial_map_params, 'log_scale': jnp.array(0.0)}

    optimized_params, history_loss = train_pipeline(
        initial_map_params,
        initial_state,
        config.target,
        config.validity,
        config.physics,
        config.training,
        map_type=config.mapping.type,
        use_shirley_chiu=config.mapping.use_shirley_chiu,
        strict_boundary_fit=config.mapping.strict_boundary_fit,
        learn_global_scale=config.mapping.learn_global_scale,
    )

    print(f"\nOptimization complete. Optimal params: {optimized_params}")

    # ── Final forward pass + visualization ────────────────────────────────────
    print("\nRunning final forward pass for visualization...")

    # Create output directory
    run_dir = None
    if config.visualization.save_outputs:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = f"data/outputs/runs/run_{timestamp}_{args.config_name}"
        os.makedirs(run_dir, exist_ok=True)
        # Copy the config file to the unified output directory
        shutil.copy(config_path, os.path.join(run_dir, "config.yaml"))
        print(f"\nCreated unified output directory: {run_dir}")

    # Plot loss history
    plot_loss_history(history_loss, config, run_dir=run_dir)

    result = forward_pipeline(
        initial_state,
        config.target,
        config.validity,
        config.physics,
        map_type=config.mapping.type,
        map_params=optimized_params,
        use_shirley_chiu=config.mapping.use_shirley_chiu,
        strict_boundary_fit=config.mapping.strict_boundary_fit
    )

    # Convert TargetConfig to dict for visualization compatibility
    target_params = {
        'type': config.target.type,
        'center': config.target.center,
        'radius': config.target.radius
    }

    visualize_pipeline_results(result, tessellation, config, target_params, args.config_name + "_trained", run_dir=run_dir)

    # ── Final Area Print ──────────────────────────────────────────────────────
    from jax_backend.geometry import compute_face_areas
    final_cnvs = result['valid_state'].centroid_node_vectors
    final_areas = compute_face_areas(final_cnvs)
    final_total_area = jnp.sum(final_areas)
    
    print("\n" + "─" * 40)
    print(f"[AREA CHECK] Final Material Area:   {final_total_area:.6f}")
    print(f"[AREA CHECK] Deviation:             {(final_total_area - initial_area):.6f}")
    print("─" * 40)

    print("\n" + "=" * 60)
    print("Training complete. Let's all pat ourselves on the back! 💕")
    print("=" * 60)
