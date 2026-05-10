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
from src.utils.training_viz import plot_training_loss

if __name__ == "__main__":

    # ── Configuration ─────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="Neural Form-Finding Training.")
    parser.add_argument("--config-dir", type=str, default="asymmetric_roots")
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
    initial_state = CentroidalState.from_tessellation(tessellation, target_cfg=config.target)

    # ── Training ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STARTING END-TO-END TRAINING")
    print("=" * 60)

    initial_map_params = config.mapping.params

    optimized_params, history_loss = train_pipeline(
        initial_map_params,
        initial_state,
        config.target,
        config.validity,
        config.physics,
        config.training,
        map_type=config.mapping.type,
        use_shirley_chiu=config.mapping.use_shirley_chiu,
        strict_boundary_fit=config.mapping.strict_boundary_fit
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

    print("\n" + "=" * 60)
    print("Training complete. Let's all pat ourselves on the back! 💕")
    print("=" * 60)
