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

from src.topology.builder import build_tessellation
from src.problem.conditions import configure_tessellation
from src.problem.config import load_and_parse_config
from src.jax_backend.state import CentroidalState
from src.jax_backend.pipeline import forward_pipeline
from src.jax_backend.training.trainer import train_pipeline
from src.utils.pipeline_viz import visualize_pipeline_results
from src.utils.training_viz import plot_training_loss
from src.jax_backend.physics_solver.statics import LBFGS

if __name__ == "__main__":

    # ── Configuration ─────────────────────────────────────────────────────────
    import argparse
    parser = argparse.ArgumentParser(description="Neural Form-Finding Training.")
    parser.add_argument("--config-dir", type=str, default="complex_mapping")
    parser.add_argument("--config-name", type=str, required=True)
    args = parser.parse_args()

    config_path = f"data/configs/{args.config_dir}/{args.config_name}.yaml"
    config = load_and_parse_config(config_path)
    print(f"Loaded config: {config_path}")

    # Apply solver hyper‑parameters to LBFGS
    LBFGS.maxiter = config.physics.solver_maxiter
    LBFGS.tol = config.physics.solver_tol

    # ── Tessellation setup ────────────────────────────────────────────────────
    from types import SimpleNamespace
    topo = config.topology
    topo_obj = SimpleNamespace(**topo)
    
    tessellation = build_tessellation(topo.get('pattern'), 
                                      topo.get('width', 5), 
                                      topo.get('height', 5))
    configure_tessellation(tessellation, topo_obj)
    initial_state = CentroidalState.from_tessellation(tessellation)

    # ── Training ──────────────────────────────────────────────────────────────
    initial_map_params = jnp.array(topo.get('map_params', []))

    print("\n" + "=" * 60)
    print("STARTING END-TO-END TRAINING")
    print("=" * 60)

    optimized_params, history_loss = train_pipeline(
        initial_map_params,
        initial_state,
        config.target,
        config.physics,
        config.training
    )

    print(f"\nOptimization complete. Optimal params: {optimized_params}")

    # ── Final forward pass + visualization ────────────────────────────────────
    print("\nRunning final forward pass for visualization...")

    result = forward_pipeline(
        initial_state,
        config.target,
        config.physics,
        map_type='conformal_polynomial',
        map_params=optimized_params,
    )

    # Convert TargetConfig to dict for visualization compatibility
    target_params = {
        'type': config.target.type,
        'center': config.target.center,
        'radius': config.target.radius
    }

    visualize_pipeline_results(result, tessellation, config, target_params, args.config_name + "_trained")
    plot_training_loss(history_loss, save_dir="data/outputs/runs/plots", show=config.visualization.save_plots)

    print("\n" + "=" * 60)
    print("Training complete. Let's all pat ourselves on the back! 💕")
    print("=" * 60)
