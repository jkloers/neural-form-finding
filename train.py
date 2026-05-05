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
from src.problem.config import load_config
from src.jax_backend.state import CentroidalState
from src.jax_backend.pipeline import forward_pipeline
from src.jax_backend.training.trainer import train_pipeline
from src.utils.pipeline_viz import visualize_pipeline_results
from src.utils.training_viz import plot_training_loss


if __name__ == "__main__":

    # ── Configuration ─────────────────────────────────────────────────────────
    config_path = "data/configs/complex_mapping/2_cs_asy_complex.yaml"
    config = load_config(config_path)
    config_name = os.path.splitext(os.path.basename(config_path))[0]
    print(f"Loaded config: {config_path}")

    target_params = {
        'type':   config.target_type,
        'center': list(config.target_center),
        'radius': config.target_radius,
    }

    # ── Tessellation setup ────────────────────────────────────────────────────
    tessellation = build_tessellation(config.pattern, config.width, config.height)
    configure_tessellation(tessellation, config)
    initial_state = CentroidalState.from_tessellation(tessellation)

    # ── Pipeline parameters ───────────────────────────────────────────────────
    geom_weights = {
        'connectivity':       config.w_connectivity,
        'non_intersection':   config.w_non_intersection,
        'target':             config.w_target,
        'arm_symmetry':       config.w_arm_symmetry,
        'void_length':        config.w_void_length,
        'void_collinear':     config.w_void_collinear,
        'boundary_rigidity':  config.w_boundary_rigidity,
    }

    pipeline_kwargs = {
        'scale_factor':      config.scale_factor,
        'geom_weights':      geom_weights,
        'use_contact':       config.use_contact,
        'k_contact':         config.k_contact,
        'min_angle':         config.min_angle * jnp.pi / 180.0,
        'cutoff_angle':      config.cutoff_angle * jnp.pi / 180.0,
        'linearized_strains': config.linearized_strains,
        'incremental':       config.incremental,
        'num_load_steps':    config.num_load_steps,
    }

    # ── Training ──────────────────────────────────────────────────────────────
    initial_map_params = jnp.array(config.map_params)

    print("\n" + "=" * 60)
    print("STARTING END-TO-END TRAINING")
    print("=" * 60)

    optimized_params, history_loss = train_pipeline(
        initial_map_params,
        initial_state,
        target_params,
        pipeline_kwargs,
        num_epochs=500,
        lr=0.01,
    )

    print(f"\nOptimization complete. Optimal params: {optimized_params}")

    # ── Final forward pass + visualization ────────────────────────────────────
    print("\nRunning final forward pass for visualization...")

    result = forward_pipeline(
        initial_state, target_params,
        map_type='conformal_polynomial',
        map_params=optimized_params,
        **pipeline_kwargs,
    )

    visualize_pipeline_results(result, tessellation, config, target_params, config_name + "_trained")
    plot_training_loss(history_loss, save_dir="data/outputs/runs/plots", show=config.save_plots)

    print("\n" + "=" * 60)
    print("Training complete. Let's all pat ourselves on the back! 💕")
    print("=" * 60)
