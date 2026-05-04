"""
main_train.py — Neural Form-Finding Training Pipeline.

Optimizes the initial map parameters (e.g., polynomial coefficients)
using gradient descent through the differentiable physics pipeline.
"""

import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax
jax.config.update("jax_enable_x64", True)

import sys
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Add src to path for imports
sys.path.append(os.path.abspath('src'))

from src.topology.builder import build_tessellation
from src.problem.conditions import (
    apply_boundary_conditions, 
    apply_loads, 
    set_material_properties
)
from src.problem.config import load_config
from src.jax_backend.state import CentroidalState
from src.jax_backend.training.trainer import train_pipeline
from src.jax_backend.pipeline import forward_pipeline
from src.utils.pipeline_viz import visualize_pipeline_results
from src.utils.training_viz import plot_training_loss

if __name__ == "__main__":
    # Load configuration
    config_dir = "complex_mapping"
    config_name = "2_cs_asy_complex"
    config_path = f"data/configs/{config_dir}/{config_name}.yaml"
    config = load_config(config_path)
    print(f"Loaded configuration from {config_path}")

    target_params = {
        'type': config.target_type,
        'center': list(config.target_center),
        'radius': config.target_radius,
    }

    # Build tessellation
    print("\n" + "=" * 60)
    print("BUILDING TESSELLATION FOR TRAINING")
    print("=" * 60)
    tessellation = build_tessellation(config.pattern, config.width, config.height)
    set_material_properties(tessellation, config)
    apply_boundary_conditions(tessellation, config)
    apply_loads(tessellation, config)

    # Export to CentroidalState. We keep topological arrays as numpy arrays 
    # so JAX treats them as static constants inside the compiled loop.
    import numpy as np
    cs_dict = tessellation.to_centroidal_state()
    state_kwargs = {}
    for k, v in cs_dict.items():
        if k in ['face_centroids', 'centroid_node_vectors', 'load_values']:
            state_kwargs[k] = jnp.array(v)
        else:
            state_kwargs[k] = np.array(v)
            
    initial_state = CentroidalState(**state_kwargs)

    geom_weights = {
        'connectivity': config.w_connectivity,
        'non_intersection': config.w_non_intersection,
        'target': config.w_target,
        'arm_symmetry': config.w_arm_symmetry,
        'void_length': config.w_void_length,
        'void_collinear': config.w_void_collinear,
        'boundary_rigidity': config.w_boundary_rigidity,
    }

    pipeline_kwargs = {
        'scale_factor': config.scale_factor,
        'geom_weights': geom_weights,
        'use_contact': config.use_contact,
        'k_contact': config.k_contact,
        'min_angle': config.min_angle * jnp.pi / 180.0,
        'cutoff_angle': config.cutoff_angle * jnp.pi / 180.0,
        'linearized_strains': config.linearized_strains,
        'incremental': config.incremental,
        'num_load_steps': config.num_load_steps,
    }

    # Initial guess for polynomial parameters [scale, c1, c2, c3, c4]
    initial_map_params = jnp.array(config.map_params)

    print("\n" + "=" * 60)
    print("STARTING END-TO-END TRAINING")
    print("=" * 60)
    
    # Run training loop
    num_epochs = 300
    learning_rate = 0.01
    
    optimized_params, history_loss = train_pipeline(
        initial_map_params, 
        initial_state, 
        target_params, 
        pipeline_kwargs, 
        num_epochs=num_epochs, 
        lr=learning_rate
    )
    
    print("\nOptimization Complete!")
    print(f"Optimal Parameters: {optimized_params}")

    # Optionally, run the forward pipeline one last time with the optimized parameters
    # to visualize the results
    print("\nRunning final forward pass with optimized parameters for visualization...")
    
    pipeline_kwargs['map_type'] = 'conformal_polynomial'
    pipeline_kwargs['map_params'] = optimized_params
    
    result = forward_pipeline(initial_state, target_params, **pipeline_kwargs)
    
    visualize_pipeline_results(result, tessellation, config, target_params, config_name + "_trained")
    
    # Plot loss curve aesthetically
    plot_training_loss(history_loss, save_dir="data/outputs/runs/plots", show=config.save_plots)
    
    print("\n" + "=" * 60)
    print("Training complete. Let's all pat ourselves on the back! 💕")
    print("=" * 60)
