"""
End-to-End Physical Loss functions for Neural Form-Finding.

This module defines the high-level objective functions that take as input the 
learnable mapping parameters (or GNN weights) and return a scalar loss 
after passing through the fully differentiable pipeline.
"""

import jax
import jax.numpy as jnp

from jax_backend.centroidal.pipeline import forward_pipeline
from jax_backend.centroidal.state import CentroidalState

def compute_end_to_end_physical_loss(
        map_params: jnp.ndarray,
        initial_state: CentroidalState,
        target_params: dict,
        pipeline_kwargs: dict) -> float:
    """Compute the physical loss for a given set of mapping parameters.
    
    This function wraps the entire forward pipeline. Because we use JAXopt with 
    implicit differentiation in the geometric validity and physics solvers, 
    this entire function is differentiable with respect to `map_params`.
    
    Args:
        map_params: The trainable parameters for the initial mapping 
                    (e.g., polynomial coefficients or GNN weights).
        initial_state: The flat tessellation CentroidalState.
        target_params: Configuration dict for the target shape (center, radius).
        pipeline_kwargs: Additional arguments for `forward_pipeline` 
                         (e.g., geom_weights, loads, use_contact).
                         
    Returns:
        Scalar loss value.
    """
    
    # 1. Forward Pass: Run the pipeline using the trainable map_params
    # We force the map_type to our parameterized mapping
    pipeline_kwargs['map_type'] = 'conformal_polynomial'
    pipeline_kwargs['map_params'] = map_params
    
    results = forward_pipeline(initial_state, target_params, **pipeline_kwargs)
    solution = results['solution']
    
    # 2. Extract Final Physical State
    # Here we look at the final displacement field after incremental loading
    final_displacements = solution.fields[-1]
    final_positions = initial_state.face_centroids + final_displacements[:, :2]
    
    # 3. Define the Physical Loss
    # Example A: Match target shape under maximum load (e.g. Chamfer distance)
    # R_target = target_params.get('radius', 1.0)
    # dists = jnp.linalg.norm(final_positions - jnp.array(target_params['center']), axis=1)
    # loss = jnp.mean((dists - R_target)**2)
    
    # Example B: Minimize total strain energy (maximize stiffness) under load
    total_energy = solution.energies['total'][-1]
    
    # For now, let's optimize for maximum stiffness (minimum energy)
    # plus a small regularization on the mapping parameters
    loss = total_energy + 1e-3 * jnp.sum(map_params**2)
    
    return loss
