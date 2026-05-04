"""
End-to-End Physical Loss functions for Neural Form-Finding.

This module defines the high-level objective functions that take as input the 
learnable mapping parameters (or GNN weights) and return a scalar loss 
after passing through the fully differentiable pipeline.
"""

import jax
import jax.numpy as jnp

from jax_backend.pipeline import forward_pipeline
from jax_backend.state import CentroidalState
from problem.targets import get_target_points

def evaluate_physical_loss(solution, valid_state, target_params=None):
    """Computes the pure physical objective from the pipeline results.
    
    This is independent of HOW the results were generated. It only looks at 
    the physical state (energies, displacements) to calculate a score.
    
    Args:
        solution: SolutionData containing the equilibrium fields and energies.
        valid_state: The CentroidalState right before physical loading.
        target_params: Optional dict for target shape objectives.
        
    Returns:
        Scalar loss value.
    """
    
    # 1. Extract Final Physical State
    final_displacements = solution.fields[-1]
    final_positions = valid_state.face_centroids + final_displacements[:, :2]
    
    # --- CHOIX DE LA PERTE PHYSIQUE ---
    
    # Example A: Match target shape under load (Chamfer distance)
    # R_target = target_params.get('radius', 1.0)
    # dists = jnp.linalg.norm(final_positions - jnp.array(target_params['center']), axis=1)
    # loss = jnp.mean((dists - R_target)**2)
    
    # # Example B: Minimize total strain energy (maximize stiffness)
    # total_energy = solution.energies['total'][-1]
    # loss = total_energy

    # Example C: Match target shape under load (Chamfer distance)
    # Extract only the boundary faces (we only want the exterior to match the target)
    b_faces = valid_state.boundary_face_node_ids[:, 0]
    boundary_positions = final_positions[b_faces]
    
    # Generate the 2D point cloud for the target
    target_cloud = get_target_points(target_params, n_points=500)
    target_cloud = jnp.asarray(target_cloud, dtype=float)
    
    # Compute pairwise squared distances: (N_boundary, M_target)
    dist_matrix = jnp.sum((boundary_positions[:, None, :] - target_cloud[None, :, :])**2, axis=-1)
    
    # Bidirectional Chamfer distance was forcing the structure to scale up 
    # to cover the ENTIRE circle. We only want the structure to lie ON the circle.
    # 1. Distance from each boundary point to the closest target point
    min_to_target = jnp.min(dist_matrix, axis=1)
    
    chamfer_loss = jnp.mean(min_to_target)
    
    # 3. Add Physical Energy Penalty
    # We want to minimize the internal strain energy (stretch + shear + rot + contact)
    # This heavily penalizes solutions that are physically invalid (e.g. extreme stretching,
    # intersections, or massive contact forces).
    u_int = (solution.energies['stretch'][-1] + 
             solution.energies['shear'][-1] + 
             solution.energies['rot'][-1] + 
             solution.energies['contact'][-1])
             
    # Weight for the energy loss (needs to be tuned so it doesn't overpower the shape loss)
    # Contact energy can be huge if there's intersection, which is exactly what we want to penalize.
    energy_loss = 1e-1 * u_int
    
    loss = chamfer_loss + energy_loss
    # # Example D: Minimize final displacement field magnitude
    # loss = jnp.mean(jnp.linalg.norm(final_displacements[:, :2], axis=1))
    
    loss = chamfer_loss + energy_loss
    
    loss_components = {
        'chamfer': chamfer_loss,
        'energy': energy_loss
    }
    
    return loss, loss_components


def compute_end_to_end_loss(map_params: jnp.ndarray, initial_state: CentroidalState, target_params: dict, pipeline_kwargs: dict) -> float:
    """Wrapper function required by JAX for gradient computation.
    
    In JAX, jax.grad needs a single function that takes parameters and 
    returns a scalar loss. This function chains the forward pass and the loss.
    """
    
    # 1. Forward Pass
    pipeline_kwargs['map_type'] = 'conformal_polynomial'
    pipeline_kwargs['map_params'] = map_params
    results = forward_pipeline(initial_state, target_params, **pipeline_kwargs)
    
    # 2. Evaluate Physical Objective
    base_loss, loss_components = evaluate_physical_loss(results['solution'], results['valid_state'], target_params)
    
    # 3. Regularization (Optional, prevents map_params from exploding)
    reg_loss = 1e-3 * jnp.sum(map_params**2)
    
    total_loss = base_loss + reg_loss
    loss_components['reg'] = reg_loss
    loss_components['total'] = total_loss
    
    return total_loss, loss_components
