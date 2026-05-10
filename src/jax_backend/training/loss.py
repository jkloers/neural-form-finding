"""
End-to-End Physical Loss functions for Neural Form-Finding.

This module defines the high-level objective functions that take as input the 
learnable mapping parameters (or GNN weights) and return a scalar loss 
after passing through the fully differentiable pipeline.
"""

import jax
import jax.numpy as jnp

from typing import Any
from jax_backend.pipeline import forward_pipeline
from jax_backend.state import CentroidalState
from problem.targets import get_target_points
from problem.config import TargetConfig, PhysicsConfig, TrainingConfig

def evaluate_physical_loss(solution, valid_state, target_cfg: TargetConfig, training_cfg: TrainingConfig):
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
    final_centroids = valid_state.face_centroids + final_displacements[:, :2]
    final_thetas = final_displacements[:, 2] # Initial theta is 0 in centroidal coords
    
    # 2. Reconstruct Geometric Points for matching
    loss_type = training_cfg.geometric_loss_type
    
    if loss_type == "boundary_vertices":
        # Use the actual exterior vertices of the boundary faces
        b_face_ids = valid_state.boundary_face_node_ids[:, 0]
        b_local_node_ids = valid_state.boundary_face_node_ids[:, 1]
        
        b_centroids = final_centroids[b_face_ids]
        b_vectors = valid_state.centroid_node_vectors[b_face_ids, b_local_node_ids]
        b_thetas = final_thetas[b_face_ids]
        
        cos_t = jnp.cos(b_thetas)
        sin_t = jnp.sin(b_thetas)
        
        rotated_vectors = jnp.stack([
            cos_t * b_vectors[:, 0] - sin_t * b_vectors[:, 1],
            sin_t * b_vectors[:, 0] + cos_t * b_vectors[:, 1]
        ], axis=-1)
        
        geometric_positions = b_centroids + rotated_vectors
    else:
        # Fallback to centroids (faster but more approximate)
        b_faces = valid_state.boundary_face_node_ids[:, 0]
        geometric_positions = final_centroids[b_faces]
    
    # 3. Compute Geometric Loss (Chamfer distance)
    # Generate the 2D point cloud for the target
    target_params = {'type': target_cfg.type, 'center': target_cfg.center, 'radius': target_cfg.radius}
    target_cloud = get_target_points(target_params, n_points=500)
    target_cloud = jnp.asarray(target_cloud, dtype=float)
    
    # Compute pairwise squared distances: (N_boundary, M_target)
    dist_matrix = jnp.sum((geometric_positions[:, None, :] - target_cloud[None, :, :])**2, axis=-1)
    
    # 1. Distance from each boundary point to the closest target point (Precision)
    min_to_target = jnp.min(dist_matrix, axis=1)
    precision_loss = jnp.mean(min_to_target)
    
    # 2. Distance from each target point to the closest boundary point (Coverage)
    # This prevents the "collapse to a single point" issue. All target points 
    # want to be satisfied (i.e. have a structure point close to them).
    min_to_structure = jnp.min(dist_matrix, axis=0)
    coverage_loss = jnp.mean(min_to_structure)
    
    # Combine them. We allow tuning the coverage weight in the config.
    coverage_weight = training_cfg.loss_weights.get("coverage", 1.0)
    chamfer_loss = precision_loss + coverage_weight * coverage_loss
    
    # 3. Add Physical Energy Penalty
    # We want to minimize the internal strain energy (stretch + shear + rot + contact)
    # This heavily penalizes solutions that are physically invalid (e.g. extreme stretching,
    # intersections, or massive contact forces).
    u_int = (solution.energies['stretch'][-1] + 
             solution.energies['shear'][-1] + 
             solution.energies['contact'][-1])
             
    # 4. Final Aggregation
    loss_weights = training_cfg.loss_weights
    
    # Weight for the energy loss
    weight_physics = loss_weights.get("physics", 0.1)
    energy_loss = weight_physics * u_int
    
    weight_geometric = loss_weights.get("geometric", 1.0)
    loss = (weight_geometric * chamfer_loss) + energy_loss
    
    loss_components = {
        'chamfer_precision': precision_loss,
        'chamfer_coverage': coverage_loss,
        'chamfer_total': chamfer_loss,
        'energy': energy_loss
    }
    
    return loss, loss_components


def compute_end_to_end_loss(
        map_params: Any, 
        initial_state: CentroidalState, 
        target_cfg: TargetConfig, 
        physics_cfg: PhysicsConfig, 
        training_cfg: TrainingConfig, 
        map_type: str = 'conformal_polynomial', 
        use_shirley_chiu: bool = True, 
        strict_boundary_fit: bool = True, 
        initial_scale_factor: float = 1.0):
    """Wrapper function required by JAX for gradient computation.
    
    In JAX, jax.grad needs a single function that takes parameters and 
    returns a scalar loss. This function chains the forward pass and the loss.
    """
    
    # 1. Run the forward pipeline
    results = forward_pipeline(
        initial_state=initial_state,
        target_cfg=target_cfg,
        physics_cfg=physics_cfg,
        map_type=map_type,
        map_params=map_params,
        use_shirley_chiu=use_shirley_chiu,
        strict_boundary_fit=strict_boundary_fit,
        initial_scale_factor=initial_scale_factor
    )
    
    # 2. Evaluate Physical Objective
    base_loss, loss_components = evaluate_physical_loss(
        results['solution'], 
        results['valid_state'], 
        target_cfg, 
        training_cfg
    )
    
    # 3. Regularization (Optional, prevents map_params from exploding)
    weight_reg = training_cfg.loss_weights.get("regularization", 1e-3)
    
    # map_params can be a JAX PyTree (dict or array)
    squared_params = jax.tree_util.tree_map(lambda x: jnp.sum(x**2), map_params)
    reg_loss = weight_reg * jax.tree_util.tree_reduce(lambda a, b: a + b, squared_params, initializer=0.0)
    
    total_loss = base_loss + reg_loss
    loss_components['reg'] = reg_loss
    loss_components['total'] = total_loss
    
    return total_loss, loss_components
