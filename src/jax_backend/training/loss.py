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
from jax_backend.geometry import compute_total_area
from jax_backend.constraints import hinge_connectivity
from problem.targets import get_target_points
from problem.config import TargetConfig, PhysicsConfig, TrainingConfig, ValidityConfig

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
    target_params = {
        'type': target_cfg.type, 
        'center': target_cfg.center, 
        'radius': target_cfg.radius
    }
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
    chamfer_loss = precision_loss + training_cfg.loss_weights.coverage * coverage_loss
    
    # 4a. Global Material Conservation — computed upstream (see compute_end_to_end_loss).
    global_material_loss = 0.0

    # 3. Add Physical Energy Penalty (Decomposed Weights)
    # Each energy component has its own weight from LossWeights
    u_stretch = solution.energies['stretch'][-1]
    u_shear = solution.energies['shear'][-1]
    u_contact = solution.energies['contact'][-1]
    
    # Bending (rot) is not explicitly tracked in energies_dict yet, 
    # but we'll add it for completeness if present, or just use 0.0
    u_bend = solution.energies.get('rot', jnp.zeros_like(u_stretch))[-1]

    # 4. Final Aggregation
    weights = training_cfg.loss_weights
    
    physics_loss = (weights.stretching * u_stretch + 
                    weights.shearing * u_shear + 
                    weights.bending * u_bend + 
                    weights.contact * u_contact)
    
    geometric_loss = weights.chamfer * chamfer_loss + weights.material_area * global_material_loss
    loss = geometric_loss + physics_loss

    # 4. Detailed Metrics Assembly
    # We store the weighted values so the plotter reflects their actual impact.
    metrics = {
        "loss_total": loss,
        "loss_geometric": geometric_loss,
        "loss_physical": physics_loss,
        "comp_geom_chamfer": weights.chamfer * chamfer_loss,
        "comp_geom_material_area": weights.material_area * global_material_loss,
        "comp_phys_stretching": weights.stretching * u_stretch,
        "comp_phys_shearing": weights.shearing * u_shear,
        "comp_phys_bending": weights.bending * u_bend,
        "comp_phys_contact": weights.contact * u_contact,
        "comp_regularization": weights.regularization
    }
    
    # Backward compatibility for existing plotter keys
    metrics.update({
        'total': loss,
        'chamfer_total': chamfer_loss,
        'global_material_area': global_material_loss,
        'energy': u_stretch + u_shear + u_contact + u_bend
    })
    
    return loss, metrics


def compute_end_to_end_loss(
        map_params: Any,
        initial_state: CentroidalState,
        target_cfg: TargetConfig,
        validity_cfg: ValidityConfig,
        physics_cfg: PhysicsConfig,
        training_cfg: TrainingConfig,
        map_type: str = 'conformal_polynomial',
        use_shirley_chiu: bool = True,
        strict_boundary_fit: bool = True,
        learn_global_scale: bool = False,
        static_features: Any = None):
    """Wrapper function required by JAX for gradient computation.

    In JAX, jax.grad needs a single function that takes parameters and
    returns a scalar loss. This function chains the forward pass and the loss.

    Args:
        learn_global_scale: When True (Option 2), a learnable log_scale parameter
            is present in map_params and the area constraint is lifted — the
            optimizer is free to dilate/contract the structure.
            When False (Option 1), the total mapped area is penalised to stay
            equal to the reference flat-tessellation area.
    """
    
    # 1. Run the forward pipeline
    results = forward_pipeline(
        initial_state=initial_state,
        target_cfg=target_cfg,
        validity_cfg=validity_cfg,
        physics_cfg=physics_cfg,
        map_type=map_type,
        map_params=map_params,
        use_shirley_chiu=use_shirley_chiu,
        strict_boundary_fit=strict_boundary_fit,
        static_features=static_features,
    )
    
    # 2. Evaluate Physical Objective
    base_loss, loss_components = evaluate_physical_loss(
        results['solution'],
        results['valid_state'],
        target_cfg,
        training_cfg
    )

    # 2a. Hinge Gap Penalty — computed at Stage 0 output (before Stage 1).
    # Penalizes vertex pairs that should coincide at hinges but don't, giving the
    # GNN a direct gradient signal to produce geometrically connected tiles.
    # hinge_node_pairs is a static NumPy array captured in the CentroidalState.
    weight_hinge_gap = training_cfg.loss_weights.hinge_gap
    if weight_hinge_gap > 0.0:
        ms = results['mapped_state']
        hinge_gap_loss = weight_hinge_gap * hinge_connectivity(
            ms.face_centroids, ms.centroid_node_vectors, ms.hinge_node_pairs)
    else:
        hinge_gap_loss = 0.0

    # 2b. Material Area Conservation (Option 1 only: learn_global_scale = False)
    # For classical maps: CNVs are Jacobian-transformed in Stage 0 → use mapped_state.
    # For GNN maps: CNVs are NOT transformed in Stage 0 (only centroids move), so
    # mapped_state area is always ≈ initial area regardless of centroid positions.
    # Use valid_state (Stage 1 output) instead — Stage 1 resizes faces to fit the
    # GNN-displaced centroids, so its area reflects the actual geometric distortion.
    weight_material = training_cfg.loss_weights.material_area
    if not learn_global_scale and weight_material > 0.0:
        if map_type.startswith('gnn_'):
            area_cnvs = results['valid_state'].centroid_node_vectors
        else:
            area_cnvs = results['mapped_state'].centroid_node_vectors
        mapped_area = compute_total_area(area_cnvs)
        target_area = jnp.sum(initial_state.initial_face_areas)
        material_area_loss = weight_material * (mapped_area - target_area) ** 2
        area_deviation = mapped_area - target_area
    else:
        material_area_loss = 0.0
        area_deviation = 0.0

    # 3. Regularization (Prevents map_params from exploding)
    weight_reg = training_cfg.loss_weights.regularization

    # map_params is a JAX PyTree (dict with 'roots', 'tx', 'ty', etc.)
    squared_params = jax.tree_util.tree_map(lambda x: jnp.sum(x**2), map_params)
    reg_loss = weight_reg * jax.tree_util.tree_reduce(lambda a, b: a + b, squared_params, initializer=0.0)

    total_loss = base_loss + material_area_loss + hinge_gap_loss + reg_loss

    # Update metrics
    loss_components['comp_regularization'] = reg_loss
    loss_components['comp_geom_material_area'] = material_area_loss
    loss_components['global_material_area'] = area_deviation
    loss_components['hinge_gap'] = hinge_gap_loss
    loss_components['loss_total'] = total_loss
    loss_components['total'] = total_loss  # Backward compatibility
    
    return total_loss, loss_components
