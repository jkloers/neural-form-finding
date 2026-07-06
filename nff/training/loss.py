"""
End-to-End Physical Loss functions for Neural Form-Finding.

This module defines the high-level objective functions that take as input the 
learnable mapping parameters (or GNN weights) and return a scalar loss 
after passing through the fully differentiable pipeline.
"""

import jax
import jax.numpy as jnp

from typing import Any, Optional
from jaxtyping import Array, Float
from nff.stages.pipeline import forward_pipeline
from nff.stages.physics.kinematics import face_to_node_kinematics_fn
from nff.stages.state import CentroidalState
from nff.stages.geometry import compute_total_area, compute_void_area
from nff.stages.constraints import hinge_connectivity
from nff.config.targets import get_target_points
from nff.config.experiment import TargetConfig, PhysicsConfig, TrainingConfig, ValidityConfig

def _circularity_loss(pts: Float[Array, "n 2"]) -> Float[Array, ""]:
    """Scale/translation-invariant circularity loss for a boundary point cloud.

    Fits the best circle algebraically (Kåsa: solve for the center and radius
    that make the points equidistant from one unknown point), then returns the
    mean squared *relative* radial residual. Zero iff the points lie on a circle
    of any size/position — so the target circle's size adapts to the
    tessellation instead of being fixed.

    Args:
        pts: (n, 2) deployed boundary vertices.

    Returns:
        Scalar circularity loss.
    """
    n = pts.shape[0]
    A = jnp.concatenate([2.0 * pts, jnp.ones((n, 1))], axis=1)        # (n, 3)
    rhs = jnp.sum(pts ** 2, axis=1)                                   # (n,)
    coeffs = jnp.linalg.solve(A.T @ A + 1e-8 * jnp.eye(3), A.T @ rhs)  # [cx, cy, c]
    center = coeffs[:2]
    radius = jnp.sqrt(jnp.clip(coeffs[2] + jnp.sum(center ** 2), 1e-12, None))
    dist = jnp.linalg.norm(pts - center[None, :], axis=1)
    return jnp.mean(((dist - radius) / radius) ** 2)


def evaluate_physical_loss(
        solution,
        valid_state,
        target_cfg: TargetConfig,
        training_cfg: TrainingConfig,
        target_cloud: Float[Array, "n_target 2"],
) -> tuple[Float[Array, ""], dict]:
    """Computes the pure physical objective from the pipeline results.

    Args:
        solution:      SolutionData containing the equilibrium fields and energies.
        valid_state:   The CentroidalState right before physical loading.
        target_cfg:    Structured TargetConfig (used only for fallback — prefer passing target_cloud).
        training_cfg:  Structured TrainingConfig with loss weights.
        target_cloud:  (n_target, 2) JAX array of target boundary points, precomputed
                       before the JIT boundary to avoid rebaking at each trace.

    Returns:
        (loss, metrics_dict)
    """
    
    # 1. Extract Final Physical State
    final_displacements = solution.fields[-1]
    final_centroids = valid_state.face_centroids + final_displacements[:, :2]
    final_thetas = final_displacements[:, 2] # Initial theta is 0 in centroidal coords
    
    # 2. Reconstruct Geometric Points for matching
    loss_type = training_cfg.geometric_loss_type
    
    if loss_type in ("boundary_vertices", "circle_fit"):
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
    
    # 3. Compute Geometric Loss
    if loss_type == "circle_fit":
        # Size/position-free objective: make the deployed boundary lie on a
        # circle whose center and radius are fit to the points themselves.
        chamfer_loss = _circularity_loss(geometric_positions)
    else:
        # Chamfer distance to a fixed target cloud (precomputed before JIT).
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
        static_features: Any = None,
        load_specs: Any = None,
        target_cloud: Optional[Float[Array, "n_target 2"]] = None,
        bond_energy_fn=None,
        stability_fn=None,
        hinge_geometry_fn=None,
) -> tuple[Float[Array, ""], dict]:
    """Chains the forward pipeline with the loss, as required by jax.value_and_grad.

    Args:
        learn_global_scale: When True, a learnable log_scale is present in map_params
            and the area constraint is lifted. When False, the total mapped area is
            penalised to stay equal to the reference flat-tessellation area.
        target_cloud: Precomputed (n_target, 2) target boundary points. Pass this from
            outside the JIT boundary (e.g. from create_train_step) to avoid rebaking
            the numpy→JAX conversion at every trace. If None, computed internally.
    """
    if target_cloud is None:
        target_params = {'type': target_cfg.type, 'center': target_cfg.center, 'radius': target_cfg.radius}
        target_cloud = jnp.asarray(get_target_points(target_params, n_points=500), dtype=jnp.float64)

    # Design-tracked hinge geometry: the surrogate's per-hinge HingeGeometry(w_lig, alpha, sec_dir) is a
    # deterministic function of the design, recomputed each step and threaded to the solver via
    # control_params (NOT closed over) so jaxopt's implicit diff carries d(loss)/d(design) through the
    # whole geometry. None -> ROM / spring energy (no surrogate).
    hinge_geometry = hinge_geometry_fn(map_params) if hinge_geometry_fn is not None else None

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
        load_specs=load_specs,
        bond_energy_fn=bond_energy_fn,
        hinge_geometry=hinge_geometry,
    )

    # 2. Evaluate Physical Objective
    base_loss, base_metrics = evaluate_physical_loss(
        results['solution'],
        results['valid_state'],
        target_cfg,
        training_cfg,
        target_cloud,
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

    # 2c. Openness loss — reward large void area at Stage 1 (before physics).
    # log1p saturation: gradient ∝ 1/(1+void_area), so chamfer dominates at
    # large void and the tessellation cannot expand without bound.
    # nan_to_num guards against Stage 1 solver divergence on hard problems.
    weight_open = training_cfg.loss_weights.openness
    if weight_open > 0.0:
        vs = results['valid_state']
        void_area = compute_void_area(
            vs.face_centroids, vs.centroid_node_vectors, vs.boundary_face_node_ids)
        void_area_safe = jnp.clip(
            jnp.nan_to_num(void_area, nan=0.0, posinf=0.0, neginf=0.0),
            0.0, 20.0)
        openness_loss = -weight_open * jnp.log1p(void_area_safe)
    else:
        openness_loss = 0.0

    # Deformation reward: mean squared face displacement across all DOFs.
    # Bounded by geometry (displacement can't exceed tessellation size), so log1p
    # saturation is well-behaved and nan_to_num guards solver divergence.
    weight_deform = training_cfg.loss_weights.deformation
    if weight_deform > 0.0:
        final_displacements = results['solution'].fields[-1]   # (n_faces, 3)
        mean_sq_disp = jnp.mean(jnp.sum(final_displacements ** 2, axis=-1))
        disp_safe = jnp.clip(
            jnp.nan_to_num(mean_sq_disp, nan=0.0, posinf=0.0, neginf=0.0),
            0.0, 10.0)
        deformation_loss = -weight_deform * jnp.log1p(disp_safe)
    else:
        deformation_loss = 0.0

    # 2d. Void closing terms — can be used independently or together.
    #
    # void_closure  (+w * log1p(void_stage2)):
    #   Penalty on remaining void area after loading. No Stage 1 reference.
    #   The model is penalised purely for open voids in the final state —
    #   cannot cheat by inflating the starting void.
    #
    # closure_delta (-w * log1p(max(0, void_stage1 - void_stage2))):
    #   Reward for the decrease in void area from Stage 1 to Stage 2.
    #   Rewards actual closing relative to the reference — a rigid-body swing
    #   leaves void area invariant (delta=0, no reward).
    #
    # Both together: void_closure ensures the final state is closed; closure_delta
    # ensures the loads are actually doing the closing (not starting already closed).
    # Gradients flow through Stage 2 (lax.scan) and Stage 1 (fori_loop) to Stage 0.

    weight_void_closure = training_cfg.loss_weights.void_closure
    weight_closure_delta = training_cfg.loss_weights.closure_delta
    need_stage2_void = (weight_void_closure > 0.0 or weight_closure_delta > 0.0)
    need_stage1_void = weight_closure_delta > 0.0

    if need_stage2_void:
        vs = results['valid_state']
        final_displacements = results['solution'].fields[-1]   # (n_faces, 3)
        deformed_centroids = vs.face_centroids + final_displacements[:, :2]
        thetas = final_displacements[:, 2]
        cos_t = jnp.cos(thetas)
        sin_t = jnp.sin(thetas)
        cnv = vs.centroid_node_vectors
        deformed_cnv = jnp.stack([
            cos_t[:, None] * cnv[:, :, 0] - sin_t[:, None] * cnv[:, :, 1],
            sin_t[:, None] * cnv[:, :, 0] + cos_t[:, None] * cnv[:, :, 1],
        ], axis=-1)
        void_stage2 = compute_void_area(
            deformed_centroids, deformed_cnv, vs.boundary_face_node_ids)
        void_stage2_safe = jnp.clip(
            jnp.nan_to_num(void_stage2, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 20.0)

        void_closure_loss = (weight_void_closure * jnp.log1p(void_stage2_safe)
                             if weight_void_closure > 0.0 else 0.0)

        if need_stage1_void:
            void_stage1 = compute_void_area(
                vs.face_centroids, vs.centroid_node_vectors, vs.boundary_face_node_ids)
            void_stage1_safe = jnp.clip(
                jnp.nan_to_num(void_stage1, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 20.0)
            delta = jnp.clip(void_stage1_safe - void_stage2_safe, 0.0, 20.0)
            closure_delta_loss = -weight_closure_delta * jnp.log1p(delta)
        else:
            closure_delta_loss = 0.0
    else:
        void_closure_loss = 0.0
        closure_delta_loss = 0.0

    # 3. Regularization (Prevents map_params from exploding)
    weight_reg = training_cfg.loss_weights.regularization

    # map_params is a JAX PyTree (dict with 'roots', 'tx', 'ty', etc.)
    squared_params = jax.tree_util.tree_map(lambda x: jnp.sum(x**2), map_params)
    reg_loss = weight_reg * jax.tree_util.tree_reduce(lambda a, b: a + b, squared_params, initializer=0.0)

    # 4. Physical-stability penalty (surrogate failure-margin + OOD), at the DEPLOYED state.
    # Chamfer is blind to structural safety; this gives the design gradient a component that
    # resists walking the shape-equivalent set into near-failure / ill-conditioned configs.
    stab_loss = jnp.asarray(0.0, dtype=jnp.float64)
    stab_metrics = {}
    if stability_fn is not None:
        # bond_connectivity indexes NODES (n_faces*n_nodes_per_face), so map face displacements
        # through the rigid-tile kinematics FIRST -- exactly as strain_energy_fn does -- else the
        # stability gather is wrong (JAX silently clamps out-of-bounds indices).
        _cnv = results['valid_state'].centroid_node_vectors
        _nf, _nn, _ = _cnv.shape
        _node_disp = face_to_node_kinematics_fn(results['solution'].fields[-1], _cnv).reshape(_nf * _nn, 3)
        # Pass the per-hinge reference bond vectors so the margin's (a, s) reduction is frame-invariant
        # and identical to the bond energy's (matters for open bonds; ~0 correction for closed hinges).
        stab_loss, stab_metrics = stability_fn(_node_disp, hinge_geometry, results['reference_bond_vectors'])

    total_loss = (base_loss + material_area_loss + hinge_gap_loss + reg_loss
                  + openness_loss + deformation_loss + void_closure_loss + closure_delta_loss
                  + stab_loss)

    all_metrics = {
        **base_metrics,
        'comp_regularization':    reg_loss,
        'comp_geom_material_area': material_area_loss,
        'global_material_area':   area_deviation,
        'hinge_gap':              hinge_gap_loss,
        'openness':               openness_loss,
        'deformation':            deformation_loss,
        'void_closure':           void_closure_loss,
        'closure_delta':          closure_delta_loss,
        'comp_stability':         stab_loss,
        **stab_metrics,
        'loss_total':             total_loss,
        'total':                  total_loss,
    }
    return total_loss, all_metrics
