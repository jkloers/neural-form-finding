"""
End-to-End Training Loop for Neural Form-Finding.

This module provides the training logic to optimize mapping parameters 
(or future GNN weights) using the differentiable physics pipeline.
"""

import jax
import jax.numpy as jnp
import optax

from jax_backend.training.loss import compute_end_to_end_loss

from problem.config import TargetConfig, PhysicsConfig, TrainingConfig, ValidityConfig

def create_train_step(initial_state, target_cfg: TargetConfig, validity_cfg: ValidityConfig, physics_cfg: PhysicsConfig, training_cfg: TrainingConfig, map_type: str = 'conformal_polynomial', use_shirley_chiu: bool = True, strict_boundary_fit: bool = True):
    """Creates a compiled training step for optimizing map_params.
    
    Args:
        initial_state: The flat tessellation CentroidalState.
        target_cfg: Structured TargetConfig.
        physics_cfg: Structured PhysicsConfig.
        training_cfg: Structured TrainingConfig.
        
    Returns:
        optimizer, train_step_fn
    """
    
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=training_cfg.learning_rate),
    )
    
    # The loss function closed over the constants
    def loss_fn(map_params):
        return compute_end_to_end_loss(
            map_params, initial_state, target_cfg, validity_cfg, physics_cfg, training_cfg, 
            map_type=map_type, use_shirley_chiu=use_shirley_chiu, 
            strict_boundary_fit=strict_boundary_fit
        )
        
    @jax.jit
    def train_step_fn(map_params, opt_state):
        (loss_val, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(map_params)

        leaves = jax.tree_util.tree_leaves(grads)
        global_grad_norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in leaves))
        grad_norms = jax.tree_util.tree_map(lambda g: jnp.sqrt(jnp.sum(g ** 2)), grads)
        aux['grad_norm'] = global_grad_norm
        aux['grad_norms'] = grad_norms

        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_map_params = optax.apply_updates(map_params, updates)

        return new_map_params, new_opt_state, loss_val, aux

    return optimizer, train_step_fn

def train_pipeline(initial_map_params, initial_state, target_cfg: TargetConfig, 
                   validity_cfg: ValidityConfig,
                   physics_cfg: PhysicsConfig, training_cfg: TrainingConfig, map_type: str = 'conformal_polynomial', use_shirley_chiu: bool = True, strict_boundary_fit: bool = True):
    """Run the training loop to find optimal mapping parameters."""
    
    optimizer, train_step_fn = create_train_step(
        initial_state, target_cfg, validity_cfg, physics_cfg, training_cfg, map_type, use_shirley_chiu, strict_boundary_fit
    )
    
    opt_state = optimizer.init(initial_map_params)
    current_params = initial_map_params
    
    history_loss = []
    
    for epoch in range(training_cfg.num_epochs):
        current_params, opt_state, loss, aux = train_step_fn(current_params, opt_state)
        history_loss.append(aux)
        
        if epoch % 5 == 0 or epoch == training_cfg.num_epochs - 1:
            grad_norm = float(aux['grad_norm'])
            flag = "  [VANISHING]" if grad_norm < 1e-6 else ("  [EXPLODING]" if grad_norm > 1e3 else "")
            per_param = " ".join(f"{k}={float(v):.2e}" for k, v in aux['grad_norms'].items())
            print(
                f"Epoch {epoch:03d} | Loss: {aux['total']:.4e} | "
                f"Chamfer: {aux['chamfer_total']:.4e} | Energy: {aux['energy']:.4e} | "
                f"‖grad‖: {grad_norm:.2e}{flag}"
            )
            print(f"         grads: {per_param}")
            
    return current_params, history_loss
