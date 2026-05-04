"""
End-to-End Training Loop for Neural Form-Finding.

This module provides the training logic to optimize mapping parameters 
(or future GNN weights) using the differentiable physics pipeline.
"""

import jax
import jax.numpy as jnp
import optax

from jax_backend.training.loss import compute_end_to_end_loss

def create_train_step(initial_state, target_params, pipeline_kwargs, learning_rate=0.01):
    """Creates a compiled training step for optimizing map_params.
    
    Args:
        initial_state: The flat tessellation CentroidalState.
        target_params: Target shape config.
        pipeline_kwargs: Keyword arguments for forward_pipeline.
        learning_rate: Learning rate for Adam optimizer.
        
    Returns:
        optimizer, opt_state, train_step_fn
    """
    
    # We use Adam for smooth optimization
    optimizer = optax.adam(learning_rate=learning_rate)
    
    # The loss function closed over the constants
    def loss_fn(map_params):
        return compute_end_to_end_loss(
            map_params, initial_state, target_params, pipeline_kwargs
        )
        
    @jax.jit
    def train_step(map_params, opt_state):
        # jax.value_and_grad computes the loss and the gradients 
        # end-to-end through the implicit physical solvers!
        (loss_val, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(map_params)
        
        updates, opt_state = optimizer.update(grads, opt_state)
        new_map_params = optax.apply_updates(map_params, updates)
        
        return new_map_params, opt_state, loss_val, aux

    return optimizer, train_step

def train_pipeline(initial_map_params, initial_state, target_params, pipeline_kwargs, num_epochs=50, lr=0.01):
    """Run the training loop to find optimal mapping parameters."""
    
    optimizer, train_step = create_train_step(
        initial_state, target_params, pipeline_kwargs, learning_rate=lr
    )
    
    opt_state = optimizer.init(initial_map_params)
    current_params = initial_map_params
    
    history_loss = []
    
    for epoch in range(num_epochs):
        current_params, opt_state, loss, aux = train_step(current_params, opt_state)
        history_loss.append(aux)
        
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch:03d} | Total Loss: {aux['total']:.6f} | Chamfer: {aux['chamfer']:.6f} | Energy: {aux['energy']:.6f}")
            
    return current_params, history_loss
