import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize
from jax_backend.objective import compute_objective_contracted, compute_objective_deployed
from jax_backend.pytrees import TessellationState

def solve_form_finding_deployed(initial_state: TessellationState, target_params, max_iter=500):
    """
    Solves the form-finding problem using Scipy L-BFGS-B 
    and JAX-computed gradients.
    """

    x0 = np.array(initial_state.X).flatten()
    dim = initial_state.X.shape[1]

    @jax.jit
    def val_and_grad_fn(x_flat):
        X_reshaped = x_flat.reshape(-1, dim)
        # Compute value and gradient with respect to X
        val, grad = jax.value_and_grad(compute_objective_deployed)(X_reshaped, initial_state, target_params)
        return val, grad.flatten()

    def scipy_wrapper(x):
        v, g = val_and_grad_fn(x)
        return np.array(v, dtype=np.float64), np.array(g, dtype=np.float64)

    print("Starting L-BFGS optimization...")
    res = minimize(
        fun=scipy_wrapper,
        x0=x0,
        jac=True,
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'disp': True}
    )

    final_X = res.x.reshape(-1, dim)
    return initial_state._replace(X=final_X), res

def solve_form_finding_contracted(initial_state: TessellationState, target_params, max_iter=500, save_every=5):
    """
    Solves the form-finding problem for the contracted shape using Scipy L-BFGS-B 
    and JAX-computed gradients. Keeps a history of states for animation.
    """

    x0 = np.array(initial_state.X).flatten()
    dim = initial_state.X.shape[1]

    @jax.jit
    def val_and_grad_fn(x_flat):
        X_reshaped = x_flat.reshape(-1, dim)
        val, grad = jax.value_and_grad(compute_objective_contracted)(X_reshaped, initial_state, target_params)
        return val, grad.flatten()

    def scipy_wrapper(x):
        v, g = val_and_grad_fn(x)
        return np.array(v, dtype=np.float64), np.array(g, dtype=np.float64)

    # Initialize history with the initial state (the deployed shape)
    history = {'energy': [], 'states': [x0.reshape(-1, dim).copy()]}
    
    # Evaluate initial energy for the history
    v0, _ = val_and_grad_fn(x0)
    history['energy'].append(float(v0))
    
    iter_count = [0]
    
    def callback(xk):
        iter_count[0] += 1
        v, _ = val_and_grad_fn(xk)
        history['energy'].append(float(v))
        
        if iter_count[0] % save_every == 0:
            history['states'].append(xk.reshape(-1, dim).copy())

    print("Starting L-BFGS optimization...")
    res = minimize(
        fun=scipy_wrapper,
        x0=x0,
        jac=True,
        method='L-BFGS-B',
        callback=callback,
        options={'maxiter': max_iter, 'disp': True}
    )

    final_X = res.x.reshape(-1, dim)
    
    # Ensure that the very last perfectly contracted state is included in the animation
    if iter_count[0] % save_every != 0:
        history['states'].append(final_X.copy())
        
    return initial_state._replace(X=final_X), res, history


    