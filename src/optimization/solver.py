import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize
from jax_backend.objective import compute_objective_contracted, compute_objective_deployed
from jax_backend.pytrees import TessellationState

def solve_form_finding_deployed(initial_state: TessellationState, target_params, max_iter=500):
    """
    Résout le problème de form-finding en utilisant Scipy L-BFGS-B 
    et les gradients calculés par JAX.
    """

    x0 = np.array(initial_state.X).flatten()
    dim = initial_state.X.shape[1]

    @jax.jit
    def val_and_grad_fn(x_flat):
        X_reshaped = x_flat.reshape(-1, dim)
        # On calcule la valeur et le gradient par rapport à X
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

def solve_form_finding_contracted(initial_state: TessellationState, target_params, max_iter=500):
    """
    Résout le problème de form-finding en utilisant Scipy L-BFGS-B 
    et les gradients calculés par JAX.
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


    