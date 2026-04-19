import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize
from jax_backend.objective import compute_objective
from jax_backend.pytrees import TessellationState

def solve_form_finding(initial_state: TessellationState, target_params, max_iter=500):
    """
    Résout le problème de form-finding en utilisant Scipy L-BFGS-B 
    et les gradients calculés par JAX.
    """
    
    # 1. On prépare les données statiques (la topologie ne change pas)
    # On aplatit X pour Scipy (qui travaille sur des vecteurs 1D)
    x0 = np.array(initial_state.X).flatten()
    dim = initial_state.X.shape[1]

    # 2. Création de la fonction de coût JAX (JIT-compilée)
    # On définit une fonction qui prend un vecteur plat et retourne (valeur, gradient)
    @jax.jit
    def val_and_grad_fn(x_flat):
        X_reshaped = x_flat.reshape(-1, dim)
        # On calcule la valeur et le gradient par rapport à X
        val, grad = jax.value_and_grad(compute_objective)(X_reshaped, initial_state, target_params)
        return val, grad.flatten()

    # 3. Wrapper pour Scipy (conversion jnp.array -> np.array)
    def scipy_wrapper(x):
        v, g = val_and_grad_fn(x)
        return np.array(v, dtype=np.float64), np.array(g, dtype=np.float64)

    # 4. Optimisation
    print("Starting L-BFGS optimization...")
    res = minimize(
        fun=scipy_wrapper,
        x0=x0,
        jac=True, # On indique à Scipy qu'on fournit le gradient
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'disp': True}
    )

    # 5. On reconstruit le résultat
    final_X = res.x.reshape(-1, dim)
    return initial_state._replace(X=final_X), res

