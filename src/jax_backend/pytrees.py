# Dans jax_backend/pytrees.py
import jax
import jax.numpy as jnp
from typing import NamedTuple

class TessellationState(NamedTuple):
    X: jnp.ndarray
    
    # (Topologie)
    F_idx: jnp.ndarray
    E_adjacent: jnp.ndarray
    A_rest: jnp.ndarray
    H_stiffness: jnp.ndarray
    V_connect: jnp.ndarray
    Anch_indices: jnp.ndarray
    #Anch_targets: jnp.ndarray  #Placeholder for anchor target positions, not defined in the current implementation

jax.tree_util.register_pytree_node(
    TessellationState,
    lambda state: ((state.X,), (state.F_idx, state.E_adjacent, state.A_rest, state.H_stiffness, state.V_connect, state.Anch_indices)),
    lambda aux, dynamic: TessellationState(dynamic[0], *aux)
)

def create_jax_state(tess_dict):
    """Converts a tessellation dictionary to a JAX-compatible state representation."""
    return TessellationState(
        X=jnp.array(tess_dict['vertices']),
        F_idx=jnp.array(tess_dict['faces']),
        E_adjacent=jnp.array(tess_dict['hinge_adjacent_vertices']),
        A_rest=jnp.array(tess_dict['angles_rest']),
        H_stiffness=jnp.array(tess_dict['hinge_stiffness']),
        V_connect=jnp.array(tess_dict['hinge_vertex_connections']),
        Anch_indices=jnp.array(tess_dict['anchor_indices'])
    )
