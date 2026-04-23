
import jax
import jax.numpy as jnp
from typing import NamedTuple

class TessellationState(NamedTuple):
    X: jnp.ndarray
    
    # Topology
    F_idx: jnp.ndarray
    F_rest_lengths_sq: jnp.ndarray # (N_faces, 6): 4 edges + 2 diagonals
    E_adjacent: jnp.ndarray
    E_opp: jnp.ndarray
    A_rest: jnp.ndarray
    H_angular_stiffness: jnp.ndarray
    H_linear_stiffness: jnp.ndarray
    V_connect: jnp.ndarray
    Boundary_indices: jnp.ndarray
    Border_edges: dict
    Border_edges_rest_lengths_sq: dict

jax.tree_util.register_pytree_node(
    TessellationState,
    lambda state: ((state.X,), (state.F_idx, state.F_rest_lengths_sq, state.E_adjacent, state.E_opp, state.A_rest, state.H_angular_stiffness, state.H_linear_stiffness, state.V_connect, state.Boundary_indices, state.Border_edges, state.Border_edges_rest_lengths_sq)),
    lambda aux, dynamic: TessellationState(dynamic[0], *aux)
)

def compute_face_lengths_sq(X, F_idx):
    """Calculates the squared lengths of the 4 edges and 2 diagonals for each face."""
    p0, p1, p2, p3 = X[F_idx[:, 0]], X[F_idx[:, 1]], X[F_idx[:, 2]], X[F_idx[:, 3]]
    edges = [p0-p1, p1-p2, p2-p3, p3-p0, p0-p2, p1-p3]
    return jnp.stack([jnp.sum(e**2, axis=-1) for e in edges], axis=1)

def create_jax_state(tess_dict):
    """Converts a tessellation dictionary to a JAX-compatible state representation."""
    X_init = jnp.array(tess_dict['vertices'])
    F_idx = jnp.array(tess_dict['faces'])
    
    # Pre-calculate initial face lengths for rigidity constraint
    rest_lengths_sq = compute_face_lengths_sq(X_init, F_idx)

    return TessellationState(
        X=X_init,
        F_idx=F_idx,
        F_rest_lengths_sq=rest_lengths_sq,
        E_adjacent=jnp.array(tess_dict['hinge_adjacent_edges']),
        E_opp=jnp.array(tess_dict['void_opposite_edges']),
        A_rest=jnp.array(tess_dict['angles_rest']),
        H_angular_stiffness=jnp.array(tess_dict['hinge_angular_stiffness']),
        H_linear_stiffness=jnp.array(tess_dict['hinge_linear_stiffness']),
        V_connect=jnp.array(tess_dict['hinge_vertex_connections']),
        Boundary_indices=jnp.array(tess_dict['boundary_indices']),
        Border_edges={k: jnp.array(v) for k, v in tess_dict.get('border_edges', {}).items()},
        Border_edges_rest_lengths_sq={
            k: jnp.array(v)
            for k, v in tess_dict.get('border_edges_rest_lengths_sq', {}).items()
        }
    )
