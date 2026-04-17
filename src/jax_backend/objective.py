import jax
import jax.numpy as jnp


def squared_length_error(p1, p2, target_length):
    """Computes (a - b)^2 for a single edge."""
    current_length = jnp.linalg.norm(p1 - p2)
    return (current_length - target_length)**2

def squared_angle_error(p_prev, p_center, p_next, target_angle):
    """Computes (alpha - beta)^2 for a single angle."""
    v1 = p_prev - p_center
    v2 = p_next - p_center
    
    v1_u = v1 / (jnp.linalg.norm(v1) + 1e-8)
    v2_u = v2 / (jnp.linalg.norm(v2) + 1e-8)
    
    cos_theta = jnp.clip(jnp.dot(v1_u, v2_u), -1.0, 1.0) #Clip to avoid numerical issues
    current_angle = jnp.arccos(cos_theta)

    return (current_angle - target_angle)**2

# Vectorization
vmap_length_error = jax.vmap(squared_length_error, in_axes=(0, 0, 0))
vmap_angle_error = jax.vmap(squared_angle_error, in_axes=(0, 0, 0, 0))

#--- Main objective function ---

@jax.jit
def compute_objective(X, edge_indices, target_lengths, angle_indices, target_angles):
    """
    X : jnp.ndarray de shape (N, 3)
    edge_indices : jnp.ndarray de shape (N_edges, 2)
    target_lengths : jnp.ndarray de shape (N_edges,)
    angle_indices : jnp.ndarray de shape (N_angles, 3) -> [idx_prev, idx_center, idx_next]
    target_angles : jnp.ndarray de shape (N_angles,)
    """
    
    P1_edges = X[edge_indices[:, 0]]
    P2_edges = X[edge_indices[:, 1]]
    
    length_errors = vmap_length_error(P1_edges, P2_edges, target_lengths)
    
    P_prev = X[angle_indices[:, 0]]
    P_center = X[angle_indices[:, 1]]
    P_next = X[angle_indices[:, 2]]
    
    angle_errors = vmap_angle_error(P_prev, P_center, P_next, target_angles)
    
    total_energy = jnp.sum(length_errors) + jnp.sum(angle_errors)
    
    M = 1
    total_energy = total_energy / M 
    
    return total_energy