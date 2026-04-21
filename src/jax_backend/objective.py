import jax
import jax.numpy as jnp
from .pytrees import TessellationState
from geometry.target_shape import DEFAULT_TARGET, get_target_points

def faces_rigidity_constraint(state: TessellationState):
    """Pénalise la déformation des quads par rapport à leurs longueurs initiales (Isométrie)."""
    X, F_idx, F_rest = state.X, state.F_idx, state.F_rest_lengths_sq
    
    # Points actuels des faces
    p0, p1, p2, p3 = X[F_idx[:, 0]], X[F_idx[:, 1]], X[F_idx[:, 2]], X[F_idx[:, 3]]
    
    # Calcul des longueurs actuelles (4 arêtes + 2 diagonales)
    edges = [p0-p1, p1-p2, p2-p3, p3-p0, p0-p2, p1-p3]
    lengths_sq = jnp.stack([jnp.sum(e**2, axis=-1) for e in edges], axis=1)
    
    # Énergie : Différence quadratique par rapport aux longueurs de repos fixes
    return jnp.sum((lengths_sq - F_rest)**2)

def opposite_edges_length_constraint(state: TessellationState):
    """Maintient la symétrie des bords des vides."""
    X, E_opp = state.X, state.E_opp
    if E_opp.shape[0] == 0:
        return 0.0
    
    p1a, p1b = X[E_opp[:, 0, 0]], X[E_opp[:, 0, 1]]
    p2a, p2b = X[E_opp[:, 1, 0]], X[E_opp[:, 1, 1]]
    
    l1_sq = jnp.sum((p1a - p1b)**2, axis=-1)
    l2_sq = jnp.sum((p2a - p2b)**2, axis=-1)
    
    return jnp.sum((l1_sq - l2_sq)**2)

def opposite_edges_collinearity_constraint(state: TessellationState):
    """Force la colinéarité des bords opposés dans les vides."""
    X, E_opp = state.X, state.E_opp
    if E_opp.shape[0] == 0:
        return 0.0
    
    p1a, p1b = X[E_opp[:, 0, 0]], X[E_opp[:, 0, 1]]
    p2a, p2b = X[E_opp[:, 1, 0]], X[E_opp[:, 1, 1]]
    
    v1 = p1b - p1a
    v2 = p2b - p2a
    
    cross_prod = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return jnp.sum(cross_prod**2)

def hinge_connectivity_constraint(state: TessellationState):
    """Énergie élastique linéaire (maintient les pivots connectés)."""
    X, V_pairs = state.X, state.V_connect
    k_linear = state.H_linear_stiffness
    
    p1, p2 = X[V_pairs[:, 0]], X[V_pairs[:, 1]]
    sq_dist = jnp.sum((p1 - p2)**2, axis=-1)
    return jnp.sum(k_linear * sq_dist)

def hinge_non_intersection_constraint(state: TessellationState, margin: float = 1e-3):
    """Empêche l'inversion locale des faces au niveau des charnières."""
    X, E_adj = state.X, state.E_adjacent
    
    pivot = X[E_adj[:, 0, 0]]
    p_adj1 = X[E_adj[:, 0, 1]]
    p_adj2 = X[E_adj[:, 1, 1]]
    
    v1 = p_adj1 - pivot
    v2 = p_adj2 - pivot
    
    det = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    norm_v1 = jnp.linalg.norm(v1, axis=-1)
    norm_v2 = jnp.linalg.norm(v2, axis=-1)
    
    normalized_det = det / (norm_v1 * norm_v2 + 1e-8)
    violations = jax.nn.relu(margin - normalized_det)
    
    return jnp.sum(violations**2)

def hinge_target_angle_constraint(state: TessellationState, target_sin: float = 0.0):
    """Pénalise l'écart par rapport à un angle cible (ex: 0 pour le repliement)."""
    X, E_adj = state.X, state.E_adjacent
    k_angular = state.H_angular_stiffness
    
    pivot = X[E_adj[:, 0, 0]]
    p_adj1 = X[E_adj[:, 0, 1]]
    p_adj2 = X[E_adj[:, 1, 1]]
    
    v1 = p_adj1 - pivot
    v2 = p_adj2 - pivot
    
    det = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    norm_v1 = jnp.linalg.norm(v1, axis=-1)
    norm_v2 = jnp.linalg.norm(v2, axis=-1)
    
    sin_theta = det / (norm_v1 * norm_v2 + 1e-8)
    return jnp.sum(k_angular * (sin_theta - target_sin)**2)

def boundary_points_hinge_arm_symmetry_constraint(state: TessellationState):
    """
    Force l'égalité des longueurs des bras pour les charnières situées au bord.
    """
    X, E_adj = state.X, state.E_adjacent
    boundary_indices = state.Boundary_indices

    # On vérifie si l'UN des sommets adjacents (b1 ou b2) est sur la bordure (masque 0/1)
    is_b1_boundary = jnp.isin(E_adj[:, 0, 1], boundary_indices)
    is_b2_boundary = jnp.isin(E_adj[:, 1, 1], boundary_indices)
    
    # Masque float (1.0 si bordure, 0.0 sinon)
    mask = (is_b1_boundary | is_b2_boundary).astype(jnp.float32)
    
    # Calcul pour TOUTES les charnières
    p1, b1 = X[E_adj[:, 0, 0]], X[E_adj[:, 0, 1]]
    p2, b2 = X[E_adj[:, 1, 0]], X[E_adj[:, 1, 1]]
    
    l1_sq = jnp.sum((p1 - b1)**2, axis=-1)
    l2_sq = jnp.sum((p2 - b2)**2, axis=-1)
    
    # On applique le masque avant la somme finale
    return jnp.sum(mask * (l1_sq - l2_sq)**2)

def boundary_shape_constraint(state: TessellationState, boundary_vertices: jnp.ndarray, boundary_set: jnp.ndarray):
    """Pénalise la distance entre les sommets de bordure et le nuage de points cible."""
    X = state.X
    pts_boundary = X[boundary_vertices]
    
    # Distance de Chamfer codée en JAX pour être différentiable
    # Distance de chaque point de bordure vers le point le plus proche de la cible
    dist_matrix = jnp.sum((pts_boundary[:, None, :] - boundary_set[None, :, :])**2, axis=-1)
    min_dist_to_target = jnp.min(dist_matrix, axis=1)
    min_dist_to_boundary = jnp.min(dist_matrix, axis=0)
    
    return jnp.mean(min_dist_to_target) + jnp.mean(min_dist_to_boundary)

#####################################################################################
#                               Objective Functions                                 #
#####################################################################################

def compute_objective_deployed(X, state: TessellationState, target_params):
    """Objectif pour la forme déployée (Fitting + Rigidité)."""
    current_state = state._replace(X=X)
    
    e_faces = faces_rigidity_constraint(current_state)
    e_hinge = hinge_connectivity_constraint(current_state)
    e_void_l = opposite_edges_length_constraint(current_state)
    e_void_c = opposite_edges_collinearity_constraint(current_state)
    e_non_inv = hinge_non_intersection_constraint(current_state)
    e_boundary_hinge = boundary_points_hinge_arm_symmetry_constraint(current_state)
    
    # Filling Cible via Chamfer Distance sur la bordure
    boundary_indices = state.Boundary_indices
    target_cloud = jnp.array(get_target_points(n_points=100))
    e_target = boundary_shape_constraint(current_state, boundary_indices, target_cloud)
    
    e_reg = jnp.sum(X**2) * 1e-4

    return (1.0 * e_faces +    # RIGIDITÉ
            1000.0 * e_hinge +    # CONNECTIVITÉ
            1000.0 * e_non_inv +  # ORIENTATION
            10000.0 * e_void_l +   
            10000.0 * e_void_c + 
            1.0    * e_target +   # Fitting Cible
            100.0    * e_boundary_hinge +   # Boundary Hinge Symmetry
            e_reg)

def compute_objective_contracted(X, state: TessellationState, target_params):
    """Objectif pour la forme contractée."""
    current_state = state._replace(X=X)
    
    e_faces = faces_rigidity_constraint(current_state)
    e_hinge = hinge_connectivity_constraint(current_state)
    e_target_angle = hinge_target_angle_constraint(current_state, target_sin=0.0)
    
    e_reg = jnp.sum(X**2) * 1e-4

    return (10000.0 * e_faces + 
            1000.0 * e_hinge + 
            1.0  * e_target_angle + 
            e_reg)
