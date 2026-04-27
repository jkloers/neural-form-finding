import jax
import jax.numpy as jnp
from .pytrees import TessellationState
from geometry.target_shape import DEFAULT_TARGET, get_target_points

def faces_rigidity_constraint(state: TessellationState):
    """Penalizes deformation of quads relative to their initial rest lengths (Isometry)."""
    X, F_idx, F_rest = state.X, state.F_idx, state.F_rest_lengths_sq
    
    # Current points of the faces
    p0, p1, p2, p3 = X[F_idx[:, 0]], X[F_idx[:, 1]], X[F_idx[:, 2]], X[F_idx[:, 3]]
    
    # Calculate current lengths (4 edges + 2 diagonals)
    edges = [p0-p1, p1-p2, p2-p3, p3-p0, p0-p2, p1-p3]
    lengths_sq = jnp.stack([jnp.sum(e**2, axis=-1) for e in edges], axis=1)
    
    # Energy: Squared difference compared to fixed rest lengths
    return jnp.sum((lengths_sq - F_rest)**2)

def opposite_edges_length_constraint(state: TessellationState):
    """Maintains symmetry of the opposite edges in the voids."""
    X, E_opp = state.X, state.E_opp
    if E_opp.shape[0] == 0:
        return 0.0
    
    p1a, p1b = X[E_opp[:, 0, 0]], X[E_opp[:, 0, 1]]
    p2a, p2b = X[E_opp[:, 1, 0]], X[E_opp[:, 1, 1]]
    
    l1_sq = jnp.sum((p1a - p1b)**2, axis=-1)
    l2_sq = jnp.sum((p2a - p2b)**2, axis=-1)
    
    return jnp.sum((l1_sq - l2_sq)**2)

def opposite_edges_collinearity_constraint(state: TessellationState):
    """Forces opposite edges in the voids to be collinear."""
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
    """Linear elastic energy to keep pivots connected."""
    X, V_pairs = state.X, state.V_connect
    k_linear = state.H_linear_stiffness
    
    p1, p2 = X[V_pairs[:, 0]], X[V_pairs[:, 1]]
    sq_dist = jnp.sum((p1 - p2)**2, axis=-1)
    return jnp.sum(k_linear * sq_dist)

def hinge_non_intersection_constraint(state: TessellationState, margin: float = 1e-3):
    """Prevents local inversion of faces at the hinges (negative angles)."""
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
    """Penalizes deviation from a target angle (e.g. 0 for folding)."""
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
    Forces the hinge arms located on the boundary to have equal lengths.
    """
    X, E_adj = state.X, state.E_adjacent
    boundary_indices = state.Boundary_indices

    # Check if either adjacent vertex (b1 or b2) is on the boundary (0/1 mask)
    is_b1_boundary = jnp.isin(E_adj[:, 0, 1], boundary_indices)
    is_b2_boundary = jnp.isin(E_adj[:, 1, 1], boundary_indices)
    
    # Float mask (1.0 if boundary, 0.0 otherwise)
    mask = (is_b1_boundary | is_b2_boundary).astype(jnp.float32)
    
    # Calculate for ALL hinges
    p1, b1 = X[E_adj[:, 0, 0]], X[E_adj[:, 0, 1]]
    p2, b2 = X[E_adj[:, 1, 0]], X[E_adj[:, 1, 1]]
    
    l1_sq = jnp.sum((p1 - b1)**2, axis=-1)
    l2_sq = jnp.sum((p2 - b2)**2, axis=-1)
    
    # Apply mask before final sum
    return jnp.sum(mask * (l1_sq - l2_sq)**2)

def border_edges_length_constraint(state: TessellationState):
    """
    Constrains the length of border edges to remain equal to their rest length.
    This prevents the borders from stretching abnormally.
    """
    X = state.X
    Border_edges = state.Border_edges
    rest_lengths_sq = state.Border_edges_rest_lengths_sq
    
    total_penalty = 0.0
    for group, edges in Border_edges.items():
        if edges.shape[0] == 0:
            continue
            
        p0 = X[edges[:, 0]]
        p1 = X[edges[:, 1]]
        
        l_sq = jnp.sum((p1 - p0)**2, axis=-1)
        target_l_sq = rest_lengths_sq[group]
        
        total_penalty += jnp.sum((l_sq - target_l_sq)**2)
        
    return total_penalty

def boundary_shape_constraint(state: TessellationState, boundary_vertices: jnp.ndarray, boundary_set: jnp.ndarray):
    """Penalizes the distance between boundary vertices and the target point cloud."""
    X = state.X
    pts_boundary = X[boundary_vertices]
    
    # Differentiable Chamfer Distance
    # Distance from each boundary point to the closest point in the target
    dist_matrix = jnp.sum((pts_boundary[:, None, :] - boundary_set[None, :, :])**2, axis=-1)
    min_dist_to_target = jnp.min(dist_matrix, axis=1)
    min_dist_to_boundary = jnp.min(dist_matrix, axis=0)
    
    return jnp.mean(min_dist_to_target) + jnp.mean(min_dist_to_boundary)

#####################################################################################
#                               Objective Functions                                 #
#####################################################################################

def compute_objective_deployed(X, state: TessellationState, target_params):
    """Objective for the deployed shape (Fitting + Rigidity)."""

    current_state = state._replace(X=X)
    
    e_faces = faces_rigidity_constraint(current_state)
    e_hinge = hinge_connectivity_constraint(current_state)
    e_void_l = opposite_edges_length_constraint(current_state)
    e_void_c = opposite_edges_collinearity_constraint(current_state)
    e_non_inv = hinge_non_intersection_constraint(current_state)
    e_boundary_hinge = boundary_points_hinge_arm_symmetry_constraint(current_state)
    e_border_length = border_edges_length_constraint(current_state)
    
    # Target fitting via Chamfer Distance on the boundary
    boundary_indices = state.Boundary_indices
    target_cloud = jnp.array(get_target_points(target_params, n_points=100))
    e_target = boundary_shape_constraint(current_state, boundary_indices, target_cloud)
    
    e_reg = jnp.sum(X**2) * 1e-4

    return (10.0 * e_faces +        # RIGIDITY
            700.0 * e_hinge +      # CONNECTIVITY
            1000.0 * e_non_inv +    # ORIENTATION (no intersection)
            10000.0 * e_void_l +     # VOID EDGE LENGTH
            10000.0 * e_void_c +     # VOID COLLINEARITY
            1.0    * e_target +     # TARGET FITTING
            1.0  * e_boundary_hinge + # HINGE ARM SYMMETRY
            10.0 * e_border_length +  # BORDER LENGTHS
            e_reg)

def compute_objective_contracted(X, state: TessellationState, target_params):
    """Objective for the contracted (folded) shape."""
    current_state = state._replace(X=X)
    
    e_faces = faces_rigidity_constraint(current_state)
    e_hinge = hinge_connectivity_constraint(current_state)
    e_target_angle = hinge_target_angle_constraint(current_state, target_sin=0.0)
    e_non_inv = hinge_non_intersection_constraint(current_state, margin=1e-3)
    
    e_reg = jnp.sum(X**2) * 1e-4

    return (10000.0 * e_faces + 
            10000.0 * e_hinge + 
            1  * e_target_angle + 
            10.0 * e_non_inv + 
            e_reg)
