import jax
import jax.numpy as jnp
from .pytrees import TessellationState

def opposite_edges_length_constraint(state: TessellationState):
    X, E_opp = state.X, state.E_opp
    if E_opp.shape[0] == 0:
        return 0.0
    
    # Vectorisation directe sans vmap
    p1a, p1b = X[E_opp[:, 0, 0]], X[E_opp[:, 0, 1]]
    p2a, p2b = X[E_opp[:, 1, 0]], X[E_opp[:, 1, 1]]
    
    l1_sq = jnp.sum((p1a - p1b)**2, axis=-1)
    l2_sq = jnp.sum((p2a - p2b)**2, axis=-1)
    
    return jnp.sum((l1_sq - l2_sq)**2)

def opposite_edges_collinearity_constraint(state: TessellationState):
    X, E_opp = state.X, state.E_opp
    if E_opp.shape[0] == 0:
        return 0.0
    
    p1a, p1b = X[E_opp[:, 0, 0]], X[E_opp[:, 0, 1]]
    p2a, p2b = X[E_opp[:, 1, 0]], X[E_opp[:, 1, 1]]
    
    v1 = p1b - p1a
    v2 = p2b - p2a
    
    # Note : strict au 2D. En 3D, il faudrait utiliser jnp.linalg.norm(jnp.cross(v1, v2, axis=-1))**2
    cross_prod = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return jnp.sum(cross_prod**2)

def hinge_connectivity_constraint(state: TessellationState):
    X, V_pairs = state.X, state.V_connect
    p1, p2 = X[V_pairs[:, 0]], X[V_pairs[:, 1]]
    return jnp.sum((p1 - p2)**2)

def hinge_non_intersection_constraint(state: TessellationState, margin: float = 1e-3):
    """
    Pénalise les charnières dont l'angle orienté entraîne une inversion locale.
    Utilise le déterminant normalisé (sinus) pour une invariance d'échelle.
    """
    X, E_adj = state.X, state.E_adjacent
    
    # Extraction des points de la charnière (p2 est le sommet central)
    p1 = X[E_adj[:, 0]]
    p2 = X[E_adj[:, 1]]
    p3 = X[E_adj[:, 2]]
    
    v1 = p1 - p2
    v2 = p3 - p2
    
    # Déterminant 2D (Produit vectoriel scalaire)
    det = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    
    # Calcul des normes avec epsilon pour éviter la division par zéro
    norm_v1 = jnp.linalg.norm(v1, axis=-1)
    norm_v2 = jnp.linalg.norm(v2, axis=-1)
    
    # Déterminant normalisé (équivaut au sinus de l'angle orienté)
    normalized_det = det / (norm_v1 * norm_v2 + 1e-8)
    
    # On pénalise si le déterminant normalisé tombe sous la marge de sécurité
    # Si margin=0, on empêche strictement l'inversion.
    # Si margin>0, on impose une convexité/ouverture minimale.
    violations = jax.nn.relu(margin - normalized_det)
    
    # Mise au carré pour une pénalité lisse et différentiable
    return jnp.sum(violations**2)

def chamfer_distance_sq(set_A, set_B):
    """
    Calcule la distance de Chamfer au carré entre deux ensembles de points.
    set_A : shape (N, D)
    set_B : shape (M, D)
    """
    # Broadcasting pour obtenir la matrice des différences paires (N, M, D)
    diffs = set_A[:, None, :] - set_B[None, :, :]
    
    # Distances au carré paires (N, M)
    sq_dists = jnp.sum(diffs**2, axis=-1)
    
    # Distance minimale de A vers B et de B vers A
    min_dist_A_to_B = jnp.min(sq_dists, axis=1) # (N,)
    min_dist_B_to_A = jnp.min(sq_dists, axis=0) # (M,)
    
    # On utilise généralement la moyenne pour être invariant au nombre de points
    return jnp.mean(min_dist_A_to_B) + jnp.mean(min_dist_B_to_A)

def anchor_chamfer_constraint(state: TessellationState, target_points: jnp.ndarray):
    """
    Contrainte rapprochant les ancres de la géométrie cible.
    On suppose que `state.anchor_indices` contient les indices des points d'ancrage.
    """
    anchors = state.X[state.anchor_indices]
    return chamfer_distance_sq(anchors, target_points)

# def laplacian_regularizer(X, F_idx):
#     """
#     Pénalise la déviation de chaque sommet par rapport au barycentre de ses voisins.
#     Maintient la qualité du maillage.
#     """
#     # Pour un maillage de quads, on peut simplifier la topologie des voisins
#     # Voisins par face (pour chaque face v0, v1, v2, v3)
#     v0, v1, v2, v3 = F_idx[:, 0], F_idx[:, 1], F_idx[:, 2], F_idx[:, 3]
    
#     # Erreur locale : (p_i - p_j)^2 pour toutes les arêtes
#     e = jnp.sum((X[v0] - X[v1])**2) + jnp.sum((X[v1] - X[v2])**2) + \
#         jnp.sum((X[v2] - X[v3])**2) + jnp.sum((X[v3] - X[v0])**2)
    
#     return e

#####################################################################################
#                               Compute Objective                                   #
#####################################################################################

def compute_objective(X, state: TessellationState, target_params):
    """
    X : les positions optimisées (ce que l'optimiseur fait varier)
    state : le reste de la topologie (face_idx, hinges, etc.)
    """
    # On remplace X dans le state pour que les fonctions de contraintes l'utilisent
    current_state = state._replace(X=X)
    
    # --- 1. Énergies de Contraintes (Topologie) ---
    # La connectivité des charnières doit être quasi-parfaite (poids élevé)
    e_hinge = hinge_connectivity_constraint(current_state)
    
    # La cohérence des longueurs dans les vides (poids moyen)
    e_void = opposite_edges_length_constraint(current_state)

    e_hinge_non_intersection = hinge_non_intersection_constraint(current_state) 
    
    # --- 2. Énergie de Cible (Fitting) ---
    # Exemple : on veut que tous les points soient sur un cercle de rayon R
    target_radius = target_params.get('radius', 1.0)
    current_radii = jnp.linalg.norm(X, axis=1)
    e_target = jnp.mean((current_radii - target_radius)**2)
    
    # --- 3. Régularisation (Optionnel) ---
    # Empêcher les points d'exploser ou de trop s'écraser
    e_reg = jnp.sum(X**2) * 1e-4

    # --- Somme totale pondérée ---
    # On donne beaucoup d'importance aux charnières pour ne pas que le maillage se déchire
    total_loss = (
        1000.0 * e_hinge +  # Hard constraint
        1000.0 * e_hinge_non_intersection +  # Hard constraint
        1000.0   * e_void  +  # Soft topological constraint
        1.0    * e_target + # Design Goal
        e_reg               # Stability
    )
    
    return total_loss



