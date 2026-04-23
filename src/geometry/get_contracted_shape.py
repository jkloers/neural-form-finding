import numpy as np
from collections import deque
from topology.core import Tessellation

def atan2_angle(v1, v2):
    """Calcule l'angle orienté entre deux vecteurs."""
    ang1 = np.arctan2(v1[1], v1[0])
    ang2 = np.arctan2(v2[1], v2[0])
    return (ang2 - ang1 + np.pi) % (2 * np.pi) - np.pi

def rotate_2d(points, theta, pivot):
    """Rotation de points autour d'un pivot."""
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return (R @ (points - pivot).T).T + pivot

def get_contracted_shape(tessellation: Tessellation, angular_tol: float = 0.1, linear_tol: float = 0.1):
    """
    Assemble le maillage dans sa configuration 'repliée'.
    S'inspire de la traversée de graphe robuste via les charnières et sommets.
    """
    X_ref = np.asarray(tessellation.vertices)
    X_new = np.zeros_like(X_ref)
    
    # Construction des maps topologiques (on utilise tes outils existants)
    primary_to_hinges = tessellation.build_primary_to_hinges()
    
    # Map face -> charnières pour le BFS
    face_to_hinges = {}
    for h in tessellation.hinges:
        face_to_hinges.setdefault(h.face1, []).append(h.id)
        face_to_hinges.setdefault(h.face2, []).append(h.id)

    print(face_to_hinges)

    visited_faces = set()
    queue = deque([0])
    visited_faces.add(0)
    
    # Position initiale de la face de référence
    f0_indices = tessellation.faces[0].vertex_indices
    X_new[f0_indices] = X_ref[f0_indices]
    
    while queue:
        f_idx = queue.popleft()
        
        # On regarde toutes les charnières de la face courante
        for h in face_to_hinges.get(f_idx, []):
            is_face1 = (h.face1 == f_idx)
            f_next_idx = h.face2 if is_face1 else h.face1
            
            if f_next_idx in visited_faces:
                continue
                
            # Identification des points d'ancrage (pins)
            # v_boundary : sur la face déjà placée
            # v_target : sur la face à placer
            v_boundary = h.vertex1 if is_face1 else h.vertex2
            v_target = h.vertex2 if is_face1 else h.vertex1
            
            # Points adjacents pour fixer l'orientation relative
            v_adj_boundary = h.vertex_adjacent1 if is_face1 else h.vertex_adjacent2
            v_adj_target = h.vertex_adjacent2 if is_face1 else h.vertex_adjacent1
            
            # --- 1. Translation ---
            # On aligne le pin de la nouvelle face sur la position déjà calculée
            f_next_indices = tessellation.faces[f_next_idx].vertex_indices
            points = X_ref[f_next_indices].copy()
            
            pivot = X_new[v_boundary]
            translation = pivot - X_ref[v_target]
            points += translation
            
            # --- 2. Rotation ---
            # Calcul de l'angle pour 'fermer' la charnière.
            # Dans le design original, vec_ref et vec_target ont une certaine orientation.
            # En configuration contractée, on force un pliage à 180° (coalescence).
            vec_ref = X_new[v_adj_boundary] - pivot
            vec_target = (X_ref[v_adj_target] + translation) - pivot
            
            # Angle pour aligner les deux bords (config repliée typique RDQK)
            angle = atan2_angle(vec_target, vec_ref)
            points = rotate_2d(points, angle + np.pi, pivot)
            
            # Mise à jour
            X_new[f_next_indices] = points
            visited_faces.add(f_next_idx)
            queue.append(f_next_idx)
            
    return X_new
