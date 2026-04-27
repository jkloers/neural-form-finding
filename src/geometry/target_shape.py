import numpy as np

# Définition centrale de la forme cible
DEFAULT_TARGET = {
    'type': 'circle',  # Changement vers 'heart'
    'center': np.array([0.0, 0.0]),
    'radius': 1     # Facteur d'échelle pour que le coeur tienne dans la vue
}

def get_target_points(target_params=None, n_points=200):
    """Génère un nuage de points discrétisant la forme cible (Cercle ou Coeur)."""
    if target_params is None:
        target_params = DEFAULT_TARGET
        
    t = np.linspace(0, 2*np.pi, n_points)
    cx, cy = target_params['center']
    scale = target_params['radius']
    shape_type = target_params['type']

    if shape_type == 'circle':
        return np.column_stack([cx + scale*np.cos(t), cy + scale*np.sin(t)])
    
    elif shape_type == 'heart':
        # Équation paramétrique du coeur
        x = 16 * np.sin(t)**3
        y = 13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t)
        
        # Normalisation pour que le coeur soit centré et à l'échelle
        points = np.column_stack([x, y])
        points = points / 16.0 * scale # Mise à l'échelle
        points[:, 0] += cx
        points[:, 1] += cy
        return points

    elif shape_type == 'convex_heart':
        # Un coeur plus arrondi et convexe, obtenu en interpolant un cercle et un coeur
        x_circle = 16 * np.sin(t)
        y_circle = 16 * np.cos(t)
        
        x_heart = 16 * np.sin(t)**3
        y_heart = 13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t)
        
        # Mélange : 60% coeur, 40% cercle (plus l'alpha est bas, plus c'est convexe/rond)
        alpha = 0.6
        x = alpha * x_heart + (1 - alpha) * x_circle
        y = alpha * y_heart + (1 - alpha) * y_circle
        
        points = np.column_stack([x, y])
        points = points / 16.0 * scale
        points[:, 0] += cx
        points[:, 1] += cy
        return points

    return np.array([])
