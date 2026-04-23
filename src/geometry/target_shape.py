import numpy as np

# Définition centrale de la forme cible
DEFAULT_TARGET = {
    'type': 'circle',  # Changement vers 'heart'
    'center': np.array([0.0, 0.0]),
    'radius': 1.0     # Facteur d'échelle pour que le coeur tienne dans la vue
}

def get_target_points(n_points=200):
    """Génère un nuage de points discrétisant la forme cible (Cercle ou Coeur)."""
    t = np.linspace(0, 2*np.pi, n_points)
    cx, cy = DEFAULT_TARGET['center']
    scale = DEFAULT_TARGET['radius']

    if DEFAULT_TARGET['type'] == 'circle':
        return np.column_stack([cx + scale*np.cos(t), cy + scale*np.sin(t)])
    
    elif DEFAULT_TARGET['type'] == 'heart':
        # Équation paramétrique du coeur
        x = 16 * np.sin(t)**3
        y = 13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t)
        
        # Normalisation pour que le coeur soit centré et à l'échelle
        points = np.column_stack([x, y])
        points = points / 16.0 * scale # Mise à l'échelle
        points[:, 0] += cx
        points[:, 1] += cy
        return points

    return np.array([])
