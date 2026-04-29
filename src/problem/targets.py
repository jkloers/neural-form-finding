import numpy as np

# Central definition of the target shape
DEFAULT_TARGET = {
    'type': 'circle',
    'center': np.array([0.0, 0.0]),
    'radius': 1.0
}

def get_target_points(target_params=None, n_points=200):
    """Generates a point cloud discretizing the target shape (Circle or Heart)."""
    if target_params is None:
        target_params = DEFAULT_TARGET
        
    t = np.linspace(0, 2*np.pi, n_points)
    cx, cy = target_params['center']
    scale = target_params['radius']
    shape_type = target_params['type']

    if shape_type == 'circle':
        return np.column_stack([cx + scale*np.cos(t), cy + scale*np.sin(t)])
    
    elif shape_type == 'heart':
        # Parametric equation of the heart
        x = 16 * np.sin(t)**3
        y = 13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t)
        
        # Normalization
        points = np.column_stack([x, y])
        points = points / 16.0 * scale
        points[:, 0] += cx
        points[:, 1] += cy
        return points

    elif shape_type == 'convex_heart':
        x_circle = 16 * np.sin(t)
        y_circle = 16 * np.cos(t)
        
        x_heart = 16 * np.sin(t)**3
        y_heart = 13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t)
        
        alpha = 0.6
        x = alpha * x_heart + (1 - alpha) * x_circle
        y = alpha * y_heart + (1 - alpha) * y_circle
        
        points = np.column_stack([x, y])
        points = points / 16.0 * scale
        points[:, 0] += cx
        points[:, 1] += cy
        return points

    return np.array([])
