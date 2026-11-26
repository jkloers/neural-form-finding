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

    elif shape_type == 'ume':
        # Plum-blossom outline r(t) = R (1 + eps cos(n t)): the circle carrying an n-fold ripple,
        # so `radius` keeps its meaning as the mean radius and `eps` is the only new quantity.
        #
        # ★ eps CROSSES A STATED GEOMETRIC BOUNDARY, which is why the shape is parameterised this
        # way rather than traced. A polar curve is convex where r^2 + 2 r'^2 - r r'' > 0; at the
        # worst point (cos n t = -1) that reduces to 1 - (n^2 + 2) eps + (n^2 + 1) eps^2 > 0.
        # That quadratic always has eps = 1 as a root, so the other is 1 / (n^2 + 1) and the
        # outline is STRICTLY CONVEX iff eps < 1 / (n^2 + 1): 1/26 = 0.038462 at n = 5, 1/17 at
        # n = 4, 1/65 at n = 8. A target above the threshold is re-entrant, and a rigid-panel
        # mechanism has to work to reach one.
        n_petals = int(target_params.get('petals', 5))
        eps = float(target_params.get('eps', 0.0))
        r = scale * (1.0 + eps * np.cos(n_petals * t))
        return np.column_stack([cx + r*np.cos(t), cy + r*np.sin(t)])

    elif shape_type == 'rectangle':
        # Rectangle perimeter; half-extents half_w / half_h (fall back to radius if not provided),
        # optionally rotated by `rot_deg` about `center`.
        #
        # ★ The rotation exists because a deployment that SWINGS is a rotated shape, and no
        # axis-aligned box hugs one: tightening an unrotated target around a rotated sheet asks for
        # a shape the mechanism cannot make, and the optimizer answers by deploying less. Giving the
        # target an orientation lets it describe the deployment instead of fighting it.
        hw = target_params.get('half_w', scale)
        hh = target_params.get('half_h', scale)
        n = max(2, n_points // 4)
        top    = np.column_stack([np.linspace(cx - hw, cx + hw, n), np.full(n, cy + hh)])
        right  = np.column_stack([np.full(n, cx + hw), np.linspace(cy + hh, cy - hh, n)])
        bottom = np.column_stack([np.linspace(cx + hw, cx - hw, n), np.full(n, cy - hh)])
        left   = np.column_stack([np.full(n, cx - hw), np.linspace(cy - hh, cy + hh, n)])
        pts = np.vstack([top, right, bottom, left])
        rot = float(target_params.get('rot_deg', 0.0) or 0.0)
        if rot != 0.0:
            a = np.radians(rot)
            R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
            c = np.array([cx, cy])
            pts = (pts - c) @ R.T + c
        return pts

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
