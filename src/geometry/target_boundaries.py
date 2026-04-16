import numpy as np

# ── Forme cible ───────────────────────────────────────────────────────────────

def target_circle(p, center, radius):
    """Projette p sur un cercle cible."""
    v = p - center
    n = np.linalg.norm(v)
    if n < 1e-10:
        return center + np.array([radius, 0.0])
    return center + (v / n) * radius


