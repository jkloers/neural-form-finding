"""Analytic target outlines for the closed-sheet kinematic inverse design.

The shield path reads its target from an SVG silhouette (``shield_target``). A circle needs
no artwork, and the optimiser only ever consumes an ``(n, 2)`` polyline, so the two are
interchangeable at that seam.

Convention, matching ``shield_target.load_shield_outline``: centred at the origin, unit
height, and ``endpoint=False`` so the first point is not repeated in the Chamfer set.
"""

import numpy as np
from jaxtyping import Float


def circle_outline_unit(n_points: int = 480) -> Float[np.ndarray, "n 2"]:
    """A circle of unit height, i.e. radius 0.5, centred at the origin.

    Args:
        n_points: samples around the circumference.

    Returns:
        (n_points, 2) counter-clockwise, uniformly spaced in arclength.
    """
    phi = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    return np.stack([0.5 * np.cos(phi), 0.5 * np.sin(phi)], axis=1)
