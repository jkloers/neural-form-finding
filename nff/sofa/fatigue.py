"""nff/sofa/fatigue.py — low-cycle fatigue life from plastic strain.

Client-side (kgnn_mac); pure NumPy, never imports ``Sofa``. Shared by the hinge
optimizer and the closing-animation viz so the cycles-to-failure formula lives in
exactly one place.
"""
from __future__ import annotations

import numpy as np


def plastic_strain(max_strain: float, yield_strain: float) -> float:
    """Per-fold plastic strain — only strain above yield fatigues the hinge."""
    return max(0.0, float(max_strain) - float(yield_strain))


def cycles_to_failure(eps_plastic: float, ductility_coeff: float,
                      ductility_exp: float) -> float:
    """Coffin-Manson cycles-to-failure ``N_f = ½·(ε_p / ε_f')^(1/c)``.

    Args:
        eps_plastic: plastic strain amplitude per fold.
        ductility_coeff: fatigue ductility coefficient ε_f'.
        ductility_exp: fatigue ductility exponent c (< 0).

    Returns:
        Cycles to failure; ``inf`` when the design stays elastic (ε_p ≈ 0).
    """
    if eps_plastic <= 1e-9:
        return float('inf')   # stays elastic → effectively unlimited cycles
    return 0.5 * (eps_plastic / ductility_coeff) ** (1.0 / ductility_exp)
