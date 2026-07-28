"""Stress-strain curves and mechanical-property extraction from tensile runs.

Strain can come from two sources:

* **Crosshead displacement + gauge length** — compliance-corrupted. At the short
  gauge lengths used here (~103 mm) with soft tape tabs, ~3/4 of the crosshead
  travel is machine + tab + toe, NOT specimen, so the modulus reads ~1 GPa vs the
  true ~3 GPa (protocol §13.5). A ``C_machine`` compliance term partly corrects it.
* **Video extensometer** (:mod:`ladder`) — true gauge strain, immune to
  machine/tab compliance. This is the trustworthy source for E.

Force-based quantities (yield, draw-plateau, UTS) are computed from force / area and
are reliable regardless of the strain source.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TensileResult:
    stress_MPa: np.ndarray
    strain: np.ndarray
    E_MPa: float | None
    yield_MPa: float | None
    uts_MPa: float | None
    strain_at_break: float | None
    draw_plateau_MPa: float | None


def crosshead_strain(
    disp_mm: np.ndarray,
    L0_mm: float,
    force_N: np.ndarray | None = None,
    C_machine_mm_per_N: float = 0.0,
) -> np.ndarray:
    """Engineering strain from crosshead travel, with optional compliance correction.

    ``disp_specimen = disp - C_machine * force`` removes the (linear) machine + grip +
    tab compliance measured on a known-modulus reference strip. With ``C_machine=0``
    this returns the raw (compliance-corrupted) crosshead strain.
    """
    disp = np.asarray(disp_mm, float)
    if C_machine_mm_per_N and force_N is not None:
        disp = disp - C_machine_mm_per_N * np.asarray(force_N, float)
    return disp / float(L0_mm)


def chord_modulus(
    stress_MPa: np.ndarray, strain: np.ndarray, lo: float = 0.004, hi: float = 0.012
) -> float | None:
    """Least-squares slope of stress vs strain over a fixed strain window [lo, hi]."""
    strain = np.asarray(strain)
    stress_MPa = np.asarray(stress_MPa)
    mask = (strain >= lo) & (strain <= hi)
    if int(mask.sum()) < 3:
        return None
    slope, _ = np.polyfit(strain[mask], stress_MPa[mask], 1)
    return float(slope)


def initial_tangent_modulus(
    stress_MPa: np.ndarray,
    strain: np.ndarray,
    window: tuple[float, float] = (0.0005, 0.004),
) -> float | None:
    """Young's modulus as the INITIAL tangent slope, before pre-yield softening.

    PET's tangent modulus falls as it approaches yield (glassy elastic -> anelastic), so a
    chord over a wide window understates E. The initial tangent over a small early window is
    the correct definition. Still a lower bound under crosshead strain (compliance in series);
    use it on video-extensometer strain for the true value.
    """
    return chord_modulus(stress_MPa, strain, *window)


def build_plastic_table(
    sigma_y_MPa: float,
    sigma_true_draw_MPa: float,
    eps_true_draw: float,
    E_MPa: float,
    intermediate: list[tuple[float, float]] | None = None,
) -> list[tuple[float, float]]:
    """Monotonic true-stress / true-plastic-strain ``*PLASTIC`` table for cold-draw PET.

    Anchored on the two measured points: yield (uniform, small strain) and the steady draw
    (true stress = lambda * plateau, true strain = ln(lambda)). Plastic strain at the draw =
    total true strain minus the elastic part sigma/E. Any ``intermediate`` (stress, eps_p)
    points are inserted; the table is de-duplicated and sorted, and CalculiX requires it to be
    monotonically increasing in stress.
    """
    eps_p_draw = eps_true_draw - sigma_true_draw_MPa / E_MPa
    table = [(sigma_y_MPa, 0.0)]
    if intermediate:
        table += list(intermediate)
    table.append((sigma_true_draw_MPa, eps_p_draw))
    table = sorted(set(table), key=lambda p: p[1])
    return table


def _yield_offset(stress: np.ndarray, strain: np.ndarray, E: float, offset: float = 0.002):
    """0.2%-offset yield: intersection of stress-strain with the E-line shifted by ``offset``."""
    if E is None or E <= 0:
        return None
    diff = stress - E * (strain - offset)
    sign = np.sign(diff)
    cross = np.where(np.diff(sign) < 0)[0]  # curve drops below the offset line
    if cross.size == 0:
        return None
    return float(stress[cross[0]])


def analyze(
    force_N: np.ndarray,
    area_mm2: float,
    strain: np.ndarray,
    *,
    modulus_window: tuple[float, float] = (0.004, 0.012),
    draw_window_mm: tuple[float, float] | None = None,
    disp_mm: np.ndarray | None = None,
) -> TensileResult:
    """Build a stress-strain curve and extract E, yield, UTS, draw plateau, break strain.

    Args:
        force_N: (n,) force.
        area_mm2: cross-section (use the manual-summary area, protocol §13).
        strain: (n,) strain from crosshead or video.
        modulus_window: strain window for the chord modulus.
        draw_window_mm: (lo, hi) crosshead-displacement window over which to median the
            force for the cold-draw plateau stress; needs ``disp_mm``.
        disp_mm: crosshead displacement, only for locating the draw window.

    Returns:
        A :class:`TensileResult`.
    """
    force_N = np.asarray(force_N, float)
    strain = np.asarray(strain, float)
    stress = force_N / float(area_mm2)

    E = chord_modulus(stress, strain, *modulus_window)
    uts = float(np.max(stress))
    y_offset = _yield_offset(stress, strain, E)
    yield_MPa = y_offset if y_offset is not None else float(np.max(stress))  # peak fallback
    strain_at_break = float(strain[-1])

    draw = None
    if draw_window_mm is not None and disp_mm is not None:
        disp_mm = np.asarray(disp_mm, float)
        lo, hi = draw_window_mm
        m = (disp_mm >= lo) & (disp_mm <= hi)
        if int(m.sum()) > 3:
            draw = float(np.median(stress[m]))

    return TensileResult(
        stress_MPa=stress,
        strain=strain,
        E_MPa=E,
        yield_MPa=yield_MPa,
        uts_MPa=uts,
        strain_at_break=strain_at_break,
        draw_plateau_MPa=draw,
    )


def true_from_engineering(sigma_eng: float, eps_eng: float) -> tuple[float, float]:
    """Engineering -> true stress/strain assuming incompressible uniform deformation.

    Valid only while deformation is uniform (pre-neck). For the cold-draw anchor use
    :func:`draw_true_point` instead, which uses the measured draw ratio.
    """
    return sigma_eng * (1.0 + eps_eng), float(np.log(1.0 + eps_eng))


def draw_true_point(plateau_eng_MPa: float, draw_ratio_lambda: float) -> tuple[float, float]:
    """Large-strain `*PLASTIC` anchor from the cold-draw plateau (protocol §13.6).

    In the propagating neck the drawn area is A0/lambda, so the true stress is
    ``lambda * plateau_eng`` and the true (logarithmic) strain is ``ln(lambda)``.
    """
    sigma_true = draw_ratio_lambda * plateau_eng_MPa
    eps_true = float(np.log(draw_ratio_lambda))
    return sigma_true, eps_true
