"""Load-train compliance: how much of the crosshead travel is NOT the specimen.

A screw-driven frame reports crosshead displacement, not specimen displacement. Everything
in series with the specimen — frame, load cell, grips, tabs, film seating — adds a nearly
linear term, so

    x_crosshead(F) = d_specimen(F) + C * F

with ``C`` in mm/N. On short, stiff specimens ``C * F`` is not a correction, it is most of
the signal: on the w=18 mm PET hinge it is ~2/3 of the crosshead travel.

``C`` is measured by running one test with an independent displacement reading. Here that is
the fillet-hole video extensometer on the w=18 mm opening test
(:data:`C_HINGE_W18`) — the excess ``x - a_true`` is linear in force to 0.06 mm rms.

Transferring ``C`` between specimens is only approximate: the frame/load-cell part is shared,
but the grip/tab part stiffens with gripped width, so a value measured on a wide sheet is a
LOWER BOUND for a narrow strip. See :func:`fit_compliance`.
"""
from __future__ import annotations

import numpy as np

# Measured on the w=18mm PET hinge opening test (video fillet-hole extensometer vs crosshead,
# 30-278 N window, residual 0.063 mm rms). Wide gripped sheet -> lower bound for narrow strips.
C_HINGE_W18 = 7.77e-3        # mm/N


def fit_compliance(
    disp_mm: np.ndarray,
    force_disp_N: np.ndarray,
    true_mm: np.ndarray,
    force_true_N: np.ndarray,
    *,
    f_lo: float = 30.0,
    f_hi_frac: float = 0.97,
) -> tuple[float, float, float]:
    """Least-squares ``C`` from a crosshead curve and an independent displacement curve.

    The two curves need not share a time base — they are paired through force on the rising
    branch, which is single-valued there.

    Args:
        disp_mm, force_disp_N: crosshead displacement and its force.
        true_mm, force_true_N: independently measured specimen displacement and its force.
        f_lo: bottom of the fit window [N]; skips the toe.
        f_hi_frac: top of the fit window as a fraction of the common peak force.

    Returns:
        ``(C_mm_per_N, offset_mm, residual_rms_mm)``.
    """
    def _rising(d, f):
        d, f = np.asarray(d, float), np.asarray(f, float)
        i = int(np.argmax(f)); d, f = d[: i + 1], f[: i + 1]
        keep = np.concatenate([[True], np.diff(f) > 1e-9])
        return d[keep], f[keep]

    xd, xf = _rising(disp_mm, force_disp_N)
    td, tf = _rising(true_mm, force_true_N)
    grid = np.linspace(f_lo, min(xf.max(), tf.max()) * f_hi_frac, 200)
    excess = np.interp(grid, xf, xd) - np.interp(grid, tf, td)
    C, off = np.polyfit(grid, excess, 1)
    rms = float(np.std(excess - (C * grid + off)))
    return float(C), float(off), rms


def correct_displacement(
    disp_mm: np.ndarray, force_N: np.ndarray, C_mm_per_N: float = C_HINGE_W18
) -> np.ndarray:
    """Crosshead travel minus the load-train term: the specimen's own displacement."""
    return np.asarray(disp_mm, float) - C_mm_per_N * np.asarray(force_N, float)


# Initial-tangent stress window for E. PET starts bending away from linear within a few MPa
# (anelastic pre-yield softening), so a chord taken at 10-20 MPa understates E by ~2x even
# after the compliance correction. Below ~1 MPa the specimen compliance is smaller than C and
# the subtraction goes singular. 1-5 MPa is the usable linear part.
TANGENT_WINDOW_MPA = (1.0, 5.0)


def modulus_from_slope(
    force_N: np.ndarray,
    disp_mm: np.ndarray,
    area_mm2: float,
    L0_mm: float,
    *,
    stress_lo: float = TANGENT_WINDOW_MPA[0],
    stress_hi: float = TANGENT_WINDOW_MPA[1],
    C_mm_per_N: float = 0.0,
) -> tuple[float, float]:
    """Young's modulus by series-compliance bookkeeping over a fixed stress window.

    ``dx/dF = C + L0/(E*A)``, so subtracting ``C`` from the measured slope leaves the
    specimen's own compliance. Working in compliance (not strain) keeps the series
    decomposition additive and makes the correction exact rather than a strain fudge.

    Take the window early (:data:`TANGENT_WINDOW_MPA`) — this is a tangent modulus, not a
    chord. Note the leverage: the specimen carries only ~1/3 of the measured slope, so the
    relative uncertainty in ``C`` is amplified ~2.7x in ``E``.

    Returns:
        ``(E_MPa, total_slope_mm_per_N)``; ``E`` is ``inf`` if the correction over-subtracts.
    """
    force_N, disp_mm = np.asarray(force_N, float), np.asarray(disp_mm, float)
    stress = force_N / float(area_mm2)
    w = (stress >= stress_lo) & (stress <= stress_hi)
    if int(w.sum()) < 3:
        return float("nan"), float("nan")
    slope = float(np.polyfit(force_N[w], disp_mm[w], 1)[0])
    c_spec = slope - C_mm_per_N
    E = L0_mm / (area_mm2 * c_spec) if c_spec > 0 else float("inf")
    return float(E), slope
