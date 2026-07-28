"""Shared figure styling: one charter, one palette, one backend switch.

Every plotting script in ``nff/scripts/figures/`` draws from here. Before this module the rcParams
block was copy-pasted verbatim into 16 scripts, ``apply_charter`` was defined four times, and two
palettes disagreed on what "ink" means (``#1A1A1A`` vs ``#212529``). The charter value wins.

Import order matters for the backend: ``use_agg()`` must run before the first ``pyplot`` import in
a process, which is why it is a function here rather than an import side effect.
"""

# ── Princeton charter palette ────────────────────────────────────────────────────────────────
ORANGE = "#F58025"   # faces / paper / cut pattern
TEAL   = "#2A9D8F"   # hinges / physical loss
RED    = "#D62828"   # loads / target / failure
GREY   = "#6C757D"   # clamps / secondary annotation
INK    = "#1A1A1A"   # text / edges / axes
PURPLE = "#6A4C93"   # boundary sliders
GREEN  = "#1F8A4C"   # deployed boundary points / geometric loss

# ── Experiment-vs-model palette (physical calibration figures) ───────────────────────────────
# Distinct from the charter on purpose: these figures contrast MEASURED against SIMULATED, a
# distinction the charter has no colour for.
SIM   = "#08519c"    # simulated / model prediction
REAL  = "#E8590C"    # measured / experimental
MUTE  = "#adb5bd"    # de-emphasised background series
LIT   = GREY         # literature / reference value
FAIL  = RED          # fracture, tear, out-of-range
TRAIN = "#C9CED4"    # faint training-set cloud behind a held-out split

_RC = {
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.6,
    "axes.edgecolor": INK, "axes.linewidth": 0.9, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": GREY, "ytick.color": GREY,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "legend.frameon": True, "legend.framealpha": 0.9, "legend.edgecolor": "#D3D6DB",
}


def use_agg():
    """Select the headless backend. Call BEFORE importing pyplot, then import it."""
    import matplotlib
    matplotlib.use("Agg")


def apply_charter(**overrides):
    """Apply the project rcParams. Extra keyword pairs override individual entries."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({**_RC, **overrides})


def finite_sorted(x, *ys):
    """Drop non-finite samples across all arrays, then sort every array by ``x``.

    Args:
        x:  (n,) sort key.
        *ys: any number of (n,) companion arrays.

    Returns:
        Tuple ``(x, *ys)`` filtered and sorted, same length in every entry.
    """
    import numpy as np
    x = np.asarray(x, float)
    mask = np.isfinite(x)
    ys = tuple(np.asarray(y, float) for y in ys)
    for y in ys:
        mask &= np.isfinite(y)
    order = np.argsort(x[mask], kind="stable")
    return (x[mask][order],) + tuple(y[mask][order] for y in ys)
