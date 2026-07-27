"""Visualize observed hinge displacement paths against what the oracle actually trains on.

Two questions this figure is built to answer at a glance:

  1. WHAT SHAPE are the paths? The oracle can only produce straight proportional rays
     ``u(lambda) = lambda * u1``. Solid curves are observed; dotted chords are the ray each endpoint
     would be reduced to. Divergence between them = what a single ray cannot represent.
  2. IS COMPRESSION NEEDED? ``sample_jobs`` samples ``eta_a`` in [0, 1] and ``_domain_barrier``
     penalizes ``a < 0`` -- the closed hinge is assumed to only ever OPEN. The red half-plane is
     that forbidden region; any path entering it is in a regime the surrogate never saw.

Run (one or more path directories written by ``diagnostics/extract_hinge_paths.py``):
    PYTHONPATH=$(pwd) conda run -n kgnn_mac python nff/scripts/figures/plot_hinge_paths.py \
        --paths data/outputs/hinge_paths/sheet_4x8ft:surrogate \
                data/outputs/hinge_paths/sheet_4x8ft_rom:ROM \
        --out data/outputs/hinge_paths/comparison.png
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from nff.closed.hinge_paths import load_paths
from nff.models.hinge_surrogate import DOMAIN

ORANGE, INK, GREY, RED = "#F58025", "#1A1A1A", "#6C757D", "#D62828"
SERIES = [ORANGE, INK, "#00838F", "#6A1B9A"]


def _box(ax, x, y):
    """Outline the region ``sample_jobs`` actually covers, in whatever pair of axes is plotted."""
    lim = {'eta_a': (0.0, DOMAIN['eta_a_max']),
           'eta_s': (-DOMAIN['eta_s_max'], DOMAIN['eta_s_max']),
           'theta': (0.0, np.degrees(DOMAIN['theta_max'])),
           'lam': None}
    (x0, x1), (y0, y1) = lim[x], lim[y]
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="none", edgecolor=GREY,
                               lw=1.1, ls="--", zorder=1))


def _shade_compression(ax, vertical=True):
    """The forbidden a<0 half-plane: sampled by no oracle job, penalized by the domain barrier.

    Called AFTER the data limits are fixed, and restores them afterwards -- spanning to an arbitrary
    -10 would otherwise rescale the axis and squash the paths into a sliver.
    """
    lo, hi = (ax.get_xlim() if vertical else ax.get_ylim())
    span = ax.axvspan if vertical else ax.axhspan
    line = ax.axvline if vertical else ax.axhline
    span(lo, 0, color=RED, alpha=0.07, zorder=0, lw=0)
    line(0, color=RED, lw=0.9, alpha=0.55, zorder=1)
    (ax.set_xlim if vertical else ax.set_ylim)(lo, hi)


def _fit_limits(ax, vals, pad=0.08, floor=None):
    """Limits from the DATA (plus the box edge), so the paths fill the panel."""
    lo, hi = float(np.min(vals)), float(np.max(vals))
    if floor is not None:
        lo, hi = min(lo, floor[0]), max(hi, floor[1])
    m = pad * max(hi - lo, 1e-9)
    return lo - m, hi + m


def build_figure(runs, out_png: str):
    """Render the two-row path figure for one or more ``(HingePaths, label)`` runs.

    Shared with ``diagnostics/extract_hinge_paths.py`` so the quick-look figure a single extraction
    writes and the multi-run comparison are the same plot, not two drifting implementations.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.6))

    # ── row 1: the (eta_a, eta_s, theta) manifold, three projections ──
    proj = [(0, 2, r"$\eta_a=a/w_{lig}$", r"$\theta$  [deg]", 'eta_a', 'theta'),
            (1, 2, r"$\eta_s=s/w_{lig}$", r"$\theta$  [deg]", 'eta_s', 'theta'),
            (0, 1, r"$\eta_a=a/w_{lig}$", r"$\eta_s=s/w_{lig}$", 'eta_a', 'eta_s')]
    lim_of = {'eta_a': (0.0, DOMAIN['eta_a_max']),
              'eta_s': (-DOMAIN['eta_s_max'], DOMAIN['eta_s_max']),
              'theta': (0.0, np.degrees(DOMAIN['theta_max']))}
    for ax, (i, j, xl, yl, kx, ky) in zip(axes[0], proj):
        allX, allY = [], []
        for (paths, _), col in zip(runs, SERIES):
            e = paths.eta
            X = np.degrees(e[..., 2]) if kx == 'theta' else e[..., i]
            Y = np.degrees(e[..., 2]) if ky == 'theta' else e[..., j]
            allX.append(X); allY.append(Y)
            for h in range(paths.n_hinges):
                # dotted chord = the proportional ray the oracle would run for this endpoint
                ax.plot([0, X[-1, h]], [0, Y[-1, h]], ":", color=col, lw=0.8, alpha=0.55, zorder=2)
                ax.plot(X[:, h], Y[:, h], "-", color=col, lw=1.3, alpha=0.9, zorder=3)
            ax.plot(X[-1], Y[-1], "o", color=col, ms=4.0, mec="white", mew=0.6, zorder=4)
        # scale to data AND box, then shade -- order matters (see _shade_compression)
        ax.set_xlim(*_fit_limits(ax, np.concatenate([a.ravel() for a in allX]), floor=lim_of[kx]))
        ax.set_ylim(*_fit_limits(ax, np.concatenate([a.ravel() for a in allY]), floor=lim_of[ky]))
        _box(ax, kx, ky)
        if kx == 'eta_a':
            _shade_compression(ax, vertical=True)
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.spines[['top', 'right']].set_visible(False)

    # ── row 2: pseudo-time traces -- where reversal and compression actually happen ──
    for ax, (k, yl) in zip(axes[1], [(0, r"$\eta_a$"), (1, r"$\eta_s$"), (2, r"$\theta$  [deg]")]):
        allY = []
        for (paths, _), col in zip(runs, SERIES):
            Y = np.degrees(paths.eta[..., 2]) if k == 2 else paths.eta[..., k]
            allY.append(Y)
            for h in range(paths.n_hinges):
                ax.plot(paths.load_fraction, Y[:, h], "-", color=col, lw=1.1, alpha=0.85)
        ax.set_ylim(*_fit_limits(ax, np.concatenate([a.ravel() for a in allY])))
        if k == 0:
            _shade_compression(ax, vertical=False)
        else:
            ax.axhline(0, color="k", lw=0.5, alpha=0.3)
        ax.set_xlabel("load fraction  $\\lambda$"); ax.set_ylabel(yl)
        ax.spines[['top', 'right']].set_visible(False)

    handles = [Line2D([], [], color=c, lw=1.6, label=l) for (_, l), c in zip(runs, SERIES)]
    handles += [Line2D([], [], color=GREY, lw=1.1, ls="--", label="oracle sampled box"),
                Line2D([], [], color=RED, lw=6, alpha=0.25,
                       label=r"compression ($a<0$): never sampled")]
    fig.legend(handles=handles, frameon=False, ncol=len(handles), loc="lower center",
               bbox_to_anchor=(0.5, -0.015), fontsize=9)
    fig.suptitle("Observed hinge displacement paths vs the surrogate's training domain", fontsize=12)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=175, bbox_inches="tight")
    plt.close(fig)
    return out_png


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--paths", nargs="+", required=True,
                   help="one or more '<dir>[:label]' written by extract_hinge_paths")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    runs = []
    for spec in args.paths:
        d, _, label = spec.partition(":")
        runs.append((load_paths(os.path.join(d, "hinge_paths")), label or os.path.basename(d)))
    print(f"  wrote {build_figure(runs, args.out)}")


if __name__ == "__main__":
    main()
