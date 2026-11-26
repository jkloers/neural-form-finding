"""The differentiable pipeline, measured: regularity, curvature, smoothness, and what isolates it.

Four panels, each a mathematical diagnostic of the pipeline rather than a picture of a design.

  a  REGULARITY. |L(t*+d) - L(t*-d)| as the bracket d shrinks around the shell |b| = 1e-4, where
     the ligament frame switches branch. A continuous objective gives slope 1; a jump gives slope
     0. The transverse ray carries its OWN control bracket taken away from the crossing, so the
     only difference between those two curves is whether the bracket straddles the shell.

  b  CURVATURE AS A LIMIT. lam_max of the design Hessian against the finite-difference step. An
     operator that exists has a limit; one that does not is not a Hessian. The naive chart is
     differencing across the discontinuity of panel (a).

  c  SMOOTHNESS, DIMENSION-CONTROLLED. beta = sup ||grad L(y1) - grad L(y2)|| / ||y1 - y2|| in mm,
     whose reciprocal 2/beta bounds the stable gradient-descent step. beta is a supremum over a
     subspace, so a 12-dimensional chart is flattered exactly as a condition number would be; the
     closed chart is therefore placed against the distribution of beta over RANDOM subspaces of
     its own dimension.

  d  WHAT SURVIVES THOSE CONTROLS. A random 12-frame has the chart's dimension and its smoothness
     and still produces no cut pattern. What remains is which twelve directions.

    JAX_PLATFORMS=cpu conda run -n kgnn_mac python nff_paper/figures/plot_pipeline_math.py
"""

import argparse
import os

import numpy as np

from nff.utils.figstyle import GREY, INK, ORANGE, RED, TEAL, apply_charter, use_agg

use_agg()
import matplotlib.pyplot as plt                                        # noqa: E402
from matplotlib.gridspec import GridSpec                               # noqa: E402

D = "data/outputs/chart_ablation_28deg"
# Arm R is a six-seed control read from its run; it is not a saved sweep.
ARM_R_TILING = np.array([0.9819, 0.9829, 0.9743, 0.9943, 0.9803, 0.9877])
FREE_TILING = 0.8189        # free vertices + projection, 72 dim, 12-seed median
CHART_TILING = 1.0000


def panel_a(ax):
    """The bracket-shrinking limit test at the frame-switch shell."""
    d = np.load(f"{D}/regularity.npz", allow_pickle=True)
    series = [('transverse, at the shell', 'transverse', 'jumps', RED, '-', 'o'),
              ('transverse, away from it', 'transverse', 'ctrl', GREY, '--', 's'),
              ('tangent to the shell', 'tangent', 'jumps', TEAL, '-', '^'),
              ('closed-chart direction', 'chart mode 1', 'jumps', ORANGE, '-', 'D')]
    for label, key, field, c, ls, mk in series:
        ax.loglog(d[f"{key}__deltas"], np.abs(d[f"{key}__{field}"]), ls, color=c, marker=mk,
                  ms=4, lw=1.6, label=label, markeredgewidth=0)
    x = d["transverse__deltas"]
    ax.loglog(x, 2.5e-4 * x / x.max(), ':', color=INK, lw=1.1, zorder=1)
    ax.text(x.max(), 3.2e-4, 'slope 1', fontsize=6.9, color=INK, ha='right', va='bottom')
    ax.set_xlabel(r'bracket half-width  $\delta$', fontsize=8.3)
    ax.set_ylabel(r'$|\,L(t^{*}{+}\delta)-L(t^{*}{-}\delta)\,|$', fontsize=8.3)
    ax.set_title('a   the objective jumps where the\nligament frame switches branch',
                 fontsize=8.7, color=INK)
    ax.legend(fontsize=6.3, loc='lower right')


def panel_b(ax):
    """Does the design Hessian exist as a limit?"""
    d = np.load(f"{D}/hessian_limit.npz")
    eps, Bm, Dm = d["eps"], d["B"], d["D"]
    ax.loglog(eps, Bm[:, 1], '-o', color=RED, ms=4.5, lw=1.7, markeredgewidth=0,
              label='free vertices, naive')
    ax.loglog(eps, Dm[:, 1], '-s', color=TEAL, ms=4.5, lw=1.7, markeredgewidth=0,
              label='free $+$ projection')
    ax.loglog(eps, Dm[:, 4], '-D', color=ORANGE, ms=4.5, lw=1.7, markeredgewidth=0,
              label='closed chart')
    ax.invert_xaxis()
    ax.set_xlabel(r'finite-difference step  $\varepsilon$   (refining $\rightarrow$)',
                  fontsize=8.3)
    ax.set_ylabel(r'$\lambda_{\max}$  of the design Hessian', fontsize=8.3)
    ax.set_title('b   the naive curvature has no limit,\nso it is not a Hessian',
                 fontsize=8.7, color=INK)
    ax.legend(fontsize=6.5, loc='upper left')


def panel_c(ax):
    """Smoothness against the null model of random subspaces of equal dimension."""
    d = np.load(f"{D}/gradient_quality.npz", allow_pickle=True)
    ax.hist(d["null_beta"], bins=11, color=GREY, alpha=0.40, edgecolor=GREY, linewidth=0.6,
            label='random 12-dim subspaces')
    ax.axvline(float(d["free, naive__beta"]), color=RED, lw=1.7, ls=':', label='naive, 72 dim')
    ax.axvline(float(d["free + projection__beta"]), color=TEAL, lw=1.7, ls='--',
               label='free $+$ projection, 72 dim')
    ax.axvline(float(d["closed chart__beta"]), color=ORANGE, lw=2.3,
               label='closed chart, 12 dim')
    ax.set_xscale('log')
    ax.set_xlabel(r'smoothness modulus  $\hat\beta$   [mm$^{-1}$]', fontsize=8.3)
    ax.set_ylabel('random subspaces', fontsize=8.3)
    ax.set_title('c   the chart is smooth, and so is\nany subspace of its dimension',
                 fontsize=8.7, color=INK)
    ax.legend(fontsize=6.1, loc='upper center')


def panel_d(ax):
    """Dimension and smoothness controlled; only the choice of subspace differs."""
    xs, cols = [0, 1, 2], [RED, GREY, ORANGE]
    vals = [FREE_TILING, float(np.median(ARM_R_TILING)), CHART_TILING]
    ax.bar(xs, vals, color=cols, alpha=0.72, width=0.58, zorder=3, edgecolor=INK, linewidth=0.6)
    ax.scatter([1] * len(ARM_R_TILING), ARM_R_TILING, color=INK, s=10, zorder=5,
               label='individual seeds')
    ax.axhline(1.0, color=INK, ls=':', lw=1.1, zorder=2)
    ax.set_ylim(0.74, 1.06)
    ax.set_xticks(xs)
    ax.set_xticklabels(['free vertices\n72 dim', 'random frame\n12 dim', 'closed chart\n12 dim'],
                       fontsize=7.4)
    ax.set_ylabel('flat panels tiling the stock sheet', fontsize=8.3)
    ax.set_title('d   with dimension controlled, only the\nright twelve directions arrive',
                 fontsize=8.7, color=INK)
    # Count labels clear the seed scatter on the middle bar, which reaches higher than its median.
    tops = [vals[0], max(ARM_R_TILING.max(), vals[1]), vals[2]]
    for x, top, n in zip(xs, tops, ['0/12', '0/6', '12/12']):
        ax.text(x, top + 0.012, n, ha='center', fontsize=7.3, color=INK)
    ax.legend(fontsize=6.3, loc='lower right')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="paper/Images/pipeline_math.png")
    ap.add_argument("--dpi", type=int, default=600)
    args = ap.parse_args()

    apply_charter()
    fig = plt.figure(figsize=(8.4, 6.4))
    gs = GridSpec(2, 2, hspace=0.44, wspace=0.30, figure=fig)
    for fn, cell in ((panel_a, gs[0, 0]), (panel_b, gs[0, 1]),
                     (panel_c, gs[1, 0]), (panel_d, gs[1, 1])):
        ax = fig.add_subplot(cell)
        fn(ax)
        ax.tick_params(labelsize=7)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
