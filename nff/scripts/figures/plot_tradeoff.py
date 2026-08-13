"""Feasibility versus optimality on the shared task.

The task is identical for both methods: minimise target-matching error, SUBJECT TO the flat sheet
being a perfect closed cut pattern. For the closed chart the constraint holds identically, so it is
a single point sitting ON the feasible axis. For the baseline the constraint must be attained
numerically while a different objective is descended, so it traces a curve, and the measurement is
how that curve behaves as the closure error is driven towards zero.

Target error is the deficit in overlap between the deployed outline and the target rectangle. It is
not the training objective of either method, so neither can reach it by moving a few boundary
vertices onto a sampled target cloud.

    JAX_PLATFORMS=cpu conda run -n kgnn_mac python nff/scripts/figures/plot_tradeoff.py
"""

import argparse
import os

import numpy as np

from nff.utils.figstyle import GREY, INK, ORANGE, RED, TEAL, apply_charter, use_agg

use_agg()
import matplotlib.pyplot as plt                                        # noqa: E402

FLOOR = 1e-16      # plotting floor for an exactly-zero closure error on a log axis


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", default="data/outputs/chart_ablation_28deg/tradeoff.npz")
    ap.add_argument("--out", default="paper/Images/feasibility_optimality.png")
    ap.add_argument("--dpi", type=int, default=600)
    args = ap.parse_args()

    apply_charter()
    d = np.load(args.npz, allow_pickle=True)
    labels = [str(x) for x in d["labels"]]
    closure, overlap, terr, w = d["closure"], d["overlap"], d["target_err"], d["w"]
    is_chart = np.array([lab.startswith('closed chart') for lab in labels])

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8))

    # ── left: the trade-off itself ───────────────────────────────────────────────────────
    ax = axes[0]
    cb = np.maximum(closure[~is_chart], FLOOR)
    order = np.argsort(w[~is_chart])
    ax.plot(cb[order], terr[~is_chart][order], '-^', color=RED, ms=5.5, lw=1.7,
            markeredgewidth=0, label='free vertices $+$ projection $+$ penalty')
    for x, y, ww in zip(cb[order], terr[~is_chart][order], w[~is_chart][order]):
        ax.annotate(f'$w{{=}}10^{{{int(np.log10(ww))}}}$', (x, y), fontsize=6.2, color=RED,
                    xytext=(3, 5), textcoords='offset points')
    ax.plot(np.maximum(closure[is_chart], FLOOR), terr[is_chart], 'o', color=ORANGE, ms=9,
            markeredgewidth=0, label='closed chart', zorder=5)
    ax.set_xscale('log')
    ax.axvline(FLOOR, color=INK, ls=':', lw=1.2)
    ax.text(FLOOR * 2.2, ax.get_ylim()[1] * 0.96, 'exactly closed', fontsize=7.0, color=INK,
            va='top')
    ax.set_xlabel('closure error  $\\max_h\\|b_h\\|$  [mm]', fontsize=8.5)
    ax.set_ylabel('target error   $1-\\mathrm{IoU}$', fontsize=8.5)
    ax.set_title('the constraint is free for one method\nand a trade-off for the other',
                 fontsize=8.9, color=INK)
    ax.legend(fontsize=6.6, loc='upper right')
    ax.tick_params(labelsize=7.4)

    # ── right: the second half of "a perfect cut pattern" ────────────────────────────────
    ax = axes[1]
    ax.plot(np.maximum(overlap[~is_chart][order], 1e-8), terr[~is_chart][order], '-^',
            color=RED, ms=5.5, lw=1.7, markeredgewidth=0, label='baseline')
    ax.plot(np.maximum(overlap[is_chart], 1e-8), terr[is_chart], 'o', color=ORANGE, ms=9,
            markeredgewidth=0, label='closed chart', zorder=5)
    ax.set_xscale('log')
    ax.set_xlabel('panel self-overlap  (fraction of area)', fontsize=8.5)
    ax.set_ylabel('target error   $1-\\mathrm{IoU}$', fontsize=8.5)
    ax.set_title('and the same for non-self-intersection', fontsize=8.9, color=INK)
    ax.legend(fontsize=6.6, loc='upper right')
    ax.tick_params(labelsize=7.4)

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {args.out}")
    for lab, c, o, t in zip(labels, closure, overlap, terr):
        print(f"    {lab:<26} closure {c:10.3e} mm   overlap {o:8.4f}   target err {t:.4f}")


if __name__ == "__main__":
    main()
