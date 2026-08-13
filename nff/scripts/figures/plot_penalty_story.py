"""Why a penalty cannot close the flat sheet: raising it wrecks the problem it is trying to fix.

Top row: the trained flat design at four penalty weights spanning six decades, against the stock
sheet it must be cut from. Bottom: what that costs. The residual hinge gap does not fall to zero at
any weight -- it bottoms out and then climbs -- and the reason is in the right-hand panel, where the
conditioning of the penalised design Hessian rises without bound and the step over which the
gradient can be trusted collapses with it.

The penalised Hessian family is EXACT rather than sampled: the hinge-gap term is ||By||^2 at the
Stage-0 output, so H(w) = H_0 + 2w B^T B follows from a single Hessian with no retraining.

    JAX_PLATFORMS=cpu conda run -n kgnn_mac python nff/scripts/figures/plot_penalty_story.py
"""

import argparse
import os

import numpy as np

from nff.utils.figstyle import GREY, INK, ORANGE, RED, TEAL, apply_charter, use_agg

use_agg()
import matplotlib.pyplot as plt                                        # noqa: E402
from matplotlib.gridspec import GridSpec                               # noqa: E402
from matplotlib.patches import Polygon as MplPolygon                   # noqa: E402


def draw(ax, y, stock, scale, F, K, title):
    n = np.asarray(y).reshape(F, K, 2)
    ax.plot(stock[:, 0] * scale, stock[:, 1] * scale, color=INK, lw=1.0, ls='--',
            dashes=(5, 3), zorder=6)
    for f in range(F):
        ax.add_patch(MplPolygon(n[f] * scale, closed=True, facecolor=ORANGE, alpha=0.6,
                                edgecolor=INK, linewidth=0.6, zorder=3))
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=8.0, color=INK, linespacing=1.35)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", default="data/outputs/chart_ablation_28deg/penalty_story.npz")
    ap.add_argument("--out", default="paper/Images/penalty_story.png")
    ap.add_argument("--dpi", type=int, default=600)
    args = ap.parse_args()

    apply_charter()
    d = np.load(args.npz)
    w, sheets, gaps, tiles = d['w_train'], d['sheets'], d['gaps'], d['tiles']
    w_an, cond_an, radii = d['w_an'], d['cond_an'], d['radii']
    stock, scale = d['stock'], 406.4
    F = sheets.shape[1] // 8 if sheets.ndim == 2 else 9
    F, K = 9, 4

    fig = plt.figure(figsize=(8.2, 5.4))
    gs = GridSpec(2, 4, height_ratios=[1.0, 0.85], hspace=0.42, wspace=0.55, figure=fig)

    # At large w the optimiser never improves on its starting point, so the recorded best design
    # IS the untrained sheet -- which is closed and tiles perfectly, and would read as a success
    # unless it is labelled for what it is.
    failed = (gaps < 1e-9) & (tiles > 0.999)
    for i in range(len(w)):
        head = f"$w=10^{{{int(np.log10(w[i]))}}}$"
        sub = ("TRAINING FAILED\nbest = untrained start" if failed[i]
               else f"gap {gaps[i]:.2g} mm\ntiling {tiles[i]:.3f}")
        ax = fig.add_subplot(gs[0, i])
        draw(ax, sheets[i], stock, scale, F, K, f"{head}\n{sub}")
        if failed[i]:
            ax.set_title(f"{head}\n{sub}", fontsize=8.0, color=RED, linespacing=1.35)

    # what you get -- residual gap only; tiling is already stated in the panel titles above, and
    # a twin axis here collides with the neighbouring panel for no information gain.
    ax = fig.add_subplot(gs[1, :2])
    live = ~failed
    w_fail = w[failed].min() if failed.any() else None
    if w_fail is not None:
        ax.axvspan(w_fail / 3, w.max() * 3, color=RED, alpha=0.09, zorder=0)
        ax.text(w_fail * 1.2, gaps[live].max() * 0.35,
                'no step improves\non the start', fontsize=7.2, color=RED, va='top')
    ax.plot(w[live], gaps[live], 'o-', color=RED, lw=1.8, ms=6, zorder=4)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(w.min() / 3, w.max() * 3)
    ax.set_ylim(1e-10, 1e3)
    ax.axhline(1e-9, color=GREY, ls=':', lw=1.1)
    ax.text(w.min() / 2.2, 2.2e-9, 'exactly closed  (never reached)', fontsize=7.2, color=GREY,
            va='bottom')
    ax.set_xlabel('hinge-gap penalty weight $w$', fontsize=8.5)
    ax.set_ylabel('residual hinge gap [mm]', fontsize=8.5)
    ax.tick_params(labelsize=7.5)
    ax.set_title('the sheet never closes, at any weight', fontsize=8.8, color=INK, pad=7)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)

    # why
    ax = fig.add_subplot(gs[1, 2:])
    ax.plot(w_an, cond_an, '-', color=INK, lw=2.0, label=r'cond$(H_0+2wB^{\mathsf{T}}B)$')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('hinge-gap penalty weight $w$', fontsize=8.5)
    ax.set_ylabel('condition number', fontsize=8.5)
    ax.tick_params(labelsize=7.5)
    ax.set_title('the gradient stays exact — and becomes unusable', fontsize=8.8, color=INK, pad=7)
    ax3 = ax.twinx()
    ax3.plot(w, radii / np.sqrt(F * K) * scale, 'o--', color=ORANGE, lw=1.6, ms=6)
    ax3.set_yscale('log'); ax3.set_ylim(1e-2, 1e1)
    ax3.set_ylabel('trust radius [mm]', fontsize=8.5, color=ORANGE)
    ax3.tick_params(labelsize=7.5, colors=ORANGE)
    ax.text(3e3, cond_an.max() * 0.35, r'$\propto w$', fontsize=9, color=INK)
    ax3.text(2e5, 0.22, 'trust radius FLAT', fontsize=7.2, color=ORANGE, va='bottom')
    for sp in ('top',):
        ax.spines[sp].set_visible(False); ax3.spines[sp].set_visible(False)

    fig.text(0.5, 0.005, "dashed outline = the stock sheet the pattern must be cut from.  The "
             "penalty is quadratic, so it adds no gradient ERROR — it adds CURVATURE, and no single "
             "step size serves both terms.",
             ha='center', fontsize=7.4, color=GREY)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {args.out}")
    for i in range(len(w)):
        print(f"    w={w[i]:8.0e}  gap {gaps[i]:9.3f} mm  tiling {tiles[i]:.4f}  "
              f"trust {radii[i]/np.sqrt(F*K)*scale:8.4f} mm")


if __name__ == "__main__":
    main()
