"""The numerical argument for the closed-state chart, in four panels.

The task is identical for both methods: match the target, subject to the flat sheet being a perfect
closed cut pattern. The closed chart satisfies the constraint identically; the strongest baseline we
could construct -- free panel vertices with a hard projection for hinge closure and penalties for
non-overlap and for tiling the stock -- must attain it numerically while descending the objective.

  A  the landscape each method presents to its optimiser
  B  whether raising the penalty ever reaches a manufacturable sheet
  C  whether the gradient is a gradient, ON the feasible set where the optimum lies
  D  what 100 independent runs produce

Panels deliberately absent: design-Hessian conditioning (the chart sits at the 96th percentile of
random subspaces of its own dimension -- the comparison is a statement about dimension, not about
the chart) and solution uniqueness (Procrustes alignment removes under 14% of the spread, so the
multi-modality is genuine for both methods).

    JAX_PLATFORMS=cpu conda run -n kgnn_mac python nff/scripts/figures/plot_pipeline_argument.py
"""

import argparse
import os

import numpy as np

from nff.utils.figstyle import GREY, INK, ORANGE, apply_charter, use_agg

use_agg()
import matplotlib.pyplot as plt                                        # noqa: E402
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec      # noqa: E402

D = "data/outputs/chart_ablation_28deg"
C_CHART, C_BASE = ORANGE, "#1f6fb2"          # colourblind-safe warm/cool pair
NAME = {'chart': 'closed chart', 'baseline': 'free vertices $+$ projection $+$ penalties'}
A_MM, CUT_MM = 2.973e6, 7329.0               # sheet area and interior cut length, for mm conversion
BAR = 1e-3                                   # manufacturability threshold: 0.41 mm, ~2x laser kerf


def panel_a(fig, cell, tj):
    """Loss slices in the plane of each optimiser's own iterates, with its trajectory."""
    gs = GridSpecFromSubplotSpec(2, 2, cell, height_ratios=[3.0, 1.05], hspace=0.46, wspace=0.16)
    lo = min(tj[f"{k}__slice"].min() for k in ('chart', 'baseline'))
    hi = tj["chart__slice"].max()
    lv = np.linspace(lo, hi, 20)
    for col, kind in enumerate(('chart', 'baseline')):
        ax = fig.add_subplot(gs[0, col])
        Z, ex = tj[f"{kind}__slice"], tj[f"{kind}__slice_extent"]
        ax.contourf(Z, levels=lv, extent=ex, cmap='Greys', alpha=0.9, extend='max')
        ax.contour(Z, levels=lv, extent=ex, colors='white', linewidths=0.3, alpha=0.5)
        P, sz = tj[f"{kind}__traj2d"], tj[f"{kind}__traj_size"]
        c = C_CHART if kind == 'chart' else C_BASE
        ax.plot(P[:, 0], P[:, 1], '-', color=c, lw=1.0, alpha=0.85, zorder=4)
        ax.scatter(P[:, 0], P[:, 1], s=sz, color=c, alpha=0.7, linewidths=0, zorder=5)
        ax.plot(P[0, 0], P[0, 1], 'o', mfc='none', mec=INK, ms=6.5, mew=1.2, zorder=6)
        ax.plot(P[-1, 0], P[-1, 1], '*', color=INK, ms=10, zorder=6)
        rg = np.log10(np.maximum(tj[f"{kind}__range"], 1e-300))
        ax.text(0.035, 0.955, f'spans $10^{{{rg[0]:.1f}}}$–$10^{{{rg[1]:.0f}}}$',
                transform=ax.transAxes, fontsize=6.4, color=INK, va='top',
                bbox=dict(fc='white', ec='none', alpha=0.8, pad=1.6))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(NAME[kind], fontsize=7.2, color=c)
        if col == 0:
            ax.set_ylabel('$\\log_{10}$ loss,\nplane of the iterates', fontsize=7.2)
        axm = fig.add_subplot(gs[1, col])
        s = tj[f"{kind}__seg"]
        axm.plot(np.linspace(0, 1, s.size), s, '-', color=c, lw=1.4)
        axm.set_yscale('log')
        axm.set_xticks([0, 1]); axm.set_xticklabels(['init', 'opt'], fontsize=6.4)
        axm.tick_params(labelsize=6.2)
        if col == 0:
            axm.set_ylabel('loss along\nthe segment', fontsize=6.8)
    fig.text(0.058, 0.517, 'marker size = inner-solver iterations, shared scale '
             '(measured comparable: 298–381 vs 306–467)', fontsize=5.8, color=GREY)


def panel_b(fig, cell, pu):
    """Does raising the tiling penalty ever produce a manufacturable sheet?"""
    ax = fig.add_subplot(cell)
    w, tile, ep, st = pu["w"], pu["tile"], pu["ep"], pu["stalled"]
    m2 = ep == 200
    ax.loglog(w[m2], tile[m2], '-^', color=C_BASE, ms=7, lw=1.8, markeredgewidth=0,
              label='tiling penalty, 200 epochs')
    m8 = ep == 800
    ax.loglog(w[m8], tile[m8], 's', color=C_BASE, ms=6.5, mfc='none', mew=1.4,
              label='4$\\times$ the budget (800 epochs)')
    # the corrected point: stalled seeds removed
    ax.loglog([1e4], [4.87e-3], 'o', color='#D62828', ms=9, markeredgewidth=0, zorder=6)
    ax.annotate('best over five decades,\nstalled runs excluded:\n4.9e-3 = 1.98 mm',
                (1e4, 4.87e-3), fontsize=6.2, color='#D62828',
                xytext=(20, 26), textcoords='offset points',
                arrowprops=dict(arrowstyle='-', color='#D62828', lw=0.8))
    ax.axhline(BAR, color=INK, ls='--', lw=1.3)
    ax.text(w.min() * 0.75, BAR * 0.30, 'manufacturable threshold\n0.41 mm $\\approx$ 2$\\times$ laser kerf',
            fontsize=6.4, color=INK, va='top')
    ax.axhline(1e-16, color=C_CHART, lw=2.4)
    ax.text(w.min(), 2.2e-16, 'closed chart — exact, at every weight and every seed',
            fontsize=6.6, color=C_CHART)
    ax.set_ylim(1e-17, 2e-1)
    ax.set_xlabel('tiling penalty weight  $w$', fontsize=8.2)
    ax.set_ylabel('tiling deficit of the flat sheet', fontsize=8.2)
    ax.legend(fontsize=6.2, loc='lower right')
    ax.tick_params(labelsize=7.2)
    axr = ax.secondary_yaxis('right', functions=(lambda t: t * A_MM / CUT_MM,
                                                 lambda g: g * CUT_MM / A_MM))
    axr.set_ylabel('equivalent gap [mm]', fontsize=7.6)
    axr.tick_params(labelsize=6.6)


def panel_c(fig, cell, ft):
    """Taylor verification ON the feasible set, with the off-manifold control."""
    ax = fig.add_subplot(cell)
    hs = ft["hs"]
    series = [('chart_feasible', C_CHART, 'o', 'closed chart, on the feasible set'),
              ('baseline_feasible', C_BASE, '^', 'baseline, on the feasible set'),
              ('baseline_generic', GREY, 's', 'baseline, away from it  (control)')]
    for key, c, mk, lab in series:
        med, q1, q3 = ft[f"{key}__med"], ft[f"{key}__q1"], ft[f"{key}__q3"]
        o = float(ft[f"{key}__order"])
        ax.fill_between(hs, q1, q3, color=c, alpha=0.20, linewidth=0)
        ax.loglog(hs, med, '-', color=c, marker=mk, ms=3.6, lw=1.7, markeredgewidth=0,
                  label=f'{lab}   —  order {o:.2f}')
        fl = float(ft[f"{key}__floor"])
        if np.isfinite(fl) and fl < 1e-5:
            ax.axhline(fl, color=c, ls=':', lw=0.9, alpha=0.8)
    ax.loglog(hs, hs ** 2 / hs.max() ** 2 * 3e-2, ':', color=INK, lw=1.2)
    ax.loglog(hs, hs / hs.max() * 3e0, '--', color=INK, lw=1.0, alpha=0.5)
    ax.text(hs.min() * 2.2, (hs.min() * 2.2) ** 2 / hs.max() ** 2 * 8e-2, 'slope 2',
            fontsize=6.8, color=INK)
    ax.text(hs.min() * 2.2, hs.min() * 2.2 / hs.max() * 7e0, 'slope 1', fontsize=6.8, color=INK)
    ax.text(0.985, 0.03, 'dotted horizontals: precision floor of each objective',
            transform=ax.transAxes, fontsize=5.9, color=GREY, ha='right')
    ax.set_xlabel(r'perturbation step  $h$', fontsize=8.2)
    ax.set_ylabel(r'Taylor remainder  $R_2$   (median, IQR band)', fontsize=8.0)
    ax.legend(fontsize=6.0, loc='upper left')
    ax.tick_params(labelsize=7.2)


def panel_d(fig, cell, pd):
    """100 independent runs: what each one produced."""
    ax = fig.add_subplot(cell)
    kind, viol, terr = pd["kind"], np.abs(pd["viol"]), pd["terr"]
    for k, c, mk in (('baseline', C_BASE, '^'), ('chart', C_CHART, 'o')):
        m = kind == k
        ax.scatter(np.maximum(viol[m], 1e-17), terr[m], s=26, marker=mk, color=c,
                   alpha=0.62, linewidths=0.3, edgecolors='white', label=NAME[k])
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.axvline(BAR, color=INK, ls='--', lw=1.2)
    ax.text(BAR * 0.6, ax.get_ylim()[0] * 1.2, 'manufacturable $\\leftarrow$', fontsize=6.6,
            color=INK, ha='right')
    for k, c in (('chart', C_CHART), ('baseline', C_BASE)):
        m = kind == k
        ax.plot(np.median(np.maximum(viol[m], 1e-17)), np.median(terr[m]), '*', color=c,
                ms=15, mec='white', mew=0.8, zorder=6)
    ax.set_xlabel('constraint violation of the flat sheet', fontsize=8.2)
    ax.set_ylabel('target error   $1-\\mathrm{IoU}$', fontsize=8.2)
    ax.legend(fontsize=6.2, loc='upper left')
    ax.tick_params(labelsize=7.2)
    n_ok = {k: int((np.abs(pd["tile"])[kind == k] < 1e-9).sum()) for k in ('chart', 'baseline')}
    ax.text(0.5, -0.19, f'exact cut patterns: {n_ok["chart"]}/100 versus {n_ok["baseline"]}/100.  '
            'Stars mark medians.', transform=ax.transAxes, fontsize=6.2, color=GREY, ha='center')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="paper/Images/pipeline_argument.png")
    ap.add_argument("--dpi", type=int, default=600)
    args = ap.parse_args()

    apply_charter()
    tj = np.load(f"{D}/traj_panels.npz", allow_pickle=True)
    pu = np.load(f"{D}/push.npz", allow_pickle=True)
    ft = np.load(f"{D}/feasible_taylor_fit.npz", allow_pickle=True)
    pd = np.load(f"{D}/panelD.npz", allow_pickle=True)

    fig = plt.figure(figsize=(13.0, 9.6))
    gs = GridSpec(2, 2, hspace=0.34, wspace=0.26, figure=fig)
    panel_a(fig, gs[0, 0], tj)
    panel_b(fig, gs[0, 1], pu)
    panel_c(fig, gs[1, 0], ft)
    panel_d(fig, gs[1, 1], pd)
    for lab, cell in zip('ABCD', [gs[0, 0], gs[0, 1], gs[1, 0], gs[1, 1]]):
        bb = cell.get_position(fig)
        fig.text(bb.x0 - 0.030, bb.y1 + 0.014, lab, fontsize=14, fontweight='bold', color=INK)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
