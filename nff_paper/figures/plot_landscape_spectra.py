"""Spectral and geometric structure of the design landscape.

  a  FREQUENCY ANALYSIS. Power spectral density of the loss along uniform transects. The decay
     exponent identifies the regularity class: a step discontinuity gives PSD ~ f^-2, a C^1 kink
     f^-4, a smooth landscape steeper than any power. Independent of the bracket-shrinking test that
     located the discontinuity, so agreement is corroboration and not restatement.

  b  HESSIAN SPECTRAL DENSITY. Conditioning as a distribution rather than a ratio. A bare condition
     number is destroyed by Cauchy interlacing; a spectrum shown against the null distribution of
     random subspaces of equal dimension is not.

  c  DEPARTURE FROM THE FEASIBLE SET. For a unit step in the design variables, how far the flat
     sheet moves off the closed-hinge set. This is the inductive bias as a single number.

    JAX_PLATFORMS=cpu conda run -n kgnn_mac python nff_paper/figures/plot_landscape_spectra.py
"""

import argparse
import os

import numpy as np

from nff.utils.figstyle import GREY, INK, ORANGE, RED, TEAL, apply_charter, use_agg

use_agg()
import matplotlib.pyplot as plt                                        # noqa: E402

D = "data/outputs/chart_ablation_28deg"
PSD_KEYS = [('transverse (crosses the shell)', RED, 'o'),
            ('tangent to the shell', TEAL, '^'),
            ('closed-chart direction', ORANGE, 'D')]


def panel_a(ax, d):
    for name, c, mk in PSD_KEYS:
        f = d[f"psd_freq__{name}"]
        p = d[f"psd_val__{name}"]
        a = float(d[f"psd_alpha__{name}"])
        m = (f > f[2]) & (f < f[-1] * 0.5)
        ax.loglog(f[m], p[m], color=c, lw=1.2, alpha=0.85,
                  label=f"{name}   $\\alpha={a:.2f}$")
    f = d[f"psd_freq__transverse (crosses the shell)"]
    fr = np.array([f[3], f[-1] * 0.5])
    p0 = float(d["psd_val__transverse (crosses the shell)"][3])
    ax.loglog(fr, p0 * (fr / fr[0]) ** -2, ':', color=INK, lw=1.3)
    ax.text(fr[1] * 0.35, p0 * (fr[1] * 0.35 / fr[0]) ** -2 * 3, r'$f^{-2}$  (a jump)',
            fontsize=7.2, color=INK)
    ax.set_xlabel('spatial frequency along the transect', fontsize=8.4)
    ax.set_ylabel('power spectral density', fontsize=8.4)
    ax.set_title('a   the loss spectrum identifies\nthe regularity of each direction',
                 fontsize=8.8, color=INK)
    ax.legend(fontsize=6.1, loc='lower left')


def panel_b(ax, d):
    bins = np.logspace(-3, 4.4, 46)
    ax.hist(np.abs(d["null_ev"]), bins=bins, color=GREY, alpha=0.38,
            weights=np.full(d["null_ev"].size, 1.0 / 60), label='random 12-dim subspaces')
    ax.hist(np.abs(d["ev_full"]), bins=bins, histtype='step', color=RED, lw=1.7,
            label='ambient, 72 dim')
    ax.hist(np.abs(d["ev_chart"]), bins=bins, histtype='step', color=ORANGE, lw=2.1,
            label='closed chart, 12 dim')
    ax.set_xscale('log')
    ax.set_xlabel(r'$|\lambda|$  of the design Hessian', fontsize=8.4)
    ax.set_ylabel('modes per subspace', fontsize=8.4)
    ax.set_title('b   conditioning as a spectrum,\nagainst its null distribution',
                 fontsize=8.8, color=INK)
    n_full = int((d["ev_full"] < 0).sum())
    n_chart = int((d["ev_chart"] < 0).sum())
    ax.text(0.03, 0.62, f'negative modes\nambient {n_full}\nchart {n_chart}',
            transform=ax.transAxes, fontsize=7.0, color=INK, va='top')
    ax.legend(fontsize=6.1, loc='upper right')


def panel_c(ax, d):
    vals = [float(d["dep_chart"]), float(d["dep_free"])]
    ax.bar([0, 1], vals, color=[ORANGE, GREY], alpha=0.78, width=0.5, edgecolor=INK, linewidth=0.6)
    ax.set_yscale('log')
    ax.set_ylim(1e-17, 5e0)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['closed chart\n12 dim', 'random frame\n12 dim'], fontsize=7.6)
    ax.set_ylabel(r'$\|B\,\mathbf{d}\|/\|\mathbf{d}\|$   for a unit design step', fontsize=8.4)
    ax.set_title('c   how fast a design step leaves\nthe feasible set', fontsize=8.8, color=INK)
    for x, v in zip([0, 1], vals):
        ax.text(x, v * 2.4, f'{v:.1e}', ha='center', fontsize=7.4, color=INK)
    ax.axhline(1e-15, color=INK, ls=':', lw=1.0)
    ax.text(1.42, 2e-15, 'machine zero', fontsize=6.9, color=INK, ha='right')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", default=f"{D}/landscape_spectra.npz")
    ap.add_argument("--out", default="paper/Images/landscape_spectra.png")
    ap.add_argument("--dpi", type=int, default=600)
    args = ap.parse_args()

    apply_charter()
    d = np.load(args.npz, allow_pickle=True)
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.7))
    panel_a(axes[0], d)
    panel_b(axes[1], d)
    panel_c(axes[2], d)
    for ax in axes:
        ax.tick_params(labelsize=7.2)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
