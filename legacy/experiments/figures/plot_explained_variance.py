"""How many design directions the ensemble actually uses, against how many the target constrains.

Two views of the same index axis. Left: the share of the ensemble's spread carried by each
principal direction, and the running total. Right: that spread laid over the curvature of the
objective, to see whether the directions the ensemble explores are the ones the target leaves
flat.

    JAX_PLATFORMS=cpu conda run -n kgnn_mac python \
        legacy/experiments/figures/plot_explained_variance.py \
        --run data/outputs/design_cloud/circle_7x7_n200_s1
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

from nff.utils.figstyle import (FS_ANNOT, GREY, INK, ORANGE, apply_charter, figsize_in,
                                page_bp, panel_letter, paper_rc, save_paper_figure, use_agg)

use_agg()
import matplotlib.pyplot as plt                                       # noqa: E402


def build_figure(run_dir, out_path):
    land = dict(np.load(os.path.join(run_dir, "landscape.npz")))
    var = np.sort(np.abs(np.asarray(land["pca_evals"], dtype=float)))[::-1]
    ratio = var / var.sum()
    cum = np.cumsum(ratio)
    curv = np.sort(np.abs(np.asarray(land["evals_shape"], dtype=float)))[::-1]
    curv = curv / curv.max()
    n_flat = int(land["n_flat"])
    k = np.arange(1, len(ratio) + 1)

    scale = 1.0
    w_bp, h_bp = page_bp(1.0, aspect=2.55)
    apply_charter(**paper_rc(scale))
    fig = plt.figure(figsize=figsize_in(w_bp, h_bp))
    ax_a = fig.add_axes([0.098, 0.235, 0.300, 0.660])
    ax_b = fig.add_axes([0.655, 0.235, 0.320, 0.660])

    # ── a: the scree, with the running total on the right axis ────────────────────
    # Only the leading directions carry anything; the tail runs to 1e-30 and would flatten
    # the whole scree into one block if it were shown.
    n_show = min(len(k), 45)
    ax_a.bar(k[:n_show], ratio[:n_show], width=0.85, color=ORANGE, edgecolor="none",
             zorder=2)
    ax_a.set_yscale("log")
    ax_a.set_ylim(1e-6, 1.0)
    ax_a.set_xlim(0, n_show + 1)
    ax_a.set_xlabel("principal direction")
    ax_a.set_ylabel("share of spread")
    ax_a.tick_params(length=2.4, pad=1.5)

    ax_a2 = ax_a.twinx()
    ax_a2.plot(k, 100 * cum, color=INK, lw=1.1, zorder=3)
    ax_a2.set_ylabel("cumulative (%)")
    ax_a2.set_ylim(0, 105)
    ax_a2.grid(False)
    ax_a2.tick_params(length=2.4, pad=1.5)
    for frac, style, ytxt in ((0.90, (0, (3, 2)), 55), (0.99, (0, (1.2, 1.4)), 25)):
        n = int(np.searchsorted(cum, frac) + 1)
        ax_a2.axvline(n, color=GREY, lw=0.7, ls=style, zorder=1)
        ax_a2.text(n + 1.5, ytxt, f"{100 * frac:.0f}% at {n}", fontsize=FS_ANNOT / scale,
                   color=GREY, va="center", ha="left")

    # ── b: spread against curvature, same index ───────────────────────────────────
    ax_b.plot(k, ratio / ratio.max(), color=ORANGE, lw=1.3, label="ensemble spread")
    ax_b.plot(np.arange(1, len(curv) + 1), np.maximum(curv, 1e-17), color=INK, lw=1.0,
              ls=(0, (1.2, 1.4)), label="curvature")
    ax_b.axvspan(len(curv) - n_flat + 0.5, len(curv) + 0.5, color=ORANGE, alpha=0.10, lw=0)
    ax_b.set_yscale("log")
    ax_b.set_ylim(1e-18, 3.0)
    ax_b.set_xlabel("design direction")
    ax_b.set_ylabel("normalised")
    ax_b.legend(loc="upper right", fontsize=FS_ANNOT / scale, handletextpad=0.5,
                borderpad=0.3, labelspacing=0.25, framealpha=0.0, handlelength=1.6)
    ax_b.tick_params(length=2.4, pad=1.5)

    for ax, letter in ((ax_a, "a"), (ax_b, "b")):
        panel_letter(ax, letter, x=-0.155, y=1.03, scale=scale)

    pdf, lo, hi = save_paper_figure(fig, out_path, w_bp, h_bp, dpi=600, scale=scale)
    plt.close(fig)
    return pdf, {"n_90pct": int(np.searchsorted(cum, 0.90) + 1),
                 "n_99pct": int(np.searchsorted(cum, 0.99) + 1),
                 "top1": float(ratio[0]), "top2": float(ratio[:2].sum()),
                 "n_flat": n_flat, "type_pt": [lo, hi]}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(args.run, "explained_variance.pdf")
    pdf, info = build_figure(args.run, out)
    print(f"  wrote {pdf}")
    for k, v in info.items():
        print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
