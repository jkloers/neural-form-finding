"""Three views of one question: does the circle determine the cut pattern?

    a  where the ensemble ends up, against how well it scores
    b  which parts of the sheet the target pins down, and which it leaves free
    c  whether the ensemble spreads along the directions the objective leaves flat

Panel c is the one that separates degeneracy from slow convergence: if variance tracks the
inverse of curvature, the spread is a property of the problem, not of the optimiser.

    JAX_PLATFORMS=cpu conda run -n kgnn_mac python \
        nff/scripts/figures/plot_design_degeneracy.py \
        --run data/outputs/design_cloud/circle_7x7_n64_s1
"""

import argparse
import json
import os
import sys

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

from nff.utils.figstyle import (FS_ANNOT, GREY, INK, ORANGE, TEAL, apply_charter,
                                figsize_in, page_bp, panel_letter, paper_rc,
                                save_paper_figure, use_agg)

use_agg()
import matplotlib.pyplot as plt                                       # noqa: E402
from matplotlib.collections import PolyCollection                     # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize      # noqa: E402


def _sequential_orange():
    return LinearSegmentedColormap.from_list("charter_seq", ["#FDF0E4", ORANGE, "#8A3D00"])


FAMILY_COLOURS = (ORANGE, TEAL, "#6A4C93")


def panel_scatter(ax, d_end, err_end, d_start, err_start, labels, scale=1.0):
    """Shape error against distance from the ensemble medoid.

    A band of endpoints at essentially one error, spread across a wide range of distances,
    is the signature: the score stops distinguishing designs long before the designs agree.
    """
    ax.scatter(d_start, err_start, s=7, facecolors="none", edgecolors=GREY, linewidths=0.5,
               label="random start", zorder=2)
    for c in np.unique(labels):
        m = labels == c
        ax.scatter(d_end[m], err_end[m], s=12, c=FAMILY_COLOURS[c % len(FAMILY_COLOURS)],
                   edgecolors=INK, linewidths=0.28, label=f"family {chr(65 + c)}", zorder=3)
    ax.set_yscale("log")
    from matplotlib.ticker import FixedLocator, ScalarFormatter
    lo, hi = min(err_end.min(), err_start.min()), max(err_end.max(), err_start.max())
    ticks = [t for t in (60, 80, 100, 150, 200, 300, 400, 600, 800) if lo * 0.85 <= t <= hi * 1.2]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_minor_locator(FixedLocator([]))
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlabel("distance from medoid (mm)")
    ax.set_ylabel("outline error (mm)")
    ax.legend(loc="upper left", fontsize=FS_ANNOT / scale, handletextpad=0.4,
              borderpad=0.35, labelspacing=0.25)
    ax.tick_params(length=2.4, pad=1.5)


def panel_determinacy(ax, tiles, node_xy, node_std, scale=1.0):
    """The medoid pattern, with every cut node sized and coloured by ensemble scatter."""
    ax.add_collection(PolyCollection(list(tiles), facecolors="#F7F7F8", edgecolors=INK,
                                     linewidths=0.3, zorder=1))
    cmap = _sequential_orange()
    norm = Normalize(vmin=float(node_std.min()), vmax=float(node_std.max()))
    ax.scatter(node_xy[:, 0], node_xy[:, 1], s=16, c=node_std, cmap=cmap, norm=norm,
               edgecolors=INK, linewidths=0.25, zorder=3)
    ax.set_aspect("equal")
    ax.axis("off")
    return cmap, norm


def panel_spectrum(ax, evals_shape, evals_total, pca_evals, n_flat, scale=1.0):
    """How many design directions the target constrains, against how many the ensemble uses.

    Curvature is normalised by its own largest value in each case: the objective is
    dimensionless and only the spread of a spectrum carries meaning, not its level.
    """
    def norm(v):
        v = np.sort(np.abs(np.asarray(v, dtype=float)))[::-1]
        return v / v.max()

    s, t, p = norm(evals_shape), norm(evals_total), norm(pca_evals)
    idx = np.arange(1, len(s) + 1)
    floor = 1e-17                       # a log axis cannot show an exact zero
    # Orange through red to black, and three line styles: the three series are different
    # KINDS of quantity, so they should not read as one family of curves.
    # Dark amber, not the charter ORANGE: that orange means "family A" in the companion
    # panel, and a curve here wearing it reads as a family.
    ax.plot(idx, np.maximum(s, floor), color="#A6611A", lw=1.2, label="shape")
    ax.plot(idx, np.maximum(t, floor), color="#B3241C", lw=1.0, ls=(0, (3.5, 2)),
            label="+ floors")
    ax.plot(np.arange(1, len(p) + 1), np.maximum(p, floor), color=INK, lw=1.0,
            ls=(0, (1.2, 1.4)), label="ensemble")
    ax.axvspan(len(s) - n_flat + 0.5, len(s) + 0.5, color=ORANGE, alpha=0.10, lw=0)
    ax.set_yscale("log")
    ax.set_ylim(1e-18, 3.0)
    ax.set_xlabel("design direction")
    ax.set_ylabel("curvature")
    # Bottom left is the only genuinely empty corner: every curve is above 1e-6 until the
    # shoulder at ~55, and the short keys end well before it.
    ax.legend(loc="lower left", fontsize=FS_ANNOT / scale, handletextpad=0.5,
              borderpad=0.3, labelspacing=0.25, framealpha=0.0, borderaxespad=0.25,
              handlelength=1.6)
    ax.tick_params(length=2.4, pad=1.5)


def build_figure(run_dir, out_path):
    from nff.closed.design_ensemble import (build_symmetry, canonical_distance,
                                            circle_problem, distance_matrix, load_ensemble,
                                            problem_statics)
    from nff.closed.analytic_targets import circle_outline_unit
    from nff.closed.shield_design import build_loss, _interior_cut_bases
    import jax.numpy as jnp

    records = load_ensemble(run_dir)
    with open(os.path.join(run_dir, "summary.json")) as fh:
        summary = json.load(fh)
    land = dict(np.load(os.path.join(run_dir, "landscape.npz")))
    meta = summary["config"]

    cfg = circle_problem(M=meta["M"], N=meta["N"], sheet_mm=meta["sheet_mm"],
                         radius_mm=meta["radius_mm"])
    statics = problem_statics(cfg)
    group = build_symmetry(cfg, statics)
    loss_fn = build_loss(statics["forward_fn"], circle_outline_unit(480), statics["plan"],
                         cfg, _interior_cut_bases(statics["struct"]))

    ends = [r["coords"] for r in records]
    starts = [r["coords_start"] for r in records]
    medoid = int(land["medoid_index"])
    d_end = np.array([canonical_distance(ends[medoid], c, group) for c in ends])
    d_start = np.array([canonical_distance(ends[medoid], c, group) for c in starts])
    err_end = np.array([float(r["shape_error_mm"]) for r in records])

    # The starting designs are scored the same way, by evaluating the objective on them --
    # a start with no score could not be put on the same axes.
    err_start = []
    for r in records:
        p = {k[len("start_params__"):]: jnp.asarray(v)
             for k, v in r.items() if k.startswith("start_params__")}
        _, m = loss_fn(p, 1.0)
        err_start.append(max(float(m["precision_mm"]), float(m["recall_mm"])))
    err_start = np.array(err_start)

    scale = 1.0
    w_bp, h_bp = page_bp(1.0, aspect=2.45)
    apply_charter(**paper_rc(scale))
    fig = plt.figure(figsize=figsize_in(w_bp, h_bp))

    ax_a = fig.add_axes([0.088, 0.235, 0.235, 0.660])
    ax_b = fig.add_axes([0.378, 0.190, 0.225, 0.780])
    ax_c = fig.add_axes([0.723, 0.235, 0.252, 0.660])

    panel_scatter(ax_a, d_end, err_end, d_start, err_start, land["labels"], scale)

    tiles = land["medoid_tiles"]
    node_xy = land["medoid_coords"].reshape(-1, 2, 2).mean(axis=1)
    cmap, norm = panel_determinacy(ax_b, tiles, node_xy, land["node_std_grid"], scale)
    cax = fig.add_axes([0.396, 0.115, 0.190, 0.019])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                      orientation="horizontal")
    cb.set_label("cut-node scatter across the ensemble (mm)", fontsize=FS_ANNOT / scale,
                 labelpad=1.0)
    cb.ax.tick_params(labelsize=FS_ANNOT / scale, length=2, pad=1.0)
    cb.outline.set_linewidth(0.5)

    panel_spectrum(ax_c, land["evals_shape"], land["evals_total"], land["pca_evals"],
                   int(land["n_flat"]), scale)

    for ax, letter, x in ((ax_a, "a", -0.215), (ax_b, "b", -0.02), (ax_c, "c", -0.225)):
        panel_letter(ax, letter, x=x, y=1.03, scale=scale)

    pdf, lo, hi = save_paper_figure(fig, out_path, w_bp, h_bp, dpi=600, scale=scale)
    plt.close(fig)
    return pdf, {"n_flat": int(land["n_flat"]),
                 "rho": float(land["var_curv_logcorr"]),
                 "type_pt": [lo, hi]}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(args.run, "design_degeneracy.pdf")
    pdf, info = build_figure(args.run, out)
    print(f"  wrote {pdf}")
    for k, v in info.items():
        print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
