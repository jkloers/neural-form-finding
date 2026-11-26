"""The force / target-shape trade for the single-tile diagonal grip.

Two panels, because the sweep answers two questions that must not be conflated:

  left   SHAPE vs TRUST -- boundary error against joint in-box fraction. The 95% line is the
         constraint the old calibration imposed by capping the force; an arm to its right is one
         the compression penalty rescued rather than one the force avoided.
  right  WHAT THE FORCE BOUGHT -- deployment angle and worst axial strain against force, with the
         trust-box floor drawn. Deployment is the result; staying above the floor is the control.

    PYTHONPATH=$(pwd) python legacy/experiments/figures/plot_grip_c_pareto.py \
        --sweep data/outputs/sweep_grip_c/stage_A.json \
        --out data/outputs/sweep_grip_c/pareto.png
"""

import argparse
import json

import numpy as np

from nff.utils.figstyle import use_agg, apply_charter, ORANGE, TEAL, RED, GREY, INK

use_agg()
import matplotlib.pyplot as plt          # noqa: E402

ETA_A_FLOOR = -0.140                     # hinge_surrogate_pet_v3 trust-box floor
IN_BOX_TARGET = 0.95                     # the pet_1t family's calibration constraint


def load_rows(paths):
    rows = []
    for p in paths:
        with open(p) as fh:
            data = json.load(fh)
        rows.extend(data if isinstance(data, list) else [data])
    return [r for r in rows if r and not r.get("error")]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", nargs="+", required=True, help="stage_*.json from sweep_grip_c.py")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = load_rows(args.sweep)
    if not rows:
        raise SystemExit("no usable arms in the given sweep files")

    apply_charter()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.4))

    force = np.array([r.get("force_N", np.nan) for r in rows], float)
    # The MULTIPLIER, not the absolute weight: parity differs per arm, so only the multiple is
    # comparable across a ladder -- and it is what "x parity" on the axis actually means.
    w_mult = np.array([r.get("compression_x_parity", np.nan) for r in rows], float)
    # A weight ladder holds the force fixed and vice versa; plot against whichever one is the
    # variable, or the figure's x axis is a constant and says nothing.
    by_weight = np.nanstd(force) < 1e-9 and np.nanstd(w_mult) > 0
    xs, xlabel = ((w_mult, "compression weight  [$\\times$ gradient parity]") if by_weight
                  else (force, "applied force  [N]"))
    # Arms carrying damage are a different experiment; they must not join the compression line.
    dmg = np.array([r.get("damage_x_parity", 0.0) or 0.0 for r in rows], float)
    inbox = np.array([r.get("in_box_joint", np.nan) for r in rows], float)
    bnd = np.array([r.get("boundary_mean_mm", np.nan) for r in rows], float)
    th = np.array([r.get("theta_max", np.nan) for r in rows], float)
    eta = np.array([r.get("min_eta_a", np.nan) for r in rows], float)
    # Headroom, not raw pull-in: the interesting axis is how much the target asks the design to
    # contract, and a marker keyed to it separates the shape arms from the force arms.
    pull = np.array([r.get("pull_in_x", np.nan) for r in rows], float)

    # ── left: shape fidelity against trust-box coverage ──
    ax1.axvline(IN_BOX_TARGET * 100, color=RED, lw=1.0, ls="--", zorder=1)
    # Axes-fraction y: reading get_ylim() here would read the DEFAULT limits, since nothing is
    # plotted yet -- the label then lands off the eventual data range and stretches the canvas.
    ax1.text(IN_BOX_TARGET * 100, 0.98, " 95% constraint", color=RED, fontsize=8,
             va="top", ha="left", rotation=90, transform=ax1.get_xaxis_transform())
    shapes = {}
    for p in sorted(set(pull[np.isfinite(pull)])):
        shapes[p] = "o" if abs(p - 0.953) < 1e-6 else ("s" if p >= 0.999 else "^")
    sc = None
    for i, r in enumerate(rows):
        if not (np.isfinite(inbox[i]) and np.isfinite(bnd[i])):
            continue
        sc = ax1.scatter(inbox[i] * 100, bnd[i], s=95, c=[xs[i]], cmap="viridis",
                         marker=shapes.get(pull[i], "o"), edgecolor=INK, linewidth=0.7,
                         zorder=3, vmin=np.nanmin(xs), vmax=np.nanmax(xs))
        ax1.annotate(r["name"].replace("gripC_", ""), (inbox[i] * 100, bnd[i]),
                     textcoords="offset points", xytext=(7, 4), fontsize=7.5, color=GREY)
    ax1.set_xlabel("joint in-box fraction of ridden hinge-states  [%]")
    ax1.set_ylabel("boundary error vs target, mean  [mm]")
    if sc is not None:
        fig.colorbar(sc, ax=ax1, label=xlabel, pad=0.02)

    # ── right: what the weight bought, against what it cost ──
    pure = np.where(dmg <= 0)[0]                      # compression-only arms define the trend
    o = pure[np.argsort(xs[pure])] if pure.size else np.argsort(xs)
    ax2.plot(xs[o], th[o], "-o", color=TEAL, lw=1.6, ms=6, mec=INK, mew=0.6,
             label="deployment angle, max")
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel("hinge angle, max  [deg]", color=TEAL)
    ax2.tick_params(axis="y", labelcolor=TEAL)

    ax3 = ax2.twinx()
    ax3.grid(False)
    ax3.plot(xs[o], eta[o], "-s", color=ORANGE, lw=1.6, ms=6, mec=INK, mew=0.6,
             label="worst axial strain")
    for i in np.where(dmg > 0)[0]:                    # damage arms: shown, but off the line
        ax2.plot(xs[i], th[i], "D", color=TEAL, ms=7, mec=INK, mew=0.8, ls="none")
        ax3.plot(xs[i], eta[i], "D", color=ORANGE, ms=7, mec=INK, mew=0.8, ls="none",
                 label="+ damage at parity")
    ax3.axhline(ETA_A_FLOOR, color=RED, lw=1.0, ls="--")
    ax3.text(xs[o][0], ETA_A_FLOOR, " trust-box floor", color=RED, fontsize=8, va="bottom")
    ax3.set_ylabel("worst axial strain  min $\\eta_a$", color=ORANGE)
    ax3.tick_params(axis="y", labelcolor=ORANGE)
    ax3.spines["right"].set_visible(True)
    ax3.spines["right"].set_color(INK)

    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax3.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"  wrote {args.out}  ({len(rows)} arms)")


if __name__ == "__main__":
    main()
