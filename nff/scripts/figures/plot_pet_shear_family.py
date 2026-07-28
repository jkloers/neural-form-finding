"""Shear prediction family across ligament width — the Experiment-1 target + scale check.

F_s vs s for w_lig = 5/10/15/25 mm (the sim-reliable range), with the chosen Experiment-1
design (w=15 mm) highlighted and predicted fracture marked. Sequential color (ordered w_lig).
"""
from __future__ import annotations

import argparse
import os

import numpy as np

RAMP = {5: "#bdd7e7", 10: "#6baed6", 15: "#E8590C", 25: "#08519c"}  # w=15 = accent (the pick)
INK, MUTE, FAIL = "#212529", "#868e96", "#D62828"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="data/experiments/processed")
    ap.add_argument("--pick", type=int, default=15)
    ap.add_argument("--out", default="data/experiments/processed/pet_shear_family.png")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#adb5bd", "text.color": INK, "axes.labelcolor": INK,
                         "xtick.color": MUTE, "ytick.color": MUTE, "axes.grid": True,
                         "grid.color": "#e9ecef", "grid.linewidth": 0.8,
                         "figure.facecolor": "white", "axes.facecolor": "white"})
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    for w in (5, 10, 15, 25):
        p = os.path.join(args.dir, f"pet_hinge_w{w}_shear.npz")
        if not os.path.exists(p):
            continue
        d = np.load(p)
        s, Fs = d["s"], np.abs(d["F_s"])
        is_pick = (w == args.pick)
        ax.plot(s, Fs, color=RAMP[w], lw=3 if is_pick else 1.8,
                label=f"w_lig = {w} mm" + ("   (Experiment 1)" if is_pick else ""),
                zorder=4 if is_pick else 2)
        if float(np.nanmax(d["peeq_p99"])) >= 1.1 * float(d["eps_f"]):
            ax.scatter([s[-1]], [Fs[-1]], marker="x", s=75, color=FAIL, lw=2.2, zorder=5)

    ax.scatter([], [], marker="x", s=75, color=FAIL, lw=2.2, label="predicted fracture (s ≈ w_lig)")
    ax.set_xlabel("shear displacement  s  [mm]")
    ax.set_ylabel("shear force  F$_s$  [N]")
    ax.set_title("Hinge shear prediction — pick w_lig = 15 mm for Experiment 1",
                 fontsize=13, loc="left", pad=10, color=INK)
    ax.text(0.015, 0.97, "force scales ~linearly with w_lig · fractures at s ≈ w_lig\n"
            "w=15 mm gives ~128 N peak (precise on the 5 kN cell), sim-reliable",
            transform=ax.transAxes, fontsize=9, color=MUTE, va="top")
    ax.legend(frameon=False, fontsize=9.5, loc="lower right")
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    fig.suptitle("PET single-hinge shear — Experiment 1 prediction (t = 0.5 mm, α = 90°)",
                 fontsize=12, fontweight="bold", x=0.02, ha="left", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=200, facecolor="white")
    print(f"figure -> {args.out}")


if __name__ == "__main__":
    main()
