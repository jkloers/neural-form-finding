"""w=18mm fold: buckle height uz(theta) — experiment vs CalculiX vs fitted closed form.

Experiment bh(theta) to 90 deg; CalculiX (stabilized) to the ~57 deg localization wall;
fitted laws  uz = A*sqrt(sin th)  (physical: post-buckling sqrt-onset + geometric saturation)
and  uz = uz_max*(1-exp(-th/th_c))  (best empirical).
"""
from __future__ import annotations

import argparse
import os

import numpy as np

REAL, SIM, FIT, FIT2, INK = "#E8590C", "#08519c", "#212529", "#868e96", "#212529"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="data/experiments/processed")
    ap.add_argument("--out", default="data/experiments/processed/pet_w18_fold_formula.png")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#adb5bd", "text.color": INK,
                         "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})

    th_e = np.array([0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90], float)
    bh_e = np.array([0, 7.62, 11.04, 18.39, 19.13, 22.12, 22.39, 23.04, 23.38, 23.92, 25.48], float)

    fig, ax = plt.subplots(figsize=(6.6, 4.8))

    # CalculiX (stabilized, to ~57 deg)
    f = os.path.join(args.dir, "pet_w18_fold90_st3e-4.npz")
    if os.path.exists(f):
        d = np.load(f)
        th = np.asarray(d["theta_deg"], float); uz = np.asarray(d["uz_max"], float)
        m = np.isfinite(th) & np.isfinite(uz); th, uz = th[m], uz[m]
        o = np.argsort(th); th, uz = th[o], uz[o]
        ax.plot(th, uz, color=SIM, lw=2.2, label="CalculiX (to localization wall)")

    # fitted laws over full range
    tt = np.linspace(0, 90, 200)
    ax.plot(tt, 25.67 * np.sqrt(np.sin(np.radians(tt))), "--", color=FIT, lw=1.8,
            label=r"fit  $u_z=25.7\,\sqrt{\sin\theta}$  (R²=0.97)")
    ax.plot(tt, 24.0 * (1 - np.exp(-tt / 15.7)), ":", color=FIT2, lw=1.8,
            label=r"fit  $u_z=24.0\,(1-e^{-\theta/15.7^\circ})$  (R²=0.99)")

    # experiment
    ax.plot(th_e, bh_e, "o", color=REAL, ms=7, label="experiment  bh", zorder=5)

    ax.set_xlabel("fold angle  θ  [deg]")
    ax.set_ylabel("out-of-plane buckle height  uz  [mm]")
    ax.set_xlim(0, 92); ax.set_ylim(0, 28)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(args.out, dpi=160, bbox_inches="tight")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
