"""Plot PET hinge RVE trials (rotation moment-angle + shear force-displacement).

Reads the .npz written by run_pet_hinge_trial.py and plots the predictions to line up
against the physical hinge experiments. Palette validated with the dataviz skill; clean/
minimal per the project viz policy (no values in titles, thin marks, direct labels).

    python -m nff.scripts.figures.plot_pet_hinge_trial \
        --rotation data/experiments/processed/pet_hinge_w5_fold90.npz \
        --shear    data/experiments/processed/pet_hinge_w5_shear.npz \
        --out data/experiments/processed/pet_hinge_trial.png
"""
from __future__ import annotations

import argparse

import numpy as np

ROT, SHR, INK, MUTE, FAIL = "#1971C2", "#E8590C", "#212529", "#868e96", "#D62828"


def _style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#adb5bd",
        "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTE, "ytick.color": MUTE,
        "axes.grid": True, "grid.color": "#e9ecef", "grid.linewidth": 0.8,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })


def _yield_idx(regime):
    yielded = np.where(np.asarray(regime) >= 1)[0]
    return int(yielded[0]) if yielded.size else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rotation")
    ap.add_argument("--shear")
    ap.add_argument("--out", default="data/experiments/processed/pet_hinge_trial.png")
    args = ap.parse_args()

    _style()
    import matplotlib.pyplot as plt

    fig, (axR, axS) = plt.subplots(1, 2, figsize=(12, 5))

    if args.rotation:
        d = np.load(args.rotation)
        th, M = d["theta_deg"], d["M_theta"]
        axR.axhline(0, color="#ced4da", lw=0.9, zorder=1)
        axR.plot(th, M, color=ROT, lw=2, zorder=3)
        kpk = int(np.nanargmax(M))
        axR.scatter([th[kpk]], [M[kpk]], s=55, color=ROT, zorder=4, edgecolor="white", lw=1.4)
        axR.annotate(f"pre-buckling peak\n{M[kpk]:.0f} N·mm @ {th[kpk]:.0f}°",
                     (th[kpk], M[kpk]), xytext=(13, -55),
                     fontsize=9, color=INK, arrowprops=dict(arrowstyle="->", color=MUTE, lw=0.9))
        axR.annotate("buckling snap-through\n(moment reverses)", (th[-1], M[-1]),
                     xytext=(8, M[-1] + 60), fontsize=9, color=INK,
                     arrowprops=dict(arrowstyle="->", color=MUTE, lw=0.9))
        axR.set_xlabel("fold angle  θ  [deg]")
        axR.set_ylabel("moment  M$_θ$  [N·mm]")
        axR.set_title("Rotation (buckling-dominated — imperfection-sensitive)",
                      fontsize=11.5, loc="left", color=INK)

    if args.shear:
        d = np.load(args.shear)
        s, Fs, reg = d["s"], d["F_s"], d["regime"]
        axS.plot(s, np.abs(Fs), color=SHR, lw=2)
        yi = _yield_idx(reg)
        if yi is not None:
            axS.scatter([s[yi]], [abs(Fs[yi])], s=55, color=SHR, zorder=4, edgecolor="white", lw=1.4)
            axS.annotate("yield", (s[yi], abs(Fs[yi])), xytext=(s[yi] + 0.2, abs(Fs[yi]) - 12),
                         fontsize=9, color=INK)
        ft = float(d["failure_theta_deg"])
        axS.annotate("robust, monotonic —\nthe calibration target", (s[-1], abs(Fs[-1])),
                     xytext=(1.3, 22), fontsize=9, color=INK,
                     arrowprops=dict(arrowstyle="->", color=MUTE, lw=0.9))
        axS.set_xlabel("shear displacement  s  [mm]")
        axS.set_ylabel("shear force  F$_s$  [N]")
        axS.set_title("Shear (Instron validation — robust)", fontsize=12, loc="left", color=INK)
        axS.set_ylim(bottom=0)

    fig.suptitle("PET hinge — CalculiX prediction (w_lig = 5 mm, t = 0.5 mm)",
                 fontsize=13, fontweight="bold", x=0.02, ha="left", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=200, facecolor="white")
    print(f"figure -> {args.out}")


if __name__ == "__main__":
    main()
