"""w_lig = 18 mm: CalculiX vs real experiment (r_win = 100 mm domain).

Panel A  fold: out-of-plane buckle uz vs rotation theta  (sim seeds imp_amp vs real bh)
Panel B  shear: force vs displacement                    (sim F_s(s) vs real crosshead)

NOTE on panel B x-axis: sim s = ligament slip; real = crosshead displacement (machine
compliance + tile stretch). Forces are directly comparable, displacement axes are not.
Sim from data/experiments/processed/pet_w18_{shear_rwin100,rot_imp*}.npz;
real from real_w18_{fold_bh,shear}.csv.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

REAL, FAIL, INK = "#E8590C", "#D62828", "#212529"
IMPC = {"0.5": "#6baed6", "1.5": "#08519c"}  # fold seed colors


def _sorted(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    o = np.argsort(x, kind="stable")
    return x[o], y[o]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="data/experiments/processed")
    ap.add_argument("--out", default="data/experiments/processed/pet_w18_calibration.png")
    args = ap.parse_args()
    D = args.dir

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#adb5bd", "text.color": INK,
                         "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})
    fig, (axR, axS) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    # ---- Panel A: fold uz vs theta ----
    for imp in ("0.5", "1.5"):
        f = os.path.join(D, f"pet_w18_rot_imp{imp}.npz")
        if not os.path.exists(f):
            continue
        d = np.load(f)
        th, uz = _sorted(d["theta_deg"], d["uz_max"])
        if len(th) == 0:
            continue
        uz = uz - uz[0]                       # baseline-correct like the real bh
        axR.plot(th, uz, color=IMPC[imp], lw=2.2, label=f"CalculiX  imp={imp} mm")
    rb = os.path.join(D, "real_w18_fold_bh.csv")
    if os.path.exists(rb):
        d = np.loadtxt(rb, delimiter=",", skiprows=1)
        axR.plot(d[:, 0], d[:, 1], "o", color=REAL, ms=6, label="experiment  bh", zorder=5)
    axR.set_xlabel("rotation  θ  [deg]")
    axR.set_ylabel("out-of-plane buckle  uz  [mm]")
    axR.set_title("Fold: buckle height", fontsize=11, loc="left")
    axR.legend(frameon=False, fontsize=9)

    # ---- Panel B: opening (tension) force vs displacement ----
    fs = os.path.join(D, "pet_w18_open_rwin25.npz")
    if os.path.exists(fs):
        d = np.load(fs)
        a, F = _sorted(d["a"], np.abs(d["F_a"]))
        if len(a):
            axS.plot(a, F, color="#08519c", lw=2.2, label="CalculiX  F_a(opening)")
            ad, Dm = _sorted(d["a"], d["damage_p99"])
            fr = np.where(Dm >= 1.0)[0]
            if len(fr):
                axS.plot(ad[fr[0]], np.interp(ad[fr[0]], a, F), "x", color=FAIL, ms=10, mew=2.6,
                         label="predicted tear")
    rs = os.path.join(D, "real_w18_shear.csv")
    if os.path.exists(rs):
        d = np.loadtxt(rs, delimiter=",", skiprows=1)
        axS.plot(d[:, 0], d[:, 1], "-", color=REAL, lw=2.0, label="experiment (crosshead)")
    axS.set_xlabel("opening displacement  [mm]   (sim: ligament · real: crosshead)")
    axS.set_ylabel("opening force  F_a  [N]")
    axS.set_title("Tension: opening curve", fontsize=11, loc="left")
    axS.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(args.out, dpi=160, bbox_inches="tight")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
