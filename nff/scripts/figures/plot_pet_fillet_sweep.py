"""Does fillet ratio move the fold buckle-height curve? (w18, alpha90)

Left:  uz(theta) family over fillet_ratio + the real experiment points.
Right: fitted prefactor A of  uz = A*sqrt(sin theta)  vs fillet_ratio.
Reads data/experiments/processed/pet_w18_fold_fr{FR}.npz.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

FRS = [0.08, 0.16, 0.24, 0.32]
RAMP = {0.08: "#bdd7e7", 0.16: "#6baed6", 0.24: "#3182bd", 0.32: "#08306b"}
REAL, INK = "#E8590C", "#212529"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="data/experiments/processed")
    ap.add_argument("--out", default="data/experiments/processed/pet_w18_fillet_sweep.png")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#adb5bd", "text.color": INK,
                         "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})

    th_e = np.array([0, 5, 10, 20, 30, 40, 50], float)
    bh_e = np.array([0, 7.62, 11.04, 18.39, 19.13, 22.12, 22.39], float)

    fig, (axC, axA) = plt.subplots(1, 2, figsize=(11, 4.4))
    A_of_fr = []
    for fr in FRS:
        f = os.path.join(args.dir, f"pet_w18_fold_fr{fr}.npz")
        if not os.path.exists(f):
            A_of_fr.append(np.nan); continue
        d = np.load(f)
        th = np.asarray(d["theta_deg"], float); uz = np.asarray(d["uz_max"], float)
        m = np.isfinite(th) & np.isfinite(uz); th, uz = th[m], uz[m]
        o = np.argsort(th); th, uz = th[o], uz[o]
        axC.plot(th, uz, color=RAMP[fr], lw=2.0, label=f"fillet {fr}  (ρ={fr*18:.1f}mm)")
        g = np.sqrt(np.sin(np.radians(th)))
        A = np.sum(uz * g) / np.sum(g * g) if np.sum(g * g) > 0 else np.nan
        A_of_fr.append(A)
    axC.plot(th_e, bh_e, "o", color=REAL, ms=6, label="experiment (fr=0.16)", zorder=5)
    axC.set_xlabel("fold angle  θ  [deg]")
    axC.set_ylabel("buckle height  uz  [mm]")
    axC.set_title("Fold buckle height vs fillet ratio", fontsize=11, loc="left")
    axC.legend(frameon=False, fontsize=8.5)

    axA.plot(FRS, A_of_fr, "o-", color="#08519c", lw=2, ms=7)
    for fr, A in zip(FRS, A_of_fr):
        if np.isfinite(A):
            axA.annotate(f"{A:.1f}", (fr, A), textcoords="offset points", xytext=(0, 8), fontsize=9)
    axA.set_xlabel("fillet ratio  ρ / w_lig")
    axA.set_ylabel("prefactor A  [mm]   (uz = A·√sin θ)")
    axA.set_title("Does the prefactor depend on fillet?", fontsize=11, loc="left")
    axA.set_ylim(0, max([a for a in A_of_fr if np.isfinite(a)] + [1]) * 1.25)

    fig.tight_layout()
    fig.savefig(args.out, dpi=160, bbox_inches="tight")
    finite = [f"{fr}:{A:.2f}" for fr, A in zip(FRS, A_of_fr) if np.isfinite(A)]
    print("A(fillet):", finite)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
