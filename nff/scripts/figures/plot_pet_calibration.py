"""Two-panel figure of the PET cold-draw tensile calibration.

Left  : the stress-strain family (9 draws + 1 break), CD vs MD, with the yield and
        draw-plateau levels marked.
Right : the calibrated true-stress *PLASTIC law fed to CalculiX, anchored on the
        measured yield and the cold-draw point (true stress = lambda * plateau).

Palette validated with the dataviz skill (orange/blue, CVD-safe); the broken specimen
uses linestyle, not a third hue. Clean/minimal per the project viz policy: no values in
titles, recessive grid, thin marks, direct labels.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from nff.calibration import bluehill_io, summary_io

CD, MD, BROKE, INK, MUTE = "#E8590C", "#1971C2", "#495057", "#212529", "#868e96"


def _style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#adb5bd", "axes.labelcolor": INK,
        "text.color": INK, "xtick.color": MUTE, "ytick.color": MUTE,
        "axes.grid": True, "grid.color": "#e9ecef", "grid.linewidth": 0.8,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="data/experiments/raw/kirigami_20260723")
    ap.add_argument("--summary", default="data/experiments/raw/kirigami_20260723/"
                    "Kirigami experiments - Tensile test (safe).csv")
    ap.add_argument("--out", default="data/experiments/processed/pet_calibration.png")
    # calibrated law (from analyze_kirigami_set)
    ap.add_argument("--yield-mpa", type=float, default=48.8)
    ap.add_argument("--plateau-mpa", type=float, default=33.1)
    ap.add_argument("--sig-true", type=float, default=109.0)
    ap.add_argument("--eps-true", type=float, default=1.19)
    ap.add_argument("--E", type=float, default=3000.0)
    args = ap.parse_args()

    _style()
    import matplotlib.pyplot as plt

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.5, 5.2), gridspec_kw={"width_ratios": [1.25, 1]})

    # ---------- Panel A: stress-strain family ----------------------------
    for m in summary_io.load_summary(args.summary):
        if not m.included:
            continue
        fn = os.path.join(args.dir, m.name.replace(".", "_") + ".csv")
        if not os.path.exists(fn):
            continue
        run = bluehill_io.load_raw_csv(fn)
        i0 = int(np.argmax(run.force_N > 5))
        eps = (run.disp_mm - run.disp_mm[i0]) / m.L_mm * 100
        sig = run.force_N / m.a_mean_mm2
        if m.lam is None:  # the broken specimen
            axA.plot(eps, sig, color=BROKE, lw=1.4, ls=(0, (4, 2)), zorder=5)
            k = int(np.argmax(sig))
            axA.annotate("4.2 — brittle break\n(edge flaw)", (eps[k], sig[k] * 0.6),
                         xytext=(9.5, 46), fontsize=9, color=BROKE, ha="left",
                         arrowprops=dict(arrowstyle="->", color=BROKE, lw=0.8))
        else:
            axA.plot(eps, sig, color=CD if m.direction == "CD" else MD, lw=1.5, alpha=0.85)

    axA.axhline(args.yield_mpa, color=MUTE, lw=0.9, ls=":", zorder=1)
    axA.axhline(args.plateau_mpa, color=MUTE, lw=0.9, ls=":", zorder=1)
    axA.text(19.3, args.yield_mpa + 0.8, "yield  ≈ 49 MPa", fontsize=9, color=MUTE, ha="right")
    axA.text(0.2, args.plateau_mpa - 3.2, "draw plateau  ≈ 33 MPa", fontsize=9, color=MUTE, ha="left")
    axA.annotate("cold drawing\n(neck propagates)", (13.5, args.plateau_mpa),
                 xytext=(9.5, 17), fontsize=9, color=INK,
                 arrowprops=dict(arrowstyle="->", color=MUTE, lw=0.9))
    from matplotlib.lines import Line2D
    axA.legend([Line2D([], [], color=CD, lw=2), Line2D([], [], color=MD, lw=2),
                Line2D([], [], color=BROKE, lw=1.4, ls=(0, (4, 2)))],
               ["CD  (n=3)", "MD  (n=6)", "broke (1)"], frameon=False, loc="lower right", fontsize=9)
    axA.set_xlabel("engineering strain  [%]")
    axA.set_ylabel("engineering stress  [MPa]")
    axA.set_title("Cold-drawing tensile response", fontsize=12, loc="left", pad=10, color=INK)
    axA.set_xlim(-0.5, 20.5)
    axA.set_ylim(0, 56)
    axA.grid(axis="x", visible=False)

    # ---------- Panel B: calibrated *PLASTIC law -------------------------
    ep_draw = args.eps_true - args.sig_true / args.E
    xs = [0.0, ep_draw]
    ys = [args.yield_mpa, args.sig_true]
    axB.plot(xs, ys, color=INK, lw=2, zorder=3)
    axB.scatter(xs, ys, s=70, color=[CD, MD], zorder=4, edgecolor="white", linewidth=1.5)

    axB.annotate("yield\n49 MPa", (0, args.yield_mpa), xytext=(0.12, 40),
                 fontsize=9.5, color=INK, ha="left")
    axB.annotate(f"cold-draw anchor\n{args.sig_true:.0f} MPa @ ε≈{args.eps_true:.1f}",
                 (ep_draw, args.sig_true), xytext=(ep_draw - 0.05, args.sig_true - 26),
                 fontsize=9.5, color=INK, ha="right",
                 arrowprops=dict(arrowstyle="->", color=MUTE, lw=0.9))
    # engineering plateau -> true, via lambda
    axB.axhline(args.plateau_mpa, color=MUTE, lw=0.8, ls=":", zorder=1)
    axB.text(0.02, args.plateau_mpa + 1.5, "eng. plateau 33 MPa", fontsize=8.5, color=MUTE)
    axB.annotate("", (ep_draw, args.sig_true - 4), (ep_draw, args.plateau_mpa + 2),
                 arrowprops=dict(arrowstyle="->", color=MD, lw=1.3))
    axB.text(ep_draw + 0.03, (args.sig_true + args.plateau_mpa) / 2,
             "× λ≈3.3", fontsize=10, color=MD, va="center")
    axB.text(0.98, 0.06, "E = 3.0 GPa (lit.)   ν = 0.40   $\\varepsilon_f \\geq$ 1.2",
             transform=axB.transAxes, fontsize=8.5, color=MUTE, ha="right")
    axB.set_xlabel("true plastic strain  [–]")
    axB.set_ylabel("true stress  [MPa]")
    axB.set_title("Calibrated plastic law for CalculiX", fontsize=12, loc="left", pad=10, color=INK)
    axB.set_xlim(-0.05, 1.35)
    axB.set_ylim(0, 125)

    fig.suptitle("PET sheet — cold-drawing tensile calibration",
                 fontsize=13.5, fontweight="bold", x=0.02, ha="left", color=INK)
    fig.text(0.02, 0.925, "10 strips (MD + CD), 1 mm/min · yield isotropic, drawability CD > MD",
             fontsize=9.5, color=MUTE, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=200, facecolor="white")
    print(f"figure -> {args.out}")


if __name__ == "__main__":
    main()
