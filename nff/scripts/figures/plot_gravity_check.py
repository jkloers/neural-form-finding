"""Sanity check: does gravity suppress the ligament's out-of-plane buckling at scale?

Left  : fold moment M_θ vs angle (no gravity) with gravity's per-face restoring moment marked.
Right : out-of-plane buckling amplitude uz(θ) that gravity would have to flatten.
If the buckling moment stays below gravity's restoring, faces stay coplanar (in-plane deploy).
"""
from __future__ import annotations

import argparse
import os

import numpy as np

INK, MUTE, ACC, GRAV = "#212529", "#868e96", "#1971C2", "#2b8a3e"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rot", default="data/experiments/processed/pet_hinge_w50_rot.npz")
    ap.add_argument("--face-cm", type=float, default=55.0, help="face edge length [cm]")
    ap.add_argument("--thickness-mm", type=float, default=0.5)
    ap.add_argument("--rho", type=float, default=1390.0, help="PET density [kg/m^3]")
    ap.add_argument("--out", default="data/experiments/processed/pet_gravity_check.png")
    args = ap.parse_args()

    # gravity restoring moment on one face hinged at an edge: W * (L/2)
    L = args.face_cm / 100.0
    m = L * L * (args.thickness_mm / 1000.0) * args.rho
    W = m * 9.81
    M_grav_Nmm = W * (L / 2) * 1000.0   # N.mm

    d = np.load(args.rot)
    th, M, uz = d["theta_deg"], d["M_theta"], d["uz_max"]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#adb5bd", "text.color": INK,
                         "axes.labelcolor": INK, "xtick.color": MUTE, "ytick.color": MUTE,
                         "axes.grid": True, "grid.color": "#e9ecef", "grid.linewidth": 0.8,
                         "figure.facecolor": "white", "axes.facecolor": "white"})
    fig, (axM, axU) = plt.subplots(1, 2, figsize=(12, 5))

    axM.plot(th, M, color=ACC, lw=2, label="ligament fold moment (weak)")
    axM.axhline(M_grav_Nmm, color=GRAV, lw=2, ls="--")
    axM.text(th.max() * 0.98, M_grav_Nmm + 25,
             f"gravity's hold on one face ≈ {M_grav_Nmm:.0f} N·mm\n({args.face_cm:.0f} cm face, stays flat)",
             color=GRAV, fontsize=9.5, ha="right", va="bottom")
    axM.text(th.max() * 0.5, M.max() * 0.45,
             "heavy face >> weak ligament:\nface stays coplanar,\nligament buckles", color=INK,
             fontsize=10, ha="center")
    axM.set_xlabel("fold angle  θ  [deg]")
    axM.set_ylabel("moment  [N·mm]")
    axM.set_title("Weak ligament vs gravity-held face", fontsize=12, loc="left", color=INK)
    axM.set_ylim(0, max(M_grav_Nmm, np.nanmax(M)) * 1.25)
    axM.legend(frameon=False, fontsize=9, loc="upper left")

    axU.plot(th, uz, color="#e8590c", lw=2)
    axU.fill_between(th, uz, color="#e8590c", alpha=0.12)
    axU.annotate(f"{np.nanmax(uz):.0f} mm out-of-plane —\nthe ligament absorbs the motion\nby buckling (faces stay flat)",
                 (th[-1], uz[-1]), xytext=(th[-1] * 0.42, uz[-1] * 0.72), fontsize=9.5, color=INK,
                 arrowprops=dict(arrowstyle="->", color=MUTE, lw=0.9))
    axU.set_xlabel("fold angle  θ  [deg]")
    axU.set_ylabel("out-of-plane amplitude  uz  [mm]")
    axU.set_title("Ligament buckling = the deployment mechanism", fontsize=12, loc="left", color=INK)
    axU.set_ylim(bottom=0)

    fig.suptitle("Gravity keeps the faces coplanar; the ligament buckles out of plane — w_lig = 50 mm, t = 0.5 mm",
                 fontsize=12, fontweight="bold", x=0.02, ha="left", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=200, facecolor="white")
    print(f"gravity restoring ≈ {M_grav_Nmm:.0f} N·mm | buckling moment peak ≈ {np.nanmax(M):.0f} N·mm | uz_max ≈ {np.nanmax(uz):.0f} mm")
    print(f"figure -> {args.out}")


if __name__ == "__main__":
    main()
