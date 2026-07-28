"""Loading-direction (φ) sweep of the PET hinge — the Tier-2 failure-locus prediction.

Each φ is a fixed-orientation Instron pull (φ=0 shear ∥ cut → φ=90 opening ⊥ cut). Plots the
projected force along the pull vs displacement along the pull, one curve per φ, with an × at
predicted fracture. φ is an ordered variable → sequential (single-hue) color, per the dataviz skill.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

# sequential blue ramp, light (shear) -> dark (opening); monotonic lightness
RAMP = {0: "#9ecae1", 30: "#4292c6", 60: "#2171b5", 90: "#08519c"}
INK, MUTE, FAIL = "#212529", "#868e96", "#D62828"


def _style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#adb5bd",
        "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTE, "ytick.color": MUTE,
        "axes.grid": True, "grid.color": "#e9ecef", "grid.linewidth": 0.8,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })


def _projected(d, phi_deg):
    """Force & displacement along the pull direction (sinφ, cosφ) in (a, s)."""
    ea, es = np.sin(np.radians(phi_deg)), np.cos(np.radians(phi_deg))
    a, s, Fa, Fs = d["a"], d["s"], d["F_a"], d["F_s"]
    disp = a * ea + s * es
    force = np.abs(Fa * ea + Fs * es)
    return disp, force


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="data/experiments/processed")
    ap.add_argument("--out", default="data/experiments/processed/pet_hinge_phi_sweep.png")
    args = ap.parse_args()

    files = {0: "pet_hinge_w5_shear.npz", 30: "pet_hinge_w5_phi30.npz",
             60: "pet_hinge_w5_phi60.npz", 90: "pet_hinge_w5_phi90.npz"}
    _style()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5.5))

    for phi, fn in files.items():
        p = os.path.join(args.dir, fn)
        if not os.path.exists(p):
            continue
        d = np.load(p)
        disp, force = _projected(d, phi)
        label = "φ=0° shear" if phi == 0 else ("φ=90° opening" if phi == 90 else f"φ={phi}°")
        ax.plot(disp, force, color=RAMP[phi], lw=2.2, label=label)
        fractured = float(np.nanmax(d["peeq_p99"])) >= 1.1 * float(d["eps_f"])
        if fractured:
            ax.scatter([disp[-1]], [force[-1]], marker="x", s=70, color=FAIL, zorder=5, lw=2)

    ax.scatter([], [], marker="x", s=70, color=FAIL, lw=2, label="predicted fracture")
    ax.set_xlabel("displacement along pull  [mm]")
    ax.set_ylabel("force along pull  [N]")
    ax.set_title("Loading-direction sweep — the failure locus", fontsize=13, loc="left",
                 pad=10, color=INK)
    ax.legend(frameon=False, fontsize=9.5, loc="lower right")
    ax.text(0.40, 0.30, "shear = most ductile · opening = fails earliest\n"
            "run each on the Instron to break —\ncalibrates eps_f0 & k",
            transform=ax.transAxes, fontsize=9, color=MUTE, va="top")
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    fig.suptitle("PET hinge — Instron φ-sweep prediction (w_lig = 5 mm, t = 0.5 mm)",
                 fontsize=12.5, fontweight="bold", x=0.02, ha="left", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=200, facecolor="white")
    print(f"figure -> {args.out}")


if __name__ == "__main__":
    main()
