"""Predicted PET shear-hinge family across ligament width (r_win = 100 mm domain).

Two panels for the calibration sweep w_lig = 10/20/30/40/50 mm:
  A  shear force F_s vs shear displacement s (the Instron target), x = predicted fracture (D>=1)
  B  out-of-plane buckle height uz_max vs s (the developable-cone amplitude)

Reads data/experiments/processed/pet_shear_w{W}_rwin100.npz. Diverged widths are skipped
with a note. Overlay real coupon curves on panel A later.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

WIDTHS = [10, 20, 30, 40, 50]
# sequential blue ramp, light (narrow) -> dark (wide)
RAMP = {10: "#bdd7e7", 20: "#6baed6", 30: "#3182bd", 40: "#08519c", 50: "#08306b"}
INK, MUTE, FAIL = "#212529", "#868e96", "#D62828"


def _clean(s: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sort by s, keep finite, monotone-s samples."""
    m = np.isfinite(s) & np.isfinite(y)
    s, y = s[m], y[m]
    o = np.argsort(s, kind="stable")
    return s[o], y[o]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="data/experiments/processed")
    ap.add_argument("--out", default="data/experiments/processed/pet_shear_sweep_rwin100.png")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#adb5bd", "text.color": INK,
                         "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})

    fig, (axF, axZ) = plt.subplots(1, 2, figsize=(11, 4.4))

    for w in WIDTHS:
        f = os.path.join(args.dir, f"pet_shear_w{w}_rwin100.npz")
        if not os.path.exists(f):
            continue
        d = np.load(f)
        s = np.asarray(d["s"], float)
        Fs = np.asarray(d["F_s"], float)
        uz = np.asarray(d["uz_max"], float)
        dam = np.asarray(d["damage_p99"], float)
        c = RAMP[w]

        # divergence guard: solve died before any meaningful displacement
        if np.nanmax(s) < 1.0 or np.isfinite(Fs).sum() < 6:
            axF.plot([], [], color=c, label=f"{w} mm  (diverged)")
            continue

        sF, F = _clean(s, np.abs(Fs))
        sZ, Z = _clean(s, uz)
        axF.plot(sF, F, color=c, lw=2.0, label=f"{w} mm")
        axZ.plot(sZ, Z, color=c, lw=2.0, label=f"{w} mm")

        # predicted fracture = first sample with damage D >= 1
        sd, D = _clean(s, dam)
        fr = np.where(D >= 1.0)[0]
        if len(fr):
            s_fr = sd[fr[0]]
            # force at that displacement
            F_fr = np.interp(s_fr, sF, F)
            axF.plot(s_fr, F_fr, "x", color=FAIL, ms=8, mew=2.2, zorder=5)

    axF.set_xlabel("shear displacement  s  [mm]")
    axF.set_ylabel("shear force  F_s  [N]")
    axF.set_title("Predicted shear response", fontsize=11, loc="left")
    axF.legend(title="w_lig", frameon=False, fontsize=9, title_fontsize=9)
    axF.plot([], [], "x", color=FAIL, ms=8, mew=2.2, label="predicted tear")

    axZ.set_xlabel("shear displacement  s  [mm]")
    axZ.set_ylabel("out-of-plane buckle  uz_max  [mm]")
    axZ.set_title("Predicted cone amplitude", fontsize=11, loc="left")

    fig.tight_layout()
    fig.savefig(args.out, dpi=160, bbox_inches="tight")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
