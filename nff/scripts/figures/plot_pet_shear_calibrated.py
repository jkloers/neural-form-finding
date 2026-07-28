"""Predicted w=18mm shear response with the committed calibrated PET (9-pt *PLASTIC, eps_f0=1.5)."""
from __future__ import annotations
import argparse, os
import numpy as np

SIM, FAIL, INK = "#08519c", "#D62828", "#212529"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="data/experiments/processed/pet_w18_shear_calibrated.npz")
    ap.add_argument("--out", default="data/experiments/processed/pet_w18_shear_calibrated.png")
    args = ap.parse_args()
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                         "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#adb5bd",
                         "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})
    d = np.load(args.npz)
    s = np.abs(np.asarray(d["s"], float)); F = np.abs(np.asarray(d["F_s"], float))
    uz = np.asarray(d["uz_max"], float); dam = np.asarray(d["damage_p99"], float)
    m = np.isfinite(s) & np.isfinite(F); s, F, uz, dam = s[m], F[m], uz[m], dam[m]
    o = np.argsort(s); s, F, uz, dam = s[o], F[o], uz[o], dam[o]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))
    a1.plot(s, F, color=SIM, lw=2.2)
    fr = np.where(dam >= 1.0)[0]
    if len(fr):
        sf = s[fr[0]]; a1.plot(sf, np.interp(sf, s, F), "x", color=FAIL, ms=11, mew=2.6)
        a1.annotate(f"predicted tear\n s≈{sf:.1f} mm", (sf, np.interp(sf, s, F)),
                    textcoords="offset points", xytext=(-90, -10), color=FAIL, fontsize=9)
    a1.set_xlabel("shear displacement  s  [mm]"); a1.set_ylabel("shear force  F_s  [N]")
    a1.set_title("Predicted shear response (calibrated PET)", fontsize=11, loc="left")
    a1.set_xlim(0, None); a1.set_ylim(0, None)

    a2.plot(s, uz, color=SIM, lw=2.2)
    a2.set_xlabel("shear displacement  s  [mm]"); a2.set_ylabel("out-of-plane buckle  uz_max  [mm]")
    a2.set_title("Out-of-plane buckling", fontsize=11, loc="left")
    a2.set_xlim(0, None); a2.set_ylim(0, None)
    fig.tight_layout(); fig.savefig(args.out, dpi=160, bbox_inches="tight")
    print(f"peak F_s={F.max():.0f}N @ s={s[np.argmax(F)]:.1f}  uz_max={uz.max():.1f}  "
          f"tear={'s≈%.1f'%s[fr[0]] if len(fr) else 'no tear (maxD=%.2f)'%dam.max()}")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
