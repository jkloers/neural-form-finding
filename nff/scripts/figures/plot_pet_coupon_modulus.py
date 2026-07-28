"""PET coupon modulus: the compliance-corrected curves and the initial-tangent fit.

A  the fit itself — corrected curves in the 1-5 MPa window with the per-specimen tangent lines.
B  the same curves out to yield — why that tangent is only the FIRST part of the curve.

    python -m nff.scripts.figures.plot_pet_coupon_modulus
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from nff.calibration import bluehill_io, compliance, summary_io

REAL, MUTE, INK, LIT, FIT = "#E8590C", "#adb5bd", "#212529", "#6C757D", "#08519c"
D = "data/experiments/processed"
RAW = "data/experiments/raw/kirigami_20260723"
SUMMARY = f"{RAW}/Kirigami experiments - Tensile test (safe).csv"
YIELD_MPA = 44.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{D}/pet_coupon_modulus.png")
    ap.add_argument("--C", type=float, default=compliance.C_HINGE_W18)
    args = ap.parse_args()
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                         "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#adb5bd",
                         "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})

    rows = [(m, bluehill_io.load_raw_csv(f"{RAW}/{m.name.replace('.', '_')}.csv"))
            for m in summary_io.load_summary(SUMMARY)
            if m.included and os.path.exists(f"{RAW}/{m.name.replace('.', '_')}.csv")]
    lo, hi = compliance.TANGENT_WINDOW_MPA
    fig, (aF, aW) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    Es = []
    for j, (m, r) in enumerate(rows):
        F, A, L = r.force_N, m.a_mean_mm2, m.L_mm
        sig = F / A
        e_raw = r.disp_mm / L
        e_cor = compliance.correct_displacement(r.disp_mm, F, args.C) / L
        E, _ = compliance.modulus_from_slope(F, r.disp_mm, A, L, C_mm_per_N=args.C)
        Es.append(E)

        z = sig <= 12.0
        aF.plot(e_raw[z] * 100, sig[z], color=MUTE, lw=0.9,
                label="raw crosshead" if j == 0 else None)
        aF.plot(e_cor[z] * 100, sig[z], color=REAL, lw=1.3,
                label="compliance-corrected" if j == 0 else None)
        w = (sig >= lo) & (sig <= hi)                       # the fitted segment, drawn on top
        b = np.polyfit(e_cor[w], sig[w], 1)
        xs = np.array([0.0, 0.0035])
        aF.plot(xs * 100, np.polyval(b, xs), "--", color=FIT, lw=0.9, alpha=0.85,
                label="per-specimen tangent fit" if j == 0 else None)

        z2 = sig <= 50.0
        aW.plot(e_cor[z2] * 100, sig[z2], color=REAL, lw=1.2,
                label="compliance-corrected" if j == 0 else None)

    Es = np.array(Es)
    aF.axhspan(lo, hi, color=FIT, alpha=0.07)
    aF.text(0.31, 0.5 * (lo + hi), "fit window", color=FIT, fontsize=8, va="center")
    aF.set_xlim(0, 0.35); aF.set_ylim(0, 12)
    aF.set_xlabel("strain  [%]"); aF.set_ylabel("stress  [MPa]")
    aF.set_title("A · Initial-tangent fit", fontsize=11, loc="left")
    aF.text(0.03, 10.9, f"E = {Es.mean()/1000:.2f} $\\pm$ {Es.std()/1000:.2f} GPa   (n={len(rows)})",
            color=FIT, fontsize=10)
    aF.legend(frameon=False, fontsize=9, loc="lower right")

    e = np.linspace(0, 0.05, 2)
    aW.plot(e * 100, Es.mean() * e, ":", color=INK, lw=1.8,
            label=f"E = {Es.mean()/1000:.1f} GPa tangent")
    aW.axhline(YIELD_MPA, color=LIT, lw=1, ls="--")
    aW.text(0.15, YIELD_MPA + 1.0, "yield 44 MPa", color=LIT, fontsize=9)
    aW.axvspan(0, hi / Es.mean() * 100, color=FIT, alpha=0.10)
    aW.annotate("fit window\n(first 0.17% strain)", xy=(hi / Es.mean() * 100, 8), xytext=(1.1, 17),
                color=FIT, fontsize=9,
                arrowprops=dict(arrowstyle="->", color=FIT, lw=1, shrinkB=2))
    aW.set_xlim(0, 5); aW.set_ylim(0, 50)
    aW.set_xlabel("strain  [%]"); aW.set_ylabel("stress  [MPa]")
    aW.set_title("B · Out to yield — the tangent is only the first 0.2%", fontsize=11, loc="left")
    aW.legend(frameon=False, fontsize=9, loc="lower right")

    fig.tight_layout(); fig.savefig(args.out, dpi=160, bbox_inches="tight")
    print(f"C = {args.C*1000:.2f} um/N   window {lo:.0f}-{hi:.0f} MPa")
    print(f"E = {Es.mean()/1000:.2f} +- {Es.std()/1000:.2f} GPa   "
          f"per specimen: {' '.join(f'{v/1000:.2f}' for v in Es)}")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
