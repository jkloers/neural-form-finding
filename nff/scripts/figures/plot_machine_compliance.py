"""Load-train compliance: measure it on the hinge video, then apply it to the coupon modulus.

A  w=18mm hinge — force vs crosshead travel and vs the TRUE ligament opening (video).
B  the difference, excess = x - a, against force: a straight line whose slope is C [mm/N].
C  coupon elastic response near the origin, raw vs compliance-corrected, against E = 3 GPa.
D  tangent modulus vs stress — why the window has to be early: PET bends within a few MPa.

    python -m nff.scripts.figures.plot_machine_compliance
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from nff.calibration import bluehill_io, compliance, summary_io

SIM, REAL, MUTE, INK, LIT = "#08519c", "#E8590C", "#adb5bd", "#212529", "#6C757D"
D = "data/experiments/processed"
RAW = "data/experiments/raw/kirigami_20260723"
SUMMARY = f"{RAW}/Kirigami experiments - Tensile test (safe).csv"


def _coupons():
    for m in summary_io.load_summary(SUMMARY):
        fn = f"{RAW}/{m.name.replace('.', '_')}.csv"
        if m.included and os.path.exists(fn):
            yield m, bluehill_io.load_raw_csv(fn)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{D}/pet_machine_compliance.png")
    args = ap.parse_args()
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                         "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#adb5bd",
                         "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})

    x = np.loadtxt(f"{D}/real_w18_open_nodamage.csv", delimiter=",", skiprows=1)
    v = np.loadtxt(f"{D}/real_w18_open_trueaxis.csv", delimiter=",", skiprows=1)
    C, off, rms = compliance.fit_compliance(x[:, 0], x[:, 1], v[:, 0], v[:, 1])

    fig, ((aA, aB), (aC, aD)) = plt.subplots(2, 2, figsize=(11, 8.6))

    # A -- the two displacement axes for the same test
    aA.plot(x[:, 0], x[:, 1], color=MUTE, lw=2, label="crosshead (machine)")
    aA.plot(v[:, 0], v[:, 1], color=REAL, lw=2.2, label="ligament opening (video)")
    aA.plot(compliance.correct_displacement(x[:, 0], x[:, 1], C), x[:, 1], "--",
            color=SIM, lw=1.6, label="crosshead $-\\,C\\,F$")
    aA.set_xlabel("displacement  [mm]"); aA.set_ylabel("force  F  [N]")
    aA.set_title("A · w=18 mm hinge, two displacement axes", fontsize=11, loc="left")
    aA.legend(frameon=False, fontsize=9); aA.set_xlim(0, None); aA.set_ylim(0, None)

    # B -- the calibration itself
    def _rise(d, f):
        i = int(np.argmax(f)); d, f = d[: i + 1], f[: i + 1]
        k = np.concatenate([[True], np.diff(f) > 1e-9]); return d[k], f[k]
    xd, xf = _rise(x[:, 0], x[:, 1]); td, tf = _rise(v[:, 0], v[:, 1])
    g = np.linspace(30.0, min(xf.max(), tf.max()) * 0.97, 200)
    exc = np.interp(g, xf, xd) - np.interp(g, tf, td)
    aB.plot(g, exc, color=REAL, lw=2.2, label="measured  $x-a$")
    aB.plot(g, C * g + off, "--", color=INK, lw=1.6,
            label=f"$C\\,F$,  C = {C*1000:.2f} $\\mu$m/N")
    aB.text(0.04, 0.9, f"residual {rms*1000:.0f} $\\mu$m rms", transform=aB.transAxes,
            fontsize=9, color=MUTE)
    aB.set_xlabel("force  F  [N]"); aB.set_ylabel("crosshead excess  $x-a$  [mm]")
    aB.set_title("B · Load-train compliance", fontsize=11, loc="left")
    aB.legend(frameon=False, fontsize=9, loc="lower right")

    # C -- coupon modulus at the ORIGIN (the curve leaves linear within a few MPa)
    rows = list(_coupons())
    lo, hi = compliance.TANGENT_WINDOW_MPA
    Er, Ec, slopes, LA = [], [], [], []
    for j, (m, run) in enumerate(rows):
        F, A, L = run.force_N, m.a_mean_mm2, m.L_mm
        d, sig = run.disp_mm, run.force_N / m.a_mean_mm2
        w = sig <= 12.0
        aC.plot(d[w] / L, sig[w], color=MUTE, lw=1.0,
                label="crosshead (raw)" if j == 0 else None)
        aC.plot(compliance.correct_displacement(d, F, C)[w] / L, sig[w], color=REAL, lw=1.2,
                label="compliance-corrected" if j == 0 else None)
        e0, s0 = compliance.modulus_from_slope(F, d, A, L)
        e1, _ = compliance.modulus_from_slope(F, d, A, L, C_mm_per_N=C)
        Er.append(e0); Ec.append(e1); slopes.append(s0); LA.append(L / A)
    e = np.linspace(0, 0.006, 2)
    aC.plot(e, 3000 * e, ":", color=LIT, lw=2, label="E = 3.0 GPa (literature)")
    aC.axhspan(lo, hi, color=REAL, alpha=0.08)
    aC.set_xlim(0, 0.006); aC.set_ylim(0, 12)
    aC.set_xlabel("strain  [-]"); aC.set_ylabel("stress  [MPa]")
    aC.set_title(f"C · Coupon modulus at the origin (n={len(rows)})", fontsize=11, loc="left")
    aC.legend(frameon=False, fontsize=9, loc="lower right")

    # D -- tangent modulus vs stress: where the curve stops being linear
    bands = [(0.5, 2.0), (1.0, 3.0), (1.0, 5.0), (2.0, 6.0), (4.0, 8.0), (6.0, 10.0),
             (8.0, 12.0), (10.0, 15.0), (15.0, 20.0), (20.0, 26.0)]
    mid = [0.5 * (a + b) for a, b in bands]
    for cc, col, lbl in ((0.0, MUTE, "raw crosshead"), (C, REAL, "compliance-corrected")):
        curve = []
        for a, b in bands:
            vals = [compliance.modulus_from_slope(r.force_N, r.disp_mm, m.a_mean_mm2, m.L_mm,
                                                  stress_lo=a, stress_hi=b, C_mm_per_N=cc)[0]
                    for m, r in rows]
            v = np.array([x for x in vals if np.isfinite(x)])
            curve.append((v.mean() / 1000, v.std() / 1000))
        mu = np.array([c[0] for c in curve]); sd = np.array([c[1] for c in curve])
        aD.plot(mid, mu, "o-", color=col, lw=2, ms=4, label=lbl)
        aD.fill_between(mid, mu - sd, mu + sd, color=col, alpha=0.15)
    aD.axhspan(2.0, 4.0, color=LIT, alpha=0.12)
    aD.text(21, 3.55, "PET literature", color=LIT, fontsize=9, ha="right")
    aD.axvspan(lo, hi, color=REAL, alpha=0.08)
    aD.text(np.mean((lo, hi)), 0.15, "fit\nwindow", color=REAL, fontsize=8, ha="center")
    aD.set_xlabel("stress window centre  [MPa]"); aD.set_ylabel("tangent modulus  E  [GPa]")
    aD.set_title("D · The curve bends early", fontsize=11, loc="left")
    aD.set_ylim(0, 5); aD.legend(frameon=False, fontsize=9)

    fig.tight_layout(); fig.savefig(args.out, dpi=160, bbox_inches="tight")

    Er, Ec, slopes, LA = map(np.array, (Er, Ec, slopes, LA))
    need = slopes - LA / 3000.0
    print(f"C = {C*1000:.2f} um/N  (offset {off*1000:+.0f} um, residual {rms*1000:.0f} um rms)")
    xm, am = np.interp(g[-1], xf, xd), np.interp(g[-1], tf, td)
    print(f"  at F={g[-1]:.0f}N: crosshead {xm:.2f} mm vs true opening {am:.2f} mm "
          f"-> {xm/am:.2f}x, {100*(xm-am)/xm:.0f}% of the travel is the machine")
    print(f"initial tangent, sigma {lo:.0f}-{hi:.0f} MPa   (mean slope {slopes.mean()*1000:.2f} um/N, "
          f"specimen share {100*(1-C/slopes.mean()):.0f}%)")
    print(f"  coupon E   raw {Er.mean()/1000:.2f} +- {Er.std()/1000:.2f} GPa"
          f"   corrected {Ec.mean()/1000:.2f} +- {Ec.std()/1000:.2f} GPa")
    print(f"  C required for E = 3.0 GPa: {need.mean()*1000:.2f} +- {need.std()*1000:.2f} um/N"
          f"  ({need.mean()/C:.2f}x the measured value)")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
