"""PET run-to-break read through the ink-ladder video extensometer.

A  the whole test on the crosshead axis, with the landmarks the flow curve is built from.
B  the flow curve: the *PLASTIC table against the two cross-sections that anchor it.
C  the elastic region on three strain axes -- crosshead, compliance-corrected, and the
   video ladder, which owes nothing to either.
D  modulus against the stress window it was fitted over.

    python -m nff.scripts.figures.plot_pet_video_stress_strain
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from nff.calibration import compliance, ladder
from nff.rve.materials.pet import PET_PLASTIC

SIM, REAL, MUTE, INK, LIT, FAIL = "#08519c", "#E8590C", "#adb5bd", "#212529", "#6C757D", "#D62828"
RAW = "data/experiments/raw/tensile_break_20260727"
OUT = "data/experiments/processed"
CACHE = f"{OUT}/pet_video_ladder.npz"

W0, T0, L0 = 18.34, 0.50, 114.12         # pre-test width, thickness, grip separation [mm]
WF, TF = 8.56, 0.18                      # cross-section at the tear [mm]
A0, AF = W0 * T0, WF * TF
FPS, T_TRACK = 5.0, 95.0                 # strip video frame rate; track up to necking
STRIP_W, STRIP_H = 440, 1080
TICK_SEED = [363.2, 429.0, 488.8, 552.1, 618.0, 681.4]   # ladder rows in frame 0
GRIP_SEED = 193.0                        # strongest horizontal grip edge in frame 0
PLATEAU_ENG = 33.6                       # MPa, the constant-force draw plateau


def _load_csv():
    a = np.char.strip(
        np.genfromtxt(f"{RAW}/tensile_with_video.csv", delimiter=",", skip_header=2, dtype=str), '"'
    ).astype(float)
    return a[:, 0], a[:, 1], a[:, 2] * 1000.0            # t [s], crosshead [mm], force [N]


def _track(force_csv, disp_csv, t_csv):
    if os.path.exists(CACHE):
        d = np.load(CACHE)
        return d["t"], d["eps"], d["resid"], float(d["offset"]), float(d["mm_per_px"])
    n = int(T_TRACK * FPS)
    t_video = np.arange(n) / FPS
    strip = f"{RAW}/video_strip_5fps.mp4"
    band = ladder.decode_gray(strip, n, STRIP_W, STRIP_H, crop=(92, STRIP_H, 140, 0))
    eps, resid = ladder.gauge_strain(ladder.track_ladder(band, TICK_SEED, x0=8))
    full = ladder.decode_gray(strip, n, STRIP_W, STRIP_H)
    offset, mm_per_px, rms = ladder.sync_by_grip(full, GRIP_SEED, t_video, t_csv, disp_csv, x0=70)
    print(f"grip sync: offset {offset:+.2f} s, {mm_per_px:.4f} mm/px, {rms*1000:.0f} um rms")
    np.savez(CACHE, t=t_video, eps=eps, resid=resid, offset=offset, mm_per_px=mm_per_px)
    return t_video, eps, resid, offset, mm_per_px


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{OUT}/pet_video_stress_strain.png")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                         "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#adb5bd",
                         "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})

    t_csv, x_csv, f_csv = _load_csv()
    t_v, eps_v, resid, offset, mm_per_px = _track(f_csv, x_csv, t_csv)
    f_v = np.interp(t_v + offset, t_csv, f_csv)
    x_v = np.interp(t_v + offset, t_csv, x_csv)
    sig_v = f_v / A0
    sig_csv = f_csv / A0
    eps_f = np.log(A0 / AF)

    # machine compliance for THIS setup, from the video/crosshead pairing
    win = (sig_v >= 1.0) & (sig_v <= 20.0)
    design = np.stack([f_v[win], np.ones(int(win.sum()))], axis=1)
    (c_fit, off_mm), *_ = np.linalg.lstsq(design, x_v[win] - eps_v[win] * L0, rcond=None)

    fig, ((aA, aB), (aC, aD)) = plt.subplots(2, 2, figsize=(11.6, 8.8))

    # ---- A: the whole test on the crosshead axis --------------------------
    aA.plot(x_csv / L0 * 100, sig_csv, color=REAL, lw=1.6)
    i = int(np.argmax(f_csv))
    aA.plot([x_csv[i] / L0 * 100], [sig_csv[i]], "o", color=SIM, ms=6, zorder=5)
    aA.annotate("upper yield", xy=(x_csv[i] / L0 * 100, sig_csv[i]), xytext=(18, 46),
                color=SIM, fontsize=9, arrowprops=dict(arrowstyle="->", color=SIM, lw=1))
    aA.axhline(PLATEAU_ENG, color=LIT, lw=1, ls="--")
    aA.text(60, PLATEAU_ENG + 1.2, "cold-draw plateau", color=LIT, fontsize=9)
    aA.plot([x_csv[-1] / L0 * 100], [sig_csv[-1]], "o", color=FAIL, ms=6, zorder=5)
    aA.annotate("tear", xy=(x_csv[-1] / L0 * 100, sig_csv[-1]), xytext=(88, 18),
                color=FAIL, fontsize=9, arrowprops=dict(arrowstyle="->", color=FAIL, lw=1))
    aA.set_xlabel("crosshead strain  [%]")
    aA.set_ylabel("engineering stress  [MPa]")
    aA.set_xlim(0, 130)
    aA.set_ylim(0, 55)
    aA.set_title("A · The whole test", fontsize=11, loc="left")

    # ---- B: the flow curve and its two anchors ----------------------------
    tbl = np.array(PET_PLASTIC)
    old = tbl[:, 1] <= 1.189 + 1e-9
    aB.plot(tbl[old, 1], tbl[old, 0], "-", color=SIM, lw=2.2, label="*PLASTIC, coupon batch")
    aB.plot(tbl[~old | (tbl[:, 1] >= 1.189), 1], tbl[~old | (tbl[:, 1] >= 1.189), 0], "--",
            color=SIM, lw=2.2, label="extension, run-to-break")
    grid = np.linspace(0.272, eps_f, 200)
    aB.plot(grid, PLATEAU_ENG * np.exp(grid), ":", color=MUTE, lw=1.4,
            label=r"constant force,  $\sigma_{pl}e^{\varepsilon}$")
    aB.plot([1.189], [PLATEAU_ENG * np.exp(1.189)], "o", color=REAL, ms=7, zorder=5)
    aB.annotate("natural draw\n(post-mortem + video)", xy=(1.189, PLATEAU_ENG * np.exp(1.189)),
                xytext=(0.45, 150), color=REAL, fontsize=9,
                arrowprops=dict(arrowstyle="->", color=REAL, lw=1))
    aB.plot([eps_f], [f_csv[-1000] / AF], "o", color=FAIL, ms=7, zorder=5)
    aB.annotate("tear section\n(measured)", xy=(eps_f, f_csv[-1000] / AF), xytext=(1.05, 60),
                color=FAIL, fontsize=9, arrowprops=dict(arrowstyle="->", color=FAIL, lw=1))
    aB.set_xlabel("true plastic strain  [-]")
    aB.set_ylabel("true stress  [MPa]")
    aB.set_xlim(0, 1.95)
    aB.set_ylim(0, 215)
    aB.set_title("B · Flow curve", fontsize=11, loc="left")
    aB.legend(frameon=False, fontsize=9, loc="upper left")

    # ---- C: the elastic region, three strain axes -------------------------
    lo, hi = 0.5, 15.0
    z = sig_csv <= 55
    aC.plot(x_csv[z] / L0 * 100, sig_csv[z], color=MUTE, lw=1.6, label="crosshead (raw)")
    over = compliance.correct_displacement(x_csv, f_csv, compliance.C_HINGE_W18)
    aC.plot(over[z] / L0 * 100, sig_csv[z], color=LIT, lw=1.6,
            label="crosshead $-\\,C_{hinge}F$")
    aC.plot(eps_v * 100, sig_v, "o", color=REAL, ms=2.4, label="video ladder")
    fitw = (sig_v >= lo) & (sig_v <= hi)
    slope, intercept = np.polyfit(eps_v[fitw], sig_v[fitw], 1)
    line = np.array([-0.2, 1.3])
    aC.plot(line, slope * line / 100 + intercept, "--", color=SIM, lw=1.5,
            label="fit to the ladder")
    aC.axhspan(lo, hi, color=SIM, alpha=0.06)
    aC.annotate("over-corrected:\nfolds back", xy=(2.85, 47), xytext=(1.55, 46),
                color=LIT, fontsize=8.5, ha="right",
                arrowprops=dict(arrowstyle="->", color=LIT, lw=1))
    aC.set_xlim(-0.25, 3.6)
    aC.set_ylim(0, 55)
    aC.set_xlabel("strain  [%]")
    aC.set_ylabel("engineering stress  [MPa]")
    aC.set_title("C · Elastic region, three strain axes", fontsize=11, loc="left")
    aC.legend(frameon=False, fontsize=9, loc="lower right")
    aC.text(-0.1, 51, f"ladder  E = {slope/1000:.2f} GPa", color=SIM, fontsize=10)

    # ---- D: the ladder is the only anchored number ------------------------
    bands = [(0.5, 3), (0.5, 5), (1, 8), (2, 10), (4, 15), (8, 20), (14, 28), (22, 38)]
    mid, lad, err = [], [], []
    for a, b in bands:
        m = (sig_v >= a) & (sig_v <= b)
        if m.sum() < 5:
            continue
        p, c = np.polyfit(eps_v[m], sig_v[m], 1)
        s = np.sqrt(((sig_v[m] - p * eps_v[m] - c) ** 2).sum() / (m.sum() - 2))
        mid.append(0.5 * (a + b))
        lad.append(p / 1000)
        err.append(s / np.sqrt(((eps_v[m] - eps_v[m].mean()) ** 2).sum()) / 1000)
    aD.axhspan(2.0, 4.0, color=LIT, alpha=0.12)
    aD.text(37, 3.72, "PET literature", color=LIT, fontsize=9, ha="right")
    w15 = (sig_csv >= 1) & (sig_csv <= 5)
    for cc, sty, lbl in ((0.0, ":", "crosshead, no correction"),
                         (c_fit, "-.", "crosshead $-\\,C_{fit}F$"),
                         (compliance.C_HINGE_W18, "--", "crosshead $-\\,C_{hinge}F$")):
        d = compliance.correct_displacement(x_csv, f_csv, cc) / L0
        e = np.polyfit(d[w15], sig_csv[w15], 1)[0] / 1000
        aD.axhline(e, ls=sty, color=MUTE if cc == 0 else LIT, lw=1.3)
        aD.text(37.5, e + 0.10, lbl, color=MUTE if cc == 0 else LIT, fontsize=8.5, ha="right")
    aD.errorbar(mid, lad, yerr=err, fmt="o-", color=REAL, lw=2.2, ms=5, capsize=3,
                label="video ladder")
    aD.set_xlabel("stress window centre  [MPa]")
    aD.set_ylabel("fitted modulus  [GPa]")
    aD.set_ylim(0, 7)
    aD.set_xlim(0, 39)
    aD.set_title("D · The ladder is the only anchored number", fontsize=11, loc="left")
    aD.legend(frameon=False, fontsize=9, loc="upper left")

    fig.tight_layout()
    fig.savefig(args.out, dpi=160, bbox_inches="tight")

    print(f"A0 = {A0:.3f} mm2   Af = {AF:.4f} mm2   eps_f = {eps_f:.3f}")
    print(f"ladder residual: median {np.median(resid):.2f} px  max {resid.max():.2f} px")
    print(f"E from the ladder over {lo:.1f}-{hi:.1f} MPa: {slope/1000:.2f} GPa "
          f"(strain intercept {-intercept/slope*100:+.3f} %)")
    print(f"this setup's compliance from the ladder pairing: C = {c_fit*1000:.2f} um/N "
          f"(hinge value {compliance.C_HINGE_W18*1000:.2f})")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
