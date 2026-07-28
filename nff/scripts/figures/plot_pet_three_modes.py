"""w=18mm PET hinge — three loading modes, calibrated model vs real data where available.

A  Out-of-plane buckling (fold): uz vs theta   -- sim + real bh   [VALIDATED]
B  Shear:                        F_s vs s       -- sim only        [PREDICTION, untested]
C  Tension (opening):            F_a vs opening -- sim + real      [VALIDATED ~11%]
"""
from __future__ import annotations
import argparse
import os
import numpy as np

from nff.utils.figstyle import SIM, REAL, FAIL, INK, MUTE, apply_charter, finite_sorted, use_agg

D = "data/experiments/processed"


def _clean(x, y):
    """finite_sorted on |x| -- these axes are magnitudes (fold angle, shear travel)."""
    return finite_sorted(np.abs(np.asarray(x, float)), y)


def _tear(ax, x, y, dam):
    xd, dd = _clean(x, dam); fr = np.where(dd >= 1.0)[0]
    if len(fr):
        xf = xd[fr[0]]; ax.plot(xf, np.interp(xf, *_clean(x, y)), "x", color=FAIL, ms=10, mew=2.4,
                                label="predicted tear")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=D, help="directory holding the processed npz/csv inputs")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dd = args.dir
    out = args.out or f"{dd}/pet_w18_three_modes.png"
    use_agg()
    import matplotlib.pyplot as plt
    apply_charter(**{"font.family": "sans-serif",
                     "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                     "xtick.color": INK, "ytick.color": INK})
    fig, (aB, aS, aT) = plt.subplots(1, 3, figsize=(15, 4.4))

    # A: fold buckling (uz material-insensitive; real bh)
    f = np.load(f"{dd}/pet_w18_fold90_st3e-4.npz")
    th, uz = _clean(f["theta_deg"], f["uz_max"])
    aB.plot(th, uz, color=SIM, lw=2.2, label="CalculiX")
    # The measured fold curve lives in the CSV, not retyped here.
    rb = f"{dd}/real_w18_fold_bh.csv"
    if os.path.exists(rb):
        e = np.loadtxt(rb, delimiter=",", skiprows=1)
        aB.plot(e[:, 0], e[:, 1], "o", color=REAL, ms=6, label="experiment", zorder=5)
    aB.set_xlabel("fold angle  θ  [deg]"); aB.set_ylabel("out-of-plane buckle  uz  [mm]")
    aB.set_title("A · Out-of-plane buckling (fold)", fontsize=11, loc="left")
    aB.legend(frameon=False, fontsize=9); aB.set_xlim(0, None); aB.set_ylim(0, None)

    # B: shear (prediction; no real data)
    p = f"{dd}/pet_w18_shear_calibrated.npz"
    if os.path.exists(p):
        d = np.load(p); s, F = _clean(d["s"], d["F_s"])
        aS.plot(s, F, color=SIM, lw=2.2, label="CalculiX")
        _tear(aS, d["s"], np.abs(d["F_s"]), d["damage_p99"])
    aS.text(0.5, 0.06, "no experiment yet — prediction", transform=aS.transAxes,
            ha="center", color=MUTE, fontsize=9, style="italic")
    aS.set_xlabel("shear displacement  s  [mm]"); aS.set_ylabel("shear force  F_s  [N]")
    aS.set_title("B · Shear", fontsize=11, loc="left")
    aS.legend(frameon=False, fontsize=9); aS.set_xlim(0, None); aS.set_ylim(0, None)

    # C: tension / opening (calibrated sim + real on TRUE ligament-opening axis from the video)
    d = np.load(f"{dd}/pet_w18_open_multipoint.npz"); a, F = _clean(d["a"], d["F_a"])
    aT.plot(a, F, color=SIM, lw=2.2, label="CalculiX")
    from nff.rve.materials.pet import PETIsotropic         # k = 0 -> D = PEEQ / eps_f0 flat
    _tear(aT, d["a"], np.abs(d["F_a"]), d["peeq"] / PETIsotropic().eps_f)
    rc = f"{dd}/real_w18_open_nodamage.csv"
    if os.path.exists(rc):                                  # crosshead, minus the measured load-train term
        from nff.calibration.compliance import C_HINGE_W18, correct_displacement
        r = np.loadtxt(rc, delimiter=",", skiprows=1)
        aT.plot(correct_displacement(r[:, 0], r[:, 1], C_HINGE_W18), r[:, 1], "--", color=REAL,
                lw=1.3, alpha=0.75, label="experiment (crosshead $-\\,C\\,F$)")
    rt = f"{dd}/real_w18_open_trueaxis.csv"
    if os.path.exists(rt):                                  # solid: video fillet-hole TRUE opening
        r = np.loadtxt(rt, delimiter=",", skiprows=1)
        aT.plot(r[:, 0], r[:, 1], color=REAL, lw=2.2, label="experiment (video, true opening)")
    aT.set_xlabel("ligament opening a  [mm]"); aT.set_ylabel("opening force  F_a  [N]")
    aT.set_title("C · Tension (opening)", fontsize=11, loc="left")
    aT.legend(frameon=False, fontsize=9); aT.set_xlim(0, None); aT.set_ylim(0, None)

    fig.tight_layout(); fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
