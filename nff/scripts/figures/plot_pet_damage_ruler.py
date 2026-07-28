"""How damage accumulates through a w=18mm PET hinge deployment, and what the experiments pin.

A  fold deployment: plastic strain and ductile damage against fold angle.
B  the three loading modes on a common deployment-fraction axis, against the D=1 tear line.
C  the "damage ruler" — the plastic-strain axis with the model's thresholds and, next to them,
   the only points the physical experiments actually constrain.

    python -m nff.scripts.figures.plot_pet_damage_ruler
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from nff.rve.materials.pet import PETIsotropic

SIM, REAL, FAIL, INK, MUTE, LIT = "#08519c", "#E8590C", "#D62828", "#212529", "#adb5bd", "#6C757D"
D = "data/experiments/processed"
EPS_F0 = PETIsotropic().eps_f      # no triaxiality locus for PET, so eps_f is stress-state-independent
W_LIG = 18.0
WHITEN_THETA = 20.0                # first visible stress-whitening in the real fold (2026-07-26)


def _clean(x, *ys):
    x = np.asarray(x, float); m = np.isfinite(x)
    for y in ys:
        m &= np.isfinite(np.asarray(y, float))
    o = np.argsort(x[m])
    return (x[m][o],) + tuple(np.asarray(y, float)[m][o] for y in ys)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{D}/pet_damage_ruler.png")
    args = ap.parse_args()
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                         "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#adb5bd",
                         "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})
    fig, (aA, aB, aC) = plt.subplots(1, 3, figsize=(15.5, 4.6))

    # ---- A: the fold, plastic strain and damage vs angle ------------------
    f = np.load(f"{D}/pet_w18_fold90_st3e-4.npz", allow_pickle=True)
    th, pe = _clean(f["theta_deg"], f["peeq_p99"])
    aA.plot(th, pe, color=SIM, lw=2.2, label="plastic strain  PEEQ$_{p99}$")
    aA.axvline(WHITEN_THETA, color=REAL, lw=1.2, ls="--")
    aA.plot([WHITEN_THETA], [np.interp(WHITEN_THETA, th, pe)], "o", color=REAL, ms=7, zorder=5)
    aA.annotate("whitening first visible\n(real fold)", xy=(WHITEN_THETA, np.interp(WHITEN_THETA, th, pe)),
                xytext=(30, 0.028), color=REAL, fontsize=9,
                arrowprops=dict(arrowstyle="->", color=REAL, lw=1))
    aA.set_xlabel("fold angle  θ  [deg]"); aA.set_ylabel("PEEQ$_{p99}$  [-]")
    aA.set_xlim(0, None); aA.set_ylim(0, None)
    aA2 = aA.twinx(); aA2.spines["top"].set_visible(False)
    aA2.plot(th, pe / EPS_F0, color=MUTE, lw=1.2)
    aA2.set_ylabel("ductile damage  D = PEEQ / $\\epsilon_f$", color=MUTE)
    aA2.tick_params(axis="y", colors=MUTE); aA2.set_ylim(0, aA.get_ylim()[1] / EPS_F0)
    aA.set_title("A · Fold deployment", fontsize=11, loc="left")
    aA.legend(frameon=False, fontsize=9, loc="upper left")

    # ---- B: three modes on a common deployment fraction -------------------
    modes = [
        ("fold  θ/90°", f["theta_deg"], f["peeq_p99"], 90.0),
        ("opening  a/w", None, None, None),
        ("shear  s/w", None, None, None),
    ]
    o = np.load(f"{D}/pet_w18_open_multipoint.npz")
    a, pe_o = _clean(o["a"], o["peeq"])
    s_npz = f"{D}/pet_w18_shear_calibrated.npz"
    for lbl, xx, yy, norm, col, ls in (
        ("fold  (θ / 90°)", th, pe, 90.0, SIM, "-"),
        ("opening  (a / w$_{lig}$)", a, pe_o, W_LIG, REAL, "-"),
    ):
        aB.plot(xx / norm, yy / EPS_F0, ls, color=col, lw=2.2, label=lbl)
    if os.path.exists(s_npz):
        sd = np.load(s_npz, allow_pickle=True)
        ss, dd = _clean(np.abs(sd["s"]), sd["damage_p99"])
        aB.plot(ss / W_LIG, dd, color=LIT, lw=2.2, label="shear  (s / w$_{lig}$)")
    aB.axhline(1.0, color=FAIL, lw=1.4, ls="--")
    aB.text(0.02, 1.05, "tear,  D = 1", color=FAIL, fontsize=9)
    aB.set_xlabel("deployment fraction  [-]"); aB.set_ylabel("ductile damage  D  [-]")
    aB.set_xlim(0, 1.0); aB.set_ylim(0, 1.5)
    aB.set_title("B · Damage per loading mode", fontsize=11, loc="left")
    aB.legend(frameon=False, fontsize=9, loc="upper left")

    # ---- C: the ruler -----------------------------------------------------
    bands = [(0.00, 0.10, "#EAF2FB", "yielded,\npermanent set", 0.99, "left"),
             (0.10, 0.27, "#D7E6F7", "pre-draw\nhardening", 0.80, "left"),
             (0.27, 1.19, "#B9D3F0", "cold drawing\n(neck propagates)", 0.99, "center"),
             (1.19, 2.10, "#F6D9C4", "post-draw\nre-hardening", 0.99, "center")]
    for lo, hi, c, lbl, y, ha in bands:
        aC.axvspan(lo, hi, color=c)
        xt = {"left": lo + 0.03, "center": 0.5 * (lo + hi)}[ha]
        aC.text(xt, y, lbl, ha=ha, va="top", fontsize=8, color=INK)
    aC.axvline(EPS_F0, color=FAIL, lw=2)
    aC.text(EPS_F0 + 0.03, 0.62, f"$\\epsilon_{{f0}}$ = {EPS_F0:.2f}\n(measured, n = 1)",
            color=FAIL, fontsize=9)

    ev = [(0.12, "fold to 57°\nreached only this", SIM),
          (1.19, "coupons drew to here\nUNBROKEN (n=9)", REAL),
          (1.35, "hinge opening survived\nthis, no tear", REAL),
          (1.99, "opening sim ends here\n(model says torn)", MUTE)]
    for i, (x, lbl, col) in enumerate(ev):
        y = 0.42 - 0.11 * i
        aC.plot([x], [y], "o", color=col, ms=8, zorder=5)
        aC.annotate(lbl, xy=(x, y), xytext=(x - 0.06, y), ha="right", va="center",
                    fontsize=8.5, color=col)
    aC.set_xlim(0, 2.1); aC.set_ylim(0, 1.0); aC.set_yticks([])
    aC.spines["left"].set_visible(False)
    aC.set_xlabel("equivalent plastic strain  PEEQ  [-]")
    aC.set_title("C · What the experiments actually pin", fontsize=11, loc="left")

    fig.tight_layout(); fig.savefig(args.out, dpi=160, bbox_inches="tight")
    print(f"fold: max PEEQ {pe.max():.3f} -> D {pe.max()/EPS_F0:.3f} at θ={th[-1]:.0f}deg")
    print(f"open: max PEEQ {pe_o.max():.3f} -> D {pe_o.max()/EPS_F0:.3f} at a={a[-1]:.1f}mm")
    print(f"PEEQ at first visible whitening (θ={WHITEN_THETA:.0f}deg): "
          f"{np.interp(WHITEN_THETA, th, pe):.3f}  ->  D = {np.interp(WHITEN_THETA, th, pe)/EPS_F0:.3f}")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
