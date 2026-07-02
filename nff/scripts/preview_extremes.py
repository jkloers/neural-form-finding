"""Preview the hinge model with a FIXED meshed window (same physical size for every hinge,
so all plots share one scale) and the mesh refined only on the uncut ligament strip.

Fig 1  strip-width sweep: deployed states for w_lig = 1, 5, 10 (fixed window, same scale).
Fig 2  alpha check: deployed states for alpha = 45, 90, 135 (w_lig=5), coloured by strain.
Fig 3  curves: energy W(theta) and moment M(theta), finely sampled from small theta.

Run:  PYTHONPATH=. conda run -n ccx python nff/scripts/preview_extremes.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

from nff.rve.geometry import RVEParams
from nff.rve.ccx_solver import deploy

WC, T, SY, EPS_F = 0.2, 0.5, 235.0, 0.25
WINDOW = 40.0                                     # FIXED meshed half-disk radius (mm), all hinges
HALF = WINDOW                                     # plot half-width (mm)


def P(wl, al):
    return RVEParams(w_lig=wl, w_c=WC, alpha_deg=al, rho=0.15 * wl, thickness=T, r_win=WINDOW)


def princ_strain(frame):                          # per-node max principal total strain
    e = frame.get("TOSTRAIN")
    if e is None or not e.size:
        return None
    out = np.zeros(len(e))
    for i, (xx, yy, zz, xy, yz, zx) in enumerate(e[:, :6]):
        out[i] = np.linalg.eigvalsh(np.array([[xx, xy, zx], [xy, yy, yz], [zx, yz, zz]]))[-1]
    return out


def run(wl, al, deg=40.0):
    p = P(wl, al)
    r = deploy(p, angle_deg=deg, n_steps=15, elastic_only=False,
               lc_min=max(0.08, 0.5 * 0.15 * wl), lc_max=WINDOW / 6, timeout=200,
               workdir=f"/tmp/pw_{wl*10:.0f}_{al}")
    fr = r["frames"]
    r["p99"] = np.array([np.percentile(np.abs(f["PE"]), 99) if "PE" in f and f["PE"].size else np.nan for f in fr])
    r["reached"] = r["theta_deg"][-1] if len(r["theta_deg"]) else 0.0
    rup = np.where(r["p99"] >= EPS_F)[0]
    r["rupture"] = r["reached"] * rup[0] / max(len(fr) - 1, 1) if len(rup) else None
    print(f"  w={wl:<4} a={al}: reached {r['reached']:.0f}deg ok={r['ok']} elems={r['n_elems']} "
          f"rupture@{None if r['rupture'] is None else round(r['rupture'],1)} p99_end={r['p99'][-1]:.2f}", flush=True)
    return r


def state(ax, r, title):
    disp = r["frames"][-1]["DISP"][:, :3]
    Pp = r["xyz"] + disp
    tris = np.array([[c[3] - 1, c[4] - 1, c[5] - 1] for c in r["conn"]])
    col = princ_strain(r["frames"][-1])
    col = np.clip(col, 0, EPS_F) if col is not None else Pp[:, 2]
    ax.tripcolor(Triangulation(Pp[:, 0], Pp[:, 1], tris), col, cmap="inferno", vmin=0, vmax=EPS_F, shading="gouraud")
    ax.plot(*r["pivot"], "c*", ms=10)
    ax.set_xlim(-HALF, HALF); ax.set_ylim(-HALF, 3); ax.set_aspect("equal")
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]"); ax.set_title(title, fontsize=9)


def main(out="data/outputs"):
    os.makedirs(out, exist_ok=True)
    ws = {wl: run(wl, 90) for wl in (1.0, 5.0, 10.0)}
    als = {al: (ws[5.0] if al == 90 else run(5.0, al)) for al in (45, 90, 135)}

    fig, ax = plt.subplots(1, 3, figsize=(16, 6))       # Fig 1: strip-width sweep, same scale
    for a, wl in zip(ax, (1.0, 5.0, 10.0)):
        r = ws[wl]
        state(a, r, f"w_lig={wl:.0f} mm  (gap≈{0.85*wl:.1f})  deployed {r['reached']:.0f}°")
    fig.suptitle("Uncut-strip width sweep — FIXED 40 mm window, same scale, colour = principal strain (cap ε_f)")
    fig.tight_layout(); fig.savefig(f"{out}/preview_widths.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(16, 6))       # Fig 2: alpha check
    for a, al in zip(ax, (45, 90, 135)):
        r = als[al]
        state(a, r, f"α={al}°  deployed {r['reached']:.0f}°  rupture@{None if r['rupture'] is None else round(r['rupture'])}°")
    fig.suptitle("α check (w_lig=5) — where does the strain concentrate? colour = principal strain")
    fig.tight_layout(); fig.savefig(f"{out}/preview_alpha.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))       # Fig 3: curves, fine theta
    for wl, c in zip((1.0, 5.0, 10.0), ("#F58025", "#2A9D8F", "#264653")):
        r = ws[wl]; th = r["theta_deg"]
        ax[0].plot(th, r["W"], "-o", color=c, ms=3, label=f"w_lig={wl:.0f}")
        ax[1].plot(th, r["M_theta"], "-o", color=c, ms=3, label=f"w_lig={wl:.0f}")
    ax[0].set_title("energy W [N·mm]"); ax[1].set_title("moment M_θ = dW/dθ [N·mm]")
    for a in ax:
        a.set_xlabel("fold angle θ [deg]"); a.legend(fontsize=8); a.set_xlim(0, None)
    fig.suptitle("Energy & moment vs fold angle (finely sampled from small θ)")
    fig.tight_layout(); fig.savefig(f"{out}/preview_curves.png", dpi=150); plt.close(fig)
    print("  saved preview_widths.png, preview_alpha.png, preview_curves.png")


if __name__ == "__main__":
    main()
