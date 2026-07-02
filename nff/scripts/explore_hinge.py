"""Exploration: pivot sweep (energy-minimising rotation centre), top-down views, and a
parameter variation (fold angle + strip width). Answers: where is the rotation centre,
is the near-tip region stretching real, and how do energy/strain scale with geometry.

Run:  PYTHONPATH=. conda run -n ccx python nff/scripts/explore_hinge.py
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

from nff.rve.geometry import RVEParams
from nff.rve.ccx_solver import deploy


def _topcap(xyz, conn, disp=None):
    P = xyz + disp if disp is not None and len(disp) == len(xyz) else xyz.copy()
    tris = np.array([[r[3] - 1, r[4] - 1, r[5] - 1] for r in conn])
    return P, tris


def _top_view(ax, xyz, conn, disp, pivot, title, vmax):
    P, tris = _topcap(xyz, conn, disp)
    uz = P[:, 2] - xyz[:, 2].mean()
    tri = Triangulation(P[:, 0], P[:, 1], tris)
    tpc = ax.tripcolor(tri, uz, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="gouraud")
    ax.plot(pivot[0], pivot[1], "k*", ms=13, zorder=5)                # rotation centre
    ax.set_aspect("equal"); ax.set_title(title, fontsize=10); ax.set_xlabel("x [mm]")
    return tpc


def main(out_dir="data/outputs"):
    os.makedirs(out_dir, exist_ok=True)

    # ── Part 1: pivot sweep (elastic + NLGEOM, fast) — find the energy-minimising centre ──
    p = RVEParams(w_lig=5.0, w_c=0.2, rho=0.4, thickness=0.5, r_win=10.0)
    fracs = [0.15, 0.35, 0.5, 0.7, 0.9]                               # fraction of w_lig below secondary
    runs, Wend = [], []
    for fr in fracs:
        r = deploy(p, angle_deg=40.0, n_steps=8, elastic_only=True, pivot=(0.0, -fr * p.w_lig),
                   lc_min=0.35, lc_max=2.0, workdir=f"/tmp/ccx_piv{int(fr*100)}")
        runs.append(r); Wend.append(r["W"][-1])
        print(f"  pivot y=-{fr*p.w_lig:.2f} ({fr:.2f}w_lig)  W(40deg)={r['W'][-1]:.0f} N.mm  "
              f"uz/t={r['uz_max'][-1]/p.thickness:.1f}")
    imin = int(np.argmin(Wend))
    print(f"  ENERGY-MIN pivot: {fracs[imin]:.2f}*w_lig  (tip=1.0, secondary=0.0)")

    fig, ax = plt.subplots(1, len(fracs), figsize=(4 * len(fracs), 4.2))
    vmax = max(np.abs(r["frames"][-1]["DISP"][:, 2]).max() for r in runs)
    for a, fr, r in zip(ax, fracs, runs):
        tpc = _top_view(a, r["xyz"], r["conn"], r["frames"][-1]["DISP"], r["pivot"],
                        f"pivot {fr:.2f}·w_lig  |  W={r['W'][-1]:.0f}"
                        + ("  ← min" if fr == fracs[imin] else ""), vmax)
    ax[0].set_ylabel("y [mm]")
    fig.colorbar(tpc, ax=ax, label="out-of-plane $u_z$ [mm]", fraction=0.02)
    fig.suptitle("pivot sweep — top view of the deployed hinge (★ = rotation centre), elastic 40°")
    fig.savefig(f"{out_dir}/hinge_pivot_sweep.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # ── Part 2: top-down initial vs deployed at the energy-min pivot (plastic 60°) ──
    piv = (0.0, -fracs[imin] * p.w_lig)
    rp = deploy(p, angle_deg=60.0, n_steps=15, elastic_only=False, pivot=piv,
                lc_min=0.3, lc_max=2.0, workdir="/tmp/ccx_base")
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    vmax = np.abs(rp["frames"][-1]["DISP"][:, 2]).max()
    _top_view(ax[0], rp["xyz"], rp["conn"], None, piv, "initial (flat, top view)", vmax)
    tpc = _top_view(ax[1], rp["xyz"], rp["conn"], rp["frames"][-1]["DISP"], piv,
                    "deployed 60° (top view, colour = out-of-plane)", vmax)
    ax[0].set_ylabel("y [mm]"); fig.colorbar(tpc, ax=ax, label="$u_z$ [mm]", fraction=0.03)
    fig.suptitle(f"top-down view, energy-min pivot at y=-{fracs[imin]*p.w_lig:.1f} mm (★)")
    fig.savefig(f"{out_dir}/hinge_topview.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # ── Part 3: parameter variation — strip width & fold angle ──
    print("\n  parameter variation (plastic):")
    cases = [("w_lig=5, 90deg", RVEParams(w_lig=5, w_c=0.2, rho=0.4, thickness=0.5, r_win=10), 90.0, (0, -fracs[imin]*5)),
             ("w_lig=8, 90deg", RVEParams(w_lig=8, w_c=0.2, rho=0.4, thickness=0.5, r_win=16), 90.0, (0, -fracs[imin]*8))]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    for name, pc, ang, pv in cases:
        r = deploy(pc, angle_deg=ang, n_steps=18, elastic_only=False, pivot=pv,
                   lc_min=0.3, lc_max=pc.r_win/5, workdir=f"/tmp/ccx_{pc.w_lig:.0f}")
        end = np.isclose((r["theta_deg"]/(ang/18)) % 1.0, 0.0, atol=3e-2)
        s95 = np.array([np.percentile(_pstrain(f["TOSTRAIN"]), 95) for f in r["frames"] if "TOSTRAIN" in f])
        thf = np.linspace(ang/len(r["frames"]), ang, len(r["frames"]))
        ax[0].plot(r["theta_deg"][end], r["W"][end], "-o", lw=2, label=name)
        ax[1].plot(thf[:len(s95)], s95, "-", lw=2, label=name)
        print(f"  {name}: W(end)={r['W'][end][-1]:.0f} N.mm  uz/t={r['uz_max'][-1]/pc.thickness:.1f}  "
              f"p95 strain end={s95[-1]:.2f}")
    ax[0].set_xlabel("θ [deg]"); ax[0].set_ylabel("W [N·mm]"); ax[0].set_title("stored energy"); ax[0].legend()
    ax[1].axhline(0.25, color="#D62828", ls=":", lw=1.5, label="ε_f (fracture)")
    ax[1].set_xlabel("θ [deg]"); ax[1].set_ylabel("p95 principal strain"); ax[1].set_title("strain / failure"); ax[1].legend()
    fig.suptitle("parameter variation: wider strip → lower strain, later failure")
    fig.savefig(f"{out_dir}/hinge_param_variation.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"\n  saved hinge_pivot_sweep.png, hinge_topview.png, hinge_param_variation.png")


def _pstrain(tostrain):
    out = np.zeros(len(tostrain))
    for i, (xx, yy, zz, xy, yz, zx) in enumerate(tostrain):
        out[i] = np.linalg.eigvalsh(np.array([[xx, xy, zx], [xy, yy, yz], [zx, yz, zz]]))[-1]
    return out


if __name__ == "__main__":
    main()
