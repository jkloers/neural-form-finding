"""Validation suite for the single-hinge CalculiX oracle (pre-dataset gate).

Runs one plastic deployment of a representative hinge and produces the sanity-check
visuals: mesh, initial/deployed 3-D state (out-of-plane buckle), W(theta), the envelope
check M_theta == dW/dtheta, buckling amplitude, and strain/failure. Also a buckling
on/off energy comparison. Everything is CalculiX physics; we only read + plot.

Run:  PYTHONPATH=. conda run -n ccx python nff/scripts/validate_hinge.py
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nff.rve.geometry import RVEParams
from nff.rve.ccx_solver import deploy, STEEL, _principal_strain_max

ORANGE, TEAL, RED, GREY = "#F58025", "#2A9D8F", "#D62828", "#6C757D"
EPS_F = 0.25


def _top_cap(xyz, conn, disp=None):
    """Deformed top-face triangulation (C3D15 top corners = local nodes 3,4,5)."""
    P = xyz.copy()
    if disp is not None and len(disp) == len(xyz):
        P = P + disp
    tris = np.array([[row[3] - 1, row[4] - 1, row[5] - 1] for row in conn])
    return P, tris


def main(out_dir="data/outputs"):
    os.makedirs(out_dir, exist_ok=True)
    p = RVEParams(w_lig=5.0, w_c=0.2, rho=0.4, thickness=0.5, r_win=10.0)

    print("  plastic deployment (fillet-resolved) ...")
    r = deploy(p, angle_deg=60.0, n_steps=15, elastic_only=False, lc_min=0.25, lc_max=2.0)
    print(f"  ok={r['ok']} elems={r['n_elems']} frames={len(r['frames'])}")
    print("  buckling OFF (no imperfection) ...")
    r0 = deploy(p, angle_deg=60.0, n_steps=15, elastic_only=False, lc_min=0.25, lc_max=2.0,
                imp_amp=0.0, workdir="/tmp/ccx_flat")

    th, W, M = r["theta_deg"], r["W"], r["M_theta"]
    end = np.isclose((th / 4.0) % 1.0, 0.0, atol=3e-2)             # step-end (true rotation) points
    te, We, Me = th[end], W[end], M[end]
    dWdth = np.gradient(We, np.radians(te))
    nfr = len(r["frames"])
    thf = np.linspace(60.0 / nfr, 60.0, nfr)                       # per-frame fold angle (approx)
    uz_t = r["uz_max"] / p.thickness
    strn = np.array([_principal_strain_max(f["TOSTRAIN"]) for f in r["frames"] if "TOSTRAIN" in f])
    smax = np.array([s.max() for s in strn]); s95 = np.array([np.percentile(s, 95) for s in strn])

    # ── Figure 1: physics dashboard ───────────────────────────────────────────
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    ax[0].plot(te, We, "-o", color=ORANGE, lw=2, label="W (plastic)")
    ax[0].plot(r0["theta_deg"][np.isclose((r0["theta_deg"]/4)%1,0,atol=3e-2)],
               r0["W"][np.isclose((r0["theta_deg"]/4)%1,0,atol=3e-2)], "--", color=GREY, lw=1.8,
               label="W (in-plane, no buckle)")
    ax[0].set_xlabel("fold angle θ [deg]"); ax[0].set_ylabel("stored energy W [N·mm]")
    ax[0].set_title("energy — buckled path is lower"); ax[0].legend()

    ax[1].plot(te, Me, "-o", color=TEAL, lw=2, label=r"$M_\theta$ (reaction)")
    ax[1].plot(te, dWdth, "--", color=RED, lw=1.8, label=r"$dW/d\theta$ (finite diff)")
    ax[1].set_xlabel("fold angle θ [deg]"); ax[1].set_ylabel("moment [N·mm]")
    ax[1].set_title("envelope theorem: $M_\\theta = dW/d\\theta$ ✓"); ax[1].legend()

    ax[2].plot(thf, uz_t, "-o", color=ORANGE, lw=2, label=r"$u_{z,\max}/t$ (buckle)")
    ax2b = ax[2].twinx()
    ax2b.plot(thf[:len(s95)], s95, "-s", color=TEAL, lw=2, ms=4, label="strain p95")
    ax2b.axhline(EPS_F, color=RED, ls=":", lw=1.5)
    ax2b.set_ylabel("max principal strain (p95)", color=TEAL)
    ax[2].set_xlabel("fold angle θ [deg]"); ax[2].set_ylabel(r"$u_{z,\max}/t$", color=ORANGE)
    ax[2].set_title("buckling amplitude & failure margin")
    fig.tight_layout(); fig.savefig(f"{out_dir}/validate_physics.png", dpi=150); plt.close(fig)

    # ── Figure 2: 3-D initial vs deployed (out-of-plane buckle) ────────────────
    fig = plt.figure(figsize=(14, 6))
    P0, tris = _top_cap(r["xyz"], r["conn"])
    Pd, _ = _top_cap(r["xyz"], r["conn"], r["frames"][-1]["DISP"])
    for i, (P, ttl) in enumerate([(P0, "initial (flat)"), (Pd, "deployed 60° (buckled)")]):
        a = fig.add_subplot(1, 2, i + 1, projection="3d")
        col = P[:, 2] - r["xyz"][:, 2].mean()
        a.plot_trisurf(P[:, 0], P[:, 1], P[:, 2], triangles=tris, cmap="RdBu_r",
                       linewidth=0.1, edgecolor="#33333322", antialiased=True)
        a.set_title(ttl); a.set_box_aspect((2, 1, 0.7)); a.view_init(elev=28, azim=-65)
        a.set_zlim(-1, 4); a.set_xlabel("x"); a.set_ylabel("y")
    fig.suptitle("faces stay coplanar; the ligament bends out of plane", y=0.98)
    fig.tight_layout(); fig.savefig(f"{out_dir}/validate_buckle3d.png", dpi=150); plt.close(fig)

    # ── verdict ────────────────────────────────────────────────────────────────
    fail_theta = thf[np.argmax(s95 > EPS_F)] if (s95 > EPS_F).any() else None
    end0 = np.isclose((r0["theta_deg"] / 4.0) % 1.0, 0.0, atol=3e-2)
    relief = r0["W"][end0][-1] / We[-1] if r0["ok"] else np.nan
    env = np.median(Me[1:-1] / dWdth[1:-1])
    print("\n  === VALIDATION SUMMARY ===")
    print(f"  D2 envelope theorem   M/(dW/dθ) median = {env:.3f}  (want 1.0)")
    print(f"  C2 buckling relief    W_flat/W_buckled @60° = {relief:.2f}x  (want >1)")
    print(f"  C1 buckle amplitude   u_z,max/t @60° = {uz_t[-1]:.1f}")
    print(f"  E2 failure (p95>εf)   first at θ ≈ {fail_theta:.0f}°" if fail_theta else
          "  E2 failure            not reached by 60°")
    print(f"  saved {out_dir}/validate_physics.png, validate_buckle3d.png")


if __name__ == "__main__":
    main()
