"""Show what the campaign will actually sample, against the region the sheet actually visits.

Three projections of the physical hinge motion (a, s, theta in mm/mm/deg), with the measured cloud
behind and the sampled jobs on top, plus the geometry marginals. The point is to see, before
spending the compute, whether the design covers the compression- and shear-dominated corners the
legacy eta box excluded.

    conda run -n kgnn_mac python -m nff.scripts.figures.plot_campaign_design \
        --prior data/fea/path_priors/envelope_v3_w18 --n 800 --out data/outputs/campaign_design.png
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from nff.rve.path_prior import measure_envelope, sample_campaign_jobs
from nff.utils.figstyle import GREY, INK, ORANGE, RED, TEAL, TRAIN, apply_charter, use_agg

use_agg()
import matplotlib.pyplot as plt                                          # noqa: E402
from matplotlib.patches import Rectangle                                 # noqa: E402


def measured_cloud(prior_dir):
    """Every measured path point, in physical mm/deg."""
    npz = np.load(os.path.join(prior_dir, "paths.npz"))
    A = {k: npz[k] for k in npz.files}
    w = float(json.load(open(os.path.join(prior_dir, "manifest.json")))["w_lig_mm"])
    eta = A["eta"]
    mask = A["hinge_mask"] if "hinge_mask" in A else np.ones(A["alpha"].shape, bool)
    pts = eta[np.broadcast_to(mask[:, None, :], eta.shape[:3])]
    return pts[:, 0] * w, pts[:, 1] * w, np.degrees(pts[:, 2])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prior", required=True)
    ap.add_argument("--n", type=int, default=800, help="campaign size to preview")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--w-lig-min", type=float, default=5.0)
    ap.add_argument("--w-lig-max", type=float, default=50.0)
    ap.add_argument("--spine-frac", type=float, default=0.25)
    ap.add_argument("--inflate", type=float, default=1.5)
    ap.add_argument("--out", default="data/outputs/campaign_design.png")
    args = ap.parse_args()

    env = measure_envelope(args.prior)
    jobs = sample_campaign_jobs(args.n, env, seed=args.seed,
                                w_lig=(args.w_lig_min, args.w_lig_max),
                                spine_frac=args.spine_frac, inflate=args.inflate)
    spine = [(g, r) for g, r in jobs if r.free_dofs]
    fan = [(g, r) for g, r in jobs if not r.free_dofs]
    fa = np.array([r.eta_a * g.w_lig for g, r in fan])
    fs = np.array([r.eta_s * g.w_lig for g, r in fan])
    ft = np.array([r.theta1_deg for _, r in fan])
    st = np.array([r.theta1_deg for _, r in spine])
    ma, ms, mt = measured_cloud(args.prior)

    apply_charter()
    fig, axes = plt.subplots(1, 4, figsize=(17.0, 4.3))
    panels = [(0, ma, mt, fa, ft, "axial  a  [mm]", "rotation  θ  [deg]", env.a, env.theta_deg),
              (1, ms, mt, fs, ft, "shear  s  [mm]", "rotation  θ  [deg]", env.s, env.theta_deg),
              (2, ma, ms, fa, fs, "axial  a  [mm]", "shear  s  [mm]", env.a, env.s)]
    for k, mx, my, sx, sy, xl, yl, bx, by in panels:
        ax = axes[k]
        ax.scatter(mx, my, s=1, c=TRAIN, alpha=0.35, linewidths=0, rasterized=True,
                   label="measured (ROM)" if k == 0 else None)
        ax.add_patch(Rectangle((bx[0], by[0]), bx[1] - bx[0], by[1] - by[0],
                               fill=False, ec=INK, lw=1.0, ls="--",
                               label="envelope" if k == 0 else None))
        ax.scatter(sx, sy, s=9, c=ORANGE, alpha=0.8, linewidths=0,
                   label="campaign fan" if k == 0 else None)
        if k < 2:
            # Spine jobs prescribe theta ONLY; (a, s) are whatever minimises energy, so they are
            # drawn on the axis as a reminder of their theta coverage, not as a claimed position.
            # Measured: the solver picks a = -4.1 mm at a 28 deg fold, i.e. well into compression.
            ax.scatter(np.zeros_like(st), st, s=14, c=TEAL, alpha=0.8, linewidths=0, marker="_",
                       label="spine: θ driven, a & s solved" if k == 0 else None)
        ax.axvline(0.0, color=GREY, lw=0.8, alpha=0.5)
        ax.set_xlabel(xl); ax.set_ylabel(yl)
    axes[0].legend(fontsize=8, loc="upper left", markerscale=2.5)
    axes[0].annotate("spine lands where the solver puts it\n(a ≈ −4 mm at 28°), not at a = 0",
                     xy=(0.0, float(np.median(st)) if len(st) else 0.0),
                     xytext=(0.42, 0.06), textcoords="axes fraction", fontsize=7.5, color=TEAL,
                     ha="left", arrowprops=dict(arrowstyle="->", color=TEAL, lw=0.8))

    ax = axes[3]
    w = np.array([g.w_lig for g, _ in jobs])
    al = np.array([g.alpha_deg for g, _ in jobs])
    ax.scatter(w, al, s=9, c=RED, alpha=0.75, linewidths=0)
    ax.set_xscale("log")
    ax.set_xlabel("ligament width  w$_{lig}$  [mm]  (log-uniform)")
    ax.set_ylabel("cut angle  α  [deg]")

    frac_c = float((fa < 0).mean())
    frac_s = float((np.abs(fs) > np.abs(fa)).mean())
    fig.text(0.5, 0.975,
             f"campaign design — {len(jobs)} jobs  ·  {len(spine)} free-DOF spine + {len(fan)} fan  ·  "
             f"{frac_c:.0%} compressive, {frac_s:.0%} shear-dominated",
             ha="center", fontsize=11, color=INK)
    fig.text(0.5, 0.935,
             f"envelope from {env.source}  ({env.n_points:,} measured points, p0.5–p99.5)  ·  "
             f"dashed box = measured envelope; points outside it are the ×{args.inflate:g} margin",
             ha="center", fontsize=8.5, color=GREY)
    fig.tight_layout(rect=(0, 0, 1, 0.915))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=180)
    plt.close(fig)

    print(f"envelope  a [{env.a[0]:+.2f}, {env.a[1]:+.2f}] mm   s [{env.s[0]:+.2f}, {env.s[1]:+.2f}] mm"
          f"   theta [0, {env.theta_deg[1]:.1f}] deg   alpha [{env.alpha_deg[0]:.0f}, {env.alpha_deg[1]:.0f}]")
    print(f"measured  a [{ma.min():+.2f}, {ma.max():+.2f}]   s [{ms.min():+.2f}, {ms.max():+.2f}]"
          f"   theta [{mt.min():.1f}, {mt.max():.1f}]   ({len(ma):,} points, {100*(ma<0).mean():.1f}% compressive)")
    print(f"sampled   a [{fa.min():+.2f}, {fa.max():+.2f}]   s [{fs.min():+.2f}, {fs.max():+.2f}]"
          f"   theta [{ft.min():.1f}, {ft.max():.1f}]   ({frac_c:.0%} compressive, {frac_s:.0%} shear-dominated)")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
