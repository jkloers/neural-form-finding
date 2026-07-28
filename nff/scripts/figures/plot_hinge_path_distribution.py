"""Visualize the DISTRIBUTION of hinge displacement paths over randomized starting tile positions.

Three questions, one per row:

  1. WHERE does the family of paths live? Row 1 pools every point of every path from every design
     into a density, against the box the oracle currently samples (dashed) and the box the
     measurement actually calls for (solid). The gap between the two is misdirected oracle budget.
  2. WHAT are the marginals? Row 2 is the per-component distribution of ray endpoints -- exactly
     the quantity ``DeploymentRay(theta1_deg, eta_a, eta_s)`` is drawn from.
  3. DOES THE RANDOM START MATTER? Row 3 puts the hinge geometry it induces, the straightness of
     the resulting paths, and the per-design endpoint clusters side by side. Overlapping clusters
     mean one sheet already samples the family; separated ones mean it does not.

Run (on a directory written by ``diagnostics/sample_hinge_paths.py``):
    PYTHONPATH=$(pwd) conda run -n kgnn_mac python nff/scripts/figures/plot_hinge_path_distribution.py \
        --ensemble data/outputs/hinge_paths/sheet_4x8ft_rom_ensemble_n32_s0.5
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from nff.models.hinge_surrogate import DOMAIN

ORANGE, INK, GREY, RED = "#F58025", "#1A1A1A", "#6C757D", "#D62828"


def _pad_limits(lo, hi, pad=0.08):
    m = pad * max(hi - lo, 1e-9)
    return lo - m, hi + m


def _rect(ax, xr, yr, **kw):
    ax.add_patch(Rectangle((xr[0], yr[0]), xr[1] - xr[0], yr[1] - yr[0],
                           facecolor="none", **kw))


def build_distribution_figure(ens, out_png: str, label: str = "") -> str:
    """Render the three-row distribution figure for a ``PathEnsemble``."""
    from nff.closed.hinge_path_ensemble import ensemble_statistics, sampling_spec

    st = ensemble_statistics(ens)
    spec = sampling_spec(ens)['sample_jobs_kwargs']
    pts, ends = ens.all_points(), ens.endpoints()
    pts = np.column_stack([pts[:, 0], pts[:, 1], np.degrees(pts[:, 2])])
    ends = np.column_stack([ends[:, 0], ends[:, 1], np.degrees(ends[:, 2])])

    cur = {'eta_a': (0.0, DOMAIN['eta_a_max']),
           'eta_s': (-DOMAIN['eta_s_max'], DOMAIN['eta_s_max']),
           'theta': (0.0, float(np.degrees(DOMAIN['theta_max'])))}
    # eta_a/eta_s ranges are measured over ALL path points, so they bound the density directly.
    # theta1 is measured over ENDPOINTS -- it names a ray, which then sweeps 0 -> theta1 -- so the
    # region such rays actually cover starts at 0, and that is what is drawn.
    new = {'eta_a': tuple(spec['eta_a']), 'eta_s': tuple(spec['eta_s']),
           'theta': (0.0, float(spec['theta1_deg'][1]))}
    axis_lbl = {'eta_a': r"$\eta_a=a/w_{lig}$", 'eta_s': r"$\eta_s=s/w_{lig}$",
                'theta': r"$\theta$  [deg]"}
    idx = {'eta_a': 0, 'eta_s': 1, 'theta': 2}

    fig, axes = plt.subplots(3, 3, figsize=(15.0, 12.6))

    # ── row 1: where the family lives ──
    for ax, (kx, ky) in zip(axes[0], [('eta_a', 'theta'), ('eta_s', 'theta'), ('eta_a', 'eta_s')]):
        X, Y = pts[:, idx[kx]], pts[:, idx[ky]]
        xl = _pad_limits(min(X.min(), cur[kx][0], new[kx][0]), max(X.max(), cur[kx][1], new[kx][1]))
        yl = _pad_limits(min(Y.min(), cur[ky][0], new[ky][0]), max(Y.max(), cur[ky][1], new[ky][1]))
        ax.hexbin(X, Y, gridsize=54, cmap="Oranges", bins="log", mincnt=1,
                  extent=(xl[0], xl[1], yl[0], yl[1]), linewidths=0.0, zorder=2)
        _rect(ax, cur[kx], cur[ky], edgecolor=GREY, lw=1.2, ls="--", zorder=3)
        _rect(ax, new[kx], new[ky], edgecolor=INK, lw=1.4, zorder=4)
        if kx == 'eta_a':                                  # compression is forbidden by the oracle
            ax.axvspan(xl[0], 0, color=RED, alpha=0.07, lw=0, zorder=1)
            ax.axvline(0, color=RED, lw=0.9, alpha=0.55, zorder=3)
        ax.set_xlim(*xl); ax.set_ylim(*yl)
        ax.set_xlabel(axis_lbl[kx]); ax.set_ylabel(axis_lbl[ky])
        ax.spines[['top', 'right']].set_visible(False)

    # ── row 2: the marginals a ray is drawn from ──
    for ax, k in zip(axes[1], ['eta_a', 'eta_s', 'theta']):
        v, i = ends[:, idx[k]], idx[k]
        ax.hist(pts[:, i], bins=60, color=GREY, alpha=0.35, density=True, label="all path points")
        ax.hist(v, bins=30, color=ORANGE, alpha=0.85, density=True, label="ray endpoints")
        for q, ls in [(1, ":"), (50, "-"), (99, ":")]:
            ax.axvline(np.percentile(v, q), color=INK, lw=1.0, ls=ls, alpha=0.8)
        if k == 'eta_a' and ax.get_xlim()[0] < 0:
            ax.axvline(0, color=RED, lw=0.9, alpha=0.55)
        ax.set_xlabel(axis_lbl[k]); ax.set_ylabel("density")
        ax.spines[['top', 'right']].set_visible(False)
        if k == 'eta_a':
            ax.legend(frameon=False, fontsize=8)

    # ── row 3: what the random start actually changed ──
    ax = axes[2][0]
    ax.hist(np.degrees(ens.alphas()), bins=30, color=ORANGE, alpha=0.85)
    _lo, _hi = ax.get_ylim()
    ax.vlines([30.0, 150.0], _lo, _hi, color=GREY, lw=1.2, ls="--")
    ax.set_ylim(_lo, _hi)
    ax.set_xlabel(r"hinge $\alpha$  [deg]  (RVE frame)"); ax.set_ylabel("count")
    ax.spines[['top', 'right']].set_visible(False)

    ax = axes[2][1]
    from nff.closed.hinge_paths import straightness
    strn = np.concatenate([[straightness(s.paths.eta[:, h, :]) for h in range(s.paths.n_hinges)]
                           for s in ens.samples])
    ax.hist(strn, bins=30, color=ORANGE, alpha=0.85)
    ax.set_xlabel("straightness  (0 = one proportional ray is exact)"); ax.set_ylabel("count")
    ax.spines[['top', 'right']].set_visible(False)

    ax = axes[2][2]
    cmap = plt.get_cmap("viridis")
    for j, s in enumerate(ens.samples):
        e = s.paths.eta[-1]
        ax.plot(e[:, 0], np.degrees(e[:, 2]), "o", ms=4.0, alpha=0.75,
                color=cmap(j / max(len(ens.samples) - 1, 1)), mec="none")
    ax.set_xlabel(axis_lbl['eta_a']); ax.set_ylabel(axis_lbl['theta'])
    ax.spines[['top', 'right']].set_visible(False)

    icc = st['variance_decomposition']
    handles = [
        Line2D([], [], color=GREY, lw=1.2, ls="--", label="currently sampled box"),
        Line2D([], [], color=INK, lw=1.4, label="measured box (this ensemble)"),
        Line2D([], [], color=RED, lw=6, alpha=0.25, label=r"compression ($a<0$): never sampled"),
        Line2D([], [], marker="o", ls="", color=GREY, label="one colour per random start"),
    ]
    fig.legend(handles=handles, frameon=False, ncol=len(handles), loc="lower center",
               bbox_to_anchor=(0.5, -0.012), fontsize=9)
    fig.suptitle("Distribution of hinge displacement paths over randomized starting tile positions"
                 + (f"  --  {label}" if label else ""), fontsize=12.5)
    fig.text(0.5, 0.955,
             f"{st['n_designs']} random designs x {st['n_hinges']} hinges     "
             f"ICC  $\\eta_a$ {icc['eta_a']['intraclass_corr']:.2f}   "
             f"$\\eta_s$ {icc['eta_s']['intraclass_corr']:.2f}   "
             f"$\\theta$ {icc['theta']['intraclass_corr']:.2f}",
             ha="center", fontsize=9.5, color=GREY)
    fig.tight_layout(rect=(0, 0.028, 1, 0.948))
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out_png


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ensemble", required=True, help="dir written by sample_hinge_paths.py")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    from nff.closed.hinge_path_ensemble import load_ensemble
    ens = load_ensemble(args.ensemble)
    out = args.out or os.path.join(args.ensemble, "distribution.png")
    print(f"  wrote {build_distribution_figure(ens, out, label=os.path.basename(args.ensemble))}")


if __name__ == "__main__":
    main()
