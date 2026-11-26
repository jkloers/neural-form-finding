"""Several penalty arms against one baseline, in the (eta_a, eta_s) hinge-strain plane.

``plot_path_shift.py`` compares TWO designs. This compares many against a shared reference, which
is what a penalty ablation needs: `compression` alone, `shear_path` alone, and both together all
read against the same shape-only run, on ONE hex grid and ONE colour scale, so the panels are
directly comparable rather than each self-normalised.

    row 1   difference hexbin per arm:  arm - baseline, red gained / blue lost
    row 2   marginals of eta_a and eta_s, every arm overlaid on the baseline

The two rows answer different questions and both are needed. The hexbins say WHERE in the strain
plane each penalty moved the paths. The marginals say whether that was a redistribution or simply
less deformation -- a penalty that shrinks the whole cloud drags the MEAN with it, while one that
retracts a tail leaves the mean where it was. Reading only the hexbins cannot tell those apart.

★ The compression fraction and mean |eta_s| are computed from the RAW paths, never from the
arc-length-resampled cloud the hexbins are drawn from. Resampling deliberately reweights points to
make the density fair to LOOK at, which is exactly what a reported statistic must not have had done
to it.

Run:
    PYTHONPATH=$(pwd) JAX_PLATFORMS=cpu conda run -n kgnn_mac \
        python nff/scripts/figures/plot_path_arms.py \
            --baseline data/outputs/hinge_paths/d70_shape:"shape only" \
            --arm data/outputs/hinge_paths/d70_comp:"+ compression" \
            --arm data/outputs/hinge_paths/d70_shear:"+ shear" \
            --arm data/outputs/hinge_paths/d70_both:"+ both" \
            --out data/outputs/path_arms_d70.png
"""

import argparse
import os

import numpy as np

from nff.utils.figstyle import use_agg, apply_charter, INK, GREY, ORANGE, TEAL, PURPLE
use_agg()
import matplotlib.pyplot as plt                                    # noqa: E402

from nff.closed.hinge_paths import load_paths                      # noqa: E402
from nff.scripts.figures.plot_path_shift import (PROJECTIONS, densify, n_unique_delta,  # noqa: E402
                                                 ID_OFFSET, GRIDSIZE, N_RESAMPLE)


ARM_COLOURS = (ORANGE, TEAL, PURPLE, "#C1121F")
_COMPRESSION_RED = "#D62828"


def _stats(paths):
    """Headline numbers straight off the raw trajectories."""
    eta = np.asarray(paths.eta)
    return {
        'compressive_pct': float((eta[..., 0] < 0).mean()) * 100.0,
        'mean_abs_eta_s': float(np.abs(eta[..., 1]).mean()),
        'mean_eta_a': float(eta[..., 0].mean()),
        'p1_eta_a': float(np.percentile(eta[..., 0], 1.0)),
        'max_theta_deg': float(np.degrees(eta[..., 2]).max()),
    }


def build_arms_figure(baseline, baseline_label, arms, out_png,
                      n_resample=N_RESAMPLE, gridsize=GRIDSIZE):
    """Difference hexbins plus overlaid marginals for every arm against one baseline.

    Args:
        baseline: reference ``HingePaths``.
        baseline_label: its legend name.
        arms: sequence of ``(HingePaths, label)``.
        out_png: destination.

    Returns:
        ``(out_png, stats)`` with ``stats`` keyed by label.
    """
    apply_charter()
    proj = PROJECTIONS['eta_a_eta_s']
    n = len(arms)

    pb, ib = densify(baseline, proj, n_resample, id_offset=0)
    dense = [densify(p, proj, n_resample, id_offset=ID_OFFSET) for p, _ in arms]

    # ONE extent over every arm, so the panels share a grid and a cell means the same thing in all.
    allpts = np.concatenate([pb] + [d[0] for d in dense], axis=0)
    lo_x, hi_x = float(allpts[:, 0].min()), float(allpts[:, 0].max())
    lo_y, hi_y = float(allpts[:, 1].min()), float(allpts[:, 1].max())
    pad_x, pad_y = 0.06 * (hi_x - lo_x + 1e-9), 0.04 * (hi_y - lo_y + 1e-9)
    extent = (lo_x - pad_x, hi_x + pad_x, lo_y - pad_y, hi_y + pad_y)

    # The two rows are laid out INDEPENDENTLY: the top row has one panel per arm, the bottom row
    # always has exactly three (eta_a marginal, eta_s marginal, summary table). Deriving the bottom
    # row from `n` breaks the moment a single arm is plotted.
    width = max(5.4 * n, 16.2)
    fig = plt.figure(figsize=(width, 10.2), dpi=170)
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.72], hspace=0.26)
    gs_top = outer[0].subgridspec(1, n, wspace=0.22)
    gs_bot = outer[1].subgridspec(1, 3, width_ratios=[1.0, 1.0, 1.25], wspace=0.28)
    top = [fig.add_subplot(gs_top[0, i]) for i in range(n)]

    hexes = []
    for ax, (pts, pid), (_, label) in zip(top, dense, arms):
        both = np.concatenate([pb, pts], axis=0)
        ids = np.concatenate([ib, pid], axis=0)
        hb = ax.hexbin(both[:, 0], both[:, 1], C=ids, reduce_C_function=n_unique_delta,
                       gridsize=gridsize, cmap="RdBu_r", mincnt=1, extent=extent,
                       linewidths=0.0, zorder=2)
        hexes.append(hb)
        ax.set_title(f"{label}  −  {baseline_label}", fontsize=11, color=INK)
        ax.axvspan(extent[0], 0.0, color=_COMPRESSION_RED, alpha=0.075, zorder=1)
        ax.axvline(0.0, color=_COMPRESSION_RED, lw=1.0, ls="--", zorder=3)
        ax.axhline(0.0, color="#1D3557", lw=1.0, ls="--", zorder=3)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_xlabel(proj.xlabel)
    top[0].set_ylabel(proj.ylabel)

    # A SHARED symmetric colour scale. Per-panel autoscaling would make a weak arm look as strong
    # as a strong one, which is the single easiest way to misread an ablation.
    lim = max(float(np.nanmax(np.abs(h.get_array()))) for h in hexes) or 1.0
    for h in hexes:
        h.set_clim(-lim, lim)
    cb = fig.colorbar(hexes[-1], ax=top[-1], fraction=0.043, pad=0.02)
    cb.set_label("paths gained (red) / lost (blue)")

    # ── marginals, all arms overlaid on the baseline ──────────────────────────────────────────
    marg = [fig.add_subplot(gs_bot[0, 0]), fig.add_subplot(gs_bot[0, 1])]
    series =[(baseline, baseline_label, GREY)] + \
             [(p, lab, ARM_COLOURS[i % len(ARM_COLOURS)]) for i, (p, lab) in enumerate(arms)]

    for ax, component, name, shade in ((marg[0], 0, r"$\eta_a$", True),
                                       (marg[1], 1, r"$\eta_s$", False)):
        lo = min(np.asarray(p.eta)[..., component].min() for p, _, _ in series)
        hi = max(np.asarray(p.eta)[..., component].max() for p, _, _ in series)
        bins = np.linspace(lo, hi, 110)
        for paths, label, colour in series:
            values = np.asarray(paths.eta)[..., component].ravel()
            ax.hist(values, bins=bins, histtype="step", color=colour, lw=1.5,
                    label=label, log=True)
        if shade:
            ax.axvspan(lo, 0.0, color=_COMPRESSION_RED, alpha=0.075, zorder=0)
            ax.axvline(0.0, color=_COMPRESSION_RED, lw=1.0, ls="--", zorder=3)
        ax.set_xlim(lo, hi)
        ax.set_xlabel(name)
        ax.set_ylabel("path points")
    marg[0].legend(fontsize=9, frameon=False, loc="upper right")

    stats = {baseline_label: _stats(baseline)}
    for paths, label in arms:
        stats[label] = _stats(paths)

    # The numbers, on the figure, next to the picture they summarise. `mean eta_a` and `theta max`
    # are the controls: a penalty that improved the strain field only by deploying less shows up
    # here as a mean or an angle that moved with it.
    table = fig.add_subplot(gs_bot[0, 2])
    table.axis("off")
    rows = [f"{'arm':<20}{'compressive':>12}{'mean|eta_s|':>13}{'mean eta_a':>12}{'theta max':>11}",
            "-" * 68]
    for label, s in stats.items():
        rows.append(f"{label:<20}{s['compressive_pct']:>11.1f}%{s['mean_abs_eta_s']:>13.5f}"
                    f"{s['mean_eta_a']:>12.5f}{s['max_theta_deg']:>10.2f}°")
    table.text(0.0, 0.95, "\n".join(rows), family="monospace", fontsize=9.5,
               va="top", color=INK, transform=table.transAxes)

    fig.suptitle("Penalty arms in the hinge-strain plane", fontsize=13.5, color=INK, y=0.98)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    return out_png, stats


def _split(spec):
    """``dir:label`` -> ``(HingePaths, label)``; label defaults to the directory name."""
    path, _, label = spec.partition(":")
    return load_paths(os.path.join(path, "hinge_paths")), (label or os.path.basename(path))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--baseline", required=True, help="dir[:label] of the reference run")
    p.add_argument("--arm", action="append", required=True, help="dir[:label], repeatable")
    p.add_argument("--out", required=True)
    p.add_argument("--resample", type=int, default=N_RESAMPLE)
    p.add_argument("--gridsize", type=int, default=GRIDSIZE)
    args = p.parse_args()

    base, base_label = _split(args.baseline)
    arms = [_split(s) for s in args.arm]
    out, stats = build_arms_figure(base, base_label, arms, args.out,
                                   n_resample=args.resample, gridsize=args.gridsize)

    ref = stats[base_label]
    print(f"  baseline: {base_label}")
    print(f"  {'arm':<24}{'compressive':>12}{'delta':>9}{'mean|eta_s|':>13}{'delta':>9}")
    for label, s in stats.items():
        d_c = s['compressive_pct'] - ref['compressive_pct']
        d_s = (s['mean_abs_eta_s'] - ref['mean_abs_eta_s']) / max(ref['mean_abs_eta_s'], 1e-12) * 100
        print(f"  {label:<24}{s['compressive_pct']:>11.2f}%{d_c:>+9.2f}"
              f"{s['mean_abs_eta_s']:>13.5f}{d_s:>+8.1f}%")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
