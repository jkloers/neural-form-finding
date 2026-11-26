"""Where training MOVED the hinge paths — before, after, and the difference, on one hex grid.

Three hexbins of ONE projection, sharing a single hex grid:

    before  |  after  |  DIFFERENCE (after - before)

``--projection`` chooses the plane. ``eta_a_theta`` is the deployment view: how far each hinge
opened against how far it rotated. ``eta_a_eta_s`` is the STRAIN-STATE view, and it is the one to
reach for when comparing penalties, for two reasons. Both axes carry the same dimensionless
``/ w_lig`` scale and the same order of magnitude (~1e-1), so a shift of either is legible at the
same zoom -- whereas against ``theta`` in degrees the whole eta_a signal is a thin sliver beside an
80-unit sweep. And it is the plane the loss terms act in: ``compression`` pushes the cloud right,
off ``eta_a < 0``; a shear penalty squeezes it toward ``eta_s = 0``. Their difference panels are
therefore directly comparable, and a run carrying both penalties shows whether the two moves
compete.

It is also the plane the FEA oracle samples ``(eta_a, eta_s)`` in, so what the sheet demands and
what the surrogate was trained on are drawn in the same coordinates.

The difference panel is the point. Two dense clouds side by side make the reader subtract by eye,
which is what makes a plain before/after read as messy; binning both on the same grid and plotting
the signed change does the subtraction for them. Red = paths gained, blue = paths lost, and the
shaded band is axial COMPRESSION (``eta_a < 0``), the region the sheet should be leaving.

★ EACH HEXAGON COUNTS DISTINCT PATHS CROSSING IT, not path sample points. The distinction is not
cosmetic. A deployment is sampled at ``num_load_steps`` points, and those samples are NOT uniformly
spaced along the trajectory -- a hinge that stalls mid-deployment drops many samples in one place.
Binning raw samples therefore shades by how SLOWLY a path moves, which is a property of the load
stepping, not of the mechanism. Counting distinct paths per cell answers the question actually being
asked: how many of the 275 hinges route through this region of path space.

Implementation: every trajectory is resampled densely along its own arc length (so a cell the path
crosses cannot be skipped between two coarse samples), each point carries its hinge id, and the hex
reduction counts UNIQUE ids. The three panels share one ``hexbin`` grid by construction -- the
difference is computed in a SINGLE call over the concatenated clouds, with before/after ids offset
into disjoint ranges, so the binning provably matches the reference panels rather than relying on
two grids happening to align.

Run:
    PYTHONPATH=$(pwd) JAX_PLATFORMS=cpu conda run -n kgnn_mac \
        python nff/scripts/figures/plot_path_shift.py \
            --before data/outputs/hinge_paths/<a> --after data/outputs/hinge_paths/<b> \
            --projection eta_a_eta_s --out <run_dir>/path_shift.png
"""

import argparse
import os
import numpy as np

from nff.utils.figstyle import use_agg, apply_charter, INK, GREY
use_agg()
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.colors import LogNorm                               # noqa: E402

from nff.closed.hinge_paths import load_paths                      # noqa: E402
from nff.closed.hinge_path_binning import (PROJECTIONS, densify, trajectories,  # noqa: E402
                                           n_unique_delta, ID_OFFSET, N_RESAMPLE)


GRIDSIZE = 44

_COMPRESSION_RED = "#D62828"
_SHEAR_BLUE = "#1D3557"


def _n_unique(values):
    """Hex reduction: how many DISTINCT paths put a point in this cell."""
    return float(np.unique(np.asarray(values)).size)


def n_unique_delta(values):
    """Hex reduction for the difference panel: (# after paths) - (# before paths) in this cell."""
    v = np.asarray(values)
    after = v[v >= ID_OFFSET]
    before = v[v < ID_OFFSET]
    return float(np.unique(after).size) - float(np.unique(before).size)


def _marginal_panel(ax, before, after, component, label_before, label_after, shade):
    """Overlaid path-point distribution of one eta component, before against after.

    The hexbin difference answers WHERE paths moved, but a contraction of the whole cloud and a
    retraction of one tail look alike in it. The marginal separates them: if the mean holds while
    the tail pulls in, the sheet is redistributing strain rather than simply straining less.
    """
    eb = np.asarray(before.eta)[..., component].ravel()
    ea = np.asarray(after.eta)[..., component].ravel()
    bins = np.linspace(min(eb.min(), ea.min()), max(eb.max(), ea.max()), 90)
    for values, colour, label in ((eb, GREY, label_before), (ea, "#E77500", label_after)):
        ax.hist(values, bins=bins, histtype="stepfilled", color=colour, alpha=0.45,
                label=label, log=True, linewidth=0.0)
        ax.axvline(values.mean(), color=colour, lw=1.2, ls="-", zorder=4)
    if shade:
        ax.axvspan(bins[0], 0.0, color=_COMPRESSION_RED, alpha=0.075, zorder=0)
        ax.axvline(0.0, color=_COMPRESSION_RED, lw=1.0, ls="--", zorder=3)
    ax.set_xlim(bins[0], bins[-1])


def build_path_shift_figure(before, after, out_png, label_before="before", label_after="after",
                            n_resample=N_RESAMPLE, gridsize=GRIDSIZE,
                            projection='eta_a_theta', marginals=False):
    """Three shared-grid hexbins counting distinct paths per cell: before, after, difference.

    Args:
        marginals: append a fourth column with the eta_a and eta_s distributions overlaid, and
            each one's mean drawn as a solid rule. Vertical rules that stay put while the tails
            pull in is the signature of redistribution rather than of less deformation.

    Returns:
        ``(out_png, stats)``; ``stats`` holds the compressive path-point fraction and the mean
        ``|eta_s|`` before and after, so a caller can report the headline without re-reducing.
    """
    apply_charter()
    proj = PROJECTIONS[projection]
    pb, ib = densify(before, proj, n_resample, id_offset=0)
    pa, ia = densify(after, proj, n_resample, id_offset=ID_OFFSET)

    both = np.concatenate([pb, pa], axis=0)
    ids = np.concatenate([ib, ia], axis=0)
    lo_x, hi_x = float(both[:, 0].min()), float(both[:, 0].max())
    lo_y, hi_y = float(both[:, 1].min()), float(both[:, 1].max())
    pad_x, pad_y = 0.06 * (hi_x - lo_x + 1e-9), 0.04 * (hi_y - lo_y + 1e-9)
    extent = (lo_x - pad_x, hi_x + pad_x, lo_y - pad_y, hi_y + pad_y)

    if marginals:
        fig = plt.figure(figsize=(21.5, 5.6), dpi=175)
        gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.85])
        axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
        sub = gs[0, 3].subgridspec(2, 1, hspace=0.45)
        marg_axes = [fig.add_subplot(sub[0, 0]), fig.add_subplot(sub[1, 0])]
    else:
        fig, ax_arr = plt.subplots(1, 3, figsize=(16.5, 5.6), dpi=175)
        axes, marg_axes = list(ax_arr), []

    counts = []
    for ax, pts, pid, title in ((axes[0], pb, ib, label_before), (axes[1], pa, ia, label_after)):
        hb = ax.hexbin(pts[:, 0], pts[:, 1], C=pid, reduce_C_function=_n_unique,
                       gridsize=gridsize, cmap="Oranges", mincnt=1, extent=extent,
                       linewidths=0.0, zorder=2)
        counts.append(hb)
        ax.set_title(title, fontsize=11, color=INK)
    # LOG scale on the counts. Every path starts at the origin, so that one cell holds all 275
    # hinges while the deployed tail holds a handful; on a linear scale the origin flattens
    # everything else to white and the figure says nothing about where the paths actually go.
    vmax = max(float(np.nanmax(h.get_array())) for h in counts)
    for h in counts:
        h.set_norm(LogNorm(vmin=1.0, vmax=max(vmax, 2.0)))
    cb0 = fig.colorbar(counts[1], ax=axes[1], fraction=0.043, pad=0.02)
    cb0.set_label("hinge paths crossing this cell")

    diff = axes[2].hexbin(both[:, 0], both[:, 1], C=ids, reduce_C_function=n_unique_delta,
                          gridsize=gridsize, cmap="RdBu_r", mincnt=1, extent=extent,
                          linewidths=0.0, zorder=2)
    lim = float(np.nanmax(np.abs(diff.get_array()))) or 1.0
    diff.set_clim(-lim, lim)
    cb = fig.colorbar(diff, ax=axes[2], fraction=0.043, pad=0.02)
    cb.set_label("paths gained (red) / lost (blue)")
    axes[2].set_title(f"difference:  {label_after}  −  {label_before}", fontsize=11, color=INK)

    # Both headline numbers are computed from the RAW paths, not the resampled cloud: arc-length
    # densification deliberately reweights points to make the density fair to draw, which is exactly
    # what a reported statistic must not have done to it.
    stats = {}
    for key, paths in (('before', before), ('after', after)):
        eta = np.asarray(paths.eta)
        stats[f'compressive_pct_{key}'] = float((eta[..., 0] < 0).mean()) * 100.0
        stats[f'mean_abs_eta_s_{key}'] = float(np.abs(eta[..., 1]).mean())

    for ax in axes:
        if proj.compression_band:
            ax.axvspan(extent[0], 0.0, color=_COMPRESSION_RED, alpha=0.075, zorder=1)
            ax.axvline(0.0, color=_COMPRESSION_RED, lw=1.0, ls="--", zorder=3)
        if proj.shear_line:
            ax.axhline(0.0, color=_SHEAR_BLUE, lw=1.0, ls="--", zorder=3)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_xlabel(proj.xlabel)
    axes[0].set_ylabel(proj.ylabel)

    if marg_axes:
        for ax, component, name, shade in ((marg_axes[0], 0, r"$\eta_a$", True),
                                           (marg_axes[1], 1, r"$\eta_s$", False)):
            _marginal_panel(ax, before, after, component, label_before, label_after, shade)
            ax.set_xlabel(f"{name}          (vertical rule: mean)", fontsize=9)
            ax.set_ylabel("path points", fontsize=9)
            ax.tick_params(labelsize=8)
        marg_axes[0].legend(fontsize=8, frameon=False, loc="upper right")
        marg_axes[0].set_title("marginals", fontsize=11, color=INK)

    headline =(f"compressive path points  {stats['compressive_pct_before']:.1f}%  →  "
                f"{stats['compressive_pct_after']:.1f}%")
    if proj.shear_line:
        headline += (f"      mean $|\\eta_s|$  {stats['mean_abs_eta_s_before']:.4f}  →  "
                     f"{stats['mean_abs_eta_s_after']:.4f}")
    fig.suptitle(f"Where training moved the hinge paths      {headline}",
                 fontsize=12.5, color=INK, y=1.02)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    return out_png, stats


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--before", required=True, help="dir written by extract_hinge_paths.py")
    p.add_argument("--after", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--label-before", default="before")
    p.add_argument("--label-after", default="after")
    p.add_argument("--resample", type=int, default=N_RESAMPLE)
    p.add_argument("--gridsize", type=int, default=GRIDSIZE)
    p.add_argument("--marginals", action="store_true",
                   help="add a 4th column with the eta_a / eta_s distributions overlaid")
    p.add_argument("--projection", default='eta_a_theta', choices=sorted(PROJECTIONS),
                   help="plane to draw; eta_a_eta_s is the strain-state view the loss terms act in")
    args = p.parse_args()

    before = load_paths(os.path.join(args.before, "hinge_paths"))
    after = load_paths(os.path.join(args.after, "hinge_paths"))
    out, stats = build_path_shift_figure(before, after, args.out, args.label_before,
                                         args.label_after, n_resample=args.resample,
                                         gridsize=args.gridsize, projection=args.projection,
                                         marginals=args.marginals)
    cb, ca = stats['compressive_pct_before'], stats['compressive_pct_after']
    sb, sa = stats['mean_abs_eta_s_before'], stats['mean_abs_eta_s_after']
    print(f"  {before.n_hinges} hinges x {before.n_steps} steps, resampled to {args.resample} "
          f"points/path along arc length, projection {args.projection}")
    print(f"  compressive path points: {cb:.2f}% -> {ca:.2f}%   ({ca - cb:+.2f} pts)")
    print(f"  mean |eta_s|:            {sb:.5f} -> {sa:.5f}   ({(sa - sb) / max(sb, 1e-12) * 100:+.1f}%)")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
