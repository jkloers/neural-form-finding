"""The ridden ``(eta_a, eta_s, theta)`` manifold in 3D, against the box the oracle samples.

The 2D projections in ``plot_hinge_path_distribution`` each collapse one axis, which hides the
thing that actually matters for aiming the oracle: the ridden set is a thin, curved SHEET inside a
box that is sampled as a solid. Volume ratios are the argument for re-aiming the sampling, and a
volume is what this draws.

Four panels: the pooled point cloud from three viewing angles (coloured by local 3D density, the
same log-count encoding as the hexbins), plus the trajectories themselves for a few designs so the
paths read as curves rather than as a cloud. Both boxes are drawn as wireframes -- dashed grey for
what ``sample_jobs`` covers today, solid ink for what this ensemble measured -- and the forbidden
compressive half-space ``a < 0`` is a translucent red plane.

Run (on a directory written by ``diagnostics/sample_hinge_paths.py``):
    PYTHONPATH=$(pwd) conda run -n kgnn_mac python nff/scripts/figures/plot_hinge_path_manifold_3d.py \
        --ensemble data/outputs/hinge_paths/sheet_4x8ft_rom_ensemble_n64_s0.5_load0.1

``--spin`` additionally writes a rotating GIF, which is far more legible than any static view for
reading the sheet's curvature -- useful for exploration, not for the paper.
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)

from nff.models.hinge_surrogate import DOMAIN

ORANGE, INK, GREY, RED = "#F58025", "#1A1A1A", "#6C757D", "#D62828"
VIEWS = [(20, -62), (16, 24), (62, -50)]


def point_density(P: np.ndarray, bins: int = 26) -> np.ndarray:
    """Count of neighbours in each point's own 3D bin -- the hexbin encoding, one dimension up.

    A histogram lookup rather than a kernel estimate: with tens of thousands of points a KDE is
    both slow and smoother than the structure we are trying to see (the sheet is thin, and
    smoothing across it would fill in exactly the void that is the point of the figure).
    """
    H, edges = np.histogramdd(P, bins=bins)
    idx = [np.clip(np.digitize(P[:, k], edges[k]) - 1, 0, H.shape[k] - 1) for k in range(3)]
    return H[idx[0], idx[1], idx[2]]


def _wire_box(ax, xr, yr, zr, **kw):
    """The 12 edges of an axis-aligned box."""
    (x0, x1), (y0, y1), (z0, z1) = xr, yr, zr
    for a, b in [((x0, y0, z0), (x1, y0, z0)), ((x0, y1, z0), (x1, y1, z0)),
                 ((x0, y0, z1), (x1, y0, z1)), ((x0, y1, z1), (x1, y1, z1)),
                 ((x0, y0, z0), (x0, y1, z0)), ((x1, y0, z0), (x1, y1, z0)),
                 ((x0, y0, z1), (x0, y1, z1)), ((x1, y0, z1), (x1, y1, z1)),
                 ((x0, y0, z0), (x0, y0, z1)), ((x1, y0, z0), (x1, y0, z1)),
                 ((x0, y1, z0), (x0, y1, z1)), ((x1, y1, z0), (x1, y1, z1))]:
        ax.plot(*zip(a, b), **kw)


def _style(ax, lims):
    ax.set_xlabel(r"$\eta_a = a/w_{lig}$", labelpad=2, fontsize=9)
    ax.set_ylabel(r"$\eta_s = s/w_{lig}$", labelpad=2, fontsize=9)
    ax.set_zlabel(r"$\theta$  [deg]", labelpad=2, fontsize=9)
    ax.set_xlim(*lims[0]); ax.set_ylim(*lims[1]); ax.set_zlim(*lims[2])
    ax.tick_params(labelsize=7.5)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_alpha(0.0)
        pane._axinfo["grid"].update(color="#DDDDDD", linewidth=0.5)


def _draw_boxes(ax, cur, new, lims):
    _wire_box(ax, *cur, color=GREY, lw=1.0, ls="--", alpha=0.85, zorder=1)
    _wire_box(ax, *new, color=INK, lw=1.2, alpha=0.9, zorder=2)
    if lims[0][0] < 0:                       # the a<0 half-space the oracle forbids
        yy, zz = np.meshgrid(np.linspace(*lims[1], 2), np.linspace(*lims[2], 2))
        ax.plot_surface(np.zeros_like(yy), yy, zz, color=RED, alpha=0.10, shade=False, zorder=0)


def build_manifold_3d_figure(ens, out_png: str, label: str = "", max_points: int = 14000,
                             n_traj_designs: int = 4, seed: int = 0) -> str:
    """Render the four-panel 3D view of the ridden manifold for a ``PathEnsemble``."""
    from nff.closed.hinge_path_ensemble import ensemble_statistics, sampling_spec

    st = ensemble_statistics(ens)
    spec = sampling_spec(ens)['sample_jobs_kwargs']
    P = ens.all_points()
    P = np.column_stack([P[:, 0], P[:, 1], np.degrees(P[:, 2])])

    cur = ((DOMAIN.get('eta_a_min', 0.0), DOMAIN['eta_a_max']), (-DOMAIN['eta_s_max'], DOMAIN['eta_s_max']),
           (float(np.degrees(DOMAIN.get('theta_min', 0.0))), float(np.degrees(DOMAIN['theta_max']))))
    new = (tuple(spec['eta_a']), tuple(spec['eta_s']), (0.0, float(spec['theta1_deg'][1])))
    lims = [(min(P[:, k].min(), cur[k][0], new[k][0]), max(P[:, k].max(), cur[k][1], new[k][1]))
            for k in range(3)]
    lims = [(lo - 0.04 * (hi - lo), hi + 0.04 * (hi - lo)) for lo, hi in lims]

    dens = point_density(P)
    keep = np.arange(len(P))
    if len(P) > max_points:                  # thin for rendering, deterministically
        keep = np.random.default_rng(seed).choice(len(P), max_points, replace=False)
    order = keep[np.argsort(dens[keep])]     # dense points drawn last, so they read on top
    Q, c = P[order], np.log10(dens[order] + 1.0)

    fig = plt.figure(figsize=(15.0, 12.2))
    for i, (elev, azim) in enumerate(VIEWS):
        ax = fig.add_subplot(2, 2, i + 1, projection="3d")
        ax.scatter(Q[:, 0], Q[:, 1], Q[:, 2], c=c, cmap="Oranges", s=2.6, alpha=0.55,
                   linewidths=0.0, vmin=0.0, zorder=3)
        _draw_boxes(ax, cur, new, lims)
        _style(ax, lims)
        ax.view_init(elev=elev, azim=azim)

    # trajectories: the same manifold read as curves, which is where reversal and snap are visible
    ax = fig.add_subplot(2, 2, 4, projection="3d")
    cmap = plt.get_cmap("viridis")
    picks = np.linspace(0, len(ens.samples) - 1, min(n_traj_designs, len(ens.samples))).astype(int)
    for j, si in enumerate(picks):
        e = ens.samples[si].paths.eta
        col = cmap(j / max(len(picks) - 1, 1))
        for h in range(e.shape[1]):
            ax.plot(e[:, h, 0], e[:, h, 1], np.degrees(e[:, h, 2]), "-", color=col, lw=1.0,
                    alpha=0.8, zorder=3)
        ax.scatter(e[-1, :, 0], e[-1, :, 1], np.degrees(e[-1, :, 2]), color=col, s=16,
                   edgecolors="white", linewidths=0.5, zorder=4)
    _draw_boxes(ax, cur, new, lims)
    _style(ax, lims)
    ax.view_init(elev=VIEWS[0][0], azim=VIEWS[0][1])

    handles = [Line2D([], [], color=GREY, lw=1.2, ls="--", label="currently sampled box"),
               Line2D([], [], color=INK, lw=1.4, label="measured box (this ensemble)"),
               Line2D([], [], color=RED, lw=6, alpha=0.25, label=r"compression ($a<0$): never sampled"),
               Line2D([], [], marker="o", ls="", color=ORANGE, label="path point (shade = local density)"),
               Line2D([], [], color=cmap(0.5), lw=1.4, label="trajectories, one colour per design")]
    fig.legend(handles=handles, frameon=False, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, 0.005), fontsize=9)
    fig.suptitle("The ridden hinge-displacement manifold in 3D"
                 + (f"  --  {label}" if label else ""), fontsize=13)
    fig.text(0.5, 0.955,
             f"{st['n_designs']} random designs x {st['n_hinges']} hinges     "
             f"{st['n_path_points']} path points     "
             f"{100 * st['in_domain_frac']:.0f}% inside the sampled box",
             ha="center", fontsize=9.5, color=GREY)
    # subplots_adjust rather than tight_layout: 3D axis labels are drawn outside the axes bbox,
    # which tight_layout does not account for, and the z-label of the right-hand panels gets
    # clipped at the figure edge.
    fig.subplots_adjust(left=0.01, right=0.93, top=0.93, bottom=0.07, wspace=0.02, hspace=0.02)
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    # not bbox_inches="tight": it measures artists and misses 3D axis labels, re-clipping the z-label
    fig.savefig(out_png, dpi=170)
    plt.close(fig)
    return out_png


def build_spin_gif(ens, out_gif: str, n_frames: int = 60, max_points: int = 9000,
                   seed: int = 0) -> str:
    """A rotating single-panel view. The sheet's curvature is only really readable in motion."""
    from matplotlib.animation import FuncAnimation, PillowWriter
    from nff.closed.hinge_path_ensemble import sampling_spec

    spec = sampling_spec(ens)['sample_jobs_kwargs']
    P = ens.all_points()
    P = np.column_stack([P[:, 0], P[:, 1], np.degrees(P[:, 2])])
    cur = ((DOMAIN.get('eta_a_min', 0.0), DOMAIN['eta_a_max']), (-DOMAIN['eta_s_max'], DOMAIN['eta_s_max']),
           (float(np.degrees(DOMAIN.get('theta_min', 0.0))), float(np.degrees(DOMAIN['theta_max']))))
    new = (tuple(spec['eta_a']), tuple(spec['eta_s']), (0.0, float(spec['theta1_deg'][1])))
    lims = [(min(P[:, k].min(), cur[k][0], new[k][0]), max(P[:, k].max(), cur[k][1], new[k][1]))
            for k in range(3)]
    lims = [(lo - 0.04 * (hi - lo), hi + 0.04 * (hi - lo)) for lo, hi in lims]

    dens = point_density(P)
    keep = np.arange(len(P))
    if len(P) > max_points:
        keep = np.random.default_rng(seed).choice(len(P), max_points, replace=False)
    order = keep[np.argsort(dens[keep])]
    Q, c = P[order], np.log10(dens[order] + 1.0)

    fig = plt.figure(figsize=(7.4, 6.6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(Q[:, 0], Q[:, 1], Q[:, 2], c=c, cmap="Oranges", s=3.0, alpha=0.6,
               linewidths=0.0, vmin=0.0, zorder=3)
    _draw_boxes(ax, cur, new, lims)
    _style(ax, lims)

    def _frame(i):
        ax.view_init(elev=18 + 10 * np.sin(2 * np.pi * i / n_frames), azim=-180 + 360 * i / n_frames)
        return ()

    os.makedirs(os.path.dirname(out_gif) or ".", exist_ok=True)
    FuncAnimation(fig, _frame, frames=n_frames, blit=False).save(
        out_gif, writer=PillowWriter(fps=18), dpi=105)
    plt.close(fig)
    return out_gif


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ensemble", required=True, help="dir written by sample_hinge_paths.py")
    p.add_argument("--out", default=None)
    p.add_argument("--spin", action="store_true", help="also write a rotating GIF")
    p.add_argument("--max-points", type=int, default=14000)
    args = p.parse_args()

    from nff.closed.hinge_path_ensemble import load_ensemble
    ens = load_ensemble(args.ensemble)
    out = args.out or os.path.join(args.ensemble, "manifold_3d.png")
    print(f"  wrote {build_manifold_3d_figure(ens, out, label=os.path.basename(args.ensemble), max_points=args.max_points)}")
    if args.spin:
        print(f"  wrote {build_spin_gif(ens, os.path.splitext(out)[0] + '_spin.gif')}")


if __name__ == "__main__":
    main()
