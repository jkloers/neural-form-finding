"""The harvested paths themselves, as curves, in the three projections of (eta_a, eta_s, theta).

The hexbin figure answers "where do hinges spend time"; this one answers "what shape is the trip".
They are different questions, and only the second one matters for an elastoplastic oracle, where W
depends on the route taken and not merely on the destination. Every path is drawn as a line from
the origin (the undeployed sheet) to wherever that hinge ended, so the fan structure, the reversals
and the compressive excursions are all visible as trajectories rather than as density.

Panels: (eta_a, theta), (eta_s, theta), (eta_a, eta_s) -- the same order as the hexbin row, so the
two figures can be read side by side.
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from nff.models.hinge_surrogate import DOMAIN

ORANGE = "#F58025"
INK = "#1A1A1A"
GREY = "#6C757D"
RED = "#D62828"

# (x index, y index, x label, y label); component order is (eta_a, eta_s, theta)
PANELS = [(0, 2, r"$\eta_a = a/w_{lig}$", r"$\theta$  [deg]"),
          (1, 2, r"$\eta_s = s/w_{lig}$", r"$\theta$  [deg]"),
          (0, 1, r"$\eta_a = a/w_{lig}$", r"$\eta_s = s/w_{lig}$")]


def _limits(v, pad=0.06):
    lo, hi = float(np.min(v)), float(np.max(v))
    m = pad * max(hi - lo, 1e-9)
    return lo - m, hi + m


def build_path_projection_figure(ds, out_png: str, *, max_paths: int = 1500, seed: int = 0,
                                 label: str = "", color_by: str = "theta") -> str:
    """Draw the harvested polylines in the three 2-D projections.

    Args:
        ds: a ``PathDataset`` (``nff.closed.path_dataset``) or anything exposing ``polylines()``.
        out_png: output path.
        max_paths: subsample cap. Thousands of overlapping translucent curves stop adding
            information and start costing minutes of render time; a random subsample of the paths
            reads identically because the paths are exchangeable within a design.
        color_by: 'theta' (final rotation), 'compression' (whether the path ever goes a < 0),
            or 'none' (flat orange).
    """
    P = np.asarray(ds.polylines())                       # (n_paths, n_steps+1, 3)
    P = P.copy()
    P[..., 2] = np.degrees(P[..., 2])                    # theta in degrees for every panel
    n = P.shape[0]
    rng = np.random.default_rng(seed)
    sel = rng.choice(n, size=min(max_paths, n), replace=False)
    Q = P[sel]

    if color_by == "theta":
        key = Q[:, -1, 2]
        cmap, clabel = plt.get_cmap("inferno"), r"final $\theta$  [deg]"
    elif color_by == "compression":
        key = (Q[..., 0].min(axis=1) < 0.0).astype(float)
        cmap, clabel = plt.get_cmap("coolwarm"), "path goes compressive"
    else:
        key, cmap, clabel = None, None, None
    if key is not None:
        lo, hi = float(np.min(key)), float(np.max(key))
        norm = (key - lo) / max(hi - lo, 1e-9)

    # the box the oracle samples today, for reference
    cur = {0: (0.0, DOMAIN['eta_a_max']),
           1: (-DOMAIN['eta_s_max'], DOMAIN['eta_s_max']),
           2: (0.0, float(np.degrees(DOMAIN['theta_max'])))}

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2), facecolor="white")
    # Axes are set from the DATA, so when the sampled box is much larger than the ridden region it
    # runs off the panel and shows only as edge lines. Say so rather than let it read as a box that
    # happens to sit on the axis limits.
    clipped = False
    for ax, (ix, iy, xl, yl) in zip(axes, PANELS):
        # forbidden compressive half-plane: the oracle assumes a >= 0, and these paths do not
        if ix == 0:
            ax.axvspan(_limits(P[..., 0])[0], 0.0, color=RED, alpha=0.06, lw=0, zorder=0)
            ax.axvline(0.0, color=RED, lw=1.0, alpha=0.55, zorder=1)
        xr, yr = cur[ix], cur[iy]
        ax.add_patch(Rectangle((xr[0], yr[0]), xr[1] - xr[0], yr[1] - yr[0], fill=False,
                               edgecolor=GREY, lw=1.4, ls="--", zorder=2))
        for k in range(Q.shape[0]):
            c = ORANGE if key is None else cmap(0.12 + 0.78 * norm[k])
            ax.plot(Q[k, :, ix], Q[k, :, iy], color=c, lw=0.5, alpha=0.28,
                    solid_capstyle="round", zorder=3)
        ax.scatter(Q[:, -1, ix], Q[:, -1, iy], s=3.5, color=INK, alpha=0.5, lw=0, zorder=4)
        ax.scatter([0.0], [0.0], s=26, color=INK, zorder=5)      # the undeployed sheet
        xlim, ylim = _limits(P[..., ix]), _limits(P[..., iy])
        clipped |= (xr[0] < xlim[0] or xr[1] > xlim[1] or yr[0] < ylim[0] or yr[1] > ylim[1])
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    box_label = "currently sampled box" + (" (larger than the axes)" if clipped else "")
    handles = [Line2D([], [], color=GREY, ls="--", lw=1.4, label=box_label),
               Line2D([], [], color=ORANGE, lw=1.4, label="hinge paths"),
               Line2D([], [], color=INK, marker="o", ls="none", ms=4, label="path end"),
               Line2D([], [], color=RED, lw=1.4, alpha=0.55, label="compression: never sampled")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))
    if label:
        fig.suptitle(f"Hinge displacement paths  --  {label}", fontsize=14)
    if key is not None and color_by == "theta":
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=lo, vmax=hi))
        sm.set_array([])
        cb = fig.colorbar(sm, ax=axes, fraction=0.018, pad=0.015)
        cb.set_label(clabel, fontsize=10)
        cb.outline.set_visible(False)
    else:
        fig.tight_layout(rect=(0, 0.04, 1, 0.95 if label else 1.0))
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_png}  ({Q.shape[0]} of {n} paths)")
    return out_png


def main():
    import argparse
    from nff.closed.path_dataset import load_dataset
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help="a harvest directory (holds paths.npz)")
    p.add_argument("--out", default=None)
    p.add_argument("--max-paths", type=int, default=1500)
    p.add_argument("--color-by", default="theta", choices=("theta", "compression", "none"))
    p.add_argument("--label", default="")
    a = p.parse_args()
    ds = load_dataset(a.dataset)
    build_path_projection_figure(ds, a.out or os.path.join(a.dataset, "path_projections.png"),
                                 max_paths=a.max_paths, color_by=a.color_by, label=a.label)


if __name__ == "__main__":
    main()
