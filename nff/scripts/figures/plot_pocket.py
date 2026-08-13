"""The pocket a local probe raises on a large sheet: its map, its reach, and its convergence.

Reads the runs `probe_pocket.py` writes and draws three things:

    a  the pocket on the sheet, one tile per marker, probe marked
    b  reach against direction, showing whether the pocket is round or aimed
    c  radius against sheet size, which is the whole boundary-independence claim

Run:
    JAX_PLATFORMS=cpu PYTHONPATH=$(pwd) python nff/scripts/figures/plot_pocket.py \
        --runs data/outputs/pocket/pocket_11 data/outputs/pocket/pocket_15 \
               data/outputs/pocket/pocket_21
"""

import argparse
import json
import os

import numpy as np

from nff.closed.one_dof import graph_distance
from nff.utils import figstyle as FS


def _load(run_dir):
    with open(os.path.join(run_dir, "summary.json")) as fh:
        meta = json.load(fh)
    return meta, np.load(os.path.join(run_dir, "pocket.npz"))


def _radius_series(meta):
    """Per-direction force radius and the direction it was measured at."""
    rows = meta["linear"]["rows"]
    ang = np.array([r["angle_deg"] for r in rows])
    rad = np.array([r.get("force_radius", np.nan) for r in rows], dtype=float)
    return ang, rad


def plot(runs, out, scale=1.0):
    import matplotlib
    FS.use_agg()
    import matplotlib.pyplot as plt

    loaded = [(_load(r)) for r in runs]
    w_bp, h_bp = FS.page_bp(width_frac=1.0, aspect=2.55)

    with matplotlib.rc_context(FS.paper_rc(scale=scale)):
        fig, axes = plt.subplots(1, 3, figsize=FS.figsize_in(w_bp, h_bp))

        # ── a: the pocket on the largest sheet ────────────────────────────────
        meta, npz = loaded[-1]
        ax = axes[0]
        xy = npz["centroids"]
        field = npz["force__0"]
        norm = field / field.max()
        ax.scatter(xy[:, 0], xy[:, 1], c=norm, s=9, cmap="magma_r", linewidths=0,
                   vmin=0.0, vmax=1.0)
        probe = [meta["sheet"]["centre_face"], meta["sheet"]["partner_face"]]
        ax.plot(xy[probe, 0], xy[probe, 1], marker="o", ms=3.0, mfc="none",
                mec=FS.RED, mew=0.8, ls="-", color=FS.RED, lw=0.8)
        ax.set_aspect("equal")
        ax.axis("off")                                   # geometry plot: no axes
        FS.panel_letter(ax, "a", scale=scale)

        # ── b: how the pocket decays away from the probe ──────────────────────
        # A shell median against hop distance, not an angular reach: any measure weighted by
        # distance is dominated by the far field on this sheet and reports the sheet rather than
        # the pocket. The shell median is what the radius is defined from, so this panel and the
        # next are the same measurement seen two ways.
        ax = axes[1]
        for meta, npz in loaded:
            dist = graph_distance(npz["face_pairs"], npz["centroids"].shape[0],
                                  [meta["sheet"]["centre_face"]])
            field = npz["force__0"]
            hops = np.arange(0, dist.max() + 1)
            med = np.array([np.median(field[dist == h]) if (dist == h).any() else np.nan
                            for h in hops]) / field.max()
            ax.semilogy(hops, med, marker="o", ms=2.0, lw=0.9,
                        label=f"{meta['sheet']['M']}x{meta['sheet']['N']}")
        ax.set_xlabel("distance from the probe (tiles)")
        ax.set_ylabel("stiff-spring energy / peak")
        ax.set_xlim(0, 10)
        ax.legend(frameon=False, fontsize=FS.printed_pt(FS.FS_ANNOT, scale))
        FS.panel_letter(ax, "b", scale=scale)

        # ── c: what converges with sheet size, and what does not ──────────────
        # The threshold radius is measured against the interior level, and the interior level IS the
        # finite sheet's own floor, so it cannot converge while the floor is still falling. The
        # shell values can, and they show the split: the near field is a property of the medium
        # while the tail is still set by the boundary at 21x21.
        ax = axes[2]
        sizes = [meta["sheet"]["M"] for meta, _ in loaded]
        shells = {}
        for meta, npz in loaded:
            dist = graph_distance(npz["face_pairs"], npz["centroids"].shape[0],
                                  [meta["sheet"]["centre_face"]])
            field = npz["force__0"]
            for h in (1, 2, 4):
                shells.setdefault(h, []).append(np.median(field[dist == h]) / field.max())
        for h, vals in shells.items():
            ax.semilogy(sizes, vals, marker="o", ms=3.0, lw=1.0, label=f"{h} tile" + "s" * (h > 1))
        ax.set_xlabel("sheet size (tiles)")
        ax.set_ylabel("energy / peak at fixed distance")
        ax.set_xticks(sizes)
        ax.legend(frameon=False, fontsize=FS.printed_pt(FS.FS_ANNOT, scale),
                  title="distance", title_fontsize=FS.printed_pt(FS.FS_ANNOT, scale))
        FS.panel_letter(ax, "c", scale=scale)

        fig.tight_layout(pad=0.4)
        FS.save_paper_figure(fig, out, w_bp, h_bp, scale=scale)
    print(f"[pocket] wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default="data/outputs/pocket/pocket.pdf")
    ap.add_argument("--scale", type=float, default=1.0)
    args = ap.parse_args()
    plot(args.runs, args.out, args.scale)


if __name__ == "__main__":
    main()
