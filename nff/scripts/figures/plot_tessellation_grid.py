"""Contact sheet of every deployed tessellation in a harvest -- a square of small kirigami sheets.

Purely an artefact of the harvest: the geometry is already stored, so drawing it costs nothing and
the result is a rather good picture of how much shape variety the random designs actually produce.

Each cell is one random tessellation, deployed, drawn at its own scale so the SHAPE reads rather
than the size (the sheets differ in extent by well over a factor of two, and a shared scale would
shrink most of them to nothing). Cells are ordered by deployed aspect ratio so the sheet reads as a
gradient rather than as noise.
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

ORANGE = "#F58025"
INK = "#1A1A1A"


def _faces_xy(verts, face_vertex_ids):
    return [verts[np.asarray(ids, dtype=int)] for ids in face_vertex_ids]


def build_grid_figure(ds, out_png: str, side=None, *, order: str = "aspect",
                      face_vertex_ids=None, color_by_angle: bool = False) -> str:
    """Draw a ``side x side`` contact sheet of deployed tessellations.

    Args:
        ds: a ``PathDataset`` from ``nff.closed.path_dataset``.
        out_png: output path.
        side: grid side length; ``None`` -> the largest square that fits the harvest.
        order: 'aspect' (deployed height/width), 'theta' (max hinge rotation) or 'seed'.
        face_vertex_ids: per-face global vertex indices. ``None`` -> taken from the harvest
            manifest, which records the (design-independent) face topology for exactly this.
        color_by_angle: shade each cell by its max hinge rotation instead of a flat orange.
    """
    ex = ds.examples
    if face_vertex_ids is None:
        face_vertex_ids = ds.meta.get('face_vertex_ids')
    n = len(ex)
    if side is None:
        side = int(np.floor(np.sqrt(n)))
    side = max(1, int(side))
    need = side * side

    theta_max = np.array([np.degrees(np.abs(e.eta[..., 2])).max() for e in ex])
    span = np.array([[np.ptp(e.verts[:, 0]), np.ptp(e.verts[:, 1])] for e in ex])
    aspect = span[:, 1] / np.maximum(span[:, 0], 1e-9)
    key = {'aspect': aspect, 'theta': theta_max}.get(order)
    idx = np.arange(n) if key is None else np.argsort(key)
    if n >= need:
        # keep an EVEN spread over the ordering rather than the first need cells, so the sheet
        # shows the whole range of shapes instead of one end of it
        idx = idx[np.linspace(0, n - 1, need).round().astype(int)]
    else:
        idx = np.resize(idx, need)

    fig, axes = plt.subplots(side, side, figsize=(side * 1.15, side * 1.6), facecolor="white")
    axes = np.atleast_2d(axes).reshape(side, side)
    cmap = plt.get_cmap("inferno")
    lo, hi = float(theta_max.min()), float(theta_max.max())
    for cell, i in enumerate(idx):
        ax = axes[cell // side, cell % side]
        e = ex[int(i)]
        v = e.verts
        if face_vertex_ids is not None:
            polys = _faces_xy(v, face_vertex_ids)
        else:
            polys = [v[k:k + 4] for k in range(0, len(v) - 3, 4)]
        c = (cmap(0.15 + 0.7 * (theta_max[int(i)] - lo) / max(hi - lo, 1e-9))
             if color_by_angle else ORANGE)
        ax.add_collection(PolyCollection(polys, facecolors=c, edgecolors=INK, linewidths=0.35))
        # per-cell limits: the shape is the subject, not the size
        pad = 0.06 * max(np.ptp(v[:, 0]), np.ptp(v[:, 1]))
        ax.set_xlim(v[:, 0].min() - pad, v[:, 0].max() + pad)
        ax.set_ylim(v[:, 1].min() - pad, v[:, 1].max() + pad)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.subplots_adjust(left=0.004, right=0.996, top=0.996, bottom=0.004, wspace=0.03, hspace=0.03)
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"  wrote {out_png}  ({side}x{side} of {n} deployed tessellations)")
    return out_png


def main():
    import argparse
    from nff.closed.path_dataset import load_dataset
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help="a harvest directory (holds paths.npz)")
    p.add_argument("--out", default=None)
    p.add_argument("--side", type=int, default=0)
    p.add_argument("--order", default="aspect", choices=("aspect", "theta", "seed"))
    p.add_argument("--color-by-angle", action="store_true")
    a = p.parse_args()
    ds = load_dataset(a.dataset)
    build_grid_figure(ds, a.out or os.path.join(a.dataset, "grid.png"),
                      side=a.side or None, order=a.order, color_by_angle=a.color_by_angle)


if __name__ == "__main__":
    main()
