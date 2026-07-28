"""Render one replayed hinge deployment: the deformed RVE, and energy/damage along the path.

Two plates from a ``replay_hinge_path`` .npz:

* ``<out>_render.png`` -- the deformed ligament in 3D after deployment, coloured by equivalent
  plastic strain so the damaged zone reads directly, with a millimetre scale bar.
* ``<out>_curves.png`` -- W, moment, damage, ligament PEEQ and out-of-plane buckle amplitude
  against rotation, i.e. everything the campaign records, along the route it was recorded on.

    conda run -n kgnn_mac python -m nff.scripts.figures.plot_hinge_replay \
        --npz data/experiments/processed/replay_sample_seed0.npz --out data/experiments/processed/replay_sample
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from nff.utils.figstyle import GREY, INK, ORANGE, RED, TEAL, apply_charter, use_agg

use_agg()
import matplotlib.pyplot as plt                                          # noqa: E402
from matplotlib.colors import Normalize                                  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection                  # noqa: E402

# C3D15 corner nodes: bottom triangle 0,1,2 / top triangle 3,4,5 (the mid-side nodes 6..14 are
# only needed for the field interpolation, not for drawing the element's skin).
_TRIS = [(0, 1, 2), (3, 4, 5)]
_QUADS = [(0, 1, 4, 3), (1, 2, 5, 4), (2, 0, 3, 5)]


def _surface_faces(conn):
    """Exterior faces of the wedge mesh: those belonging to exactly one element."""
    seen, owner = {}, {}
    for e, row in enumerate(conn):
        c = [row[i] - 1 for i in range(6)]
        for f in _TRIS + _QUADS:
            nodes = tuple(c[i] for i in f)
            key = tuple(sorted(nodes))
            seen[key] = seen.get(key, 0) + 1
            owner[key] = (nodes, e)
    return [owner[k] for k, n in seen.items() if n == 1]


def _draw(ax, polys, vals, vmax, ctr, rad, elev, azim, label, lw=0.08):
    pc = Poly3DCollection(polys, cmap="inferno", norm=Normalize(0.0, vmax),
                          linewidths=lw, edgecolors=(0, 0, 0, 0.22))
    pc.set_array(vals)
    ax.add_collection3d(pc)
    for setlim, c in ((ax.set_xlim, 0), (ax.set_ylim, 1), (ax.set_zlim, 2)):
        setlim(ctr[c] - rad, ctr[c] + rad)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()

    # millimetre scale bar, drawn in the model's own units at the front-bottom corner
    bar = 10.0 ** np.floor(np.log10(rad))
    bar = bar * (5 if rad / bar > 5 else (2 if rad / bar > 2 else 1))
    x0, y0, z0 = ctr[0] - 0.85 * rad, ctr[1] - 0.9 * rad, ctr[2] - 0.92 * rad
    ax.plot([x0, x0 + bar], [y0, y0], [z0, z0], color=INK, lw=2.2, zorder=10)
    ax.text(x0 + bar / 2, y0, z0 - 0.09 * rad, f"{bar:g} mm", color=INK,
            ha="center", va="top", fontsize=8.5)
    ax.text2D(0.03, 0.92, label, transform=ax.transAxes, color=GREY, fontsize=9)
    return pc


def plot_render(d, out_path, zoom=2.2):
    apply_charter()
    xyz, conn = d["xyz"], d["conn"]
    defo = xyz + d["disp"][:, :3]
    peeq, lig = d["elem_peeq"], d["lig_mask"].astype(bool)
    w_lig = float(d["w_lig"])

    faces = _surface_faces(conn)
    polys = [defo[list(nodes)] for nodes, _ in faces]
    vals = np.array([peeq[e] for _, e in faces])
    vmax = float(np.nanpercentile(peeq[lig], 99.5)) if np.isfinite(peeq[lig]).any() else 1.0
    vmax = max(vmax, 1e-6)

    # The Saint-Venant window is r_win = 100 mm around an 18 mm ligament, so a whole-RVE view is
    # 95 % undeformed panel. Frame the ligament instead and keep the full window as context only.
    cell = np.array([defo[np.asarray(conn, int)[e, :6] - 1].mean(axis=0) for e in range(len(conn))])
    ctr_l = cell[lig].mean(axis=0) if lig.any() else defo.mean(axis=0)
    rad_l = zoom * w_lig
    full = defo.min(axis=0), defo.max(axis=0)
    ctr_f = 0.5 * (full[0] + full[1])
    rad_f = 0.55 * float(np.max(full[1] - full[0]))

    fig = plt.figure(figsize=(13.4, 5.0))
    _draw(fig.add_subplot(1, 3, 1, projection="3d"), polys, vals, vmax, ctr_f, rad_f,
          24, -62, f"full RVE  (r_win {float(d['r_win']):g} mm)", lw=0.05)
    pc = _draw(fig.add_subplot(1, 3, 2, projection="3d"), polys, vals, vmax, ctr_l, rad_l,
               24, -62, "ligament — oblique", lw=0.15)
    _draw(fig.add_subplot(1, 3, 3, projection="3d"), polys, vals, vmax, ctr_l, rad_l,
          88, -90, "ligament — plan", lw=0.15)

    cax = fig.add_axes([0.35, 0.07, 0.30, 0.028])
    cb = fig.colorbar(pc, cax=cax, orientation="horizontal")
    cb.set_label("equivalent plastic strain (PEEQ)", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    prov = json.loads(str(d["provenance"]))
    fig.text(0.5, 0.965,
             f"deployed hinge  ·  w_lig {w_lig:g} mm  ·  t {float(d['thickness']):g} mm  ·  "
             f"alpha {float(d['alpha_deg']):.0f}°  ·  PET",
             ha="center", fontsize=10.5, color=INK)
    fig.text(0.5, 0.925,
             f"{prov['prior']}  example {prov['example']}  hinge {prov['hinge']}  ·  "
             f"ligament disc = {float(d['lig_radius_frac']):g}·w_lig",
             ha="center", fontsize=8.5, color=GREY)
    fig.savefig(out_path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return vmax


def plot_curves(d, out_path):
    apply_charter()
    th = d["theta_deg"]
    eps_f = float(d["eps_f"])
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 6.6))

    panels = [
        (axes[0, 0], d["W"], "stored + dissipated work  W  [N·mm]", TEAL, None),
        (axes[0, 1], d["M_theta"], "moment  M$_\\theta$  [N·mm]", ORANGE, None),
        (axes[0, 2], d["uz_max"], "out-of-plane buckle  u$_z$  [mm]", GREY, None),
        (axes[1, 0], d["damage"], "damage  $\\Delta$ = ⟨PEEQ⟩$_{lig}$ / $\\epsilon_f$", RED, None),
        (axes[1, 1], d["peeq_lig"], "peak ligament PEEQ", RED, eps_f),
        (axes[1, 2], None, "in-plane handle motion  [mm]", None, None),
    ]
    for ax, y, label, colour, ref in panels:
        if y is not None:
            ax.plot(th, y, color=colour, lw=1.7)
            ax.scatter(th[-1], y[-1], s=18, color=colour, zorder=5)
        ax.set_xlabel("rotation  θ  [deg]")
        ax.set_ylabel(label)
        if ref is not None:
            ax.axhline(ref, color=INK, lw=1.0, ls="--", alpha=0.6)
            ax.text(th.min(), ref, f"  $\\epsilon_f$ = {ref:g}", va="bottom", ha="left",
                    fontsize=8, color=INK)
            ax.set_ylim(0, max(ref * 1.08, float(np.nanmax(y)) * 1.15))

    ax = axes[1, 2]
    ax.plot(th, d["a"], color=TEAL, lw=1.7, label="axial  a")
    ax.plot(th, d["s"], color=ORANGE, lw=1.7, label="shear  s")
    ax.axhline(0.0, color=GREY, lw=0.8, alpha=0.6)
    ax.legend(fontsize=8.5)

    fig.text(0.5, 0.975, "hinge response along the replayed deployment path",
             ha="center", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(out_path, dpi=190)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True, help="output prefix (no extension)")
    ap.add_argument("--zoom", type=float, default=2.2, help="ligament view radius, in w_lig")
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    d = {k: z[k] for k in z.files}
    vmax = plot_render(d, args.out + "_render.png", zoom=args.zoom)
    plot_curves(d, args.out + "_curves.png")
    print(f"peak element PEEQ (p99.5 in ligament) = {vmax:.4f}")
    print(f"wrote {args.out}_render.png and {args.out}_curves.png")


if __name__ == "__main__":
    main()
