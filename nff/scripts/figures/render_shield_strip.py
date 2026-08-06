"""Re-render the deployment strip from a cached shield design.

``plot_shield_kirigami.py`` writes ``design.npz`` beside its figures. This script
reloads it and re-renders the strip at any panel count or resolution without
repeating the optimisation, which is the expensive part.

Run:
  JAX_PLATFORMS=cpu conda run -n kgnn_mac python nff/scripts/figures/render_shield_strip.py \
      --run-dir data/outputs/shield/princeton_paper --svg ~/Desktop/Princeton_seal.svg \
      --steps 4 --paper-dir paper
"""

import argparse
import os

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np

from nff.utils.figstyle import use_agg, apply_charter

use_agg()
import matplotlib.pyplot as plt

from nff.topology.closed_builder_jax import build_deploy_structure
from nff.closed.shield_kinematics import build_deploy_plan
from nff.closed.shield_target import load_shield_raster
from nff.closed.shield_render import render_deployment_strip


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, help="Directory holding design.npz")
    p.add_argument("--svg", required=True, help="Target artwork, for the texture")
    p.add_argument("--grid", default="16x22")
    p.add_argument("--steps", type=int, default=4, help="Panels, closed to open")
    p.add_argument("--sheet", default="210x297", help="Closed sheet size in mm")
    p.add_argument("--raster-px", type=int, default=3000)
    p.add_argument("--dpi", type=int, default=600)
    p.add_argument("--no-reference", dest="reference", action="store_false",
                   help="Omit the undeformed reference panel")
    p.add_argument("--no-labels", dest="labels", action="store_false",
                   help="Omit the per-panel labels")
    p.add_argument("--label-size", type=float, default=None,
                   help="Label point size in the source figure (scaled down in print)")
    p.add_argument("--paper-dir", default=None)
    p.add_argument("--paper-stem", default="shield_pipeline")
    return p.parse_args()


def main():
    args = parse_args()
    apply_charter()
    M, N = (int(v) for v in args.grid.lower().split("x"))
    sheet_w, sheet_h = (float(v) for v in args.sheet.lower().split("x"))

    d = np.load(os.path.join(args.run_dir, "design.npz"))
    struct = build_deploy_structure(M, N)
    plan = build_deploy_plan(struct, M, N, sheet_w / M, sheet_h / N)

    rgb, extent = load_shield_raster(args.svg, args.raster_px)
    fig = render_deployment_strip(
        plan, d["flat_tiles"], d["deployed"], rgb, extent, d["target_unit"],
        float(d["scale"]), d["shift"], d["offset"], float(d["psi"]), n_steps=args.steps,
        reference=args.reference, labels=args.labels or None,
        label_size=args.label_size)

    targets = [(args.run_dir, f"pipeline_{args.steps}_steps")]
    if args.paper_dir:
        targets.append((os.path.join(args.paper_dir, "Images"), args.paper_stem))
    # PNG only: each panel is a few hundred clipped rasters, so a PDF wraps the same
    # pixels at ~90x the size without becoming vector.
    for directory, stem in targets:
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"{stem}.png")
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight", pad_inches=0.01,
                    facecolor="white")
        print(f"  wrote {path}  ({os.path.getsize(path) / 1e6:.1f} MB)")
    plt.close(fig)


if __name__ == "__main__":
    main()
