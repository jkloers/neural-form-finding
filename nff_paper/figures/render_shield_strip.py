"""Re-render the deployment strip from a cached shield design.

``plot_shield_kirigami.py`` writes ``design.npz`` beside its figures. This script
reloads it and re-renders the strip at any panel count or resolution without
repeating the optimisation, which is the expensive part.

The strip is authored at ``\textwidth`` and written as a VECTOR PDF. Each panel is one clipped
raster per tile -- several hundred of them -- which is why this figure used to ship as a PNG; the
tile layer is now flattened to ONE image per panel at save resolution and the labels stay real
text, which takes the file from the 109 MB a naive PDF cost to well under one.

⚠ ``--svg`` is the ONLY input not cached in ``design.npz``. The artwork is rasterised fresh on
every run, so the figure cannot be rebuilt without it.

Run (``--steps 3`` with the reference panel is the manuscript's four-panel Figure 4):
  JAX_PLATFORMS=cpu conda run -n kgnn_mac python nff_paper/figures/render_shield_strip.py \
      --run-dir data/outputs/shield/princeton_paper --svg <path to Princeton_seal.svg> \
      --steps 3 --paper-dir paper
"""

import argparse
import os

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np

from nff.utils.figstyle import (FS_BASE, TEXT_W_BP, apply_charter, paper_rc, save_paper_figure,
                                use_agg)

use_agg()
import matplotlib.pyplot as plt

from nff.topology.closed_builder_jax import build_deploy_structure
from nff_paper.closed_support.shield_kinematics import build_deploy_plan
from nff_paper.closed_support.shield_target import load_shield_raster
from nff_paper.closed_support.shield_render import render_deployment_strip


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
    apply_charter(**paper_rc())
    M, N = (int(v) for v in args.grid.lower().split("x"))
    sheet_w, sheet_h = (float(v) for v in args.sheet.lower().split("x"))

    d = np.load(os.path.join(args.run_dir, "design.npz"))
    struct = build_deploy_structure(M, N)
    plan = build_deploy_plan(struct, M, N, sheet_w / M, sheet_h / N)

    # The strip is authored at the width it prints at, so the labels are printed point sizes. The
    # panel count sets the panel width; the height keeps the composition's 5.7:6.4 proportion.
    n_panels = args.steps + (1 if args.reference else 0)
    panel_w_bp = TEXT_W_BP / n_panels
    panel_h_bp = panel_w_bp * 6.4 / 5.7
    rgb, extent = load_shield_raster(args.svg, args.raster_px)
    fig = render_deployment_strip(
        plan, d["flat_tiles"], d["deployed"], rgb, extent, d["target_unit"],
        float(d["scale"]), d["shift"], d["offset"], float(d["psi"]), n_steps=args.steps,
        panel_w=panel_w_bp / 72.0, panel_h=panel_h_bp / 72.0,
        reference=args.reference, labels=args.labels or None,
        label_size=args.label_size or FS_BASE, rasterize=True)

    targets = [(args.run_dir, f"pipeline_{args.steps}_steps")]
    if args.paper_dir:
        targets.append((os.path.join(args.paper_dir, "Images"), args.paper_stem))
    for directory, stem in targets:
        pdf, lo, hi = save_paper_figure(fig, os.path.join(directory, stem),
                                        TEXT_W_BP, panel_h_bp, dpi=args.dpi)
        print(f"  wrote {pdf}  ({os.path.getsize(pdf) / 1e6:.1f} MB, "
              f"{TEXT_W_BP:.1f} x {panel_h_bp:.1f} bp, type {lo:.1f}-{hi:.1f} pt)")
    plt.close(fig)


if __name__ == "__main__":
    main()
