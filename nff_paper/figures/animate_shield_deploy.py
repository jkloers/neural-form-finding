"""Animate the cached shield design deploying, then resolving into the artwork.

The deployment is closed-form and one-DOF: a single scalar drives every tile, so the three
panels of the deployment strip are three samples of a motion that costs nothing extra to
sample densely. This script sweeps that scalar, holds on the open sheet, and then lets the
finished shield spread outward from the motto: behind that front the voids are already
filled and the sheet reaching past the silhouette is already gone, so the spill is
swallowed as the front passes rather than cleared afterwards.

Writes a PNG sequence always -- ``animategraphics`` indexes it -- plus an MP4 by default and
a GIF on request. The opening sweep costs one clipped raster per tile per frame, a few
tenths of a second; the front is a per-pixel blend of two fixed images and is far cheaper.

⚠ ``--svg`` is the ONLY input not cached in ``design.npz``. The artwork is rasterised fresh on
every run, so the animation cannot be rebuilt without it.

Nothing is written into ``paper/``: that directory is two-way Overleaf-synced.

Run:
  JAX_PLATFORMS=cpu conda run -n kgnn_mac python nff_paper/figures/animate_shield_deploy.py \
      --run-dir data/outputs/shield/princeton_paper --svg <path to Princeton_seal.svg>
"""

import argparse
import os
import shutil

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np

from nff.utils.figstyle import apply_charter, paper_rc, use_agg

use_agg()

from nff.topology.closed_builder_jax import build_deploy_structure
from nff_paper.closed_support.shield_kinematics import build_deploy_plan
from nff_paper.closed_support.shield_target import load_shield_raster
from nff_paper.closed_support.shield_render import render_deploy_animation


def _parse_centre(value):
    """'motto' / 'shield' pass through; anything else is an explicit 'x,y' in mm."""
    if value in ("motto", "shield"):
        return value
    return [float(v) for v in value.replace(",", " ").split()]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, help="Directory holding design.npz")
    p.add_argument("--svg", required=True, help="Target artwork, for the texture")
    p.add_argument("--grid", default="16x22")
    p.add_argument("--sheet", default="210x297", help="Closed sheet size in mm")
    p.add_argument("--out-dir", default=None, help="Default <run-dir>/deploy_anim")
    p.add_argument("--stem", default="deploy", help="Basename of the movie and the GIF")

    p.add_argument("--n-frames", type=int, default=120, help="Frames in the opening sweep")
    p.add_argument("--wipe-frames", type=int, default=36,
                   help="Frames for the finished shield to spread outward")
    p.add_argument("--hold-closed", type=int, default=6)
    p.add_argument("--hold-open", type=int, default=12)
    p.add_argument("--hold-logo", type=int, default=24)
    p.add_argument("--no-merge", dest="merge", action="store_false",
                   help="Stop on the open sheet; never resolve into the artwork")
    p.add_argument("--style", choices=("wipe", "fade"), default="wipe",
                   help="wipe: the shield spreads outward, swallowing the spill as it goes. "
                        "fade: the artwork fades in everywhere, then the sheet fades out")
    p.add_argument("--wipe-centre", default="motto",
                   help="'motto', 'shield', or explicit 'x,y' in mm")
    p.add_argument("--wipe-feather", type=float, default=25.0,
                   help="Softness of the advancing edge, in mm")
    p.add_argument("--wipe-rate", choices=("area", "linear", "smoothstep"), default="area",
                   help="area: steady revealed area; linear: steady radius")
    p.add_argument("--clean-frames", type=int, default=12,
                   help="Only for --style fade: frames fading the sheet back out")
    p.add_argument("--ease", choices=("smoothstep", "linear"), default="smoothstep")

    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--figsize", type=float, nargs=2, default=(5.4, 6.5),
                   help="Inches; sets the window aspect")
    p.add_argument("--margin", type=float, default=0.03)
    p.add_argument("--raster-px", type=int, default=1400)
    p.add_argument("--edge", default=None,
                   help="Tile outline colour, e.g. #1A1A1A -- makes the closed sheet legible")
    p.add_argument("--edge-lw", type=float, default=0.06)

    p.add_argument("--no-mp4", dest="mp4", action="store_false")
    p.add_argument("--gif", action="store_true", help="Also write a GIF (palette flicker)")
    p.add_argument("--crf", type=int, default=18)
    p.add_argument("--no-keep-frames", dest="keep_frames", action="store_false")
    p.add_argument("--slides-dir", default=None, help="Copy the movie and GIF here as well")
    return p.parse_args()


def main():
    args = parse_args()
    apply_charter(**paper_rc())
    M, N = (int(v) for v in args.grid.lower().split("x"))
    sheet_w, sheet_h = (float(v) for v in args.sheet.lower().split("x"))
    out_dir = args.out_dir or os.path.join(args.run_dir, "deploy_anim")

    d = np.load(os.path.join(args.run_dir, "design.npz"))
    struct = build_deploy_structure(M, N)
    plan = build_deploy_plan(struct, M, N, sheet_w / M, sheet_h / N)
    rgb, extent = load_shield_raster(args.svg, args.raster_px)

    result = render_deploy_animation(
        plan, d["flat_tiles"], d["deployed"], rgb, extent, d["target_unit"],
        float(d["scale"]), d["shift"], d["offset"], float(d["psi"]), out_dir,
        n_frames=args.n_frames,
        wipe_frames=args.wipe_frames if args.merge else 0,
        hold_closed=args.hold_closed, hold_open=args.hold_open,
        clean_frames=args.clean_frames if args.merge else 0,
        hold_logo=args.hold_logo if args.merge else 0,
        style=args.style, wipe_centre=_parse_centre(args.wipe_centre),
        wipe_feather=args.wipe_feather, wipe_rate=args.wipe_rate,
        ease=args.ease, dpi=args.dpi, figsize=tuple(args.figsize), margin=args.margin,
        edge=args.edge, lw=args.edge_lw, stem=args.stem, mp4=args.mp4, gif=args.gif,
        fps=args.fps, crf=args.crf, keep_frames=args.keep_frames)

    n = len(result["schedule"])
    origin = result["origin_mm"]
    print(f"  {n} frames, {n / args.fps:.1f} s at {args.fps} fps, "
          f"window {2 * result['half'][0]:.0f} x {2 * result['half'][1]:.0f} mm")
    if args.merge and args.style == "wipe":
        print(f"  front spreads from ({origin[0]:.1f}, {origin[1]:.1f}) mm "
              f"[{args.wipe_centre}], {args.wipe_rate} rate, {args.wipe_feather:.0f} mm edge")
    for key in ("mp4", "gif"):
        path = result[key]
        if path:
            print(f"  wrote {path}  ({os.path.getsize(path) / 1e6:.1f} MB)")
            if args.slides_dir:
                os.makedirs(args.slides_dir, exist_ok=True)
                dest = os.path.join(args.slides_dir, f"shield_{os.path.basename(path)}")
                shutil.copyfile(path, dest)
                print(f"  staged {dest}")
    if result["frames"]:
        print(f"  wrote {len(result['frames'])} frames in {out_dir}")


if __name__ == "__main__":
    main()
