"""CLI for the compression-shift figure. The figure itself is ``nff/closed/compression_shift_viz``.

Run:
    PYTHONPATH=$(pwd) JAX_PLATFORMS=cpu conda run -n kgnn_mac \
        python nff/scripts/figures/plot_compression_shift.py \
            --pair data/outputs/hinge_paths/<before>,data/outputs/hinge_paths/<after> \
            --endpoints --out data/outputs/compression_shift.png

``--pair`` is repeatable; several pairs are averaged per cell, each weighted equally.
"""

import argparse
import os

import numpy as np

from nff.closed.hinge_paths import load_paths
from nff.closed.compression_shift_viz import (build_figure, GRIDSIZE, GRIDSIZE_ENDPOINTS,
                                              N_RESAMPLE)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pair", action="append", required=True, metavar="BEFORE_DIR,AFTER_DIR",
                   help="one sub-problem: the shape-only run and the compression run, comma "
                        "separated. Repeat for each sub-problem; they are averaged per cell.")
    p.add_argument("--out", required=True)
    p.add_argument("--show-paths", action="store_true",
                   help="overlay the hinge trajectories themselves, old (grey) against new (ink)")
    p.add_argument("--n-paths", type=int, default=None,
                   help="draw only this many randomly chosen hinges per design (default: all)")
    p.add_argument("--endpoints", action="store_true",
                   help="bin only each hinge's DEPLOYED state instead of its whole trajectory. "
                        "Exactly conserved, so the field shows redistribution and nothing else -- "
                        "but it cannot show transient compression, which is what the penalty acts on.")
    p.add_argument("--gridsize", type=int, default=None,
                   help=f"hexagons across x (default {GRIDSIZE}, or {GRIDSIZE_ENDPOINTS} with --endpoints)")
    p.add_argument("--resample", type=int, default=N_RESAMPLE)
    args = p.parse_args()

    pairs = []
    for spec in args.pair:
        b, _, a = spec.partition(",")
        pairs.append((load_paths(os.path.join(b.strip(), "hinge_paths")),
                      load_paths(os.path.join(a.strip(), "hinge_paths"))))

    gridsize = args.gridsize if args.gridsize is not None else (
        GRIDSIZE_ENDPOINTS if args.endpoints else GRIDSIZE)
    out, stats = build_figure(pairs, args.out, gridsize=gridsize, n_resample=args.resample,
                              use_endpoints=args.endpoints, show_paths=args.show_paths,
                              n_paths=args.n_paths)

    n = stats['n_sub_problems']
    cb = np.array(stats['compressive_pct_before'])
    ca = np.array(stats['compressive_pct_after'])
    tb = np.array(stats['max_theta_deg_before'])
    ta = np.array(stats['max_theta_deg_after'])
    view = "DEPLOYED ENDPOINT only" if args.endpoints else f"whole path, {args.resample} pts"
    print(f"  {n} sub-problem(s), {pairs[0][0].n_hinges} hinges each, {view}")
    if args.endpoints:
        # The conservation the endpoint view is chosen FOR. One point per hinge per design means
        # the signed change must sum to zero over the plane; if it does not, the binning dropped
        # points and the figure is not the redistribution map it claims to be.
        print(f"  conservation check: sum of signed change = {stats['delta_sum']:+.6g} "
              f"(exactly 0 expected)")
        eb = np.array(stats['endpoint_compressive_pct_before'])
        ea = np.array(stats['endpoint_compressive_pct_after'])
        print(f"  compressive AT THE ENDPOINT {eb.mean():6.2f}% -> {ea.mean():6.2f}%   "
              f"mean change {(ea - eb).mean():+.2f} pts")
    print(f"  compressive path points  {cb.mean():6.2f}% -> {ca.mean():6.2f}%   "
          f"mean change {(ca - cb).mean():+.2f} pts   (sd {(ca - cb).std():.2f})")
    print(f"    per sub-problem: " + "  ".join(f"{d:+.1f}" for d in (ca - cb)))
    # theta is the CONTROL: with the load held fixed each grip deploys to its own angle, so a
    # compression gain bought by deploying less has to be visible rather than folded into the mean.
    print(f"  theta max                {tb.mean():6.2f} deg -> {ta.mean():6.2f} deg   "
          f"mean change {(ta - tb).mean():+.2f} deg")
    print(f"    spread across grips (shape-only): {tb.min():.2f} .. {tb.max():.2f} deg")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
