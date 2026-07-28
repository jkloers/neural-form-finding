"""Harvest a hinge-displacement-path dataset from random tessellations.

Each example is a random flat design (random void aspect ratios + boundary points) deployed with
ONE clamped tile on one side and ONE pulling tile on the other. Every hinge's full path through
(eta_a, eta_s, theta) is recorded -- the whole polyline, so it can later be replayed into the
oracle verbatim rather than approximated by a straight ray.

Run:
    PYTHONPATH=$(pwd) JAX_PLATFORMS=cpu conda run -n kgnn_mac \
        python nff/scripts/diagnostics/harvest_paths.py \
            --config data/configs/closed/sheet_4x8ft_rom_1tile.yaml --n 256 --noise 0.5

Writes into ``data/fea/path_priors/<name>_n<N>_s<noise>/``:
    paths.npz          stacked polylines, alpha, w_lig, designs, deployed + flat geometry
    manifest.json      provenance: config, BCs, loads, length_scale, w_lig, failures
    distribution.json  pooled marginals / correlations / variance split / measured oracle spec
    distribution.png   the density figure
    grid.png           contact sheet of the deployed tessellations
"""

import argparse
import json
import os

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np

from nff.config.experiment import load_and_parse_config
from nff.closed.path_dataset import harvest_paths, save_dataset
from nff.closed.hinge_path_ensemble import (PathEnsemble, EnsembleSample, ensemble_statistics,
                                            sampling_spec)
from nff.closed.hinge_paths import HingePaths


def _as_ensemble(ds) -> PathEnsemble:
    """Reuse the ensemble statistics stack (marginals, ICC, oracle spec) on a harvested dataset."""
    ens = PathEnsemble(config_path=ds.meta.get('config_path', ''),
                       length_scale=float(ds.meta.get('length_scale_mm_per_unit', 1.0)),
                       n_load_steps=int(ds.meta.get('n_load_steps', 0)))
    for e in ds.examples:
        ens.samples.append(EnsembleSample(
            seed=e.seed, noise=float(ds.meta.get('noise', 0.0)),
            paths=HingePaths(u=e.eta, eta=e.eta, w_lig=e.w_lig, alpha=e.alpha,
                             length_scale=ens.length_scale, load_fraction=e.load_frac)))
    return ens


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="data/configs/closed/sheet_4x8ft_rom_1tile.yaml",
                   help="closed-pipeline YAML; the default is the single-tile grip standard")
    p.add_argument("--n", type=int, default=256, help="number of random tessellations")
    p.add_argument("--noise", type=float, default=0.5,
                   help="init_noise: Gaussian sigma on the design logits (z and bnd_logits)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=30,
                   help="load steps = polyline resolution (what the oracle will replay)")
    p.add_argument("--out", default=None)
    p.add_argument("--angle-max", type=float, default=50.0,
                   help="each grip's force is calibrated so its WORST hinge reaches this angle; "
                        "forces are then drawn as a fraction of that ceiling. Slightly above the "
                        "45 deg working point so the surrogate gets margin past it")
    p.add_argument("--load-lo", type=float, default=0.02,
                   help="lowest force as a FRACTION of the grip ceiling (log-uniform)")
    p.add_argument("--load-hi", type=float, default=3.0,
                   help="highest force as a fraction of the grip ceiling. >1 deliberately drives "
                        "PAST the first physical limit so the surrogate sees the over-driven "
                        "regime the optimizer can wander into, rather than extrapolating blindly")
    p.add_argument("--absolute-load", action="store_true",
                   help="treat --load-lo/--load-hi as newtons and skip the per-grip calibration")
    p.add_argument("--fixed-grip", action="store_true",
                   help="keep the config's single (clamp, pull) pair instead of varying it")
    p.add_argument("--grid", type=int, default=0,
                   help="ALSO write a contact sheet this many cells wide (0 = skip)")
    p.add_argument("--chunk", type=int, default=180,
                   help="run the harvest as subprocesses of this many examples each, then merge. "
                        "forward_pipeline recompiles per call and JAX never evicts, so a single "
                        "long process leaks to OOM; a process boundary returns the memory. Use a "
                        "multiple of the grip count so each shard stays balanced. 0 = one process")
    p.add_argument("--shard", type=int, default=-1, help=argparse.SUPPRESS)   # internal
    args = p.parse_args()

    config = load_and_parse_config(args.config)
    name = os.path.splitext(os.path.basename(args.config))[0]
    out_dir = args.out or os.path.join("data", "fea", "path_priors",
                                       f"{name}_n{args.n}_s{args.noise:g}")

    from nff.closed.path_dataset import grip_pairs, calibrate_grip_loads, merge_datasets
    grips = None if args.fixed_grip else 'all'
    pairs = grip_pairs(config) if grips == 'all' else [(int(config.topology['bc_clamped'][0]),
                                                        int(config.topology['loads'][0]['face']))]
    gl_path = os.path.join(out_dir, "grip_loads.json")

    def _load_grip_loads():
        if args.absolute_load or not os.path.exists(gl_path):
            return None
        with open(gl_path) as f:
            return {tuple(int(v) for v in k.split("->")): float(x)
                    for k, x in json.load(f).items()}

    # ── shard mode: harvest one chunk in this process and exit (see --chunk) ──
    if args.shard >= 0:
        n_this = min(args.chunk, args.n - args.shard * args.chunk)
        ds = harvest_paths(config, n_examples=n_this, noise=args.noise,
                           seed0=args.seed + args.shard * args.chunk, n_load_steps=args.steps,
                           config_path=args.config, grips=grips,
                           load_range=(args.load_lo, args.load_hi),
                           grip_loads=_load_grip_loads(), progress_every=0, verbose=False)
        if ds.examples:
            save_dataset(ds, os.path.join(out_dir, "shards", f"shard_{args.shard:04d}"))
        print(f"  shard {args.shard:4d}: {ds.n_examples}/{n_this} deployed "
              f"({len(ds.failures)} failed)", flush=True)
        return

    print(f"\n  {args.n} random tessellations from {args.config}")
    print(f"  GRIPS ({len(pairs)}): " + ", ".join(f"{c}->{l}" for c, l in pairs))
    print(f"    clamped tile always on one side, pulled tile always on the other; "
          f"dofs {config.topology.get('clamped_dofs') or 'all'}")
    print(f"  DESIGN init_noise={args.noise}  seeds {args.seed}..{args.seed + args.n - 1}  "
          f"{args.steps} load steps")

    os.makedirs(out_dir, exist_ok=True)
    grip_loads = None
    if not args.absolute_load:
        # Grips differ ~5x in compliance, so an absolute force means a different deployment depth
        # for each one. Calibrate each grip's own limit first, then draw a fraction of it. Done
        # ONCE here and written to disk so the shards read it instead of repeating it.
        print(f"\n  calibrating each grip (theta_max {args.angle_max:g} deg or eta 1.0, "
              f"whichever binds) ...")
        grip_loads = calibrate_grip_loads(config, pairs, angle_max_deg=args.angle_max)
        with open(gl_path, "w") as f:
            json.dump({f"{c}->{l}": v for (c, l), v in grip_loads.items()}, f, indent=2)
        print(f"  FORCE  {args.load_lo:g}..{args.load_hi:g} x each grip's ceiling, log-uniform\n")
    else:
        print(f"  FORCE  {args.load_lo:g}..{args.load_hi:g} N absolute, log-uniform\n")

    if args.chunk and args.chunk < args.n:
        import subprocess
        import sys
        n_shards = (args.n + args.chunk - 1) // args.chunk
        print(f"  running {n_shards} shards of <={args.chunk} in separate processes "
              f"(JAX leaks ~55 MB/example inside one process)\n")
        base = [sys.executable, "-u", os.path.abspath(__file__),
                "--config", args.config, "--n", str(args.n), "--noise", str(args.noise),
                "--seed", str(args.seed), "--steps", str(args.steps), "--out", out_dir,
                "--chunk", str(args.chunk), "--load-lo", str(args.load_lo),
                "--load-hi", str(args.load_hi), "--angle-max", str(args.angle_max)]
        if args.fixed_grip:
            base.append("--fixed-grip")
        if args.absolute_load:
            base.append("--absolute-load")
        for k in range(n_shards):
            r = subprocess.run(base + ["--shard", str(k)], env={**os.environ, "PYTHONUNBUFFERED": "1"})
            if r.returncode != 0:
                print(f"  ! shard {k} exited {r.returncode} -- keeping what it wrote and continuing")
        shard_dirs = sorted(os.path.join(out_dir, "shards", d)
                            for d in os.listdir(os.path.join(out_dir, "shards")))
        ds = merge_datasets(shard_dirs)
    else:
        ds = harvest_paths(config, n_examples=args.n, noise=args.noise, seed0=args.seed,
                           n_load_steps=args.steps, config_path=args.config,
                           grips=grips, load_range=(args.load_lo, args.load_hi),
                           grip_loads=grip_loads, progress_every=max(1, args.n // 25))
    if not ds.examples:
        raise SystemExit("every design failed to deploy -- nothing to store")

    npz = save_dataset(ds, out_dir)
    print(f"\n  {ds.n_examples}/{args.n} deployed  ({len(ds.failures)} failed)  -> {npz}")

    ens = _as_ensemble(ds)
    stats = ensemble_statistics(ens)
    spec = sampling_spec(ens)
    with open(os.path.join(out_dir, "distribution.json"), "w") as f:
        json.dump({'statistics': stats, 'sampling_spec': spec, 'manifest': ds.meta}, f, indent=2)

    m = stats['endpoint_marginals']
    print(f"\n  ENDPOINT MARGINALS ({ds.n_examples} designs x {ds.meta.get('n_hinges', '?')} hinges)")
    for k in ('eta_a', 'eta_s', 'theta_deg'):
        q = m[k]['q']
        print(f"    {k:10s} p1 {q['p1']:+8.3f}  p50 {q['p50']:+8.3f}  p99 {q['p99']:+8.3f}")
    q = stats['alpha_deg']['q']
    print(f"    {'alpha_deg':10s} p1 {q['p1']:+8.3f}  p50 {q['p50']:+8.3f}  p99 {q['p99']:+8.3f}")
    c = stats['compression']
    print(f"    compression: {c['design_frac_with_any']:.0%} of designs, "
          f"{c['hinge_frac']:.0%} of hinges, min eta_a {c['min_eta_a']:+.3f}")
    print(f"    in-domain: {stats['in_domain_frac']:.1%} of path points")
    print(f"    straightness: mean {stats['straightness']['mean']:.3f} "
          f"(0 = one proportional ray is exact)")

    from nff.scripts.figures.plot_hinge_path_distribution import build_distribution_figure
    from nff.scripts.figures.plot_hinge_path_projections import build_path_projection_figure
    build_distribution_figure(ens, os.path.join(out_dir, "distribution.png"), label=name)
    build_path_projection_figure(ds, os.path.join(out_dir, "path_projections.png"),
                                 max_paths=2500, label=name)
    if args.grid:
        from nff.scripts.figures.plot_tessellation_grid import build_grid_figure
        build_grid_figure(ds, os.path.join(out_dir, "grid.png"), side=args.grid)
    print(f"\n  wrote {out_dir}\n")


if __name__ == "__main__":
    main()
