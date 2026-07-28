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
    p.add_argument("--grid", type=int, default=0,
                   help="side length of the contact sheet (0 = the largest square that fits)")
    args = p.parse_args()

    config = load_and_parse_config(args.config)
    name = os.path.splitext(os.path.basename(args.config))[0]
    out_dir = args.out or os.path.join("data", "fea", "path_priors",
                                       f"{name}_n{args.n}_s{args.noise:g}")

    clamped = config.topology.get('bc_clamped', [])
    loaded = [s['face'] for s in config.topology.get('loads', [])]
    print(f"\n  {args.n} random tessellations from {args.config}")
    print(f"  HOLD faces {clamped}  (dofs {config.topology.get('clamped_dofs') or 'all'})   "
          f"PULL faces {loaded}")
    print(f"  init_noise={args.noise}  seeds {args.seed}..{args.seed + args.n - 1}  "
          f"{args.steps} load steps\n")

    ds = harvest_paths(config, n_examples=args.n, noise=args.noise, seed0=args.seed,
                       n_load_steps=args.steps, config_path=args.config,
                       progress_every=max(1, args.n // 20))
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
    from nff.scripts.figures.plot_tessellation_grid import build_grid_figure
    build_distribution_figure(ens, os.path.join(out_dir, "distribution.png"), label=name)
    build_grid_figure(ds, os.path.join(out_dir, "grid.png"), side=args.grid or None)
    print(f"\n  wrote {out_dir}/{{paths.npz, manifest.json, distribution.json, "
          f"distribution.png, grid.png}}\n")


if __name__ == "__main__":
    main()
