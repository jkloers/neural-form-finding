"""Sample the DISTRIBUTION of hinge displacement paths over randomized starting tile positions.

``extract_hinge_paths.py`` measures one deployment. This runs the same setup from many random
starting designs -- Gaussian noise on the cut aspect ratios and boundary sliders, which is what
sets where the tiles sit -- and pools the paths, so the oracle can be aimed at the FAMILY of paths
a tessellation rides rather than at one sheet's worth.

Run:
    PYTHONPATH=$(pwd) JAX_PLATFORMS=cpu conda run -n kgnn_mac \
        python nff/scripts/diagnostics/sample_hinge_paths.py \
            --config data/configs/closed/sheet_4x8ft_rom.yaml --n-samples 32 --noise 0.5

Writes into ``data/outputs/hinge_paths/<name>_ensemble_n<N>_s<noise>/``:
    designs/seed<NNN>.{npz,json}   per-design paths
    distribution.json              pooled marginals, correlations, variance split, sample_jobs spec
    distribution.png               the figure
"""

import argparse
import os

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
from nff.config.experiment import load_and_parse_config
from nff.closed.hinge_path_ensemble import (run_path_ensemble, ensemble_statistics, sampling_spec,
                                            save_ensemble)
from nff.scripts.figures.plot_hinge_path_distribution import build_distribution_figure


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="path to the closed-pipeline YAML")
    p.add_argument("--n-samples", type=int, default=32, help="number of random starting designs")
    p.add_argument("--noise", type=float, default=0.5,
                   help="init_noise: Gaussian sigma on the design logits (z and bnd_logits). "
                        "0.5 moves r=sigmoid(z) over roughly 0.33..0.57 around r_init=0.45")
    p.add_argument("--seed", type=int, default=0, help="init_seed of the first design")
    p.add_argument("--steps", type=int, default=30, help="load steps (path resolution)")
    p.add_argument("--load-scale", type=float, default=1.0, help="multiply every load by this")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    config = load_and_parse_config(args.config)
    name = os.path.splitext(os.path.basename(args.config))[0]
    out_dir = args.out or os.path.join("data", "outputs", "hinge_paths",
                                       f"{name}_ensemble_n{args.n_samples}_s{args.noise:g}")

    print(f"\n  {args.n_samples} random starting designs  init_noise={args.noise}  "
          f"seeds {args.seed}..{args.seed + args.n_samples - 1}  ({args.steps} load steps)\n")
    ens = run_path_ensemble(config, n_samples=args.n_samples, noise=args.noise, seed0=args.seed,
                            n_load_steps=args.steps, load_scale=args.load_scale,
                            config_path=args.config)
    if not ens.samples:
        raise SystemExit("every design failed to deploy -- nothing to summarize")

    summary = save_ensemble(ens, out_dir, meta={'load_scale': args.load_scale})
    build_distribution_figure(ens, os.path.join(out_dir, "distribution.png"), label=name)

    st, spec = summary['statistics'], summary['sampling_spec']
    print(f"\n  {st['n_designs']} designs x {st['n_hinges']} hinges "
          f"= {st['n_endpoints']} rays, {st['n_path_points']} path points"
          + (f"   ({st['n_failed']} designs failed)" if st['n_failed'] else ""))

    print("\n  ENDPOINT MARGINALS (the distribution a DeploymentRay is drawn from)")
    print(f"    {'':10s} {'p1':>8s} {'p25':>8s} {'p50':>8s} {'p75':>8s} {'p99':>8s} {'mean+-std':>16s}")
    for key, lbl in [('eta_a', 'eta_a'), ('eta_s', 'eta_s'), ('theta_deg', 'theta [deg]')]:
        m = st['endpoint_marginals'][key]
        q = m['q']
        print(f"    {lbl:10s} {q['p1']:8.3f} {q['p25']:8.3f} {q['p50']:8.3f} {q['p75']:8.3f} "
              f"{q['p99']:8.3f} {m['mean']:8.3f}+-{m['std']:<6.3f}")
    a = st['alpha_deg']
    print(f"    {'alpha[deg]':10s} {a['q']['p1']:8.3f} {a['q']['p25']:8.3f} {a['q']['p50']:8.3f} "
          f"{a['q']['p75']:8.3f} {a['q']['p99']:8.3f} {a['mean']:8.3f}+-{a['std']:<6.3f}")

    c = st['endpoint_correlation']['matrix']
    print("\n  ENDPOINT CORRELATION (independent LHS over a box assumes these are 0)")
    print(f"    eta_a-eta_s {c[0][1]:+.3f}   eta_a-theta {c[0][2]:+.3f}   eta_s-theta {c[1][2]:+.3f}")

    print("\n  DOES THE RANDOM START MATTER? (intraclass corr: ~0 within-sheet, ~1 per-design clusters)")
    for comp, v in st['variance_decomposition'].items():
        print(f"    {comp:8s} between={v['between_design_var']:.4g}  "
              f"within={v['within_design_var']:.4g}  ICC={v['intraclass_corr']:.3f}")

    cm = st['compression']
    print(f"\n  compression   {100*cm['design_frac_with_any']:.0f}% of designs show a<0 "
          f"({100*cm['hinge_frac']:.0f}% of hinges, min eta_a={cm['min_eta_a']:+.3f})")
    print(f"  straightness  mean={st['straightness']['mean']:.3f}  p99={st['straightness']['q']['p99']:.3f}"
          f"   (0 = one proportional ray reproduces the path exactly)")
    print(f"  monotonic     eta_a={st['monotonic_frac']['eta_a']:.2f} "
          f"eta_s={st['monotonic_frac']['eta_s']:.2f} theta={st['monotonic_frac']['theta']:.2f}")
    occ = st['box_occupancy']
    print(f"  box occupancy eta_a={100*occ['eta_a']:.0f}%  eta_s={100*occ['eta_s']:.0f}%  "
          f"theta={100*occ['theta']:.0f}%   in-domain {100*st['in_domain_frac']:.1f}% of points")

    kw = spec['sample_jobs_kwargs']
    print("\n  MEASURED SAMPLING SPEC -- splat straight into nff.rve.dataset.sample_jobs:")
    print(f"    alpha_deg={tuple(kw['alpha_deg'])}, theta1_deg={tuple(kw['theta1_deg'])},")
    print(f"    eta_a={tuple(kw['eta_a'])}, eta_s={tuple(kw['eta_s'])}, spine_frac={kw['spine_frac']}")
    if spec['box_volume_vs_current'] is not None:
        print(f"    -> {100 * spec['box_volume_vs_current']:.1f}% of the box currently sampled")
    if spec['requires_negative_eta_a']:
        print("    !! eta_a range is NEGATIVE at its low end -- sample_jobs and _domain_barrier both "
              "assume a >= 0 today")
    print(f"\n  PROVISIONAL: {spec['PROVISIONAL']}")
    print(f"\n  wrote {out_dir}/{{distribution.json, distribution.png, designs/}}")


if __name__ == "__main__":
    main()
