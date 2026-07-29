"""Extract and visualize the ``(a, s, theta)`` paths a tessellation's hinges ride during deployment.

The oracle that trains the surrogate samples STRAIGHT proportional rays over a guessed box
(``nff/rve/dataset.py:sample_jobs``). This measures what the pipeline actually traverses, so the
sampling can be aimed at it instead. Works on a ROM run -- no surrogate checkpoint needed.

Run:
    PYTHONPATH=$(pwd) JAX_PLATFORMS=cpu conda run -n kgnn_mac \
        python nff/scripts/diagnostics/extract_hinge_paths.py \
            --config data/configs/closed/sheet_4x8ft_rom.yaml --steps 40

Writes into ``data/outputs/hinge_paths/<name>/``:
    hinge_paths.npz     per-hinge (a, s, theta) and (eta_a, eta_s, theta) trajectories
    hinge_paths.json    diagnostics (straightness, monotonicity, domain coverage) + fitted rays
    hinge_paths.png     the paths in the oracle's own coordinates, against the sampled box
"""

import argparse
import copy
import os
import pickle

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
from nff.config.experiment import load_and_parse_config
from nff.stages.pipeline import forward_pipeline
from nff.closed.setup import build_closed_initial_state, init_closed_les_params, build_surrogate_energy
from nff.closed.hinge_paths import (build_hinge_geometry, extract_hinge_paths, path_diagnostics,
                                    fit_deployment_rays, save_paths)
from nff.models.hinge_surrogate import DOMAIN
from nff.scripts.figures.plot_hinge_paths import build_figure


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="path to the closed-pipeline YAML")
    p.add_argument("--steps", type=int, default=40,
                   help="load steps to resolve the path with (more = finer path shape)")
    p.add_argument("--params", default=None,
                   help="best_params.pkl from a trained run; omit to use the untrained design")
    p.add_argument("--out", default=None, help="output dir (default data/outputs/hinge_paths/<name>)")
    p.add_argument("--load-scale", type=float, default=1.0,
                   help="multiply every load by this (probe whether path SHAPE is load-dependent)")
    p.add_argument("--barrier", type=float, default=None,
                   help="override hinge_model.barrier. The OOD barrier penalizes a<0, so setting 0 "
                        "asks whether it is MASKING a compressive regime (surrogate runs only)")
    args = p.parse_args()

    config = load_and_parse_config(args.config)
    name = os.path.splitext(os.path.basename(args.config))[0]
    out_dir = args.out or os.path.join("data", "outputs", "hinge_paths", name)
    os.makedirs(out_dir, exist_ok=True)

    # Resolve the path finely: num_load_steps IS the sampling resolution of the trajectory. The
    # design/target/BCs are untouched, so this is the same deployment, just observed more often.
    config = copy.deepcopy(config)
    config.physics.num_load_steps = int(args.steps)
    if args.barrier is not None and getattr(config, 'hinge_model', None) is not None:
        config.hinge_model.barrier = float(args.barrier)
    if args.load_scale != 1.0:
        # Loads are BAKED INTO the state at build time, so this must happen before
        # build_closed_initial_state -- editing load_specs afterwards silently does nothing.
        config.topology['loads'] = [{**dict(l), 'value': float(l['value']) * args.load_scale}
                                    for l in config.topology.get('loads', [])]

    initial_state, _ = build_closed_initial_state(config)
    params, static_features = init_closed_les_params(config)
    bond_energy, _, geometry_fn, _, w_lig_logit0 = build_surrogate_energy(
        config, static_features, initial_state, params)
    if w_lig_logit0 is not None:
        params = {**params, 'w_lig_logit': w_lig_logit0}
    if args.params:
        with open(args.params, "rb") as f:
            params = pickle.load(f)
        print(f"  design: trained params from {args.params}")
    else:
        print("  design: untrained (config r_init / flat boundary)")

    load_specs = config.topology.get('loads', [])
    result = forward_pipeline(initial_state, config.target, config.validity, config.physics,
                              map_type=config.mapping.type, map_params=params,
                              static_features=static_features, load_specs=load_specs,
                              bond_energy_fn=bond_energy,
                              hinge_geometry=(geometry_fn(params) if geometry_fn else None))

    # length_scale is a pure GEOMETRIC bridge (mm per pipeline unit), so it is meaningful even on the
    # ROM branch where the physics never reads it. Falls back to the real panel pitch if declared.
    hm = getattr(config, 'hinge_model', None)
    ls = float(getattr(hm, 'length_scale', 0.0) or 0.0) if hm is not None else 0.0
    if ls <= 0.0:
        sheet_w = float(config.topology.get('sheet_width_mm', 0.0) or 0.0)
        ls = (sheet_w / (int(config.topology['M']) * float(config.topology.get('spacing', 1.0)))
              if sheet_w > 0 else 1.0)
    geom = build_hinge_geometry(config, static_features, initial_state, params)

    paths = extract_hinge_paths(result['solution'], result['valid_state'], geom, ls,
                                reference_bond_vectors=result.get('reference_bond_vectors'))
    diag = path_diagnostics(paths)
    rays = fit_deployment_rays(paths)
    summary = save_paths(paths, os.path.join(out_dir, "hinge_paths"),
                         meta={'config': args.config, 'n_load_steps': args.steps,
                               'trained_params': args.params,
                               'fitted_rays': [{'tag': r.tag, 'theta1_deg': r.theta1_deg,
                                                'eta_a': r.eta_a, 'eta_s': r.eta_s,
                                                'w_lig_mm': g.w_lig, 'alpha_deg': g.alpha_deg}
                                               for g, r in rays]})
    build_figure([(paths, name)], os.path.join(out_dir, "hinge_paths.png"))

    d = summary['diagnostics']
    print(f"\n  {d['n_hinges']} hinges x {d['n_steps']} steps   length_scale={ls:.1f} mm/unit")
    print(f"  eta_a  {d['eta_a_range'][0]:+.3f} .. {d['eta_a_range'][1]:+.3f}"
          f"   (box {DOMAIN.get('eta_a_min', 0.0)} .. {DOMAIN['eta_a_max']})")
    print(f"  eta_s  {d['eta_s_range'][0]:+.3f} .. {d['eta_s_range'][1]:+.3f}"
          f"   (box +-{DOMAIN['eta_s_max']})")
    print(f"  theta  {d['theta_range_deg'][0]:+.1f} .. {d['theta_range_deg'][1]:+.1f} deg"
          f"   (box {np.degrees(DOMAIN.get('theta_min', -DOMAIN['theta_max'])):+.1f}"
          f" .. {np.degrees(DOMAIN['theta_max']):.1f})")
    print(f"  straightness  mean={d['straightness_mean']:.4f}  max={d['straightness_max']:.4f}"
          f"   (0 = a proportional ray reproduces it exactly)")
    print(f"  monotonic frac  eta_a={d['monotonic_frac']['eta_a']:.2f} "
          f"eta_s={d['monotonic_frac']['eta_s']:.2f} theta={d['monotonic_frac']['theta']:.2f}")
    print(f"  in surrogate domain: {100*d['in_domain_frac']:.1f}% of samples, "
          f"{100*d['endpoint_in_domain_frac']:.1f}% of endpoints")
    c = d['compression']
    print(f"  COMPRESSION  {c['n_hinges_compressive']}/{d['n_hinges']} hinges go a<0 "
          f"(min eta_a={c['min_eta_a']:+.3f}, {100*c['sample_frac']:.1f}% of samples)"
          f"   -- oracle samples eta_a>=0 ONLY")
    occ = d['box_occupancy']
    print(f"  box occupancy  eta_a={100*occ['eta_a']:.0f}%  eta_s={100*occ['eta_s']:.0f}%  "
          f"theta={100*occ['theta']:.0f}%   (low = oracle budget spent off the ridden manifold)")
    print(f"\n  wrote {out_dir}/hinge_paths.{{npz,json,png}}")


if __name__ == "__main__":
    main()
