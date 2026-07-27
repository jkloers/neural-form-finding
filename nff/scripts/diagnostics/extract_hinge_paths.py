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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nff.config.experiment import load_and_parse_config
from nff.stages.pipeline import forward_pipeline
from nff.closed.setup import build_closed_initial_state, init_closed_les_params, build_surrogate_energy
from nff.closed.hinge_paths import (build_hinge_geometry, extract_hinge_paths, path_diagnostics,
                                    fit_deployment_rays, save_paths)
from nff.models.hinge_surrogate import DOMAIN

ORANGE, GREY, RED = "#F58025", "#6C757D", "#D62828"


def _plot(paths, diag, out_png, title):
    """Three projections of every hinge path, drawn over the oracle's sampled box.

    The box is what ``sample_jobs`` currently covers; the paths are what the pipeline visits. The
    gap between them is the point of the figure -- wherever the box is empty, oracle budget is being
    spent on states the tessellation never reaches.
    """
    eta = paths.eta
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    th_deg = np.degrees(eta[..., 2])
    combos = [(0, 2, r"$\eta_a = a/w_{lig}$", r"$\theta$ [deg]"),
              (1, 2, r"$\eta_s = s/w_{lig}$", r"$\theta$ [deg]"),
              (0, 1, r"$\eta_a = a/w_{lig}$", r"$\eta_s = s/w_{lig}$")]
    lim = {0: (0.0, DOMAIN['eta_a_max']), 1: (-DOMAIN['eta_s_max'], DOMAIN['eta_s_max']),
           2: (0.0, np.degrees(DOMAIN['theta_max']))}

    for ax, (i, j, xl, yl) in zip(axes, combos):
        X = eta[..., i] if i != 2 else th_deg
        Y = th_deg if j == 2 else eta[..., j]
        (x0, x1), (y0, y1) = lim[i], lim[j]
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=GREY, alpha=0.13,
                                   edgecolor=GREY, lw=1.0, ls="--", zorder=0,
                                   label="oracle sampled box"))
        for h in range(paths.n_hinges):
            ax.plot(X[:, h], Y[:, h], "-", color=ORANGE, lw=1.3, alpha=0.85, zorder=2)
            ax.plot(X[-1, h], Y[-1, h], "o", color=RED, ms=4.5, zorder=3)
        # the straight ray the oracle WOULD run for each observed endpoint -- the visual gap between
        # a dashed chord and its solid path is exactly what `straightness` quantifies.
        for h in range(paths.n_hinges):
            ax.plot([0, X[-1, h]], [0, Y[-1, h]], ":", color=GREY, lw=0.9, alpha=0.9, zorder=1)
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.axhline(0, color="k", lw=0.5, alpha=0.3); ax.axvline(0, color="k", lw=0.5, alpha=0.3)
        ax.spines[['top', 'right']].set_visible(False)

    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="path to the closed-pipeline YAML")
    p.add_argument("--steps", type=int, default=40,
                   help="load steps to resolve the path with (more = finer path shape)")
    p.add_argument("--params", default=None,
                   help="best_params.pkl from a trained run; omit to use the untrained design")
    p.add_argument("--out", default=None, help="output dir (default data/outputs/hinge_paths/<name>)")
    args = p.parse_args()

    config = load_and_parse_config(args.config)
    name = os.path.splitext(os.path.basename(args.config))[0]
    out_dir = args.out or os.path.join("data", "outputs", "hinge_paths", name)
    os.makedirs(out_dir, exist_ok=True)

    # Resolve the path finely: num_load_steps IS the sampling resolution of the trajectory. The
    # design/target/BCs are untouched, so this is the same deployment, just observed more often.
    config = copy.deepcopy(config)
    config.physics.num_load_steps = int(args.steps)

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
    _plot(paths, diag, os.path.join(out_dir, "hinge_paths.png"),
          f"observed hinge displacement paths — {name} ({paths.n_hinges} hinges, {args.steps} steps)")

    d = summary['diagnostics']
    print(f"\n  {d['n_hinges']} hinges x {d['n_steps']} steps   length_scale={ls:.1f} mm/unit")
    print(f"  eta_a  {d['eta_a_range'][0]:+.3f} .. {d['eta_a_range'][1]:+.3f}"
          f"   (box 0 .. {DOMAIN['eta_a_max']})")
    print(f"  eta_s  {d['eta_s_range'][0]:+.3f} .. {d['eta_s_range'][1]:+.3f}"
          f"   (box +-{DOMAIN['eta_s_max']})")
    print(f"  theta  {d['theta_range_deg'][0]:+.1f} .. {d['theta_range_deg'][1]:+.1f} deg"
          f"   (box +-{np.degrees(DOMAIN['theta_max']):.1f})")
    print(f"  straightness  mean={d['straightness_mean']:.4f}  max={d['straightness_max']:.4f}"
          f"   (0 = a proportional ray reproduces it exactly)")
    print(f"  monotonic frac  eta_a={d['monotonic_frac']['eta_a']:.2f} "
          f"eta_s={d['monotonic_frac']['eta_s']:.2f} theta={d['monotonic_frac']['theta']:.2f}")
    print(f"  in surrogate domain: {100*d['in_domain_frac']:.1f}% of samples, "
          f"{100*d['endpoint_in_domain_frac']:.1f}% of endpoints")
    occ = d['box_occupancy']
    print(f"  box occupancy  eta_a={100*occ['eta_a']:.0f}%  eta_s={100*occ['eta_s']:.0f}%  "
          f"theta={100*occ['theta']:.0f}%   (low = oracle budget spent off the ridden manifold)")
    print(f"\n  wrote {out_dir}/hinge_paths.{{npz,json,png}}")


if __name__ == "__main__":
    main()
