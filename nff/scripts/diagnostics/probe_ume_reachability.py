"""Can the closed sheet reach a plum-blossom outline, and at what ripple depth.

The closing figure needs a target the GEOMETRY can make and the PHYSICS then fails to hold. That
only works if the kinematics reach the target cleanly, so the ripple depth is chosen by
measurement here rather than by eye.

The target is the calibrated circle carrying an n-fold ripple, ``r(t) = R (1 + eps cos(n t))``.
``eps`` crosses a stated boundary: a polar curve is convex where ``r^2 + 2 r'^2 - r r'' > 0``,
which at the worst point reduces to ``1 - (n^2 + 2) eps + (n^2 + 1) eps^2 > 0``, whose roots are
``1 / (n^2 + 1)`` and ``1``. So the outline is strictly convex iff ``eps < 1 / (n^2 + 1)`` and
re-entrant above it.

⚠ THE SHEET IS 4-FOLD SYMMETRIC AND A 5-PETAL TARGET IS NOT. That mismatch is the reachability
risk this probe exists to measure, which is why ``n = 4`` and ``n = 8`` are swept as
symmetry-compatible controls rather than assumed to be equivalent.

Deployment is closed form (1-DOF, machine-precision pin closure), so no equilibrium solve enters
here at all. This measures what geometry PROMISES, nothing else.

Run:
    PYTHONPATH=$(pwd) JAX_PLATFORMS=cpu conda run -n kgnn_mac \
        python nff/scripts/diagnostics/probe_ume_reachability.py \
            --config data/configs/closed/sheet_7x7_circle.yaml
"""

import argparse
import json
import os

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np                                                          # noqa: E402

from nff.closed.hinge_paths import resolve_length_scale                     # noqa: E402
from nff.closed.shield_design import ShieldDesignConfig, optimize_shield    # noqa: E402
from nff.closed.shield_kinematics import build_deploy_plan, pin_residual    # noqa: E402
from nff.config.experiment import load_and_parse_config                     # noqa: E402
from nff.config.targets import get_target_points                            # noqa: E402
from nff.scripts.diagnostics.design_force_blind import (build_design_config,  # noqa: E402
                                                        start_params)
from nff.topology.closed_builder_jax import (build_boundary_edges,          # noqa: E402
                                             build_deploy_structure)
from nff.utils.figstyle import INK, ORANGE, apply_charter, use_agg          # noqa: E402

use_agg()
import matplotlib.pyplot as plt                                             # noqa: E402


def convex_eps(n_petals: int) -> float:
    """Largest ripple depth for which ``r = R (1 + eps cos(n t))`` is still strictly convex."""
    return 1.0 / (n_petals ** 2 + 1.0)


def ume_target(config, n_petals: int, eps: float, n_points: int = 480):
    """The blossom outline in PIPELINE UNITS, plus the normalisation ``optimize_shield`` wants.

    ``build_loss`` forms ``target_unit * scale``, so dividing by the outline's y-extent and
    passing that extent as ``fixed_scale`` reproduces the target exactly. This mirrors
    ``design_force_blind.unit_target`` for the rectangle case.
    """
    topo = config.topology
    cx = float(topo.get('target_cx', config.target.center[0]))
    cy = float(topo.get('target_cy', config.target.center[1]))
    radius = float(topo.get('target_r', config.target.radius))
    cloud = get_target_points({'type': 'ume', 'center': (cx, cy), 'radius': radius,
                               'petals': n_petals, 'eps': eps}, n_points=n_points)
    height = float(cloud[:, 1].max() - cloud[:, 1].min())
    return np.asarray(cloud), np.asarray(cloud) / height, height, (cx, cy, radius)


def run_case(config, cfg_base, n_petals, eps, seeds, theta_start_deg, z_sigma, length_scale,
             schedule, overrides=None):
    """Design to one (n, eps) target by geometry alone and report what it promises."""
    cloud, tgt_unit, height, (cx, cy, radius) = ume_target(config, n_petals, eps)
    cfg = ShieldDesignConfig(**{**cfg_base.__dict__, 'fixed_scale': height,
                                'schedule': schedule, **(overrides or {})})
    struct = build_deploy_structure(cfg.M, cfg.N)
    bnd = build_boundary_edges(struct, cfg.spacing, cfg.spacing_y)
    plan = build_deploy_plan(struct, cfg.M, cfg.N, cfg.spacing, cfg.spacing_y)
    r_init = float(config.topology.get('r_init', 0.45))
    psi0 = np.radians(theta_start_deg) / 2.0

    runs = []
    for seed in seeds:
        p0 = start_params(cfg, bnd["n_logits"], seed, r_init, z_sigma, psi0)
        res = optimize_shield(cfg, tgt_unit, seed=seed, params0=p0, verbose=False)
        runs.append((seed, res))
    seed, best = min(runs, key=lambda sr: float(sr[1]['metrics']['loss']))

    m = best['metrics']
    residual = float(np.abs(np.asarray(pin_residual(plan, best['forward']['deployed']))).max())
    outline = np.asarray(best['forward']['outline'])
    row = {
        'petals': int(n_petals), 'eps': float(eps),
        'eps_star': convex_eps(n_petals),
        'convex': bool(eps < convex_eps(n_petals)),
        'chosen_seed': int(seed),
        'rel_rms': float(m['rel_rms']),
        'precision_mm': float(m['precision_mm']) * length_scale,
        'recall_mm': float(m['recall_mm']) * length_scale,
        'worst_recall_mm': float(m['worst_recall_mm']) * length_scale,
        'theta_deg': float(np.degrees(2.0 * abs(float(best['forward']['psi'])))),
        'area_cv': float(m['area_cv']),
        'min_quality': float(m['min_quality']),
        'r_min': float(np.asarray(best['forward']['r']).min()),
        'r_max': float(np.asarray(best['forward']['r']).max()),
        'pin_residual_max': residual,
        'target_mean_radius_mm': radius * length_scale,
    }
    return row, cloud, outline, np.asarray(best['forward']['deployed'])


def contact_sheet(cases, out_png):
    """One panel per case: what the geometry makes against what was asked for."""
    apply_charter()
    n = len(cases)
    ncol = min(4, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 3.3 * nrow), squeeze=False)
    for ax in axes.ravel():
        ax.axis('off')
    for k, (row, cloud, outline, deployed) in enumerate(cases):
        ax = axes[k // ncol][k % ncol]
        ax.axis('off')
        ax.set_aspect('equal')
        # Place the design on the target the way the metric does: both are centred, since the
        # design's translation is a free variable that the supports fix later.
        d = deployed - deployed.reshape(-1, 2).mean(axis=0)
        c = cloud - cloud.mean(axis=0)
        o = outline - outline.mean(axis=0)
        for quad in d:
            ax.fill(quad[:, 0], quad[:, 1], facecolor=ORANGE, edgecolor='white', lw=0.3, alpha=0.9)
        ax.plot(np.append(c[:, 0], c[0, 0]), np.append(c[:, 1], c[0, 1]),
                color=INK, lw=1.0, ls='--')
        ax.plot(np.append(o[:, 0], o[0, 0]), np.append(o[:, 1], o[0, 1]), color=INK, lw=0.8)
        flag = 'convex' if row['convex'] else 're-entrant'
        ax.set_title(f"n={row['petals']}  eps={row['eps']:.3f}  ({flag})\n"
                     f"rel_rms {row['rel_rms']:.4f}   worst rec {row['worst_recall_mm']:.0f} mm",
                     fontsize=7)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    print(f"  wrote {out_png}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="data/configs/closed/sheet_7x7_circle.yaml")
    ap.add_argument("--petals", type=int, nargs="*", default=[5])
    ap.add_argument("--eps", type=float, nargs="*",
                    default=[0.0, 0.02, 0.0385, 0.06, 0.10, 0.15])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0])
    ap.add_argument("--theta-start-deg", type=float, default=85.0,
                    help="the 7x7 sheet is calibrated to ~89.4 deg, so the latent starts near it")
    ap.add_argument("--z-sigma", type=float, default=0.05)
    ap.add_argument("--steps", type=int, nargs=3, default=[400, 600, 300],
                    help="probe schedule; the full design run uses the ShieldDesignConfig default")
    ap.add_argument("--w-quality", type=float, default=0.0,
                    help="one-sided quad-regularity floor. 0 reproduces the paper's force-blind "
                         "control exactly; ~5 keeps tiles off slivers at a small cost in match")
    ap.add_argument("--q-min", type=float, default=0.30)
    ap.add_argument("--area-frac-min", type=float, default=0.25)
    ap.add_argument("--w-uniform", type=float, default=0.0,
                    help="one-sided tile-area floor, as a fraction of the mean")
    ap.add_argument("--out", default="data/outputs/ume_probe")
    args = ap.parse_args()

    config = load_and_parse_config(args.config)
    cfg_base = build_design_config(config)
    length_scale = resolve_length_scale(config)
    schedule = ((args.steps[0], 3e-2, 0.0), (args.steps[1], 3e-2, 1.0), (args.steps[2], 5e-3, 1.0))
    overrides = {'w_quality': args.w_quality, 'q_min': args.q_min,
                 'w_uniform': args.w_uniform, 'area_frac_min': args.area_frac_min}

    os.makedirs(args.out, exist_ok=True)
    print(f"[ume probe] sheet {cfg_base.M}x{cfg_base.N}  length_scale {length_scale:.1f} mm/u  "
          f"schedule {sum(args.steps)} steps  seeds {args.seeds}", flush=True)
    for n in args.petals:
        print(f"           n={n}: convex iff eps < 1/(n^2+1) = {convex_eps(n):.6f}")

    cases, rows = [], []
    for n in args.petals:
        for eps in args.eps:
            row, cloud, outline, deployed = run_case(
                config, cfg_base, n, eps, args.seeds, args.theta_start_deg, args.z_sigma,
                length_scale, schedule, overrides)
            rows.append(row)
            cases.append((row, cloud, outline, deployed))
            flag = 'convex' if row['convex'] else 're-entrant'
            print(f"  n={n} eps={eps:.4f} ({flag:>10s})  rel_rms {row['rel_rms']:.4f}  "
                  f"prec {row['precision_mm']:6.1f} mm  rec {row['recall_mm']:6.1f} mm  "
                  f"worstRec {row['worst_recall_mm']:6.1f} mm  theta {row['theta_deg']:.1f} deg  "
                  f"pin {row['pin_residual_max']:.1e}", flush=True)

    with open(os.path.join(args.out, "probe.json"), "w") as fh:
        json.dump({'config': args.config, 'length_scale_mm': length_scale,
                   'schedule_steps': list(args.steps), 'seeds': list(args.seeds),
                   'overrides': overrides, 'rows': rows}, fh, indent=1)
    contact_sheet(cases, os.path.join(args.out, "probe.png"))
    print(f"  wrote {args.out}/probe.json")


if __name__ == "__main__":
    main()
