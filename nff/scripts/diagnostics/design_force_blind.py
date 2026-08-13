"""Form-find a cut pattern against the target using geometry alone, with no force anywhere in it.

The control this produces answers what the physics in the objective actually buys. It is designed
exactly as Section 3.1 designs: the closed sheet deploys in CLOSED FORM as a 1-DOF mechanism
(``shield_kinematics``), so a design can be matched to a target outline with no equilibrium, no
load and no solver. The design therefore never sees the actuation it will later be driven by.

Because every row of the figure matrix shares ONE target rectangle, there is exactly one force-blind
design for all five problems. It is written in the ``closed_les`` parameterisation, so
``score_designs.py`` activates it at each row's own calibrated load with no special case.

The two parameterisations are made to coincide rather than converted approximately:
``shield_design`` uses ``r = 0.5 + 0.5 * (r_hi - r_lo) * tanh(z)``, so setting ``r_lo = 0`` and
``r_hi = 1`` gives ``r = sigmoid(2 z)`` against ``closed_les``' ``r = sigmoid(z)``, and the transfer
is exactly ``z_les = 2 * z_shield``. It also leaves the design free over the whole chart, which is
the bound the trained designs had.

⚠ The design objective is SYMMETRIC (precision + recall + a soft-max worst term), while the
comparison in ``score_designs.py`` is the trained runs' one-sided precision. That asymmetry is
deliberate and necessary: the opening angle is a free design variable here, so a precision-only
objective is minimised by barely opening at all and parking a small sheet inside the rectangle.

Run:
    PYTHONPATH=$(pwd) JAX_PLATFORMS=cpu conda run -n kgnn_mac \
        python nff/scripts/diagnostics/design_force_blind.py \
            --config data/configs/closed/figmat_1_deploy.yaml --seeds 0 1 2
"""

import argparse
import json
import os
import pickle

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np                                                          # noqa: E402

from nff.closed.hinge_paths import resolve_length_scale                     # noqa: E402
from nff.closed.setup import init_closed_les_params                         # noqa: E402
from nff.closed.shield_design import ShieldDesignConfig, init_params, optimize_shield  # noqa: E402
from nff.closed.shield_kinematics import build_deploy_plan, pin_residual    # noqa: E402
from nff.config.experiment import load_and_parse_config                     # noqa: E402
from nff.scripts.diagnostics.score_designs import target_cloud              # noqa: E402
from nff.topology.closed_builder_jax import (build_boundary_edges,          # noqa: E402
                                             build_deploy_structure)

# theta = 2 * psi and the square-tile mechanism locks at theta = 90 deg, so 45 deg is the
# kinematic ceiling rather than a tuned bound. The shield's own 0.45 rad cannot reach the ~70 deg
# opening this sheet is calibrated to.
PSI_MAX = np.pi / 4


def build_design_config(config, **overrides) -> ShieldDesignConfig:
    """A ``ShieldDesignConfig`` for this sheet, in PIPELINE UNITS.

    Units matter for more than tidiness: ``build_deploy_structure`` and ``build_boundary_edges``
    must produce byte-identical structures to the ones ``init_closed_les_params`` builds, or the
    boundary logits do not transfer. Working at ``spacing = 1`` guarantees it.

    ⚠ The ``_mm`` suffixes on ``ShieldDesignConfig``'s barrier lengths and on the metrics it prints
    are therefore pipeline units here. They are inert while the barrier weights are zero.
    """
    topo = config.topology
    M, N = int(topo['M']), int(topo['N'])
    spacing = float(topo.get('spacing', 1.0))
    spacing_y = float(topo.get('spacing_y', spacing))
    cfg = ShieldDesignConfig(
        M=M, N=N, sheet_w=M * spacing, sheet_h=N * spacing_y,
        psi_max=PSI_MAX,
        r_lo=0.0, r_hi=1.0,          # => r = sigmoid(2 z); the closed_les chart, unrestricted
        w_barrier=0.0, w_degen=0.0,  # geometry only: no manufacturability, no homogeneity
        w_quality=0.0, w_uniform=0.0, w_homog=0.0,
        fixed_shift=None,            # translation stays free; the supports place the loaded sheet
    )
    return ShieldDesignConfig(**{**cfg.__dict__, **overrides})


def unit_target(config):
    """The shared target rectangle as ``(target_unit, fixed_scale)``.

    ``build_loss`` forms ``target_unit * scale``, so normalising by the rectangle's height makes
    ``fixed_scale`` the target height and reproduces the rectangle exactly.
    """
    cloud, (cx, cy, hw, hh, rot) = target_cloud(config)
    if rot:
        raise SystemExit(f"target_rot_deg is {rot:+.2f} but this tool normalises by the rectangle's "
                         "y-extent, which is not its height once the rectangle is rotated. Rework "
                         "the normalisation before using a rotated target here.")
    height = 2.0 * hh
    return np.asarray(cloud) / height, float(height), (cx, cy, hw, hh)


def start_params(cfg: ShieldDesignConfig, n_logits: int, seed: int, r_init: float,
                 z_sigma: float, psi0: float) -> dict:
    """A start on the uniform sheet the trained runs began from, opened to a usable angle.

    ``r_init`` is matched to the config's own so the control and the trained designs start from the
    same sheet; the opening angle is started near the calibrated deployment because ``psi`` enters
    through a sigmoid and a start at 36 deg of hinge angle wastes the first stage climbing.
    """
    z_mean = float(np.arctanh(2.0 * r_init - 1.0))       # r = 0.5 + 0.5*tanh(z)
    params = init_params(cfg, n_logits, seed=seed, z_sigma=z_sigma, z_mean=z_mean)
    # psi = -psi_max * sigmoid(psi_raw)
    frac = min(max(abs(psi0) / cfg.psi_max, 1e-3), 1.0 - 1e-3)
    params['psi_raw'] = jax.numpy.asarray(float(np.log(frac / (1.0 - frac))))
    return params


def to_closed_les(result) -> dict:
    """Shield design params -> the ``closed_les`` design the physics pipeline consumes."""
    return {'z': 2.0 * np.asarray(result['params']['z'], dtype=float),
            'bnd_logits': np.asarray(result['params']['bnd_logits'], dtype=float)}


def kinematic_report(cfg: ShieldDesignConfig, result, length_scale: float, rect) -> dict:
    """What the geometry PROMISES: the closed-form deployment this design was chosen for."""
    plan = build_deploy_plan(build_deploy_structure(cfg.M, cfg.N), cfg.M, cfg.N,
                             cfg.spacing, cfg.spacing_y)
    deployed = np.asarray(result['forward']['deployed'])
    outline = np.asarray(result['forward']['outline'])
    lo, hi = outline.min(axis=0), outline.max(axis=0)
    residual = np.asarray(pin_residual(plan, result['forward']['deployed']))
    psi = float(result['forward']['psi'])
    _, _, hw, hh = rect
    return {
        'psi_deg': float(np.degrees(abs(psi))),
        'theta_deg': float(np.degrees(2.0 * abs(psi))),   # hinge angle = 2 psi
        'bbox_w': float(hi[0] - lo[0]), 'bbox_h': float(hi[1] - lo[1]),
        'bbox_w_mm': float((hi[0] - lo[0]) * length_scale),
        'bbox_h_mm': float((hi[1] - lo[1]) * length_scale),
        'target_w_mm': float(2 * hw * length_scale), 'target_h_mm': float(2 * hh * length_scale),
        'precision_mm': float(result['metrics']['precision_mm'] * length_scale),
        'recall_mm': float(result['metrics']['recall_mm'] * length_scale),
        'worst_recall_mm': float(result['metrics']['worst_recall_mm'] * length_scale),
        'rel_rms': float(result['metrics']['rel_rms']),
        'r_min': float(np.asarray(result['forward']['r']).min()),
        'r_max': float(np.asarray(result['forward']['r']).max()),
        'area_cv': float(result['metrics']['area_cv']),
        'pin_residual_max': float(np.abs(residual).max()),
        'n_tiles': int(deployed.shape[0]),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="data/configs/closed/figmat_1_deploy.yaml",
                    help="any row: the sheet, the target and r_init are shared across all five")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2],
                    help="multi-start; the best-scoring design is kept, so the control is the "
                         "geometric method's best shot rather than a strawman")
    ap.add_argument("--z-sigma", type=float, default=0.05)
    ap.add_argument("--theta-start-deg", type=float, default=70.0,
                    help="hinge angle the opening latent starts at (psi = theta / 2)")
    ap.add_argument("--out", default="data/outputs/kinematic_designs/figmat_geom")
    args = ap.parse_args()

    config = load_and_parse_config(args.config)
    cfg = build_design_config(config)
    tgt_unit, scale, rect = unit_target(config)
    cfg = ShieldDesignConfig(**{**cfg.__dict__, 'fixed_scale': scale})

    length_scale = resolve_length_scale(config)                   # mm per pipeline unit

    struct = build_deploy_structure(cfg.M, cfg.N)
    bnd = build_boundary_edges(struct, cfg.spacing, cfg.spacing_y)
    r_init = float(config.topology.get('r_init', 0.45))
    psi0 = np.radians(args.theta_start_deg) / 2.0

    reference, _ = init_closed_les_params(config)                 # shapes the transfer must match
    print(f"[force-blind] sheet {cfg.M}x{cfg.N}  target {2*rect[2]:.4f} x {2*rect[3]:.4f} u  "
          f"({2*rect[2]*length_scale:.0f} x {2*rect[3]*length_scale:.0f} mm)  "
          f"r_init {r_init}  psi_max {np.degrees(cfg.psi_max):.1f} deg", flush=True)

    runs = []
    for seed in args.seeds:
        print(f"\n[force-blind] seed {seed}", flush=True)
        p0 = start_params(cfg, bnd["n_logits"], seed, r_init, args.z_sigma, psi0)
        result = optimize_shield(cfg, tgt_unit, seed=seed, params0=p0, verbose=True, log_every=250)
        runs.append((seed, result))

    seed, best = min(runs, key=lambda sr: float(sr[1]['metrics']['loss']))
    design = to_closed_les(best)
    for key, ref in reference.items():
        got = np.shape(design[key])
        if got != np.shape(ref):
            raise SystemExit(f"design cannot enter the physics pipeline: {key} has shape {got}, "
                             f"closed_les expects {np.shape(ref)}")

    report = kinematic_report(cfg, best, length_scale, rect)
    if report['pin_residual_max'] > 1e-8:
        raise SystemExit(f"the closed-form deployment does not close its pins "
                         f"(max residual {report['pin_residual_max']:.3e}); the design is not a "
                         f"valid 1-DOF configuration and must not be scored")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "best_params.pkl"), "wb") as fh:
        pickle.dump(design, fh)
    payload = {
        'config': args.config, 'chosen_seed': seed, 'seeds': list(args.seeds),
        'promised': report,
        'per_seed': [{'seed': s, 'loss': float(r['metrics']['loss']),
                      'precision_mm': float(r['metrics']['precision_mm'] * length_scale),
                      'recall_mm': float(r['metrics']['recall_mm'] * length_scale),
                      'theta_deg': float(np.degrees(2 * abs(float(r['forward']['psi']))))}
                     for s, r in runs],
    }
    with open(os.path.join(args.out, "report.json"), "w") as fh:
        json.dump(payload, fh, indent=1)

    print(f"\n[force-blind] kept seed {seed}")
    print(f"  PROMISED  {report['bbox_w_mm']:.0f} x {report['bbox_h_mm']:.0f} mm  against target "
          f"{report['target_w_mm']:.0f} x {report['target_h_mm']:.0f} mm")
    print(f"            hinge angle {report['theta_deg']:.2f} deg   r in "
          f"[{report['r_min']:.3f}, {report['r_max']:.3f}]   area CV {report['area_cv']:.3f}")
    print(f"            precision {report['precision_mm']:.1f} mm   recall {report['recall_mm']:.1f} mm"
          f"   worst recall {report['worst_recall_mm']:.1f} mm")
    print(f"            pin closure {report['pin_residual_max']:.2e} (machine precision required)")
    print(f"  Wrote {args.out}/best_params.pkl")


if __name__ == "__main__":
    main()
