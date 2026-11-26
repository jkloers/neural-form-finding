"""Sweep a disturbance against the deployment drive and find where each design stops opening.

The sheet is a ONE-DOF mechanism driven by a single moment. Superposing a transverse load on a
distant panel sets up a competition: below a critical ratio the moment wins and the sheet reaches
its deployed configuration, above it the solve settles in a barely-open configuration and the
deployment is not reached. The ratio at which that happens is a property of the CUT PATTERN, so it
separates designs that merely trace the target from designs that can still be driven to it.

``rho`` is the disturbance in units of the reference disturbance ``--f-ref``, so ``rho`` is the
only quantity swept and the moment is held fixed throughout. The moment must be held: 90 deg is a
kinematic LOCK on this sheet, so raising the drive re-closes the pattern instead of opening it
further.

Every level is an INDEPENDENT incremental solve from the flat sheet, never a continuation, so a
collapse is a property of the load and not of the path the sweep took to it.

Run:
    PYTHONPATH=$(pwd) JAX_PLATFORMS=cpu conda run -n kgnn_mac \
        python nff_paper/analysis/sweep_ume_collapse.py \
            --config data/configs/closed/ume_7x7.yaml \
            --arm uniform= \
            --arm geometry=data/outputs/kinematic_designs/ume_geom/best_params.pkl \
            --out data/outputs/ume_collapse
"""

import argparse
import copy
import json
import os

import jax
jax.config.update("jax_enable_x64", True)   # ⚠ before any solve: float32 moves the hinge states

import numpy as np                                                          # noqa: E402

from nff.closed.deploy import _boundary_cloud                               # noqa: E402
from nff.closed.hinge_paths import resolve_length_scale                     # noqa: E402
from nff.closed.load_paths import extract_force_field, solve_deployed_state  # noqa: E402
from nff_paper.closed_support.one_dof import fit_one_dof                                  # noqa: E402
from nff_paper.closed_support.shield_kinematics import _checkerboard_signs                # noqa: E402
from nff.config.experiment import load_and_parse_config                     # noqa: E402
from nff_paper.analysis.score_designs import (_deployed_tiles, shape_metrics,  # noqa: E402
                                                   target_cloud, target_record, target_spec)


def parse_arm(text: str):
    """``label=/path/to/best_params.pkl``; an empty path means the uniform (untrained) sheet."""
    if '=' not in text:
        raise argparse.ArgumentTypeError(f"--arm wants label=PATH (got {text!r})")
    label, path = text.split('=', 1)
    return label, (path or None)


def with_disturbance(config, face: int, dof: int, value: float):
    """A copy of the config carrying the drive it already has plus one transverse load."""
    cfg = copy.deepcopy(config)
    loads = [dict(spec) for spec in cfg.topology['loads']]
    loads.append({'face': int(face), 'dof': int(dof), 'value': float(value)})
    cfg.topology['loads'] = loads
    return cfg


def solve_one(config, params_path, cloud, length_scale, M, N):
    """One independent solve, reported as how far it opened and how well it matched.

    ⚠ Equilibrium is checked at every level and not assumed. A collapse is being read off the
    opening angle, and an unconverged solve would produce exactly the same signature as one.
    """
    results, _, bond_energy, hinge_geometry, cut_axis = solve_deployed_state(config, params_path)
    vs = results['valid_state']
    disp = np.asarray(results['solution'].fields[-1])
    sigma = _checkerboard_signs(M, N).astype(float)
    fit = fit_one_dof(disp[:, 2], sigma)
    boundary = _boundary_cloud(vs, results['solution'].fields[-1])
    shape = shape_metrics(boundary, cloud, length_scale)
    row = {
        'psi_deg': float(np.degrees(abs(fit.psi))),
        'theta_deg': float(np.degrees(2.0 * abs(fit.psi))),
        'swing_deg': float(np.degrees(fit.swing)),
        'one_dof_r2': float(fit.r2),
        'one_dof_rms_over_psi': float(fit.rms_over_psi),
        'rms_boundary_mm': shape['rms_boundary_mm'],
        'chamfer_precision': shape['chamfer_precision'],
        'chamfer_coverage': shape['chamfer_coverage'],
    }
    try:
        field = extract_force_field(results, config.physics, bond_energy_fn=bond_energy,
                                    hinge_geometry=hinge_geometry, cut_axis=cut_axis)
        row['converged'] = bool(field.checks.get('converged', False))
        row['equilibrium_residual'] = float(field.checks.get('equilibrium_residual', np.nan))
    except Exception as exc:              # a diagnostic must not sink the measurement
        row['converged'] = None
        row['equilibrium_error'] = f"{type(exc).__name__}: {exc}"
    return row, np.asarray(_deployed_tiles(vs, results['solution'].fields[-1])), np.asarray(boundary)


def collapse_level(levels, theta, drop_frac: float = 0.5):
    """First swept level whose opening has fallen below ``drop_frac`` of the plateau.

    The plateau is the median opening over the lowest quarter of the sweep rather than the value at
    the first level, so one anomalous solve at the bottom cannot define the reference.
    """
    levels, theta = np.asarray(levels, float), np.asarray(theta, float)
    k = max(1, len(theta) // 4)
    plateau = float(np.median(theta[:k]))
    below = np.flatnonzero(theta < drop_frac * plateau)
    return (float(levels[below[0]]) if below.size else None), plateau


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="data/configs/closed/ume_7x7.yaml")
    ap.add_argument("--arm", type=parse_arm, action="append", required=True,
                    help="label=PATH to a best_params.pkl; label= alone is the uniform sheet")
    ap.add_argument("--face", type=int, default=48,
                    help="panel the disturbance acts on; default is the far corner from the clamp")
    ap.add_argument("--dof", type=int, default=1, help="0=x, 1=y, 2=theta")
    ap.add_argument("--f-ref", type=float, default=100.0,
                    help="disturbance at rho = 1, in newtons")
    ap.add_argument("--rho", type=float, nargs=3, default=[0.0, 3.0, 16],
                    metavar=("MIN", "MAX", "N"), help="linear grid of disturbance ratios")
    ap.add_argument("--drop-frac", type=float, default=0.5,
                    help="fraction of the plateau opening below which the deployment is lost")
    ap.add_argument("--out", default="data/outputs/ume_collapse")
    args = ap.parse_args()

    config = load_and_parse_config(args.config)
    length_scale = resolve_length_scale(config)
    cloud, spec = target_cloud(config)
    M, N = int(config.topology['M']), int(config.topology['N'])
    levels = np.linspace(args.rho[0], args.rho[1], int(args.rho[2]))

    os.makedirs(args.out, exist_ok=True)
    print(f"[collapse] {args.config}  target {target_record(spec)}")
    print(f"[collapse] drive {config.topology['loads']}  disturbance face {args.face} dof "
          f"{args.dof} at rho x {args.f_ref} N  levels {levels[0]:.2f}..{levels[-1]:.2f} "
          f"({len(levels)})", flush=True)

    out, store = {}, {}
    for label, path in args.arm:
        rows = []
        for rho in levels:
            cfg = with_disturbance(config, args.face, args.dof, -args.f_ref * float(rho))
            row, tiles, boundary = solve_one(cfg, path, cloud, length_scale, M, N)
            row['rho'] = float(rho)
            rows.append(row)
            store[f"tiles__{label}__{len(rows)-1}"] = tiles
            store[f"boundary__{label}__{len(rows)-1}"] = boundary
            print(f"  {label:>10s}  rho {rho:5.2f}  theta {row['theta_deg']:6.2f} deg  "
                  f"r2 {row['one_dof_r2']:6.3f}  boundary {row['rms_boundary_mm']:7.1f} mm  "
                  f"eq {row.get('converged')}", flush=True)
        lost, plateau = collapse_level(levels, [r['theta_deg'] for r in rows], args.drop_frac)
        out[label] = {'params': path or 'uniform', 'rows': rows,
                      'plateau_theta_deg': plateau, 'lost_at_rho': lost}
        print(f"  {label:>10s}  plateau {plateau:.2f} deg   deployment lost at rho "
              f"{'never in range' if lost is None else f'{lost:.3f}'}", flush=True)

    store['levels'] = levels
    store['target'] = cloud
    np.savez_compressed(os.path.join(args.out, "sweep.npz"), **store)
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump({'config': args.config, 'target': target_record(spec),
                   'length_scale_mm': length_scale, 'face': args.face, 'dof': args.dof,
                   'f_ref': args.f_ref, 'levels': levels.tolist(),
                   'drive': list(config.topology['loads']),
                   'drop_frac': args.drop_frac, 'arms': out}, fh, indent=1)
    print(f"  wrote {args.out}/summary.json and sweep.npz")


if __name__ == "__main__":
    main()
