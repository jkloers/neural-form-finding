"""Score several designs on one problem, so a trained design and a control can be compared.

Every number here is read off a fresh solve of the saved design, never off the training history.
Two reasons, both load-bearing:

⚠ ``loss_history.json`` does not describe ``best_params.pkl``. ``run_closed`` selects with
``best = (tot, state.params, ch)`` where ``state`` is the POST-update state and ``aux`` the metrics
evaluated BEFORE that update, so the saved design sits one Adam step past its logged row.

⚠ A design trained without a path penalty has no hinge metrics at all. ``build_hinge_model``
builds the ROM kinematic probe only when a probe-backed weight is already non-zero, so a control run
with ``compression: 0`` logs no ``eta_a`` / ``eta_s``. ``extract_hinge_paths`` reads them off the
corotated hinge frame of a solved state instead, which is independent of whatever objective found
the design -- the only way a control and a trained design can be scored on one ruler.

The shape metric is the one the runs were trained on: the one-sided precision chamfer from the
deployed boundary VERTICES to a 400-point sample of the fixed target rectangle (``coverage: 0`` in
every figmat config, so ``chamfer_total`` IS the precision term). Scoring the uniform design must
therefore reproduce a run's epoch-0 ``chamfer_total`` exactly, which ``--check`` asserts.

Run:
    PYTHONPATH=$(pwd) JAX_PLATFORMS=cpu conda run -n kgnn_mac \
        python nff_paper/analysis/score_designs.py \
            --run data/outputs/runs/run_20260806_161121_figmat_1_deploy \
            --case untrained=data/configs/closed/figmat_1_deploy.yaml \
            --out data/outputs/geometric_baseline.json
"""

import argparse
import json
import os
import pickle
import time

import jax
jax.config.update("jax_enable_x64", True)   # ⚠ before any solve: float32 moves the hinge states

import numpy as np                                                          # noqa: E402

from nff.closed.deploy import _boundary_cloud                               # noqa: E402
from nff.hinge.paths import (build_hinge_geometry, extract_hinge_paths,               # noqa: E402
                                    resolve_length_scale)
from nff.analysis.load_paths import extract_force_field, solve_deployed_state  # noqa: E402
from nff_paper.closed_support.one_dof import fit_configuration, fit_one_dof               # noqa: E402
from nff.closed.setup import build_closed_initial_state, init_closed_les_params  # noqa: E402
from nff_paper.closed_support.shield_kinematics import (_checkerboard_signs,              # noqa: E402
                                          build_deploy_plan, deploy_tiles, pin_residual)
from nff.topology.closed_builder_jax import build_deploy_structure          # noqa: E402
from nff.config.experiment import load_and_parse_config                     # noqa: E402
from nff.config.targets import get_target_points                            # noqa: E402

TARGET_POINTS = 400          # must match run_closed's own sampling of the fixed rectangle


# ══════════════════════════════════════════════════════════════════════════════
# The three measurement blocks
# ══════════════════════════════════════════════════════════════════════════════

def target_spec(config) -> dict:
    """The FIXED target this config declares, as a spec ``get_target_points`` consumes.

    The shape is taken from ``config.target.type`` and its extents from the ``tessellation``
    block, which is where a fixed target has always been written (``target_half_w`` for the
    rectangle). An auto-fitted target is refused: this tool compares designs against ONE target,
    and a target that adapts to each deploy is not one.
    """
    topo = config.topology
    kind = str(config.target.type).lower()

    if kind == 'ume':
        missing = [k for k in ('target_cx', 'target_cy', 'target_r') if k not in topo]
        if missing:
            raise KeyError(f"config declares a ume target but carries no fixed outline "
                           f"(missing {missing})")
        return {'type': 'ume',
                'center': np.array([float(topo['target_cx']), float(topo['target_cy'])]),
                'radius': float(topo['target_r']),
                'petals': int(topo.get('target_petals', 5)),
                'eps': float(topo.get('target_eps', 0.0))}

    missing = [k for k in ('target_cx', 'target_cy', 'target_half_w', 'target_half_h')
               if k not in topo]
    if missing:
        raise KeyError(f"config carries no fixed target rectangle (missing {missing}); this tool "
                       "compares designs against ONE target and cannot score an auto-fitted one")
    hw, hh = float(topo['target_half_w']), float(topo['target_half_h'])
    return {'type': 'rectangle',
            'center': np.array([float(topo['target_cx']), float(topo['target_cy'])]),
            'radius': max(hw, hh), 'half_w': hw, 'half_h': hh,
            'rot_deg': float(topo.get('target_rot_deg', 0.0) or 0.0)}


def target_record(spec: dict) -> dict:
    """The target as it goes into the JSON: every field that defines it, and no invented ones."""
    cx, cy = float(spec['center'][0]), float(spec['center'][1])
    if spec['type'] == 'ume':
        return {'shape': 'ume', 'cx': cx, 'cy': cy, 'radius': float(spec['radius']),
                'petals': int(spec['petals']), 'eps': float(spec['eps'])}
    return {'shape': 'rectangle', 'cx': cx, 'cy': cy,
            'half_w': float(spec['half_w']), 'half_h': float(spec['half_h']),
            'rot_deg': float(spec['rot_deg'])}


def target_cloud(config, n_points: int = TARGET_POINTS):
    """The fixed target, sampled exactly as ``run_closed`` samples it.

    Returns:
        ``(cloud, spec)`` with ``spec`` as returned by :func:`target_spec`.
    """
    spec = target_spec(config)
    return np.asarray(get_target_points(spec, n_points=n_points), dtype=float), spec


def shape_metrics(boundary, target, length_scale: float) -> dict:
    """One-sided precision chamfer and its millimetre reading.

    ``chamfer_total`` is a MEAN SQUARED distance in pipeline units, so the millimetre figure is
    ``sqrt(chamfer) * length_scale`` -- an RMS boundary error, not a mean one.
    """
    sq = ((boundary[:, None, :] - target[None, :, :]) ** 2).sum(axis=-1)
    precision = float(sq.min(axis=1).mean())
    coverage = float(sq.min(axis=0).mean())
    lo, hi = boundary.min(axis=0), boundary.max(axis=0)
    return {
        'chamfer_precision': precision,
        'chamfer_coverage': coverage,                 # reported, never scored: coverage = 0 here
        'rms_boundary_mm': float(np.sqrt(precision) * length_scale),
        'bbox_w': float(hi[0] - lo[0]), 'bbox_h': float(hi[1] - lo[1]),
        'bbox_w_mm': float((hi[0] - lo[0]) * length_scale),
        'bbox_h_mm': float((hi[1] - lo[1]) * length_scale),
    }


def _flat_tiles(valid_state) -> np.ndarray:
    """(n_faces, 4, 2) undeployed panel corners, in the physics pipeline's own face/corner order."""
    return (np.asarray(valid_state.face_centroids)[:, None, :]
            + np.asarray(valid_state.centroid_node_vectors))


def _boundary_of(tiles: np.ndarray, valid_state) -> np.ndarray:
    """The same boundary-vertex gather ``_boundary_cloud`` applies to a deployed state."""
    b = np.asarray(valid_state.boundary_face_node_ids)
    return tiles[b[:, 0], b[:, 1]]


def _fit_translation(points: np.ndarray, target: np.ndarray, n_iter: int = 12):
    """Translation minimising the one-sided precision chamfer, by nearest-point iteration.

    The kinematic configuration carries no supports, so its placement is free and must be chosen
    before it can be scored. Choosing the placement that flatters it is deliberate: the baseline is
    then reported at its best, and any shortfall cannot be blamed on where it was put.
    """
    t = target.mean(axis=0) - points.mean(axis=0)
    for _ in range(n_iter):
        moved = points + t
        sq = ((moved[:, None, :] - target[None, :, :]) ** 2).sum(axis=-1)
        t = t + (target[sq.argmin(axis=1)] - moved).mean(axis=0)
    return t


def kinematic_promise(valid_state, displacement, target, length_scale, plan) -> dict:
    """What this design claims under IDEAL uniform opening, on the ruler the loaded solve is scored on.

    The literature's geometric inverse design deploys a cut pattern by opening every hinge at once:
    one global angle, rigid panels, perfect pins. That is exactly ``deploy_tiles``. Scoring it here
    with the same boundary-vertex chamfer against the same target makes the design's own claim and
    its behaviour under discrete actuation directly comparable, which is the whole question.

    Deploying the PHYSICS flat tiles sidesteps the corner-ordering question entirely: the tiles are
    already in the pipeline's face/corner order, so the boundary gather is the identical one.

    Two figures, and they answer different questions:

    ``best`` scans the opening angle and places the result optimally. This is the design's claim:
    the best any member of its own 1-DOF family can do against the target.

    ``at_delivered`` is the ideal configuration nearest the LOADED state, rigidly aligned onto it by
    ``fit_configuration``. It is the counterfactual sheet that deployed exactly as a mechanism to
    the same extent, in the place the supports put it, so the difference between it and the delivery
    is the part of the shape error that the kinematic model cannot see.
    """
    flat = _flat_tiles(valid_state)
    pitch = float(np.linalg.norm(flat[1].mean(axis=0) - flat[0].mean(axis=0)))

    best = None
    for psi in np.linspace(-np.pi / 4, -0.05, 120):
        pts = _boundary_of(np.asarray(deploy_tiles(plan, flat, float(psi))), valid_state)
        t = _fit_translation(pts, target)
        sq = ((pts + t)[:, None, :] - target[None, :, :]) ** 2
        precision = float(sq.sum(axis=-1).min(axis=1).mean())
        if best is None or precision < best[0]:
            lo, hi = (pts + t).min(axis=0), (pts + t).max(axis=0)
            best = (precision, float(psi), float(hi[0] - lo[0]), float(hi[1] - lo[1]))

    actual = _deployed_tiles(valid_state, displacement)
    fit = fit_configuration(plan, flat, actual)
    ideal = np.asarray(fit['nodes'])
    sq = (_boundary_of(ideal, valid_state)[:, None, :] - target[None, :, :]) ** 2
    at_delivered = float(sq.sum(axis=-1).min(axis=1).mean())
    dev = np.linalg.norm(actual - ideal, axis=-1) / pitch

    return {
        'best_chamfer': best[0],
        'best_rms_mm': float(np.sqrt(best[0]) * length_scale),
        'best_theta_deg': float(np.degrees(2.0 * abs(best[1]))),   # hinge angle = 2 psi
        'best_bbox_w_mm': float(best[2] * length_scale),
        'best_bbox_h_mm': float(best[3] * length_scale),
        'at_delivered_chamfer': at_delivered,
        'at_delivered_rms_mm': float(np.sqrt(at_delivered) * length_scale),
        'at_delivered_theta_deg': float(np.degrees(2.0 * abs(fit['psi']))),
        'one_dof_dev_rms_pitch': float(fit['rmse'] / pitch),
        'one_dof_dev_max_pitch': float(dev.max()),
        'one_dof_dev_rms_mm': float(fit['rmse'] * length_scale),
        'pin_residual_max': float(np.abs(np.asarray(pin_residual(plan, ideal))).max()),
    }


def _deployed_tiles(valid_state, displacement) -> np.ndarray:
    """(n_faces, 4, 2) deployed panel corners, by the same rigid-tile kinematics as the boundary."""
    fc = np.asarray(valid_state.face_centroids) + np.asarray(displacement)[:, :2]
    th = np.asarray(displacement)[:, 2]
    cnv = np.asarray(valid_state.centroid_node_vectors)
    ct, st = np.cos(th)[:, None], np.sin(th)[:, None]
    rot = np.stack([ct * cnv[..., 0] - st * cnv[..., 1],
                    st * cnv[..., 0] + ct * cnv[..., 1]], axis=-1)
    return fc[:, None, :] + rot


def deployment_metrics(displacement, paths, sigma) -> dict:
    """How far the sheet opened, and how much of that is the 1-DOF mechanism.

    ``swing`` is the mean face rotation. The deployment mode counter-rotates neighbours and averages
    to ~0, so a large mean is the sheet turning bodily about its supports rather than deploying.
    """
    theta = np.abs(paths.eta[-1, :, 2])               # endpoint hinge angles, rad
    fit = fit_one_dof(np.asarray(displacement)[:, 2], sigma)
    return {
        'theta_max_deg': float(np.degrees(theta.max())),
        'theta_mean_deg': float(np.degrees(theta.mean())),
        'theta_min_deg': float(np.degrees(theta.min())),
        'theta_spread_deg': float(np.degrees(theta.max() - theta.min())),
        'swing_deg': float(np.degrees(fit.swing)),
        'psi_deg': float(np.degrees(fit.psi)),
        'one_dof_rms_deg': float(np.degrees(fit.rms)),
        'one_dof_rms_over_psi': float(fit.rms_over_psi),
        'one_dof_r2': float(fit.r2),
    }


def ligament_metrics(paths) -> dict:
    """The hinge-strain state the design puts its ligaments through, over the WHOLE load path.

    ``eta_a = a / w_lig`` is compressive where negative: a ligament being pushed shorter than its
    manufactured width, which is the buckling regime the linear-spring reference model cannot
    represent and the surrogate was never trained on. Path, not endpoint: compression that appears
    mid-deployment and relaxes has still happened to the sheet.
    """
    # Step 0 is the prepended undeformed origin, where every hinge sits at exactly zero. It belongs
    # in the path but not in a population fraction: including it dilutes every share by 1/(n+1) and
    # the figures then disagree with the training aux by exactly that factor.
    eta_a, eta_s = paths.eta[1:, :, 0], paths.eta[1:, :, 1]
    a_mm = paths.u[1:, :, 0]
    hinge_min = eta_a.min(axis=0)                     # (n_hinges,) most compressive value reached
    return {
        'n_hinges': int(paths.n_hinges),
        'min_eta_a': float(eta_a.min()),
        'compressive_sample_frac': float(np.mean(eta_a < -1e-6)),
        'n_hinges_compressive': int((hinge_min < -1e-6).sum()),
        'frac_hinges_compressive': float((hinge_min < -1e-6).mean()),
        'worst_compression_mm': float(a_mm.min()),
        'mean_compression_depth_mm': float(-np.minimum(a_mm.min(axis=0), 0.0).mean()),
        'max_abs_eta_s': float(np.abs(eta_s).max()),
        'mean_abs_eta_s': float(np.abs(eta_s).mean()),
        'w_lig_mm': float(np.asarray(paths.w_lig).mean()),
    }


# ══════════════════════════════════════════════════════════════════════════════
# One case
# ══════════════════════════════════════════════════════════════════════════════

def score(config_path: str, params_path, label: str, *, load_scale: float = 1.0,
          check_equilibrium: bool = True) -> dict:
    """Solve one design on one problem and measure it. ``params_path=None`` = the uniform design.

    ``load_scale`` multiplies every applied load. Each row's load was bisected so that the UNIFORM
    sheet reaches the shared target width, so a design that is stiffer or softer than uniform lands
    somewhere else at the same force. Scaling separates "this design cannot do the job" from "this
    design was handed the wrong force".
    """
    t0 = time.time()
    config = load_and_parse_config(config_path)
    if load_scale != 1.0:
        config.topology['loads'] = [{**spec, 'value': float(spec['value']) * load_scale}
                                    for spec in config.topology['loads']]
    results, _, bond_energy, hinge_geometry, cut_axis = solve_deployed_state(config, params_path)

    # The hinge frame needs the design and the UNDEPLOYED state; both are cheap to rebuild and
    # neither is returned by solve_deployed_state.
    initial_state, _ = build_closed_initial_state(config)
    params, static_features = init_closed_les_params(config)
    if params_path:
        with open(params_path, 'rb') as fh:
            params = {k: np.asarray(v) for k, v in pickle.load(fh).items()}

    length_scale = resolve_length_scale(config)
    geometry = build_hinge_geometry(config, static_features, initial_state, params)
    paths = extract_hinge_paths(results['solution'], results['valid_state'], geometry, length_scale,
                                reference_bond_vectors=results.get('reference_bond_vectors'))

    cloud, spec = target_cloud(config)
    boundary = _boundary_cloud(results['valid_state'], results['solution'].fields[-1])
    M, N = int(config.topology['M']), int(config.topology['N'])
    sigma = _checkerboard_signs(M, N).astype(float)

    plan = build_deploy_plan(build_deploy_structure(M, N), M, N,
                             float(config.topology.get('spacing', 1.0)),
                             float(config.topology.get('spacing_y',
                                                       config.topology.get('spacing', 1.0))))
    displacement = np.asarray(results['solution'].fields[-1])

    row = {
        'label': label,
        'config': config_path,
        'params': params_path or 'uniform',
        'target': target_record(spec),
        'length_scale_mm': float(length_scale),
        'load_scale': float(load_scale),
        'loads': list(config.topology.get('loads', [])),
        'shape': shape_metrics(boundary, cloud, length_scale),
        'promise': kinematic_promise(results['valid_state'], displacement, cloud, length_scale, plan),
        'deployment': deployment_metrics(displacement, paths, sigma),
        'ligament': ligament_metrics(paths),
        'wall_s': None,
    }

    # Equilibrium is not a formality here: a control design is being pushed by a force calibrated
    # for a different sheet, and an unconverged solve would make every hinge number fiction.
    if check_equilibrium:
        try:
            field = extract_force_field(results, config.physics, bond_energy_fn=bond_energy,
                                        hinge_geometry=hinge_geometry, cut_axis=cut_axis)
            row['equilibrium'] = {'converged': bool(field.checks.get('converged', False)),
                                  'residual': float(field.checks.get('equilibrium_residual', np.nan))}
        except Exception as exc:                      # a diagnostic must not sink the measurement
            row['equilibrium'] = {'converged': None, 'error': f"{type(exc).__name__}: {exc}"}

    row['wall_s'] = round(time.time() - t0, 1)
    return row


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_case(spec: str):
    """``LABEL=CONFIG[,PARAMS]`` -> ``(label, config, params_or_None)``."""
    if '=' not in spec:
        raise ValueError(f"--case wants LABEL=CONFIG[,PARAMS], got {spec!r}")
    label, rest = spec.split('=', 1)
    config, _, params = rest.partition(',')
    return label, config, (params or None)


def _from_run(run_dir: str):
    """A shipped run directory -> the case that reproduces it."""
    name = os.path.basename(run_dir.rstrip('/'))
    label = name.split('_figmat_')[-1] if '_figmat_' in name else name
    return (f"trained:{label}", os.path.join(run_dir, "config.yaml"),
            os.path.join(run_dir, "best_params.pkl"))


def _print_table(rows):
    head = (f"{'label':<26} {'PROMISE':>8} {'DELIVER':>8} {'gap x':>6} {'ideal@':>7} "
            f"{'1DOF mm':>8} {'th prom':>8} {'th deliv':>8} {'spread':>7} {'comp%':>7} {'eq':>3}")
    print("\n" + head)
    print("-" * len(head))
    for r in rows:
        s, p, d, l = r['shape'], r['promise'], r['deployment'], r['ligament']
        eq = r.get('equilibrium', {}).get('converged')
        gap = s['rms_boundary_mm'] / p['best_rms_mm'] if p['best_rms_mm'] else float('nan')
        print(f"{r['label']:<26} {p['best_rms_mm']:8.1f} {s['rms_boundary_mm']:8.1f} {gap:6.2f} "
              f"{p['at_delivered_rms_mm']:7.1f} {p['one_dof_dev_rms_mm']:8.2f} "
              f"{p['best_theta_deg']:8.2f} {d['theta_mean_deg']:8.2f} {d['theta_spread_deg']:7.2f} "
              f"{100 * l['compressive_sample_frac']:7.2f} "
              f"{'ok' if eq else ('--' if eq is None else 'NO'):>3}")
    print("\nPROMISE = best the design's own uniform-opening 1-DOF family reaches, placed optimally.")
    print("DELIVER = same metric on the loaded equilibrium.  ideal@ = ideal mechanism nearest the")
    print("loaded state, where the supports put it.  1DOF mm = how far the loaded state sits from it.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", nargs="*", default=[],
                    help="shipped run directory; scores its best_params.pkl on its own config.yaml")
    ap.add_argument("--case", nargs="*", default=[],
                    help="LABEL=CONFIG[,PARAMS]; PARAMS omitted scores the uniform design")
    ap.add_argument("--out", default="data/outputs/geometric_baseline.json")
    ap.add_argument("--load-scale", type=float, default=1.0,
                    help="multiply every applied load; separates a bad design from a bad force")
    ap.add_argument("--no-equilibrium", action="store_true",
                    help="skip the equilibrium residual check (one extra force-field extraction)")
    ap.add_argument("--check", nargs="*", default=[], metavar="LABEL=VALUE",
                    help="assert a scored label's chamfer_precision equals VALUE (the run's own "
                         "epoch-0 chamfer_total, for the uniform design)")
    args = ap.parse_args()

    cases = [_from_run(d) for d in args.run] + [_parse_case(c) for c in args.case]
    if not cases:
        ap.error("nothing to score: pass --run and/or --case")

    rows = []
    for label, config_path, params_path in cases:
        print(f"\n[score] {label}  config={config_path}  params={params_path or 'uniform'}",
              flush=True)
        rows.append(score(config_path, params_path, label, load_scale=args.load_scale,
                          check_equilibrium=not args.no_equilibrium))
        r = rows[-1]
        print(f"  {r['shape']['rms_boundary_mm']:.1f} mm rms  |  theta mean "
              f"{r['deployment']['theta_mean_deg']:.2f} deg  |  min eta_a "
              f"{r['ligament']['min_eta_a']:.4f}  |  {r['wall_s']} s", flush=True)
        jax.clear_caches()

    _print_table(rows)

    by_label = {r['label']: r for r in rows}
    for spec in args.check:
        label, _, value = spec.partition('=')
        got = by_label[label]['shape']['chamfer_precision']
        want = float(value)
        ok = abs(got - want) <= 1e-5 * max(1.0, abs(want))
        print(f"\n[check] {label}: chamfer_precision {got:.6f} vs expected {want:.6f}  "
              f"{'OK' if ok else 'MISMATCH'}")
        if not ok:
            raise SystemExit(f"shape metric does not reproduce the trained runs' own chamfer for "
                             f"{label}: {got!r} != {want!r}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(rows, fh, indent=1)
    print(f"\nWrote {args.out}  ({len(rows)} designs)")


if __name__ == "__main__":
    main()
