"""Multi-start ensembles for the Stage-0 chart ablation.

Answers the question a condition number cannot: do independent starts reach the SAME design,
or merely the same loss, or neither? Three outcomes must be told apart --

    same loss, same design        well-posed
    same loss, DIFFERENT design   flat / gauge-degenerate optimum (benign, but say so)
    different loss                genuine bad local minima

so the ensemble records the converged DESIGN, not only scalars. A basin sweep over the start
perturbation turns "is it well-posed?" into a curve.

    JAX_PLATFORMS=cpu conda run -n kgnn_mac python legacy/experiments/diagnostics/ensemble_charts.py \
        --config closed/sheet_4x8ft_rom --seeds 12 --basin
"""

import argparse
import json
import math
import os
import subprocess
import time
from copy import deepcopy

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np

from nff.config.experiment import load_and_parse_config
from nff.closed.chart_ablation import (
    arm_extra_loss, arm_training_cfg, arm_validity_cfg, closed_start_geometry, deploy_validity,
    deployed_boundary, fixed_target, flat_validity, geom_coords, geom_shape, hinge_gap_operator,
    init_direct_from_closed, project_if_needed, r_interval, rms_displacement,
)
from nff.closed.hinge_path_ensemble import resolve_length_scale
from nff.stages.mapping import apply_closed_les_mapping, apply_direct_mapping
from nff.training.trainer import TrainState, create_train_step

ARMS = ('A', 'B', 'C', 'D', 'E')
HINGE_GAP_W = 1.0e3          # arm C's penalty; see the analytic Pareto in diagnose_chart --phase1
OVERLAP_W = 1.0e3            # arm E's penalty; swept separately to measure the O(1/w) law


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _setup(config, arm, seed, noise, overlap_w=OVERLAP_W):
    """Starting params + everything the trainer needs, for one (arm, seed, noise)."""
    cfg = deepcopy(config)
    cfg.topology = {**cfg.topology, 'init_seed': int(seed), 'init_noise': float(noise)}
    target, cloud = fixed_target(cfg)

    if arm == 'A':
        _, params, sf, state, tess = closed_start_geometry(cfg, seed)
        map_type = 'closed_les'
        training_cfg = arm_training_cfg(cfg, hinge_gap=0.0)
    else:
        params, state, tess = init_direct_from_closed(cfg, seed)
        sf, map_type = None, 'direct_vertices'
        training_cfg = arm_training_cfg(cfg, hinge_gap=HINGE_GAP_W if arm == 'C' else 0.0)
    # Arms D/E earn Stage-1 projection; the returned cfg is what create_train_step and
    # deploy_validity both see, so the arm's mechanism is applied consistently everywhere.
    cfg = arm_validity_cfg(cfg, arm)
    extra_loss_fn = arm_extra_loss(cfg, arm, state, w_overlap=overlap_w)
    return cfg, params, sf, state, tess, map_type, training_cfg, target, cloud, extra_loss_fn


def run_one(config, arm, seed, noise, epochs, overlap_w=OVERLAP_W):
    """Train one (arm, seed, noise) and audit the result."""
    (cfg, params, sf, state, tess, map_type,
     training_cfg, target, cloud, extra_loss_fn) = _setup(config, arm, seed, noise, overlap_w)
    length_scale = resolve_length_scale(cfg)
    n_faces, max_nodes = geom_shape(state)
    n_nodes = n_faces * max_nodes
    B = hinge_gap_operator(state)

    def _geom(p):
        mapped = (apply_closed_les_mapping(state, p, sf) if arm == 'A'
                  else apply_direct_mapping(state, p))
        # The sheet that reaches Stage 2 is the POST-projection one, so that is what gets audited.
        # Auditing the Stage-0 output for arms D/E would report a hinge gap the physics never saw.
        return np.asarray(geom_coords(project_if_needed(mapped, cfg)))

    y_start = _geom(params)
    optimizer, step = create_train_step(
        state, target, cfg.validity, cfg.physics, training_cfg,
        map_type=map_type, use_jit=True, load_specs=cfg.topology.get('loads', []),
        static_features=sf, target_cloud=cloud, extra_loss_fn=extra_loss_fn)
    ts = TrainState(params=params, opt_state=optimizer.init(params), rng=jax.random.PRNGKey(0))

    # `step` returns the loss evaluated at the params BEFORE the update, so the loss must be paired
    # with THOSE params. Pairing it with ts.params afterwards reports one design's loss against a
    # different design's geometry -- negligible once steps are small, but not while they are large.
    best = (math.inf, params)
    t0 = time.time()
    for _ in range(epochs):
        prev = ts.params
        ts, loss, aux = step(ts)
        tot = float(loss)
        if math.isfinite(tot) and tot < best[0]:
            best = (tot, prev)
    best_loss, best_params = best

    y_end = _geom(best_params)
    fv = flat_validity(y_end, state, B=B)
    dv = deploy_validity(cfg, best_params, state, map_type, sf)
    try:
        cloud_out = deployed_boundary(cfg, best_params, state, map_type, sf)
    except Exception:
        cloud_out = np.zeros((0, 2))

    r_lo, r_hi = r_interval(best_params)

    jax.clear_caches()      # forward_pipeline builds a fresh jitted closure per call, never evicted
    return {
        'arm': arm, 'seed': int(seed), 'noise': float(noise), 'epochs': int(epochs),
        'final_loss': best_loss,
        # Containment of the closed chart on its admissible set. The validity theorem lives on the
        # OPEN interval r in (0,1); logging the interval a run actually occupies attaches a measured
        # domain to "the chart is exact" instead of leaving it as an assertion.
        'r_min': r_lo, 'r_max': r_hi,
        'flat_overlap_fraction': fv['overlap_fraction'],
        # Distance from the UNPERTURBED reference sheet. Measuring against
        # closed_start_geometry(cfg) would compare the perturbed start against itself and
        # return zero, silently flattening the basin axis.
        'start_offset_mm': rms_displacement(y_start, np.asarray(
            closed_start_geometry(config)[0]), n_nodes, length_scale),
        'flat_connected': fv['connected'], 'flat_non_overlapping': fv['non_overlapping'],
        'flat_non_inverted': fv['non_inverted'],
        'max_hinge_gap_mm': fv['max_hinge_gap'] * length_scale,
        'deploy_finite': dv['finite'], 'deploy_non_overlapping': dv['non_overlapping'],
        'converged': bool(math.isfinite(best_loss) and fv['valid'] and dv['valid']),
        'y_end': y_end, 'boundary': cloud_out, 'secs': time.time() - t0,
    }


def summarize(rows, length_scale, n_nodes, label):
    """Loss spread AND design spread -- the pair is what separates the three outcomes."""
    print(f"\n  ── {label} ──────────────────────────────────────────────")
    for arm in ARMS:
        rs = [r for r in rows if r['arm'] == arm]
        if not rs:
            continue
        losses = np.array([r['final_loss'] for r in rs])
        ok = np.array([r['converged'] for r in rs])
        Y = np.array([r['y_end'] for r in rs])
        pw = [rms_displacement(Y[i], Y[j], n_nodes, length_scale)
              for i in range(len(Y)) for j in range(i + 1, len(Y))]
        pw = np.array(pw) if pw else np.array([np.nan])
        gaps = np.array([r['max_hinge_gap_mm'] for r in rs])
        # The four criteria are reported SEPARATELY: they have different scales, and collapsing
        # them into one flag hides which one each arm actually violates -- the whole point here.
        conn = np.array([r['flat_connected'] for r in rs], dtype=float)
        novl = np.array([r['flat_non_overlapping'] for r in rs], dtype=float)
        ninv = np.array([r['flat_non_inverted'] for r in rs], dtype=float)
        dep = np.array([r['deploy_non_overlapping'] for r in rs], dtype=float)
        ovf = np.array([r['flat_overlap_fraction'] for r in rs], dtype=float)
        print(f"    arm {arm}  n={len(rs):3d}  loss med {np.median(losses):.4f} "
              f"[{losses.min():.4f}, {losses.max():.4f}]"
              f" | design spread med {np.median(pw):7.1f} mm"
              f" | converged {ok.mean():5.0%}")
        print(f"             closed {conn.mean():5.0%} (gap med {np.median(gaps):9.2e} mm) "
              f" no-self-inter {novl.mean():5.0%} (overlap med {100*np.median(ovf):5.2f}%) "
              f" non-inverted {ninv.mean():5.0%}  deploys {dep.mean():5.0%}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="closed/sheet_4x8ft_rom")
    ap.add_argument("--out", default="data/outputs/chart_ablation")
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--noise", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--overlap-w", type=float, default=OVERLAP_W,
                    help="arm E's non-overlap penalty weight; sweep it to measure the O(1/w) law")
    ap.add_argument("--basin", action="store_true")
    ap.add_argument("--basin-epochs", type=int, default=100)
    ap.add_argument("--basin-seeds", type=int, default=6)
    args = ap.parse_args()

    config = load_and_parse_config(f"data/configs/{args.config}.yaml")
    os.makedirs(args.out, exist_ok=True)
    length_scale = resolve_length_scale(config)
    _, _, _, state0, _ = closed_start_geometry(config)
    n_faces, max_nodes = geom_shape(state0)
    n_nodes = n_faces * max_nodes

    main_rows = []
    for arm in args.arms:
        for seed in range(1, args.seeds + 1):
            r = run_one(config, arm, seed, args.noise, args.epochs, args.overlap_w)
            main_rows.append(r)
            print(f"    {arm} seed {seed:2d}  loss {r['final_loss']:.5f}  "
                  f"conv {int(r['converged'])}  gap {r['max_hinge_gap_mm']:9.2e} mm  "
                  f"overlap {100*r['flat_overlap_fraction']:5.2f}%  ({r['secs']:.0f}s)")
    summarize(main_rows, length_scale, n_nodes, f"multi-start, noise {args.noise}, "
                                                f"{args.epochs} epochs")

    basin_rows = []
    if args.basin:
        for noise in (0.125, 0.25, 0.5, 1.0):
            for arm in args.arms:
                for seed in range(1, args.basin_seeds + 1):
                    basin_rows.append(run_one(config, arm, seed, noise, args.basin_epochs,
                                              args.overlap_w))
            summarize([r for r in basin_rows if r['noise'] == noise],
                      length_scale, n_nodes, f"basin: noise {noise}")

    def _dump(rows, name):
        if not rows:
            return
        np.savez_compressed(
            os.path.join(args.out, name),
            y_end=np.array([r['y_end'] for r in rows]),
            meta=np.array([[ARMS.index(r['arm']), r['seed'], r['noise'], r['final_loss'],
                            r['converged'], r['max_hinge_gap_mm'], r['start_offset_mm'],
                            r['deploy_non_overlapping'], r['flat_connected'],
                            r['flat_non_overlapping'], r['flat_overlap_fraction'],
                            r['flat_non_inverted'], r['r_min'], r['r_max']] for r in rows]),
            cols='arm_idx, seed, noise, final_loss, converged, gap_mm, start_mm, '
                 'deploy_ok, flat_connected, flat_non_overlapping, flat_overlap_fraction, '
                 'flat_non_inverted, r_min, r_max',
            boundaries=np.array([r['boundary'] for r in rows], dtype=object), allow_pickle=True,
            length_scale=length_scale, git_sha=_git_sha())

    _dump(main_rows, "ensemble_main.npz")
    _dump(basin_rows, "ensemble_basin.npz")
    with open(os.path.join(args.out, "ensemble_summary.json"), "w") as fh:
        json.dump([{k: v for k, v in r.items() if k not in ('y_end', 'boundary')}
                   for r in main_rows + basin_rows], fh, indent=2, default=float)
    print(f"\n  wrote {args.out}/ensemble_main.npz (+ basin, + summary json)")


if __name__ == "__main__":
    main()
