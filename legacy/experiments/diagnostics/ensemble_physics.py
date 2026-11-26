"""The multi-start experiment again, with the physics on.

Same sheet and same target as the kinematic ensemble, but the sheet is now deployed by solving
Stage-2 equilibrium under a moment rather than by the closed-form one-degree-of-freedom map.
The centre tile is clamped on all three degrees of freedom and a moment drives its neighbour,
calibrated so the sheet reaches full deployment.

The design vector loses a dimension. Kinematically the opening angle was chosen by the
optimiser; here it is an output of the equilibrium solve, so the design is the chart alone.

    JAX_PLATFORMS=cpu conda run -n kgnn_mac python \
        legacy/experiments/diagnostics/ensemble_physics.py --seeds 200 --epochs 150

The question is whether the physics narrows the solution set. The kinematic problem admitted
two families; a physical deployment may admit fewer.
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
from copy import deepcopy

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np                                                   # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

from nff.analysis.chart_ablation import (deploy_validity, flat_validity,  # noqa: E402
                                       geom_coords, geom_shape, hinge_gap_operator,
                                       rms_displacement)
from nff_paper.closed_support.design_ensemble import (build_symmetry, canonical_distance,  # noqa: E402
                                        circle_problem, distance_matrix, problem_statics,
                                        split_families)
from nff.hinge.path_ensemble import resolve_length_scale       # noqa: E402
from nff.closed.setup import _flat_coords_from_design                 # noqa: E402
from nff.config.experiment import load_and_parse_config               # noqa: E402
from legacy.experiments.diagnostics.ensemble_charts import _setup            # noqa: E402
from nff_paper.analysis.validity_ladder import outline_error_mm  # noqa: E402
from nff.stages.mapping import apply_closed_les_mapping               # noqa: E402
from nff.training.trainer import TrainState, create_train_step        # noqa: E402


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       text=True).strip()
    except Exception:                                                 # noqa: BLE001
        return "unknown"


def design_coords(static_features, params, centre, length_scale):
    """Flat cut vertices in mm, centred on the sheet.

    The same quantity the kinematic ensemble compares, and deliberately so: both charts solve
    the identical collinearity system through ``solve_cut_vertices_jax``, so the vertices are
    indexed the same way and the C4 quotient built for one transfers to the other. The charts
    differ only in how they squash the latent into ``r``, which changes what a given parameter
    vector means, never what a design is.
    """
    return np.asarray(_flat_coords_from_design(static_features, params)) * length_scale - centre


def run_one(config, seed, noise, epochs, snapshot_every, symmetry, statics_kin,
            time_cap=None):
    """One random start to one converged design, with the deployed sheet audited."""
    (cfg, params, sf, state, tess, map_type,
     training_cfg, target, cloud, extra_loss_fn) = _setup(config, "A", seed, noise)
    length_scale = resolve_length_scale(cfg)
    n_faces, max_nodes = geom_shape(state)
    n_nodes = n_faces * max_nodes
    gap_op = hinge_gap_operator(state)
    centre = statics_kin["centre"]

    optimizer, step = create_train_step(
        state, target, cfg.validity, cfg.physics, training_cfg, map_type=map_type,
        use_jit=True, load_specs=cfg.topology.get("loads", []), static_features=sf,
        target_cloud=cloud, extra_loss_fn=extra_loss_fn)
    ts = TrainState(params=params, opt_state=optimizer.init(params),
                    rng=jax.random.PRNGKey(0))

    coords_start = design_coords(sf, params, centre, length_scale)
    traj_steps, traj_params, traj_loss = [], [], []
    # The step returns the loss at the params BEFORE its update, so the two are recorded
    # together rather than paired across a step boundary.
    best = (math.inf, params)
    t_seed = time.time()
    for ep in range(epochs):
        # A design that tears or self-overlaps makes the Stage-2 solve ill-conditioned, so
        # L-BFGS runs to solver_maxiter on every load step and one seed can cost more than
        # ten. Such a seed is abandoned rather than allowed to stall the ensemble, and it is
        # COUNTED: dropping the awkward starts silently would bias the sample toward the
        # well-behaved ones, which is the opposite of what a multi-start study is for.
        if time_cap and time.time() - t_seed > time_cap:
            raise TimeoutError(f"seed exceeded {time_cap:.0f} s at epoch {ep}/{epochs}")
        prev = ts.params
        if snapshot_every and ep % snapshot_every == 0:
            traj_steps.append(ep)
            traj_params.append(np.concatenate([np.asarray(prev[k]).reshape(-1)
                                               for k in sorted(prev)]))
        ts, loss, aux = step(ts)
        value = float(loss)
        if snapshot_every and ep % snapshot_every == 0:
            traj_loss.append(value)
        if math.isfinite(value) and value < best[0]:
            best = (value, prev)
    best_loss, best_params = best

    coords_end = design_coords(sf, best_params, centre, length_scale)
    flat = flat_validity(np.asarray(geom_coords(
        apply_closed_les_mapping(state, best_params, sf))), state, B=gap_op)
    dep = deploy_validity(cfg, best_params, state, map_type, sf)

    tgt = config.topology
    err = float("nan")
    if dep.get("finite") and dep.get("boundary") is not None:
        err = outline_error_mm(dep["boundary"], float(tgt["target_cx"]),
                               float(tgt["target_cy"]), float(tgt["target_r"]), length_scale)

    rec = {
        "seed": int(seed), "noise": float(noise), "epochs": int(epochs),
        "final_loss": best_loss,
        "shape_error_mm": err,
        "coords": coords_end, "coords_start": coords_start,
        "params": {k: np.asarray(v) for k, v in best_params.items()},
        "start_params": {k: np.asarray(v) for k, v in params.items()},
        "travel_mm": rms_displacement(coords_end, coords_start, n_nodes, 1.0),
        # The physics arm's own validity audits: a design that tore or overlapped is not a
        # cut pattern, whatever it scored.
        "max_hinge_gap_mm": float(flat["max_hinge_gap"]) * length_scale,
        "flat_overlap_fraction": float(flat["overlap_fraction"]),
        "flat_connected": bool(flat.get("connected", False)),
        "deploy_overlap_fraction": float(dep.get("overlap_fraction", float("nan"))),
        "deploys": bool(dep.get("valid", False)),
    }
    if traj_params:
        rec["traj_steps"] = np.asarray(traj_steps, dtype=int)
        rec["traj_params"] = np.stack(traj_params)
        rec["traj_loss"] = np.asarray(traj_loss)
    jax.clear_caches()          # forward_pipeline builds a fresh jitted closure per call
    return rec


def load_existing(out_dir):
    """Records already on disk, keyed by seed.

    Each seed is written the moment it finishes, so a run that is interrupted keeps every
    design it completed. Resuming reloads them rather than recomputing an hour of physics.
    """
    import glob
    out = {}
    for path in sorted(glob.glob(os.path.join(out_dir, "designs", "seed*.npz"))):
        with np.load(path) as z:
            rec = {k: z[k] for k in z.files}
        for k in ("seed", "epochs"):
            rec[k] = int(rec[k])
        for k in ("final_loss", "shape_error_mm", "travel_mm", "max_hinge_gap_mm",
                  "flat_overlap_fraction", "deploy_overlap_fraction", "noise"):
            if k in rec:
                rec[k] = float(rec[k])
        rec["deploys"] = bool(rec.get("deploys", False))
        out[rec["seed"]] = rec
    return out


def summarise(records, symmetry):
    """Loss spread and design spread, the pair that separates the three outcomes."""
    ends = [r["coords"] for r in records]
    starts = [r["coords_start"] for r in records]
    d_end = distance_matrix(ends, symmetry)
    d_start = distance_matrix(starts, symmetry)
    off = ~np.eye(len(records), dtype=bool)
    err = np.array([r["shape_error_mm"] for r in records], dtype=float)
    labels, fam = split_families(d_end)
    medoid = int(np.argmin(d_end.sum(axis=1)))

    def stats(x):
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        return {"median": float(np.median(x)), "min": float(x.min()), "max": float(x.max()),
                "mean": float(x.mean()), "std": float(x.std())}

    return {
        "n": len(records),
        "shape_error_mm": stats(err),
        "shape_error_spread_pct_of_median": float(
            100 * (np.nanmax(err) - np.nanmin(err)) / np.nanmedian(err)),
        "design_spread_end_mm": stats(d_end[off]),
        "design_spread_start_mm": stats(d_start[off]),
        "spread_ratio_end_over_start": float(np.median(d_end[off]) / np.median(d_start[off])),
        "travel_mm": stats([r["travel_mm"] for r in records]),
        "final_loss": stats([r["final_loss"] for r in records]),
        "max_hinge_gap_mm": float(np.nanmax([r["max_hinge_gap_mm"] for r in records])),
        "max_flat_overlap": float(np.nanmax([r["flat_overlap_fraction"] for r in records])),
        "deploys_frac": float(np.mean([r["deploys"] for r in records])),
        "families": fam,
        "labels": labels.tolist(),
        "medoid_seed": int(records[medoid]["seed"]),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="data/configs/closed/sheet_7x7_circle.yaml")
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--noise", type=float, default=1.0, help="start spread")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--snapshot-every", type=int, default=5)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed-timeout", type=float, default=600.0,
                    help="abandon a seed that exceeds this many seconds")
    ap.add_argument("--resume", action="store_true",
                    help="reuse seeds already written under <out>/designs")
    args = ap.parse_args()

    config = load_and_parse_config(args.config)
    out_dir = args.out or (f"data/outputs/design_cloud/physics_n{args.seeds}"
                           f"_e{args.epochs}_s{args.noise:g}")
    os.makedirs(os.path.join(out_dir, "designs"), exist_ok=True)

    # The C4 quotient is built once from the kinematic problem and reused here: both charts
    # index their cut vertices identically, so the relabelling transfers.
    cfg_kin = circle_problem(M=int(config.topology["M"]), N=int(config.topology["N"]),
                             sheet_mm=float(config.topology["sheet_width_mm"]))
    statics_kin = problem_statics(cfg_kin)
    symmetry = build_symmetry(cfg_kin, statics_kin)

    print(f"  {config.topology['M']}x{config.topology['N']} sheet, "
          f"{float(config.topology['sheet_width_mm']):.1f} mm square, ROM hinge law")
    print(f"  clamped {config.topology.get('bc_clamped')}, "
          f"loads {config.topology.get('loads')}")
    print(f"  symmetry group {sorted(symmetry)} (order {len(symmetry)})")
    print(f"  {args.seeds} starts x {args.epochs} epochs -> {out_dir}\n")

    t0 = time.time()
    done = load_existing(out_dir) if args.resume else {}
    if done:
        print(f"  resuming: {len(done)} seeds already on disk, "
              f"{args.seeds - len(done)} to go\n", flush=True)
    records, failures = list(done.values()), []
    for seed in range(1, args.seeds + 1):
        if seed in done:
            continue
        try:
            rec = run_one(config, seed, args.noise, args.epochs, args.snapshot_every,
                          symmetry, statics_kin, time_cap=args.seed_timeout)
            records.append(rec)
            print(f"    seed {seed:3d}  err {rec['shape_error_mm']:7.2f} mm  "
                  f"loss {rec['final_loss']:8.4f}  gap {rec['max_hinge_gap_mm']:8.2e} mm  "
                  f"deploys {int(rec['deploys'])}  [{time.time() - t0:6.0f}s]", flush=True)
            np.savez_compressed(
                os.path.join(out_dir, "designs", f"seed{seed:03d}.npz"),
                **{k: v for k, v in rec.items()
                   if isinstance(v, np.ndarray) or np.isscalar(v)},
                **{f"params__{k}": v for k, v in rec["params"].items()},
                **{f"start_params__{k}": v for k, v in rec["start_params"].items()})
            if len(records) % 20 == 0:
                with open(os.path.join(out_dir, "summary.json"), "w") as fh:
                    json.dump({"partial": True, **summarise(records, symmetry)}, fh, indent=2)
        except Exception as exc:                                      # noqa: BLE001
            failures.append({"seed": seed, "error": repr(exc)})
            print(f"    seed {seed:3d}  FAILED {exc!r}", flush=True)

    if not records:
        raise SystemExit("every start failed")

    records.sort(key=lambda r: r["seed"])
    summary = summarise(records, symmetry)
    summary["config"] = {"path": args.config, "seeds": args.seeds, "epochs": args.epochs,
                         "noise": args.noise, "git_sha": _git_sha(),
                         "M": int(config.topology["M"]), "N": int(config.topology["N"]),
                         "sheet_mm": float(config.topology["sheet_width_mm"])}
    summary["failures"] = failures
    summary["n_failed"] = len(failures)
    summary["failed_seeds"] = [f["seed"] for f in failures]
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    s = summary
    print(f"\n  ── physics, {s['n']} starts ─────────────────────────────")
    e, de, ds = s["shape_error_mm"], s["design_spread_end_mm"], s["design_spread_start_mm"]
    print(f"    shape error    median {e['median']:8.2f} mm  [{e['min']:.2f}, {e['max']:.2f}]"
          f"   spread {s['shape_error_spread_pct_of_median']:.1f}% of median")
    print(f"    design spread  starts {ds['median']:8.2f} mm   ends {de['median']:8.2f} mm"
          f"   ratio {s['spread_ratio_end_over_start']:.3f}")
    print(f"    travel         median {s['travel_mm']['median']:8.2f} mm")
    f = s["families"]
    print(f"    families       sizes {f['sizes']}   within "
          f"{['%.1f' % w for w in f['within_mean_mm']]} mm   "
          f"separation ratio {f['separation_ratio']:.2f}")
    print(f"    AUDIT  max hinge gap {s['max_hinge_gap_mm']:.2e} mm   "
          f"max flat overlap {100 * s['max_flat_overlap']:.3f}%   "
          f"deploys {100 * s['deploys_frac']:.0f}%")
    print(f"\n  total {time.time() - t0:.0f}s -> {out_dir}")


if __name__ == "__main__":
    main()
