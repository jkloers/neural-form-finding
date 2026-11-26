"""Does the target determine the cut pattern? Many random starts, one kinematic problem.

The Fig. 5(d) task -- a square closed sheet deployed onto a fixed circle -- solved from N
random starting patterns with no physics anywhere: deployment is the closed-form 1-DOF map,
so pin closure is exact at every iterate and the only thing that varies is where the
optimiser began.

    JAX_PLATFORMS=cpu conda run -n kgnn_mac python \
        nff/scripts/diagnostics/ensemble_circle.py --seeds 64

``--basin`` repeats the ensemble across start spreads, turning "is it unique?" into a curve.
"""

import argparse
import json
import os
import subprocess
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

from nff.closed.design_ensemble import (build_symmetry, circle_problem, draw_start,
                                        ensemble_summary, problem_statics, run_ensemble,
                                        save_ensemble, symmetry_closure)


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       text=True).strip()
    except Exception:                                              # noqa: BLE001
        return "unknown"


def _print_summary(summary: dict, label: str):
    s = summary
    print(f"\n  ── {label} ─────────────────────────────────────────────")
    print(f"    n = {s['n']}   symmetry group order {s['group_order']}   "
          f"medoid seed {s['medoid_seed']}")
    e, de, ds = s["shape_error_mm"], s["design_spread_end_mm"], s["design_spread_start_mm"]
    print(f"    shape error      median {e['median']:8.2f} mm   "
          f"[{e['min']:.2f}, {e['max']:.2f}]   spread {s['shape_error_spread_pct_of_median']:.1f}% of median")
    print(f"    design spread    starts {ds['median']:8.2f} mm   "
          f"ends {de['median']:8.2f} mm   ratio {s['spread_ratio_end_over_start']:.3f}")
    print(f"    travel           median {s['travel_mm']['median']:8.2f} mm")
    t = s["theta_deg"]
    print(f"    opening theta    median {t['median']:8.2f} deg  [{t['min']:.2f}, {t['max']:.2f}]")
    print(f"    panel area CV    median {s['area_cv']['median']:8.3f}   "
          f"radius {s['radius_mean_mm']['median']:.1f} mm")
    print(f"    AUDIT  max pin gap {s['max_pin_gap_mm']:.2e} mm   "
          f"max flat-area rel err {s['max_flat_area_rel_err']:.2e}")
    g = s["grad_norm"]
    print(f"    stationarity     |grad| median {g['median']:.2e}  max {g['max']:.2e}")
    print(f"    quality floor active on median "
          f"{100 * s['quality_floor_active_frac']['median']:.1f}% of tiles")


def _verdict(summary: dict) -> str:
    """Name the outcome in the taxonomy, using the two spreads."""
    err_spread = summary["shape_error_spread_pct_of_median"]
    ratio = summary["spread_ratio_end_over_start"]
    design_med = summary["design_spread_end_mm"]["median"]
    if err_spread > 25.0:
        return "DIFFERENT shape error -> genuine local minima"
    if design_med < 1.0:
        return "same shape error, same design -> well posed"
    return (f"same shape error (within {err_spread:.1f}%), DIFFERENT design "
            f"(median {design_med:.0f} mm apart, {ratio:.2f} of the starting spread) "
            f"-> the target underdetermines the pattern")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=64)
    ap.add_argument("--sigma", type=float, default=1.0, help="start spread")
    ap.add_argument("--grid", default="7x7")
    ap.add_argument("--radius", type=float, default=2171.0,
                    help="target radius in mm; the natural deployed size at 7x7")
    ap.add_argument("--sheet", type=float, default=2844.8)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--basin", action="store_true",
                    help="repeat over start spreads 0.25 / 0.5 / 1.0 / 2.0")
    ap.add_argument("--basin-seeds", type=int, default=16)
    ap.add_argument("--snapshot-every", type=int, default=None,
                    help="record the design every N steps, for plotting the path taken")
    args = ap.parse_args()

    M, N = (int(v) for v in args.grid.lower().split("x"))
    out_dir = args.out or (f"data/outputs/design_cloud/"
                           f"circle_{M}x{N}_n{args.seeds}_s{args.sigma:g}")
    os.makedirs(out_dir, exist_ok=True)

    cfg = circle_problem(M=M, N=N, sheet_mm=args.sheet, radius_mm=args.radius,
                         steps=args.steps)
    statics = problem_statics(cfg)
    group = build_symmetry(cfg, statics)

    start0 = draw_start(cfg, statics, 0, args.sigma)
    n_design = sum(int(np.asarray(v).size) for v in start0.values())
    print(f"  {M}x{N} closed sheet, {M * N} panels, {args.sheet:.1f} mm square")
    print(f"  target circle r = {args.radius:.1f} mm, fixed a priori")
    print(f"  design vector: {n_design} scalars "
          f"({', '.join(f'{k} {np.asarray(v).size}' for k, v in sorted(start0.items()))})")
    print(f"  symmetry group: {sorted(group)} (order {len(group)})")
    print(f"  writing to {out_dir}\n")

    t0 = time.time()

    def progress(rec):
        print(f"    seed {rec['seed']:3d}  err {rec['shape_error_mm']:7.2f} mm  "
              f"theta {2 * abs(np.degrees(rec['psi'])):5.1f} deg  "
              f"CV {rec['area_cv']:.2f}  pin {rec['max_pin_gap_mm']:.1e} mm  "
              f"[{time.time() - t0:5.0f}s]")

    records, failures, statics = run_ensemble(cfg, range(1, args.seeds + 1),
                                              sigma=args.sigma, progress=progress,
                                              snapshot_every=args.snapshot_every)
    if failures:
        print(f"\n  {len(failures)} start(s) failed: {failures}")
    if not records:
        raise SystemExit("every start failed")

    summary = ensemble_summary(records, group)
    summary["verdict"] = _verdict(summary)
    # Two more full distance matrices; on a large ensemble that costs more than the solves,
    # and the check only needs enough pairs to see whether any collapse.
    chir_n = min(len(records), 48)
    summary["chirality_check"] = symmetry_closure(
        [records[i]["coords"] for i in np.linspace(0, len(records) - 1, chir_n).astype(int)],
        cfg, statics)
    summary["chirality_check"]["n_designs_checked"] = chir_n
    summary["config"] = {"M": M, "N": N, "sheet_mm": args.sheet, "radius_mm": args.radius,
                         "steps": args.steps, "sigma": args.sigma,
                         "seeds": args.seeds, "git_sha": _git_sha()}
    summary["per_seed"] = [
        {k: r[k] for k in ("seed", "shape_error_mm", "precision_mm", "recall_mm",
                           "worst_recall_mm", "loss", "psi", "area_cv", "min_quality",
                           "radius_mean_mm", "radius_min_mm", "radius_max_mm",
                           "max_pin_gap_mm", "flat_area_rel_err", "grad_norm",
                           "quality_floor_active_frac")}
        for r in records]

    save_ensemble(out_dir, records, cfg)
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    _print_summary(summary, f"sigma = {args.sigma}")
    ch = summary["chirality_check"]
    print(f"\n    chirality: closest pair under rotations {ch['min_pair_rotations_mm']:.1f} mm, "
          f"with reflections forced in {ch['min_pair_with_reflections_mm']:.1f} mm")
    print(f"\n  VERDICT: {summary['verdict']}")

    if args.basin:
        basin = []
        for sigma in (0.25, 0.5, 1.0, 2.0):
            recs, _, _ = run_ensemble(cfg, range(1001, 1001 + args.basin_seeds), sigma=sigma)
            if not recs:
                continue
            s = ensemble_summary(recs, group)
            basin.append({"sigma": sigma, **{k: s[k] for k in (
                "shape_error_mm", "design_spread_end_mm", "design_spread_start_mm",
                "spread_ratio_end_over_start", "theta_deg")}})
            print(f"    sigma {sigma:4.2f}: err med {s['shape_error_mm']['median']:6.2f} mm  "
                  f"design spread {s['design_spread_end_mm']['median']:7.1f} mm  "
                  f"ratio {s['spread_ratio_end_over_start']:.3f}")
        with open(os.path.join(out_dir, "basin.json"), "w") as fh:
            json.dump(basin, fh, indent=2)

    print(f"\n  total {time.time() - t0:.0f}s -> {out_dir}")


if __name__ == "__main__":
    main()
