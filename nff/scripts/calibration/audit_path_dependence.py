"""Measure how path-dependent the hinge energy is -- the surrogate's stated error bar.

``W(a, s, theta)`` is fitted as a STATE function, but the material is elasto-plastic, so the
stored energy is really a path-dependent WORK. This script quantifies the gap the state-function
assumption papers over: for each sampled deployment path it drives the single-hinge RVE twice --

    A. along the measured polyline  (the route the sheet actually rides)
    B. along a straight proportional ray to the SAME endpoint  (what the campaign trains on)

-- and reports the paired distribution of ``W_path/W_ray`` and ``D_path/D_ray``. A single earlier
sample gave +65% on W and 3.2x on damage; this turns that anecdote into a distribution.

Read the result as a diagnostic, NOT as a law. The measured polylines come from a Stage-2 solve,
so they minimise the energy of whatever hinge model was in the loop. Harvested under the ROM they
are rotation-first (the ROM's linear ``k_rot`` is the buckled secant, so it under-prices the first
few degrees). The number that matters for the deployed surrogate is the one measured on paths
harvested with THAT surrogate in the loop -- run this once per generation and watch it shrink.

Requires the campaign branch (``DeploymentPath``, ``pick_path``, ``resample_polyline``) and a
working ``ccx``; runs in the ccx env, not kgnn_mac:

    conda run -n ccx --no-capture-output python -m nff.scripts.calibration.audit_path_dependence \
        --prior data/fea/path_priors/envelope_v3_w18 --n 20 --w-lig 18 \
        --out data/surrogates/path_dependence_pet_v2.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

try:
    from nff.rve.hinge_function import DeploymentPath
    from nff.scripts.calibration.replay_hinge_path import pick_path, resample_polyline
except ImportError as exc:                      # pragma: no cover - branch-ordering guard
    raise SystemExit(
        "audit_path_dependence needs the PET campaign branch (DeploymentPath, pick_path, "
        f"resample_polyline). Merge it before running this. Underlying error: {exc}")

from nff.rve.ccx_solver import deploy
from nff.rve.hinge_function import (HingeConstants, HingeGeometry, assemble_response,
                                    solver_kwargs, to_rve_params)
from nff.rve.materials import get_material


def _run(states, geo, const, tag, *, timeout, ncpus, workdir):
    """One CalculiX deployment along an explicit state list -> the endpoint scalars we compare."""
    path = DeploymentPath(polyline=states, tag=tag)
    kw = solver_kwargs(geo, path, const)
    t0 = time.time()
    parsed = deploy(to_rve_params(geo, const), timeout=timeout, eps_f=None, ncpus=ncpus,
                    workdir=workdir, **kw)
    res = assemble_response(geo, path, const, parsed)
    if res.n_samples == 0:
        return None
    return dict(
        W=float(res.W[-1]), damage=float(res.damage[-1]), peeq_lig=float(res.peeq_lig[-1]),
        uz_max=float(res.uz_max[-1]), theta_deg=float(res.theta_deg[-1]),
        theta_req_deg=float(path.theta1_deg), n_samples=int(res.n_samples),
        ok=bool(parsed["ok"]), wall_s=round(time.time() - t0, 1))


def _stats(name, ratios):
    r = np.asarray([x for x in ratios if np.isfinite(x)], float)
    if r.size == 0:
        return {name: None}
    return {name: dict(n=int(r.size), median=float(np.median(r)),
                       p25=float(np.percentile(r, 25)), p75=float(np.percentile(r, 75)),
                       p90=float(np.percentile(r, 90)),
                       min=float(r.min()), max=float(r.max()))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prior", default="data/fea/path_priors/envelope_v3_w18")
    ap.add_argument("--n", type=int, default=20, help="paths to sample (2 ccx solves each)")
    ap.add_argument("--seed0", type=int, default=0, help="first path seed; uses seed0..seed0+n-1")
    ap.add_argument("--w-lig", type=float, default=18.0)
    ap.add_argument("--thickness", type=float, default=0.5)
    ap.add_argument("--alpha", type=float, default=None, help="override each path's alpha [deg]")
    ap.add_argument("--fillet", type=float, default=0.16)
    ap.add_argument("--kerf", type=float, default=0.2)
    ap.add_argument("--r-win", type=float, default=100.0)
    ap.add_argument("--n-through", type=int, default=2)
    ap.add_argument("--resample", type=int, default=30, help="uniform arc-length steps per route")
    ap.add_argument("--material", default="pet")
    ap.add_argument("--ncpus", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--tol", type=float, default=0.98,
                    help="a route must reach this fraction of its requested theta to be comparable")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    const = HingeConstants(
        thickness=args.thickness, w_c=args.kerf, r_win=args.r_win, fillet_ratio=args.fillet,
        material=get_material(args.material), n_through=args.n_through, stop_at_fracture=False)

    pairs, skipped = [], []
    for seed in range(args.seed0, args.seed0 + args.n):
        poly, alpha_deg, prov = pick_path(args.prior, seed)
        if args.alpha is not None:
            alpha_deg = args.alpha
        if args.resample:
            poly = resample_polyline(poly, args.resample, args.w_lig)
        states = poly[1:]                                   # origin is implicit at t = 0
        n = len(states)
        ray = np.outer(np.arange(1, n + 1) / n, states[-1])  # same endpoint, straight route
        geo = HingeGeometry(w_lig=args.w_lig, alpha_deg=alpha_deg, fillet_ratio=args.fillet)

        print(f"[{seed - args.seed0 + 1}/{args.n}] seed {seed}  alpha {alpha_deg:.1f} deg  "
              f"endpoint a={states[-1, 0]:+.3f} s={states[-1, 1]:+.3f} "
              f"theta={np.degrees(states[-1, 2]):.1f} deg", flush=True)

        wd = f"/tmp/hinge_audit/s{seed}"
        a = _run(states, geo, const, f"poly{seed}", timeout=args.timeout, ncpus=args.ncpus,
                 workdir=wd + "_poly")
        b = _run(ray, geo, const, f"ray{seed}", timeout=args.timeout, ncpus=args.ncpus,
                 workdir=wd + "_ray")

        # Both routes must actually REACH the shared endpoint, or the ratio compares two different
        # states rather than two different routes. Reject explicitly and count it -- a silently
        # dropped pair would bias the distribution toward whichever route survives more often.
        why = None
        if a is None or b is None:
            why = "no increments parsed"
        elif a["theta_deg"] < args.tol * a["theta_req_deg"]:
            why = f"polyline truncated ({a['theta_deg']:.1f}/{a['theta_req_deg']:.1f} deg)"
        elif b["theta_deg"] < args.tol * b["theta_req_deg"]:
            why = f"ray truncated ({b['theta_deg']:.1f}/{b['theta_req_deg']:.1f} deg)"
        if why:
            print(f"    SKIP: {why}")
            skipped.append(dict(seed=seed, reason=why, provenance=prov))
            continue

        rec = dict(seed=seed, alpha_deg=alpha_deg, provenance=prov, polyline=a, ray=b,
                   W_ratio=a["W"] / b["W"] if b["W"] else float("nan"),
                   damage_ratio=a["damage"] / b["damage"] if b["damage"] else float("nan"),
                   peeq_ratio=a["peeq_lig"] / b["peeq_lig"] if b["peeq_lig"] else float("nan"))
        pairs.append(rec)
        print(f"    W {a['W']:8.2f} vs {b['W']:8.2f}  ratio {rec['W_ratio']:5.2f}   |   "
              f"D {a['damage']:.5f} vs {b['damage']:.5f}  ratio {rec['damage_ratio']:5.2f}",
              flush=True)

    summary = {}
    for key, label in (("W_ratio", "W_ratio"), ("damage_ratio", "damage_ratio"),
                       ("peeq_ratio", "peeq_ratio")):
        summary.update(_stats(label, [p[key] for p in pairs]))

    out = dict(prior=args.prior, n_requested=args.n, n_compared=len(pairs),
               n_skipped=len(skipped), skipped=skipped, tol=args.tol,
               geometry=dict(w_lig=args.w_lig, thickness=args.thickness, r_win=args.r_win,
                             fillet_ratio=args.fillet, material=args.material,
                             resample=args.resample),
               summary=summary, pairs=pairs)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\ncompared {len(pairs)} pairs, skipped {len(skipped)} "
          f"(of {args.n} requested -- skips are listed in the JSON, not hidden)")
    for k, v in summary.items():
        if v:
            print(f"  {k:14s} median {v['median']:.3f}  IQR [{v['p25']:.3f}, {v['p75']:.3f}]  "
                  f"p90 {v['p90']:.3f}  range [{v['min']:.3f}, {v['max']:.3f}]")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
