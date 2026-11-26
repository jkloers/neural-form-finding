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

    conda run -n ccx --no-capture-output python -m nff.scripts.surrogate.audit_path_dependence \
        --prior data/fea/path_priors/envelope_v3_w18 --n 3 --seed0 0 --seed-stride 10 --w-lig 18 \
        --out data/surrogates/path_dependence_pet_v2.json
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time

import numpy as np

try:
    from nff.rve.hinge_function import DeploymentPath
    from nff.scripts.surrogate.replay_hinge_path import pick_path, resample_polyline
except ImportError as exc:                      # pragma: no cover - branch-ordering guard
    raise SystemExit(
        "audit_path_dependence needs the PET campaign branch (DeploymentPath, pick_path, "
        f"resample_polyline). Merge it before running this. Underlying error: {exc}")

from nff.rve.ccx_solver import deploy
from nff.rve.hinge_function import (HingeConstants, HingeGeometry, assemble_response,
                                    solver_kwargs, to_rve_params)
from nff.rve.materials import get_material
from nff.utils.routes import nonproportionality


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


def endpoint_reason(states, w_lig, args):
    """Why this path is not worth two solves, or ``None`` if it is.

    A harvested prior samples every hinge of every deployment, so a uniform draw over it is mostly
    hinges that barely moved and a tail that moved absurdly. Neither answers the question:

    * below ``--min-theta-deg`` the hinge has not opened, ``W`` is near zero on both routes, and the
      ratio is one small number divided by another.
    * outside the trust region the surrogate is never read, so the gap there prices nothing the
      design loop relies on. That tail also carries the expensive and divergence-prone solves: the
      w18 prior reaches an axial ratio of 3.25, which is 58 mm of opening on an 18 mm ligament.

    Screening here costs nothing, because it happens before CalculiX is invoked.
    """
    departure = max(nonproportionality(states[:, 0], states[:, 2], w_lig),
                    nonproportionality(states[:, 1], states[:, 2], w_lig))
    if not (args.min_departure <= departure <= args.max_departure):
        return f"route departure {departure:.3f} outside the requested band"

    a, s, theta = states[-1]
    eta_a, eta_s, theta_deg = a / w_lig, s / w_lig, np.degrees(theta)
    if theta_deg < args.min_theta_deg:
        return f"endpoint rotation {theta_deg:.1f} deg below the {args.min_theta_deg:.0f} deg floor"
    if theta_deg > args.max_theta_deg:
        return f"endpoint rotation {theta_deg:.1f} deg above the trust region"
    if not (args.min_eta_a <= eta_a <= args.max_eta_a):
        return f"axial ratio {eta_a:+.3f} outside the trust region"
    if abs(eta_s) > args.max_abs_eta_s:
        return f"shear ratio {eta_s:+.3f} outside the trust region"
    return None


def _stats(name, ratios):
    r = np.asarray([x for x in ratios if np.isfinite(x)], float)
    if r.size == 0:
        return {name: None}
    return {name: dict(n=int(r.size), median=float(np.median(r)),
                       p25=float(np.percentile(r, 25)), p75=float(np.percentile(r, 75)),
                       p90=float(np.percentile(r, 90)),
                       min=float(r.min()), max=float(r.max()))}


def summarise(pairs):
    """Paired ratio statistics over the records collected so far."""
    summary = {}
    for key in ("W_ratio", "damage_ratio", "peeq_ratio"):
        summary.update(_stats(key, [p[key] for p in pairs]))
    return summary


def _dump(path, args, pairs, skipped, rejected=()):
    """Write the report. Called after EVERY pair, not only at the end.

    Each solve costs about half an hour, so a shard that dies late would otherwise discard hours of
    finished work. Writing as we go also lets a fan-out be stopped early and still merged.
    """
    out = dict(prior=args.prior, n_requested=args.n, n_compared=len(pairs),
               n_skipped=len(skipped), skipped=skipped, tol=args.tol,
               n_rejected_unsolved=len(rejected), rejected=list(rejected),
               seed0=args.seed0, seed_stride=args.seed_stride,
               band=dict(min_theta_deg=args.min_theta_deg, max_theta_deg=args.max_theta_deg,
                         min_eta_a=args.min_eta_a, max_eta_a=args.max_eta_a,
                         max_abs_eta_s=args.max_abs_eta_s),
               geometry=dict(w_lig=args.w_lig, thickness=args.thickness, r_win=args.r_win,
                             fillet_ratio=args.fillet, material=args.material,
                             resample=args.resample),
               summary=summarise(pairs), pairs=pairs)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=2)
    os.replace(tmp, path)                  # atomic, so a reader never sees a half-written file
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prior", default="data/fea/path_priors/envelope_v3_w18")
    ap.add_argument("--n", type=int, default=20, help="paths to sample (2 ccx solves each)")
    ap.add_argument("--seed0", type=int, default=0, help="first path seed")
    ap.add_argument("--seed-stride", dest="seed_stride", type=int, default=1,
                    help="step between seeds. Set it to the shard COUNT and seed0 to the shard "
                         "index: seeds then interleave, so a shard that scans past its quota can "
                         "never reach another shard's paths.")
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
    ap.add_argument("--workdir-root", dest="workdir_root", default="/tmp/hinge_audit",
                    help="scratch root for the ccx runs; seeds already namespace the subdirectories")
    ap.add_argument("--keep-scratch", dest="keep_scratch", action="store_true",
                    help="keep the ccx working directories of SUCCESSFUL pairs (~160 MB each)")
    # Defaults are the shipped checkpoint's trust region (hinge_surrogate_pet_v3.pkl, stats
    # ["domain"]), so the gap is measured where the surrogate is actually read.
    ap.add_argument("--min-theta-deg", dest="min_theta_deg", type=float, default=10.0)
    ap.add_argument("--max-theta-deg", dest="max_theta_deg", type=float, default=71.9)
    ap.add_argument("--min-eta-a", dest="min_eta_a", type=float, default=-0.140)
    ap.add_argument("--max-eta-a", dest="max_eta_a", type=float, default=0.233)
    ap.add_argument("--max-abs-eta-s", dest="max_abs_eta_s", type=float, default=0.220)
    # A run restricted to nearly straight routes is the NULL CONTROL: there the two legs almost
    # coincide, so a ratio far from one would be solver noise or a mismatched endpoint rather than
    # path dependence. The prior carries no such routes at its own sampling, so they must be asked
    # for deliberately.
    ap.add_argument("--min-departure", dest="min_departure", type=float, default=0.0)
    ap.add_argument("--max-departure", dest="max_departure", type=float, default=float("inf"))
    ap.add_argument("--max-seed-scan", dest="max_seed_scan", type=int, default=4000,
                    help="give up after this many seeds; guards against an empty acceptance band")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    const = HingeConstants(
        thickness=args.thickness, w_c=args.kerf, r_win=args.r_win, fillet_ratio=args.fillet,
        material=get_material(args.material), n_through=args.n_through, stop_at_fracture=False)

    pairs, skipped, rejected = [], [], []
    seed, scanned = args.seed0, 0
    while len(pairs) < args.n and scanned < args.max_seed_scan:
        seed, scanned = seed + args.seed_stride, scanned + 1
        this_seed = seed - args.seed_stride
        poly, alpha_deg, prov = pick_path(args.prior, this_seed)
        if args.alpha is not None:
            alpha_deg = args.alpha
        if args.resample:
            poly = resample_polyline(poly, args.resample, args.w_lig)
        states = poly[1:]                                   # origin is implicit at t = 0
        n = len(states)
        ray = np.outer(np.arange(1, n + 1) / n, states[-1])  # same endpoint, straight route
        geo = HingeGeometry(w_lig=args.w_lig, alpha_deg=alpha_deg, fillet_ratio=args.fillet)

        # Screened BEFORE any solve, so a path outside the band costs nothing.
        reason = endpoint_reason(states, args.w_lig, args)
        if reason:
            rejected.append(dict(seed=this_seed, reason=reason))
            continue

        print(f"[{len(pairs) + 1}/{args.n}] seed {this_seed}  alpha {alpha_deg:.1f} deg  "
              f"endpoint a={states[-1, 0]:+.3f} s={states[-1, 1]:+.3f} "
              f"theta={np.degrees(states[-1, 2]):.1f} deg", flush=True)

        wd = os.path.join(args.workdir_root, f"s{this_seed}")
        a = _run(states, geo, const, f"poly{this_seed}", timeout=args.timeout, ncpus=args.ncpus,
                 workdir=wd + "_poly")
        b = _run(ray, geo, const, f"ray{this_seed}", timeout=args.timeout, ncpus=args.ncpus,
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
            skipped.append(dict(seed=this_seed, reason=why, provenance=prov))
            _dump(args.out, args, pairs, skipped, rejected)
            continue

        # How far this route departs from the ray it is being compared against. Recording it makes
        # the result a relation rather than a single number: the gap should vanish as the route
        # straightens, and a pair set that shows otherwise is measuring something else.
        departure = max(nonproportionality(states[:, 0], states[:, 2], args.w_lig),
                        nonproportionality(states[:, 1], states[:, 2], args.w_lig))
        rec = dict(seed=this_seed, alpha_deg=alpha_deg, provenance=prov, polyline=a, ray=b,
                   departure=float(departure),
                   W_ratio=a["W"] / b["W"] if b["W"] else float("nan"),
                   damage_ratio=a["damage"] / b["damage"] if b["damage"] else float("nan"),
                   peeq_ratio=a["peeq_lig"] / b["peeq_lig"] if b["peeq_lig"] else float("nan"))
        pairs.append(rec)
        print(f"    W {a['W']:8.2f} vs {b['W']:8.2f}  ratio {rec['W_ratio']:5.2f}   |   "
              f"D {a['damage']:.5f} vs {b['damage']:.5f}  ratio {rec['damage_ratio']:5.2f}",
              flush=True)
        _dump(args.out, args, pairs, skipped, rejected)

        # Each solve leaves ~80 MB of .frd. A skipped pair keeps its scratch so the failure can be
        # inspected; a compared one has nothing left to say that the record does not already hold.
        if not args.keep_scratch:
            for d in (wd + "_poly", wd + "_ray"):
                shutil.rmtree(d, ignore_errors=True)

    summary = _dump(args.out, args, pairs, skipped, rejected)["summary"]

    print(f"\ncompared {len(pairs)} pairs, skipped {len(skipped)} "
          f"(of {args.n} requested -- skips are listed in the JSON, not hidden)")
    for k, v in summary.items():
        if v:
            print(f"  {k:14s} median {v['median']:.3f}  IQR [{v['p25']:.3f}, {v['p75']:.3f}]  "
                  f"p90 {v['p90']:.3f}  range [{v['min']:.3f}, {v['max']:.3f}]")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
