"""Verify a campaign is what it claims to be BEFORE spending 16 hours on it (runs in the ccx env).

Every check here exists because the corresponding mistake would produce a perfectly well-formed,
completely useless dataset that only reveals itself at training time:

  material    ``--material`` used to default to "steel"; omitting it yielded a valid STEEL dataset
              with no warning. eps_f also has to come from the material, not a campaign constant.
  modulus     PET E was 3000 MPa in code while the compliance-free ladder read ~2300-2500.
  envelope    the harvest stores eta at w_lig 5 mm while the standard is 18, so an envelope taken
              in eta rather than mm is silently rescaled by 3.6x; and its WIDTH depends on a
              grip-ceiling calibration that has already moved 4.9x once.
  damage      Delta must land near 1e-3..1e-2. At 1e-5 the design loss term w_damage*mean(D^2) is
              ~1e-8 against chamfer terms of order 1 -- silently dead.
  disk        a job used to write a 1.2 GB .frd; nine of them filled 10 GB. Scratch is now deleted
              per job and fields are written once per state, but the headroom is worth confirming.
  solver      that ccx actually runs, on this machine, with this deck.

    conda run -n ccx --no-capture-output python -m nff.scripts.calibration.preflight_campaign \
        --path-prior data/fea/path_priors/envelope_v3_w18 --material pet --n 700
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time

import numpy as np

from nff.rve.dataset import responses_to_columns, run_jobs
from nff.rve.hinge_function import HingeConstants
from nff.rve.materials import get_material
from nff.rve.path_prior import measure_envelope, sample_campaign_jobs

OK, BAD = "  ok  ", " FAIL "


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--path-prior", dest="path_prior", required=True)
    ap.add_argument("--material", required=True)
    ap.add_argument("--n", type=int, default=700, help="the campaign size you intend to run")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--max-load", dest="max_load", type=float, default=300.0)
    ap.add_argument("--w-lig-min", dest="w_lig_min", type=float, default=5.0)
    ap.add_argument("--w-lig-max", dest="w_lig_max", type=float, default=50.0)
    ap.add_argument("--thickness", type=float, default=0.5)
    ap.add_argument("--r-win", dest="r_win", type=float, default=100.0)
    ap.add_argument("--expect-E", dest="expect_E", type=float, default=2500.0)
    ap.add_argument("--expect-eps-f", dest="expect_eps_f", type=float, default=1.784)
    ap.add_argument("--smoke", type=int, default=3, help="jobs to actually solve (0 to skip)")
    ap.add_argument("--parallel", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=1200)
    ap.add_argument("--min-free-GB", dest="min_free_gb", type=float, default=20.0)
    args = ap.parse_args()

    fails = []

    def check(name, good, detail):
        print(f"[{OK if good else BAD}] {name:34s} {detail}")
        if not good:
            fails.append(name)

    print("── material ─────────────────────────────────────────────────────────────────")
    mat = get_material(args.material)
    const = HingeConstants(thickness=args.thickness, w_c=0.2, r_win=args.r_win, fillet_ratio=0.16,
                           material=mat, n_through=2, el_fields="PEEQ, S")
    check("material is the one requested", const.material.name.lower().startswith(args.material[:3]),
          f"{const.material.name}")
    check("eps_f comes from the material", abs(const.eps_f - args.expect_eps_f) < 1e-9,
          f"{const.eps_f} (expected {args.expect_eps_f})")
    check("elastic modulus", abs(mat.params["E"] - args.expect_E) < 1e-6,
          f"E = {mat.params['E']:.0f} MPa (expected {args.expect_E:.0f})")
    check("stop-at-fracture enabled", const.stop_at_fracture,
          "on -- this is what calibrates Delta_tear")

    print("\n── envelope ─────────────────────────────────────────────────────────────────")
    env = measure_envelope(args.path_prior, max_load_N=args.max_load)
    check("envelope is in millimetres", abs(env.a[1]) < 100.0,
          f"a [{env.a[0]:+.2f}, {env.a[1]:+.2f}] mm   s [{env.s[0]:+.2f}, {env.s[1]:+.2f}] mm")
    check("rotation within the mechanism", 0.0 <= env.theta_deg[0] and env.theta_deg[1] <= 90.0,
          f"theta [0, {env.theta_deg[1]:.1f}] deg")
    check("compression is inside the box", env.a[0] < 0.0,
          f"a_min = {env.a[0]:+.2f} mm -- 40% of measured points are compressive")

    jobs = sample_campaign_jobs(args.n, env, seed=args.seed, n_steps=args.steps,
                                w_lig=(args.w_lig_min, args.w_lig_max))
    fan = [(g, r) for g, r in jobs if not r.free_dofs]
    a = np.array([r.eta_a * g.w_lig for g, r in fan])
    s = np.array([r.eta_s * g.w_lig for g, r in fan])
    w = np.array([g.w_lig for g, _ in jobs])
    check("campaign size", len(jobs) == args.n, f"{len(jobs)} jobs, {len(jobs) - len(fan)} free-DOF spine")
    check("compression is sampled", (a < 0).mean() > 0.25, f"{(a < 0).mean():.0%} of fan jobs")
    check("shear-dominated sampled", (np.abs(s) > np.abs(a)).mean() > 0.25,
          f"{(np.abs(s) > np.abs(a)).mean():.0%} of fan jobs")
    check("w_lig spans the design range", w.min() < 6.0 and w.max() > 40.0,
          f"{w.min():.1f} .. {w.max():.1f} mm")

    print("\n── machine ──────────────────────────────────────────────────────────────────")
    free_gb = shutil.disk_usage("/tmp").free / 1e9
    check("scratch headroom", free_gb >= args.min_free_gb, f"{free_gb:.0f} GB free on /tmp")
    ccx = os.environ.get("CCX_BIN", "ccx")
    check("ccx binary", shutil.which(ccx) is not None or os.path.exists(ccx), ccx)

    if args.smoke:
        print(f"\n── smoke: solving {args.smoke} real jobs ────────────────────────────────────")
        t0 = time.time()
        probe = jobs[:args.smoke]
        responses = run_jobs(probe, const, n_parallel=args.parallel, timeout=args.timeout,
                             root="/tmp/hinge_preflight")
        wall = time.time() - t0
        good = [r for r in responses if r is not None and r.n_samples > 0]
        check("jobs produced data", len(good) == len(probe), f"{len(good)}/{len(probe)}")
        if good:
            cols, meta = responses_to_columns(good, const)
            dmax = float(np.nanmax(cols["damage"]))
            check("damage is on a usable scale", 1e-5 < dmax < 1.0,
                  f"max Delta = {dmax:.2e}  (want 1e-3..1e-2; 1e-5 kills the design-loss term)")
            check("energy is finite and positive", np.isfinite(cols["W"]).all() and
                  np.nanmax(cols["W"]) > 0, f"max W = {np.nanmax(cols['W']):.1f} N.mm")
            check("stop reasons recorded", all(m.get("stop_reason", "unknown") != "unknown"
                                               for m in meta),
                  ", ".join(sorted({m["stop_reason"] for m in meta})))
            per_job = wall / max(len(probe), 1)      # jobs ran concurrently, so this IS the mean
            for par in (args.parallel, 9):
                rate = 3600.0 * par / max(per_job, 1e-9)
                print(f"        ~{per_job:.0f}s per job -> {rate:.0f} jobs/hour at {par} parallel; "
                      f"{args.n} jobs ~ {args.n / max(rate, 1e-9):.1f} h")
        shutil.rmtree("/tmp/hinge_preflight", ignore_errors=True)

    print()
    if fails:
        print(f"PREFLIGHT FAILED: {len(fails)} check(s) -- " + "; ".join(fails))
        sys.exit(1)
    print("PREFLIGHT PASSED -- safe to launch")


if __name__ == "__main__":
    main()
