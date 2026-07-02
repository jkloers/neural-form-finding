"""Parallel batch runner for the hinge dataset sweep.

Speed strategy (from benchmarks): PARDISO isn't available and single-job threading barely
helps; the wins are a coarse (converged) mesh + running many jobs at once. gmsh is NOT
thread-safe, so we phase the batch: build all meshes/decks SERIALLY, run all ccx solves
in PARALLEL (each is an external subprocess, safe), then parse SERIALLY. Workers return only
scalar labels (energy, forces, buckle, plastic strain).
"""

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from nff.rve.geometry import RVEParams
from nff.rve.ccx_solver import prepare_job, solve_job, parse_job, STEEL

_LABELS = ("theta_deg", "W", "M_theta", "F_a", "F_s", "uz_max", "peeq_max")


def run_batch(jobs, n_parallel=8, timeout=1800):
    """jobs: list of dicts of deploy kwargs (with 'params' = RVEParams kwargs). Returns labels."""
    os.makedirs("/tmp/hinge_sweep", exist_ok=True)

    metas = []                                                   # phase 1: build decks (serial)
    for i, job in enumerate(jobs):
        job = dict(job)
        try:
            p = RVEParams(**job.pop("params"))
            metas.append(prepare_job(p, workdir=f"/tmp/hinge_sweep/job{i:05d}", **job))
        except Exception as e:
            metas.append({"error": f"prepare: {type(e).__name__}: {e}"})

    def _solve(m):                                               # phase 2: solve (parallel)
        if "error" in m:
            return None
        try:
            return solve_job(m, ncpus=1, timeout=timeout)
        except Exception as e:
            m["error"] = f"solve: {type(e).__name__}: {e}"
            return None
    with ThreadPoolExecutor(max_workers=n_parallel) as ex:
        runs = list(ex.map(_solve, metas))

    out = []                                                     # phase 3: parse (serial)
    for m, run in zip(metas, runs):
        if "error" in m:
            out.append({"ok": False, "error": m["error"]}); continue
        try:
            r = parse_job(m, run.stdout if run else "")
            out.append({**{k: np.asarray(r[k]) for k in _LABELS}, "ok": r["ok"], "n_elems": r["n_elems"]})
        except Exception as e:
            out.append({"ok": False, "error": f"parse: {type(e).__name__}: {e}"})
    return out


def failure_theta(res, eps_f=0.25):
    """First fold angle where PEEQ >= eps_f (ductile fracture), else None (survives)."""
    pk, th = res.get("peeq_max"), res.get("theta_deg")
    if pk is None or th is None or not len(pk):
        return None
    m = np.asarray(pk) >= eps_f
    if not m.any():
        return None
    return float(np.argmax(m) / max(len(pk) - 1, 1) * th[-1]) if len(th) else None
