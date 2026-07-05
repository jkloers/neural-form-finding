"""Dataset generation: sample the hinge function's inputs, evaluate it, write to disk.

The dataset is samples of the constitutive map ``hinge(geometry, u) -> (W, dW/du, validity)``
(see ``nff.rve.hinge_function``). One *job* is one deployment ray on one geometry; it yields a
whole *path* of samples (one per solved increment). We LHS-sample the joint 5-D input space

    (w_lig, alpha, theta1, ra, rs)

so every job is a distinct hinge driven along a distinct 3-DOF ray -- failure is resolved as a
function of all of ``(a, s, theta)``, not rotation alone.

Speed: gmsh is not thread-safe, so we phase the batch -- build all decks SERIALLY, run all
``ccx`` solves in PARALLEL (each an external subprocess), parse SERIALLY.
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

import numpy as np

from nff.rve.ccx_solver import prepare_job, solve_job, parse_job
from nff.rve.hinge_function import (HingeConstants, HingeGeometry, DeploymentRay, REGIME_NAME,
                                    to_rve_params, descriptor, solver_kwargs, assemble_response)


# ── sampling ────────────────────────────────────────────────────────────────────

def _lhs(n: int, d: int, seed: int) -> np.ndarray:
    """Latin-hypercube points in [0, 1)^d (one per stratified cell, shuffled per axis)."""
    rng = np.random.default_rng(seed)
    edges = np.arange(n)[:, None] / n
    pts = edges + rng.uniform(size=(n, d)) / n
    for j in range(d):
        rng.shuffle(pts[:, j])
    return pts


def _lerp(u, lo, hi):
    return lo + u * (hi - lo)


def _loglerp(u, lo, hi):
    return float(np.exp(_lerp(u, np.log(lo), np.log(hi))))


def sample_jobs(n, seed=0, *, w_lig=(1.0, 10.0), alpha_deg=(30.0, 150.0),
                theta1_deg=(60.0, 60.0), eta_a=(0.0, 1.0), eta_s=(-0.7, 0.7),
                n_steps=20, spine_frac=0.25):
    """Sample the hinge function's inputs -> list of (geometry, ray).

    A fraction ``spine_frac`` are the PURE-ROTATION spine (eta_a=eta_s=0) over (w_lig, alpha,
    theta1) -- the manifold the deployed pipeline rides, sampled densely. The rest fan out over
    GENEROUS (eta_a, eta_s): buckling can accommodate large in-plane shear/stretch, so we cover
    it broadly and let each ray fail where it fails, measuring the (a, s, theta) failure surface.
    """
    n_spine = int(round(spine_frac * n))
    jobs = []
    us = _lhs(n_spine, 3, seed)                                   # spine: pure rotation
    for i in range(n_spine):
        geo = HingeGeometry(_loglerp(us[i, 0], *w_lig), float(_lerp(us[i, 1], *alpha_deg)))
        ray = DeploymentRay(float(_lerp(us[i, 2], *theta1_deg)), 0.0, 0.0, n_steps, f"s{i:05d}")
        jobs.append((geo, ray))
    uf = _lhs(n - n_spine, 5, seed + 1)                           # fan: full 3-DOF
    for i in range(n - n_spine):
        geo = HingeGeometry(_loglerp(uf[i, 0], *w_lig), float(_lerp(uf[i, 1], *alpha_deg)))
        ray = DeploymentRay(float(_lerp(uf[i, 2], *theta1_deg)),
                            float(_lerp(uf[i, 3], *eta_a)), float(_lerp(uf[i, 4], *eta_s)),
                            n_steps, f"f{i:05d}")
        jobs.append((geo, ray))
    return jobs


# ── parallel evaluation ───────────────────────────────────────────────────────────

def run_jobs(jobs, const=HingeConstants(), *, n_parallel=6, timeout=900,
             root="/tmp/hinge_campaign", fracture_margin=1.1):
    """Evaluate many (geometry, ray) jobs -> list[HingeResponse | None].

    Phased: prepare (serial, gmsh) -> solve (parallel, ccx subprocess) -> parse (serial).
    A failed/timed-out job returns ``None`` (partial data is still parsed when present).
    """
    os.makedirs(root, exist_ok=True)
    metas = []                                                    # phase 1: decks (serial)
    for i, (geo, ray) in enumerate(jobs):
        try:
            metas.append(prepare_job(to_rve_params(geo, const),
                                     workdir=f"{root}/job{i:05d}",
                                     **solver_kwargs(geo, ray, const)))
        except Exception as e:
            metas.append({"error": f"prepare: {type(e).__name__}: {e}"})

    def _solve(m):                                                # phase 2: solve (parallel)
        if "error" in m:
            return ""
        try:
            return solve_job(m, ncpus=1, timeout=timeout, eps_f=const.eps_f,
                             fracture_margin=fracture_margin).stdout
        except Exception:
            return ""                                             # parse whatever completed
    with ThreadPoolExecutor(max_workers=n_parallel) as ex:
        stdouts = list(ex.map(_solve, metas))

    out = []                                                      # phase 3: parse (serial)
    for (geo, ray), m, so in zip(jobs, metas, stdouts):
        if "error" in m:
            out.append(None); continue
        try:
            out.append(assemble_response(geo, ray, const, parse_job(m, so)))
        except Exception:
            out.append(None)
    return out


# ── on-disk dataset ───────────────────────────────────────────────────────────────

# every column is one flat array with one entry per solved increment (sample)
_KIN = ["a", "s", "theta"]                                       # the hinge function's input u
_RESP = ["W", "F_a", "F_s", "M_theta"]                           # W and its gradient dW/du
_AUX = ["peeq_p99", "damage_p99", "uz_max", "theta_deg", "regime"]


def responses_to_columns(responses, const, job_id_offset=0):
    """Flatten paths into aligned 1-D columns: descriptor + u + (W, dW/du) + validity."""
    cols, meta = {}, []
    for local, r in enumerate(responses):
        job_id = job_id_offset + local
        if r is None or r.n_samples == 0:
            meta.append(dict(job_id=job_id, ok=False)); continue
        desc = descriptor(r.geo, const)
        keys = list(desc) + _KIN + _RESP + _AUX + ["job_id"]
        n = r.n_samples
        row = {**{k: np.full(n, desc[k]) for k in desc},
               **{k: getattr(r, k) for k in _KIN + _RESP + _AUX},
               "job_id": np.full(n, job_id)}
        for k in keys:
            cols.setdefault(k, []).append(np.asarray(row[k]))
        meta.append(dict(job_id=job_id, ok=r.ok, tag=r.geo.tag, n_samples=n,
                         failure_theta_deg=r.failure_theta_deg,
                         w_lig=r.geo.w_lig, alpha_deg=r.geo.alpha_deg,
                         eta_a=r.ray.eta_a, eta_s=r.ray.eta_s, theta1_deg=r.ray.theta1_deg))
    cols = {k: np.concatenate(v) for k, v in cols.items()}
    return cols, meta


def _write_checkpoint(out_path, acc, meta, const):
    """Write the cumulative dataset (npz) + summary/meta (json). Overwrites in place."""
    cols = {k: np.concatenate(v) for k, v in acc.items()} if acc else {}
    np.savez_compressed(out_path + ".npz", **cols)
    regime = cols.get("regime", np.array([]))
    n_usable = sum("n_samples" in m for m in meta)                # produced data (the success metric)
    summary = dict(n_jobs=len(meta), n_usable=n_usable, n_errored=len(meta) - n_usable,
                   n_finished_to_cap=sum(m.get("ok", False) for m in meta),  # survived without fracture
                   n_samples=int(len(regime)),
                   n_elastic=int((regime == 0).sum()), n_plastic=int((regime == 1).sum()),
                   n_failed=int((regime == 2).sum()),
                   columns=list(cols), regime_names=REGIME_NAME, const=asdict(const), jobs=meta)
    with open(out_path + ".json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    return summary


def generate_dataset(jobs, out_path, const=HingeConstants(), *, n_parallel=9, timeout=300,
                     batch_size=50, root="/tmp/hinge_campaign", fracture_margin=1.1):
    """Run the campaign in BATCHES, checkpointing the cumulative dataset after each batch.

    Overnight-safe: a crash / machine-sleep loses at most one in-flight batch — everything parsed
    so far is already on disk (``<out_path>.npz`` + ``.json``). Per-job failures are isolated in
    run_jobs (a failed/timed-out job returns None and is skipped), so one bad simulation never
    aborts the campaign. Batching also overlaps the serial gmsh meshing with parallel solves
    instead of meshing all N up front.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    acc, meta, n = {}, [], len(jobs)
    n_batches = -(-n // batch_size)
    for bi, b0 in enumerate(range(0, n, batch_size)):
        responses = run_jobs(jobs[b0:b0 + batch_size], const, n_parallel=n_parallel,
                             timeout=timeout, root=root, fracture_margin=fracture_margin)
        cols, bmeta = responses_to_columns(responses, const, job_id_offset=b0)
        for k, v in cols.items():
            acc.setdefault(k, []).append(v)
        meta += bmeta
        summary = _write_checkpoint(out_path, acc, meta, const)   # checkpoint after every batch
        print(f"  batch {bi + 1}/{n_batches}  ({b0 + len(bmeta)}/{n} jobs): "
              f"{summary['n_usable']} usable, {summary['n_errored']} errored, "
              f"{summary['n_samples']} samples -> checkpointed", flush=True)
    return summary if meta else _write_checkpoint(out_path, acc, meta, const)
