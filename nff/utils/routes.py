"""Route-family classification shared by training and diagnostics. NumPy only -- deliberately no JAX.

A campaign job is an ``origin_ray`` when its ``(a, s)`` are proportional to ``theta`` through the
origin, and a ``free_path`` when they are not. The distinction is load-bearing, not cosmetic:

Origin rays **foliate** the state space -- two of them meet only at ``u = 0`` -- so on a
prescribed-ray campaign the path that reached a state is recoverable from the state itself
(direction ``u/||u||``, earlier states ``lambda*u``). History is then a function of state, ``W`` is
single-valued in ``(a, s, theta)``, and a state-function surrogate is the right object to fit.

A free-DOF job (``DeploymentRay(..., free_dofs=("a","s"))``) has the solver choose ``(a, s)`` at each
imposed rotation, tracing a curve that **cuts across** that foliation. It crosses origin rays
belonging to other jobs, and at each crossing ``W`` takes two different values -- the ray's
accumulated work and the curve's. ``W`` is a path functional for an elastoplastic oracle, so mixing
the two families makes the training target multi-valued in the network's own input.

Measured on the PET campaign (2026-07-30): cross-family sigma_pair is 2.50x the same-family value,
the two families carry opposite-signed residual bias, and a checkpoint scoring 5.97% on
``origin_ray`` rows scores 43.5% on ``free_path`` rows.

Lives here rather than in the diagnostics CLI so the trainer can filter by family without importing
a script (same reasoning as :mod:`nff.utils.splits`).
"""

import numpy as np

ORIGIN_RAY = "origin_ray"
FREE_PATH = "free_path"
UNKNOWN = "unknown"

DEF_TOL = 0.02


def nonproportionality(x, theta, w_lig):
    """Residual fraction of the best origin ray ``x = k*theta``.

    Least squares THROUGH THE ORIGIN, because that is what an origin ray is: the undeformed state is
    implicit at lambda=0 and every sampled state is ``lambda*u1``. The denominator carries a
    ``w_lig`` floor so a pure-rotation spine (``x`` identically 0, residual identically 0) scores 0
    rather than 0/0.
    """
    tt = float(theta @ theta)
    k = float(x @ theta) / tt if tt > 0 else 0.0
    resid = float(np.sqrt(np.mean((x - k * theta) ** 2)))
    return resid / (float(np.sqrt(np.mean(x ** 2))) + 1e-3 * max(float(w_lig), 1e-9))


def classify_jobs(a, s, theta, w_lig, job_id, *, tol=DEF_TOL, min_rows=3):
    """Per-job route family from the ROWS alone -> (family dict, per-job diagnostics).

    Definitional test, so it needs no manifest and works on any campaign: a job is an ``origin_ray``
    iff both ``a`` and ``s`` are proportional to ``theta`` through the origin. A free-DOF job's
    ``(a, s)`` are solver outputs tracing a curve, so they are not.
    """
    a, s, theta = np.asarray(a, float), np.asarray(s, float), np.asarray(theta, float)
    w_lig, job_id = np.asarray(w_lig, float), np.asarray(job_id)
    fam, diag = {}, {}
    for j in np.unique(job_id):
        m = job_id == j
        if int(m.sum()) < min_rows:
            fam[int(j)] = UNKNOWN
            diag[int(j)] = dict(n_rows=int(m.sum()), nonprop=float("nan"))
            continue
        wl = float(np.median(w_lig[m]))
        na = nonproportionality(a[m], theta[m], wl)
        ns = nonproportionality(s[m], theta[m], wl)
        nonprop = max(na, ns)
        fam[int(j)] = ORIGIN_RAY if nonprop <= tol else FREE_PATH
        diag[int(j)] = dict(n_rows=int(m.sum()), nonprop=float(nonprop),
                            nonprop_a=float(na), nonprop_s=float(ns),
                            w_lig=wl, max_abs_a=float(np.abs(a[m]).max()),
                            theta_max_deg=float(np.degrees(theta[m].max())))
    return fam, diag


def route_family_mask(a, s, theta, w_lig, job_id, family, *, tol=DEF_TOL):
    """Row mask selecting one route family; ``family='all'`` keeps everything.

    ``origin_ray`` is the subset a state-function surrogate can legitimately fit -- see the module
    docstring. Filtering is by whole JOB, never by row: a job is one path, and keeping part of it
    would leave a truncated plastic history, which is the very confound being removed.
    """
    if family in (None, "all"):
        return np.ones(len(np.asarray(job_id)), bool)
    fam, _ = classify_jobs(a, s, theta, w_lig, job_id, tol=tol)
    job_id = np.asarray(job_id)
    return np.array([fam.get(int(j)) == family for j in job_id], bool)
