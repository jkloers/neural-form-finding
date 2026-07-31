"""The route-family audit: does it actually separate origin rays from free-DOF paths?

The audit's whole job is to decide whether the surrogate's target is multi-valued. These tests pin
the three things that decision rests on:

  * the classifier calls a proportional ray a ray -- INCLUDING the prescribed pure-rotation spine
    (``a = s = 0``), which is the steel campaign's entire spine and must not be mistaken for a
    free-DOF path just because its translations are zero;
  * the gradient correction is what makes ``sigma_pair`` work. Uncorrected, the first-order change
    in ``W`` across the gap swamps the signal and the test returns a FALSE NEGATIVE;
  * ``split_by_job`` still is the trainer's own function after being moved to ``nff.utils.splits``.
"""

import numpy as np
import pytest

from nff.scripts.diagnostics.surrogate_route_audit import (
    FREE_PATH, ORIGIN_RAY, _nonproportionality, classify_jobs, manifest_families,
    sigma_pair_by_family)


def _job(job_id, w, alpha, kind, n=20, theta1=0.9, eta_a=0.4, eta_s=-0.2, seed=0):
    """One synthetic job's columns. ``W`` is deliberately path-dependent: a state part plus the arc
    length travelled, so an origin ray is single-valued and a curve is not."""
    lam = np.linspace(1.0 / n, 1.0, n)
    th = theta1 * lam
    if kind == "ray":
        a, s = eta_a * w * lam, eta_s * w * lam
        meta_eta = (eta_a, eta_s)
    elif kind == "spine_prescribed":                 # steel: a and s IMPOSED at zero
        a, s = np.zeros(n), np.zeros(n)
        meta_eta = (0.0, 0.0)
    else:                                            # PET: solver-chosen, curved, compressive
        a = -0.35 * w * np.sin(np.pi * lam) - 0.10 * w * lam ** 2
        s = 0.12 * w * np.sin(2 * np.pi * lam)
        meta_eta = (0.0, 0.0)
    v = np.stack([a / w, s / w, th], -1)
    dv = np.diff(np.vstack([np.zeros(3), v]), axis=0)
    seg = np.linalg.norm(dv, axis=-1)
    that = dv / np.maximum(seg, 1e-12)[:, None]
    W = 40.0 * w * (0.5 * v[:, 2] ** 2 + 0.3 * v[:, 0] ** 2 + 0.2 * v[:, 1] ** 2) \
        + 55.0 * w * np.cumsum(seg)
    F_v = 40.0 * w * np.stack([0.6 * v[:, 0], 0.4 * v[:, 1], v[:, 2]], -1) + 55.0 * w * that
    cols = dict(w_lig=np.full(n, w), alpha_deg=np.full(n, alpha), a=a, s=s, theta=th, W=W,
                F_a=F_v[:, 0] / w, F_s=F_v[:, 1] / w, M_theta=F_v[:, 2],
                job_id=np.full(n, job_id))
    meta = dict(job_id=job_id, eta_a=meta_eta[0], eta_s=meta_eta[1])
    return cols, meta


def _cat(jobs):
    cols = {k: np.concatenate([j[0][k] for j in jobs]) for k in jobs[0][0]}
    return cols, [j[1] for j in jobs]


# ── the classifier ──────────────────────────────────────────────────────────────

def test_nonproportionality_is_zero_for_a_ray_and_for_identically_zero():
    th = np.linspace(0.05, 1.0, 20)
    assert _nonproportionality(0.7 * th, th, 10.0) < 1e-9
    # the steel spine: a is identically 0, so residual and scale are BOTH 0. Must be 0, not 0/0.
    assert _nonproportionality(np.zeros_like(th), th, 10.0) < 1e-9


def test_prescribed_zero_spine_is_an_origin_ray_not_a_free_path():
    """The steel campaign's spine has eta_a = eta_s = 0 exactly like PET's free-DOF spine. What
    separates them is that steel's rows really are zero. Confusing the two would make the steel
    control come back as free_path and invert the entire diagnosis."""
    cols, _ = _cat([_job(0, 10.0, 90.0, "spine_prescribed")])
    fam, _ = classify_jobs(cols)
    assert fam[0] == ORIGIN_RAY


def test_classifier_separates_rays_from_curves():
    cols, _ = _cat([_job(0, 10.0, 90.0, "ray"), _job(1, 12.0, 100.0, "free"),
                    _job(2, 8.0, 80.0, "spine_prescribed")])
    fam, diag = classify_jobs(cols)
    assert fam == {0: ORIGIN_RAY, 1: FREE_PATH, 2: ORIGIN_RAY}
    assert diag[1]["nonprop"] > diag[0]["nonprop"]


def test_manifest_agrees_with_the_row_based_classifier():
    """The manifest label is a cross-check, not the definition -- but a disagreement means the .json
    and .npz are out of sync, so the two must line up on a well-formed dataset."""
    cols, meta = _cat([_job(0, 10.0, 90.0, "ray"), _job(1, 12.0, 100.0, "free"),
                       _job(2, 8.0, 80.0, "spine_prescribed")])
    fam, _ = classify_jobs(cols)
    assert manifest_families(meta, cols) == fam


# ── sigma_pair ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def mixed_campaign():
    """Many near-identical geometries so pairs actually match, half rays and half curves."""
    rng = np.random.default_rng(0)
    jobs = []
    for j in range(120):
        w = 10.0 * float(np.exp(rng.normal(0, 0.01)))       # inside the geometry tolerance
        al = 90.0 + float(rng.normal(0, 0.5))
        kind = "ray" if j % 2 == 0 else "free"
        jobs.append(_job(j, w, al, kind, eta_a=float(rng.uniform(-0.5, 0.5)),
                         eta_s=float(rng.uniform(-0.3, 0.3)), theta1=float(rng.uniform(0.6, 1.0))))
    return _cat(jobs)


def test_gradient_correction_collapses_same_family_spread(mixed_campaign):
    """Within one family W IS single-valued here, so once the first-order term is subtracted the
    residual spread must be far smaller than the raw difference."""
    cols, _ = mixed_campaign
    fam, _ = classify_jobs(cols)
    sig, _ = sigma_pair_by_family(cols, fam, u_tol=0.15, seed=0)
    same = sig["same_origin_ray"]
    assert same["gradient_corrected"]
    assert same["sigma_pair"] < 0.4 * same["sigma_pair_uncorrected"]


def test_cross_family_spread_survives_the_correction(mixed_campaign):
    """THE decisive assertion. Two routes to the same state carry different plastic work, which no
    gradient correction can explain away -- so cross-family spread stays large while same-family
    collapses. Without the correction the ratio is near 1 and the audit reports a false negative."""
    cols, _ = mixed_campaign
    fam, _ = classify_jobs(cols)
    sig, _ = sigma_pair_by_family(cols, fam, u_tol=0.15, seed=0)
    ratio = sig["cross_family"]["sigma_pair"] / sig["same_origin_ray"]["sigma_pair"]
    raw_ratio = (sig["cross_family"]["sigma_pair_uncorrected"]
                 / sig["same_origin_ray"]["sigma_pair_uncorrected"])
    assert ratio >= 2.0, f"corrected ratio {ratio:.2f} should flag multi-valuedness"
    assert ratio > raw_ratio, "the correction must sharpen the contrast, not blur it"


# ── the split move ──────────────────────────────────────────────────────────────

def test_split_by_job_is_still_the_trainers_own_function():
    """Moved to nff.utils.splits so a numpy-only diagnostic need not import JAX. The trainer
    re-exports it, and plot_surrogate_parity imports it from there -- that must keep working."""
    from nff.scripts.train_hinge_surrogate import split_by_job as from_trainer
    from nff.utils.splits import split_by_job as from_utils
    assert from_trainer is from_utils


def test_split_by_job_is_job_disjoint_and_covers_every_row():
    from nff.utils.splits import split_by_job
    data = {"job_id": np.repeat(np.arange(40), 7)}
    tr, va, te = split_by_job(data, 0.15, 0, test_frac=0.15)
    assert (tr | va | te).all()
    assert not (tr & va).any() and not (tr & te).any() and not (va & te).any()
    for m1, m2 in ((tr, va), (tr, te), (va, te)):
        assert not set(data["job_id"][m1]) & set(data["job_id"][m2])
