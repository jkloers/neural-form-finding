"""The solver-convergence quarantine: rows a DIVERGED solve left behind are not ground truth.

A diverged CalculiX job stopped at the buckling bifurcation, and the increments it retained sit on
an equilibrium branch the converged jobs never occupy. On the PET campaign those rows were 4% of the
data and 50% of the surrogate's squared error. The reason was recorded only in the sidecar manifest,
so nothing reading the npz could tell them apart -- these tests pin both the new per-row column and
the manifest back-fill that makes every EXISTING campaign filterable.
"""
import json
import os

import numpy as np
import pytest

from nff.rve.hinge_function import (STOP_COMPLETED, STOP_DIVERGED, STOP_UNCONVERGED, STOP_UNKNOWN,
                                    STOP_NAME, stop_code)
from nff.scripts.train_hinge_surrogate import _stop_codes, converged_mask, load_dataset, split_by_job

MERGED = "data/fea/hinge_dataset_pet_merged"


def test_stop_code_round_trips_every_name():
    for code, name in STOP_NAME.items():
        assert stop_code(name) == code
    assert stop_code("something ccx never says") == STOP_UNKNOWN


def test_only_diverged_and_unknown_are_quarantined():
    """timeout/fractured solves CONVERGED -- they just stopped early, so their rows are valid."""
    assert STOP_DIVERGED in STOP_UNCONVERGED and STOP_UNKNOWN in STOP_UNCONVERGED
    assert stop_code("timeout") not in STOP_UNCONVERGED
    assert stop_code("fractured") not in STOP_UNCONVERGED
    assert STOP_COMPLETED not in STOP_UNCONVERGED


def test_stop_codes_prefers_the_npz_column_over_the_manifest():
    d = {"stop_code": np.array([STOP_DIVERGED, STOP_COMPLETED])}
    out = _stop_codes("/nonexistent/path", d, np.array([0, 1]))
    assert list(out) == [STOP_DIVERGED, STOP_COMPLETED]


def test_stop_codes_returns_none_when_the_campaign_never_said(tmp_path):
    """None, NOT a column of `unknown` -- `unknown` is quarantined, so guessing drops everything."""
    assert _stop_codes(str(tmp_path / "absent"), {}, np.array([0, 1])) is None
    stem = str(tmp_path / "nojobs")
    with open(stem + ".json", "w") as f:
        json.dump({"columns": []}, f)
    assert _stop_codes(stem, {}, np.array([0, 1])) is None


def test_converged_mask_is_all_true_without_labels():
    assert converged_mask({"W": np.zeros(5)}).all()


def test_manifest_backfill_maps_job_ids_not_row_order(tmp_path):
    stem = str(tmp_path / "ds")
    with open(stem + ".json", "w") as f:
        json.dump({"jobs": [{"job_id": 7, "stop_reason": "diverged"},
                            {"job_id": 3, "stop_reason": "completed"}]}, f)
    out = _stop_codes(stem, {}, np.array([3, 7, 7, 3]))
    assert list(out) == [STOP_COMPLETED, STOP_DIVERGED, STOP_DIVERGED, STOP_COMPLETED]
    # a job absent from the manifest cannot be vouched for
    assert _stop_codes(stem, {}, np.array([99]))[0] == STOP_UNKNOWN


@pytest.mark.skipif(not os.path.exists(MERGED + ".npz"), reason="data/ is gitignored")
def test_pet_campaign_backfill_reproduces_the_measured_counts():
    """The numbers the 2026-07-30 investigation was built on -- if these move, its conclusions do."""
    data, _, _ = load_dataset(MERGED, w_lig_max=25.0)
    assert "stop_code" in data, "the merged PET campaign must be filterable via its manifest"
    conv = converged_mask(data)
    assert int((~conv).sum()) == 658
    assert len(np.unique(data["job_id"][~conv])) == 85


@pytest.mark.skipif(not os.path.exists(MERGED + ".npz"), reason="data/ is gitignored")
def test_quarantine_leaves_job_membership_untouched():
    """The policy is applied AFTER the split, so the three policies stay comparable."""
    data, _, _ = load_dataset(MERGED, w_lig_max=25.0)
    tr, va = split_by_job(data, 0.15, 0)
    conv = converged_mask(data)
    # every row the quarantine removes was already in exactly one of the splits
    assert not (tr & va).any()
    assert ((tr | va) & ~conv).sum() == (~conv).sum()
    # and it never promotes a row from one split into another
    assert ((tr & conv) | (va & conv)).sum() == int(conv.sum())


def test_stabilize_is_refused_not_silently_ignored():
    """CalculiX 2.23 has no *STATIC, STABILIZE -- it ignores the parameter and returns UNDAMPED
    results that look damped. Four on/off pairs matched W to 5 significant figures before this
    guard, which is exactly how a dead knob survives as a documented lever."""
    from nff.rve.ccx_solver import _write_deck
    with pytest.raises(NotImplementedError, match="STABILIZE"):
        _write_deck("/tmp/unused.inp", np.zeros((1, 3)), [], [0], [0], (0.0, 0.0),
                    None, [(0.0, 0.0, 0.1)], False, 1, stabilize=2e-4)


# ── force-sign probe ────────────────────────────────────────────────────────────

def _ray_job(job_id, n=8, sign=+1.0, pure_rotation=False):
    """One proportional ray with W = 0.5*k*|u|^2 and F = +dW/du (times `sign`)."""
    lam = np.linspace(0.1, 1.0, n)
    u1 = np.array([0.0, 0.0, 0.5]) if pure_rotation else np.array([0.3, -0.2, 0.5])
    u = lam[:, None] * u1
    W = 0.5 * np.sum(u ** 2, axis=1)
    return dict(job_id=np.full(n, job_id), u=u, W=W, F=sign * u)


def _merge(*jobs):
    return {k: np.concatenate([j[k] for j in jobs]) for k in jobs[0]}


def test_force_sign_uses_the_work_identity_when_no_pure_rotation_ray_exists():
    """The PET failure mode: no job has a=s=0, so the spine probe finds nothing. It must NOT
    silently return +1 -- the old code did, on every PET run."""
    from nff.scripts.train_hinge_surrogate import check_force_sign
    sign, ratio, how = check_force_sign(_merge(_ray_job(0), _ray_job(1)))
    assert how == "work"
    assert sign == 1.0 and abs(ratio - 1.0) < 0.05


def test_force_sign_detects_a_flipped_convention():
    """A check that cannot fail is not a check: a negated F must come back -1."""
    from nff.scripts.train_hinge_surrogate import check_force_sign
    sign, ratio, how = check_force_sign(_merge(_ray_job(0, sign=-1.0), _ray_job(1, sign=-1.0)))
    assert sign == -1.0 and ratio < 0 and how == "work"


def test_force_sign_prefers_the_direct_spine_probe_when_one_exists():
    from nff.scripts.train_hinge_surrogate import check_force_sign
    _, _, how = check_force_sign(_merge(_ray_job(0, pure_rotation=True), _ray_job(1)))
    assert how == "spine"


def test_force_sign_reports_unverified_rather_than_passing_silently():
    from nff.scripts.train_hinge_surrogate import check_force_sign
    sign, ratio, how = check_force_sign(
        dict(job_id=np.zeros(2), u=np.zeros((2, 3)), W=np.zeros(2), F=np.zeros((2, 3))))
    assert how == "unverified" and np.isnan(ratio) and sign == 1.0
