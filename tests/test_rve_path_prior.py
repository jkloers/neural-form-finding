"""The campaign sampler must explore the measured REGION, not reproduce the ROM's density."""
import numpy as np
import pytest

from nff.rve.path_prior import Envelope, sample_campaign_jobs

ENV = Envelope(a=(-2.2, 5.6), s=(-3.0, 2.5), theta_deg=(0.0, 76.9),
               alpha_deg=(34.9, 151.4), n_points=1_000_000, source="test")


def _split(jobs):
    return ([(g, r) for g, r in jobs if r.free_dofs],
            [(g, r) for g, r in jobs if not r.free_dofs])


def test_the_fan_targets_three_dimensional_points_not_just_rotation():
    """A theta-only campaign would never see a compression- or shear-dominated hinge."""
    _, fan = _split(sample_campaign_jobs(600, ENV, seed=0))
    a = np.array([r.eta_a * g.w_lig for g, r in fan])
    s = np.array([r.eta_s * g.w_lig for g, r in fan])

    assert (a < 0).mean() > 0.15, "compression must be sampled -- 40% of measured points have a<0"
    assert (a > 0).mean() > 0.3
    assert (np.abs(s) > np.abs(a)).mean() > 0.2, "shear-dominated hinges must appear"
    assert np.corrcoef(a, s)[0, 1] == pytest.approx(0.0, abs=0.2)   # axes sampled independently


def test_spine_leaves_the_translations_to_the_solver():
    spine, fan = _split(sample_campaign_jobs(400, ENV, seed=0, spine_frac=0.25))
    assert len(spine) == 100 and len(fan) == 300
    assert all(r.free_dofs == ("a", "s") for _, r in spine)
    assert all(r.eta_a == 0.0 and r.eta_s == 0.0 for _, r in spine)


def test_rotation_is_never_inflated_past_the_mechanism_limit():
    """Beyond 90 deg the rotating-tile mechanism has closed on itself; below 0 tiles interpenetrate."""
    jobs = sample_campaign_jobs(600, ENV, seed=1, inflate=3.0, inflate_frac=1.0)
    th = np.array([r.theta1_deg for _, r in jobs])
    assert th.min() >= 0.0 and th.max() <= Envelope.THETA_MAX_DEG

    wide = ENV.inflate(3.0)
    assert wide.theta_deg[1] == Envelope.THETA_MAX_DEG
    assert wide.a[0] == pytest.approx(3.0 * ENV.a[0])      # translations DO inflate


def test_eta_stays_physical_when_w_lig_is_small():
    """The envelope is in mm and w_lig is drawn independently, so the ratio needs a cap."""
    jobs = sample_campaign_jobs(800, ENV, seed=2, w_lig=(5.0, 50.0), eta_cap=1.2)
    _, fan = _split(jobs)
    assert max(abs(r.eta_a) for _, r in fan) <= 1.2 + 1e-9
    assert max(abs(r.eta_s) for _, r in fan) <= 1.2 + 1e-9


def test_w_lig_spans_the_design_range_log_uniformly():
    jobs = sample_campaign_jobs(600, ENV, seed=3, w_lig=(5.0, 50.0))
    w = np.array([g.w_lig for g, _ in jobs])
    assert w.min() < 5.5 and w.max() > 45.0
    # log-uniform => the median sits near the geometric mean, not the arithmetic one
    assert np.median(w) == pytest.approx(np.sqrt(5.0 * 50.0), rel=0.25)


def test_partial_runs_stay_representative():
    """Jobs are shuffled, so a campaign killed at 30% is not 30% of one corner."""
    jobs = sample_campaign_jobs(400, ENV, seed=4)
    first = [bool(r.free_dofs) for _, r in jobs[:120]]
    assert 0.1 < np.mean(first) < 0.45          # spine interleaved, not front-loaded
