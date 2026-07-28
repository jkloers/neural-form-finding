"""Path distribution over randomized starting tile positions (``nff.closed.hinge_path_ensemble``).

Two things carry the weight here. First, the variance decomposition must actually separate
between-design from within-design spread -- it is the statistic the whole exercise turns on ("does
randomizing the start change the distribution, or does one sheet already sample it?"), and a sign
error in it would flip the conclusion silently. Second, ``sampling_spec`` claims its output can be
splatted into ``sample_jobs``; that claim is tested by splatting it.
"""
import os

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)   # the pipeline runs in x64; keep this file self-sufficient
import pytest

from nff.closed.hinge_paths import HingePaths
from nff.closed.hinge_path_ensemble import (EnsembleSample, PathEnsemble, ensemble_statistics,
                                            sampling_spec, save_ensemble, load_ensemble,
                                            run_path_ensemble, resolve_length_scale)


CFG = "data/configs/closed/sheet_4x8ft_rom.yaml"


def _paths(endpoints, n_steps=6, alpha=None):
    """A HingePaths whose hinges ride straight rays to the given endpoints."""
    endpoints = np.asarray(endpoints, dtype=float)                # (n_hinges, 3)
    lam = np.linspace(0.0, 1.0, n_steps + 1)
    eta = lam[:, None, None] * endpoints[None, :, :]
    w = np.full(len(endpoints), 5.0)
    return HingePaths(u=eta.copy(), eta=eta, w_lig=w,
                      alpha=np.full(len(endpoints), 1.5) if alpha is None else np.asarray(alpha),
                      length_scale=406.4, load_fraction=lam)


def _ensemble(per_design_endpoints, alphas=None):
    ens = PathEnsemble(length_scale=406.4, n_load_steps=6)
    for i, ends in enumerate(per_design_endpoints):
        a = None if alphas is None else alphas[i]
        ens.samples.append(EnsembleSample(seed=i, noise=0.5, paths=_paths(ends, alpha=a)))
    return ens


# ── the statistic the conclusion rests on ─────────────────────────────────────────

def test_icc_is_one_when_every_design_is_its_own_cluster():
    """Hinges identical within a design, designs differing -> all variance is between-design."""
    ens = _ensemble([[[1.0, 0.0, 0.5]] * 4, [[3.0, 0.0, 0.5]] * 4])
    v = ensemble_statistics(ens)['variance_decomposition']['eta_a']
    assert v['within_design_var'] == pytest.approx(0.0, abs=1e-12)
    assert v['intraclass_corr'] == pytest.approx(1.0)


def test_icc_is_zero_when_the_random_start_changes_nothing():
    """Same spread inside every design -> the start is irrelevant; one sheet samples the family."""
    ends = [[1.0, 0.0, 0.5], [3.0, 0.0, 0.5]]
    v = ensemble_statistics(_ensemble([ends, ends]))['variance_decomposition']['eta_a']
    assert v['between_design_var'] == pytest.approx(0.0, abs=1e-12)
    assert v['intraclass_corr'] == pytest.approx(0.0)


def test_variance_parts_sum_to_the_pooled_variance():
    """between + within must reconstruct the total, or the split is not a decomposition."""
    ens = _ensemble([[[0.3, -0.2, 0.4], [1.7, 0.5, 0.6]], [[2.1, 0.1, 0.5], [0.9, -0.4, 0.7]]])
    v = ensemble_statistics(ens)['variance_decomposition']['eta_a']
    pooled = float(np.var(ens.endpoints()[:, 0]))
    assert v['between_design_var'] + v['within_design_var'] == pytest.approx(pooled)


# ── pooling ───────────────────────────────────────────────────────────────────────

def test_pooling_shapes_and_endpoint_selection():
    ens = _ensemble([[[1.0, 0.0, 0.5]] * 3, [[2.0, 0.1, 0.6]] * 3])
    assert ens.endpoints().shape == (6, 3)
    assert ens.all_points().shape == (2 * 7 * 3, 3)               # (n_steps + 1) points per path
    assert ens.alphas().shape == (6,)
    assert np.allclose(ens.endpoints()[:3, 0], 1.0)               # endpoints, not any other step
    assert np.allclose(ens.endpoints()[3:, 0], 2.0)


def test_compression_is_reported_from_whole_paths_not_just_endpoints():
    """A hinge that dips compressive mid-path but returns must still be counted."""
    ens = _ensemble([[[1.0, 0.0, 0.5]]])
    ens.samples[0].paths.eta[2, 0, 0] = -0.4                      # a dip that the endpoint hides
    c = ensemble_statistics(ens)['compression']
    assert c['min_eta_a'] == pytest.approx(-0.4)
    assert c['sample_frac'] > 0.0


# ── the bridge back into the oracle ───────────────────────────────────────────────

def test_sampling_spec_splats_into_sample_jobs():
    """The advertised contract: the spec's kwargs ARE sample_jobs' kwargs."""
    from nff.rve.dataset import sample_jobs
    ens = _ensemble([[[0.4, -0.2, 0.5], [1.2, 0.3, 0.7]], [[0.9, 0.1, 0.6], [1.6, -0.4, 0.8]]],
                    alphas=[[1.2, 1.4], [1.3, 1.5]])
    kw = sampling_spec(ens)['sample_jobs_kwargs']
    jobs = sample_jobs(12, seed=0, **kw)
    assert len(jobs) == 12
    for geo, ray in jobs:
        assert kw['alpha_deg'][0] - 1e-9 <= geo.alpha_deg <= kw['alpha_deg'][1] + 1e-9
        assert kw['theta1_deg'][0] - 1e-9 <= ray.theta1_deg <= kw['theta1_deg'][1] + 1e-9


def test_sampling_spec_brackets_the_observed_data():
    """Padded quantile ranges must contain the bulk of what was measured."""
    rng = np.random.default_rng(0)
    ens = _ensemble([rng.normal([1.0, 0.0, 0.6], [0.3, 0.2, 0.05], size=(8, 3)) for _ in range(6)])
    kw = sampling_spec(ens)['sample_jobs_kwargs']
    pts = ens.all_points()
    inside = np.mean((pts[:, 0] >= kw['eta_a'][0]) & (pts[:, 0] <= kw['eta_a'][1]))
    assert inside > 0.97, inside


def test_sampling_spec_flags_a_compressive_requirement():
    """The oracle assumes eta_a >= 0; a spec that needs otherwise must say so."""
    pos = _ensemble([[[1.0, 0.0, 0.5], [2.0, 0.0, 0.6]]] * 4)
    assert sampling_spec(pos)['requires_negative_eta_a'] is False
    neg = _ensemble([[[-0.8, 0.0, 0.5], [-0.6, 0.0, 0.6]]] * 4)
    assert sampling_spec(neg)['requires_negative_eta_a'] is True


def test_sampling_spec_refuses_to_invent_a_w_lig_range():
    """The ROM has no learnable ligament width, so measuring it would be degenerate."""
    spec = sampling_spec(_ensemble([[[1.0, 0.0, 0.5]] * 3] * 3))
    assert 'w_lig' not in spec['sample_jobs_kwargs']
    assert 'NOT MEASURED' in spec['w_lig']


def test_statistics_reject_an_empty_ensemble():
    with pytest.raises(ValueError, match="empty ensemble"):
        ensemble_statistics(PathEnsemble())


# ── persistence ───────────────────────────────────────────────────────────────────

def test_save_load_roundtrip(tmp_path):
    ens = _ensemble([[[1.0, 0.2, 0.5], [1.4, -0.1, 0.6]], [[2.0, 0.3, 0.7], [0.8, 0.0, 0.4]]])
    summary = save_ensemble(ens, str(tmp_path), meta={'note': 'unit-test'})
    assert summary['note'] == 'unit-test' and 'sampling_spec' in summary
    back = load_ensemble(str(tmp_path))
    assert back.n_samples == 2 and [s.seed for s in back.samples] == [0, 1]
    assert np.allclose(back.endpoints(), ens.endpoints())


# ── against the real pipeline ─────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def rom_config():
    if not os.path.exists(CFG):
        pytest.skip(f"{CFG} not present (data/ is gitignored)")
    from nff.config.experiment import load_and_parse_config
    return load_and_parse_config(CFG)


def test_randomizing_the_start_actually_moves_the_paths(rom_config):
    """The whole premise: init_noise must change the deployment, not just the design vector.

    Guards against the failure where the perturbed design is built but never reaches Stage 2 (e.g.
    the state is rebuilt at r_init and the map params are dropped), which would silently produce
    ``n_samples`` copies of one deployment and a fake ICC of zero.
    """
    ens = run_path_ensemble(rom_config, n_samples=2, noise=0.5, seed0=0, n_load_steps=3,
                            verbose=False)
    assert ens.n_samples == 2, ens.failures
    a, b = ens.samples[0].paths.eta[-1], ens.samples[1].paths.eta[-1]
    assert not np.allclose(a, b), "different random starts produced identical paths"
    assert np.all(np.isfinite(a)) and np.all(np.isfinite(b))


def test_zero_noise_is_the_deterministic_start(rom_config):
    """noise = 0 must reproduce one design exactly, whatever the seed -- the degeneracy check."""
    ens = run_path_ensemble(rom_config, n_samples=2, noise=0.0, seed0=7, n_load_steps=3,
                            verbose=False)
    assert ens.n_samples == 2, ens.failures
    assert np.allclose(ens.samples[0].paths.eta, ens.samples[1].paths.eta, atol=1e-12)


def test_length_scale_prefers_the_declared_panel_pitch(rom_config):
    assert resolve_length_scale(rom_config) == pytest.approx(406.4)


def test_geometry_fn_matches_the_one_shot_builder(rom_config):
    """The hoisted-static-work closure must equal the convenience wrapper it replaced."""
    from nff.closed.setup import build_closed_initial_state, init_closed_les_params
    from nff.closed.hinge_paths import build_hinge_geometry, build_hinge_geometry_fn
    state, _ = build_closed_initial_state(rom_config)
    params, sf = init_closed_les_params(rom_config)
    a = build_hinge_geometry(rom_config, sf, state, params)
    b = build_hinge_geometry_fn(rom_config, sf, state)(params)
    assert np.allclose(np.asarray(a.alpha), np.asarray(b.alpha), atol=1e-12)
    assert np.allclose(np.asarray(a.sec_dir), np.asarray(b.sec_dir), atol=1e-12)
    assert np.allclose(np.asarray(a.w_lig), np.asarray(b.w_lig), atol=1e-12)
