"""The disturbed-radius measure, gated on synthetic fields whose radius is known by construction."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nff.closed.perturbation import perturbation_radius  # noqa: E402


def _grid(M=15, N=10):
    """Four-neighbour tile adjacency, face id = col * N + row (the sheet's own convention)."""
    pairs = []
    for i in range(M):
        for j in range(N):
            k = i * N + j
            if i + 1 < M:
                pairs.append([k, (i + 1) * N + j])
            if j + 1 < N:
                pairs.append([k, i * N + j + 1])
    return np.asarray(pairs, dtype=np.int64), M * N


def _hops(pairs, n, src):
    from nff.closed.one_dof import graph_distance
    return graph_distance(pairs, n, [src])


def test_uniform_field_has_no_disturbed_zone():
    """A flat field is nowhere above twice its own median, so every radius is absent, not zero."""
    pairs, n = _grid()
    out = perturbation_radius(np.full(n, 0.4), pairs, [145, 9, 0])
    assert all(np.isnan(r) for r in out.radius.values())
    assert np.isclose(out.baseline, 0.4)


@pytest.mark.parametrize("planted", [0, 1, 2, 3])
def test_recovers_a_planted_radius(planted):
    """A bump of the given radius around one source is measured back exactly."""
    pairs, n = _grid()
    src = 145
    d = _hops(pairs, n, src)
    v = np.full(n, 1.0)
    v[d <= planted] = 10.0                    # well above 2x the interior median of 1.0
    out = perturbation_radius(v, pairs, [src], factor=2.0)
    assert out.radius[src] == float(planted)


def test_two_sources_do_not_claim_each_others_tiles():
    """Attribution is by nearest source, so one bump cannot inflate the other's radius."""
    pairs, n = _grid()
    a, b = 145, 0
    da = _hops(pairs, n, a)
    v = np.full(n, 1.0)
    v[da <= 2] = 10.0
    out = perturbation_radius(v, pairs, [a, b], factor=2.0)
    assert out.radius[a] == 2.0
    assert np.isnan(out.radius[b])


def test_baseline_excludes_tiles_near_any_source():
    """The interior must be far from EVERY source, or a boundary layer contaminates the baseline."""
    pairs, n = _grid()
    srcs = [145, 9, 0]
    v = np.full(n, 1.0)
    for s in srcs:
        v[_hops(pairs, n, s) <= 1] = 50.0
    out = perturbation_radius(v, pairs, srcs, factor=2.0, interior_min_hops=4)
    assert np.isclose(out.baseline, 1.0), "a contaminated baseline would exceed 1.0"
    assert out.n_interior > 0


def test_factor_widens_the_zone_monotonically():
    """A lower threshold can only ever report the same radius or a larger one."""
    pairs, n = _grid()
    src = 145
    d = _hops(pairs, n, src).astype(float)
    v = 1.0 + 20.0 * np.exp(-d / 2.0)
    r_loose = perturbation_radius(v, pairs, [src], factor=1.5).radius[src]
    r_tight = perturbation_radius(v, pairs, [src], factor=4.0).radius[src]
    assert r_loose >= r_tight


def test_raises_when_the_sheet_is_too_small_for_a_baseline():
    pairs, n = _grid(4, 3)
    with pytest.raises(ValueError, match="no interior"):
        perturbation_radius(np.ones(n), pairs, [0, n - 1], interior_min_hops=9)
