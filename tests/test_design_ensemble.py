"""Gates for the multi-start design ensemble.

The load-bearing claim is the symmetry quotient: if the group or its relabelling is wrong,
a design and its own image read as maximally different and the whole experiment inverts.
"""

import os
import sys

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nff.closed.design_ensemble import (  # noqa: E402
    REFLECTIONS, apply_symmetry, build_symmetry, canonical_distance, circle_problem,
    design_coords, draw_start, problem_statics)
from nff.closed.shield_design import init_params  # noqa: E402


@pytest.fixture(scope="module")
def problem():
    cfg = circle_problem(steps=200)
    statics = problem_statics(cfg)
    return cfg, statics, build_symmetry(cfg, statics)


def test_group_is_the_rotations(problem):
    """The cut family is chiral: the rotations close, the reflections do not."""
    cfg, statics, group = problem
    assert set(group) == {"e", "r90", "r180", "r270"}

    with_refl = build_symmetry(cfg, statics, include_reflections=True)
    assert not (set(with_refl) & set(REFLECTIONS)), "a reflection unexpectedly closed"


def test_symmetry_images_are_at_zero_distance(problem):
    """A design and its own rotated image are the SAME design."""
    cfg, statics, group = problem
    coords = design_coords(statics["forward_fn"](draw_start(cfg, statics, 7, 1.0)), statics)

    for name, element in group.items():
        image = apply_symmetry(coords, element)
        d = canonical_distance(coords, image, group)
        assert d < 1e-9, f"{name} image sits {d:.3e} mm from its original"


def test_distinct_designs_are_far_apart(problem):
    """The quotient must not collapse genuinely different designs."""
    cfg, statics, group = problem
    a = design_coords(statics["forward_fn"](draw_start(cfg, statics, 1, 1.0)), statics)
    b = design_coords(statics["forward_fn"](draw_start(cfg, statics, 2, 1.0)), statics)
    assert canonical_distance(a, b, group) > 10.0


def test_relabelling_is_a_permutation(problem):
    """Every group element must relabel vertices bijectively."""
    _, _, group = problem
    for name, (_, perm) in group.items():
        assert len(set(perm.tolist())) == len(perm), f"{name} is not a bijection"


def test_random_starts_are_valid_by_construction(problem):
    """Tutte validity: r stays inside its box and every tile keeps positive area."""
    cfg, statics, _ = problem
    for seed in range(6):
        out = statics["forward_fn"](draw_start(cfg, statics, seed, 1.0))
        r = np.asarray(out["r"])
        assert r.min() > cfg.r_lo and r.max() < cfg.r_hi
        from nff.closed.shield_kinematics import tile_areas
        assert np.all(np.asarray(tile_areas(out["flat_tiles"])) > 0)


def test_start_draw_is_reproducible(problem):
    """Same seed, same start -- the ensemble must be re-runnable."""
    cfg, statics, _ = problem
    a = draw_start(cfg, statics, 11, 1.0)
    b = draw_start(cfg, statics, 11, 1.0)
    for k in a:
        assert np.array_equal(np.asarray(a[k]), np.asarray(b[k]))


def test_fixed_target_drops_the_free_variables(problem):
    """A frozen target must leave the design vector, not sit in it as a dead direction."""
    cfg, statics, _ = problem
    params = init_params(cfg, statics["bnd"]["n_logits"], seed=0)
    assert set(params) == {"z", "bnd_logits", "psi_raw"}


def test_init_params_defaults_unchanged():
    """Backward compatibility: the shield's own start must be bit-identical."""
    from nff.closed.shield_design import ShieldDesignConfig
    from nff.topology.closed_builder_jax import build_boundary_edges, build_deploy_structure
    cfg = ShieldDesignConfig(M=6, N=8)
    struct = build_deploy_structure(cfg.M, cfg.N)
    bnd = build_boundary_edges(struct, cfg.spacing, cfg.spacing_y)

    params = init_params(cfg, bnd["n_logits"], seed=0)
    expected = 0.8 + 0.02 * np.random.default_rng(0).normal(size=(cfg.M + 1, cfg.N + 1))
    assert np.array_equal(np.asarray(params["z"]), expected)
    assert np.all(np.asarray(params["bnd_logits"]) == 0.0)
    assert float(params["psi_raw"]) == pytest.approx(-0.4055)
    assert "log_scale" in params and "target_shift" in params
