"""Guards for the loss term registry, the path reductions, and the compression sign.

These cover the three ways this refactor could go wrong silently: a config quietly changing meaning
because a default moved, a path reduction that does not actually see the path, and a compression
term that penalizes the wrong sign because the hinge frame is oriented the other way.
"""
import warnings

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from nff.config.experiment import LossWeights, _parse_loss_weights, _migrate_hinge_model_weights  # noqa: E402
from nff.training.loss import TERMS, _softmax_max, _softmax_min  # noqa: E402


# ── the registry ──────────────────────────────────────────────────────────────

def test_every_term_maps_to_a_real_weight():
    """A typo in a TERMS row would silently drop that term from the objective forever."""
    fields = set(LossWeights.__dataclass_fields__)
    for name, attr, fn, needs_probe in TERMS:
        assert attr in fields, f"term {name!r} points at LossWeights.{attr}, which does not exist"
        assert callable(fn)
        assert isinstance(needs_probe, bool)


def test_term_names_are_unique():
    names = [name for name, *_ in TERMS]
    assert len(names) == len(set(names))


def test_additive_weights_default_to_zero():
    """A term must not enter the loss unless a config asks for it.

    The previous defaults (material_area 1.0, contact 1.0, stretching/shearing/bending 0.1) meant a
    config that merely omitted the block trained against contact and spring energies without ever
    saying so.
    """
    w = LossWeights()
    for name, attr, _, _ in TERMS:
        if attr == "chamfer":
            continue                      # a loss with no shape term is not an experiment
        assert getattr(w, attr) == 0.0, f"{attr} still defaults to {getattr(w, attr)}"


# ── config parsing ────────────────────────────────────────────────────────────

def test_removed_weights_warn_but_still_load():
    """Old configs carry openness/deformation. They must load, not raise."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        w = _parse_loss_weights({"chamfer": 3.0, "openness": 1.0, "deformation": 2.0})
    assert w.chamfer == 3.0
    assert any("openness" in str(c.message) for c in caught)
    assert any("deformation" in str(c.message) for c in caught)


def test_removed_weights_at_zero_are_silent():
    """Writing `openness: 0.0` was the idiom for switching it off; that should not nag."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _parse_loss_weights({"chamfer": 1.0, "openness": 0.0, "deformation": 0.0})
    assert not [c for c in caught if "openness" in str(c.message)]


def test_unknown_weights_warn_instead_of_raising():
    """The strict splat this replaced made whole config files unloadable."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        w = _parse_loss_weights({"chamfer": 2.0, "face_inversion": 1.0, "void_length": 3.0})
    assert w.chamfer == 2.0
    assert any("face_inversion" in str(c.message) for c in caught)


def test_hinge_model_weights_migrate_into_loss_weights():
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        merged = _migrate_hinge_model_weights({"w_damage": 5.0, "w_ood": 1e-3}, {"chamfer": 50.0})
    assert merged == {"chamfer": 50.0, "damage": 5.0, "ood": 1e-3}


def test_explicit_loss_weight_beats_the_legacy_hinge_model_key():
    """A config already migrated must not be overridden by a stale key left behind."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        merged = _migrate_hinge_model_weights({"w_damage": 5.0}, {"damage": 0.25})
    assert merged["damage"] == 0.25
    assert any("both set" in str(c.message) for c in caught)


# ── path reductions ───────────────────────────────────────────────────────────

def test_softmax_max_and_min_bracket_the_true_extrema():
    x = jnp.asarray(np.random.RandomState(0).randn(7, 4))
    assert np.all(np.asarray(_softmax_max(x, axis=0)) <= np.asarray(jnp.max(x, axis=0)) + 1e-9)
    assert np.all(np.asarray(_softmax_min(x, axis=0)) >= np.asarray(jnp.min(x, axis=0)) - 1e-9)


def test_path_max_exceeds_the_endpoint_on_a_non_monotone_path():
    """The whole reason damage reads the path: an excursion that returns.

    Real PEEQ is monotone along any path, but the surrogate's D is a state function, so the endpoint
    alone forgets the excursion. The path max is what recovers it.
    """
    path = jnp.asarray([[0.0], [0.2], [0.9], [0.3], [0.1]])   # (n_steps, 1 hinge)
    path_max = float(_softmax_max(path, axis=0)[0])
    endpoint = float(path[-1, 0])
    assert path_max > endpoint
    assert path_max == pytest.approx(0.9, abs=0.05)


def test_path_min_finds_transient_compression_the_endpoint_hides():
    """A hinge compressed mid-deployment has already buckled, whatever the final state says."""
    eta_a = jnp.asarray([[0.0], [-0.4], [0.2]])
    assert float(_softmax_min(eta_a, axis=0)[0]) < 0.0
    assert float(eta_a[-1, 0]) > 0.0            # the endpoint alone reports pure tension


def test_softmax_reductions_are_differentiable():
    """A hard max/min would give a piecewise-constant gradient across load steps."""
    g = jax.grad(lambda x: _softmax_max(x, axis=0).sum())(jnp.asarray([[0.0], [0.5], [0.1]]))
    assert np.all(np.isfinite(np.asarray(g)))
    assert np.abs(np.asarray(g)).sum() > 0.0


# ── compression sign ──────────────────────────────────────────────────────────

def test_compression_is_negative_eta_a_in_the_hinge_frame():
    """Pin the sign the compression term depends on.

    ``sec_dir`` is the axial (secondary-cut) direction, oriented so that separating the two tiles
    ALONG it is positive ``a`` -- the RVE's arcA->arcB direction. Nothing else in the codebase
    asserts this, and the pipeline-side sign also depends on bond_connectivity column order
    (``corotated_bond_deformation`` computes DOFs2 - DOFs1), so a silent flip here would make the
    compression term reward exactly what it is meant to penalize.
    """
    from nff.models.hinge_surrogate import hinge_kinematics

    sec = jnp.asarray([[1.0, 0.0]])                       # axial along +x
    ref = jnp.asarray([[0.0, 0.0]])                       # closed hinge
    still = jnp.zeros((1, 3))

    apart = jnp.asarray([[0.3, 0.0, 0.0]])                # node 2 moves +x, away from node 1
    a_open, _, _ = hinge_kinematics((still, apart), sec, 1.0, reference_vector=ref)
    assert float(a_open[0]) > 0.0, "separation must read as POSITIVE a (tension)"

    together = jnp.asarray([[-0.3, 0.0, 0.0]])            # node 2 moves -x, into node 1
    a_close, _, _ = hinge_kinematics((still, together), sec, 1.0, reference_vector=ref)
    assert float(a_close[0]) < 0.0, "approach must read as NEGATIVE a (compression)"


# ── no silent config drift ────────────────────────────────────────────────────

_REMOVED = ("openness", "deformation")


def _configs_with_loss_weights():
    import glob
    import yaml
    for path in sorted(glob.glob("data/configs/**/*.yaml", recursive=True)):
        with open(path) as fh:
            raw = yaml.safe_load(fh) or {}
        if isinstance(raw, dict) and isinstance(raw.get("loss_weights"), dict):
            yield path, raw


def test_every_written_weight_survives_parsing():
    """What a config writes is what the loss uses -- no key silently dropped, renamed or rescaled.

    (The one-time old-vs-new default comparison is not repeated here: it was run across the whole
    config tree at migration and is meaningless now that the configs rely on the new defaults. This
    is the invariant that stays true, and it is the one that catches a future rename.)
    """
    known = set(LossWeights.__dataclass_fields__)
    checked = 0
    for path, raw in _configs_with_loss_weights():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = _parse_loss_weights(raw["loss_weights"])
        for key, value in raw["loss_weights"].items():
            if key not in known:
                continue                  # removed/unknown: covered by the warning tests
            assert float(getattr(parsed, key)) == pytest.approx(float(value)), (
                f"{path}: wrote {key}={value}, loss sees {getattr(parsed, key)}")
        checked += 1
    assert checked > 0, "no configs found to check -- the guard would pass vacuously"


def test_omitted_weights_are_off():
    """A config that does not mention a term must not pay for it."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w = _parse_loss_weights({"chamfer": 50.0})
    for name, attr, _, _ in TERMS:
        if attr == "chamfer":
            continue
        assert getattr(w, attr) == 0.0, f"omitting {attr} still costs {getattr(w, attr)}"


def test_dropping_a_deleted_term_is_never_silent():
    """Any config that actually relied on openness/deformation must say so on load."""
    for path, raw in _configs_with_loss_weights():
        lw = raw["loss_weights"]
        if not any(float(lw.get(k) or 0.0) != 0.0 for k in _REMOVED):
            continue
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _parse_loss_weights(lw)
        assert any("is removed" in str(c.message) for c in caught), (
            f"{path} relies on a deleted term but loading it warned about nothing")


def test_every_shipped_config_still_parses():
    """The unknown-key filter exists so a stale weight cannot make a file unloadable."""
    from nff.config.experiment import load_and_parse_config
    for path, _ in _configs_with_loss_weights():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            load_and_parse_config(path)


def test_compression_penalty_is_one_sided():
    """Tension must cost nothing: the term resists compression, it does not pin eta_a to zero."""
    from nff.training.loss import _term_compression, LossContext

    def penalty(eta_a_rows):
        ctx = LossContext(results=None, initial_state=None, map_params=None, map_type='closed_les',
                          training_cfg=None, target_cloud=None, hinge_geometry=None,
                          hinge_probe={'eta_a': jnp.asarray(eta_a_rows)}, learn_global_scale=False)
        return float(_term_compression(ctx)[0])

    assert penalty([[0.0], [0.5], [0.9]]) == pytest.approx(0.0, abs=1e-12)   # pure tension: free
    assert penalty([[0.0], [-0.5], [0.2]]) > 0.0                            # transient compression
