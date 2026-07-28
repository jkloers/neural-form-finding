"""Observed hinge displacement paths (``nff.closed.hinge_paths``).

The load-bearing guarantee is that the ROM-side geometry reconstruction is EXACTLY the surrogate's
own -- otherwise paths extracted from a ROM run would be expressed in a different cut frame than the
one the oracle and the surrogate use, and the whole point (feeding observed paths back into
training) silently breaks.
"""
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)   # the exact-invariance assertions below need float64;
                                            # without this the file only passes when some earlier
                                            # test module happens to have enabled it first
import jax.numpy as jnp
import pytest

from nff.closed.hinge_paths import (build_hinge_geometry, extract_hinge_paths, straightness,
                                    path_diagnostics, save_paths, load_paths, HingePaths)


CFG = "data/configs/closed/sheet_4x8ft.yaml"


@pytest.fixture(scope="module")
def surrogate_setup():
    """The 4x8ft surrogate config, set up once (loading the net + descriptor structure is slow)."""
    import os
    if not os.path.exists(CFG):
        pytest.skip(f"{CFG} not present (data/ is gitignored)")
    from nff.config.experiment import load_and_parse_config
    from nff.closed.setup import (build_closed_initial_state, init_closed_les_params,
                                  build_surrogate_energy)
    config = load_and_parse_config(CFG)
    state, _ = build_closed_initial_state(config)
    params, sf = init_closed_les_params(config)
    bond_energy, _, geometry_fn, _, w0 = build_surrogate_energy(config, sf, state, params)
    if w0 is not None:
        params = {**params, 'w_lig_logit': w0}
    return config, state, sf, params, bond_energy, geometry_fn


def test_geometry_matches_the_surrogates_own(surrogate_setup):
    """build_hinge_geometry (works without a checkpoint) == setup's hinge_geometry_from_design."""
    config, state, sf, params, _, geometry_fn = surrogate_setup
    ours = build_hinge_geometry(config, sf, state, params)
    theirs = geometry_fn(params)
    assert np.allclose(np.asarray(ours.alpha), np.asarray(theirs.alpha), atol=1e-12)
    assert np.allclose(np.asarray(ours.sec_dir), np.asarray(theirs.sec_dir), atol=1e-12)
    assert np.allclose(np.asarray(ours.w_lig), np.asarray(theirs.w_lig), atol=1e-12)


def test_geometry_without_learnable_w_lig_falls_back_to_config(surrogate_setup):
    """A ROM design carries no w_lig_logit -- the manufactured width is used uniformly."""
    config, state, sf, params, _, _ = surrogate_setup
    rom_params = {k: v for k, v in params.items() if k != 'w_lig_logit'}
    geo = build_hinge_geometry(config, sf, state, rom_params)
    w = np.asarray(geo.w_lig)
    assert w.shape == (np.asarray(state.bond_connectivity).shape[0],)
    assert np.allclose(w, float(config.hinge_model.w_lig_mm))


def _rigid_solution(state, motions):
    """A stand-in SolutionData whose steps are TRUE rigid-body motions of the whole sheet.

    Rotating by theta about the origin displaces each centroid by ``R(theta)x - x`` -- NOT a uniform
    offset -- and spins each face by the same theta. (Applying one common ``(dx, dy, dtheta)`` to
    every face is *not* rigid: it translates the sheet while each face spins about its own centroid.)
    """
    X = np.asarray(state.face_centroids)

    class _Sol:
        pass
    steps = []
    for dx, dy, th in motions:
        c, s = np.cos(th), np.sin(th)
        rot = X @ np.array([[c, s], [-s, c]]) - X              # row-vector convention
        steps.append(np.concatenate([rot + np.array([dx, dy]), np.full((len(X), 1), th)], axis=1))
    sol = _Sol()
    sol.fields = jnp.asarray(np.stack(steps))
    return sol


def test_pure_translation_gives_exactly_zero_path(surrogate_setup):
    """Translating the whole sheet is not deformation, whatever reference vector is used."""
    config, state, sf, params, _, geometry_fn = surrogate_setup
    sol = _rigid_solution(state, [(0.3, -0.2, 0.0), (1.1, 0.7, 0.0)])
    paths = extract_hinge_paths(sol, state, geometry_fn(params), length_scale=406.4)
    assert np.abs(paths.u).max() < 1e-12, "translation leaked into (a, s, theta)"


def test_rigid_rotation_is_exactly_zero_under_the_closed_hinge_convention(surrogate_setup):
    """With the closed-hinge reference (ref = 0), rigid-body motion reduces to EXACTLY zero.

    This is the invariant `corotated_bond_deformation` exists to provide, and it pins that the
    extraction wires the reduction up correctly (right node gather, right corotation).
    """
    config, state, sf, params, _, geometry_fn = surrogate_setup
    sol = _rigid_solution(state, [(0.3, -0.2, 0.0), (0.0, 0.0, 0.25), (1.1, 0.7, -0.4)])
    n_hinges = np.asarray(state.bond_connectivity).shape[0]
    paths = extract_hinge_paths(sol, state, geometry_fn(params), length_scale=406.4,
                               reference_bond_vectors=np.zeros((n_hinges, 2)))
    assert np.abs(paths.u).max() < 1e-12, "rigid-body motion leaked into (a, s, theta)"


def test_nominal_l0_leaves_only_the_documented_second_order_residue(surrogate_setup):
    """The pipeline's own reference vector is NOT zero for closed hinges -- and that shows up here.

    ``build_reference_bond_vectors`` gives degenerate (closed) hinges a nominal ``l0_nom = 1e-4``
    vector along the cut axis, because the true bond ``p2 - p1`` vanishes and its direction would be
    arbitrary. That nominal vector corresponds to no real material, so a large rigid rotation leaves
    a residue ``|R(-theta) - I| * l0_nom * length_scale ~ theta * l0_nom * length_scale``.

    We assert the residue stays within that bound rather than pretending it is zero: at theta = 0.4
    it is ~0.016 mm, four orders below the ~0.1-75 mm deformations these paths actually record, so
    it is irrelevant in practice -- but it is real, and the real solve feeds the surrogate this same
    vector, so the extraction reproduces it faithfully instead of hiding it.
    """
    config, state, sf, params, _, geometry_fn = surrogate_setup
    th_max, l0_nom, ls = 0.4, 1e-4, 406.4
    sol = _rigid_solution(state, [(0.3, -0.2, 0.0), (0.0, 0.0, 0.25), (1.1, 0.7, -th_max)])
    paths = extract_hinge_paths(sol, state, geometry_fn(params), length_scale=ls)  # nominal ref
    resid = np.abs(paths.u[..., :2]).max()                     # (a, s) only; theta is exact
    assert resid < 2.0 * th_max * l0_nom * ls, f"residue {resid:.4g} mm exceeds the l0_nom bound"
    assert resid > 0.0, "expected the nominal-l0 artifact to be present, not zero"


def test_straightness_is_zero_for_a_proportional_ray():
    """A DeploymentRay is u(lambda)=lambda*u1 -- exactly what the oracle can produce."""
    lam = np.linspace(0, 1, 25)[:, None]
    ray = lam * np.array([[0.4, -0.2, 0.5]])
    assert straightness(ray) < 1e-12


def test_straightness_detects_a_bowed_path():
    """A path that bows off its own chord cannot be represented by one proportional ray."""
    lam = np.linspace(0, 1, 25)
    bowed = np.stack([lam, 0.25 * np.sin(np.pi * lam), np.zeros_like(lam)], axis=1)
    s = straightness(bowed)
    assert 0.2 < s < 0.4, s                       # ~0.25 bow over a unit-length chord


def test_straightness_of_a_motionless_hinge_is_zero():
    assert straightness(np.zeros((10, 3))) == 0.0


def test_diagnostics_and_roundtrip(tmp_path):
    """Diagnostics are well-formed and save/load preserves the arrays."""
    lam = np.linspace(0, 1, 11)
    u = np.stack([np.outer(lam, [1.0, 0.2, 0.1]), np.outer(lam, [2.0, -0.3, 0.15])], axis=1)
    paths = HingePaths(u=u, eta=u.copy(), w_lig=np.array([5.0, 5.0]), alpha=np.array([1.5, 1.6]),
                       length_scale=406.4, load_fraction=lam)
    d = path_diagnostics(paths)
    assert d['n_hinges'] == 2 and d['n_steps'] == 10
    assert d['monotonic_frac']['theta'] == 1.0    # every component increases monotonically here
    assert d['straightness_max'] < 1e-12          # both are proportional rays

    out = str(tmp_path / "p")
    save_paths(paths, out, meta={'config': 'unit-test'})
    back = load_paths(out)
    assert np.allclose(back.u, paths.u) and np.allclose(back.eta, paths.eta)
    assert back.length_scale == pytest.approx(406.4)
