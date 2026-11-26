"""Gate tests for the per-hinge expansion law of App. sec:expansion-derivation.

Three properties are load-bearing for the target derivation and are asserted here
rather than assumed:

  1. the square-root law reproduces the pitch ratio deploy_tiles actually produces,
     for every hinge of a non-congruent sheet;
  2. the law is invariant to which panel of a pair is called k, so the sign of the
     opening angle is read from the checkerboard and never guessed;
  3. Eq. (5) of the manuscript is the zero-eccentricity slice, and that slice is
     exactly r = 1/2 with the rectangular outline -- the sheet whose cuts have zero
     length. Every design with real cuts has a nonzero eccentricity.
"""

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np                                                     # noqa: E402
import jax.numpy as jnp                                                # noqa: E402
import pytest                                                          # noqa: E402

from nff.topology.closed_builder_jax import (                          # noqa: E402
    build_deploy_structure, build_boundary_edges, boundary_flat_from_logits,
    boundary_points_flat, solve_cut_vertices_jax)
from nff_paper.closed_support.shield_kinematics import (                             # noqa: E402
    build_deploy_plan, deploy_tiles, pin_residual)

A4_W, A4_H = 210.0, 297.0
PSI_VALUES = (-0.05, -0.15, -0.25, -0.35, -0.45, -0.70, 0.20)


def _design(M, N, seed=None, r_value=None, spacing=None, spacing_y=None, jitter=True):
    """One admissible sheet.

    Args:
        seed: draws random per-cut aspect ratios; ignored when r_value is given.
        r_value: uniform aspect ratio on every cut.
        jitter: True randomises the boundary positions, False pins them to the grid.

    Returns:
        (plan, flat_tiles, (spacing, spacing_y)).
    """
    sp = spacing if spacing is not None else A4_W / M
    spy = spacing_y if spacing_y is not None else A4_H / N
    struct = build_deploy_structure(M, N)
    plan = build_deploy_plan(struct, M, N, sp, spy)
    if jitter:
        rng = np.random.default_rng(0 if seed is None else seed)
        bnd = build_boundary_edges(struct, sp, spy)
        boundary = boundary_flat_from_logits(
            bnd, jnp.asarray(rng.normal(size=bnd["n_logits"]) * 0.4))
    else:
        boundary = jnp.asarray(boundary_points_flat(struct, sp, spy))
    if r_value is not None:
        r = np.full((M + 1, N + 1), float(r_value))
    else:
        r = 0.5 + 0.42 * np.tanh(np.random.default_rng(seed).normal(size=(M + 1, N + 1)))
    coords = solve_cut_vertices_jax(struct, boundary, jnp.asarray(r))
    return plan, np.asarray(coords[np.asarray(struct["corner_pid"])]), (sp, spy)


def _hinge_params(plan, flat_tiles):
    """Flat per-hinge geometry, with panel f1 taken as k throughout.

    Taking every quantity from the same panel is what makes a half-swap of the
    sign convention structurally impossible.

    Returns:
        dict of (f1, f2, p0, q, psi, ecc, axis) arrays, one entry per hinge.
    """
    pins = np.asarray(plan["pin_pairs"])
    f1, c1, f2 = pins[:, 0], pins[:, 1], pins[:, 2]
    centroids = flat_tiles.mean(axis=1)
    pivot = flat_tiles[f1, c1]
    span = centroids[f2] - centroids[f1]
    p0 = np.linalg.norm(span, axis=1)
    axial = span / p0[:, None]
    normal = np.stack([-axial[:, 1], axial[:, 0]], axis=1)
    a_k = np.einsum("ij,ij->i", pivot - centroids[f1], axial)
    q = np.einsum("ij,ij->i", pivot - centroids[f1], normal)
    return dict(f1=f1, f2=f2, p0=p0, q=q,
                psi=np.arctan2(2.0 * q, p0),
                ecc=(p0 - 2.0 * a_k) / p0,
                axis=np.where(f1 // plan["N"] != f2 // plan["N"], "x", "y"))


def _opening(plan, params, psi_deploy):
    """Per-hinge relative rotation, signed by the checkerboard."""
    return -2.0 * np.asarray(plan["sigma"])[params["f1"]] * psi_deploy


def _lambda_law(params, opening):
    """The square-root law. Degrades to Eq. (5) wherever the eccentricity vanishes."""
    bracket = (np.cos(opening / 2)
               + (2.0 * params["q"] / params["p0"]) * np.sin(opening / 2))
    return np.hypot(bracket, params["ecc"] * np.sin(opening / 2))


def _lambda_measured(plan, flat_tiles, params, psi_deploy):
    deployed = np.asarray(deploy_tiles(plan, jnp.asarray(flat_tiles), float(psi_deploy)))
    moved = deployed.mean(axis=1)
    ratio = np.linalg.norm(moved[params["f2"]] - moved[params["f1"]], axis=1) / params["p0"]
    return ratio, deployed


# ── the law itself ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("M,N", [(6, 6), (12, 17), (20, 28)])
@pytest.mark.parametrize("seed", [0, 1])
def test_square_root_law_reproduces_the_deployed_pitch(M, N, seed):
    """The law is exact on non-congruent panels, not a small-angle approximation."""
    plan, flat_tiles, _ = _design(M, N, seed=seed)
    params = _hinge_params(plan, flat_tiles)
    assert np.all(np.asarray(plan["sigma"])[params["f1"]]
                  == -np.asarray(plan["sigma"])[params["f2"]])
    for psi in PSI_VALUES:
        measured, deployed = _lambda_measured(plan, flat_tiles, params, psi)
        gap = float(jnp.max(jnp.abs(pin_residual(plan, jnp.asarray(deployed)))))
        assert gap < 1e-9, f"{M}x{N} seed={seed} psi={psi}: off the mechanism, gap {gap:.2e}"
        worst = np.abs(_lambda_law(params, _opening(plan, params, psi)) - measured).max()
        assert worst < 1e-12, f"{M}x{N} seed={seed} psi={psi}: law off by {worst:.3e}"


@pytest.mark.parametrize("M,N", [(6, 6), (12, 17)])
def test_law_is_invariant_to_which_panel_is_called_k(M, N):
    """Relabelling the pair must change nothing; half-relabelling must change a lot."""
    plan, flat_tiles, _ = _design(M, N, seed=0)
    params = _hinge_params(plan, flat_tiles)
    swapped = dict(params, f1=params["f2"], f2=params["f1"],
                   q=-params["q"], ecc=-params["ecc"])
    for psi in (-0.15, -0.35):
        law = _lambda_law(params, _opening(plan, params, psi))
        assert np.allclose(_lambda_law(swapped, _opening(plan, swapped, psi)), law,
                           rtol=0, atol=1e-15)
    # the half-swap: normal flipped, opening left alone. This is the O(1) trap.
    opening = _opening(plan, params, -0.35)
    half = _lambda_law(dict(params, q=-params["q"]), opening)
    measured, _ = _lambda_measured(plan, flat_tiles, params, -0.35)
    assert np.abs(half - measured).max() > 0.1


# ── the congruent slice, and its scope ──────────────────────────────────────

@pytest.mark.parametrize("M,N", [(8, 8), (6, 9)])
@pytest.mark.parametrize("spacing,spacing_y", [(1.0, 1.0), (1.0, 1.7)])
def test_zero_eccentricity_slice_is_r_one_half_on_the_grid(M, N, spacing, spacing_y):
    """Eq. (5) holds per hinge exactly there, and psi is the half-pitch angle."""
    plan, flat_tiles, (sx, sy) = _design(M, N, r_value=0.5, jitter=False,
                                         spacing=spacing, spacing_y=spacing_y)
    params = _hinge_params(plan, flat_tiles)
    assert np.abs(params["ecc"]).max() < 1e-12
    on_x, on_y = params["axis"] == "x", params["axis"] == "y"
    assert np.allclose(np.abs(np.tan(params["psi"][on_x])), sy / sx, rtol=0, atol=1e-12)
    assert np.allclose(np.abs(np.tan(params["psi"][on_y])), sx / sy, rtol=0, atol=1e-12)
    # at the first lock the two axes multiply to exactly two, whatever the cell shape
    psi_x, psi_y = np.arctan2(sy / 2, sx / 2), np.arctan2(sx / 2, sy / 2)
    opening = 2.0 * min(psi_x, psi_y)
    law = _lambda_law(params, _opening(plan, params, -opening / 2))
    assert np.isclose(law[on_x].mean() * law[on_y].mean(), 2.0, rtol=0, atol=1e-12)


@pytest.mark.parametrize("M,N", [(8, 8), (12, 17)])
@pytest.mark.parametrize("seed", [0, 1])
def test_eccentricity_is_not_negligible_on_a_real_design(M, N, seed):
    """Guards against simplifying the law back to Eq. (5) for the general sheet."""
    plan, flat_tiles, _ = _design(M, N, seed=seed)
    params = _hinge_params(plan, flat_tiles)
    assert np.abs(params["ecc"]).max() > 0.1
    opening = _opening(plan, params, -0.35)
    measured, _ = _lambda_measured(plan, flat_tiles, params, -0.35)
    eq5 = np.abs(np.cos(params["psi"] - opening / 2) / np.cos(params["psi"]))
    assert np.abs(eq5 - measured).max() > 1e-2


@pytest.mark.parametrize("M,N", [(8, 8), (12, 17)])
@pytest.mark.parametrize("r_value", [0.15, 0.30, 0.40])
def test_psi_spectrum_is_symmetric_under_r_to_one_minus_r(M, N, r_value):
    """A cut and its mirror place the hinges at the same angles, in some order."""
    def spectrum(r):
        plan, flat_tiles, _ = _design(M, N, r_value=r, jitter=False,
                                      spacing=1.0, spacing_y=1.0)
        params = _hinge_params(plan, flat_tiles)
        return np.sort(np.abs(params["psi"])), np.abs(params["ecc"]).max()

    low, ecc_low = spectrum(r_value)
    high, ecc_high = spectrum(1.0 - r_value)
    assert np.allclose(low, high, rtol=0, atol=1e-11)
    assert np.isclose(ecc_low, ecc_high, rtol=0, atol=1e-11)
