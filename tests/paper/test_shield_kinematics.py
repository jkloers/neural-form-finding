"""Gate tests for the closed-form rigid-tile deployment.

Two properties are load-bearing for the whole shield pipeline and are asserted
here rather than optimised toward:

  1. every pin closes to machine precision at any opening angle, for any design;
  2. the flat sheet is an exact, gapless, zero-void tiling of the rectangle.
"""

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
import pytest
from shapely.geometry import Polygon
from shapely.ops import unary_union

from nff.topology.closed_builder_jax import (
    build_deploy_structure, build_boundary_edges, boundary_flat_from_logits,
    solve_cut_vertices_jax)
from nff_paper.closed_support.shield_kinematics import (
    build_deploy_plan, deploy_tiles, pin_residual, deployed_outline,
    tile_areas, polygon_area)

A4_W, A4_H = 210.0, 297.0
PSI_VALUES = (-0.05, -0.15, -0.25, -0.35, -0.45)


def _design(M, N, seed):
    """A random valid design: random aspect ratios and jittered boundary sliders."""
    rng = np.random.default_rng(seed)
    sp, spy = A4_W / M, A4_H / N
    struct = build_deploy_structure(M, N)
    bnd = build_boundary_edges(struct, sp, spy)
    plan = build_deploy_plan(struct, M, N, sp, spy)

    r = 0.5 + 0.42 * np.tanh(rng.normal(size=(M + 1, N + 1)))
    logits = rng.normal(size=bnd["n_logits"]) * 0.4
    boundary_flat = boundary_flat_from_logits(bnd, jnp.asarray(logits))
    coords = solve_cut_vertices_jax(struct, boundary_flat, jnp.asarray(r))
    return plan, coords[np.asarray(struct["corner_pid"])], struct


@pytest.mark.parametrize("M,N", [(6, 6), (12, 17), (20, 28)])
@pytest.mark.parametrize("seed", [0, 1])
def test_pin_closure_is_machine_precision(M, N, seed):
    """All pins close, including the ~2*n_pins - 3*n_tiles redundant ones."""
    plan, flat_tiles, _ = _design(M, N, seed)
    for psi in PSI_VALUES:
        deployed = deploy_tiles(plan, flat_tiles, psi)
        worst = float(jnp.max(jnp.abs(pin_residual(plan, deployed))))
        assert worst < 1e-9, f"{M}x{N} seed={seed} psi={psi}: pin gap {worst:.3e} mm"


@pytest.mark.parametrize("M,N", [(6, 6), (12, 17), (20, 28)])
def test_pin_system_is_overdetermined(M, N):
    """The closed form is only meaningful because there are more pins than DOFs."""
    plan, _, _ = _design(M, N, 0)
    n_equations = 2 * plan["pin_pairs"].shape[0]
    n_unknowns = 3 * plan["n_tiles"]
    assert n_equations > n_unknowns


@pytest.mark.parametrize("M,N", [(6, 6), (12, 17), (20, 28)])
@pytest.mark.parametrize("seed", [0, 1])
def test_flat_sheet_is_an_exact_closed_tiling(M, N, seed):
    """Gapless, non-overlapping, zero-void: the sheet starts closed."""
    _, flat_tiles, _ = _design(M, N, seed)
    total = float(jnp.sum(jnp.abs(tile_areas(flat_tiles))))
    rect = A4_W * A4_H
    assert abs(total - rect) / rect < 1e-9

    union = unary_union([Polygon(np.asarray(flat_tiles[k]))
                         for k in range(flat_tiles.shape[0])])
    assert abs(union.area - total) / rect < 1e-9


@pytest.mark.parametrize("M,N", [(6, 6), (12, 17), (20, 28)])
def test_outward_edge_count(M, N):
    """The outline is built from exactly one tile edge per boundary cut interval."""
    plan, _, _ = _design(M, N, 0)
    assert plan["outward_edges"].shape[0] == 2 * (M + N)


@pytest.mark.parametrize("psi", PSI_VALUES)
def test_deployment_does_not_overlap_and_conserves_area(psi):
    """Tiles are rigid, so area is conserved; on the psi < 0 branch they never collide."""
    plan, flat_tiles, _ = _design(12, 17, 0)
    deployed = deploy_tiles(plan, flat_tiles, psi)

    flat_area = float(jnp.sum(jnp.abs(tile_areas(flat_tiles))))
    dep_area = float(jnp.sum(jnp.abs(tile_areas(deployed))))
    assert abs(dep_area - flat_area) / flat_area < 1e-9

    union = unary_union([Polygon(np.asarray(deployed[k]))
                         for k in range(deployed.shape[0])])
    assert (dep_area - union.area) / dep_area < 1e-6


def test_deployed_outline_is_simple_and_encloses_the_tiles():
    """An ordered, non-self-intersecting perimeter with the tiles inside it."""
    plan, flat_tiles, _ = _design(12, 17, 0)
    deployed = deploy_tiles(plan, flat_tiles, -0.30)
    outline = deployed_outline(plan, deployed, n_sub=6)

    assert Polygon(np.asarray(outline)).is_valid
    hull = abs(float(polygon_area(outline)))
    material = float(jnp.sum(jnp.abs(tile_areas(deployed))))
    assert hull > material


@pytest.mark.parametrize("ease", ["linear", "smoothstep"])
def test_animation_sweep_is_valid_and_hits_both_endpoints(ease):
    """The frame schedule starts closed, ends open, and stays kinematically valid.

    The animation camera is fixed once, on the open state and the artwork outline, so it
    is only correct if every intermediate state stays inside that window.
    """
    from nff_paper.closed_support.shield_render import _ease_fractions, _fixed_camera

    psi = -0.45
    plan, flat_tiles, _ = _design(12, 17, 0)
    deployed = deploy_tiles(plan, flat_tiles, psi)
    flat_area = float(jnp.sum(jnp.abs(tile_areas(flat_tiles))))

    fractions = _ease_fractions(24, ease)
    assert fractions[0] == 0.0 and fractions[-1] == 1.0

    for f in fractions:
        state = deploy_tiles(plan, flat_tiles, float(f) * psi)
        assert float(jnp.max(jnp.abs(pin_residual(plan, state)))) < 1e-9
        assert abs(float(jnp.sum(jnp.abs(tile_areas(state)))) - flat_area) / flat_area < 1e-9

    # Eased endpoints are exact, so the sweep reproduces both reference states.
    assert np.allclose(np.asarray(deploy_tiles(plan, flat_tiles, 0.0)),
                       np.asarray(flat_tiles), atol=1e-12)
    assert np.allclose(np.asarray(deploy_tiles(plan, flat_tiles, float(fractions[-1]) * psi)),
                       np.asarray(deployed), atol=1e-12)

    # A target outline reaching past the tiles is the case the camera has to cover.
    target_mm = np.asarray(deployed).reshape(-1, 2) * 1.05
    centre, half = _fixed_camera(deployed, target_mm, (5.4, 6.5))
    lo, hi = centre - half, centre + half
    for f in fractions:
        pts = np.asarray(deploy_tiles(plan, flat_tiles, float(f) * psi)).reshape(-1, 2)
        assert np.all(pts >= lo) and np.all(pts <= hi)
    assert np.all(target_mm >= lo) and np.all(target_mm <= hi)


def test_animation_schedule_wipes_after_the_sheet_opens():
    """The front only starts once the sheet is fully open, and it finishes the job."""
    from nff_paper.closed_support.shield_render import _animation_schedule

    schedule = _animation_schedule(-0.45, n_frames=10, wipe_frames=6, hold_closed=2,
                                   hold_open=2, hold_logo=3)
    assert len(schedule) == 2 + 10 + 2 + 6 + 3

    kinds = [k for k, _, _ in schedule]
    assert set(kinds) == {"tiles", "wipe"}
    assert kinds.index("wipe") > max(i for i, k in enumerate(kinds) if k == "tiles")

    tiles = [a for k, a, _ in schedule if k == "tiles"]
    assert tiles[0] == 0.0 and tiles[-1] == -0.45

    wipe = [a for k, a, _ in schedule if k == "wipe"]
    assert wipe[0] == 0.0 and wipe[-1] == 1.0
    assert all(b >= a for a, b in zip(wipe, wipe[1:]))


def test_wipe_mask_is_monotone_and_saturates():
    """Fully behind the front the frame is the finished shield, so no spill survives it."""
    from nff_paper.closed_support.shield_render import _wipe_mask

    dist = np.linspace(0.0, 400.0, 401)
    mask = _wipe_mask(dist, radius_px=200.0, feather_px=40.0)[:, 0]

    assert np.all(np.diff(mask) <= 1e-12)          # never sharpens back up with distance
    assert mask[0] == 1.0 and mask[-1] == 0.0
    assert 0.0 < mask[200] < 1.0                   # the front itself is soft

    # Past the end of the sweep the mask must saturate EVERYWHERE, or stray tile pixels
    # outlive the animation. This is the property the ending depends on.
    reach, feather = 300.0, 40.0
    full = _wipe_mask(dist, radius_px=reach + 1.5 * feather, feather_px=feather)[:, 0]
    assert np.all(full[dist <= reach + feather] == 1.0)


def test_animation_schedule_fade_style_never_cross_fades():
    """The retained fade style still never washes the overlap out."""
    from nff_paper.closed_support.shield_render import _animation_schedule

    schedule = _animation_schedule(-0.45, n_frames=10, wipe_frames=5, hold_closed=2,
                                   hold_open=2, clean_frames=4, hold_logo=3, style="fade")
    for kind, a, b in schedule:
        if kind == "fade":
            assert a >= 1.0 or b >= 1.0
    assert schedule[-1][1] == 1.0 and schedule[-1][2] == 0.0
