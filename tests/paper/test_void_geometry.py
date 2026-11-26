"""Void polygons of the closed sheet -- gated on the rigid deployment, where the answer is exact."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import jax                                                            # noqa: E402
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp                                               # noqa: E402

from nff_paper.closed_support.shield_kinematics import (build_deploy_plan, deploy_tiles,  # noqa: E402
                                          pin_residual)
from nff_paper.closed_support.void_geometry import (build_void_index, parallelogram_defect,  # noqa: E402
                                      pin_gaps, polygon_areas, rhombus_defect, void_octagons,
                                      void_quads, void_report, void_sides)
from nff.topology.closed_builder import build_closed_tessellation  # noqa: E402
from nff.topology.closed_builder_jax import (boundary_flat_from_logits,  # noqa: E402
                                             build_boundary_edges, build_deploy_structure,
                                             solve_cut_vertices_jax)


def _sheet(M, N, r, spacing=1.0, spacing_y=1.0):
    struct = build_deploy_structure(M, N)
    bnd = build_boundary_edges(struct, spacing, spacing_y)
    bflat = boundary_flat_from_logits(bnd, jnp.asarray(bnd['init_logits']))
    coords = solve_cut_vertices_jax(struct, bflat, r)
    flat = np.asarray(coords[np.asarray(struct['corner_pid'])])
    plan = build_deploy_plan(struct, M, N, spacing=spacing, spacing_y=spacing_y)
    return struct, plan, flat


# ── indexing ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("M,N", [(4, 4), (6, 5), (15, 10)])
def test_void_count_is_interior_lattice_sites(M, N):
    struct, _, _ = _sheet(M, N, 0.45)
    idx = build_void_index(struct)
    assert idx.n_voids == (M - 1) * (N - 1)
    assert idx.pids.shape == (idx.n_voids, 4)
    assert idx.owner_face.shape == (idx.n_voids, 4, 2)


def test_every_void_corner_is_a_pin_of_two_panels():
    """A void corner shared by only one panel would be a free boundary corner, not a pin."""
    struct, _, _ = _sheet(6, 5, 0.45)
    idx = build_void_index(struct)
    # owners were duplicated only for single-owner pids; a genuine pin has two DISTINCT panels
    same = idx.owner_face[:, :, 0] == idx.owner_face[:, :, 1]
    assert not same.any(), "some void corner is owned by a single panel"


# ── the flat sheet: a collapsed slit ──────────────────────────────────────────

@pytest.mark.parametrize("r", [0.35, 0.45, 0.5])
def test_flat_void_is_a_collapsed_parallelogram(r):
    """The four void corners are distinct but collinear when flat, so the hole has zero area."""
    struct, _, flat = _sheet(8, 6, r)
    idx = build_void_index(struct)
    quads = void_quads(idx, flat)
    assert np.abs(polygon_areas(quads)).max() < 1e-12
    assert parallelogram_defect(quads).max() < 1e-12


# ── the rigid deployment: an exact parallelogram ──────────────────────────────

@pytest.mark.parametrize("psi", [-0.05, -0.3, -np.pi / 4])
@pytest.mark.parametrize("r", [0.4, 0.5])
def test_rigid_deployment_gives_exact_parallelograms(r, psi):
    """Lemma 1 of the LES construction. This is the gate the compliant measurement rests on."""
    struct, plan, flat = _sheet(8, 6, r)
    idx = build_void_index(struct)
    dep = np.asarray(deploy_tiles(plan, flat, psi))
    assert np.abs(np.asarray(pin_residual(plan, dep))).max() < 1e-11
    assert parallelogram_defect(void_quads(idx, dep)).max() < 1e-11
    assert pin_gaps(idx, dep).max() < 1e-11


def test_rigid_voids_are_open_and_positively_oriented():
    struct, plan, flat = _sheet(8, 6, 0.45)
    idx = build_void_index(struct)
    dep = np.asarray(deploy_tiles(plan, flat, -0.3))
    areas = polygon_areas(void_quads(idx, dep))
    assert (areas > 0).all(), "void quads are not CCW, or a void has closed"


# ── the rhombus condition separates r = 1/2 from a general design ─────────────

def test_rhombus_defect_is_the_design_constant_two_r_minus_one():
    """Void sides are (1-r)L and rL, so the spread over the mean is |1-2r|/0.5 -- and it is
    invariant in psi, because the tiles are rigid. That makes the rhombus defect a reading of the
    DESIGN, while the parallelogram defect reads the DEFORMATION."""
    for r in (0.40, 0.45):
        struct, plan, flat = _sheet(8, 6, r)
        idx = build_void_index(struct)
        for psi in (-0.1, -0.4):
            dep = np.asarray(deploy_tiles(plan, flat, psi))
            expected = abs(1.0 - 2.0 * r) / 0.5
            assert np.allclose(rhombus_defect(void_quads(idx, dep)), expected, atol=1e-9)


def test_perfect_squares_give_exact_rhombi():
    struct, plan, flat = _sheet(8, 6, 0.5)
    idx = build_void_index(struct)
    dep = np.asarray(deploy_tiles(plan, flat, -0.4))
    assert rhombus_defect(void_quads(idx, dep)).max() < 1e-11


# ── the metric must actually respond to a departure ───────────────────────────

def test_defect_detects_a_perturbed_panel():
    """A metric that is zero on the rigid family is only useful if it is non-zero off it."""
    struct, plan, flat = _sheet(8, 6, 0.45)
    idx = build_void_index(struct)
    dep = np.asarray(deploy_tiles(plan, flat, -0.3))
    assert parallelogram_defect(void_quads(idx, dep)).max() < 1e-11
    moved = dep.copy()
    moved[10, 2] += np.array([0.05, -0.02])
    d = parallelogram_defect(void_quads(idx, moved))
    assert d.max() > 1e-3
    assert (d > 1e-9).sum() <= 4, "one moved corner should disturb only the voids that touch it"


def test_octagon_collapses_onto_the_quad_when_pins_are_closed():
    struct, plan, flat = _sheet(6, 5, 0.45)
    idx = build_void_index(struct)
    dep = np.asarray(deploy_tiles(plan, flat, -0.3))
    octs = void_octagons(idx, dep)
    quads = void_quads(idx, dep)
    assert np.abs(octs[:, 0::2] - quads).max() < 1e-11
    assert np.abs(octs[:, 1::2] - quads).max() < 1e-11


def test_void_report_scales_lengths_but_not_defects():
    struct, plan, flat = _sheet(6, 5, 0.45)
    idx = build_void_index(struct)
    dep = np.asarray(deploy_tiles(plan, flat, -0.3))
    a = void_report(idx, dep, length_scale=1.0)
    b = void_report(idx, dep, length_scale=1000.0)
    assert np.isclose(b['area_total'], a['area_total'] * 1e6)
    assert np.isclose(b['para_mean'], a['para_mean'])


def test_index_matches_the_numpy_builder_corner_order():
    """`build_closed_initial_state` goes through the NumPy builder; the void recipe is derived from
    the JAX one. Their corner ordering must agree or every void polygon is scrambled."""
    M, N, r = 6, 5, 0.45
    struct, _, flat_jax = _sheet(M, N, r)
    tess = build_closed_tessellation(M, N, r=r, spacing=1.0, spacing_y=1.0)
    V = np.asarray(tess.vertices)
    flat_np = np.stack([V[np.asarray(f.vertex_indices)] for f in tess.faces])
    assert np.abs(flat_np - flat_jax).max() < 1e-12


def test_void_sides_reproduces_a_known_parallelogram():
    """Two sides of 3 and 1 at sixty degrees, walked CCW."""
    u = np.array([3.0, 0.0])
    v = np.array([np.cos(np.radians(60.0)), np.sin(np.radians(60.0))])
    quad = np.array([[0.0, 0.0], u, u + v, v])[None]
    long_side, short_side, opening = void_sides(quad)
    assert np.isclose(long_side[0], 3.0)
    assert np.isclose(short_side[0], 1.0)
    assert np.isclose(opening[0], 60.0)


def test_void_sides_orders_long_before_short_whichever_way_the_quad_is_walked():
    u = np.array([1.0, 0.0])
    v = np.array([0.0, 4.0])
    walked = np.array([[0.0, 0.0], u, u + v, v])[None]
    swapped = np.array([[0.0, 0.0], v, u + v, u])[None]
    for quad in (walked, swapped):
        long_side, short_side, opening = void_sides(quad)
        assert np.isclose(long_side[0], 4.0)
        assert np.isclose(short_side[0], 1.0)
        assert np.isclose(opening[0], 90.0)


def test_void_sides_rejects_a_non_quad():
    with pytest.raises(ValueError):
        void_sides(np.zeros((3, 5, 2)))
