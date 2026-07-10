"""Tests for the analytic hinge descriptor map (nff.topology.hinge_descriptor).

Covers: static structure integrity, descriptor values against the known 4x4
geometry, differentiability in r, scale-invariance of the dimensionless
descriptor, the crowding filter, and consistency of the inference path.

Run: JAX_PLATFORMS=cpu conda run -n kgnn_mac python -m pytest tests/test_hinge_descriptor.py -q
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import jax
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)
import pytest

from nff.topology.closed_builder_jax import boundary_points_flat
from nff.topology.hinge_descriptor import (
    build_hinge_descriptor_structure,
    compute_hinge_descriptors,
    hinge_descriptors_from_design,
    ManufacturingParams,
    DESCRIPTOR_FIELDS,
)

M = N = 4
R = 0.4
MANUF = ManufacturingParams(ligament_width=0.02, kerf_width=0.005, thickness=0.001)


@pytest.fixture(scope="module")
def hstruct():
    return build_hinge_descriptor_structure(M, N, ref_r=R)


@pytest.fixture(scope="module")
def design(hstruct):
    struct = hstruct["deploy_struct"]
    boundary_flat = jnp.asarray(boundary_points_flat(struct, spacing=1.0))
    r_arr = jnp.full((struct["rows"], struct["cols"]), R)
    return boundary_flat, r_arr


@pytest.fixture(scope="module")
def out(hstruct, design):
    boundary_flat, r_arr = design
    return hinge_descriptors_from_design(hstruct, boundary_flat, r_arr, MANUF)


# ── Static structure ──────────────────────────────────────────────────────────

def test_structure_shapes_and_counts(hstruct):
    H = len(hstruct["pivot_pid"])
    assert H == 24                                   # 4x4 -> 24 corner hinges
    for key, shape in [("pivot_pid", (H,)), ("edge_pid", (H, 2)),
                       ("main_end_pid", (H, 2)), ("sec_end_pid", (H, 2)),
                       ("face_pairs", (H, 2)), ("is_interior", (H,))]:
        assert hstruct[key].shape == shape, key


@pytest.mark.parametrize("grid", [(4, 4), (6, 6)])
def test_rom_green_superposition(grid):
    """The descriptor hinges must superimpose the closed-builder ROM hinges (the
    source of truth): same count and same positions. Boundary hinges (ends of
    border cuts) are included — no spurious hinges are invented or dropped."""
    from scipy.spatial import cKDTree
    from nff.topology.closed_builder import build_closed_tessellation
    from nff.topology.closed_builder_jax import boundary_points_flat, solve_cut_vertices_jax
    m, n = grid
    tess = build_closed_tessellation(m, n, r=R, spacing=1.0)
    rom = np.array([tess.vertices[h.vertex1] for h in tess.hinges])

    hs = build_hinge_descriptor_structure(m, n, ref_r=R)
    bf = jnp.asarray(boundary_points_flat(hs["deploy_struct"], 1.0))
    coords = np.asarray(solve_cut_vertices_jax(hs["deploy_struct"], bf,
                                               jnp.full((n + 1, n + 1), R)))
    piv = coords[np.asarray(hs["pivot_pid"])]

    assert len(piv) == len(rom)                      # same number
    d, _ = cKDTree(rom).query(piv)
    assert d.max() < 1e-4                             # every green hinge on a ROM hinge
    d2, _ = cKDTree(piv).query(rom)
    assert d2.max() < 1e-4                            # and vice-versa (bijection)


def test_structure_pids_in_range(hstruct):
    P = hstruct["deploy_struct"]["P"]
    for key in ("pivot_pid", "edge_pid", "main_end_pid", "sec_end_pid"):
        arr = np.asarray(hstruct[key])
        assert arr.min() >= 0 and arr.max() < P, key


def test_face_pairs_distinct(hstruct):
    fp = np.asarray(hstruct["face_pairs"])
    assert np.all(fp[:, 0] != fp[:, 1])


# ── Descriptor values ─────────────────────────────────────────────────────────

def test_descriptor_columns(out):
    assert out["descriptor"].shape == (24, len(DESCRIPTOR_FIELDS))


def test_all_finite(out):
    for k in ("descriptor", "L_main", "alpha", "min_gap"):
        assert bool(jnp.all(jnp.isfinite(out[k]))), k


def test_alpha_bounds(out):
    a = out["alpha"]
    assert float(a.min()) > 0.0 and float(a.max()) < np.pi


def test_interior_descriptor_sane(out):
    im = np.asarray(out["is_interior"])
    d = np.asarray(out["descriptor"])
    L = np.asarray(out["L_main"])
    assert np.all(L[im] > 0)
    assert np.all((d[im, 0] > 0.5) & (d[im, 0] < 2.0))          # L_sec/L_main
    assert np.all((d[im, 1] > np.radians(60)) & (d[im, 1] < np.radians(120)))  # alpha


def test_known_hinge_h10(out):
    """H10 is an interior hinge with a hand-checked geometry.

    alpha is now the RVE-frame value (compute_hinge_descriptors delegates to compute_hinge_frame):
    87.1deg, the supplement of the old edge-based 92.9deg. For this near-90deg hinge the two branches
    nearly coincide; the frame resolves the correct (material-side) one. See test_hinge_frame.py.
    """
    assert bool(out["is_interior"][10])
    assert float(out["L_main"][10]) == pytest.approx(1.990, abs=2e-3)
    assert float(out["descriptor"][10, 0]) == pytest.approx(1.001, abs=2e-3)
    assert np.degrees(float(out["alpha"][10])) == pytest.approx(87.1, abs=0.2)


def test_manufacturing_ratios(out):
    """Columns 2-4 must equal manuf param / L_main exactly."""
    L = out["L_main"]
    np.testing.assert_allclose(out["descriptor"][:, 2], MANUF.ligament_width / L, rtol=1e-9)
    np.testing.assert_allclose(out["descriptor"][:, 3], MANUF.kerf_width / L, rtol=1e-9)
    np.testing.assert_allclose(out["descriptor"][:, 4], MANUF.thickness / L, rtol=1e-9)


# ── Differentiability ─────────────────────────────────────────────────────────

def test_grad_wrt_r_finite_nonzero(hstruct, design):
    boundary_flat, r_arr = design

    def loss(rv):
        o = hinge_descriptors_from_design(hstruct, boundary_flat, rv, MANUF)
        return jnp.sum(jnp.where(o["is_interior"], o["descriptor"][:, 0], 0.0))

    g = jax.grad(loss)(r_arr)
    assert bool(jnp.all(jnp.isfinite(g)))
    assert float(jnp.linalg.norm(g)) > 1e-6


# ── Scale invariance ──────────────────────────────────────────────────────────

def test_scale_invariance(hstruct, design, out):
    boundary_flat, r_arr = design
    out2 = hinge_descriptors_from_design(hstruct, boundary_flat * 2.0, r_arr, MANUF)
    im = np.asarray(out["is_interior"])
    # alpha and L_sec/L_main invariant under uniform scaling
    np.testing.assert_allclose(np.asarray(out2["alpha"])[im],
                               np.asarray(out["alpha"])[im], atol=1e-10)
    np.testing.assert_allclose(np.asarray(out2["descriptor"])[im, 0],
                               np.asarray(out["descriptor"])[im, 0], atol=1e-10)
    # L_main scales linearly
    np.testing.assert_allclose(out2["L_main"], 2.0 * out["L_main"], rtol=1e-9)


# ── Crowding filter ───────────────────────────────────────────────────────────

def test_crowding_toggles_with_ligament_width(hstruct, design):
    boundary_flat, r_arr = design
    tight = ManufacturingParams(ligament_width=1.0, kerf_width=0.005, thickness=0.001)
    loose = ManufacturingParams(ligament_width=1e-4, kerf_width=0.005, thickness=0.001)
    o_tight = hinge_descriptors_from_design(hstruct, boundary_flat, r_arr, tight)
    o_loose = hinge_descriptors_from_design(hstruct, boundary_flat, r_arr, loose)
    assert bool(jnp.any(o_tight["crowded"]))          # huge ligament -> some crowded
    assert not bool(jnp.any(o_loose["crowded"]))      # tiny ligament -> none crowded


def test_min_gap_matches_bruteforce(out, hstruct, design):
    boundary_flat, r_arr = design
    from nff.topology.closed_builder_jax import solve_cut_vertices_jax
    coords = np.asarray(solve_cut_vertices_jax(hstruct["deploy_struct"], boundary_flat, r_arr))
    piv = coords[np.asarray(hstruct["pivot_pid"])]
    D = np.linalg.norm(piv[:, None] - piv[None], axis=-1)
    np.fill_diagonal(D, np.inf)
    np.testing.assert_allclose(np.asarray(out["min_gap"]), D.min(axis=1), rtol=1e-9)


# ── Inference-path consistency ────────────────────────────────────────────────

def test_from_design_matches_compute(hstruct, design):
    boundary_flat, r_arr = design
    from nff.topology.closed_builder_jax import solve_cut_vertices_jax
    coords = solve_cut_vertices_jax(hstruct["deploy_struct"], boundary_flat, r_arr)
    a = compute_hinge_descriptors(hstruct, coords, MANUF)
    b = hinge_descriptors_from_design(hstruct, boundary_flat, r_arr, MANUF)
    np.testing.assert_allclose(np.asarray(a["descriptor"]), np.asarray(b["descriptor"]))


def test_determinism(hstruct, design):
    boundary_flat, r_arr = design
    a = hinge_descriptors_from_design(hstruct, boundary_flat, r_arr, MANUF)["descriptor"]
    b = hinge_descriptors_from_design(hstruct, boundary_flat, r_arr, MANUF)["descriptor"]
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
