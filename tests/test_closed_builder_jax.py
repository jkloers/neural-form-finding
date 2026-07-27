"""The differentiable flat-sheet forward map (vectorized collinearity LES).

Guards that ``solve_cut_vertices_jax`` (single-scatter assembly + dense solve) stays bit-identical to
the NumPy reference ``closed_builder`` across grid sizes -- the vectorization that lifted the ~20x20
compile ceiling must never silently change the geometry -- and that it is differentiable in ``r``.
"""
import numpy as np
import jax, jax.numpy as jnp
import pytest

from nff.topology.closed_builder import build_square_boundary_points, _assemble_les
from nff.topology.closed_builder_jax import (
    build_deploy_structure, boundary_points_flat, solve_cut_vertices_jax, build_boundary_edges)


@pytest.mark.parametrize("M,N", [(4, 4), (8, 8), (17, 12), (20, 20)])
def test_vectorized_les_matches_numpy_reference(M, N):
    struct = build_deploy_structure(M, N)
    T = struct["T"]
    bf = boundary_points_flat(struct)
    r = np.random.default_rng(0).uniform(0.28, 0.64, size=(struct["rows"], struct["cols"]))
    coords_np = np.linalg.solve(*_assemble_les(T, build_square_boundary_points(T), r))
    coords_jax = np.asarray(solve_cut_vertices_jax(struct, jnp.asarray(bf), jnp.asarray(r)))
    assert np.allclose(coords_jax, coords_np, atol=1e-9, rtol=0)


def test_les_missing_index_cache_falls_back():
    """A struct without the precomputed pattern still solves (indices recomputed from T)."""
    struct = build_deploy_structure(6, 6)
    stripped = {k: v for k, v in struct.items() if k != "les_idx"}
    r = np.full((struct["rows"], struct["cols"]), 0.45)
    a = solve_cut_vertices_jax(struct, jnp.asarray(boundary_points_flat(struct)), jnp.asarray(r))
    b = solve_cut_vertices_jax(stripped, jnp.asarray(boundary_points_flat(struct)), jnp.asarray(r))
    assert np.allclose(np.asarray(a), np.asarray(b), atol=1e-12)


@pytest.mark.parametrize("M,N,spacing,spacing_y", [(3, 3, 1.0, 2.0), (4, 4, 2.0, 0.5)])
def test_anisotropic_pitch_matches_numpy_reference(M, N, spacing, spacing_y):
    """A non-square grid pitch must flow identically through both LES paths."""
    struct = build_deploy_structure(M, N)
    T = struct["T"]
    bpts = build_square_boundary_points(T, spacing=spacing, spacing_y=spacing_y)
    bf = boundary_points_flat(struct, spacing=spacing, spacing_y=spacing_y)
    r = np.random.default_rng(0).uniform(0.28, 0.64, size=(struct["rows"], struct["cols"]))
    coords_np = np.linalg.solve(*_assemble_les(T, bpts, r))
    coords_jax = np.asarray(solve_cut_vertices_jax(struct, jnp.asarray(bf), jnp.asarray(r)))
    assert np.allclose(coords_jax, coords_np, atol=1e-9, rtol=0)


def test_spacing_y_defaults_to_spacing():
    """Omitting spacing_y reproduces the square sheet exactly (backward compatibility)."""
    struct = build_deploy_structure(5, 5)
    assert np.array_equal(boundary_points_flat(struct, spacing=1.3),
                          boundary_points_flat(struct, spacing=1.3, spacing_y=1.3))
    edges_default = build_boundary_edges(struct, spacing=1.3)
    edges_explicit = build_boundary_edges(struct, spacing=1.3, spacing_y=1.3)
    assert [e["hi"] for e in edges_default["edges"]] == [e["hi"] for e in edges_explicit["edges"]]


def test_anisotropic_sheet_spans_the_declared_rectangle():
    """3x3 panels on a 1:2 sheet: the boundary sliders span the true rectangle edges."""
    struct = build_deploy_structure(3, 3)
    bnd = build_boundary_edges(struct, spacing=1.0, spacing_y=2.0)
    spans = sorted(e["hi"] for e in bnd["edges"])
    assert spans == pytest.approx([3.0, 3.0, 6.0, 6.0])   # x edges 3 units, y edges 6 units


def test_les_differentiable_in_r():
    struct = build_deploy_structure(8, 8)
    bf = jnp.asarray(boundary_points_flat(struct))
    r0 = jnp.asarray(np.full((struct["rows"], struct["cols"]), 0.45))

    def scalar(r):
        return jnp.sum(solve_cut_vertices_jax(struct, bf, r) ** 2)

    g = np.asarray(jax.grad(scalar)(r0))
    assert np.all(np.isfinite(g)) and np.linalg.norm(g) > 1e-6
