"""Unit cells and their families -- gated on the rigid deployment, where the geometry is exact.

The load-bearing claim of ``nff_paper/closed_support/cell_families.py`` is that a unit cell has exactly two
geometric degrees of freedom, so the tests here are mostly identities rather than tolerances: the
two side lengths must be unchanged by deployment, and the void area must equal the closed form
``r (1-r) L^2 sin(psi)`` to machine precision on a rigid sheet.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import jax                                                            # noqa: E402
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp                                               # noqa: E402

from nff_paper.closed_support.cell_families import (CellLattice, cell_descriptors,  # noqa: E402
                                      cell_face_values, cell_gain, cell_lattice, family_map,
                                      fit_families)
from nff_paper.closed_support.shield_kinematics import build_deploy_plan, deploy_tiles  # noqa: E402
from nff_paper.closed_support.void_geometry import (VoidIndex, build_void_index,    # noqa: E402
                                      polygon_areas, void_quads)
from nff.topology.closed_builder_jax import (boundary_flat_from_logits,  # noqa: E402
                                             build_boundary_edges, build_deploy_structure,
                                             solve_cut_vertices_jax)


def _sheet(M, N, r, spacing=1.0, spacing_y=1.0):
    """Flat panel corners and a deployment plan, the same construction the void tests use."""
    struct = build_deploy_structure(M, N)
    bnd = build_boundary_edges(struct, spacing, spacing_y)
    bflat = boundary_flat_from_logits(bnd, jnp.asarray(bnd['init_logits']))
    coords = solve_cut_vertices_jax(struct, bflat, r)
    flat = np.asarray(coords[np.asarray(struct['corner_pid'])])
    plan = build_deploy_plan(struct, M, N, spacing=spacing, spacing_y=spacing_y)
    return struct, plan, flat


def _varied_r(M, N, seed=0):
    """A cut field that is not uniform, so the cells actually differ from one another."""
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.uniform(0.28, 0.62, size=(M + 1, N + 1)))


# ── the lattice ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("M,N,cells,shape", [(15, 10, 35, (7, 5)), (8, 6, 12, (4, 3))])
def test_lattice_tiles_the_sheet_with_disjoint_cells(M, N, cells, shape):
    struct, _, _ = _sheet(M, N, 0.45)
    lattice = cell_lattice(build_void_index(struct))
    assert lattice.n_cells == cells
    assert lattice.shape == shape
    assert lattice.owner_face.shape == (cells, 4)
    flat = lattice.owner_face.ravel()
    assert np.unique(flat).size == flat.size, "the unit cells share a panel"
    assert flat.size <= M * N


def test_shared_panels_are_refused():
    """The guard must read OWNERSHIP: a silently shared panel would take two family colours."""
    struct, _, _ = _sheet(8, 6, 0.45)
    index = build_void_index(struct)
    doctored = VoidIndex(sites=index.sites, pids=index.pids,
                         owner_face=np.zeros_like(index.owner_face), owner_node=index.owner_node)
    with pytest.raises(ValueError, match="four panels"):
        cell_lattice(doctored)


# ── the two degrees of freedom ────────────────────────────────────────────────

@pytest.mark.parametrize("psi", [-0.05, -0.3, -np.pi / 4])
def test_side_lengths_are_fixed_by_the_cut_pattern(psi):
    """Deployment rotates rigid panels, so it opens the angle and changes no side length."""
    M, N = 8, 6
    struct, plan, flat = _sheet(M, N, _varied_r(M, N))
    index = build_void_index(struct)
    lattice = cell_lattice(index)

    near = cell_descriptors(index, lattice, np.asarray(deploy_tiles(plan, flat, -0.02)))
    far = cell_descriptors(index, lattice, np.asarray(deploy_tiles(plan, flat, psi)))

    assert np.abs(near.long_side - far.long_side).max() < 1e-10
    assert np.abs(near.short_side - far.short_side).max() < 1e-10
    assert np.abs(near.aspect - far.aspect).max() < 1e-10


def test_aspect_and_span_reparameterise_the_two_sides():
    M, N = 8, 6
    struct, plan, flat = _sheet(M, N, _varied_r(M, N))
    index = build_void_index(struct)
    lattice = cell_lattice(index)
    d = cell_descriptors(index, lattice, np.asarray(deploy_tiles(plan, flat, -0.3)),
                         length_scale=406.4)

    assert np.abs(d.aspect * d.span - d.short_side).max() < 1e-9
    assert np.abs((1.0 - d.aspect) * d.span - d.long_side).max() < 1e-9
    # r is recoverable only up to r <-> 1-r; the branch returned is the lower one
    assert (d.aspect > 0.0).all() and (d.aspect <= 0.5 + 1e-12).all()


@pytest.mark.parametrize("psi", [-0.1, -0.4, -np.pi / 4])
def test_void_area_equals_the_closed_form(psi):
    """``A = r (1-r) L^2 sin(psi)``. This identity is what makes the gain a property of the cell."""
    M, N = 8, 6
    struct, plan, flat = _sheet(M, N, _varied_r(M, N))
    index = build_void_index(struct)
    lattice = cell_lattice(index)
    deployed = np.asarray(deploy_tiles(plan, flat, psi))

    d = cell_descriptors(index, lattice, deployed)
    predicted = d.gain * np.sin(np.radians(d.opening_deg))
    measured = np.abs(polygon_areas(void_quads(index, deployed)))[lattice.keep]

    assert np.abs(predicted - measured).max() < 1e-10 * max(measured.max(), 1.0)


def test_gain_is_symmetric_in_r_and_peaks_at_a_half():
    r = np.linspace(0.02, 0.98, 97)
    assert np.allclose(cell_gain(r, 3.0), cell_gain(1.0 - r, 3.0))
    assert np.argmax(cell_gain(r, 3.0)) == np.argmin(np.abs(r - 0.5))


# ── families ──────────────────────────────────────────────────────────────────

def test_fitted_model_relabels_its_own_fitting_set():
    """Needed so a sheet that took no part in the fit can be labelled by the same families."""
    M, N = 8, 6
    index = build_void_index(build_deploy_structure(M, N))
    lattice = cell_lattice(index)
    pooled = []
    for seed in range(4):
        struct, plan, flat = _sheet(M, N, _varied_r(M, N, seed))
        pooled.append(cell_descriptors(build_void_index(struct), lattice,
                                       np.asarray(deploy_tiles(plan, flat, -0.3)),
                                       length_scale=406.4).sides)
    pooled = np.vstack(pooled)

    model, labels = fit_families(pooled, 3)
    assert model.n_families == 3
    assert (model.label(pooled) == labels).all()
    assert model.centroids.shape == (3, 2)
    # centroids are returned in the units they were given, not in log space
    assert model.centroids.min() > pooled.min() * 0.5


def test_family_map_reports_regions_against_chance():
    lattice = CellLattice(keep=np.arange(12), shape=(4, 3),
                          owner_face=np.arange(48).reshape(12, 4))
    striped = np.array([[0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1]]).ravel()
    scattered = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0], [1, 0, 1]]).ravel()

    assert family_map(striped, lattice)['contiguity']['ratio'] > 1.0
    assert family_map(scattered, lattice)['contiguity']['ratio'] < 1.0
    assert family_map(striped, lattice)['n_clustered'] == 12
    assert family_map(scattered, lattice)['n_clustered'] == 0


# ── per-panel fields onto cells ───────────────────────────────────────────────

def test_cell_face_values_reductions():
    lattice = CellLattice(keep=np.arange(2), shape=(1, 2),
                          owner_face=np.array([[0, 1, 2, 3], [4, 5, 6, 7]]))
    values = np.array([1.0, 2.0, 3.0, 4.0, -9.0, 0.0, 1.0, 2.0])

    assert np.allclose(cell_face_values(values, lattice, "mean"), [2.5, -1.5])
    assert np.allclose(cell_face_values(values, lattice, "sum"), [10.0, -6.0])
    assert np.allclose(cell_face_values(values, lattice, "max"), [4.0, 2.0])
    assert np.allclose(cell_face_values(values, lattice, "absmax"), [4.0, -9.0])
    with pytest.raises(ValueError, match="unknown reduction"):
        cell_face_values(values, lattice, "median")
