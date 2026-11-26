"""The per-cell shape tensor and its regime, gated on the case the literature already solved.

The load-bearing test is :func:`test_rotating_squares_is_conformal`: a uniform ``r = 1/2`` sheet is
the rotating-squares pattern, for which Zheng et al. (Proc. R. Soc. A 479:20220665, Eq. 28) give
``nu_21 = -1`` identically. Reproducing that from our own kinematics, to machine precision and at
every opening angle, is what licenses reading the same quantity off the optimised sheets, where no
published answer exists to check against.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax                                                            # noqa: E402
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp                                               # noqa: E402

from nff.closed.cell_families import CellLattice, cell_lattice        # noqa: E402
from nff.closed.effective_medium import (boundary_cells,              # noqa: E402
                                         build_cell_frame, cell_centres, cell_deformation,
                                         neighbour_mismatch, poisson_ratio,
                                         shape_tensor_parameters)
from nff.closed.shield_kinematics import build_deploy_plan            # noqa: E402
from nff.closed.void_geometry import build_void_index                 # noqa: E402
from nff.topology.closed_builder_jax import (boundary_flat_from_logits,  # noqa: E402
                                             build_boundary_edges, build_deploy_structure,
                                             solve_cut_vertices_jax)


def _sheet(M, N, r, spacing=1.0, spacing_y=1.0):
    struct = build_deploy_structure(M, N)
    bnd = build_boundary_edges(struct, spacing, spacing_y)
    bflat = boundary_flat_from_logits(bnd, jnp.asarray(bnd['init_logits']))
    flat = solve_cut_vertices_jax(struct, bflat, r)[np.asarray(struct['corner_pid'])]
    plan = build_deploy_plan(struct, M, N, spacing=spacing, spacing_y=spacing_y)
    lattice = cell_lattice(build_void_index(struct))
    return plan, flat, lattice, build_cell_frame(lattice)


# ── the published case ────────────────────────────────────────────────────────

@pytest.mark.parametrize("M,N", [(8, 8), (10, 8)])
@pytest.mark.parametrize("psi", [-0.15, -0.35, -0.6])
def test_rotating_squares_is_conformal(M, N, psi):
    """Eq. 28 of Zheng et al.: the rotating-squares pattern has ``nu_21 = -1`` at every opening."""
    plan, flat, lattice, frame = _sheet(M, N, 0.5)

    nu = np.asarray(poisson_ratio(plan, flat, lattice, frame, psi))
    assert np.abs(nu + 1.0).max() < 1e-9, "the conformal case must return exactly -1"

    params = shape_tensor_parameters(plan, flat, lattice, frame, psi)
    # nu_21 = -1 identically iff beta = -alpha, so the defect is the conformality measure
    assert np.abs(params['defect']).max() < 1e-9
    # and their diagonal form is exact here: no lattice shear at all
    assert np.abs(params['shear_deg']).max() < 1e-9


def test_a_general_design_is_not_conformal():
    """The gate above must not pass trivially: an off-square design has to depart from -1."""
    rng = np.random.default_rng(0)
    r = jnp.asarray(rng.uniform(0.25, 0.65, size=(9, 7)))
    plan, flat, lattice, frame = _sheet(8, 6, r)

    params = shape_tensor_parameters(plan, flat, lattice, frame, -0.4)
    assert np.abs(params['defect']).max() > 1e-3
    assert np.abs(params['shear_deg']).max() > 1e-3


# ── internal consistency ──────────────────────────────────────────────────────

def test_areal_ratio_matches_its_own_stretches_and_shear():
    """``det F`` must equal ``lambda_1 lambda_2 sin(theta_dep) / sin(theta_flat)``."""
    rng = np.random.default_rng(1)
    r = jnp.asarray(rng.uniform(0.3, 0.6, size=(9, 7)))
    plan, flat, lattice, frame = _sheet(8, 6, r)

    from nff.closed.effective_medium import frame_vectors
    flat_v = np.asarray(frame_vectors(frame, cell_centres(lattice, jnp.asarray(flat))))

    def angle(v):
        cosine = ((v[:, :, 0] * v[:, :, 1]).sum(axis=1)
                  / (np.linalg.norm(v[:, :, 0], axis=1) * np.linalg.norm(v[:, :, 1], axis=1)))
        return np.arccos(np.clip(cosine, -1.0, 1.0))

    psi = -0.4
    d = cell_deformation(plan, flat, lattice, frame, psi)
    theta_flat = angle(flat_v)
    theta_dep = theta_flat + np.radians(np.asarray(d.shear_deg))
    predicted = np.asarray(d.lambda_1) * np.asarray(d.lambda_2) * np.sin(theta_dep) / np.sin(theta_flat)

    assert np.abs(predicted - np.abs(np.asarray(d.areal))).max() < 1e-9


def test_flat_sheet_is_undeformed():
    plan, flat, lattice, frame = _sheet(8, 6, 0.45)
    d = cell_deformation(plan, flat, lattice, frame, 0.0)
    assert np.abs(np.asarray(d.lambda_1) - 1.0).max() < 1e-12
    assert np.abs(np.asarray(d.lambda_2) - 1.0).max() < 1e-12
    assert np.abs(np.asarray(d.shear_deg)).max() < 1e-10


def test_alpha_beta_undefined_on_the_closed_sheet():
    plan, flat, lattice, frame = _sheet(8, 6, 0.45)
    with pytest.raises(ValueError, match="undefined at psi = 0"):
        shape_tensor_parameters(plan, flat, lattice, frame, 0.0)


def test_poisson_ratio_is_invariant_to_the_sign_convention_of_psi():
    """It is a ratio of log-derivatives, so opening 'negative' must not change the answer."""
    plan, flat, lattice, frame = _sheet(8, 6, 0.5)
    a = np.asarray(poisson_ratio(plan, flat, lattice, frame, -0.3))
    b = np.asarray(poisson_ratio(plan, flat, lattice, frame, +0.3))
    assert np.abs(a - b).max() < 1e-9


# ── the frame ─────────────────────────────────────────────────────────────────

def test_frame_uses_a_backward_neighbour_on_the_last_row_and_column():
    lattice = CellLattice(keep=np.arange(6), shape=(2, 3),
                          owner_face=np.arange(24).reshape(6, 4))
    frame = build_cell_frame(lattice)
    # cell 2 is at (0, 2): no forward column neighbour, so it takes cell 1 with a sign flip
    assert frame.neighbour_1[2] == 1 and frame.sign_1[2] == -1.0
    # cell 0 is at (0, 0): forward in both
    assert frame.neighbour_1[0] == 1 and frame.sign_1[0] == 1.0
    assert frame.neighbour_2[0] == 3 and frame.sign_2[0] == 1.0


def test_a_degenerate_lattice_is_refused():
    lattice = CellLattice(keep=np.arange(3), shape=(1, 3),
                          owner_face=np.arange(12).reshape(3, 4))
    with pytest.raises(ValueError, match="cannot span a frame"):
        build_cell_frame(lattice)


# ── neighbour statistics ──────────────────────────────────────────────────────

def test_neighbour_mismatch_and_boundary_cells():
    lattice = CellLattice(keep=np.arange(6), shape=(2, 3),
                          owner_face=np.arange(24).reshape(6, 4))
    values = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])          # a step between the two rows
    mismatch = neighbour_mismatch(lattice, values)
    # The mismatch is a MEAN over however many neighbours a cell has, so a corner cell (two
    # neighbours, one across the step) reads 1/2 and a middle-of-edge cell (three) reads 1/3.
    assert np.allclose(mismatch, [0.5, 1 / 3, 0.5, 0.5, 1 / 3, 0.5])

    uniform = neighbour_mismatch(lattice, np.ones(6))
    assert np.allclose(uniform, 0.0)

    labels = np.array([0, 0, 0, 1, 1, 1])
    assert boundary_cells(lattice, labels).all(), "a two-row step puts every cell on the wall"
    assert not boundary_cells(lattice, np.zeros(6, int)).any()
