"""Each unit cell as a piece of effective medium: its shape tensor, and the regime that follows.

Zheng, Niloy, Tobasco, Celli & Plucinsky (Proc. R. Soc. A 479:20220665, 2023) coarse-grain a planar
kirigami into a generalised continuum. Each cell carries a **shape tensor**, their Eq. (4),

    A(xi) = (cos xi - alpha sin xi) e1 (x) e1 + (cos xi + beta sin xi) e2 (x) e2,

fixing the deformation that cell prefers at slit actuation ``xi``; the effective deformation obeys
``F^T F = A^2(xi)``, their Eq. (5). The **effective Poisson ratio** of their Eq. (17) then sets the
type of the PDE governing the actuation field: negative is elliptic and actuation decays away from
where it is applied, positive is hyperbolic and actuation persists into the bulk. Substituting
``beta = -alpha`` gives ``nu_21 = -1`` identically -- the conformal, rotating-squares case -- so
``alpha + beta`` is a one-number conformality defect.

★ Their algebra is NOT ported here, for two reasons. Their cell is parameterised by three edge
lengths and two sector angles with congruent mirrored panels and RHOMBUS slits; ours has irregular
quadrilateral panels (median 14.7 degrees off square) and PARALLELOGRAM voids, which is the
generalisation their conclusions name ("by modifying the shape tensor in Eq. (4), we can model
planar kirigami with parallelogram slits"). Porting would mean re-deriving their reduction for a
different cell and hoping the mapping survived. Instead the shape tensor is read straight off this
sheet's own exact rigid kinematics, which are already differentiable in the opening angle.

The identity that makes that legitimate: writing ``lambda_1, lambda_2`` for the stretches along the
two cell axes, Eq. (17) is algebraically

    nu_21 = - (d ln lambda_2 / d xi) / (d ln lambda_1 / d xi),

so ``jax.grad`` through :func:`nff.closed.shield_kinematics.deploy_tiles` gives their quantity with
no parameterisation in between. ``alpha`` and ``beta`` are then recovered from Eq. (4) itself.

⚠ **The cell frame must come from the cell LATTICE, never from the void's sides.** The two void side
lengths are invariant under deployment -- the whole opening lives in the angle between them -- and
in the flat sheet the void is a collapsed slit whose edge vectors are parallel, so a frame built on
them is singular at ``psi = 0``. Neighbouring cell centres stay independent at every opening.

⚠ **A(xi) is diagonal in their class; ours is not.** The manuscript records that the lattice shear
of an optimised sheet reaches 32 degrees, so the off-diagonal is an O(1) part of the deformation and
is reported here as :attr:`CellDeformation.shear_deg` rather than dropped. Whether ``alpha`` and
``beta`` come out independent of ``psi`` is the test of how far this sheet sits from their form, and
:func:`shape_tensor_drift` measures it.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int

from nff.closed.cell_families import CellLattice


@dataclass
class CellFrame:
    """Static index arrays picking each cell's two lattice neighbours, and the sign to use.

    A cell on the last row or column has no forward neighbour, so it takes the backward one and the
    sign flips. Built once from the lattice and reused at every opening angle.
    """
    neighbour_1: Int[np.ndarray, "n_cells"]
    neighbour_2: Int[np.ndarray, "n_cells"]
    sign_1: Float[np.ndarray, "n_cells"]
    sign_2: Float[np.ndarray, "n_cells"]


@dataclass
class CellDeformation:
    """The effective deformation of every unit cell at one opening angle.

    ``lambda_1`` and ``lambda_2`` are the stretches along the two cell-lattice axes, the diagonal of
    the shape tensor in the sense of Eq. (4); ``shear_deg`` is the change in angle between those two
    axes, which their diagonal form has no slot for; ``areal`` is ``det F``.
    """
    lambda_1: Float[Array, "n_cells"]
    lambda_2: Float[Array, "n_cells"]
    shear_deg: Float[Array, "n_cells"]
    areal: Float[Array, "n_cells"]


def build_cell_frame(lattice: CellLattice) -> CellFrame:
    """Pick, for each cell, the two lattice neighbours whose offsets span its frame."""
    rows, cols = lattice.shape
    n_1, n_2 = np.empty(lattice.n_cells, int), np.empty(lattice.n_cells, int)
    s_1, s_2 = np.ones(lattice.n_cells), np.ones(lattice.n_cells)
    if rows < 2 or cols < 2:
        raise ValueError(f"a {rows}x{cols} cell lattice cannot span a frame; need at least 2x2")
    for c in range(lattice.n_cells):
        r, k = divmod(c, cols)
        forward_k, forward_r = k + 1 < cols, r + 1 < rows
        n_1[c] = r * cols + (k + 1 if forward_k else k - 1)
        n_2[c] = (r + 1 if forward_r else r - 1) * cols + k
        s_1[c] = 1.0 if forward_k else -1.0
        s_2[c] = 1.0 if forward_r else -1.0
    return CellFrame(neighbour_1=n_1, neighbour_2=n_2, sign_1=s_1, sign_2=s_2)


def cell_centres(lattice: CellLattice,
                 tiles: Float[Array, "n_faces 4 2"]) -> Float[Array, "n_cells 2"]:
    """The centre of each unit cell, as the mean of its four panels' centroids."""
    return jnp.asarray(tiles).mean(axis=1)[jnp.asarray(lattice.owner_face)].mean(axis=1)


def frame_vectors(frame: CellFrame,
                  centres: Float[Array, "n_cells 2"]) -> Float[Array, "n_cells 2 2"]:
    """The two lattice vectors of each cell, as columns of a 2x2 matrix."""
    v_1 = jnp.asarray(frame.sign_1)[:, None] * (centres[frame.neighbour_1] - centres)
    v_2 = jnp.asarray(frame.sign_2)[:, None] * (centres[frame.neighbour_2] - centres)
    return jnp.stack([v_1, v_2], axis=-1)


def cell_deformation(plan, flat_tiles: Float[Array, "n_faces 4 2"], lattice: CellLattice,
                     frame: CellFrame, psi) -> CellDeformation:
    """Deploy to ``psi`` and measure every cell's effective deformation.

    Args:
        plan: from :func:`nff.closed.shield_kinematics.build_deploy_plan`.
        flat_tiles: flat panel corners of this design.
        lattice: the disjoint unit cells.
        frame: from :func:`build_cell_frame`, on the same lattice.
        psi: opening angle, traced or concrete.

    Returns:
        Stretches along the two cell axes, the lattice shear, and the areal ratio.
    """
    from nff.closed.shield_kinematics import deploy_tiles

    flat_v = frame_vectors(frame, cell_centres(lattice, jnp.asarray(flat_tiles)))
    depl_v = frame_vectors(frame, cell_centres(lattice, deploy_tiles(plan, flat_tiles, psi)))

    def norms(v):
        return jnp.linalg.norm(v[:, :, 0], axis=1), jnp.linalg.norm(v[:, :, 1], axis=1)

    flat_1, flat_2 = norms(flat_v)
    depl_1, depl_2 = norms(depl_v)

    def between(v, n_1, n_2):
        cosine = (v[:, :, 0] * v[:, :, 1]).sum(axis=1) / jnp.maximum(n_1 * n_2, 1e-30)
        return jnp.arccos(jnp.clip(cosine, -1.0, 1.0))

    shear = between(depl_v, depl_1, depl_2) - between(flat_v, flat_1, flat_2)
    return CellDeformation(
        lambda_1=depl_1 / flat_1,
        lambda_2=depl_2 / flat_2,
        shear_deg=jnp.degrees(shear),
        areal=jnp.linalg.det(depl_v) / jnp.linalg.det(flat_v),
    )


def poisson_ratio(plan, flat_tiles, lattice: CellLattice, frame: CellFrame,
                  psi: float) -> Float[Array, "n_cells"]:
    """``nu_21`` per cell at opening ``psi`` -- Eq. (17), by differentiating the kinematics.

    Negative is elliptic: a disturbance in the actuation field decays away from where it is applied.
    Positive is hyperbolic: it persists into the bulk. The rotating-squares case is exactly ``-1``.

    The ratio of log-derivatives is invariant under reversing the sign convention of ``psi``, so it
    does not matter that this sheet's openings are recorded negative.
    """
    def log_stretches(angle):
        d = cell_deformation(plan, flat_tiles, lattice, frame, angle)
        return jnp.stack([jnp.log(d.lambda_1), jnp.log(d.lambda_2)])

    rate = jax.jacfwd(log_stretches)(float(psi))
    return -rate[1] / rate[0]


def shape_tensor_parameters(plan, flat_tiles, lattice: CellLattice, frame: CellFrame,
                            psi: float) -> dict:
    """``alpha`` and ``beta`` of Eq. (4), read off the measured stretches at one opening.

    ``lambda_1 = cos psi - alpha sin psi`` and ``lambda_2 = cos psi + beta sin psi``, so both follow
    from one deployment. Their published rotating-squares example is ``alpha = -0.9, beta = 0.9``;
    the conformal condition is ``beta = -alpha``, and ``alpha + beta`` is the defect.

    ⚠ In their class ``alpha`` and ``beta`` are constants of the cell. Here they are read at a given
    ``psi`` and need not be; use :func:`shape_tensor_drift` before treating them as constants.
    """
    d = cell_deformation(plan, flat_tiles, lattice, frame, psi)
    sin_psi = np.sin(float(psi))
    if abs(sin_psi) < 1e-9:
        raise ValueError("alpha and beta are undefined at psi = 0; the sheet is closed there")
    alpha = (np.cos(float(psi)) - np.asarray(d.lambda_1)) / sin_psi
    beta = (np.asarray(d.lambda_2) - np.cos(float(psi))) / sin_psi
    return {'alpha': alpha, 'beta': beta, 'defect': alpha + beta,
            'lambda_1': np.asarray(d.lambda_1), 'lambda_2': np.asarray(d.lambda_2),
            'shear_deg': np.asarray(d.shear_deg), 'areal': np.asarray(d.areal)}


def shape_tensor_drift(plan, flat_tiles, lattice: CellLattice, frame: CellFrame,
                       angles) -> dict:
    """How far this sheet departs from Eq. (4), by re-reading ``alpha``/``beta`` at several openings.

    In their class both are constants of the cell geometry. A spread across ``angles`` measures the
    error of assuming their diagonal, angle-independent form on a sheet whose panels are irregular
    and whose lattice shears.
    """
    stacked = [shape_tensor_parameters(plan, flat_tiles, lattice, frame, a) for a in angles]
    alpha = np.stack([s['alpha'] for s in stacked])
    beta = np.stack([s['beta'] for s in stacked])
    return {'angles': np.asarray(angles), 'alpha': alpha, 'beta': beta,
            'alpha_spread': alpha.max(axis=0) - alpha.min(axis=0),
            'beta_spread': beta.max(axis=0) - beta.min(axis=0)}


def neighbour_mismatch(lattice: CellLattice,
                       values: Float[np.ndarray, "n_cells"]) -> Float[np.ndarray, "n_cells"]:
    """Mean absolute difference between a cell's value and its lattice neighbours'.

    The frustration of a non-periodic sheet is a mismatch between what neighbouring cells prefer,
    so the driver of a departure from the mechanism is pairwise and cannot be written as a per-cell
    label. This is the simplest scalar version of that mismatch.
    """
    rows, cols = lattice.shape
    grid = np.asarray(values, dtype=float).reshape(rows, cols)
    total = np.zeros_like(grid)
    count = np.zeros_like(grid)
    for shift, axis in ((1, 0), (-1, 0), (1, 1), (-1, 1)):
        rolled = np.roll(grid, shift, axis=axis)
        valid = np.ones_like(grid, dtype=bool)
        if axis == 0:
            valid[0 if shift == 1 else -1, :] = False
        else:
            valid[:, 0 if shift == 1 else -1] = False
        total += np.where(valid, np.abs(grid - rolled), 0.0)
        count += valid
    return (total / np.maximum(count, 1)).ravel()


def boundary_cells(lattice: CellLattice,
                   labels: Int[np.ndarray, "n_cells"]) -> Float[np.ndarray, "n_cells"]:
    """True where a cell has at least one lattice neighbour of a different family.

    These are the domain walls of the family map: the places where, if frustration is driven by
    neighbour incompatibility, a departure from the mechanism should concentrate.
    """
    rows, cols = lattice.shape
    grid = np.asarray(labels, dtype=int).reshape(rows, cols)
    edge = np.zeros(grid.shape, dtype=bool)
    edge[:-1, :] |= grid[:-1, :] != grid[1:, :]
    edge[1:, :] |= grid[:-1, :] != grid[1:, :]
    edge[:, :-1] |= grid[:, :-1] != grid[:, 1:]
    edge[:, 1:] |= grid[:, :-1] != grid[:, 1:]
    return edge.ravel()
