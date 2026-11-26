"""Unit cells of a closed RDPQK sheet, their two-parameter geometry, and their families.

A unit cell is the four panels that meet around one void. The sheet is tiled by DISJOINT unit
cells, taken on every second void site, so no panel is claimed twice; overlapping windows share
two panels with each neighbour and that overlap alone holds spatial coherence at chance.

★ A unit cell has exactly TWO geometric degrees of freedom. With ``A`` and ``B`` the two
neighbouring cut endpoints the collinearity system links the cell to, and ``L = ||A - B||``, the
cut's own endpoints are ``r A + (1-r) B`` and ``(1-r) A + r B``, so the void's four sides are
``r L``, ``(1-r) L``, ``r L``, ``(1-r) L`` in order. Deployment rotates rigid panels and therefore
opens the angle between those sides without changing any of them: the deployed void is an exact
parallelogram of sides ``r L`` and ``(1-r) L`` at every opening angle. ``(long, short)`` and
``(r, L)`` are the same two numbers written two ways, and nothing else about the cell is free.

Two consequences worth stating before anyone extends this module:

- **The taxonomy is complete, not heuristic.** A family is a region of the ``(r, L)`` plane. Kites,
  darts and trapezoids are unreachable for the void by construction, so a shape-name classifier
  would have exactly two names to give out, parallelogram and (at ``r = 1/2``) rhombus.
- **``r`` is a local design freedom, ``L`` is not.** Only ``r_ij = sigmoid(z_ij)`` is chosen per
  cell; ``L_ij`` falls out of the global collinearity solve and so carries the whole sheet's
  design. A family therefore mixes a local choice with a sheet-wide consequence, and the two are
  reported separately rather than summed into one number.

``r`` is recoverable from the geometry only up to ``r <-> 1-r``, because the two sides are
interchangeable. :func:`cell_descriptors` returns the branch in ``(0, 1/2]``.

This module holds what the inverse-design matrix figure and the family analysis must agree on. It
was factored out of ``nff_paper/figures/plot_inverse_design_matrix.py`` so a figure and a
measurement of the same sheet cannot disagree, the same reason ``tension_routes`` is shared.
"""

from dataclasses import dataclass

import numpy as np
from jaxtyping import Bool, Float, Int

from nff.analysis.force_stats import clustered_groups, family_contiguity, motif_families
from nff_paper.closed_support.void_geometry import VoidIndex, void_quads, void_sides


@dataclass
class CellLattice:
    """The disjoint unit cells of a sheet, and where each sits on its own lattice.

    ``keep`` indexes ``VoidIndex.sites``; ``shape`` is the ``(R, C)`` lattice the cells form, in
    which a family map is drawn and neighbour statistics are taken; ``owner_face`` is the four
    panels each cell owns, which is what makes disjointness checkable.
    """
    keep: Int[np.ndarray, "n_cells"]
    shape: tuple
    owner_face: Int[np.ndarray, "n_cells 4"]

    @property
    def n_cells(self) -> int:
        return int(self.keep.size)


@dataclass
class CellDescriptors:
    """Per-cell geometry, all of it fixed by the cut pattern except ``opening_deg``.

    ``long_side`` and ``short_side`` are in the units of the node field they were measured from,
    scaled by ``length_scale``; ``aspect`` is ``r`` on its ``(0, 1/2]`` branch; ``span`` is ``L``;
    ``gain`` is ``r (1-r) L^2``, the cell's whole kinematic gain (see :func:`cell_gain`).
    """
    long_side: Float[np.ndarray, "n_cells"]
    short_side: Float[np.ndarray, "n_cells"]
    opening_deg: Float[np.ndarray, "n_cells"]
    aspect: Float[np.ndarray, "n_cells"]
    span: Float[np.ndarray, "n_cells"]
    gain: Float[np.ndarray, "n_cells"]

    @property
    def sides(self) -> Float[np.ndarray, "n_cells 2"]:
        """``(long, short)`` stacked, the descriptor the families are fitted in."""
        return np.column_stack([self.long_side, self.short_side])


@dataclass
class FamilyModel:
    """A fitted partition of the ``(long, short)`` plane, and the scaling it was fitted under.

    Carrying the log-space mean and spread alongside the centroids is what lets a sheet that took
    no part in the fit be labelled by the same families -- needed whenever a design is scored
    against families fitted elsewhere.
    """
    centroids: Float[np.ndarray, "n_families 2"]
    log_mean: Float[np.ndarray, "2"]
    log_spread: Float[np.ndarray, "2"]

    @property
    def n_families(self) -> int:
        return int(self.centroids.shape[0])

    def label(self, sides: Float[np.ndarray, "n_cells 2"]) -> Int[np.ndarray, "n_cells"]:
        """Nearest-centroid family of each cell, in the space the centroids were fitted in."""
        z = self._standardise(np.asarray(sides, dtype=float))
        centres = self._standardise(self.centroids)
        return np.argmin(((z[:, None, :] - centres[None]) ** 2).sum(axis=2), axis=1)

    def _standardise(self, sides: np.ndarray) -> np.ndarray:
        return (np.log(sides) - self.log_mean) / np.maximum(self.log_spread, 1e-12)


def cell_lattice(void_index: VoidIndex) -> CellLattice:
    """Tile the sheet with disjoint unit cells, one on every second void site.

    Args:
        void_index: from :func:`nff_paper.closed_support.void_geometry.build_void_index`.

    Returns:
        The selected cells and their lattice.

    Raises:
        ValueError: if the selected cells do not own four distinct panels each, or share any.
    """
    rows = void_index.sites[:, 0] - void_index.sites[:, 0].min()
    cols = void_index.sites[:, 1] - void_index.sites[:, 1].min()
    keep = np.flatnonzero((rows % 2 == 0) & (cols % 2 == 0))
    shape = (int(rows[keep].max() // 2 + 1), int(cols[keep].max() // 2 + 1))

    # ⚠ Disjointness is asserted from the OWNERSHIP, never from the stride arithmetic. A shared
    # panel would take two family colours, one of them silently overwritten, and any map built on
    # it would assert a clustering the sheet does not have.
    owned = [np.unique(f) for f in void_index.owner_face[keep].reshape(len(keep), -1)]
    if any(len(f) != 4 for f in owned):
        raise ValueError("a unit cell does not own exactly four panels; the void lattice and the "
                         "panel lattice have gone out of step")
    flat = np.concatenate(owned)
    if len(np.unique(flat)) != flat.size:
        raise ValueError("the selected unit cells share panels; they must tile the sheet")
    return CellLattice(keep=keep, shape=shape, owner_face=np.stack(owned))


def cell_descriptors(void_index: VoidIndex,
                     lattice: CellLattice,
                     nodes: Float[np.ndarray, "n_faces n_nodes 2"],
                     length_scale: float = 1.0) -> CellDescriptors:
    """Measure every unit cell of one deployed sheet.

    Args:
        void_index: from :func:`nff_paper.closed_support.void_geometry.build_void_index`.
        lattice: from :func:`cell_lattice`, on the same ``void_index``.
        nodes: deployed panel corners.
        length_scale: multiplies the two side lengths, to report them in mm.

    Returns:
        One :class:`CellDescriptors` for the sheet.
    """
    long_side, short_side, opening = void_sides(void_quads(void_index, nodes))
    long_side = long_side[lattice.keep] * length_scale
    short_side = short_side[lattice.keep] * length_scale
    span = long_side + short_side
    aspect = short_side / np.maximum(span, 1e-12)
    return CellDescriptors(long_side=long_side, short_side=short_side,
                           opening_deg=opening[lattice.keep], aspect=aspect, span=span,
                           gain=cell_gain(aspect, span))


def cell_gain(aspect: Float[np.ndarray, "..."],
              span: Float[np.ndarray, "..."]) -> Float[np.ndarray, "..."]:
    """``r (1-r) L^2`` -- the whole kinematic gain of a cell, in one number.

    The deployed void is a parallelogram of sides ``r L`` and ``(1-r) L`` at opening ``psi``, so
    its area is ``r (1-r) L^2 sin(psi)`` and its stroke ``dA/dpsi`` is ``r (1-r) L^2 cos(psi)``.
    Every cell of a sheet shares one ``psi`` to within half a degree, so this prefactor is the
    only thing separating one cell's contribution to the opening from another's. It is maximised
    at ``r = 1/2``, the rhombus, and vanishes as a cell degenerates to a slit.

    Args:
        aspect: ``r``, on either branch -- the expression is symmetric in ``r <-> 1-r``.
        span: ``L``, the distance between the two neighbouring cut endpoints.

    Returns:
        The gain, in the square of ``span``'s units.
    """
    r = np.asarray(aspect, dtype=float)
    return r * (1.0 - r) * np.asarray(span, dtype=float) ** 2


def fit_families(sides: Float[np.ndarray, "n_cells 2"],
                 n_families: int) -> tuple:
    """Split the ``(long, short)`` plane into ``n_families`` families of similar shape.

    Clusters on LOG side lengths, so a family is a constant RATIO band rather than a constant
    difference in mm, and the fit is not dominated by the largest cells. Pool every sheet that is
    to be compared before calling this: a per-sheet fit makes a family mean a different shape in
    every sheet, and the comparison stops being one.

    ⚠ Do not enrich ``sides``. The measurement is in :func:`nff.analysis.force_stats.motif_families`:
    appending panel-derived features makes the families measurably less coherent, because
    deployment counter-rotates neighbouring panels. ``r`` and ``L`` are safe to derive from these
    two but must not be appended to them -- they are the same two numbers.

    Args:
        sides: ``(n_cells, 2)`` from :attr:`CellDescriptors.sides`, pooled over every sheet.
        n_families: how many families, at least two.

    Returns:
        ``(model, labels)`` -- a :class:`FamilyModel` with centroids in the units of ``sides``,
        and the family of each pooled cell.
    """
    x = np.log(np.asarray(sides, dtype=float))
    labels, centroids_log = motif_families(x, n_families)
    return FamilyModel(centroids=np.exp(centroids_log),
                       log_mean=x.mean(axis=0), log_spread=x.std(axis=0)), labels


def family_map(labels: Int[np.ndarray, "n_cells"],
               lattice: CellLattice,
               min_size: int = 3) -> dict:
    """Where the families of one sheet form REGIONS, measured against their own chance level.

    Similarity and proximity are separate claims. ``mask`` is the second one: a cell is in it only
    where at least ``min_size`` same-family cells touch, so anything drawn from it asserts both.

    Args:
        labels: this sheet's cells, in ``lattice`` order.
        lattice: from :func:`cell_lattice`.
        min_size: smallest connected run of one family that counts as a group.

    Returns:
        ``grid`` ``(R, C)`` labels, ``mask`` ``(R, C)`` bool, ``groups`` ``(family, size)`` pairs
        large first, ``contiguity`` from
        :func:`nff.analysis.force_stats.family_contiguity`, and ``n_clustered``.
    """
    grid = np.asarray(labels, dtype=int).reshape(lattice.shape)
    mask, groups = clustered_groups(grid, min_size=min_size)
    return {'grid': grid, 'mask': mask, 'groups': groups,
            'contiguity': family_contiguity(grid), 'n_clustered': int(mask.sum())}


def cell_face_values(values: Float[np.ndarray, "n_faces"],
                     lattice: CellLattice,
                     reduce: str = "mean") -> Float[np.ndarray, "n_cells"]:
    """Reduce a per-panel field onto the unit cells that own those panels.

    Args:
        values: one scalar per panel.
        lattice: from :func:`cell_lattice`.
        reduce: ``mean``, ``max``, ``absmax`` or ``sum`` over a cell's four panels.

    Returns:
        One scalar per cell.
    """
    per_cell = np.asarray(values, dtype=float)[lattice.owner_face]
    if reduce == "mean":
        return per_cell.mean(axis=1)
    if reduce == "sum":
        return per_cell.sum(axis=1)
    if reduce == "max":
        return per_cell.max(axis=1)
    if reduce == "absmax":
        return per_cell[np.arange(len(per_cell)), np.abs(per_cell).argmax(axis=1)]
    raise ValueError(f"unknown reduction {reduce!r}")
