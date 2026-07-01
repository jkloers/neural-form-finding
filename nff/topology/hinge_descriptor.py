"""Analytic local-geometry descriptor for closed-state (RDPQK) hinges.

Each hinge is the small strip of sheet left *uncut* between two tiles, at the
interior end of a cut slit (see ``nff.utils.visualization.plot_cut_pattern``). Its
local geometry is characterised by:

    - the MAIN cut   : the slit that terminates at the hinge (its end leaves the
                       ligament),
    - the SECONDARY cut: the woven slit the hinge lies on (passes through the
                       pivot), meeting the main cut at an angle ``alpha``,
    - the two cut lengths ``L_main``, ``L_sec`` (full collinear slit spans),
    - and three finite-width MANUFACTURING parameters that the idealised
      zero-width builder does not carry: ligament width, cut kerf, sheet
      thickness.

The in-plane part (cut lengths, ``alpha``, pivot positions) is an analytic,
differentiable function of the LES-solved cut vertices — hence of the design
parameters ``{r, boundary}`` (``closed_builder_jax.solve_cut_vertices_jax``). The
finite-width part is injected. This module is therefore both the training-time
descriptor extractor and the **on-the-fly inference evaluator** used when the
surrogate energy is queried inside the physics solver.

Two-part design (mirrors ``build_deploy_structure`` + ``solve_cut_vertices_jax``):

    build_hinge_descriptor_structure(M, N)     -> static NumPy index data
    compute_hinge_descriptors(hstruct, coords, manuf) -> differentiable JAX arrays

The topological classification (which slit is main/secondary for each hinge) is
invariant for ``r in (0, 1)`` (the tessellation never self-intersects), so it is
resolved once on a reference sheet and reused for every design.
"""

from typing import NamedTuple

import numpy as np
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from nff.topology.closed_builder import (
    build_topology_matrix, build_square_boundary_points,
    solve_cut_vertices, _assemble_panels, _build_hinges,
)
from nff.topology.closed_builder_jax import build_deploy_structure, solve_cut_vertices_jax


# Descriptor column layout (dimensionless, reference length L_main).
DESCRIPTOR_FIELDS = ("L_sec_over_L_main", "alpha", "lig_over_L_main",
                     "kerf_over_L_main", "t_over_L_main")

_ON_SLIT_TOL = 1e-3          # point-on-slit-line tolerance on the reference sheet
_AT_END_TOL = 1e-3          # pivot-at-slit-endpoint tolerance


class ManufacturingParams(NamedTuple):
    """Finite-width hinge parameters absent from the idealised cut model [m]."""
    ligament_width: float = 0.02
    kerf_width: float = 0.005
    thickness: float = 0.001


# ══════════════════════════════════════════════════════════════════════════════
# Static structure (NumPy, precomputed once per grid)
# ══════════════════════════════════════════════════════════════════════════════

def _all_cut_lines(T: Int[np.ndarray, "rows cols"], cols: int) -> dict:
    """Map each cut ``(i, j)`` to its full collinear slit as flat cut-vertex indices.

    Horizontal cut spans ``x_{i-1,j} -> x'_{i+1,j}``; vertical cut spans
    ``x'_{i,j-1} -> x_{i,j+1}`` (Dang et al. collinearity). **Boundary cuts are
    included**: their out-of-grid collinear neighbour is replaced by the cut's own
    pinned border vertex, so the slit runs to the sheet edge. Without these, the
    hinges at the interior ends of border cuts look like they have no terminating
    cut (spurious "through-only" flags). Degenerate corner cuts (both neighbours
    out of grid) are skipped.
    """
    rows = T.shape[0]

    def pid(i, j, s):
        return 2 * (i * cols + j) + s

    def in_grid(i, j):
        return 0 <= i < rows and 0 <= j < cols

    lines = {}
    for i in range(rows):
        for j in range(cols):
            if T[i, j] == 0:
                continue
            if abs(int(T[i, j])) == 1:                      # horizontal
                (ai, aj, as_), (bi, bj, bs) = (i - 1, j, 0), (i + 1, j, 1)
            else:                                           # vertical
                (ai, aj, as_), (bi, bj, bs) = (i, j - 1, 1), (i, j + 1, 0)
            pa = pid(ai, aj, as_) if in_grid(ai, aj) else pid(i, j, 0)
            pb = pid(bi, bj, bs) if in_grid(bi, bj) else pid(i, j, 0)
            if pa != pb:
                lines[(i, j)] = (pa, pb)
    return lines


def _classify_incident_cuts(pivot_xy, cut_lines, coords_ref):
    """Split the slits incident to a pivot into (through, terminating).

    A slit is incident if the pivot lies on it (reference geometry). It is
    'terminating' if the pivot coincides with a slit endpoint (leaving the
    ligament) and 'through' if the pivot is interior to the slit.
    """
    through, terminating = [], []
    for ij, (pa, pb) in cut_lines.items():
        a, b = coords_ref[pa], coords_ref[pb]
        ab = b - a
        L2 = float(ab @ ab)
        if L2 < 1e-12:
            continue
        t = np.clip((pivot_xy - a) @ ab / L2, 0.0, 1.0)
        if np.linalg.norm(pivot_xy - (a + t * ab)) > _ON_SLIT_TOL:
            continue
        end_dist = min(np.linalg.norm(pivot_xy - a), np.linalg.norm(pivot_xy - b))
        (terminating if end_dist < _AT_END_TOL else through).append(ij)
    return through, terminating


def build_hinge_descriptor_structure(M: int, N: int, ref_r: float = 0.4,
                                     spacing: float = 1.0) -> dict:
    """Precompute the static index data linking each hinge to its two cuts.

    Resolves — on a reference sheet — the pivot, the two adjacent tile-edge
    endpoints (for ``alpha``), and the main/secondary slit endpoints (for the cut
    lengths). Boundary hinges (whose second cut is the sheet edge, not an interior
    slit) are flagged ``is_interior = False``; their descriptor is still computed
    from the one available cut but should be filtered downstream.

    Returns a dict with (all NumPy):
        deploy_struct : from ``build_deploy_structure`` (T, rows, cols, P, ...).
        pivot_pid     : (H,) int  — cut-vertex index of each hinge pivot.
        edge_pid      : (H, 2) int — cut-vertex indices of the two adjacent
                        tile-edge endpoints (define ``alpha``).
        main_end_pid  : (H, 2) int — endpoints of the main (terminating) slit.
        sec_end_pid   : (H, 2) int — endpoints of the secondary (through) slit.
        face_pairs    : (H, 2) int — the two tiles the hinge joins.
        is_interior   : (H,) bool.
    """
    deploy_struct = build_deploy_structure(M, N)
    T, cols = deploy_struct["T"], deploy_struct["cols"]

    bpts = build_square_boundary_points(T, spacing=spacing)
    cut_vertices = solve_cut_vertices(T, bpts, ref_r)
    vertices, faces, corner_pid = _assemble_panels(T, cut_vertices)
    hinges = _build_hinges(vertices, faces, corner_pid)

    P = deploy_struct["P"]
    coords_ref = np.zeros((P, 2))
    for (i, j), xx in cut_vertices.items():
        xx = np.asarray(xx)
        coords_ref[2 * (i * cols + j) + 0] = xx[0]
        coords_ref[2 * (i * cols + j) + 1] = xx[1]

    # Global panel-vertex index -> flat cut-vertex index (one-to-one).
    vtx_to_pid = np.full(len(vertices), -1, dtype=np.int64)
    for f_id, face in enumerate(faces):
        for local, gv in enumerate(face):
            vtx_to_pid[gv] = corner_pid[f_id, local]

    cut_lines = _all_cut_lines(T, cols)
    H = len(hinges)
    pivot_pid = np.zeros(H, dtype=np.int64)
    edge_pid = np.zeros((H, 2), dtype=np.int64)
    main_end_pid = np.zeros((H, 2), dtype=np.int64)
    sec_end_pid = np.zeros((H, 2), dtype=np.int64)
    face_pairs = np.zeros((H, 2), dtype=np.int64)
    is_interior = np.zeros(H, dtype=bool)

    for hid, h in enumerate(hinges):
        p_pid = int(vtx_to_pid[h["vertex1"]])
        pivot_pid[hid] = p_pid
        edge_pid[hid] = [vtx_to_pid[h["vertex_adjacent1"]],
                         vtx_to_pid[h["vertex_adjacent2"]]]
        face_pairs[hid] = [h["face1"], h["face2"]]

        through, terminating = _classify_incident_cuts(coords_ref[p_pid], cut_lines, coords_ref)
        # Every hinge has a cut TERMINATING at it (the retracted end leaves the
        # ligament) — an interior cut, or a boundary cut running to the sheet edge.
        # main = terminating cut; sec = the through cut.
        main_ij = terminating[0] if terminating else (through[0] if through else next(iter(cut_lines)))
        sec_ij = through[0] if through else main_ij
        # "interior" hinge = both incident cuts are interior slits (T > 0); a boundary
        # hinge has one cut running to the sheet edge (different RVE outer boundary).
        is_interior[hid] = bool(T[main_ij] > 0 and T[sec_ij] > 0)
        main_end_pid[hid] = cut_lines[main_ij]
        sec_end_pid[hid] = cut_lines[sec_ij]

    return {
        "deploy_struct": deploy_struct,
        "pivot_pid": pivot_pid,
        "edge_pid": edge_pid,
        "main_end_pid": main_end_pid,
        "sec_end_pid": sec_end_pid,
        "face_pairs": face_pairs,
        "is_interior": is_interior,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Differentiable descriptor (JAX)
# ══════════════════════════════════════════════════════════════════════════════

def compute_hinge_descriptors(
        hstruct: dict,
        cut_vertices_flat: Float[Array, "P 2"],
        manufacturing: ManufacturingParams = ManufacturingParams(),
        crowding_factor: float = 3.0) -> dict:
    """Per-hinge dimensionless descriptor from solved cut vertices (differentiable).

    Args:
        hstruct: output of ``build_hinge_descriptor_structure``.
        cut_vertices_flat: (P, 2) flat cut vertices from ``solve_cut_vertices_jax``.
        manufacturing: finite-width hinge parameters [m].
        crowding_factor: a hinge is flagged crowded when the nearest other pivot is
            closer than ``crowding_factor * ligament_width``.

    Returns a dict of JAX arrays:
        descriptor : (H, 5) dimensionless — columns ``DESCRIPTOR_FIELDS``.
        L_main     : (H,) main-cut length [m] (dimensional energy prefactor).
        alpha      : (H,) tile-wedge angle [rad].
        min_gap    : (H,) distance to nearest other pivot [m].
        crowded    : (H,) bool — min_gap below the crowding threshold.
        is_interior: (H,) bool — passthrough of the boundary-hinge flag.
    """
    coords = cut_vertices_flat
    pivots = coords[hstruct["pivot_pid"]]                       # (H, 2)

    # Wedge angle from the two adjacent tile edges (unambiguous, physical).
    e1 = coords[hstruct["edge_pid"][:, 0]] - pivots
    e2 = coords[hstruct["edge_pid"][:, 1]] - pivots
    n1 = jnp.linalg.norm(e1, axis=1); n2 = jnp.linalg.norm(e2, axis=1)
    cos_a = jnp.sum(e1 * e2, axis=1) / (n1 * n2 + 1e-12)
    alpha = jnp.arccos(jnp.clip(cos_a, -1.0, 1.0))              # (H,)

    # Cut lengths (full collinear slit spans).
    main_vec = coords[hstruct["main_end_pid"][:, 1]] - coords[hstruct["main_end_pid"][:, 0]]
    sec_vec = coords[hstruct["sec_end_pid"][:, 1]] - coords[hstruct["sec_end_pid"][:, 0]]
    L_main = jnp.linalg.norm(main_vec, axis=1)                 # (H,)
    L_sec = jnp.linalg.norm(sec_vec, axis=1)

    lig = manufacturing.ligament_width
    wc = manufacturing.kerf_width
    t = manufacturing.thickness
    inv_Lm = 1.0 / (L_main + 1e-12)
    descriptor = jnp.stack([
        L_sec * inv_Lm,
        alpha,
        lig * inv_Lm,
        wc * inv_Lm,
        t * inv_Lm,
    ], axis=1)                                                 # (H, 5)

    # Crowding: nearest other pivot.
    d = jnp.linalg.norm(pivots[:, None, :] - pivots[None, :, :], axis=-1)
    d = d + jnp.eye(pivots.shape[0]) * 1e9
    min_gap = jnp.min(d, axis=1)
    crowded = min_gap < (crowding_factor * lig)

    return {
        "descriptor": descriptor,
        "L_main": L_main,
        "alpha": alpha,
        "min_gap": min_gap,
        "crowded": crowded,
        "is_interior": jnp.asarray(hstruct["is_interior"]),
    }


def hinge_descriptors_from_design(
        hstruct: dict,
        boundary_flat: Float[Array, "P 2"],
        r: Float[Array, "rows cols"],
        manufacturing: ManufacturingParams = ManufacturingParams(),
        crowding_factor: float = 3.0) -> dict:
    """Convenience/inference path: ``{r, boundary}`` -> per-hinge descriptors.

    Solves the LES for cut vertices, then evaluates ``compute_hinge_descriptors``.
    Differentiable in ``r`` and ``boundary_flat``.
    """
    coords = solve_cut_vertices_jax(hstruct["deploy_struct"], boundary_flat, r)
    return compute_hinge_descriptors(hstruct, coords, manufacturing, crowding_factor)
