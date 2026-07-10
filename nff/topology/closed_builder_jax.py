"""Differentiable JAX forward map for closed-state RDPQK kirigami.

Implements the Dang et al. (2021) inverse-design forward map as a differentiable
function of the per-cut aspect ratios ``r`` and the boundary-slider positions:

    {r, boundary logits} -> LES -> flat cut vertices.

The static topology (which cut endpoint each panel corner takes) is precomputed
once with NumPy in ``build_deploy_structure``; the differentiable part
(``solve_cut_vertices_jax``) is pure JAX and JIT/grad-friendly. The flat panel
vertices it produces are mapped into a ``CentroidalState`` by
``apply_closed_les_mapping`` (Stage 0); the actual deployment is then performed by
the Stage-2 physics solver.

Genus-0 patterns built from the LES have parallelogram voids (Lemma 1) and convex
boundary (ordered sliders), satisfying Tutte's embedding conditions — so the
assembled flat tessellation is non-self-intersecting for any r in (0, 1).
"""

import numpy as np
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from nff.topology.closed_builder import build_topology_matrix, build_square_boundary_points


# ══════════════════════════════════════════════════════════════════════════════
# Static structure (NumPy, precomputed once)
# ══════════════════════════════════════════════════════════════════════════════

def _panel_corner_sources(T):
    """For each panel corner, the source cut-point index into the flat (P, 2) array.

    Uses the validated panel->vertex rule (depends on the panel's own cut type
    ``h = |t_{i,j}| == 1``). Point index of endpoint ``s`` of cut ``(ci, cj)`` is
    ``2 * (ci * cols + cj) + s`` — matching the LES ordering.

    Returns:
        corner_pid: (n_panels, 4) int — point index per panel corner (BL, BR, TR, TL).
    """
    rows, cols = T.shape
    M, N = rows - 1, cols - 1

    def pid(ci, cj, s):
        return 2 * (ci * cols + cj) + s

    corner_pid = []
    for i in range(M):
        for j in range(N):
            h = abs(int(T[i, j])) == 1
            corner_pid.append([
                pid(i, j, 1 if h else 0),          # BL
                pid(i + 1, j, 1),                  # BR
                pid(i + 1, j + 1, 0 if h else 1),  # TR
                pid(i, j + 1, 0),                  # TL
            ])
    return np.array(corner_pid, dtype=np.int64)


def _les_assembly_indices(T):
    """Static sparsity pattern of the collinearity LES, precomputed once (NumPy).

    The LES is ``A x = b`` with ``A`` = identity diagonal plus, for each INTERIOR cut point, two
    off-diagonal entries with coefficients ``-r`` and ``-(1-r)``; boundary points fix ``x = boundary``.
    We precompute the off-diagonal ``(row, col, which r, r-is-r-or-1minusr)`` pattern and the boundary
    mask so ``solve_cut_vertices_jax`` assembles ``A`` with a single vectorized scatter -- instead of a
    Python double-loop of ``.at[].set`` that unrolled O(P) scatter ops into the traced graph and blew
    up compile time (the old ~20x20 ceiling). See ``closed_builder._assemble_les`` for the same rule.

    Returns off_rows/off_cols/off_ridx (int, length E=4*n_interior) + off_is_r (bool) + boundary_mask.
    """
    rows, cols = T.shape
    P = 2 * rows * cols

    def pid(i, j, s):
        return 2 * (i * cols + j) + s

    off_rows, off_cols, off_ridx, off_is_r = [], [], [], []
    boundary_mask = np.zeros(P, dtype=bool)
    for i in range(rows):
        for j in range(cols):
            t = int(T[i, j])
            if t < 0:                                       # boundary cut: x = boundary_flat
                boundary_mask[pid(i, j, 0)] = True
                boundary_mask[pid(i, j, 1)] = True
                continue
            a0, b0 = ((pid(i - 1, j, 0), pid(i + 1, j, 1)) if t == 1        # horizontal
                      else (pid(i, j - 1, 1), pid(i, j + 1, 0)))            # t == 2, vertical
            p0, p1 = pid(i, j, 0), pid(i, j, 1)
            # p0 = r*a0 + (1-r)*b0 ;  p1 = (1-r)*a0 + r*b0  (matches the sign convention in _assemble_les)
            for row, col, is_r in ((p0, a0, True), (p0, b0, False), (p1, a0, False), (p1, b0, True)):
                off_rows.append(row); off_cols.append(col)
                off_ridx.append(i * cols + j); off_is_r.append(is_r)
    return {
        "off_rows": np.array(off_rows, dtype=np.int64),
        "off_cols": np.array(off_cols, dtype=np.int64),
        "off_ridx": np.array(off_ridx, dtype=np.int64),
        "off_is_r": np.array(off_is_r, dtype=bool),
        "boundary_mask": boundary_mask,
    }


def build_deploy_structure(M: int, N: int):
    """Precompute the static (NumPy) data needed for the differentiable forward map.

    Returns the topology matrix ``T``, the cut-grid dimensions (``rows``, ``cols``) and flat
    cut-vertex count ``P``, ``corner_pid`` — each panel corner's index into the flat cut-vertex array —
    and ``les_idx``, the precomputed LES sparsity pattern for the vectorized solve.
    """
    T = build_topology_matrix(M, N)
    rows, cols = T.shape
    P = 2 * rows * cols
    corner_pid = _panel_corner_sources(T)
    return {"T": T, "rows": rows, "cols": cols, "P": P, "corner_pid": corner_pid,
            "les_idx": _les_assembly_indices(T)}


def boundary_points_flat(struct, spacing: float = 1.0) -> np.ndarray:
    """(P, 2) array with boundary-cut positions filled (uniform square), 0 elsewhere."""
    T, cols, P = struct["T"], struct["cols"], struct["P"]
    bpts = build_square_boundary_points(T, spacing=spacing)
    flat = np.zeros((P, 2), dtype=float)
    for (i, j), xy in bpts.items():
        base = 2 * (i * cols + j)
        flat[base] = xy
        flat[base + 1] = xy
    return flat


# ── Ordered (non-crossing) boundary parameterization ──────────────────────────
# Boundary sliders on each edge are kept strictly ordered via a cumulative-softmax
# of unconstrained logits. This preserves the convexity of the boundary polygon —
# the single precondition Tutte's embedding theorem needs — so the assembled flat
# tessellation is GUARANTEED non-self-intersecting for any r in (0, 1), with no
# validity loss and full placement freedom along each edge.

def build_boundary_edges(struct, spacing: float = 1.0):
    """Group non-corner boundary cuts by edge, in order, for monotonic sliders.

    Returns a dict with the (P, 2) template (corners filled), a list of edges
    (each with ordered point indices, free axis, and [lo, hi] span), the per-edge
    logit counts, and an all-zeros init that reproduces the uniform grid spacing.
    """
    T, rows, cols = struct["T"], struct["rows"], struct["cols"]
    template = boundary_points_flat(struct, spacing=spacing)
    x_max = (rows - 1) * spacing      # bottom/top edges span x
    y_max = (cols - 1) * spacing      # left/right edges span y

    edges = []
    for j in (0, cols - 1):           # bottom, top — sliders vary in i (x)
        pb = [2 * (i * cols + j) for i in range(1, rows - 1)]
        edges.append({"pbase": np.array(pb, dtype=np.int64), "axis": 0, "lo": 0.0, "hi": x_max})
    for i in (0, rows - 1):           # left, right — sliders vary in j (y)
        pb = [2 * (i * cols + j) for j in range(1, cols - 1)]
        edges.append({"pbase": np.array(pb, dtype=np.int64), "axis": 1, "lo": 0.0, "hi": y_max})

    logit_sizes = [len(e["pbase"]) + 1 for e in edges]   # k sliders -> k+1 gaps

    return {
        "template": template,
        "edges": edges,
        "logit_sizes": logit_sizes,
        "n_logits": int(sum(logit_sizes)),
        "init_logits": np.zeros(int(sum(logit_sizes)), dtype=float),  # uniform spacing
    }


def boundary_flat_from_logits(bnd, logits: Float[Array, "n_logits"]) -> Float[Array, "P 2"]:
    """Build the (P, 2) boundary array from per-edge logits (differentiable).

    Per edge with k sliders: softmax over (k+1) logits gives positive gaps that
    sum to 1; the cumulative sum (dropping the last) yields k strictly increasing
    positions in (0, 1), mapped onto the edge span. Strictly increasing ⟹ sliders
    never cross ⟹ the boundary stays convex. The flat outline therefore stays a
    square (sliders move only tangentially along each edge).
    """
    flat = jnp.array(bnd["template"])
    off = 0
    for e, sz in zip(bnd["edges"], bnd["logit_sizes"]):
        gaps = jax.nn.softmax(logits[off:off + sz])
        pos = e["lo"] + (e["hi"] - e["lo"]) * jnp.cumsum(gaps)[:-1]   # (k,) strictly increasing
        flat = flat.at[e["pbase"], e["axis"]].set(pos)
        flat = flat.at[e["pbase"] + 1, e["axis"]].set(pos)
        off += sz
    return flat


# ══════════════════════════════════════════════════════════════════════════════
# Differentiable forward map (JAX)
# ══════════════════════════════════════════════════════════════════════════════

def solve_cut_vertices_jax(
        struct,
        boundary_flat: Float[Array, "P 2"],
        r: Float[Array, "rows cols"]) -> Float[Array, "P 2"]:
    """Solve the collinearity LES for flat cut vertices — differentiable in r.

    Vectorized assembly from the precomputed sparsity pattern (``_les_assembly_indices``): a single
    scatter builds ``A = I + off-diagonals``, so the traced graph is O(1) scatter + one dense solve
    rather than the old O(P) unrolled ``.at[].set`` loop. Mirrors ``closed_builder._assemble_les``.
    """
    rows, cols, P = struct["rows"], struct["cols"], struct["P"]
    idx = struct.get("les_idx") or _les_assembly_indices(struct["T"])
    r_flat = jnp.broadcast_to(r, (rows, cols)).reshape(-1)

    coeff = jnp.where(jnp.asarray(idx["off_is_r"]),
                      r_flat[idx["off_ridx"]], 1.0 - r_flat[idx["off_ridx"]])
    A = jnp.eye(P).at[idx["off_rows"], idx["off_cols"]].add(-coeff)
    b = jnp.asarray(idx["boundary_mask"], dtype=boundary_flat.dtype)[:, None] * boundary_flat
    return jnp.linalg.solve(A, b)
