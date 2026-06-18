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


def build_deploy_structure(M: int, N: int):
    """Precompute the static (NumPy) data needed for the differentiable forward map.

    Returns the topology matrix ``T``, the cut-grid dimensions (``rows``, ``cols``)
    and flat cut-vertex count ``P``, and ``corner_pid`` — each panel corner's index
    into the flat cut-vertex array produced by ``solve_cut_vertices_jax``.
    """
    T = build_topology_matrix(M, N)
    rows, cols = T.shape
    P = 2 * rows * cols
    corner_pid = _panel_corner_sources(T)
    return {"T": T, "rows": rows, "cols": cols, "P": P, "corner_pid": corner_pid}


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

    Mirrors ``closed_builder._assemble_les`` with ``jnp`` so gradients flow
    through ``jnp.linalg.solve``.
    """
    T, rows, cols, P = struct["T"], struct["rows"], struct["cols"], struct["P"]
    r_arr = jnp.broadcast_to(r, (rows, cols))

    def pid(i, j, s):
        return 2 * (i * cols + j) + s

    A = jnp.zeros((P, P))
    b = jnp.zeros((P, 2))
    for i in range(rows):
        for j in range(cols):
            t = int(T[i, j])
            rij = r_arr[i, j]
            if t < 0:
                for s in (0, 1):
                    p = pid(i, j, s)
                    A = A.at[p, p].set(1.0)
                    b = b.at[p].set(boundary_flat[p])
            elif t == 1:
                p0 = pid(i, j, 0)
                A = A.at[p0, p0].set(1.0)
                A = A.at[p0, pid(i - 1, j, 0)].add(-rij)
                A = A.at[p0, pid(i + 1, j, 1)].add(-(1.0 - rij))
                p1 = pid(i, j, 1)
                A = A.at[p1, p1].set(1.0)
                A = A.at[p1, pid(i - 1, j, 0)].add(-(1.0 - rij))
                A = A.at[p1, pid(i + 1, j, 1)].add(-rij)
            else:  # t == 2
                p0 = pid(i, j, 0)
                A = A.at[p0, p0].set(1.0)
                A = A.at[p0, pid(i, j - 1, 1)].add(-rij)
                A = A.at[p0, pid(i, j + 1, 0)].add(-(1.0 - rij))
                p1 = pid(i, j, 1)
                A = A.at[p1, p1].set(1.0)
                A = A.at[p1, pid(i, j - 1, 1)].add(-(1.0 - rij))
                A = A.at[p1, pid(i, j + 1, 0)].add(-rij)

    return jnp.linalg.solve(A, b)
