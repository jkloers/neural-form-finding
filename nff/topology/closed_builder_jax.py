"""Differentiable JAX forward map for closed-state RDPQK kirigami.

Implements the Dang et al. (2021) inverse-design forward map as a differentiable
function of the per-cut aspect ratios ``r`` (and, optionally, boundary points):

    r -> LES (flat cut vertices) -> rigid kinematic deployment at angle omega
      -> deployed panel vertices.

The static topology (which cut endpoint each panel corner takes, the hinge graph,
the panel rotation classes) is precomputed once with NumPy in
``build_deploy_structure``; the differentiable part (``forward_deploy``) is pure
JAX and JIT/grad-friendly.

Deployment (Eqs. 5-9): each panel's absolute rotation is the identity if its cut
type ``|t_{i,j}| == 1`` else ``R_omega``; the per-panel translations are fixed by
requiring shared hinge vertices to coincide, propagated over a spanning tree of
the panel adjacency graph. Genus-0 patterns built from the LES have parallelogram
voids (Lemma 1), so the propagation is path-independent.
"""

from collections import deque

import numpy as np
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from nff.topology.closed_builder import build_topology_matrix, build_square_boundary_points


# ══════════════════════════════════════════════════════════════════════════════
# Static structure (NumPy, precomputed once)
# ══════════════════════════════════════════════════════════════════════════════

def _panel_corner_sources(T):
    """For each panel corner, the source cut point index into the flat (P,2) array.

    Uses the validated panel->vertex rule (depends on the panel's own cut type
    ``h = |t_{i,j}| == 1``). Point index of endpoint ``s`` of cut ``(ci, cj)`` is
    ``2 * (ci * cols + cj) + s`` — matching the LES ordering.

    Returns:
        corner_pid: (n_panels, 4) int — point index per panel corner (BL,BR,TR,TL).
        panel_ij:   (n_panels, 2) int — (i, j) of each panel.
    """
    rows, cols = T.shape
    M, N = rows - 1, cols - 1

    def pid(ci, cj, s):
        return 2 * (ci * cols + cj) + s

    corner_pid = []
    panel_ij = []
    for i in range(M):
        for j in range(N):
            h = abs(int(T[i, j])) == 1
            corner_pid.append([
                pid(i, j, 1 if h else 0),          # BL
                pid(i + 1, j, 1),                  # BR
                pid(i + 1, j + 1, 0 if h else 1),  # TR
                pid(i, j + 1, 0),                  # TL
            ])
            panel_ij.append([i, j])
    return np.array(corner_pid, dtype=np.int64), np.array(panel_ij, dtype=np.int64)


def _boundary_corner_mask(T):
    """(n_panels, 4) bool — True where the corner's source cut is a boundary cut."""
    rows, cols = T.shape
    M, N = rows - 1, cols - 1
    mask = []
    for i in range(M):
        for j in range(N):
            cuts = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]
            mask.append([T[ci, cj] < 0 for (ci, cj) in cuts])
    return np.array(mask, dtype=bool)


def _bfs_panel_tree(corner_pid):
    """Spanning tree over panels adjacent through a shared cut point (hinge).

    Two panels are hinged where they own a corner with the same point index.

    Returns:
        order: list of panels in BFS order (root first).
        parent_edge: dict panel -> (parent_panel, shared_point_index); root maps to None.
    """
    n_panels = corner_pid.shape[0]
    point_to_panels = {}
    for p in range(n_panels):
        for c in range(4):
            point_to_panels.setdefault(int(corner_pid[p, c]), []).append(p)

    adj = {p: set() for p in range(n_panels)}
    shared_pid = {}
    for pid_val, panels in point_to_panels.items():
        for a in range(len(panels)):
            for b in range(a + 1, len(panels)):
                pa, pb = panels[a], panels[b]
                if pa == pb:
                    continue
                adj[pa].add(pb)
                adj[pb].add(pa)
                shared_pid[(pa, pb)] = pid_val
                shared_pid[(pb, pa)] = pid_val

    order = []
    parent_edge = {0: None}
    seen = {0}
    q = deque([0])
    while q:
        p = q.popleft()
        order.append(p)
        for nb in sorted(adj[p]):
            if nb not in seen:
                seen.add(nb)
                parent_edge[nb] = (p, shared_pid[(p, nb)])
                q.append(nb)
    if len(order) != n_panels:
        raise ValueError(f"panel adjacency graph not connected ({len(order)}/{n_panels})")
    return order, parent_edge


def build_deploy_structure(M: int, N: int):
    """Precompute all static (NumPy) data needed for the differentiable forward map."""
    T = build_topology_matrix(M, N)
    rows, cols = T.shape
    P = 2 * rows * cols

    corner_pid, panel_ij = _panel_corner_sources(T)
    bnd_mask = _boundary_corner_mask(T)
    order, parent_edge = _bfs_panel_tree(corner_pid)

    # Panel rotation class: True (rotates by omega) if |t_{i,j}| == 2.
    panel_rotates = np.array([abs(int(T[i, j])) == 2 for (i, j) in panel_ij], dtype=bool)

    return {
        "T": T, "rows": rows, "cols": cols, "P": P,
        "corner_pid": corner_pid, "panel_ij": panel_ij,
        "bnd_mask": bnd_mask, "order": order, "parent_edge": parent_edge,
        "panel_rotates": panel_rotates,
    }


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


def build_boundary_sliders(struct, spacing: float = 1.0):
    """Parameterize non-corner boundary cuts as sliders along the square outline.

    Corners stay fixed; every other boundary cut slides tangentially along its
    edge (its perpendicular coordinate is held, preserving the square shape of
    the flat sheet — Dang et al. Sec. V).

    Returns a dict with:
        template:   (P, 2) base positions (corners + initial slider positions).
        point_base: (n_sliders,) int  — first point index of each slider cut.
        free_axis:  (n_sliders,) int  — 0 (x free) or 1 (y free).
        init:       (n_sliders,) float — initial free coordinate (grid position).
        lo, hi:     (n_sliders,) float — slider range along the edge.
    """
    T, rows, cols = struct["T"], struct["rows"], struct["cols"]
    template = boundary_points_flat(struct, spacing=spacing)

    x_max = (rows - 1) * spacing      # bottom/top edges span x in [0, x_max]
    y_max = (cols - 1) * spacing      # left/right edges span y in [0, y_max]

    point_base, free_axis, init, lo, hi = [], [], [], [], []
    for i in range(rows):
        for j in range(cols):
            if T[i, j] >= 0:
                continue                                # interior cut
            on_v_edge = (i == 0 or i == rows - 1)       # left/right
            on_h_edge = (j == 0 or j == cols - 1)       # bottom/top
            if on_v_edge and on_h_edge:
                continue                                # corner: fixed
            base = 2 * (i * cols + j)
            if on_v_edge:                               # slide in y
                point_base.append(base); free_axis.append(1)
                init.append(j * spacing); lo.append(0.0); hi.append(y_max)
            else:                                       # slide in x
                point_base.append(base); free_axis.append(0)
                init.append(i * spacing); lo.append(0.0); hi.append(x_max)

    return {
        "template": template,
        "point_base": np.array(point_base, dtype=np.int64),
        "free_axis": np.array(free_axis, dtype=np.int64),
        "init": np.array(init, dtype=float),
        "lo": np.array(lo, dtype=float),
        "hi": np.array(hi, dtype=float),
    }


def boundary_flat_from_sliders(sliders, bnd_free: Float[Array, "n_sliders"]) -> Float[Array, "P 2"]:
    """Build the (P, 2) boundary array from slider free-coordinates (differentiable)."""
    flat = jnp.array(sliders["template"])
    pb = sliders["point_base"]
    ax = sliders["free_axis"]
    flat = flat.at[pb, ax].set(bnd_free)
    flat = flat.at[pb + 1, ax].set(bnd_free)
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
    never cross ⟹ the boundary stays convex.
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


def forward_deploy(
        struct,
        boundary_flat: Float[Array, "P 2"],
        r: Float[Array, "rows cols"],
        omega: float) -> Float[Array, "n_panels 4 2"]:
    """Full differentiable map: r -> flat LES -> rigid deployment at omega.

    Returns deployed panel vertices, shape (n_panels, 4, 2).
    """
    coords = solve_cut_vertices_jax(struct, boundary_flat, r)        # (P, 2) flat
    corner_pid = struct["corner_pid"]
    panel_flat = coords[corner_pid]                                  # (n_panels, 4, 2)

    c, s = jnp.cos(omega), jnp.sin(omega)
    R_rot = jnp.array([[c, -s], [s, c]])
    eye = jnp.eye(2)
    panel_rotates = struct["panel_rotates"]
    n_panels = panel_flat.shape[0]
    R = [R_rot if panel_rotates[p] else eye for p in range(n_panels)]

    # Propagate translations over the spanning tree: t_child = t_parent +
    # (R_parent - R_child) @ p_shared.
    t = [None] * n_panels
    t[struct["order"][0]] = jnp.zeros(2)
    for p in struct["order"][1:]:
        parent, shared_pid = struct["parent_edge"][p]
        p_shared = coords[shared_pid]
        t[p] = t[parent] + (R[parent] - R[p]) @ p_shared

    deployed = jnp.stack([
        (R[p] @ panel_flat[p].T).T + t[p] for p in range(n_panels)
    ])
    return deployed
