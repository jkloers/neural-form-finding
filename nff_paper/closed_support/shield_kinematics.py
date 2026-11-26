"""Closed-form rigid-tile deployment of an RDPQK closed sheet.

The flat closed sheet built by the collinearity LES is a one-degree-of-freedom
mechanism: rigid quad tiles joined by perfect pin hinges. Its deployment has an
analytic solution rather than requiring a constrained solve.

Every tile rotates by ``theta_k = sigma_k * psi`` where ``sigma_k = +-1`` on a
checkerboard, and with the rotations fixed the pin equations become linear in the
tile translations, so those propagate along a spanning tree of the pin graph. The
redundant pins (there are roughly ``2 * n_pins - 3 * n_tiles`` more equations than
unknowns) are satisfied to machine precision only because the cut vertices lie on
the collinear family -- collinearity is exactly the mobility condition. Perturbing
off it drops the mechanism nullity from 4 to 2 and the sheet stops deploying.

Sign convention: the flat sheet is the compact, zero-void state, so ``psi < 0``
opens the voids. ``psi > 0`` drives the tiles into each other.
"""

import numpy as np
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from nff.topology.closed_builder_jax import boundary_points_flat


# ══════════════════════════════════════════════════════════════════════════════
# Static pin and tree topology (NumPy, precomputed once)
# ══════════════════════════════════════════════════════════════════════════════

def build_pin_pairs(struct) -> Int[np.ndarray, "n_pins 4"]:
    """Identify pin hinges as panel corners sharing a cut-endpoint index.

    Two panel corners are the same physical pivot iff they draw from the same
    entry of the flat cut-vertex array. This is a topological test, not a
    geometric one, which matters at boundary cuts where ``x`` and ``x'`` are
    coincident but belong to distinct pivots.

    Returns:
        (n_pins, 4) int array of ``[face1, corner1, face2, corner2]``.
    """
    corner_pid = np.asarray(struct["corner_pid"])
    n_tiles = corner_pid.shape[0]

    owners: dict[int, list[tuple[int, int]]] = {}
    for k in range(n_tiles):
        for c in range(4):
            owners.setdefault(int(corner_pid[k, c]), []).append((k, c))

    pins = []
    for pid in sorted(owners):
        group = owners[pid]
        if len(group) == 2:
            (f1, c1), (f2, c2) = group
            pins.append([f1, c1, f2, c2])
        elif len(group) > 2:
            raise ValueError(f"cut endpoint {pid} is shared by {len(group)} corners; expected <= 2")
    return np.array(pins, dtype=np.int64).reshape(-1, 4)


def _checkerboard_signs(M: int, N: int) -> Int[np.ndarray, "n_tiles"]:
    """Per-tile rotation sign, ``+1`` where ``(i + j)`` is even."""
    i = np.repeat(np.arange(M), N)
    j = np.tile(np.arange(N), M)
    return np.where((i + j) % 2 == 0, 1, -1).astype(np.int64)


def _spanning_tree(pin_pairs, n_tiles: int, root: int):
    """Breadth-first spanning tree over the pin graph, oriented away from root.

    Returns:
        tree_edges: (n_tiles-1, 4) int -- ``[parent, parent_corner, child, child_corner]``.
        order: (n_tiles,) int -- BFS visit order, root first.
    """
    adjacency: dict[int, list[tuple[int, int, int]]] = {k: [] for k in range(n_tiles)}
    for f1, c1, f2, c2 in pin_pairs:
        adjacency[int(f1)].append((int(f2), int(c1), int(c2)))
        adjacency[int(f2)].append((int(f1), int(c2), int(c1)))

    visited = np.zeros(n_tiles, dtype=bool)
    visited[root] = True
    queue, order, tree_edges = [root], [root], []
    while queue:
        parent = queue.pop(0)
        for child, parent_corner, child_corner in adjacency[parent]:
            if not visited[child]:
                visited[child] = True
                tree_edges.append([parent, parent_corner, child, child_corner])
                order.append(child)
                queue.append(child)

    if not visited.all():
        raise ValueError(f"pin graph is disconnected: {int((~visited).sum())} tiles unreachable")
    return np.array(tree_edges, dtype=np.int64), np.array(order, dtype=np.int64)


def _path_matrix(tree_edges, n_tiles: int, root: int) -> Float[np.ndarray, "n_tiles n_edges"]:
    """Accumulation matrix ``d = path_matrix @ delta`` over the rooted tree.

    Row ``k`` marks the tree edges on the root-to-``k`` path. Because every edge
    is oriented parent-to-child and ``delta`` is the child-minus-parent
    translation jump, all signs are ``+1``.
    """
    n_edges = tree_edges.shape[0]
    P = np.zeros((n_tiles, n_edges), dtype=float)
    edge_of_child = {int(e[2]): idx for idx, e in enumerate(tree_edges)}
    parent_of = {int(e[2]): int(e[0]) for e in tree_edges}

    for k in range(n_tiles):
        node = k
        while node != root:
            e = edge_of_child[node]
            P[k, e] = 1.0
            node = parent_of[node]
    return P


def _outward_edges(struct, M: int, N: int, spacing: float, spacing_y: float):
    """Tile edges lying on the sheet outline, ordered around the perimeter.

    A tile edge is on the outline iff both endpoints are boundary cut vertices on
    the same side line of the rectangle. Boundary sliders move only tangentially,
    so this classification is invariant to the design parameters.

    Returns:
        (2*(M+N), 3) int array of ``[tile, corner_a, corner_b]`` in perimeter order.
    """
    corner_pid = np.asarray(struct["corner_pid"])
    cols = struct["cols"]
    T = struct["T"]
    W, H = M * spacing, N * spacing_y

    # Side membership of every cut-vertex index, from the (invariant) grid position.
    side_of = {}
    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            if T[i, j] >= 0:
                continue
            x, y = i * spacing, j * spacing_y
            sides = []
            if abs(x) < 1e-9:
                sides.append("x0")
            if abs(x - W) < 1e-9:
                sides.append("x1")
            if abs(y) < 1e-9:
                sides.append("y0")
            if abs(y - H) < 1e-9:
                sides.append("y1")
            for s in (0, 1):
                side_of[2 * (i * cols + j) + s] = set(sides)

    flat_template = np.asarray(struct["boundary_template"])
    edges = []
    for k in range(corner_pid.shape[0]):
        for c in range(4):
            a, b = int(corner_pid[k, c]), int(corner_pid[k, (c + 1) % 4])
            shared = side_of.get(a, set()) & side_of.get(b, set())
            if shared:
                mid = 0.5 * (flat_template[a] + flat_template[b])
                edges.append((k, c, (c + 1) % 4, _perimeter_param(mid, W, H)))

    edges.sort(key=lambda e: e[3])
    return np.array([[e[0], e[1], e[2]] for e in edges], dtype=np.int64)


def _perimeter_param(p, W: float, H: float) -> float:
    """Arclength position of a boundary point, counter-clockwise from the origin."""
    x, y = float(p[0]), float(p[1])
    if abs(y) < 1e-7:
        return x
    if abs(x - W) < 1e-7:
        return W + y
    if abs(y - H) < 1e-7:
        return W + H + (W - x)
    return 2 * W + H + (H - y)


def build_deploy_plan(struct, M: int, N: int,
                      spacing: float = 1.0, spacing_y: float | None = None) -> dict:
    """Precompute every static array the closed-form deployment needs.

    Args:
        struct: Output of ``build_deploy_structure(M, N)``.
        M: Panel count along x.
        N: Panel count along y.
        spacing: Panel pitch along x.
        spacing_y: Panel pitch along y; defaults to ``spacing``.

    Returns:
        Dict with ``sigma``, ``pin_pairs``, ``tree_edges``, ``path_matrix``,
        ``outward_edges``, ``root``, and the sheet dimensions.
    """
    sy = spacing if spacing_y is None else spacing_y
    n_tiles = M * N
    struct = {**struct, "boundary_template": boundary_points_flat(struct, spacing, sy)}

    pin_pairs = build_pin_pairs(struct)
    root = (M // 2) * N + (N // 2)
    tree_edges, order = _spanning_tree(pin_pairs, n_tiles, root)

    return {
        "M": M, "N": N, "n_tiles": n_tiles,
        "sheet_w": M * spacing, "sheet_h": N * sy,
        "sigma": _checkerboard_signs(M, N),
        "pin_pairs": pin_pairs,
        "tree_edges": tree_edges,
        "path_matrix": _path_matrix(tree_edges, n_tiles, root),
        "outward_edges": _outward_edges(struct, M, N, spacing, sy),
        "root": root,
        "bfs_order": order,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Closed-form deployment (JAX, differentiable in psi and in the flat geometry)
# ══════════════════════════════════════════════════════════════════════════════

def deploy_tiles(plan, flat_tiles: Float[Array, "n_tiles 4 2"],
                 psi: float) -> Float[Array, "n_tiles 4 2"]:
    """Place every rigid tile at opening angle ``psi``.

    Rotations are the checkerboard field ``sigma_k * psi``; translations are
    accumulated along the spanning tree so that every tree pin closes exactly.
    The redundant pins then close to machine precision as a consequence of the
    collinearity family -- check with ``pin_residual``.
    """
    sigma = jnp.asarray(plan["sigma"], dtype=flat_tiles.dtype)
    theta = sigma * psi
    cos_t, sin_t = jnp.cos(theta), jnp.sin(theta)

    centroids = flat_tiles.mean(axis=1)                       # (n_tiles, 2)
    local = flat_tiles - centroids[:, None, :]
    rotated = jnp.stack([
        cos_t[:, None] * local[..., 0] - sin_t[:, None] * local[..., 1],
        sin_t[:, None] * local[..., 0] + cos_t[:, None] * local[..., 1],
    ], axis=-1)
    placed = rotated + centroids[:, None, :]                  # before translation

    edges = plan["tree_edges"]
    parent, parent_corner = edges[:, 0], edges[:, 1]
    child, child_corner = edges[:, 2], edges[:, 3]
    delta = placed[parent, parent_corner] - placed[child, child_corner]

    translation = jnp.asarray(plan["path_matrix"], dtype=flat_tiles.dtype) @ delta
    return placed + translation[:, None, :]


def pin_residual(plan, deployed: Float[Array, "n_tiles 4 2"]) -> Float[Array, "n_pins 2"]:
    """Gap at every pin, including the redundant ones. Diagnostic only."""
    pins = plan["pin_pairs"]
    return deployed[pins[:, 0], pins[:, 1]] - deployed[pins[:, 2], pins[:, 3]]


def deployed_outline(plan, deployed: Float[Array, "n_tiles 4 2"],
                     n_sub: int = 6) -> Float[Array, "n_out 2"]:
    """Dense, ordered outer boundary of the deployed sheet.

    Each outward tile edge is subdivided ``n_sub`` times. The result traverses the
    perimeter in order, so the shoelace area and turning number are meaningful.
    """
    edges = plan["outward_edges"]
    a = deployed[edges[:, 0], edges[:, 1]]                    # (n_edges, 2)
    b = deployed[edges[:, 0], edges[:, 2]]
    t = jnp.linspace(0.0, 1.0, n_sub, endpoint=False)[None, :, None]
    pts = a[:, None, :] * (1.0 - t) + b[:, None, :] * t
    return pts.reshape(-1, 2)


def polygon_area(points: Float[Array, "n 2"]) -> Float[Array, ""]:
    """Signed shoelace area of an ordered closed polygon."""
    x, y = points[:, 0], points[:, 1]
    return 0.5 * jnp.sum(x * jnp.roll(y, -1) - jnp.roll(x, -1) * y)


def tile_areas(tiles: Float[Array, "n_tiles 4 2"]) -> Float[Array, "n_tiles"]:
    """Signed shoelace area of every quad tile."""
    x, y = tiles[..., 0], tiles[..., 1]
    return 0.5 * jnp.sum(x * jnp.roll(y, -1, axis=-1) - jnp.roll(x, -1, axis=-1) * y, axis=-1)
