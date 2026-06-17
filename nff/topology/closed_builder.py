"""Closed-state (flat-sheet) kirigami builder.

Constructs the *undeployed* configuration of an interwoven-cut kirigami from a
topology matrix, fixed boundary points, and per-cut void aspect ratios, by
solving the collinearity linear system (LES). It emits a ``Tessellation`` in the
SAME representation produced by the deployed-state builder
(``nff.topology.builder.build_tessellation``), so ``CentroidalState`` and every
downstream stage are reused unchanged.

This is the reverse of the historical workflow: instead of starting from a
deployed tessellation and closing it, we start from a flat sheet, introduce
cuts, and (later) deploy it.

Reference
---------
Dang, Feng, Duan, Wang, "Theorem for the design of deployable kirigami
tessellations with different topologies" (2021), arXiv:2106.15891. Section II
(genus-0) gives the topology matrix Eq. (2) and the collinearity LES Eqs. (3-4).

Index convention
----------------
A tessellation of ``M × N`` panels ``P_{i,j}`` is bounded by cuts on an
``(M+1) × (N+1)`` grid, indexed ``(i, j)``. Cut ``C_{i,j}`` sits at the shared
corner of panels ``P_{i-1,j-1}, P_{i,j-1}, P_{i-1,j}, P_{i,j}`` and owns two
endpoint vertices ``x_ij`` and ``x'_ij`` (coincident only in the degenerate
``r = 0.5`` case). The topology matrix ``T`` (shape ``(M+1, N+1)``) encodes,
per cut:

    |T| == 1   horizontal cut, collinear with x_{i-1,j} and x'_{i+1,j}
    |T| == 2   vertical cut,   collinear with x'_{i,j-1} and x_{i,j+1}
    T  <  0    boundary cut (pinned: x == x' == x_bound)
    T  >  0    interior cut (position solved from the LES)

The collinearity equations (paper construction), each a 2-vector equation:

    horizontal (t = 1):
        x_ij  = r·x_{i-1,j} + (1-r)·x'_{i+1,j}
        x'_ij = (1-r)·x_{i-1,j} + r·x'_{i+1,j}
    vertical (t = 2):
        x_ij  = r·x'_{i,j-1} + (1-r)·x_{i,j+1}
        x'_ij = (1-r)·x'_{i,j-1} + r·x_{i,j+1}

NumPy only. Matrix assembly is kept separate from the solve so the solve can
later be ported to ``jnp.linalg.solve`` for a differentiable Stage-0 (with the
aspect ratios and boundary points as design variables).
"""

import numpy as np
from jaxtyping import Float, Int

from nff.topology.core import Tessellation


# ══════════════════════════════════════════════════════════════════════════════
# Topology matrix and boundary points
# ══════════════════════════════════════════════════════════════════════════════

def build_topology_matrix(M: int, N: int) -> Int[np.ndarray, "M+1 N+1"]:
    """Build the staggered topology matrix for an M×N panel tessellation.

    Implements Dang et al. Eq. (2): the cut grid is ``(M+1) × (N+1)``. Cut type
    alternates in a checkerboard — horizontal (1) where ``(i+j)`` is even,
    vertical (2) otherwise. Cuts on the grid border are boundary cuts (negative
    sign); interior cuts are positive. Verified against the paper's
    ``T⁰_{4×4}`` (Eq. 1).

    Args:
        M: Number of panel columns (``M >= 3``).
        N: Number of panel rows (``N >= 3``).

    Returns:
        (M+1, N+1) int array of signed cut codes in {-2, -1, 1, 2}.
    """
    rows, cols = M + 1, N + 1
    T = np.zeros((rows, cols), dtype=np.int32)
    for i in range(rows):
        for j in range(cols):
            ctype = 1 if (i + j) % 2 == 0 else 2
            on_border = (i == 0 or i == rows - 1 or j == 0 or j == cols - 1)
            T[i, j] = -ctype if on_border else ctype
    return T


def build_square_boundary_points(
        T: Int[np.ndarray, "M N"],
        spacing: float = 1.0) -> dict:
    """Place every boundary cut at its grid position, forming a square sheet.

    Boundary cuts (``T < 0``) are pinned to these coordinates; interior cuts are
    solved from the LES. Useful as the canonical validation case: a uniform
    square outline.

    Args:
        T: Topology matrix from ``build_topology_matrix``.
        spacing: Physical grid spacing.

    Returns:
        Mapping ``(i, j) -> (2,) float`` for every boundary cut.
    """
    M, N = T.shape
    boundary_points = {}
    for i in range(M):
        for j in range(N):
            if T[i, j] < 0:
                boundary_points[(i, j)] = np.array([i * spacing, j * spacing], dtype=float)
    return boundary_points


# ══════════════════════════════════════════════════════════════════════════════
# Linear equation system for the closed-state cut vertices
# ══════════════════════════════════════════════════════════════════════════════

def _assemble_les(
        T: Int[np.ndarray, "M N"],
        boundary_points: dict,
        r: Float[np.ndarray, "M N"]) -> tuple:
    """Assemble the dense LES ``A @ coords = b`` for all cut endpoint vertices.

    Each cut contributes exactly two point-rows (for ``x`` and ``x'``), so the
    system is square: ``(2MN, 2MN)``. The x and y components are decoupled and
    share ``A``, so ``b`` carries both as two columns.

    Returns:
        (A, b): A is (2MN, 2MN); b is (2MN, 2).
    """
    M, N = T.shape
    P = 2 * M * N
    A = np.zeros((P, P), dtype=float)
    b = np.zeros((P, 2), dtype=float)
    r_arr = np.broadcast_to(np.asarray(r, dtype=float), (M, N))

    def pid(i, j, s):
        """Row/column index of point s in {0:x, 1:x'} of cut (i, j)."""
        return 2 * (i * N + j) + s

    for i in range(M):
        for j in range(N):
            t = int(T[i, j])
            rij = float(r_arr[i, j])

            if t < 0:                       # boundary cut: x == x' == x_bound
                bp = np.asarray(boundary_points[(i, j)], dtype=float)
                for s in (0, 1):
                    p = pid(i, j, s)
                    A[p, p] = 1.0
                    b[p] = bp

            elif t == 1:                    # horizontal: links (i-1,j).x, (i+1,j).x'
                p0 = pid(i, j, 0)
                A[p0, p0] = 1.0
                A[p0, pid(i - 1, j, 0)] -= rij
                A[p0, pid(i + 1, j, 1)] -= (1.0 - rij)

                p1 = pid(i, j, 1)
                A[p1, p1] = 1.0
                A[p1, pid(i - 1, j, 0)] -= (1.0 - rij)
                A[p1, pid(i + 1, j, 1)] -= rij

            elif t == 2:                    # vertical: links (i,j-1).x', (i,j+1).x
                p0 = pid(i, j, 0)
                A[p0, p0] = 1.0
                A[p0, pid(i, j - 1, 1)] -= rij
                A[p0, pid(i, j + 1, 0)] -= (1.0 - rij)

                p1 = pid(i, j, 1)
                A[p1, p1] = 1.0
                A[p1, pid(i, j - 1, 1)] -= (1.0 - rij)
                A[p1, pid(i, j + 1, 0)] -= rij

            else:
                raise ValueError(f"Unexpected topology code {t} at cut ({i}, {j}).")

    return A, b


def solve_cut_vertices(
        T: Int[np.ndarray, "M N"],
        boundary_points: dict,
        r) -> dict:
    """Solve the collinearity LES for every cut's two endpoint vertices.

    Args:
        T: Topology matrix.
        boundary_points: ``(i, j) -> (2,)`` for boundary cuts.
        r: Per-cut aspect ratio, scalar or ``(M, N)`` array.

    Returns:
        Mapping ``(i, j) -> (x, x_prime)``, each a ``(2,)`` float array.
    """
    M, N = T.shape
    A, b = _assemble_les(T, boundary_points, r)
    coords = np.linalg.solve(A, b)          # (2MN, 2)

    cut_vertices = {}
    for i in range(M):
        for j in range(N):
            base = 2 * (i * N + j)
            cut_vertices[(i, j)] = (coords[base], coords[base + 1])
    return cut_vertices


# ══════════════════════════════════════════════════════════════════════════════
# Panel and hinge assembly
# ══════════════════════════════════════════════════════════════════════════════

def _assemble_panels(T: Int[np.ndarray, "M+1 N+1"], cut_vertices: dict) -> tuple:
    """Assemble quad panels from the solved cut vertices.

    Panel ``P_{i,j}`` (``i = 0..M-1``, ``j = 0..N-1``) has its four corners at
    the surrounding cuts ``C_{i,j}`` (bottom-left), ``C_{i+1,j}`` (bottom-right),
    ``C_{i+1,j+1}`` (top-right), ``C_{i,j+1}`` (top-left). Each corner takes one
    of the cut's two endpoints (``x`` or ``x'``). The choice is fixed by the
    shared-hinge pivots in the deployment relations Eqs. (7-8) of Dang et al.,
    and depends only on the panel's own cut type via ``h = (|t_{i,j}| == 1)``:

        bottom-left  C_{i,j}     -> x' if h else x
        bottom-right C_{i+1,j}   -> x'              (always)
        top-right    C_{i+1,j+1} -> x  if h else x'
        top-left     C_{i,j+1}   -> x               (always)

    Returns:
        (vertices, faces): vertices is (V, 2); faces is a list of length-4
        index lists in CCW order. Coincident corners across panels are NOT
        merged — that is what lets the hinge step recover pivots as
        coincident-but-distinct vertex pairs.
    """
    rows, cols = T.shape
    M, N = rows - 1, cols - 1
    vertices = []
    faces = []

    def endpoint(cut_ij, which):
        """which: 0 -> x, 1 -> x'."""
        return cut_vertices[cut_ij][which]

    for i in range(M):
        for j in range(N):
            h = abs(int(T[i, j])) == 1
            corner_pts = [
                endpoint((i, j),         1 if h else 0),  # BL
                endpoint((i + 1, j),     1),              # BR: x'
                endpoint((i + 1, j + 1), 0 if h else 1),  # TR
                endpoint((i, j + 1),     0),              # TL: x
            ]
            base = len(vertices)
            vertices.extend(corner_pts)
            faces.append([base, base + 1, base + 2, base + 3])

    return np.array(vertices, dtype=float), faces


def _build_hinges(vertices: Float[np.ndarray, "V 2"], faces: list, tol: float = 1e-6) -> list:
    """Detect corner hinges as coincident vertex pairs between distinct faces.

    Two panels form a hinge where one vertex of each coincides (a pivot). The
    hinge's adjacent vertices are the next vertices CCW within each face; the
    cross-product orientation check mirrors ``builder.py`` so the CCW
    convention is consistent across the codebase.

    Returns:
        List of dicts with keys face1, face2, vertex1, vertex2,
        vertex_adjacent1, vertex_adjacent2.
    """
    # Local-node lookup: global vertex -> (face_id, local_id).
    v_to_fn = {}
    for f_id, f in enumerate(faces):
        for local, gv in enumerate(f):
            v_to_fn[gv] = (f_id, local)

    hinges = []
    seen = set()
    n_faces = len(faces)
    for f1 in range(n_faces):
        for f2 in range(f1 + 1, n_faces):
            for v1 in faces[f1]:
                for v2 in faces[f2]:
                    if np.sum((vertices[v1] - vertices[v2]) ** 2) >= tol ** 2:
                        continue
                    key = (f1, f2)
                    if key in seen:
                        continue                # one pivot per face pair
                    seen.add(key)

                    l1 = v_to_fn[v1][1]
                    l2 = v_to_fn[v2][1]
                    adj1 = faces[f1][(l1 + 1) % 4]
                    adj2 = faces[f2][(l2 + 1) % 4]

                    face1, face2 = f1, f2
                    vertex1, vertex2 = v1, v2
                    vertex_adjacent1, vertex_adjacent2 = adj1, adj2

                    cross = np.cross(
                        vertices[vertex1] - vertices[vertex_adjacent1],
                        vertices[vertex2] - vertices[vertex_adjacent2],
                    )
                    if cross < 0:               # enforce CCW, as in builder.py
                        face1, face2 = face2, face1
                        vertex1, vertex2 = vertex2, vertex1
                        vertex_adjacent1, vertex_adjacent2 = vertex_adjacent2, vertex_adjacent1

                    hinges.append({
                        'face1': face1, 'face2': face2,
                        'vertex1': vertex1, 'vertex2': vertex2,
                        'vertex_adjacent1': vertex_adjacent1,
                        'vertex_adjacent2': vertex_adjacent2,
                    })
    return hinges


def _detect_voids(tessellation: Tessellation) -> None:
    """Detect voids (opposite hinge pairs) and register them on the tessellation.

    Replicates the generic void-discovery walk used by
    ``nff.topology.builder.build_tessellation`` so the closed-state builder is
    self-contained.
    """
    primary_to_hinges = tessellation.build_primary_to_hinges()
    adjacents_to_hinge = tessellation.build_adjacents_to_hinge()
    unvisited = set(range(len(tessellation.hinges)))
    discovered = set()

    def walk(h_id):
        if h_id not in unvisited:
            return
        unvisited.remove(h_id)
        h0 = tessellation.hinges[h_id]
        a1, a2 = h0.vertex_adjacent1, h0.vertex_adjacent2

        for hs1_id in primary_to_hinges.get(a1, []):
            if hs1_id == h_id:
                continue
            hs1 = tessellation.hinges[hs1_id]
            p1 = hs1.vertex2 if hs1.vertex1 == a1 else hs1.vertex1
            for hs2_id in primary_to_hinges.get(a2, []):
                if hs2_id == h_id:
                    continue
                hs2 = tessellation.hinges[hs2_id]
                p2 = hs2.vertex2 if hs2.vertex1 == a2 else hs2.vertex1
                target_pair = frozenset([p1, p2])
                if target_pair in adjacents_to_hinge:
                    h_opp_id = adjacents_to_hinge[target_pair]
                    if h_opp_id != h_id:
                        sig = tuple(sorted([h_id, h_opp_id]))
                        if sig not in discovered:
                            discovered.add(sig)
                            tessellation.add_void(h_id, h_opp_id)
                        break

        for v in [h0.vertex1, h0.vertex2, h0.vertex_adjacent1, h0.vertex_adjacent2]:
            for next_h_id in primary_to_hinges.get(v, []):
                walk(next_h_id)

    while unvisited:
        walk(next(iter(unvisited)))


# ══════════════════════════════════════════════════════════════════════════════
# Public builder
# ══════════════════════════════════════════════════════════════════════════════

def build_closed_tessellation(
        M: int,
        N: int,
        boundary_points: dict | None = None,
        r=0.4,
        spacing: float = 1.0,
        k_stretch: float = 1.0,
        k_shear: float = 1.0,
        k_rot: float = 1.0) -> Tessellation:
    """Build the flat closed-state kirigami tessellation.

    Pipeline: topology matrix → LES solve for cut vertices → quad panels →
    corner hinges → void detection. The result is a ``Tessellation`` ready for
    ``CentroidalState.from_tessellation``.

    Args:
        M, N: Number of panel columns/rows (``M, N >= 3``).
        boundary_points: ``(i, j) -> (2,)`` for boundary cuts. Defaults to a
            uniform square outline at grid positions.
        r: Per-cut aspect ratio, scalar or ``(M+1, N+1)`` array.
        spacing: Grid spacing for the default square boundary.
        k_stretch, k_shear, k_rot: Default hinge stiffnesses.

    Returns:
        Configured ``Tessellation``.
    """
    T = build_topology_matrix(M, N)
    if boundary_points is None:
        boundary_points = build_square_boundary_points(T, spacing=spacing)

    cut_vertices = solve_cut_vertices(T, boundary_points, r)
    vertices, faces = _assemble_panels(T, cut_vertices)
    hinges = _build_hinges(vertices, faces)

    tessellation = Tessellation(vertices=vertices, faces=faces, dim=2)
    for h in hinges:
        tessellation.add_hinge(**h)
    _detect_voids(tessellation)
    tessellation.set_hinge_properties(k_stretch=k_stretch, k_shear=k_shear, k_rot=k_rot)

    return tessellation
