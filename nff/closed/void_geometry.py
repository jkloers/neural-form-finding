"""Void (hole) polygons of a closed RDPQK sheet, and their departure from a parallelogram.

The generic void detector finds nothing here. ``Tessellation``'s notion of a void is a pair of
OPPOSITE hinges bounding it, which matches an open RDQK cell; in the closed sheet every corner of
every void is a pin, so the walk in ``closed_builder._detect_voids`` never matches,
``void_opposite_node_pairs`` comes out ``(0, 2, 3)``, and every void term in ``nff/stages/`` is
inert on this pattern. Measured: 0 voids at both ``r = 0.45`` and ``r = 0.5`` on a 15x10 sheet.

The voids are nonetheless fixed by ``corner_pid``. Each interior lattice site ``(i, j)`` carries
one void whose four corners are the site's own two cut endpoints plus the two endpoints the
collinearity system links it to -- the same two points the LES uses, which is why the void closes
flat. In the flat sheet those four points are distinct but COLLINEAR, so the void is a fully
collapsed parallelogram (a slit); deployment opens the angle without changing the four side
lengths, because the tiles are rigid.

★ In the rigid 1-DOF deployment every void is an EXACT parallelogram (Lemma 1 of the LES
construction; measured to 5e-15). So the parallelogram defect of a compliant deployment measures
precisely the departure from rigid-tile kinematics, the same departure ``nff/closed/one_dof.py``
sees in the angle field. In a compliant state the pins also separate, so the hole is really an
octagon; the quad is recovered by collapsing each pin onto the midpoint of its two owners, and the
separation is reported alongside as the slip.
"""

from dataclasses import dataclass

import numpy as np
from jaxtyping import Float, Int


@dataclass
class VoidIndex:
    """Topological addressing of every interior void. Built once, reused at every load step.

    ``owner_face``/``owner_node`` give, for each void and each of its four pins, the two
    ``(face, local node)`` pairs that meet there. A pin on a boundary cut can have a single owner;
    its second slot repeats the first, so the midpoint is still correct and its gap is zero.
    """
    sites: Int[np.ndarray, "n_voids 2"]
    pids: Int[np.ndarray, "n_voids 4"]
    owner_face: Int[np.ndarray, "n_voids 4 2"]
    owner_node: Int[np.ndarray, "n_voids 4 2"]

    @property
    def n_voids(self) -> int:
        return int(self.sites.shape[0])


def build_void_index(struct) -> VoidIndex:
    """Enumerate the ``(M-1) * (N-1)`` interior voids from the cut-lattice topology.

    Args:
        struct: ``closed_builder_jax.build_deploy_structure`` output, or any mapping carrying
            ``T`` (rows, cols), ``rows``, ``cols`` and ``corner_pid`` (n_faces, 4).

    Returns:
        ``VoidIndex``.
    """
    T = np.asarray(struct["T"])
    rows, cols = int(struct["rows"]), int(struct["cols"])
    corner_pid = np.asarray(struct["corner_pid"])

    owners: dict[int, list[tuple[int, int]]] = {}
    for k in range(corner_pid.shape[0]):
        for c in range(corner_pid.shape[1]):
            owners.setdefault(int(corner_pid[k, c]), []).append((k, c))

    def pid(i: int, j: int, s: int) -> int:
        return 2 * (i * cols + j) + s

    sites, pids, of, on = [], [], [], []
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if abs(int(T[i, j])) == 1:                       # horizontal cut
                a, c_ = pid(i - 1, j, 0), pid(i + 1, j, 1)
            else:                                            # vertical cut
                a, c_ = pid(i, j - 1, 1), pid(i, j + 1, 0)
            quad = [a, pid(i, j, 0), c_, pid(i, j, 1)]       # CCW: A, x, C, x'
            faces, nodes = [], []
            for p in quad:
                own = owners.get(p, [])
                if not own:
                    raise ValueError(f"cut endpoint {p} at site ({i},{j}) has no owning panel")
                if len(own) > 2:
                    raise ValueError(f"cut endpoint {p} has {len(own)} owners, expected 1 or 2")
                pair = own if len(own) == 2 else [own[0], own[0]]
                faces.append([pair[0][0], pair[1][0]])
                nodes.append([pair[0][1], pair[1][1]])
            sites.append([i, j])
            pids.append(quad)
            of.append(faces)
            on.append(nodes)

    return VoidIndex(sites=np.asarray(sites, dtype=np.int64),
                     pids=np.asarray(pids, dtype=np.int64),
                     owner_face=np.asarray(of, dtype=np.int64),
                     owner_node=np.asarray(on, dtype=np.int64))


# ══════════════════════════════════════════════════════════════════════════════
# Polygons from a deployed node field
# ══════════════════════════════════════════════════════════════════════════════

def void_quads(index: VoidIndex,
               nodes: Float[np.ndarray, "n_faces n_nodes 2"]) -> Float[np.ndarray, "n_voids 4 2"]:
    """The four pin positions of each void, CCW, each taken as the midpoint of its two owners."""
    nd = np.asarray(nodes, dtype=float)
    p0 = nd[index.owner_face[:, :, 0], index.owner_node[:, :, 0]]
    p1 = nd[index.owner_face[:, :, 1], index.owner_node[:, :, 1]]
    return 0.5 * (p0 + p1)


def void_sides(quads: Float[np.ndarray, "n_voids 4 2"]) -> tuple:
    """``(side_long, side_short, opening_deg)`` per void, from its mean opposite-side lengths.

    The two side lengths are INDEPENDENT properties of the design. Both are fixed by the cut
    pattern and neither changes on deployment, because the panels are rigid; measured over five
    deployed sheets their correlation is 0.16. Reporting only their ratio therefore discards a
    whole dimension of the void's shape, and two voids of equal ratio can differ in size by a
    factor of two.

    ``opening_deg`` is the angle between the two side directions and carries the whole of the
    deployment: ninety degrees at the fully open state, zero on the flat closed sheet. It is
    near-uniform over one sheet -- within a deployed sheet it varies by under half a degree -- so
    it separates one sheet from another, not one void from its neighbour.

    Args:
        quads: from :func:`void_quads`.

    Returns:
        ``(side_long, side_short, opening_deg)``, each ``(n_voids,)``, the lengths in the units of
        ``quads``.
    """
    q = np.asarray(quads, dtype=float)
    if q.ndim != 3 or q.shape[1] != 4:
        raise ValueError(f"void_sides expects (n_voids, 4, 2) quads, got shape {q.shape}")
    edges = q - np.roll(q, -1, axis=1)
    lengths = np.linalg.norm(edges, axis=2)
    side_a = 0.5 * (lengths[:, 0] + lengths[:, 2])
    side_b = 0.5 * (lengths[:, 1] + lengths[:, 3])
    u = edges[:, 0] / np.maximum(np.linalg.norm(edges[:, 0], axis=1, keepdims=True), 1e-12)
    v = edges[:, 1] / np.maximum(np.linalg.norm(edges[:, 1], axis=1, keepdims=True), 1e-12)
    opening = np.degrees(np.arccos(np.clip(np.abs((u * v).sum(axis=1)), 0.0, 1.0)))
    return np.maximum(side_a, side_b), np.minimum(side_a, side_b), opening


def void_octagons(index: VoidIndex,
                  nodes: Float[np.ndarray, "n_faces n_nodes 2"]
                  ) -> Float[np.ndarray, "n_voids 8 2"]:
    """The true hole boundary of a compliant deployment: both owners of each pin, in CCW order."""
    nd = np.asarray(nodes, dtype=float)
    p0 = nd[index.owner_face[:, :, 0], index.owner_node[:, :, 0]]     # (n_voids, 4, 2)
    p1 = nd[index.owner_face[:, :, 1], index.owner_node[:, :, 1]]
    out = np.empty((p0.shape[0], 8, 2), dtype=float)
    out[:, 0::2] = p0
    out[:, 1::2] = p1
    return out


def pin_gaps(index: VoidIndex,
             nodes: Float[np.ndarray, "n_faces n_nodes 2"]) -> Float[np.ndarray, "n_voids 4"]:
    """Separation of the two panel corners meeting at each of a void's four pins.

    Zero for a rigid-tile, perfect-pin deployment. This is the hinge slip seen from the void.
    """
    nd = np.asarray(nodes, dtype=float)
    p0 = nd[index.owner_face[:, :, 0], index.owner_node[:, :, 0]]
    p1 = nd[index.owner_face[:, :, 1], index.owner_node[:, :, 1]]
    return np.linalg.norm(p0 - p1, axis=-1)


# ══════════════════════════════════════════════════════════════════════════════
# Shape defects
# ══════════════════════════════════════════════════════════════════════════════

def _side_lengths(quads: Float[np.ndarray, "n_voids 4 2"]) -> Float[np.ndarray, "n_voids 4"]:
    return np.linalg.norm(np.roll(quads, -1, axis=1) - quads, axis=-1)


def parallelogram_defect(quads: Float[np.ndarray, "n_voids 4 2"], *,
                         normalise: bool = True) -> Float[np.ndarray, "n_voids"]:
    """``|Q0 - Q1 + Q2 - Q3|``, the quad's deviation from a parallelogram.

    Same measure ``nff/topology/quad_warp.py`` documents for the bilinear warp. Normalised by the
    mean side length, so it reads as a fraction of the void's own size and is comparable across a
    non-uniform sheet.
    """
    q = np.asarray(quads, dtype=float)
    d = np.linalg.norm(q[:, 0] - q[:, 1] + q[:, 2] - q[:, 3], axis=-1)
    if not normalise:
        return d
    scale = _side_lengths(q).mean(axis=1)
    return np.where(scale > 0, d / np.maximum(scale, 1e-300), 0.0)


def rhombus_defect(quads: Float[np.ndarray, "n_voids 4 2"]) -> Float[np.ndarray, "n_voids"]:
    """Spread of the four side lengths over their mean.

    At ``r = 0.5`` the ideal void is a rhombus (all four sides ``L/2``), so this is the extra
    condition a perfect-rotating-squares sheet must satisfy and a general RDPQK sheet need not.
    """
    L = _side_lengths(np.asarray(quads, dtype=float))
    m = L.mean(axis=1)
    return np.where(m > 0, (L.max(axis=1) - L.min(axis=1)) / np.maximum(m, 1e-300), 0.0)


def polygon_areas(polys: Float[np.ndarray, "n m 2"]) -> Float[np.ndarray, "n"]:
    """Shoelace area of each polygon, signed positive for CCW ordering."""
    p = np.asarray(polys, dtype=float)
    x, y = p[..., 0], p[..., 1]
    return 0.5 * np.sum(x * np.roll(y, -1, axis=1) - y * np.roll(x, -1, axis=1), axis=1)


def void_report(index: VoidIndex, nodes: Float[np.ndarray, "n_faces n_nodes 2"], *,
                length_scale: float = 1.0) -> dict:
    """Every per-void quantity at one deployed state, plus summary statistics.

    Args:
        index: from :func:`build_void_index`.
        nodes: (n_faces, n_nodes, 2) deployed panel corners, pipeline units.
        length_scale: mm per pipeline unit, applied to the gap and area outputs only. The two
            defects are already dimensionless.
    """
    quads = void_quads(index, nodes)
    octs = void_octagons(index, nodes)
    gaps = pin_gaps(index, nodes) * float(length_scale)
    para = parallelogram_defect(quads)
    rhom = rhombus_defect(quads)
    area_q = polygon_areas(quads) * float(length_scale) ** 2
    area_o = polygon_areas(octs) * float(length_scale) ** 2
    return {
        "quads": quads, "octagons": octs,
        "pin_gap": gaps, "para_defect": para, "rhombus_defect": rhom,
        "area_quad": area_q, "area_octagon": area_o,
        "para_mean": float(para.mean()), "para_max": float(para.max()),
        "rhombus_mean": float(rhom.mean()), "rhombus_max": float(rhom.max()),
        "gap_mean": float(gaps.mean()), "gap_max": float(gaps.max()),
        "area_total": float(area_o.sum()),
        "n_negative_area": int((area_o <= 0).sum()),
    }
