"""The region a local load disturbs on a large sheet, measured away from the supports.

A concentrated load on a kirigami sheet raises a pocket of deformation around itself. Measuring
that pocket needs two things the edge-loaded studies in this package cannot provide.

The first is a load that the supports do not react. A sheet held anywhere carries the applied
resultant through to its supports, and the support then raises a pocket of its own -- on the 15x10
sheet a wider one than the load did. A load whose net force and net moment both vanish is carried
entirely inside the sheet, so the supports see nothing and only fix the coordinate frame.
``self_equilibrate`` makes any local load set self-equilibrated by removing its three resultants,
which is the smallest change to the load that achieves it. Because a three-degree-of-freedom gauge
is statically determinate, its reaction is exactly the negated load resultant, so nulling the
resultants nulls the reaction identically rather than approximately.

The second is a separation of the response. An RDPQK sheet is a one-degree-of-freedom mechanism, and
the part of any load that drives that mechanism does not decay: it opens the whole sheet at once, at
any size. Only the part orthogonal to the mechanism is localised. ``mechanism_split`` removes the
mechanism component so the remainder is the pocket, and the shape measures below then describe
something that has a size at all.

Positions are tile centroids, so a distance here is a length in pipeline units; the hop-count radius
lives in ``nff_paper.closed_support.perturbation`` and the two are complementary.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from jaxtyping import Float, Int


# ══════════════════════════════════════════════════════════════════════════════
# Building a probe the supports cannot feel
# ══════════════════════════════════════════════════════════════════════════════

def rigid_body_modes(centroids: Float[np.ndarray, "n_cluster 2"],
                     origin: Optional[Float[np.ndarray, "2"]] = None
                     ) -> Float[np.ndarray, "3 n_cluster 3"]:
    """The three rigid-body modes of a set of tiles, as generalized-force test vectors.

    Pairing a load with each of these gives that load's net x force, net y force and net moment, so
    a load orthogonal to all three is self-equilibrated. The modes are written in the same
    ``(n_cluster, 3)`` layout as the load itself, one row per tile holding ``(fx, fy, m)``.

    Args:
        centroids: (n_cluster, 2) tile centroids.
        origin: moment reference; the centroid of the cluster when omitted. The span of the three
            modes does not depend on it, so neither does anything built from them.

    Returns:
        (3, n_cluster, 3) stack, ordered as x translation, y translation, rotation.
    """
    xy = np.asarray(centroids, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"centroids must be (n_cluster, 2), got {xy.shape}")
    o = xy.mean(axis=0) if origin is None else np.asarray(origin, dtype=float)
    rel = xy - o

    modes = np.zeros((3, xy.shape[0], 3), dtype=float)
    modes[0, :, 0] = 1.0                                  # net force along x
    modes[1, :, 1] = 1.0                                  # net force along y
    modes[2, :, 0] = -rel[:, 1]                           # net moment: r x F, plus the couples
    modes[2, :, 1] = rel[:, 0]
    modes[2, :, 2] = 1.0
    return modes


def rigid_resultants(load: Float[np.ndarray, "n_cluster 3"],
                     centroids: Float[np.ndarray, "n_cluster 2"],
                     origin: Optional[Float[np.ndarray, "2"]] = None
                     ) -> Float[np.ndarray, "3"]:
    """Net x force, net y force and net moment of a per-tile generalized load."""
    f = np.asarray(load, dtype=float)
    modes = rigid_body_modes(centroids, origin)
    return np.array([float(np.sum(m * f)) for m in modes])


def self_equilibrate(load: Float[np.ndarray, "n_cluster 3"],
                     centroids: Float[np.ndarray, "n_cluster 2"],
                     origin: Optional[Float[np.ndarray, "2"]] = None
                     ) -> Float[np.ndarray, "n_cluster 3"]:
    """Remove a load's three resultants, keeping it on the same tiles.

    The correction is the Euclidean projection onto the complement of the rigid-body span, so it is
    the smallest change that leaves nothing for a support to react. An opposed force pair along the
    line joining two tiles is already self-equilibrated and comes back untouched; the same pair
    applied across that line carries a net moment, and the projection pays for it with a balancing
    couple on each tile rather than by moving the forces.

    Args:
        load: (n_cluster, 3) per-tile ``(fx, fy, m)``.
        centroids: (n_cluster, 2) centroids of those same tiles, in the same order.
        origin: moment reference, defaulting to the cluster centroid.

    Returns:
        (n_cluster, 3) load with all three resultants at machine zero.

    Raises:
        ValueError: if the cluster has fewer than two tiles, where the three constraints exhaust the
            three available components and only the zero load survives.
    """
    f = np.asarray(load, dtype=float)
    xy = np.asarray(centroids, dtype=float)
    if f.shape != (xy.shape[0], 3):
        raise ValueError(f"load must be (n_cluster, 3) matching centroids {xy.shape}, got {f.shape}")
    if xy.shape[0] < 2:
        raise ValueError(
            "a self-equilibrated probe needs at least two tiles; on one tile the three resultants "
            "fix all three components and nothing but the zero load remains")

    basis = rigid_body_modes(xy, origin).reshape(3, -1).T          # (3 n_cluster, 3)
    q, _ = np.linalg.qr(basis)
    flat = f.reshape(-1)
    return (flat - q @ (q.T @ flat)).reshape(f.shape)


def scatter_load(load: Float[np.ndarray, "n_cluster 3"],
                 faces: Sequence[int],
                 n_faces: int) -> Float[np.ndarray, "n_faces 3"]:
    """Place a cluster load onto the whole sheet, zero everywhere else."""
    f = np.asarray(load, dtype=float)
    out = np.zeros((int(n_faces), 3), dtype=float)
    for row, face in zip(f, faces):
        out[int(face)] += row
    return out


def load_specs(load: Float[np.ndarray, "n_cluster 3"],
               faces: Sequence[int],
               *, tol: float = 0.0) -> list:
    """Convert a per-tile generalized load into the pipeline's ``loads:`` entries.

    A repeated ``(face, dof)`` pair overwrites rather than sums downstream, so each pair is emitted
    once. Components at or below ``tol`` are dropped, which keeps a pinch probe from carrying three
    entries per tile when it only needs one.
    """
    out = []
    for row, face in zip(np.asarray(load, dtype=float), faces):
        for dof in range(3):
            if abs(row[dof]) > tol:
                out.append({'face': int(face), 'dof': int(dof), 'value': float(row[dof])})
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Separating the mechanism from the pocket
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MechanismSplit:
    """A response separated into the non-decaying mechanism part and the localised remainder."""
    amplitude: float
    mechanism: Float[np.ndarray, "n_free"]
    pocket: Float[np.ndarray, "n_free"]
    share: float

    def as_dict(self) -> dict:
        return {"amplitude": self.amplitude, "share": self.share}


def mechanism_split(response: Float[np.ndarray, "n_free"],
                    mode: Float[np.ndarray, "n_free"]) -> MechanismSplit:
    """Project a response onto the sheet's mechanism mode and keep the remainder.

    ``share`` is the fraction of the response's squared norm carried by the mechanism. It is close
    to one whenever the probe drives the mechanism, and that is not a defect of the probe: it says
    the response has no finite size, because the mechanism component is the same everywhere.

    Args:
        response: (n_free,) displacement response in free-DOF coordinates.
        mode: (n_free,) mechanism mode, normalised internally.

    Returns:
        ``MechanismSplit``.
    """
    d = np.asarray(response, dtype=float)
    v = np.asarray(mode, dtype=float)
    nrm = float(np.linalg.norm(v))
    if nrm == 0:
        raise ValueError("mechanism mode is the zero vector")
    v = v / nrm

    amplitude = float(v @ d)
    mech = amplitude * v
    pocket = d - mech
    total = float(d @ d)
    share = float(amplitude ** 2 / total) if total > 0 else float("nan")
    return MechanismSplit(amplitude=amplitude, mechanism=mech, pocket=pocket, share=share)


# ══════════════════════════════════════════════════════════════════════════════
# The shape of the pocket
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SubspaceSplit:
    """A response with a whole subspace of global content removed."""
    amplitudes: Float[np.ndarray, "k"]
    removed: Float[np.ndarray, "n_free"]
    pocket: Float[np.ndarray, "n_free"]
    share: float

    def as_dict(self) -> dict:
        return {"amplitudes": [float(a) for a in self.amplitudes], "share": self.share}


def subspace_split(response: Float[np.ndarray, "n_free"],
                   modes: Float[np.ndarray, "k n_free"]) -> SubspaceSplit:
    """Remove every component of a response that lies in a given subspace.

    One vector is not always enough. A probe on a deployed sheet does two things at once: it raises
    a local disturbance, and it deploys the sheet slightly further. The second is not a mode of the
    sheet but the sheet's whole response to its own actuation, boundary layer at the actuator
    included, so removing only the softest eigenvector leaves that boundary layer behind and the
    remainder peaks at the actuator rather than at the probe.

    Args:
        response: (n_free,) response to split.
        modes: (k, n_free) directions spanning the global content. They need not be orthogonal or
            normalised; the span is what matters.

    Returns:
        ``SubspaceSplit``. ``amplitudes`` are the coefficients against an orthonormal basis of the
        span, so they are comparable to each other but not to the raw ``modes``.
    """
    d = np.asarray(response, dtype=float)
    m = np.atleast_2d(np.asarray(modes, dtype=float))
    if m.shape[1] != d.size:
        raise ValueError(f"modes must be (k, {d.size}), got {m.shape}")

    q, _ = np.linalg.qr(m.T)                            # (n_free, k) orthonormal span
    amplitudes = q.T @ d
    removed = q @ amplitudes
    pocket = d - removed
    total = float(d @ d)
    share = float(removed @ removed / total) if total > 0 else float("nan")
    return SubspaceSplit(amplitudes=amplitudes, removed=removed, pocket=pocket, share=share)


@dataclass
class PocketShape:
    """Second-moment description of a per-tile departure field around the probe.

    ``gyration`` is the field-weighted root-mean-square distance from the probe, so it is a reach in
    length units. ``aspect`` is at least one and equals one for a round pocket; ``angle`` is the
    direction of the long axis in radians, and is meaningless as ``aspect`` approaches one.
    """
    gyration: float
    semi_major: float
    semi_minor: float
    aspect: float
    angle: float
    offset: Float[np.ndarray, "2"]
    mass: float

    def as_dict(self) -> dict:
        return {"gyration": self.gyration, "semi_major": self.semi_major,
                "semi_minor": self.semi_minor, "aspect": self.aspect, "angle": self.angle,
                "offset_x": float(self.offset[0]), "offset_y": float(self.offset[1]),
                "mass": self.mass}


def pocket_shape(values: Float[np.ndarray, "n_faces"],
                 centroids: Float[np.ndarray, "n_faces 2"],
                 origin: Float[np.ndarray, "2"]) -> PocketShape:
    """Size, elongation and direction of a departure field, as its weighted second moment.

    The moment is taken about the probe rather than about the field's own centre of mass, because
    the question is how far the disturbance reaches from where it was applied. The centre of mass is
    reported separately as ``offset``, which says whether the pocket leans to one side.

    Args:
        values: (n_faces,) per-tile departure; the absolute value is used as the weight.
        centroids: (n_faces, 2) tile centroids.
        origin: (2,) position of the probe.

    Returns:
        ``PocketShape``. Every length is in the units of ``centroids``.
    """
    w = np.abs(np.asarray(values, dtype=float))
    xy = np.asarray(centroids, dtype=float)
    if xy.shape != (w.size, 2):
        raise ValueError(f"centroids must be ({w.size}, 2), got {xy.shape}")
    total = float(w.sum())
    if total <= 0:
        raise ValueError("departure field is identically zero, so it has no shape")

    rel = xy - np.asarray(origin, dtype=float)
    second = (w[:, None, None] * rel[:, :, None] * rel[:, None, :]).sum(axis=0) / total
    evals, evecs = np.linalg.eigh(second)
    evals = np.clip(evals, 0.0, None)                      # a weighted second moment is PSD
    major, minor = float(evals[1]), float(evals[0])
    axis = evecs[:, 1]

    semi_major, semi_minor = float(np.sqrt(major)), float(np.sqrt(minor))
    return PocketShape(
        gyration=float(np.sqrt(major + minor)),
        semi_major=semi_major,
        semi_minor=semi_minor,
        aspect=semi_major / semi_minor if semi_minor > 0 else float("inf"),
        angle=float(np.arctan2(axis[1], axis[0])) % np.pi,
        offset=(w[:, None] * rel).sum(axis=0) / total,
        mass=total,
    )


def angular_profile(values: Float[np.ndarray, "n_faces"],
                    centroids: Float[np.ndarray, "n_faces 2"],
                    origin: Float[np.ndarray, "2"],
                    n_sectors: int = 12) -> dict:
    """Weighted reach of the departure field per angular sector around the probe.

    The second moment gives one direction and one elongation, which is the right summary for an
    elliptical pocket and the wrong one for a pocket with lobes. This resolves the reach as a
    function of direction so a lobed pocket is visible as one.

    Args:
        values: (n_faces,) per-tile departure.
        centroids: (n_faces, 2) tile centroids.
        origin: (2,) position of the probe.
        n_sectors: number of equal angular bins over the full turn.

    Returns:
        Dict with ``angle`` (sector centres, radians), ``reach`` (weighted mean distance in each
        sector, NaN where the sector holds no weight), ``mass`` (weight per sector) and ``n``
        (tiles per sector). The probe's own tile sits at zero radius and is excluded.
    """
    w = np.abs(np.asarray(values, dtype=float))
    rel = np.asarray(centroids, dtype=float) - np.asarray(origin, dtype=float)
    dist = np.hypot(rel[:, 0], rel[:, 1])
    keep = dist > 0

    edges = np.linspace(-np.pi, np.pi, int(n_sectors) + 1)
    ang = np.arctan2(rel[:, 1], rel[:, 0])
    which = np.clip(np.digitize(ang[keep], edges) - 1, 0, int(n_sectors) - 1)

    reach = np.full(int(n_sectors), np.nan)
    mass = np.zeros(int(n_sectors))
    count = np.zeros(int(n_sectors), dtype=np.int64)
    for s in range(int(n_sectors)):
        sel = which == s
        count[s] = int(sel.sum())
        if not sel.any():
            continue
        ws, ds = w[keep][sel], dist[keep][sel]
        mass[s] = float(ws.sum())
        if mass[s] > 0:
            reach[s] = float(np.sum(ws * ds) / mass[s])
    return {"angle": 0.5 * (edges[:-1] + edges[1:]), "reach": reach, "mass": mass, "n": count}


# ══════════════════════════════════════════════════════════════════════════════
# Sheet bookkeeping
# ══════════════════════════════════════════════════════════════════════════════

def tile_adjacency(bond_connectivity: Int[np.ndarray, "n_hinges 2"],
                   n_nodes: int) -> Int[np.ndarray, "n_hinges 2"]:
    """Tile pairs from the node-level bond list, which stores ``face * n_nodes + corner``."""
    fp = np.asarray(bond_connectivity)
    return np.column_stack([fp[:, 0] // int(n_nodes), fp[:, 1] // int(n_nodes)])


def energy_decomposition(plus: Float[np.ndarray, "n"],
                         zero: Float[np.ndarray, "n"],
                         minus: Float[np.ndarray, "n"]) -> tuple:
    """Split a probe's effect on the stored energy into the two things it is doing at once.

    The stored energy is quadratic in the displacement, so the change a probe makes to it is not
    proportional to the probe. It is a term linear in the probe, which is the probe REDISTRIBUTING
    the stress the background already carries, plus a term quadratic in it, which is the probe's OWN
    stress field. Those answer different questions and only their sum was being measured, so the
    measured shape drifted with probe magnitude for a reason that says nothing about the sheet.

    Evaluating the energy at ``+delta``, ``0`` and ``-delta`` separates them exactly, by parity, and
    costs no solve. Both parts are then homogeneous in the probe, so in the linear regime each has a
    shape that cannot depend on its magnitude at all.

    Args:
        plus: energy at ``background + delta``.
        zero: energy at ``background``.
        minus: energy at ``background - delta``.

    Returns:
        ``(own, redistribution)``. ``own`` scales as the square of the probe, ``redistribution``
        linearly, and ``own + redistribution == plus - zero`` identically.
    """
    p, z, m = (np.asarray(a, dtype=float) for a in (plus, zero, minus))
    return 0.5 * (p + m) - z, 0.5 * (p - m)


def decomposition_precision(plus: Float[np.ndarray, "n"],
                            zero: Float[np.ndarray, "n"],
                            minus: Float[np.ndarray, "n"]) -> dict:
    """Relative round-off floor on the split, which a too-small probe walks straight into.

    ``own`` is second order in the probe but is computed by differencing energies that are not, so
    halving the probe quarters the signal while leaving the cancellation error where it was. Below
    some probe size the returned ``own`` is round-off wearing the shape of an answer, and nothing
    downstream can tell. This estimates the damage so the caller can refuse rather than plot it.

    Returns:
        Dict with ``own_rel`` and ``redist_rel``, the estimated relative error on each part, and
        ``scale``, the energy the cancellation is measured against. Keep ``own_rel`` below about
        1e-6; raise the probe magnitude if it is not.
    """
    p, z, m = (np.asarray(a, dtype=float) for a in (plus, zero, minus))
    own, redist = energy_decomposition(p, z, m)
    scale = float(np.max(np.abs(np.stack([p, z, m]))))
    noise = float(np.finfo(float).eps) * scale
    own_peak, redist_peak = float(np.max(np.abs(own))), float(np.max(np.abs(redist)))
    return {"own_rel": noise / own_peak if own_peak > 0 else float("inf"),
            "redist_rel": noise / redist_peak if redist_peak > 0 else float("inf"),
            "scale": scale}


@dataclass
class DecayProfile:
    """How a per-tile field falls off with hop distance from the probe.

    ``ratios[h]`` is the drop from shell ``h`` to shell ``h+1``, so ``ratios[0]`` is the step from
    the probe to its neighbours and the later entries are the per-tile rate. ``decay_length`` is in
    tiles, fitted only over the shells given, because the outer shells sit on the finite sheet's own
    floor rather than on the decay.
    """
    shells: Float[np.ndarray, "n_hops"]
    ratios: Float[np.ndarray, "n_hops-1"]
    decay_length: float
    r2: float
    fit_hops: tuple

    def as_dict(self) -> dict:
        return {"shells": [float(v) for v in self.shells],
                "ratios": [float(v) for v in self.ratios],
                "decay_length": self.decay_length, "r2": self.r2,
                "fit_hops": list(self.fit_hops)}


def decay_ratios(values: Float[np.ndarray, "n_faces"],
                 distance: Int[np.ndarray, "n_faces"],
                 fit_hops: Sequence[int] = (1, 2, 3)) -> DecayProfile:
    """Shell medians against hop distance, the drop between consecutive shells, and a decay length.

    The median rather than the mean, because a shell holds a handful of tiles and one of them may be
    the probe's own neighbour. The fit is restricted to ``fit_hops`` and reports its ``r2``: a
    decay length quoted without one is a slope through whatever the boundary happened to leave.

    Args:
        values: (n_faces,) non-negative per-tile field.
        distance: (n_faces,) hop distance, negative where unreachable.
        fit_hops: shells the exponential fit uses.

    Returns:
        ``DecayProfile``.
    """
    v = np.abs(np.asarray(values, dtype=float))
    d = np.asarray(distance)
    hops = np.arange(0, int(d.max()) + 1)
    shells = np.array([np.median(v[d == h]) if (d == h).any() else np.nan for h in hops])

    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = shells[:-1] / shells[1:]

    keep = [h for h in fit_hops if h < shells.size and np.isfinite(shells[h]) and shells[h] > 0]
    if len(keep) < 2:
        return DecayProfile(shells=shells, ratios=ratios, decay_length=float("nan"),
                            r2=float("nan"), fit_hops=tuple(fit_hops))
    x = np.array(keep, dtype=float)
    y = np.log(shells[keep])
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    ss = float(np.sum((y - y.mean()) ** 2))
    # A flat set of shells fits a slope of order 1e-17 rather than exactly zero, and inverting that
    # returns 1e16, which reads as a measured decay length. Anything that shallow is no decay.
    return DecayProfile(
        shells=shells, ratios=ratios,
        decay_length=float(-1.0 / slope) if slope < -1e-12 else float("inf"),
        r2=1.0 - float(np.sum(resid ** 2)) / ss if ss > 0 else float("nan"),
        fit_hops=tuple(fit_hops))


def scatter_hinge_to_tiles(values: Float[np.ndarray, "n_hinges"],
                           face_pairs: Int[np.ndarray, "n_hinges 2"],
                           n_faces: int) -> Float[np.ndarray, "n_faces"]:
    """Spread a per-hinge quantity onto tiles, half to each side of every hinge.

    The shape measures work on tiles because that is where positions live, while the energy the
    stiff springs store is a per-hinge quantity. Splitting evenly conserves the total.
    """
    v = np.asarray(values, dtype=float)
    fp = np.asarray(face_pairs)
    out = np.zeros(int(n_faces), dtype=float)
    np.add.at(out, fp[:, 0], 0.5 * v)
    np.add.at(out, fp[:, 1], 0.5 * v)
    return out


def centre_face(M: int, N: int) -> int:
    """Index of the centre tile, which exists only on an odd grid.

    Faces are numbered ``col * N + row`` (``nff/topology/closed_builder.py``). An even grid has no
    tile at the sheet centre, and probing half a pitch off it breaks the four-fold symmetry the
    measurement relies on, so this refuses rather than rounding.
    """
    if int(M) % 2 == 0 or int(N) % 2 == 0:
        raise ValueError(
            f"grid {M}x{N} has no centre tile; a centred probe needs both dimensions odd")
    return (int(M) // 2) * int(N) + int(N) // 2
