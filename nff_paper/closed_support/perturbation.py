"""How far a concentrated boundary condition disturbs the sheet, as a radius in tiles.

The departure from ideal 1-DOF kinematics is not spread through the sheet: it sits in boundary layers
around whatever is applied to it -- the grip and each support -- and the interior stays close to the
mechanism. This measures the SIZE of those zones, so "the disturbance is local" becomes a number that
can be compared across deployments.

The radius is defined by a THRESHOLD, not by fitting a decay: the extent over which the departure
exceeds ``factor`` times its interior level. Nothing is assumed about the shape of the decay, which
matters here -- fitting a single exponential in distance-from-the-grip is the wrong model on this
sheet, because the residual is also large at the supports, so a global fit produces a slope that
means nothing.

Distances are hop counts on the tile-adjacency graph, so a radius is in TILES; multiply by the panel
pitch for millimetres.
"""

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
from jaxtyping import Float, Int

from nff_paper.closed_support.one_dof import graph_distance


@dataclass
class RadiusResult:
    """Per-source disturbed radius, with the baseline it was measured against.

    ``radius`` is ``NaN`` for a source whose own tile is already below the threshold -- that is not a
    radius of zero, it is an absence of any disturbed zone, and the two should not be averaged
    together.
    """
    radius: Dict[int, float]
    radius_interp: Dict[int, float]
    baseline: float
    n_interior: int
    factor: float

    def as_list(self, sources: Sequence[int], interp: bool = True):
        src = self.radius_interp if interp else self.radius
        return [src[int(s)] for s in sources]


def perturbation_radius(values: Float[np.ndarray, "n_faces"],
                        face_pairs: Int[np.ndarray, "n_hinges 2"],
                        sources: Sequence[int],
                        factor: float = 2.0,
                        interior_min_hops: int = 4) -> RadiusResult:
    """Radius of the disturbed zone around each source, in tiles.

    Args:
        values: (n_faces,) per-tile departure, any non-negative measure.
        face_pairs: (n_hinges, 2) tile adjacency, used for hop distances.
        sources: the tiles carrying a concentrated boundary condition (grip, pin, roller).
        factor: a tile is disturbed when its value exceeds ``factor`` times the interior level.
        interior_min_hops: tiles at least this far from EVERY source define the interior baseline.

    Returns:
        ``RadiusResult``.

    Raises:
        ValueError: if no tile is far enough from every source to form a baseline, which means the
            sheet is too small for this measurement rather than that the radius is large.
    """
    v = np.abs(np.asarray(values, dtype=float))
    n_faces = v.size
    dists = {int(s): graph_distance(face_pairs, n_faces, [int(s)]) for s in sources}
    nearest = np.min(np.stack([dists[s] for s in dists]), axis=0)

    interior = nearest >= int(interior_min_hops)
    if not interior.any():
        raise ValueError(
            f"no tile is {interior_min_hops} hops from every source, so there is no interior to "
            "compare against; lower interior_min_hops or use a larger sheet")
    baseline = float(np.median(v[interior]))
    cut = factor * baseline

    # Each tile is attributed to its nearest source, so two boundary layers cannot claim the same
    # tile and inflate both radii.
    owner = np.full(n_faces, -1, dtype=np.int64)
    best = np.full(n_faces, np.iinfo(np.int64).max)
    for s, d in dists.items():
        closer = d < best
        owner[closer] = s
        best[closer] = d[closer]

    radius, radius_interp = {}, {}
    for s, d in dists.items():
        mine = (owner == s) & (d >= 0)
        if not mine.any():
            radius[s] = radius_interp[s] = float("nan")
            continue
        # Walk outwards; the radius is the last shell whose median still exceeds the threshold.
        prof = []
        for hop in range(int(d[mine].max()) + 1):
            shell = mine & (d == hop)
            if shell.any():
                prof.append((float(hop), float(np.median(v[shell]))))
        r = float("nan")
        r_i = float("nan")
        for idx, (hop, med) in enumerate(prof):
            if med > cut:
                r = hop
                continue
            # Linear crossing between the last shell above the cut and this one, in log(value):
            # hop counts are integers, so without this the radius can only be an integer and a
            # smoothly varying zone reads as a square wave.
            if idx == 0:
                r_i = float("nan")
            else:
                h0, v0 = prof[idx - 1]
                lo, hi = np.log(max(med, 1e-300)), np.log(max(v0, 1e-300))
                w = 0.0 if hi == lo else (np.log(cut) - hi) / (lo - hi)
                r_i = h0 + float(np.clip(w, 0.0, 1.0)) * (hop - h0)
            break
        else:
            r_i = r                     # never dropped below the cut inside this source's territory
        radius[s] = r
        radius_interp[s] = r_i if not np.isnan(r) else float("nan")

    return RadiusResult(radius=radius, radius_interp=radius_interp, baseline=baseline,
                        n_interior=int(interior.sum()), factor=float(factor))
