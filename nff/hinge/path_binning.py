"""Reduce hinge deployment paths to point clouds a hex grid can bin.

The primitives every hinge-path figure shares: which plane to project onto, how to turn a trajectory
into points that bin fairly, and how to count paths per cell. They live here rather than beside the
figures because ``run_closed`` emits one of those figures as part of a run, and ``nff/closed/`` may
not import ``nff/scripts/`` -- the dependency graph runs strictly downward.

★ THE BINNING COUNTS DISTINCT PATHS, not path sample points. A deployment is sampled at
``num_load_steps`` points, and those are NOT uniformly spaced along the trajectory -- a hinge that
stalls mid-deployment drops many samples in one place. Binning raw samples therefore shades by how
SLOWLY a path moves, which is a property of the load stepping rather than of the mechanism.
:func:`densify` resamples along arc length and tags every point with its hinge id, so a reduction
over unique ids answers the question actually being asked: how many hinges route through here.
"""

from typing import NamedTuple

import numpy as np


N_RESAMPLE = 400          # points per trajectory after arc-length densification

# Keeps "after" ids disjoint from "before" ids inside a single shared reduction, so one hexbin call
# over the concatenated clouds can return their signed difference. Larger than any hinge count.
ID_OFFSET = 1_000_000


class Projection(NamedTuple):
    """Which two of ``(eta_a, eta_s, theta)`` to draw, and how to annotate that plane.

    Args:
        xi, yi: component indices into ``paths.eta``'s last axis.
        degrees: components to convert from radians (``theta`` is stored in radians).
        xlabel, ylabel: axis labels.
        compression_band: shade ``eta_a < 0`` as a vertical band. Only meaningful when x is eta_a.
        shear_line: draw a horizontal guide at y = 0, the shear-free state.
    """
    xi: int
    yi: int
    degrees: tuple
    xlabel: str
    ylabel: str
    compression_band: bool
    shear_line: bool


PROJECTIONS = {
    'eta_a_theta': Projection(
        xi=0, yi=2, degrees=(2,),
        xlabel=r"$\eta_a = a / w_{lig}$        (shaded: COMPRESSION)",
        ylabel=r"$\theta$  [deg]",
        compression_band=True, shear_line=False),
    'eta_a_eta_s': Projection(
        xi=0, yi=1, degrees=(),
        xlabel=r"$\eta_a = a / w_{lig}$        (shaded: COMPRESSION)",
        ylabel=r"$\eta_s = s / w_{lig}$        (dashed: pure opening)",
        compression_band=True, shear_line=True),
}


def trajectories(paths, proj):
    """(n_hinges, n_steps+1, 2) per-hinge curves through the chosen plane, origin included."""
    eta = np.asarray(paths.eta)
    axes = []
    for i in (proj.xi, proj.yi):
        comp = eta[..., i].T
        axes.append(np.degrees(comp) if i in proj.degrees else comp)
    return np.stack(axes, axis=-1)


def densify(paths, proj, n_resample=N_RESAMPLE, id_offset=0):
    """Resample every path along its arc length -> (points (N,2), hinge ids (N,)).

    Coarse load steps leave gaps a path visibly crosses but drops no sample into, which would punch
    holes in the density. Resampling by arc length closes them and makes the count independent of
    how the deployment happened to be stepped.
    """
    traj = trajectories(paths, proj)
    n_hinges = traj.shape[0]
    # Arc length in the PLOTTED coordinates, after scaling theta to eta_a's range, so neither axis
    # dominates the parameterisation just because degrees are numerically large.
    scale = np.array([1.0, 1.0 / max(np.ptp(traj[..., 1]), 1e-9) * max(np.ptp(traj[..., 0]), 1e-9)])
    out_pts, out_ids = [], []
    for h in range(n_hinges):
        curve = traj[h]
        seg = np.linalg.norm(np.diff(curve * scale, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        if s[-1] <= 0.0:                       # a hinge that never moved: keep its single location
            out_pts.append(curve[:1])
            out_ids.append(np.full(1, h + id_offset))
            continue
        q = np.linspace(0.0, s[-1], n_resample)
        out_pts.append(np.stack([np.interp(q, s, curve[:, 0]), np.interp(q, s, curve[:, 1])], -1))
        out_ids.append(np.full(n_resample, h + id_offset))
    return np.concatenate(out_pts, 0), np.concatenate(out_ids, 0)


def endpoints(paths, proj, id_offset=0):
    """One point per hinge: the DEPLOYED strain state, ignoring how it got there.

    The whole-path view has an artefact that no amount of styling removes. Path COUNT is conserved
    (the hinge count is the same either way) but the number of cells a path CROSSES is not, so a
    design that deploys less draws shorter trajectories, touches fewer cells, and tilts the whole
    field toward "fewer hinges route here" without rerouting anything.

    Endpoints are conserved EXACTLY: each hinge contributes one point to each design, so the signed
    change sums to zero over the plane and every gain is paid for by a loss.

    ⚠ The trade is real, and in the direction of UNDER-reporting. ``compression`` is a PATH penalty
    -- it acts on the worst state a hinge passes through, and a ligament squeezed at mid-deployment
    has already buckled whatever its final state says. Use this view to prove redistribution; use
    the path view to see what is being penalised.
    """
    traj = trajectories(paths, proj)
    return traj[:, -1, :], np.arange(traj.shape[0]) + id_offset


def n_unique(values):
    """Hex reduction: how many DISTINCT paths put a point in this cell."""
    return float(np.unique(np.asarray(values)).size)


def n_unique_delta(values):
    """Hex reduction for a difference field: (# after paths) - (# before paths) in this cell."""
    v = np.asarray(values)
    after = v[v >= ID_OFFSET]
    before = v[v < ID_OFFSET]
    return float(np.unique(after).size) - float(np.unique(before).size)
