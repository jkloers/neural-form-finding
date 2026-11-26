"""How far a compliant deployment departs from the sheet's single kinematic degree of freedom.

The closed RDPQK sheet is a 1-DOF mechanism: with rigid tiles and perfect pins every panel
rotates by ``sigma_k * psi`` on a checkerboard of signs, and one scalar ``psi`` fixes the whole
configuration (``nff/closed/shield_kinematics.py``). A Stage-2 equilibrium has compliant hinges,
so it only approximates that family. These functions measure the gap.

The angle fit carries a rigid-body rotation alongside ``psi`` because the deployment mode
counter-rotates neighbours and averages to zero: a non-zero mean face rotation is the sheet
swinging bodily about its supports, which is a support artifact rather than deployment, and
lumping it into ``psi`` would hide it.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from jaxtyping import Float, Int


# ══════════════════════════════════════════════════════════════════════════════
# The 1-DOF angle fit
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class OneDofFit:
    """Least-squares fit of ``theta_k = swing + sigma_k * psi`` to a measured angle field.

    ``psi`` and ``swing`` are radians. ``residual`` is the per-tile part of the angle field that
    the 1-DOF family cannot represent.
    """
    psi: float
    swing: float
    residual: Float[np.ndarray, "n_faces"]
    rms: float
    rms_over_psi: float
    r2: float

    def as_dict(self) -> dict:
        return {"psi": self.psi, "swing": self.swing, "rms": self.rms,
                "rms_over_psi": self.rms_over_psi, "r2": self.r2}


def fit_one_dof(theta: Float[np.ndarray, "n_faces"],
                sigma: Int[np.ndarray, "n_faces"]) -> OneDofFit:
    """Fit the two-parameter 1-DOF model to a per-tile rotation field.

    With ``sigma`` in ``{-1, +1}`` and the two colours equinumerous the two regressors are
    exactly orthogonal, so the estimates are plain means. An odd tile count cannot balance, and a
    sheet with an odd number of tiles is exactly what a centred probe needs, so the unbalanced case
    is solved as an ordinary two-column least squares instead of being refused. The closed form is
    kept when it applies because it is exact and allocation-free, and the two agree to machine
    precision wherever both are defined.

    Args:
        theta: (n_faces,) measured tile rotations, radians.
        sigma: (n_faces,) checkerboard signs in {-1, +1}.

    Returns:
        ``OneDofFit``. ``r2`` is measured against the swing-removed field, so it reports how much
        of the *deployment* is 1-DOF rather than how much of the total motion is.
    """
    theta = np.asarray(theta, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if set(np.unique(sigma)) - {-1.0, 1.0}:
        raise ValueError(f"sigma must be +-1, got values {np.unique(sigma)}")

    if int((sigma > 0).sum()) * 2 == sigma.size:
        swing = float(np.mean(theta))
        psi = float(np.mean(sigma * theta))
    else:
        design = np.column_stack([np.ones_like(sigma), sigma])
        (swing, psi), *_ = np.linalg.lstsq(design, theta, rcond=None)
        swing, psi = float(swing), float(psi)
    residual = theta - swing - sigma * psi
    rms = float(np.sqrt(np.mean(residual ** 2)))
    centred_var = float(np.var(theta - swing))
    r2 = 1.0 - float(np.var(residual)) / centred_var if centred_var > 0 else float("nan")
    return OneDofFit(psi=psi, swing=swing, residual=residual, rms=rms,
                     rms_over_psi=rms / abs(psi) if psi != 0 else float("inf"), r2=r2)


def relative_rotations(theta: Float[np.ndarray, "n_faces"],
                       face_pairs: Int[np.ndarray, "n_hinges 2"]) -> Float[np.ndarray, "n_hinges"]:
    """Signed relative rotation across every hinge, radians."""
    theta = np.asarray(theta, dtype=float)
    fp = np.asarray(face_pairs)
    return theta[fp[:, 1]] - theta[fp[:, 0]]


def localisation_ratio(theta: Float[np.ndarray, "n_faces"],
                       face_pairs: Int[np.ndarray, "n_hinges 2"],
                       psi: float) -> float:
    """Worst hinge rotation over the 1-DOF prediction ``2|psi|``.

    Exactly 1 when every hinge opens by the mechanism amount. Large values mean the rotation is
    concentrated on a few hinges while the rest of the sheet has barely moved, which is what a
    localised deformation looks like in the angle field.
    """
    if psi == 0:
        return float("inf")
    rel = np.abs(relative_rotations(theta, face_pairs))
    return float(rel.max() / (2.0 * abs(psi)))


# ══════════════════════════════════════════════════════════════════════════════
# Spatial structure: how far the motion reaches from the load
# ══════════════════════════════════════════════════════════════════════════════

def graph_distance(face_pairs: Int[np.ndarray, "n_hinges 2"], n_faces: int,
                   sources: Sequence[int]) -> Int[np.ndarray, "n_faces"]:
    """Hop distance to the nearest source tile on the tile-adjacency graph.

    Uses the hinge list rather than tile positions, so the metric follows the load path the sheet
    actually has. Unreachable tiles get ``-1``.
    """
    fp = np.asarray(face_pairs)
    adj = [[] for _ in range(n_faces)]
    for a, b in fp:
        adj[int(a)].append(int(b))
        adj[int(b)].append(int(a))
    dist = np.full(n_faces, -1, dtype=np.int64)
    frontier = [int(s) for s in sources]
    for s in frontier:
        dist[s] = 0
    while frontier:
        nxt = []
        for u in frontier:
            for v in adj[u]:
                if dist[v] < 0:
                    dist[v] = dist[u] + 1
                    nxt.append(v)
        frontier = nxt
    return dist


def decay_length(values: Float[np.ndarray, "n"], distance: Int[np.ndarray, "n"], *,
                 floor: float = 1e-14) -> dict:
    """Fit ``log|v| = a - d / lam`` and return the decay length in hops.

    A localised field decays over a few tiles; a mechanism-wide field is flat, giving a very large
    ``lam``. Values at or below ``floor`` are dropped rather than clipped, since a clipped floor
    invents a plateau and flattens the fit.
    """
    v = np.abs(np.asarray(values, dtype=float))
    d = np.asarray(distance, dtype=float)
    keep = (v > floor) & (d >= 0)
    if keep.sum() < 3:
        return {"lam": float("nan"), "inv_lam": float("nan"), "r2": float("nan"),
                "n": int(keep.sum())}
    y = np.log(v[keep])
    x = d[keep]
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - float(np.sum((y - pred) ** 2)) / ss if ss > 0 else float("nan")
    # A flat field fits a slope of order 1e-17 rather than exactly zero, which would otherwise be
    # inverted into a decay length of 1e16 and read as a number. Anything that shallow is no decay.
    lam = float(-1.0 / slope) if slope < -1e-12 else float("inf")
    # ``inv_lam`` is the plottable form: 0 for a field that does not decay at all (the whole sheet
    # moving together), large for one confined to a few tiles around the load. ``lam`` is infinite
    # in the first case, which is correct and useless on an axis.
    return {"lam": lam, "inv_lam": float(-slope), "r2": float(r2), "n": int(keep.sum()),
            "slope": float(slope)}


def arrival_fractions(theta_history: Float[np.ndarray, "n_steps n_faces"],
                      sigma: Int[np.ndarray, "n_faces"],
                      load_fraction: Float[np.ndarray, "n_steps"],
                      frac: float = 0.5,
                      rel_floor: float = 1e-3) -> Float[np.ndarray, "n_faces"]:
    """Load fraction at which each tile first reaches ``frac`` of its final rotation.

    Identical values across tiles mean the sheet moves as one; a spread means a front. Tiles whose
    final rotation is below ``rel_floor`` of the largest are returned as NaN rather than given a
    meaningless arrival, since dividing by a near-zero endpoint manufactures arbitrary values.
    """
    m = np.asarray(theta_history, dtype=float) * np.asarray(sigma, dtype=float)[None, :]
    lf = np.asarray(load_fraction, dtype=float)
    final = m[-1]
    scale = np.max(np.abs(final))
    out = np.full(m.shape[1], np.nan)
    for k in range(m.shape[1]):
        if scale <= 0 or abs(final[k]) < rel_floor * scale:
            continue
        target = frac * final[k]
        track = m[:, k] / final[k]                      # normalised, so target is `frac`
        hit = np.nonzero(track >= frac)[0]
        if hit.size == 0:
            continue
        i = int(hit[0])
        if i == 0:
            out[k] = lf[0]
            continue
        t0, t1 = track[i - 1], track[i]
        w = 0.0 if t1 == t0 else (frac - t0) / (t1 - t0)
        out[k] = lf[i - 1] + w * (lf[i] - lf[i - 1])
    return out


# ══════════════════════════════════════════════════════════════════════════════
# The stronger test: fit the whole configuration, not just the angles
# ══════════════════════════════════════════════════════════════════════════════

def _kabsch_2d(src: Float[np.ndarray, "n 2"], dst: Float[np.ndarray, "n 2"]):
    """Optimal rotation + translation carrying ``src`` onto ``dst`` (no scaling)."""
    cs, cd = src.mean(axis=0), dst.mean(axis=0)
    H = (src - cs).T @ (dst - cd)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, d]) @ U.T
    return R, cd - R @ cs


def fit_configuration(plan, flat_tiles: Float[np.ndarray, "n_tiles 4 2"],
                      target_nodes: Float[np.ndarray, "n_tiles 4 2"], *,
                      psi_bounds=(-0.7854, 0.0), n_grid: int = 97,
                      n_refine: int = 40) -> dict:
    """Best rigid-body-aligned 1-DOF configuration matching a deployed node field.

    Stronger than :func:`fit_one_dof`: it compares positions, so it also sees translation error,
    which the angle fit is blind to. ``psi`` is found by grid search plus golden-section refinement
    because the objective is smooth and one-dimensional -- the rigid alignment inside is closed
    form, so each evaluation is cheap.

    Args:
        plan: ``shield_kinematics.build_deploy_plan`` output.
        flat_tiles: (n_tiles, 4, 2) undeployed panel corners.
        target_nodes: (n_tiles, 4, 2) deployed corners to match.

    Returns:
        ``{'psi', 'rmse', 'max_err', 'per_tile', 'nodes'}``; errors in the units of ``target_nodes``.
        ``nodes`` is the fitted 1-DOF configuration itself, rigidly aligned onto ``target_nodes``, so
        the caller can difference the two without repeating the deployment and the alignment.
    """
    from nff.closed.shield_kinematics import deploy_tiles

    tgt = np.asarray(target_nodes, dtype=float).reshape(-1, 2)

    def cost(psi: float):
        dep = np.asarray(deploy_tiles(plan, flat_tiles, float(psi)), dtype=float).reshape(-1, 2)
        R, t = _kabsch_2d(dep, tgt)
        aligned = dep @ R.T + t
        err = np.linalg.norm(aligned - tgt, axis=1)
        return float(np.sqrt(np.mean(err ** 2))), err, aligned

    grid = np.linspace(psi_bounds[0], psi_bounds[1], int(n_grid))
    vals = [cost(p)[0] for p in grid]
    i = int(np.argmin(vals))
    lo = grid[max(i - 1, 0)]
    hi = grid[min(i + 1, grid.size - 1)]
    phi = 0.5 * (np.sqrt(5.0) - 1.0)
    a, b = lo, hi
    c, d = b - phi * (b - a), a + phi * (b - a)
    fc, fd = cost(c)[0], cost(d)[0]
    for _ in range(int(n_refine)):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = cost(c)[0]
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = cost(d)[0]
    psi = 0.5 * (a + b)
    rmse, err, aligned = cost(psi)
    per_tile = err.reshape(-1, 4).mean(axis=1)
    return {"psi": float(psi), "rmse": float(rmse), "max_err": float(err.max()),
            "per_tile": per_tile, "nodes": aligned.reshape(np.shape(target_nodes))}


def amplify_departure(ideal_nodes: Float[np.ndarray, "n_tiles 4 2"],
                      actual_nodes: Float[np.ndarray, "n_tiles 4 2"],
                      factor: float) -> Float[np.ndarray, "n_tiles 4 2"]:
    """Redraw a configuration with its departure from a reference scaled by ``factor``.

    Each panel's departure is decomposed into the rigid motion carrying its reference position onto
    its actual one, and the rotation and the centroid offset are scaled separately. Interpolating
    the corner coordinates instead would shrink every panel, by roughly ``(factor * angle)**2 / 2``
    of its side, so an exaggerated drawing would carry a second distortion that is nothing to do
    with the departure being shown.

    The result is a drawing, not a configuration the sheet can occupy: scaling the motions preserves
    every panel but breaks the pin closure between them. Report that gap -- ``factor`` times the
    true one -- alongside any figure that uses this.

    Args:
        ideal_nodes: (n_tiles, 4, 2) reference panel corners.
        actual_nodes: (n_tiles, 4, 2) measured panel corners, panel for panel.
        factor: 1 reproduces ``actual_nodes`` exactly when the panels are rigid; 0 gives the
            reference.

    Returns:
        (n_tiles, 4, 2) exaggerated corners, congruent panel by panel with ``ideal_nodes``.
    """
    ideal = np.asarray(ideal_nodes, dtype=float)
    actual = np.asarray(actual_nodes, dtype=float)
    if ideal.shape != actual.shape:
        raise ValueError(f"shape mismatch: ideal {ideal.shape} vs actual {actual.shape}")

    out = np.empty_like(ideal)
    for k in range(ideal.shape[0]):
        R, _ = _kabsch_2d(ideal[k], actual[k])
        angle = factor * float(np.arctan2(R[1, 0], R[0, 0]))
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        centre = ideal[k].mean(axis=0)
        local = ideal[k] - centre
        out[k] = np.column_stack([cos_a * local[:, 0] - sin_a * local[:, 1],
                                  sin_a * local[:, 0] + cos_a * local[:, 1]])
        out[k] += centre + factor * (actual[k].mean(axis=0) - centre)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Distribution shape
# ══════════════════════════════════════════════════════════════════════════════

def distribution_stats(values: Float[np.ndarray, "n"]) -> dict:
    """Concentration of a non-negative field: effective count, Gini, quartile spread."""
    v = np.abs(np.asarray(values, dtype=float))
    total = float(v.sum())
    n_eff = float(total ** 2 / float((v ** 2).sum())) if total > 0 else 0.0
    s = np.sort(v)
    n = s.size
    gini = (float((2.0 * np.arange(1, n + 1) - n - 1).dot(s)) / (n * total)) if total > 0 else 0.0
    q1, q3 = np.percentile(v, [25, 75])
    return {"n_eff": n_eff, "gini": float(gini), "iqr": float(q3 - q1),
            "max": float(v.max()), "min": float(v.min()), "mean": float(v.mean()),
            "top_decile_share": (float(np.sort(v)[-max(1, n // 10):].sum() / total)
                                 if total > 0 else 0.0)}
