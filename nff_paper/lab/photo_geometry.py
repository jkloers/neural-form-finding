"""Fitting known rigid geometry to a photographed specimen.

The sheet is nine rigid tiles, so a photograph of it is explained by nine planar poses. Rather than
segmenting tiles out of the image and hoping the segments are tiles, the KNOWN tile polygons are
posed until their edges lie on the tape -- which turns an image-processing problem into a
27-parameter fit whose parameters are exactly the simulation's own degrees of freedom.

Fitting happens in IMAGE space, not in a rectified raster. The pose parameters stay in world
millimetres and are projected through the homography for evaluation, so the mask is used at native
resolution and no resampling error enters the measurement. Rectified rasters are produced only for
display.

Why not a plain distance transform: it is exactly zero across the whole width of a tape strip, so a
~20 mm plateau carries no gradient -- the same size as the errors being measured. A Gaussian-blurred
mask has a gradient everywhere and a single maximum on the strip's centreline.

Nothing here imports from ``nff``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import minimize


# ── Fields sampled by the fit ─────────────────────────────────────────────────────────────────

def matched_field(mask: np.ndarray, sigma_px: float) -> np.ndarray:
    """Gaussian-blurred mask, normalised to a peak of 1.

    Smooth, bounded, maximal on the tape centreline, and with a basin whose width is set by
    ``sigma_px`` -- so a coarse-to-fine sequence turns a global search into a local one.
    """
    f = ndi.gaussian_filter(mask.astype(np.float32), sigma_px)
    peak = float(f.max())
    return f / peak if peak > 0 else f


def sample_field(field: np.ndarray, pts_px: np.ndarray, *, cval: float = 0.0) -> np.ndarray:
    """Bilinear sample of ``field`` at (N, 2) image-pixel points."""
    coords = np.stack([pts_px[:, 1], pts_px[:, 0]])
    return ndi.map_coordinates(field, coords, order=1, mode="constant", cval=cval)


def ridge_distance_mm(mask: np.ndarray, px_per_mm: float) -> np.ndarray:
    """Distance in millimetres to the nearest tape CENTRELINE.

    The interior distance transform peaks along the centre of a strip; thresholding its local
    maxima gives a one-pixel-wide skeleton whose exterior distance transform is a residual with
    units, which the matched-filter cost does not have.
    """
    d_in = ndi.distance_transform_edt(mask)
    ridge = (d_in >= ndi.maximum_filter(d_in, size=3)) & (d_in > 1.0)
    if not ridge.any():
        ridge = mask
    return ndi.distance_transform_edt(~ridge) / px_per_mm


# ── Polygons and poses ────────────────────────────────────────────────────────────────────────

def as_ccw(poly: np.ndarray) -> np.ndarray:
    """Polygon with counterclockwise winding, so the outward normal is unambiguous."""
    x, y = poly[:, 0], poly[:, 1]
    area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    return poly if area > 0 else poly[::-1]


def polygon_edge_samples(poly: np.ndarray, spacing: float = 5.0,
                         offset: float = 0.0) -> np.ndarray:
    """Points along a polygon's edges, optionally offset along the outward normal.

    Args:
        poly: (n, 2) polygon, any winding.
        spacing: sample spacing in the polygon's own units.
        offset: signed outward displacement; positive moves samples out of the polygon. This is
            the tape datum -- whether a strip sits on, inside or across the edge it marks.

    Returns:
        (N, 2) samples.
    """
    p = as_ccw(np.asarray(poly, float))
    out = []
    for a, b in zip(p, np.roll(p, -1, axis=0)):
        seg = b - a
        length = float(np.hypot(*seg))
        if length < 1e-9:
            continue
        n = np.array([seg[1], -seg[0]]) / length          # outward for CCW winding
        t = np.linspace(0.0, 1.0, max(2, int(round(length / spacing))), endpoint=False)
        out.append(a + t[:, None] * seg + offset * n)
    return np.concatenate(out) if out else np.zeros((0, 2))


def apply_pose(pts: np.ndarray, pose: Sequence[float], centre: np.ndarray) -> np.ndarray:
    """Rigid transform ``(dx, dy, dtheta_deg)`` applied about ``centre``."""
    dx, dy, dth = float(pose[0]), float(pose[1]), np.radians(float(pose[2]))
    c, s = np.cos(dth), np.sin(dth)
    q = np.asarray(pts, float) - centre
    return np.stack([c * q[:, 0] - s * q[:, 1], s * q[:, 0] + c * q[:, 1]], -1) + centre + [dx, dy]


def pose_matrix(pose: Sequence[float], centre: np.ndarray) -> np.ndarray:
    """The same transform as a 3x3 matrix, for composing with a homography."""
    dx, dy, dth = float(pose[0]), float(pose[1]), np.radians(float(pose[2]))
    c, s = np.cos(dth), np.sin(dth)
    R = np.array([[c, -s], [s, c]])
    t = centre + np.array([dx, dy]) - R @ centre
    return np.block([[R, t[:, None]], [np.zeros((1, 2)), np.ones((1, 1))]])


# ── The cost ──────────────────────────────────────────────────────────────────────────────────

@dataclass
class FitContext:
    """Everything a pose fit needs that does not change during it."""
    field: np.ndarray                 # matched filter on the tape mask, image space
    weight: np.ndarray                # 1 where the photograph carries evidence, 0 where it cannot
    to_image: np.ndarray              # world mm -> image px (the inverse homography)
    shape: tuple                      # image shape, for the in-bounds test

    def project(self, pts_mm: np.ndarray) -> np.ndarray:
        q = np.column_stack([pts_mm, np.ones(len(pts_mm))]) @ self.to_image.T
        return q[:, :2] / q[:, 2:3]

    def score(self, pts_mm: np.ndarray) -> float:
        """Mean field value along a set of world points; 1 is a perfect lie on the tape.

        The denominator is the FULL sample count, not the count that survived. Dividing by the
        surviving weight instead would let a pose raise its score by pushing part of the object out
        of frame or behind an occluder -- scoring the evidence it kept rather than the evidence it
        was asked to explain. Occlusion therefore caps a tile's attainable score, which is honest,
        and is reported as ``coverage``.
        """
        n = len(pts_mm)
        if n < 8:
            return 0.0
        px = self.project(pts_mm)
        inside = ((px[:, 0] >= 0) & (px[:, 0] < self.shape[1] - 1)
                  & (px[:, 1] >= 0) & (px[:, 1] < self.shape[0] - 1))
        if inside.sum() < 8:
            return 0.0
        px = px[inside]
        v = sample_field(self.field, px) * sample_field(self.weight, px)
        return float(v.sum() / n)

    def coverage(self, pts_mm: np.ndarray, frac: float = 0.25) -> float:
        """Fraction of samples that actually land on tape, as opposed to merely near it."""
        px = self.project(pts_mm)
        inside = ((px[:, 0] >= 0) & (px[:, 0] < self.shape[1] - 1)
                  & (px[:, 1] >= 0) & (px[:, 1] < self.shape[0] - 1))
        if inside.sum() < 8:
            return 0.0
        v = sample_field(self.field, px[inside])
        return float((v > frac * self.field.max()).mean())


def hinge_links(polys_mm: np.ndarray, face_pairs: np.ndarray) -> list:
    """Which corner of one tile meets which corner of its neighbour, per hinge.

    Read off the FLAT configuration, where the pairing is exact (the coincident corners are the
    hinge). Nearest-neighbour on a deployed configuration would occasionally pick a different pair
    once the tiles have rotated.

    Returns:
        list of ``(face_i, corner_a, face_j, corner_b)``.
    """
    out = []
    for i, j in np.asarray(face_pairs):
        d = np.linalg.norm(polys_mm[i][:, None, :] - polys_mm[j][None, :, :], axis=-1)
        a, b = np.unravel_index(int(np.argmin(d)), d.shape)
        out.append((int(i), int(a), int(j), int(b)))
    return out


def fit_pose(ctx: FitContext, samples_mm: np.ndarray, centre: np.ndarray,
             x0: Sequence[float] = (0.0, 0.0, 0.0), *,
             coarse: Optional[tuple] = None,
             penalty=None,
             bounds: tuple = (150.0, 25.0)) -> tuple[np.ndarray, float]:
    """Best rigid pose of one sample set against the tape field.

    Args:
        ctx: the image-space evidence.
        samples_mm: (N, 2) world points on the object's edges.
        centre: rotation centre in world millimetres.
        x0: initial ``(dx, dy, dtheta_deg)``.
        coarse: ``(dxy_mm, step_mm, dth_deg, step_deg)`` grid searched before the local solve;
            None skips it.
        bounds: ``(max_translation_mm, max_rotation_deg)`` about ``x0``.

    Returns:
        ``(pose, score)``.
    """
    def neg(p):
        v = -ctx.score(apply_pose(samples_mm, p, centre))
        return v + penalty(p) if penalty is not None else v

    best = np.asarray(x0, float)
    if coarse is not None:
        dxy, sxy, dth, sth = coarse
        grid_xy = np.arange(-dxy, dxy + 1e-9, sxy)
        grid_th = np.arange(-dth, dth + 1e-9, sth)
        best_val = neg(best)
        for gx in grid_xy:
            for gy in grid_xy:
                for gt in grid_th:
                    cand = np.array([x0[0] + gx, x0[1] + gy, x0[2] + gt])
                    val = neg(cand)
                    if val < best_val:
                        best_val, best = val, cand
    lo = np.array([x0[0] - bounds[0], x0[1] - bounds[0], x0[2] - bounds[1]])
    hi = np.array([x0[0] + bounds[0], x0[1] + bounds[0], x0[2] + bounds[1]])
    res = minimize(neg, best, method="Powell",
                   bounds=list(zip(lo, hi)), options=dict(xtol=1e-3, ftol=1e-6, maxiter=4000))
    return np.asarray(res.x, float), float(-res.fun)


RESTART_OFFSETS = ((0.0, 0.0, 0.0), (25.0, 0.0, 0.0), (-25.0, 0.0, 0.0),
                   (0.0, 25.0, 0.0), (0.0, -25.0, 0.0), (0.0, 0.0, 4.0), (0.0, 0.0, -4.0))


def fit_all_tiles(ctxs: Sequence[FitContext], polys_mm: np.ndarray, *,
                  offset: float = 0.0, spacing: float = 5.0,
                  x0: Optional[np.ndarray] = None,
                  coarse: Optional[tuple] = (60.0, 15.0, 8.0, 2.0),
                  bounds: tuple = (150.0, 25.0),
                  repair_frac: float = 0.5,
                  links: Optional[Sequence] = None, link_weight: float = 1.0,
                  link_scale: float = 60.0, link_sweeps: int = 3
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pose every tile independently, coarse-to-fine over a sequence of blur scales.

    A tile whose fit collapses is REPAIRED rather than reported. Powell occasionally falls into a
    local minimum on the tape's near-parallel lines: measured on the pull tile, the same tile scores
    0.716 from six of seven seeds and 0.068 from the seventh, so the failure is an order of magnitude
    away from the good answer and is trivially detectable. Left in, one such tile moves a
    whole-sheet descriptor by 120 mm.

    Args:
        ctxs: fit contexts ordered from the coarsest blur to the finest.
        polys_mm: (n_tiles, n_corners, 2) tile polygons in world millimetres.
        offset: tape datum, passed to :func:`polygon_edge_samples`.
        spacing: edge sample spacing, millimetres.
        x0: (n_tiles, 3) initial poses; zeros if omitted.
        coarse: grid search, applied at the coarsest scale only.
        bounds: pose bounds about ``x0``.
        repair_frac: refit any tile scoring below this fraction of the best tile in the sheet,
            from several offset restarts, keeping the best. 0 disables.
        links: ``(face_i, corner_a, face_j, corner_b)`` per hinge, from :func:`hinge_links`. Given,
            a connectivity sweep follows the independent fits and removes the sliding degeneracy.
        link_weight: strength of the hinge tie relative to the image score.
        link_scale: millimetres at which a hinge discrepancy costs ``link_weight``.
        link_sweeps: block-coordinate sweeps over the tiles.

    Returns:
        ``(poses (n_tiles, 3), scores (n_tiles,), coverage (n_tiles,))``.
    """
    n = len(polys_mm)
    poses = np.zeros((n, 3)) if x0 is None else np.asarray(x0, float).copy()
    scores = np.zeros(n)
    cover = np.zeros(n)
    starts = np.zeros((n, 3)) if x0 is None else np.asarray(x0, float).copy()

    def run(t, start):
        samples = polygon_edge_samples(polys_mm[t], spacing=spacing, offset=offset)
        centre = polys_mm[t].mean(0)
        p = np.asarray(start, float).copy()
        sc = 0.0
        for k, ctx in enumerate(ctxs):
            p, sc = fit_pose(ctx, samples, centre, p,
                             coarse=coarse if k == 0 else None, bounds=bounds)
        return p, sc, ctxs[-1].coverage(apply_pose(samples, p, centre))

    for t in range(n):
        poses[t], scores[t], cover[t] = run(t, starts[t])

    if repair_frac > 0 and n > 1:
        threshold = repair_frac * float(scores.max())
        for t in np.nonzero(scores < threshold)[0]:
            for off in RESTART_OFFSETS[1:]:
                p, sc, cv = run(t, starts[t] + np.asarray(off))
                if sc > scores[t]:
                    poses[t], scores[t], cover[t] = p, sc, cv

    if links:
        poses, scores, cover = _tie_hinges(ctxs, polys_mm, poses, links, offset, spacing,
                                           bounds, link_weight, link_scale, link_sweeps)
    return poses, scores, cover


def _tie_hinges(ctxs, polys_mm, poses, links, offset, spacing, bounds,
                weight, scale, sweeps):
    """Re-fit each tile against the tape AND against where its neighbours put their shared hinges.

    Nine tiles fitted independently have a degeneracy the sheet does not: a long thin tile can slide
    along its own tape and score almost as well, and one measured here slid 122 mm that way. The
    sheet is a mechanism, so the corners that meet at a hinge must keep meeting. Penalising the
    CHANGE in each hinge's corner offset (rather than driving it to zero) works whether the model
    has the corners coincident, as when flat, or a millimetre or two apart, as when deployed.

    Block coordinate descent: every tile is re-solved against its neighbours' current poses, a few
    sweeps. Cheap, and it converges because each sweep can only lower the objective.
    """
    n = len(polys_mm)
    centres = np.array([p.mean(0) for p in polys_mm])
    by_tile = {t: [] for t in range(n)}
    for (i, a, j, b) in links:
        by_tile[i].append((a, j, b))
        by_tile[j].append((b, i, a))

    def node(t, corner, pose):
        return apply_pose(polys_mm[t][corner][None, :], pose, centres[t])[0]

    for _ in range(sweeps):
        for t in range(n):
            mine = by_tile[t]
            if not mine:
                continue
            targets = [(a, node(j, b, poses[j]) + (polys_mm[t][a] - polys_mm[j][b]))
                       for (a, j, b) in mine]

            def pen(p, t=t, targets=targets):
                e = [node(t, a, p) - x for a, x in targets]
                return weight * float(np.mean(np.sum(np.square(e), axis=-1))) / (scale ** 2)

            samples = polygon_edge_samples(polys_mm[t], spacing=spacing, offset=offset)
            p = poses[t].copy()
            for ctx in ctxs[-2:]:
                p, _ = fit_pose(ctx, samples, centres[t], p, coarse=None,
                                penalty=pen, bounds=bounds)
            poses[t] = p

    scores = np.zeros(n)
    cover = np.zeros(n)
    for t in range(n):
        samples = polygon_edge_samples(polys_mm[t], spacing=spacing, offset=offset)
        moved = apply_pose(samples, poses[t], centres[t])
        scores[t] = ctxs[-1].score(moved)
        cover[t] = ctxs[-1].coverage(moved)
    return poses, scores, cover


# ── Point-based estimation from the hinge markers ─────────────────────────────────────────────

def blob_centroids(mask: np.ndarray, *, min_px: int = 150) -> np.ndarray:
    """(N, 2) image-pixel centroids of the connected components of a marker mask."""
    lab, n = ndi.label(mask)
    if n == 0:
        return np.zeros((0, 2))
    counts = np.bincount(lab.ravel(), minlength=n + 1)
    keep = [i for i in range(1, n + 1) if counts[i] >= min_px]
    if not keep:
        return np.zeros((0, 2))
    cy_cx = ndi.center_of_mass(mask, lab, keep)
    return np.array([[c, r] for r, c in cy_cx])


def solve_poses_from_points(pred_mm: np.ndarray, obs_mm: np.ndarray, owner: np.ndarray,
                            centres_mm: np.ndarray, n_tiles: int) -> np.ndarray:
    """Per-tile rigid poses from marker correspondences, linearised about zero rotation.

    Each observation contributes two rows in ``dx, dy, dtheta`` for its owning tile. A tile with
    fewer than two markers is left at zero rather than being fitted from an underdetermined system.

    Args:
        pred_mm: (N, 2) predicted marker positions.
        obs_mm: (N, 2) observed marker positions.
        owner: (N,) tile index per marker.
        centres_mm: (n_tiles, 2) rotation centres.
        n_tiles: tile count.

    Returns:
        (n_tiles, 3) poses, ``dtheta`` in degrees.
    """
    poses = np.zeros((n_tiles, 3))
    for t in range(n_tiles):
        sel = owner == t
        if sel.sum() < 2:
            continue
        r = pred_mm[sel] - centres_mm[t]
        d = obs_mm[sel] - pred_mm[sel]
        # [1 0 -ry; 0 1 rx] @ [dx dy dtheta]^T = d
        A = np.zeros((2 * sel.sum(), 3))
        A[0::2, 0] = 1.0
        A[1::2, 1] = 1.0
        A[0::2, 2] = -r[:, 1]
        A[1::2, 2] = r[:, 0]
        sol, *_ = np.linalg.lstsq(A, d.ravel(), rcond=None)
        poses[t] = [sol[0], sol[1], np.degrees(sol[2])]
    return poses


# ── Comparison ────────────────────────────────────────────────────────────────────────────────

def kabsch_2d(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Rigid transform carrying ``P`` onto ``Q``, without scale.

    Scale is metric here and must not be absorbed: a fit that rescales would hide exactly the
    systematic error the calibration exists to measure.

    Returns:
        ``(R (2, 2), t (2,), rms)``.
    """
    P, Q = np.asarray(P, float), np.asarray(Q, float)
    mp, mq = P.mean(0), Q.mean(0)
    H = (P - mp).T @ (Q - mq)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, d]) @ U.T
    t = mq - R @ mp
    resid = (P @ R.T + t) - Q
    return R, t, float(np.sqrt(np.mean(np.sum(resid ** 2, axis=1))))


def tile_rotations(flat: np.ndarray, deployed: np.ndarray) -> np.ndarray:
    """Signed best-fit rotation carrying each flat tile onto its deployed tile, in degrees."""
    P = flat - flat.mean(-2, keepdims=True)
    Q = deployed - deployed.mean(-2, keepdims=True)
    H = np.einsum("...ki,...kj->...ij", P, Q)
    return np.degrees(np.arctan2(H[..., 0, 1] - H[..., 1, 0], H[..., 0, 0] + H[..., 1, 1]))


def hinge_opening(rot_deg: np.ndarray, face_pairs: np.ndarray) -> np.ndarray:
    """Relative rotation across every hinge, wrapped into [0, 180)."""
    d = np.abs(rot_deg[..., face_pairs[:, 0]] - rot_deg[..., face_pairs[:, 1]]) % 360.0
    return np.minimum(d, 360.0 - d)


def regularise_rectangle(corners: np.ndarray) -> np.ndarray:
    """The rectangle closest, corner for corner, to a quadrilateral that ought to be one.

    A taped rectangle reaches a design frame through a homography fitted on something ELSE, and it
    arrives as a quadrilateral: the residual is whatever the anchor object and the tape do not
    agree about. Here that is a couple of degrees, and none of it is removable -- squaring the tape
    un-squares the anchor, which is a rectangle of known size and the metric standard.

    So the quadrilateral is kept as the measurement and this is used only where the shape is DRAWN,
    on a plate whose reader has the panel border to compare against and no way to know that a
    hand-taped floor is not square. It is a five-parameter least squares -- centre, two sides, one
    angle -- so the answer is the rectangle, not a rectangle that happens to touch some corners.

    ⛔ Never feed the result back into a measurement, a mask or a homography. It is not the tape.

    Args:
        corners: (4, 2) quadrilateral, in order around the shape.

    Returns:
        (4, 2) rectangle, in the same order and winding.
    """
    from scipy.optimize import least_squares
    q = np.asarray(corners, float)
    local = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])
    u, v = q[1] - q[0], q[2] - q[1]
    if u[0] * v[1] - u[1] * v[0] < 0:                   # match the input's winding
        local = local[::-1]
    e = np.linalg.norm(np.roll(q, -1, axis=0) - q, axis=1)

    def build(p):
        c, s = np.cos(p[4]), np.sin(p[4])
        return p[:2] + (local * p[2:4]) @ np.array([[c, -s], [s, c]]).T

    a = np.arctan2(*(q[1] - q[0])[::-1])
    r = least_squares(lambda p: (build(p) - q).ravel(),
                      np.r_[q.mean(0), 0.5 * (e[0] + e[2]), 0.5 * (e[1] + e[3]), a])
    return build(r.x)


def min_area_rect(points: np.ndarray) -> tuple[float, float, float]:
    """``(short, long, angle_deg)`` of the minimum-area rectangle enclosing a point cloud."""
    from scipy.spatial import ConvexHull
    hull = points[ConvexHull(points).vertices]
    edges = np.roll(hull, -1, axis=0) - hull
    best = None
    for a in np.arctan2(edges[:, 1], edges[:, 0]):
        c, s = np.cos(-a), np.sin(-a)
        rot = hull @ np.array([[c, -s], [s, c]]).T
        w, h = np.ptp(rot[:, 0]), np.ptp(rot[:, 1])
        if best is None or w * h < best[0] * best[1]:
            best = (w, h, np.degrees(a))
    w, h, a = best
    return (min(w, h), max(w, h), a)
