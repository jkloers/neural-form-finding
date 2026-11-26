"""Out-of-plane signature of a taped ligament, read from one oblique photograph.

The deployment series in :mod:`nff_paper.lab.photo_rectify` measures a sheet that is assumed
PLANAR: one homography per frame, tiles posed as rigid polygons on a floor. That machinery is blind
to the one thing a close-up of a single hinge shows, which is the ligament leaving the plane.

What a single uncalibrated frame can and cannot give:

**It can give the apparent-width profile.** A tape strip of constant true width, imaged along its
length, has an apparent width set by two things: the perspective factor ``1/Z``, and the angle its
surface makes with the viewing ray. Both are smooth along a strip lying flat on a floor, so the
flat-lying spans at the two ends fix a baseline, and the departure from that baseline where the
strip lifts is a direct observable. It needs no camera calibration, no scale, and no assumption
about the buckle's shape.

**It can give the chord bow.** A strip lying flat on a plane images to a straight line, because a
homography maps lines to lines. The observed centreline's departure from the straight chord joining
its two flat ends is therefore entirely due to material leaving that plane, up to the strip's own
in-plane curvature. Reported normalised by the strip length, it is again assumption-free.

**It cannot give an absolute millimetre amplitude without a ruler and a camera.** Both are supplied
from outside: a known length in frame (the ligament width, or a relief-hole diameter from the cut
package) and an estimate of the camera's elevation above the floor. Both enter
:func:`amplitude_estimate`, which returns a band rather than a number, and the band is wide. Quote
the dimensionless observables first and the millimetres second.

⚠ The two observables answer different questions and must not be conflated. A strip that BENDS out
of plane about its transverse axis keeps its apparent width and bows its centreline. A strip that
TWISTS about its own tangent turns its face toward the camera and changes its apparent width while
its centreline stays put. A buckled ligament does both, and separating them is the point.

Nothing here imports from ``nff``: numpy, scipy and PIL only, matching
:mod:`nff_paper.lab.photo_rectify`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import ndimage as ndi


# ── The strip ─────────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Ribbon:
    """A tape strip resolved along its own length, in pixels.

    Attributes:
        xy: (n, 2) centreline samples, ordered along the strip.
        width_px: (n,) apparent width PERPENDICULAR to the local centreline.
        s_px: (n,) cumulative arc length along the centreline, starting at 0.
        mask: (H, W) bool, the strip the samples came from.
    """
    xy: np.ndarray
    width_px: np.ndarray
    s_px: np.ndarray
    mask: np.ndarray

    @property
    def length_px(self) -> float:
        return float(self.s_px[-1])

    @property
    def axis(self) -> np.ndarray:
        """Unit vector from the first centreline sample to the last."""
        d = self.xy[-1] - self.xy[0]
        return d / (np.linalg.norm(d) + 1e-12)


def largest_component(mask: np.ndarray, *, fill: bool = True) -> np.ndarray:
    """The biggest connected piece of ``mask``, holes filled.

    Filling matters here rather than being tidiness: hardware threaded through the ligament punches
    a hole straight through the strip, and an unfilled hole splits the distance transform's ridge in
    two exactly where the buckle crest is.
    """
    lab, n = ndi.label(mask)
    if n == 0:
        return mask
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    out = lab == int(counts.argmax())
    return ndi.binary_fill_holes(out) if fill else out


def ribbon_from_mask(mask: np.ndarray, *, n_bins: int = 90, min_px: int = 20,
                     trim_frac: float = 0.03) -> Ribbon:
    """Resolve a strip-shaped mask into a centreline and an apparent-width profile.

    The width is taken from the Euclidean distance transform rather than by scanning image rows or
    columns. A row scan measures the strip's vertical extent, which is the true width divided by the
    cosine of the angle the strip happens to make with the image axis, so a strip that curves in the
    image acquires a spurious width bulge exactly where it turns. The distance transform's ridge
    value is half the width measured perpendicular to the boundary, which is the quantity wanted and
    carries no dependence on how the strip is oriented.

    Args:
        mask: (H, W) bool, one strip.
        n_bins: samples along the strip.
        min_px: bins holding fewer pixels than this are dropped.
        trim_frac: fraction of the length discarded at EACH end. The end caps pull the distance
            transform down, so the last samples read narrow whatever the surface is doing.

    Returns:
        The :class:`Ribbon`.
    """
    m = np.asarray(mask, bool)
    edt = ndi.distance_transform_edt(m)
    ys, xs = np.nonzero(m)
    pts = np.stack([xs, ys], axis=1).astype(float)

    # Order the strip by its own principal axis. A strip is long and thin, so the first singular
    # vector is its length direction; using it as the bin coordinate is stable even where the strip
    # curves, because the binning only has to be monotone along the strip, not exact.
    centre = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - centre, full_matrices=False)
    axis = vt[0]
    t = (pts - centre) @ axis

    lo, hi = t.min(), t.max()
    span = hi - lo
    lo, hi = lo + trim_frac * span, hi - trim_frac * span
    edges = np.linspace(lo, hi, n_bins + 1)

    rows = []
    for i in range(n_bins):
        sel = (t >= edges[i]) & (t < edges[i + 1])
        if int(sel.sum()) < min_px:
            continue
        e = edt[ys[sel], xs[sel]]
        j = int(np.argmax(e))
        rows.append((xs[sel][j], ys[sel][j], 2.0 * float(e[j])))
    if len(rows) < 4:
        raise ValueError("strip too small to resolve: fewer than 4 usable bins")

    arr = np.asarray(rows, float)
    xy, width = arr[:, :2], arr[:, 2]
    # The ridge wanders by a pixel or two between bins; smoothing it keeps the arc length from
    # accumulating that jitter as spurious length.
    xy = np.stack([ndi.uniform_filter1d(xy[:, k], 5, mode="nearest") for k in range(2)], axis=1)
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    return Ribbon(xy=xy, width_px=width, s_px=s, mask=m)


# ── Observables that need nothing from outside ────────────────────────────────────────────────

def flat_width_baseline(rib: Ribbon, *, ends_frac: float = 0.18) -> np.ndarray:
    """Apparent width the strip WOULD have along its length if it stayed in the floor plane.

    Under a homography the image scale of a planar object varies as ``1 / (a x + b y + c)``, so the
    RECIPROCAL of the apparent width is linear along any line in that plane. Fitting ``1 / w``
    linearly against arc length on the two end spans, where the strip is known to be lying flat,
    therefore extrapolates the flat baseline across the lifted middle with one straight line and no
    free shape.

    Args:
        rib: the strip.
        ends_frac: fraction of the length at EACH end taken as lying flat.

    Returns:
        (n,) baseline width in pixels.
    """
    s = rib.s_px / (rib.length_px + 1e-12)
    ends = (s <= ends_frac) | (s >= 1.0 - ends_frac)
    if int(ends.sum()) < 4:
        raise ValueError("not enough samples in the end spans to fit a baseline")
    coef = np.polyfit(rib.s_px[ends], 1.0 / rib.width_px[ends], 1)
    inv = np.polyval(coef, rib.s_px)
    return 1.0 / np.maximum(inv, 1e-6)


def face_turn(rib: Ribbon, baseline: np.ndarray) -> np.ndarray:
    """Angle by which the strip's face turns toward the camera, in radians, along its length.

    A strip lying flat presents its surface to the camera at some fixed obliquity and images at the
    baseline width. When it twists about its own tangent it turns its face toward the viewing ray
    and images WIDER, saturating at its true width when the face is square to the ray. Taking the
    baseline as the reference,

        cos(turn) = baseline / width       for width >= baseline

    which is zero wherever the strip images no wider than flat. The sign is not recoverable from one
    view: a face turning away images narrower than baseline and is reported as zero turn, not as a
    negative one.

    ⚠ This is a turn RELATIVE TO THE FLAT-LYING ATTITUDE, not an angle from the floor. Converting to
    a floor-referenced tilt needs the camera elevation, which one frame does not carry.
    """
    ratio = np.clip(baseline / np.maximum(rib.width_px, 1e-6), 0.0, 1.0)
    return np.arccos(ratio)


def chord_bow(rib: Ribbon) -> tuple[np.ndarray, float]:
    """Signed perpendicular offset of the centreline from its own end-to-end chord, in pixels.

    A strip lying in a plane images to a straight line, because a homography maps lines to lines. So
    for a strip whose two ends are flat on the floor, every pixel of departure from the chord is
    material that left the floor, up to the strip's own in-plane curvature.

    Returns:
        ``(offset, peak)`` with ``offset`` (n,) signed and ``peak`` the largest absolute value.
    """
    a, b = rib.xy[0], rib.xy[-1]
    d = b - a
    d = d / (np.linalg.norm(d) + 1e-12)
    nrm = np.array([-d[1], d[0]])
    off = (rib.xy - a) @ nrm
    return off, float(np.abs(off).max())


# ── Millimetres, and what they cost ───────────────────────────────────────────────────────────

def scale_from_width(rib: Ribbon, true_width_mm: float, *, ends_frac: float = 0.18) -> float:
    """Millimetres per pixel on the floor, from the strip's own known true width.

    Read on the FLAT end spans only. There the strip lies in the floor plane, so its apparent width
    is the floor's own image scale in the transverse direction, which is the scale the chord bow
    needs. Reading it at the crest instead would return the scale of a surface that is neither the
    floor nor at the floor's depth.

    ⚠ This assumes the taped strip's true width. Cross-check it: if the strip runs from one hinge to
    the next, the length it implies must come out at the panel pitch.
    """
    s = rib.s_px / (rib.length_px + 1e-12)
    ends = (s <= ends_frac) | (s >= 1.0 - ends_frac)
    return float(true_width_mm / np.median(rib.width_px[ends]))


def amplitude_estimate(rib: Ribbon, baseline: np.ndarray, mm_per_px: float,
                       *, camera_elev_deg: float = 45.0,
                       elev_band_deg: float = 15.0) -> dict:
    """Out-of-plane amplitude in millimetres, with the band its assumptions earn.

    The amplitude comes from the CHORD BOW alone. A point sitting a height ``h`` above the floor
    images displaced from its own floor footprint along the picture's vertical direction, and a
    vertical segment viewed from an elevation ``eps`` above the floor plane images foreshortened by
    ``cos(eps)``. So ``h = bow_px * mm_per_px / cos(eps)``, and the elevation is the one thing a
    single frame does not carry, which is why the result is a band.

    ⚠ THE FACE TURN IS NOT A SECOND ROUTE TO THE AMPLITUDE, and an earlier version of this function
    wrongly integrated it as one. The turn is a rotation of the strip about its OWN TANGENT, so
    ``sin(turn)`` is not the tangent's inclination and its integral along the strip is not a height.
    Integrating it here returned 173 mm on a strip whose bow says 25 mm. The turn is reported as
    what it is: an independent observable measuring twist, not rise.

    Args:
        rib: the strip.
        baseline: flat-lying width baseline, from :func:`flat_width_baseline`.
        mm_per_px: floor scale, from :func:`scale_from_width` or a relief-hole diameter.
        camera_elev_deg: camera elevation above the floor plane.
        elev_band_deg: half-width of the elevation uncertainty, propagated into the band.

    Returns:
        ``{"bow_mm", "lo_mm", "hi_mm", "turn_deg_max", "bow_px", "bow_frac"}``.
    """
    _, bow_px = chord_bow(rib)
    turn = face_turn(rib, baseline)

    def _bow(elev_deg: float) -> float:
        return float(bow_px * mm_per_px / max(np.cos(np.radians(elev_deg)), 1e-3))

    lo_e = max(camera_elev_deg - elev_band_deg, 0.0)
    hi_e = min(camera_elev_deg + elev_band_deg, 85.0)
    return dict(bow_mm=_bow(camera_elev_deg), lo_mm=_bow(lo_e), hi_mm=_bow(hi_e),
                turn_deg_max=float(np.degrees(turn.max())), bow_px=bow_px,
                bow_frac=float(bow_px / rib.length_px))


# ── The model's own strip, for a like-for-like comparison ─────────────────────────────────────

def band_indicator(xyz: np.ndarray, *, half_width_mm: float, along: str = "x") -> np.ndarray:
    """1 on the band of the RVE the tape covers, 0 elsewhere, as a per-node field.

    The RVE frame puts the secondary cut on ``y = 0`` with material below it, so the tape laid along
    that cut is the band ``-2 * half_width <= y <= 0``. Passing this as the render's scalar field
    paints the taped strip a distinct colour, after which the SAME strip reader used on the
    photograph can be run on the render and the two apparent-width profiles compared directly, with
    no camera model in between.
    """
    if along != "x":
        raise ValueError("only the secondary-cut band (along x) is defined")
    y = np.asarray(xyz, float)[:, 1]
    return ((y <= 0.0) & (y >= -2.0 * half_width_mm)).astype(float)


def height_profile_3d(xyz: np.ndarray, disp: np.ndarray, *, half_width_mm: float,
                      thickness_mm: float = 0.5, n_bins: int = 70,
                      edge_frac: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    """``(x_mm, uz_mm)`` along the cut edge the tape's centreline lies over.

    The model's counterpart to the photograph's chord bow. Three choices make it the SAME quantity
    the photograph measures rather than merely a similar one.

    ⚠ **Read the material nearest the CUT, not the whole taped band.** The tape spans the secondary
    cut, so its centreline sits on ``y = 0``, and that free edge is exactly where the buckle is
    largest: the crest reaches 24.2 mm there against 19.4 mm on the band's mid-line. Averaging over
    the full band width therefore under-reads the model's own peak by a quarter and makes the model
    look flatter than it is. ``edge_frac`` is the fraction of the band, measured from the cut, that
    is read.

    Only the TOP CAP is taken, because it is the surface the render draws and the tape sticks to.
    And each bin is reduced by its MEDIAN rather than its extreme, because an extreme picks a
    different node in every bin and the profile inherits that jitter as a saw-tooth.
    """
    ref = np.asarray(xyz, float)
    P = ref + np.asarray(disp, float)[:, :3]
    band = (ref[:, 1] <= 0.0) & (ref[:, 1] >= -2.0 * half_width_mm * edge_frac)
    cap = ref[:, 2] > 0.5 * thickness_mm
    sel0 = band & cap
    if int(sel0.sum()) < n_bins:
        sel0 = band                                   # a coarse mesh may have no separate top cap
    x, z = P[sel0, 0], P[sel0, 2]
    if x.size < n_bins:
        raise ValueError("band holds too few nodes")
    edges = np.linspace(x.min(), x.max(), n_bins + 1)
    idx = np.clip(np.digitize(x, edges) - 1, 0, n_bins - 1)
    xs, zs = [], []
    for i in range(n_bins):
        sel = idx == i
        if int(sel.sum()) < 3:
            continue
        xs.append(0.5 * (edges[i] + edges[i + 1]))
        zs.append(float(np.median(z[sel])))
    return np.asarray(xs), np.asarray(zs)


def planarity(x_mm: np.ndarray, z_mm: np.ndarray) -> float:
    """RMS departure of a height profile from its own straight chord, over the peak, dimensionless.

    Small means the buckle is a single planar arch, which is the assumption the photograph's chord
    bow falls back on. Large means the strip is doing something the one-view reconstruction cannot
    represent, and the millimetre numbers should be quoted with that said.
    """
    chord = np.interp(x_mm, [x_mm[0], x_mm[-1]], [z_mm[0], z_mm[-1]])
    resid = z_mm - chord
    peak = float(np.abs(resid).max())
    return float(np.sqrt(np.mean((resid - resid.mean()) ** 2)) / (peak + 1e-12))


def bow_height_profile(rib: Ribbon, mm_per_px: float, *, camera_elev_deg: float = 45.0,
                       centre_on_peak: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """``(s_mm, uz_mm)`` along the photographed strip, from its chord bow.

    The same conversion :func:`amplitude_estimate` applies at the peak, applied at every sample so
    the measured SHAPE can be laid over the model's rather than only its height. Centred on the
    crest by default, because the strip's own origin in the picture is arbitrary while the model's
    ``x = 0`` is the ligament.
    """
    off, _ = chord_bow(rib)
    uz = np.abs(off) * mm_per_px / max(np.cos(np.radians(camera_elev_deg)), 1e-3)
    s = rib.s_px * mm_per_px
    if centre_on_peak:
        s = s - s[int(np.argmax(uz))]
    return s, uz


# ── The relief hole: a circle the photograph and the model SHARE ──────────────────────────────

@dataclass(frozen=True)
class Ellipse:
    """An imaged circle. ``angle`` is the major axis, in radians, measured from +x toward +y."""
    cx: float
    cy: float
    major: float          # full length of the major axis, in pixels
    minor: float
    angle: float

    @property
    def ratio(self) -> float:
        return float(self.minor / max(self.major, 1e-9))

    @property
    def tilt_deg(self) -> float:
        """Angle between the viewing ray and the circle's own normal, in degrees.

        A circle seen down its normal images as a circle (ratio 1, tilt 0); seen edge-on it images
        as a line (ratio 0, tilt 90).
        """
        return float(np.degrees(np.arccos(np.clip(self.ratio, 0.0, 1.0))))

    def mm_per_px(self, true_diameter_mm: float) -> float:
        """Scale at the circle's own depth.

        ⚠ Read off the MAJOR axis, which is the only diameter tilt does not foreshorten: a circle
        tilted about any axis keeps its extent along that axis. So this scale is free of the
        orientation error that dogs every other ruler in these frames.
        """
        return float(true_diameter_mm / max(self.major, 1e-9))


def fit_ellipse(pts: np.ndarray) -> Ellipse:
    """Direct least-squares ellipse through a point cloud (Halir and Flusser).

    The numerically stable form of Fitzgibbon's method: the design matrix is split into its
    quadratic and linear halves so the constrained eigenproblem is solved on a 3x3 rather than a
    near-singular 6x6, which matters here because the rim spans only a few hundred pixels.
    """
    p = np.asarray(pts, float)
    if len(p) < 6:
        raise ValueError("need at least 6 points to fit an ellipse")
    # Normalise: a conic fit is not scale invariant and a rim sitting at (2000, 1400) in pixel
    # coordinates otherwise conditions the eigenproblem badly.
    mu = p.mean(axis=0)
    s = float(np.sqrt(((p - mu) ** 2).sum(axis=1)).mean()) or 1.0
    x, y = ((p - mu) / s).T

    D1 = np.column_stack([x * x, x * y, y * y])
    D2 = np.column_stack([x, y, np.ones_like(x)])
    S1, S2, S3 = D1.T @ D1, D1.T @ D2, D2.T @ D2
    T = -np.linalg.solve(S3, S2.T)
    M = S1 + S2 @ T
    C = np.array([[0.0, 0.0, 2.0], [0.0, -1.0, 0.0], [2.0, 0.0, 0.0]])
    _, vec = np.linalg.eig(np.linalg.solve(C, M))
    cond = 4.0 * vec[0] * vec[2] - vec[1] ** 2          # the ellipse branch of the eigenvectors
    k = int(np.argmax(np.where(cond > 0, cond, -np.inf)))
    if not np.isfinite(cond[k]) or cond[k] <= 0:
        raise ValueError("no elliptical solution: the points are not on a conic arc")
    a1 = np.real(vec[:, k])
    a, b, c = a1
    d, e, f = np.real(T @ a1)

    den = b * b - 4 * a * c
    if abs(den) < 1e-12:
        raise ValueError("degenerate conic")
    cx = (2 * c * d - b * e) / den
    cy = (2 * a * e - b * d) / den

    # Axes and orientation from the QUADRATIC FORM's eigenvectors rather than from the closed-form
    # arctan. The closed form needs a branch to say which axis the angle belongs to, and getting
    # that branch wrong reports the minor axis' direction as the major's -- a silent 90 deg error
    # that leaves the axis LENGTHS correct, so it survives any test that only checks those.
    Q = np.array([[a, 0.5 * b], [0.5 * b, c]])
    F = a * cx * cx + b * cx * cy + c * cy * cy + d * cx + e * cy + f
    lam, vec = np.linalg.eigh(Q)
    if np.any(lam == 0) or -F / lam[0] < 0 or -F / lam[1] < 0:
        raise ValueError("degenerate conic")
    semi = np.sqrt(-F / lam)                    # semi-axis along each eigenvector
    k = int(np.argmax(semi))                    # the major axis is the LARGER semi-axis
    ang = np.arctan2(vec[1, k], vec[0, k])
    return Ellipse(cx=float(cx * s + mu[0]), cy=float(cy * s + mu[1]),
                   major=float(2 * semi[k] * s), minor=float(2 * semi[1 - k] * s),
                   angle=float(np.mod(ang, np.pi)))


def rim_ridge_points(gray: np.ndarray, centre: tuple[float, float], r_guess: float,
                     *, n_rays: int = 240, band: tuple[float, float] = (0.35, 1.6),
                     smooth: float = 1.5, rel: float = 0.45) -> np.ndarray:
    """Points on a roughly circular rim, found by walking rays out from a seed.

    Along each ray the rim shows as a gradient peak, which is a far more selective test than
    thresholding the image: the hole's edge is a step in brightness whatever happens to lie behind
    it, and through a transparent sheet that background changes around the circle.

    ⚠ Take the FIRST peak outward, not the strongest. The hole is see-through, so its interior is
    plain floor and the rim is the innermost edge on every ray. The strongest edge is often
    something else entirely: beside this hole sits a torn tab whose fold catches the light harder
    than the rim does, and taking the maximum pulled the fitted ellipse bodily onto it.

    Returns the (n, 2) hit points; callers should still reject outliers, because a ray running down
    the cut meets no rim at all before it meets the cut's own edge.
    """
    g = ndi.gaussian_filter(np.asarray(gray, float), smooth)
    gy, gx = np.gradient(g)
    mag = np.hypot(gx, gy)
    cx, cy = centre
    rr = np.arange(band[0] * r_guess, band[1] * r_guess, 0.5)
    out = []
    for th in np.linspace(0, 2 * np.pi, n_rays, endpoint=False):
        xs, ys = cx + rr * np.cos(th), cy + rr * np.sin(th)
        ok = (xs >= 0) & (ys >= 0) & (xs < g.shape[1] - 1) & (ys < g.shape[0] - 1)
        if ok.sum() < 5:
            continue
        prof = ndi.map_coordinates(mag, [ys[ok], xs[ok]], order=1)
        thr = rel * prof.max()
        above = prof >= thr
        if not above.any():
            continue
        start = int(np.argmax(above))
        end = start + int(np.argmin(above[start:])) if not above[start:].all() else len(prof)
        j = start + int(np.argmax(prof[start:end]))          # crest of the FIRST qualifying peak
        out.append((xs[ok][j], ys[ok][j]))
    return np.asarray(out, float)


def fit_hole(gray: np.ndarray, centre: tuple[float, float], r_guess: float,
             *, n_iter: int = 4, keep: float = 0.72) -> Ellipse:
    """Ellipse of the relief hole, refined from a coarse seed.

    Iterated because the seed is placed by eye: each pass re-centres on the current fit, re-walks
    the rays, and drops the worst-fitting third of the hits before refitting. The rejected ones are
    real features, not noise -- the cut's edge and the torn tab beside the hole both produce a
    stronger gradient than the rim does on the rays that cross them.
    """
    c = (float(centre[0]), float(centre[1]))
    r = float(r_guess)
    ell = None
    for _ in range(n_iter):
        pts = rim_ridge_points(gray, c, r)
        if len(pts) < 12:
            raise ValueError("too few rim points; check the seed")
        if ell is not None:
            pts = pts[_ellipse_residual(ell, pts) <
                      np.quantile(_ellipse_residual(ell, pts), keep)]
        ell = fit_ellipse(pts)
        c, r = (ell.cx, ell.cy), 0.5 * ell.major
    return ell


def radial_dark_peaks(gray: np.ndarray, centre: tuple[float, float], r_px: float,
                      *, n_peaks: int = 2, band: tuple[float, float] = (1.4, 5.0),
                      sector: tuple[float, float] | None = None,
                      sep_deg: float = 20.0) -> list[tuple[float, float]]:
    """Directions of the dark rays leaving a point, as ``(angle_rad, score)`` strongest first.

    The two edges of the opened main cut radiate from the relief hole, so averaging a local darkness
    measure along rays and reading off the angular peaks finds them without fitting any lines. That
    matters because the cut is thin, broken by specular highlights, and curves slightly: a line fit
    needs a clean inlier set, whereas an angular histogram only needs the cut to be darker than the
    sheet on average, which it is everywhere.

    Args:
        centre: the hole centre, in pixels.
        r_px: the hole's radius; the band is measured in multiples of it.
        sector: optional ``(lo, hi)`` in radians restricting the search, for when only one side of
            the hole carries the feature wanted.
        sep_deg: minimum angular separation between reported peaks.
    """
    g = np.asarray(gray, float)
    gs = ndi.gaussian_filter(g, 2.0)
    dark = np.clip(ndi.uniform_filter(gs, max(int(2 * r_px), 3)) - gs, 0.0, None)
    th = np.linspace(0.0, 2.0 * np.pi, 720, endpoint=False)
    rr = np.arange(band[0] * r_px, band[1] * r_px, 2.0)
    cx, cy = centre
    prof = np.zeros(len(th))
    for i, t in enumerate(th):
        xs, ys = cx + rr * np.cos(t), cy + rr * np.sin(t)
        ok = (xs >= 0) & (ys >= 0) & (xs < g.shape[1] - 1) & (ys < g.shape[0] - 1)
        if ok.sum() > 5:
            prof[i] = ndi.map_coordinates(dark, [ys[ok], xs[ok]], order=1).mean()
    prof = ndi.gaussian_filter1d(prof, 4, mode="wrap")
    mask = np.ones_like(prof, bool)
    if sector is not None:
        a = np.mod(th - sector[0], 2 * np.pi)
        mask = a <= np.mod(sector[1] - sector[0], 2 * np.pi)
    work = np.where(mask, prof, -np.inf)
    out = []
    half = int(round(sep_deg / 360.0 * len(th)))
    for _ in range(n_peaks):
        i = int(np.argmax(work))
        if not np.isfinite(work[i]):
            break
        out.append((float(th[i]), float(prof[i])))
        work[np.arange(i - half, i + half + 1) % len(work)] = -np.inf
    return out


def _ellipse_residual(e: Ellipse, pts: np.ndarray) -> np.ndarray:
    """Radial distance from each point to the ellipse, in units of its own radius."""
    ca, sa = np.cos(-e.angle), np.sin(-e.angle)
    dx, dy = pts[:, 0] - e.cx, pts[:, 1] - e.cy
    u = (dx * ca - dy * sa) / max(0.5 * e.major, 1e-9)
    v = (dx * sa + dy * ca) / max(0.5 * e.minor, 1e-9)
    return np.abs(np.hypot(u, v) - 1.0)


def buckle_law(theta_deg: float | np.ndarray, u0_mm: float = 25.7) -> np.ndarray:
    """``u_z = u0 sqrt(sin theta)`` [mm], the empirical fold law of ``eq:buckle-law``.

    ⚠ Fitted at ``w_lig = 18 mm`` on a single-ligament fold coupon. The prefactor is expected to
    scale with ligament width and that has never been tested, so applying it at another width is an
    extrapolation and must be labelled one.
    """
    th = np.radians(np.asarray(theta_deg, float))
    return u0_mm * np.sqrt(np.clip(np.sin(th), 0.0, None))
