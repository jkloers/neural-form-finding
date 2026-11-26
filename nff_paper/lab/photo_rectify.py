"""Perspective rectification of deployment photographs onto a metric millimetre frame.

A kirigami sheet lying on a floor is planar, so one homography per photograph maps image pixels
exactly onto floor millimetres. The fiducial is a rectangle of KNOWN size taped on the floor, which
is static across a photo series -- so every photograph lands in the same frame and the sheet can be
measured, not merely looked at.

The corner finder does NOT take the largest connected component. Tape gets crossed by the specimen
and breaks into pieces, so the fiducial arrives as several fragments covering different sides. Four
total-least-squares lines fitted to the union of all fragments recover the rectangle from partial
evidence, and with thousands of pixels per line the fitted line is the tape CENTRELINE rather than
one of its edges.

Nothing here imports from ``nff`` -- numpy, scipy, PIL and matplotlib's colour conversion only.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from matplotlib.colors import rgb_to_hsv
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull

# Hue windows in degrees, measured on the 2026-08-14 PET series and separated by empty troughs.
# ``h`` wraps when lo > hi. Keep these together with the regression test that pins the separation:
# a re-shoot under different lighting must fail loudly rather than mis-mask silently.
TAPE_HUES = {
    "red":    dict(h=(348.0, 12.0), s_min=0.35),
    "orange": dict(h=(15.0, 48.0),  s_min=0.35),
    "green":  dict(h=(70.0, 115.0), s_min=0.30),
}


# ── Loading and colour ────────────────────────────────────────────────────────────────────────

def load_rgb(path: str) -> np.ndarray:
    """(H, W, 3) float32 in [0, 1], with the EXIF orientation applied.

    A rotated EXIF flag would silently transpose every coordinate downstream, so it is resolved
    here and never carried further.
    """
    from PIL import Image, ImageOps
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        return np.asarray(im, dtype=np.float32) / 255.0


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def to_hsv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(hue_deg, saturation, value), hue in [0, 360)."""
    hsv = rgb_to_hsv(rgb)
    return hsv[..., 0] * 360.0, hsv[..., 1], hsv[..., 2]


def local_value(v: np.ndarray, frac: float = 0.125) -> np.ndarray:
    """Box-blurred value channel, as the local brightness a pixel should be judged against.

    A constant brightness floor rejects tape in shadow and accepts floor in glare. Hue is already
    illumination-invariant; only the gate needs to be local.
    """
    size = max(3, int(frac * min(v.shape)))
    return ndi.uniform_filter(v, size=size)


def _hue_window(h: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (h >= lo) | (h < hi) if lo > hi else (h >= lo) & (h < hi)


def colour_masks(rgb: np.ndarray, *, hues: Optional[dict] = None,
                 v_floor_rel: float = 0.45, v_floor_abs: float = 0.10) -> dict[str, np.ndarray]:
    """Boolean mask per tape colour.

    Args:
        rgb: (H, W, 3) in [0, 1].
        hues: colour -> ``dict(h=(lo, hi), s_min=...)``; defaults to :data:`TAPE_HUES`.
        v_floor_rel: a pixel must be at least this fraction of its LOCAL brightness.
        v_floor_abs: absolute floor, to reject noise in deep shadow.

    Returns:
        colour -> (H, W) bool.
    """
    hues = TAPE_HUES if hues is None else hues
    h, s, v = to_hsv(rgb)
    gate = (v > v_floor_rel * local_value(v)) & (v > v_floor_abs)
    return {name: _hue_window(h, *spec["h"]) & (s >= spec["s_min"]) & gate
            for name, spec in hues.items()}


def no_evidence_mask(rgb: np.ndarray, *, dilate: int = 5, min_grey_px: int = 400) -> np.ndarray:
    """Where the photograph carries no information about tape: blown highlights and grey objects.

    Samples landing here are given zero weight rather than being read as "no tape". A hole burned
    in the evidence must not pull a fit, and a lab weight sitting on the sheet is a hole.
    """
    _, s, v = to_hsv(rgb)
    spec = (v > 0.92) & (s < 0.15)
    grey = (s < 0.12) & (v > 0.30) & (v < 0.70)
    lab, n = ndi.label(grey)
    if n:
        keep = np.zeros(n + 1, dtype=bool)
        counts = np.bincount(lab.ravel(), minlength=n + 1)
        keep[1:] = counts[1:] >= min_grey_px
        grey = keep[lab]
    return ndi.binary_dilation(spec | grey, iterations=dilate)


def clean_mask(mask: np.ndarray, *, open_px: int = 2, min_area_frac: float = 0.02) -> np.ndarray:
    """Open by ``open_px``, then drop components smaller than a fraction of the largest.

    JPEG stores chroma at half resolution, so every mask carries a one-pixel halo and stray
    speckle; the area filter also removes tape offcuts lying on the floor.
    """
    m = ndi.binary_opening(mask, np.ones((open_px, open_px), bool)) if open_px > 1 else mask
    lab, n = ndi.label(m)
    if n == 0:
        return m
    counts = np.bincount(lab.ravel(), minlength=n + 1)
    counts[0] = 0
    keep = counts >= min_area_frac * counts.max()
    keep[0] = False
    return keep[lab]


# ── Lines, quads, homography ──────────────────────────────────────────────────────────────────

def fit_line_tls(pts: np.ndarray) -> tuple[np.ndarray, float]:
    """Total-least-squares line through a point cloud.

    Returns:
        ``(n, c)`` with ``n`` a unit normal and the line ``{p : n.p = c}``.
    """
    mu = pts.mean(0)
    cov = np.cov((pts - mu).T)
    n = np.linalg.eigh(cov)[1][:, 0]                 # smallest-eigenvalue direction = the normal
    return n, float(n @ mu)


def line_distance(pts: np.ndarray, n: np.ndarray, c: float) -> np.ndarray:
    return np.abs(pts @ n - c)


def intersect(l0: tuple, l1: tuple) -> np.ndarray:
    """Intersection of two lines given as ``(n, c)``."""
    A = np.stack([l0[0], l1[0]])
    return np.linalg.solve(A, np.array([l0[1], l1[1]]))


def min_area_rect_lines(pts: np.ndarray) -> list[tuple[np.ndarray, float]]:
    """Four ``(n, c)`` lines of the minimum-area rectangle enclosing ``pts``, in cyclic order.

    Rotating calipers over the convex hull. Used only to initialise :func:`fit_quad_klines`.
    """
    hull = pts[ConvexHull(pts).vertices]
    edges = np.roll(hull, -1, axis=0) - hull
    best = None
    for a in np.arctan2(edges[:, 1], edges[:, 0]):
        u = np.array([np.cos(a), np.sin(a)])
        w = np.array([-u[1], u[0]])
        pu, pw = hull @ u, hull @ w
        area = np.ptp(pu) * np.ptp(pw)
        if best is None or area < best[0]:
            best = (area, u, w, pu.min(), pu.max(), pw.min(), pw.max())
    _, u, w, u0, u1, w0, w1 = best
    # cyclic: -w side, +u side, +w side, -u side -> consecutive pairs meet at the four corners
    return [(w, w0), (u, u1), (w, w1), (u, u0)]


@dataclass
class QuadFit:
    """A fiducial rectangle recovered from a broken tape mask."""
    corners: np.ndarray                      # (4, 2) image pixels, cyclic
    lines: list                              # 4 x (n, c)
    residual_px: np.ndarray                  # (4,) median |distance| of assigned pixels
    counts: np.ndarray                       # (4,) pixels per line
    n_components: int = 0
    along: list = field(default_factory=list)  # per line: (t, signed distance) for the bow test


def fit_quad_klines(mask: np.ndarray, *, min_area_frac: float = 0.02, n_iter: int = 30,
                    trim_pct: float = 95.0, end_trim: float = 0.05,
                    n_polish: int = 6, reject_mad: float = 4.0) -> QuadFit:
    """Recover a rectangle's four corners from a mask that may be broken into fragments.

    Four lines are initialised from the minimum-area enclosing rectangle, then alternately
    re-assigned and re-fitted. Corners come from intersecting consecutive lines, so a corner that
    is occluded or cut is still recovered from the two sides that reach it.

    Args:
        mask: boolean tape mask.
        min_area_frac: keep components at least this fraction of the largest.
        n_iter: assign/refit sweeps.
        trim_pct: per sweep, drop pixels beyond this percentile of |distance|.
        end_trim: per line, drop this fraction from each end before refitting, so that pixels
            near a corner -- which are close to two lines -- do not bias either.
        n_polish: extra sweeps that reject on an ABSOLUTE distance instead of a percentile.
        reject_mad: in those sweeps, drop pixels beyond this many robust deviations of their line.
            A percentile trim keeps a fixed FRACTION however far away it is, so a stray blob of the
            same colour lying inside the frame -- tape offcuts and debris, present in the real
            photographs -- survives it and drags the line it is nearest. Measured on a synthetic
            fiducial, a decoy costs 2.5 px of corner accuracy without this pass and nothing with it.

    Returns:
        :class:`QuadFit`.
    """
    m = clean_mask(mask, min_area_frac=min_area_frac)
    _, n_comp = ndi.label(m)
    ys, xs = np.nonzero(m)
    pts = np.column_stack([xs, ys]).astype(np.float64)
    if pts.shape[0] < 100:
        raise ValueError(f"fiducial mask has only {pts.shape[0]} pixels after cleaning")

    lines = min_area_rect_lines(pts)
    for _ in range(n_iter):
        d = np.stack([line_distance(pts, n, c) for n, c in lines])       # (4, N)
        who = d.argmin(0)
        new = []
        for i, (n, c) in enumerate(lines):
            sel = pts[who == i]
            if sel.shape[0] < 50:
                new.append((n, c))
                continue
            di = line_distance(sel, n, c)
            sel = sel[di <= np.percentile(di, trim_pct)]
            t = sel @ np.array([-n[1], n[0]])                            # along the line
            lo, hi = np.quantile(t, [end_trim, 1.0 - end_trim])
            core = sel[(t >= lo) & (t <= hi)]
            new.append(fit_line_tls(core if core.shape[0] >= 50 else sel))
        lines = new

    for _ in range(n_polish):
        d = np.stack([line_distance(pts, n, c) for n, c in lines])
        who = d.argmin(0)
        new = []
        for i, (n, c) in enumerate(lines):
            sel = pts[who == i]
            if sel.shape[0] < 50:
                new.append((n, c))
                continue
            di = line_distance(sel, n, c)
            keep = di <= reject_mad * max(1.4826 * np.median(di), 1e-6)
            sel = sel[keep] if keep.sum() >= 50 else sel
            t = sel @ np.array([-n[1], n[0]])
            lo, hi = np.quantile(t, [end_trim, 1.0 - end_trim])
            core = sel[(t >= lo) & (t <= hi)]
            new.append(fit_line_tls(core if core.shape[0] >= 50 else sel))
        lines = new

    d = np.stack([line_distance(pts, n, c) for n, c in lines])
    who = d.argmin(0)
    residual = np.array([np.median(d[i][who == i]) if (who == i).any() else np.nan
                         for i in range(4)])
    counts = np.array([int((who == i).sum()) for i in range(4)])
    corners = np.stack([intersect(lines[i], lines[(i + 1) % 4]) for i in range(4)])
    along = []
    for i, (n, c) in enumerate(lines):
        sel = pts[who == i]
        along.append((sel @ np.array([-n[1], n[0]]), sel @ n - c))
    return QuadFit(corners=corners, lines=lines, residual_px=residual, counts=counts,
                   n_components=n_comp, along=along)


def homography_dlt(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Hartley-normalised 4-point DLT homography carrying ``src`` onto ``dst``."""
    def normalise(p):
        mu = p.mean(0)
        scale = np.sqrt(2.0) / np.sqrt(((p - mu) ** 2).sum(1)).mean()
        T = np.array([[scale, 0, -scale * mu[0]], [0, scale, -scale * mu[1]], [0, 0, 1.0]])
        return (p - mu) * scale, T

    s, Ts = normalise(np.asarray(src, float))
    d, Td = normalise(np.asarray(dst, float))
    rows = []
    for (x, y), (u, v) in zip(s, d):
        rows += [[-x, -y, -1, 0, 0, 0, u * x, u * y, u],
                 [0, 0, 0, -x, -y, -1, v * x, v * y, v]]
    H = np.linalg.svd(np.asarray(rows))[2][-1].reshape(3, 3)
    H = np.linalg.inv(Td) @ H @ Ts
    return H / H[2, 2]


def apply_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a homography to (..., 2) points."""
    p = np.asarray(pts, float)
    shape = p.shape
    p = p.reshape(-1, 2)
    q = np.column_stack([p, np.ones(len(p))]) @ H.T
    return (q[:, :2] / q[:, 2:3]).reshape(shape)


def rect_corners(width: float, height: float, *, cyclic_from: str = "lr") -> np.ndarray:
    """(4, 2) corners of an axis-aligned ``width x height`` rectangle centred on the origin.

    Cyclic order matches :func:`fit_quad_klines`, whose first corner is the intersection of the
    first two lines.
    """
    w, h = width / 2.0, height / 2.0
    order = {"lr": [(w, -h), (w, h), (-w, h), (-w, -h)],
             "ll": [(-w, -h), (w, -h), (w, h), (-w, h)]}[cyclic_from]
    return np.asarray(order, float)


def label_variants(corners: np.ndarray) -> list[np.ndarray]:
    """The 8 rotations/reflections of a cyclic corner list.

    Image y runs downward, so image-counterclockwise is world-clockwise: the handedness cannot be
    assumed and every labelling has to be scored.
    """
    out = []
    for flip in (False, True):
        base = corners[::-1] if flip else corners
        out += [np.roll(base, k, axis=0) for k in range(4)]
    return out


def warp_to_grid(img: np.ndarray, H: np.ndarray, x_mm: np.ndarray, y_mm: np.ndarray, *,
                 order: int = 1, fill: float = np.nan) -> np.ndarray:
    """Resample ``img`` onto a world grid.

    Args:
        img: (H, W) or (H, W, C) source, in image pixels.
        H: homography carrying image pixels -> world millimetres.
        x_mm, y_mm: 1-D grid coordinates; the output is ``(len(y_mm), len(x_mm), ...)``.
        order: 1 for bilinear. Never 0 -- nearest-neighbour aliases thin tape strips away.
        fill: value outside the source.

    Returns:
        the resampled array, with ``y_mm`` increasing along axis 0.
    """
    X, Y = np.meshgrid(np.asarray(x_mm, float), np.asarray(y_mm, float))
    src = apply_h(np.linalg.inv(H), np.stack([X.ravel(), Y.ravel()], axis=-1))
    coords = np.stack([src[:, 1], src[:, 0]])                          # (row, col)
    if img.ndim == 2:
        out = ndi.map_coordinates(img.astype(np.float32), coords, order=order,
                                  mode="constant", cval=fill)
        return out.reshape(X.shape)
    chans = [ndi.map_coordinates(img[..., k].astype(np.float32), coords, order=order,
                                 mode="constant", cval=fill).reshape(X.shape)
             for k in range(img.shape[2])]
    return np.stack(chans, axis=-1)


def corner_jitter_ensemble(corners: np.ndarray, dst: np.ndarray, sigma_px: float, n: int,
                           rng: np.random.Generator) -> list[np.ndarray]:
    """``n`` homographies from corners perturbed by ``N(0, sigma_px)``.

    The datum ambiguity of a taped fiducial is systematic, not statistical, so ``sigma_px`` should
    be the tape half-width rather than the fitted corner standard error -- deliberately pessimistic.
    """
    return [homography_dlt(corners + rng.normal(0.0, sigma_px, corners.shape), dst)
            for _ in range(n)]


# ── Manifest ──────────────────────────────────────────────────────────────────────────────────

def load_manifest(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


def save_manifest(path: str, manifest: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(manifest, fh, indent=2)


def clean_render(rgb: np.ndarray, masks: dict[str, np.ndarray],
                 colours: Optional[dict] = None, *,
                 floor_lo: float = 0.62, floor_hi: float = 0.97) -> np.ndarray:
    """Photograph reduced to a line drawing: floor flattened to grey, tape repainted flat.

    The specimen is transparent, so the tape IS the geometry; everything else is context that
    competes with it for the reader's attention.
    """
    colours = colours or {"red": (0.839, 0.157, 0.157), "orange": (0.961, 0.502, 0.145),
                          "green": (0.122, 0.541, 0.298)}
    lum = rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=rgb.dtype)
    lo, hi = np.nanpercentile(lum, [2, 98])
    grey = np.clip((lum - lo) / max(hi - lo, 1e-6), 0, 1) * (floor_hi - floor_lo) + floor_lo
    out = np.repeat(grey[..., None], 3, axis=-1).astype(np.float32)
    for name, colour in colours.items():
        m = masks.get(name)
        if m is not None and m.any():
            out[m] = colour
    return out
