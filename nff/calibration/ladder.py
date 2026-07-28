"""Ink-ladder video extensometer: hand-drawn tick marks -> true gauge strain.

A ladder of ink ticks drawn along the specimen gauge is tracked by 2D normalised
cross-correlation of a patch cut from the first frame. Fitting an affine map
``y_i(t) = a(t) + b(t) * y_i(0)`` across all ticks gives the gauge stretch directly:
``strain = b - 1``. Because it is a ratio of pixel distances the result needs no
pixel scale and carries no machine, grip or tab compliance -- unlike the crosshead.

The tick pitch does NOT need to be accurate. Hand-drawn marks at a nominal 10 mm
are fine: the affine fit references every frame against frame 0, so pitch errors
cancel. Only the *change* in spacing is used.

:func:`sync_by_grip` tracks the moving grip instead, whose displacement IS the
crosshead displacement, and recovers the video-to-Bluehill time offset (and the
pixel scale at the grip plane) by least squares.

Both entry points read frames through ffmpeg as raw greyscale -- no OpenCV.

Known systematic: a specimen that is initially bowed and straightens under load
moves through the depth of field, changing the apparent scale. At typical phone
working distances ~2 mm of straightening mimics a few 1e-3 of strain, comparable
to the elastic range. Most of it happens in the first few newtons and is absorbed
by fitting the modulus with a free intercept; shoot against a matte backdrop and
track the specimen width as an internal check when it matters.
"""
from __future__ import annotations

import subprocess

import numpy as np


def decode_gray(path: str, n_frames: int, width: int, height: int,
                crop: tuple[int, int, int, int] | None = None) -> np.ndarray:
    """Decode the first ``n_frames`` as greyscale.

    Args:
        path: video file.
        n_frames: how many frames to read from the start.
        width, height: frame size of the file.
        crop: optional ``(w, h, x, y)`` ffmpeg crop applied before decoding.

    Returns:
        (n, h, w) float32 luma.
    """
    vf = [] if crop is None else ["-vf", "crop={}:{}:{}:{}".format(*crop)]
    w, h = (width, height) if crop is None else (crop[0], crop[1])
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-frames:v", str(n_frames), *vf,
         "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        capture_output=True, check=True,
    )
    buf = np.frombuffer(proc.stdout, np.uint8)
    n = len(buf) // (w * h)
    return buf[: n * w * h].reshape(n, h, w).astype(np.float32)


def _ncc(window: np.ndarray, template: np.ndarray) -> float:
    a = window - window.mean()
    b = template - template.mean()
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def _track_one(frames: np.ndarray, y0: float, x0: int, tw: int, th: int,
               search: int, dx: int) -> np.ndarray:
    """Follow a single patch down the stack, parabolic-refined to sub-pixel."""
    h = frames.shape[1]
    tpl = frames[0][int(round(y0)) - th // 2: int(round(y0)) + th // 2, x0: x0 + tw].copy()
    out = np.empty(len(frames))
    out[0] = y0
    cur = float(y0)
    for i in range(1, len(frames)):
        scores, ys = [], []
        for d in range(-search, search + 1):
            y = int(round(cur)) + d
            if y - th // 2 < 0 or y + th // 2 > h:
                scores.append(-2.0)
                ys.append(y)
                continue
            best = -2.0
            for s in range(-dx, dx + 1):
                best = max(best, _ncc(frames[i][y - th // 2: y + th // 2, x0 + s: x0 + s + tw], tpl))
            scores.append(best)
            ys.append(y)
        s = np.asarray(scores)
        k = int(np.argmax(s))
        y = float(ys[k])
        if 0 < k < len(s) - 1:
            y += 0.5 * (s[k - 1] - s[k + 1]) / (s[k - 1] - 2 * s[k] + s[k + 1] + 1e-12)
        cur = y
        out[i] = y
    return out


def track_ladder(frames: np.ndarray, seed_y: list[float], *, x0: int, tw: int = 74,
                 th: int = 46, search: int = 7, dx: int = 2) -> np.ndarray:
    """Track every tick of the ladder.

    Args:
        frames: (n, h, w) greyscale stack from :func:`decode_gray`.
        seed_y: tick row positions in frame 0, any order.
        x0: left edge of the template window (should span digit + tick).
        tw, th: template width and height in pixels.
        search: vertical search half-range per frame.
        dx: horizontal search half-range, absorbing specimen sway.

    Returns:
        (n, n_ticks) tick rows per frame.
    """
    return np.stack([_track_one(frames, y, x0, tw, th, search, dx) for y in seed_y], axis=1)


def gauge_strain(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Affine stretch of the tracked ladder, referenced to frame 0.

    Args:
        positions: (n, n_ticks) from :func:`track_ladder`.

    Returns:
        ``(strain, residual_px)`` -- strain is ``b - 1``; the residual is the rms
        departure from a pure affine map and rises when the strain field stops
        being uniform (necking) or a tick is lost.
    """
    ref = positions[0]
    centred = ref - ref.mean()
    rows = positions - positions.mean(axis=1, keepdims=True)
    b = (centred * rows).sum(axis=1) / (centred * centred).sum()
    fit = positions.mean(axis=1, keepdims=True) + np.outer(b, centred)
    residual = np.sqrt(((positions - fit) ** 2).mean(axis=1))
    return b - 1.0, residual


def sync_by_grip(frames: np.ndarray, seed_y: float, video_t: np.ndarray,
                 csv_t: np.ndarray, csv_disp_mm: np.ndarray, *, x0: int, tw: int = 300,
                 th: int = 60, search: int = 6,
                 offsets: np.ndarray | None = None) -> tuple[float, float, float]:
    """Recover the video-to-Bluehill time offset from the moving grip.

    The grip is rigidly attached to the crosshead, so its travel is the crosshead
    displacement exactly -- immune to any specimen behaviour, including slip.

    Args:
        frames: (n, h, w) greyscale stack covering the whole grip.
        seed_y: a strong horizontal grip edge in frame 0.
        video_t: frame times.
        csv_t, csv_disp_mm: the Bluehill time and displacement columns.
        x0, tw, th, search: template placement and search range.
        offsets: candidate offsets in seconds; default -15..15 at 0.1 s.

    Returns:
        ``(offset_s, mm_per_px, rms_mm)``. ``csv_t = video_t + offset_s``.
    """
    y = _track_one(frames, seed_y, x0, tw, th, search, 0)
    travel_px = y[0] - y                                   # grip rises, rows decrease
    grid = np.arange(-15.0, 15.001, 0.1) if offsets is None else np.asarray(offsets, float)
    best = (np.inf, 0.0, 0.0)
    denom = float((travel_px ** 2).sum())
    for off in grid:
        ref = np.interp(video_t + off, csv_t, csv_disp_mm)
        scale = float((travel_px * ref).sum() / denom) if denom > 0 else 0.0
        resid = float(((ref - scale * travel_px) ** 2).sum())
        if resid < best[0]:
            best = (resid, float(off), scale)
    resid, off, scale = best
    return off, scale, float(np.sqrt(resid / len(video_t)))
