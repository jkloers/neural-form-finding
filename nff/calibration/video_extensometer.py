"""Two-dot (and lateral) video extensometer for true gauge strain.

Tracks high-contrast marks painted on the specimen gauge through a test video and
returns their positions per frame. Axial mark separation -> true axial strain,
immune to machine / grip / tape compliance (protocol §13.5). Optional lateral marks
-> transverse strain -> Poisson's ratio.

Pipeline:  video -> :func:`track_marks` -> :func:`marks_to_strain` -> sync with the
Bluehill force curve (:func:`sync_to_force`) -> true stress-strain in
:mod:`stress_strain`.

``opencv-python`` is an optional dependency, imported lazily so the rest of the
package (CSV parsing, stress-strain) works without it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _cv2():
    try:
        import cv2  # noqa: PLC0415
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "video_extensometer needs opencv-python: `pip install opencv-python`"
        ) from e
    return cv2


@dataclass
class MarkTrack:
    """Tracked mark centroids over a video."""

    frame_time_s: np.ndarray            # (n_frames,)
    positions_px: np.ndarray            # (n_frames, n_marks, 2) in (x, y) pixels
    mm_per_px: float | None = None      # scale from calibrate_scale(), else None


def calibrate_scale(known_mm: float, measured_px: float) -> float:
    """mm-per-pixel from a known length (ruler in frame, or initial dot spacing)."""
    return float(known_mm) / float(measured_px)


def _detect_blobs(gray, dark_on_light: bool, min_area: int, max_area: int):
    cv2 = _cv2()
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = float(min_area)
    params.maxArea = float(max_area)
    params.filterByColor = True
    params.blobColor = 0 if dark_on_light else 255
    params.filterByCircularity = False
    params.filterByInertia = False
    params.filterByConvexity = False
    detector = cv2.SimpleBlobDetector_create(params)
    kps = detector.detect(gray)
    return np.array([kp.pt for kp in kps], dtype=float)  # (k, 2) in (x, y)


def track_marks(
    video_path: str,
    n_marks: int = 2,
    *,
    dark_on_light: bool = True,
    min_area: int = 20,
    max_area: int = 5000,
    roi: tuple[int, int, int, int] | None = None,
) -> MarkTrack:
    """Track ``n_marks`` painted dots through a video by nearest-neighbour association.

    Assumes the marks are the strongest blobs of the chosen polarity and move only a
    little frame-to-frame (true for a slow tensile pull). First frame seeds the marks
    (sorted top-to-bottom by y); later frames match each mark to its nearest detected
    blob.

    Args:
        video_path: path to the test video.
        n_marks: number of marks (2 axial; 4 to add a lateral pair for Poisson).
        dark_on_light: True for dark marks on translucent PET; False for white marks.
        min_area / max_area: blob-area gate in pixels (tune to your dot size).
        roi: optional (x, y, w, h) crop to the gauge region to reject grip clutter.

    Returns:
        A :class:`MarkTrack` (``mm_per_px`` unset — call :func:`calibrate_scale`).
    """
    cv2 = _cv2()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    times: list[float] = []
    positions: list[np.ndarray] = []
    prev: np.ndarray | None = None
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if roi is not None:
            x, y, w, h = roi
            frame = frame[y : y + h, x : x + w]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pts = _detect_blobs(gray, dark_on_light, min_area, max_area)

        if prev is None:
            if pts.shape[0] < n_marks:
                frame_idx += 1
                continue  # wait for a frame where all marks are visible
            order = np.argsort(pts[:, 1])  # top -> bottom
            cur = pts[order][:n_marks]
        else:
            cur = np.empty_like(prev)
            for m in range(prev.shape[0]):
                if pts.shape[0] == 0:
                    cur[m] = prev[m]  # hold last known if nothing detected
                    continue
                d = np.linalg.norm(pts - prev[m], axis=1)
                cur[m] = pts[int(np.argmin(d))]

        positions.append(cur)
        times.append(frame_idx / fps)
        prev = cur
        frame_idx += 1

    cap.release()
    return MarkTrack(
        frame_time_s=np.asarray(times),
        positions_px=np.asarray(positions),
    )


def marks_to_strain(track: MarkTrack, axial_pair: tuple[int, int] = (0, 1)) -> np.ndarray:
    """Engineering axial strain from the separation of two axial marks over time."""
    i, j = axial_pair
    sep = np.linalg.norm(
        track.positions_px[:, i, :] - track.positions_px[:, j, :], axis=1
    )
    sep0 = sep[0]
    return (sep - sep0) / sep0  # scale cancels -> mm_per_px not needed for strain


def marks_to_poisson(
    track: MarkTrack, axial_pair: tuple[int, int], lateral_pair: tuple[int, int]
) -> np.ndarray:
    """Instantaneous Poisson ratio -lateral_strain / axial_strain (needs 4 marks)."""
    eps_ax = marks_to_strain(track, axial_pair)
    eps_lat = marks_to_strain(track, lateral_pair)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(eps_ax != 0, -eps_lat / eps_ax, np.nan)


def sync_to_force(
    strain: np.ndarray,
    frame_time_s: np.ndarray,
    force_time_s: np.ndarray,
    force_N: np.ndarray,
    t_offset_s: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample force onto the video timebase (shifted by ``t_offset_s``).

    Determine ``t_offset_s`` from a shared trigger (LED flash / clap at test start).

    Returns:
        (strain, force_N) aligned on the video frames.
    """
    f = np.interp(frame_time_s + t_offset_s, force_time_s, force_N,
                  left=np.nan, right=np.nan)
    good = ~np.isnan(f)
    return strain[good], f[good]
