"""Retouching of a rectified photograph for publication: empty corners, glare, loose hardware.

A rectification warps a keystoned photograph onto an axis-aligned millimetre grid, so the output
rectangle always reaches outside the source and those pixels carry no data. Left at white they read
as torn corners; what belongs there is the floor the photograph would have shown, so it is
reconstructed from the boundary rather than declared.

The same reconstruction serves the other three subjects a laboratory floor supplies: a specular
highlight on a transparent sheet, a weight holding the specimen down, and a cable crossing the
frame. None of them is the measurement -- the homography is fitted on the UNMODIFIED photograph, and
these operations run afterwards, on the panel that is drawn -- but all three compete with the
specimen for the reader's attention.

⚠ Every function here returns its mask statistics alongside the image, and the caller is expected to
print them. A retouched photograph in a paper is defensible when the reader can see what was
removed and how much of the frame it covered; it is not defensible silently.

Nothing here imports from ``nff`` -- numpy and scipy only.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from scipy import ndimage as ndi

# A cable is a few millimetres across but tens of centimetres long, so a mask built by a hue or
# brightness rule fragments on it. Named regions are given in MILLIMETRES of the rectified frame,
# which is the only frame in which a lab object has a fixed position across re-renders.
Shape = dict


def _stats(mask: np.ndarray, gx: Optional[np.ndarray] = None,
           gy: Optional[np.ndarray] = None) -> dict:
    """Area of a boolean mask in pixels, in square millimetres and as a fraction of the frame."""
    n = int(mask.sum())
    out = dict(px=n, frac=float(n) / float(mask.size))
    if gx is not None and gy is not None and len(gx) > 1 and len(gy) > 1:
        cell = abs(float(gx[1] - gx[0]) * float(gy[1] - gy[0]))
        out["mm2"] = n * cell
    return out


def mask_stats(mask: np.ndarray, gx: Optional[np.ndarray] = None,
               gy: Optional[np.ndarray] = None) -> dict:
    """Public name for :func:`_stats`, so a caller can put its own mask in the ledger."""
    return _stats(mask, gx, gy)


def empty_mask(img: np.ndarray) -> np.ndarray:
    """Where an inverse warp landed outside its source, as marked by ``fill=np.nan``."""
    a = np.asarray(img)
    return ~np.isfinite(a).all(axis=-1) if a.ndim == 3 else ~np.isfinite(a)


def specular_mask(img: np.ndarray, inside: Optional[np.ndarray] = None, *,
                  where: Optional[np.ndarray] = None, q: float = 97.0, q_grow: float = 86.0,
                  sat_max: float = 0.22, min_area_px: int = 250, close_px: int = 5,
                  dilate_px: int = 6) -> np.ndarray:
    """A specular highlight: bright, colourless, compact, and on the specimen.

    The threshold is a PERCENTILE of the region it is asked about rather than an absolute level.
    A gamma or a white balance has usually run before this point, so an absolute level would have
    to be retuned every time the tone curve moved, while the highlight stays the brightest thing on
    the sheet under any of them.

    ⚠ The mask is grown by HYSTERESIS, not by a margin. A highlight has a hard clipped crest and a
    soft skirt that decays over many times its own width, so a mask cut at one level covers the
    crest and leaves the skirt as a halo -- reconstructing it then swaps a bright blob for a bright
    ring, which reads worse. A high percentile SEEDS, a low one GROWS, and only the components that
    the seed reaches are kept, so the low level never floods the whole specimen on its own.

    ⚠ A brightness percentile taken over a whole specimen is the wrong statistic when the specimen
    is not uniformly lit. On the deployment frame the panels sit at visibly different levels, so
    lowering the seed far enough to reach one highlight's skirt swallows an entire brighter panel --
    measured, 1.6% of the panel at one threshold and 46% one step down. ``where`` is the answer: a
    DECLARED region bounds the search, and the detector then only decides the highlight's exact
    shape within it. What gets erased is bounded by a stated intention rather than by a threshold.

    Args:
        img: (H, W, 3) in [0, 1]; NaN is ignored.
        inside: (H, W) bool, the region to search -- the specimen, not the floor. The floor is
            bright everywhere, so an unrestricted search returns the floor.
        where: (H, W) bool bounding the search further, from :func:`region_mask`. ⚠ Both
            percentiles are then taken WITHIN it, so ``q`` and ``q_grow`` change meaning with it
            and must be retuned when it is supplied.
        q: luminance percentile that seeds a highlight.
        q_grow: the lower percentile the seed is allowed to grow through.
        sat_max: a specular reflection is the light source's colour, so it is near-neutral.
        min_area_px: components below this are grain, not a reflection.
        close_px: closes the pinholes a cut line punches through a highlight.
        dilate_px: covers the last soft pixels at the grown edge.

    Returns:
        (H, W) bool.
    """
    a = np.asarray(img, float)
    lum = np.nanmean(a, axis=2)
    mx = np.nanmax(a, axis=2)
    mn = np.nanmin(a, axis=2)
    sat = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    region = np.isfinite(lum) if inside is None else (inside & np.isfinite(lum))
    if where is not None:
        region = region & where
    if not region.any():
        return np.zeros(lum.shape, bool)
    neutral = region & (sat < sat_max)
    # ⚠ The comparison is >=, not >. A specular highlight CLIPS: its crest sits on a plateau of
    # saturated pixels, so when that plateau is wider than (100 - q)% of the region the percentile
    # lands exactly on the maximum and a strict > returns an empty mask -- brightest possible
    # subject, nothing found.
    seed = neutral & (lum >= float(np.nanpercentile(lum[region], q)))
    seed = drop_small(seed, min_area_px)
    if not seed.any():
        return seed
    grow = neutral & (lum >= float(np.nanpercentile(lum[region], min(q_grow, q))))
    m = ndi.binary_propagation(seed, mask=grow)
    if close_px > 1:
        m = ndi.binary_closing(m, np.ones((close_px, close_px), bool)) & region
    if dilate_px > 1:
        m = ndi.binary_dilation(m, np.ones((dilate_px, dilate_px), bool)) & region
    return drop_small(m, min_area_px)


def drop_small(mask: np.ndarray, min_area_px: int) -> np.ndarray:
    """Remove connected components below ``min_area_px``."""
    lab, n = ndi.label(mask)
    if n == 0:
        return mask
    counts = np.bincount(lab.ravel(), minlength=n + 1)
    counts[0] = 0
    keep = counts >= int(min_area_px)
    keep[0] = False
    return keep[lab]


def region_mask(gx: np.ndarray, gy: np.ndarray, shapes: Sequence[Shape]) -> np.ndarray:
    """Named regions of the rectified frame, in millimetres.

    A hand-placed region is the honest tool for a fixed object in a single photograph. A detector
    tuned to find a dark cable also finds the sheet's own cut lines, which are the subject; a
    region says exactly what is being covered, is reproducible, and can be read off the source.

    Args:
        gx: (nx,) grid abscissae in millimetres.
        gy: (ny,) grid ordinates in millimetres.
        shapes: each ``dict(kind=...)`` with

            - ``disc``: ``xy=(x, y)``, ``r`` in millimetres;
            - ``segment``: ``a=(x, y)``, ``b=(x, y)``, ``w`` full width in millimetres;
            - ``polygon``: ``pts`` as (n, 2) in millimetres.

            Every shape may carry a ``why`` string; it is documentation, not geometry.

    Returns:
        (ny, nx) bool.
    """
    X, Y = np.meshgrid(np.asarray(gx, float), np.asarray(gy, float))
    out = np.zeros(X.shape, bool)
    for s in shapes:
        kind = s.get("kind", "disc")
        if kind == "disc":
            cx, cy = s["xy"]
            out |= (X - cx) ** 2 + (Y - cy) ** 2 <= float(s["r"]) ** 2
        elif kind == "segment":
            a = np.asarray(s["a"], float)
            b = np.asarray(s["b"], float)
            ab = b - a
            L2 = float(ab @ ab)
            if L2 <= 0.0:
                continue
            t = np.clip(((X - a[0]) * ab[0] + (Y - a[1]) * ab[1]) / L2, 0.0, 1.0)
            dx = X - (a[0] + t * ab[0])
            dy = Y - (a[1] + t * ab[1])
            out |= dx * dx + dy * dy <= (0.5 * float(s["w"])) ** 2
        elif kind == "polygon":
            from matplotlib.path import Path
            pts = np.stack([X.ravel(), Y.ravel()], axis=-1)
            out |= Path(np.asarray(s["pts"], float)).contains_points(pts).reshape(X.shape)
        else:
            raise ValueError(f"unknown shape kind {kind!r}")
    return out


def inpaint(img: np.ndarray, mask: np.ndarray, *, iters: int = 240, smooth_px: int = 3,
            source: Optional[np.ndarray] = None) -> np.ndarray:
    """Reconstruct ``mask`` by diffusion from its boundary, leaving everything else bit-exact.

    A nearest-source fill seeds the region and a normalised Jacobi smoothing relaxes it, which is a
    discrete harmonic extension: the result carries the surrounding gradient and no texture. That
    flatness is a feature, not a limitation. A reconstructed patch that is visibly smoother than the
    floor around it cannot be mistaken for evidence.

    ⚠ ``source`` is what stops the reconstruction from inventing the wrong SUBJECT. Diffusion pulls
    from whatever borders the mask, and on this plate the highlight, the cable and the weight all
    touch orange marking tape -- so an unrestricted fill spread tape colour across the sheet and
    left a brown smudge where the glare had been. Excluding the tape from the sources, and
    normalising each sweep by the weight that actually contributed, keeps the fill the colour of
    the sheet it is standing in for.

    Args:
        img: (H, W, 3); NaN inside ``mask`` is expected and fine.
        mask: (H, W) bool, the region to reconstruct.
        iters: Jacobi sweeps. Large regions need more; the seed is already close.
        smooth_px: width of the averaging kernel driving each sweep.
        source: (H, W) bool, the pixels the reconstruction may draw from. Defaults to everything
            outside the mask.

    Returns:
        (H, W, 3), a copy.
    """
    a = np.array(img, dtype=float, copy=True)
    m = np.asarray(mask, bool)
    if not m.any():
        return a
    known = ~m & np.isfinite(a).all(axis=-1)
    if source is not None:
        known = known & np.asarray(source, bool)
    if not known.any():
        raise ValueError("nothing outside the mask to reconstruct from")
    # Seed from the nearest source pixel: the relaxation then only has to remove the seams of that
    # Voronoi fill rather than carry a constant across the whole region.
    _, idx = ndi.distance_transform_edt(~known, return_indices=True)
    k = max(int(smooth_px), 3)
    ker = np.ones((k, k), float) / float(k * k)
    weight = (known | m).astype(float)                # everything else contributes nothing
    den = np.maximum(ndi.convolve(weight, ker, mode="nearest"), 1e-6)
    for c in range(a.shape[2]):
        ch = a[..., c]
        ch[m] = ch[idx[0][m], idx[1][m]]
        ch = np.nan_to_num(ch, nan=float(np.nanmean(ch[known])))
        for _ in range(int(iters)):
            ch[m] = (ndi.convolve(ch * weight, ker, mode="nearest") / den)[m]
        a[..., c] = ch
    return a


def fill_empty(img: np.ndarray, *, iters: int = 240, gx: Optional[np.ndarray] = None,
               gy: Optional[np.ndarray] = None,
               source: Optional[np.ndarray] = None) -> tuple[np.ndarray, dict]:
    """Reconstruct the warp's empty region, so a rectification stops showing torn white corners.

    Args:
        img: (H, W, 3) with ``np.nan`` wherever the inverse warp left its source.

    Returns:
        ``(image, stats)``; ``stats`` carries the empty area.
    """
    m = empty_mask(img)
    st = _stats(m, gx, gy)
    if not m.any():
        return np.asarray(img, float), st
    return inpaint(img, m, iters=iters, source=source), st


def neutralise(img: np.ndarray, mask: np.ndarray,
               tone: Sequence[float] = (0.80, 0.80, 0.82)) -> np.ndarray:
    """Repaint ``mask`` in one flat neutral tone.

    For a fiducial that must stay VISIBLE but must stop carrying a hue the plate has already spent:
    red means disagreement wherever else this paper draws it, so red tape in a photograph on the
    same plate is a collision.

    ⚠ FLAT, not shaded. Scaling the tone by each pixel's own luminance looks like the careful
    choice and is not: tape is the most specular thing in frame, so its highlights divide out above
    one and clip, and a band of white glitter replaces the band of red. This is the same flat
    repaint ``photo_rectify.clean_render`` already uses, for the same reason.
    """
    a = np.array(img, dtype=float, copy=True)
    m = np.asarray(mask, bool)
    if not m.any():
        return a
    a[m] = np.clip(np.asarray(tone, float), 0.0, 1.0)
    return a


def retouch(img: np.ndarray, *, gx: np.ndarray, gy: np.ndarray,
            inside: Optional[np.ndarray] = None,
            objects: Sequence[Shape] = (),
            neutral_mask: Optional[np.ndarray] = None,
            neutral_tone: Sequence[float] = (0.80, 0.80, 0.82),
            glare: bool = True, glare_where: Optional[np.ndarray] = None,
            source: Optional[np.ndarray] = None, iters: int = 240,
            glare_kw: Optional[dict] = None) -> tuple[np.ndarray, dict]:
    """Every retouch of one rectified panel, in one pass, with a ledger of what it did.

    Order matters. The empty region is reconstructed first so that later masks see a complete
    image; the tape is neutralised before the glare search so that a bright tape edge cannot be
    read as a highlight; the named objects go last because they are reconstructed, not measured.

    Args:
        img: (H, W, 3) rectified panel, NaN outside the warp's source.
        gx, gy: the millimetre grid ``img`` was sampled on.
        inside: (H, W) bool, the specimen -- the region the glare search is restricted to.
        objects: named regions to reconstruct, as :func:`region_mask` takes them.
        neutral_mask: (H, W) bool to repaint neutral, typically the target tape.
        glare: whether to search for and reconstruct a specular highlight.
        glare_where: the declared region that search is bounded to; see :func:`specular_mask`.
        source: the pixels every reconstruction may draw from; see :func:`inpaint`. Pass the frame
            with the marking tape excluded, or the tape's colour spreads into the fills.

    Returns:
        ``(image, ledger)``. The ledger maps each operation to its area statistics, and is written
        into the figure's provenance sidecar.
    """
    out = np.asarray(img, float)
    ledger: dict = {}

    out, ledger["empty"] = fill_empty(out, iters=iters, gx=gx, gy=gy, source=source)

    if neutral_mask is not None and neutral_mask.any():
        out = neutralise(out, neutral_mask, neutral_tone)
        ledger["neutralised"] = _stats(neutral_mask, gx, gy)

    if glare:
        gm = specular_mask(out, inside, where=glare_where, **(glare_kw or {}))
        if gm.any():
            out = inpaint(out, gm, iters=iters, source=source)
        ledger["glare"] = _stats(gm, gx, gy)

    if objects:
        om = region_mask(gx, gy, objects)
        if om.any():
            out = inpaint(out, om, iters=iters, source=source)
        ledger["objects"] = _stats(om, gx, gy)
        ledger["objects"]["named"] = [s.get("why", s.get("kind", "?")) for s in objects]

    return np.clip(out, 0.0, 1.0), ledger


def ledger_lines(ledger: dict) -> list[str]:
    """The ledger as sidecar prose, one line per operation."""
    name = {"empty": "empty corners of the rectification, reconstructed by diffusion",
            "neutralised": "target tape, repainted neutral",
            "glare": "specular highlight, reconstructed by diffusion",
            "objects": "loose hardware, reconstructed by diffusion",
            "completed": "target tape completed along its measured corners"}
    out = []
    for key, st in ledger.items():
        area = f", {st['mm2'] / 1e6:.3f} m2" if "mm2" in st else ""
        why = "" if not st.get("named") else "  [" + "; ".join(st["named"]) + "]"
        out.append(f"  {name.get(key, key)}: {st['frac'] * 100:.2f}% of the panel{area}{why}")
    return out
