"""Rendering: project artwork onto the deployed tiles and pull it back to the flat sheet.

Because every tile moves rigidly, the pull-back is exact. A tile's colour -- or its
whole artwork fragment -- is read where the tile lands when deployed, then painted
where that same tile sits on the flat closed sheet. The flat A4 therefore carries a
scrambled image that reassembles into the target on deployment.
"""

import os

import numpy as np
from jaxtyping import Float
from matplotlib.collections import PolyCollection
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.transforms import Affine2D

from nff.closed.shield_kinematics import deploy_tiles
from nff.closed.shield_target import sample_artwork


def to_artwork_frame(points_mm, target_unit, scale: float, shift, outline_offset):
    """Map deployed millimetre coordinates back into the artwork's unit frame."""
    pts = np.asarray(points_mm, dtype=float) - np.asarray(outline_offset)
    return (pts - np.asarray(shift)) / scale + np.mean(np.asarray(target_unit), axis=0)


def tile_colours(deployed, rgb, extent, target_unit, scale, shift, outline_offset,
                 samples: int = 6) -> Float[np.ndarray, "n_tiles 3"]:
    """Mean artwork colour under each deployed tile.

    Bilinear supersampling over the quad, so a tile straddling a colour boundary
    averages rather than snapping to whichever colour sits under its centroid.
    """
    dep = np.asarray(deployed)
    t = (np.arange(samples) + 0.5) / samples
    u, v = np.meshgrid(t, t, indexing="ij")
    u, v = u.reshape(-1), v.reshape(-1)

    bl, br, tr, tl = dep[:, 0], dep[:, 1], dep[:, 2], dep[:, 3]
    bottom = bl[:, None, :] * (1 - u)[None, :, None] + br[:, None, :] * u[None, :, None]
    top = tl[:, None, :] * (1 - u)[None, :, None] + tr[:, None, :] * u[None, :, None]
    pts = bottom * (1 - v)[None, :, None] + top * v[None, :, None]

    flat = pts.reshape(-1, 2)
    unit = to_artwork_frame(flat, target_unit, scale, shift, outline_offset)
    cols = sample_artwork(rgb, extent, unit).reshape(dep.shape[0], -1, 3)
    return cols.mean(axis=1) / 255.0


def _tile_polys(tiles):
    return [np.asarray(tiles[k]) for k in range(np.asarray(tiles).shape[0])]


def draw_tiles(ax, tiles, colours, edge="#1A1A1A", lw=0.15):
    """Filled tiles with per-tile colour, no transparency."""
    ax.add_collection(PolyCollection(_tile_polys(tiles), facecolors=colours,
                                     edgecolors=edge, linewidths=lw, zorder=2))
    _fit(ax, tiles)
    return ax


def _fit(ax, *tile_sets, pad_frac=0.03):
    pts = np.concatenate([np.asarray(t).reshape(-1, 2) for t in tile_sets], axis=0)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    pad = pad_frac * float(np.max(hi - lo))
    ax.set_xlim(lo[0] - pad, hi[0] + pad)
    ax.set_ylim(lo[1] - pad, hi[1] + pad)
    ax.set_aspect("equal")
    ax.axis("off")


def render_tiles(tiles, colours, out_png, *, dpi=220, figsize=(6.5, 8.5),
                 edge="#1A1A1A", lw=0.15, background="white"):
    """Standalone figure of one tile set."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(background)
    draw_tiles(ax, tiles, colours, edge=edge, lw=lw)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight", facecolor=background)
    plt.close(fig)
    return out_png


def _crop_to(rgb, extent, points_unit, pad_px: int = 2):
    """Smallest sub-image of the artwork covering ``points_unit``, with its own extent.

    Each tile needs a few thousand pixels of a multi-megapixel artwork. Handing
    matplotlib the whole raster once per tile and clipping it away is what made these
    figures take tens of minutes; cropping first makes the cost proportional to the
    tile rather than to the image.
    """
    x_min, x_max, y_min, y_max = extent
    H, W = rgb.shape[:2]
    dx = (x_max - x_min) / W
    dy = (y_max - y_min) / H

    pts = np.asarray(points_unit, dtype=float)
    c = np.floor((pts[:, 0] - x_min) / dx).astype(int)
    r = np.floor((y_max - pts[:, 1]) / dy).astype(int)

    c0 = int(np.clip(c.min() - pad_px, 0, W - 1))
    c1 = int(np.clip(c.max() + 1 + pad_px, c0 + 1, W))
    r0 = int(np.clip(r.min() - pad_px, 0, H - 1))
    r1 = int(np.clip(r.max() + 1 + pad_px, r0 + 1, H))

    sub_extent = (x_min + c0 * dx, x_min + c1 * dx,
                  y_max - r1 * dy, y_max - r0 * dy)
    return rgb[r0:r1, c0:c1], sub_extent


def _unit_to_mm(target_unit, scale, shift, outline_offset):
    """Affine mapping the artwork's unit frame onto deployed millimetres."""
    mean_tu = np.mean(np.asarray(target_unit), axis=0)
    return (Affine2D().translate(*(-mean_tu)).scale(scale)
            .translate(*np.asarray(shift)).translate(*np.asarray(outline_offset)))


def render_texture_deployed(ax, deployed, rgb, extent, target_unit, scale, shift,
                            outline_offset, *, edge=None, lw=0.1):
    """Print the artwork on the deployed tessellation, clipped tile by tile.

    The voids between tiles stay empty, so this is the image as it would actually
    appear on the opened sheet -- the thing the closed state is reverse-engineered
    from.
    """
    dep = np.asarray(deployed)
    unit_to_mm = _unit_to_mm(target_unit, scale, shift, outline_offset)
    dep_unit = to_artwork_frame(dep.reshape(-1, 2), target_unit, scale, shift,
                                outline_offset).reshape(dep.shape)

    for k in range(dep.shape[0]):
        patch = MplPolygon(dep[k], closed=True, facecolor="none", edgecolor="none",
                           transform=ax.transData)
        ax.add_patch(patch)
        sub, sub_extent = _crop_to(rgb, extent, dep_unit[k])
        im = ax.imshow(sub, extent=sub_extent, origin="upper",
                       transform=unit_to_mm + ax.transData,
                       interpolation="bilinear", zorder=2)
        im.set_clip_path(patch)
        if edge is not None:
            ax.add_patch(MplPolygon(dep[k], closed=True, facecolor="none",
                                    edgecolor=edge, linewidth=lw, zorder=3))
    _fit(ax, dep)
    return ax


def render_texture_at(ax, tiles_state, deployed, rgb, extent, target_unit, scale, shift,
                      outline_offset, sigma, psi_state, psi_deployed, *,
                      edge=None, lw=0.1, fit=True, alpha: float = 1.0):
    """Print each tile's artwork fragment at an arbitrary deployment state.

    Each tile carries the fragment it covers when deployed. Because the motion is
    rigid, transporting that fragment to any other state is exact: rotate by the
    difference in tile angle about the deployed centroid, then translate onto the
    state centroid. ``psi_state = 0`` gives the closed sheet.
    """
    state = np.asarray(tiles_state)
    dep = np.asarray(deployed)
    unit_to_mm = _unit_to_mm(target_unit, scale, shift, outline_offset)

    dep_c = dep.mean(axis=1)
    state_c = state.mean(axis=1)
    d_theta = np.asarray(sigma) * (float(psi_state) - float(psi_deployed))

    # The artwork each tile needs is the region under its DEPLOYED footprint: the
    # per-tile transform carries that region onto the state polygon.
    dep_unit = to_artwork_frame(dep.reshape(-1, 2), target_unit, scale, shift,
                                outline_offset).reshape(dep.shape)

    for k in range(state.shape[0]):
        patch = MplPolygon(state[k], closed=True, facecolor="none",
                           edgecolor="none", transform=ax.transData)
        ax.add_patch(patch)
        tr = (unit_to_mm
              + Affine2D().translate(*(-dep_c[k])).rotate(d_theta[k]).translate(*state_c[k])
              + ax.transData)
        sub, sub_extent = _crop_to(rgb, extent, dep_unit[k])
        im = ax.imshow(sub, extent=sub_extent, origin="upper", transform=tr,
                       interpolation="bilinear", zorder=2)
        im.set_clip_path(patch)
        if alpha < 1.0:
            im.set_alpha(alpha)
        if edge is not None:
            ax.add_patch(MplPolygon(state[k], closed=True, facecolor="none",
                                    edgecolor=edge, linewidth=lw, zorder=3,
                                    alpha=alpha))
    if fit:
        _fit(ax, state)
    return ax


def render_texture_pullback(ax, flat_tiles, deployed, rgb, extent, target_unit,
                            scale, shift, outline_offset, sigma, psi, *, edge=None, lw=0.1):
    """Closed-state special case of ``render_texture_at`` (``psi_state = 0``)."""
    return render_texture_at(ax, flat_tiles, deployed, rgb, extent, target_unit, scale,
                             shift, outline_offset, sigma, 0.0, psi, edge=edge, lw=lw)


def target_polygon_mm(target_unit, scale, shift, outline_offset):
    """The target outline in deployed millimetre coordinates."""
    tu = np.asarray(target_unit)
    return (tu - tu.mean(axis=0)) * scale + np.asarray(shift) + np.asarray(outline_offset)


def render_reference(ax, rgb, extent, target_unit, scale, shift, outline_offset,
                     ref_rgb=None, ref_extent=None, *, alpha: float = 1.0,
                     zorder: int = 2):
    """Undeformed reference artwork, clipped to its silhouette, at deployed scale.

    Drawn in the same millimetre frame as the deployed panels, so it reads as a
    like-for-like comparison rather than a detached thumbnail.
    """
    target_mm = target_polygon_mm(target_unit, scale, shift, outline_offset)
    patch = MplPolygon(target_mm, closed=True, facecolor="none", edgecolor="none",
                       transform=ax.transData)
    ax.add_patch(patch)
    im = ax.imshow(rgb if ref_rgb is None else ref_rgb,
                   extent=extent if ref_extent is None else ref_extent, origin="upper",
                   transform=_unit_to_mm(target_unit, scale, shift, outline_offset) + ax.transData,
                   interpolation="bilinear", zorder=zorder)
    im.set_clip_path(patch)
    if alpha < 1.0:
        im.set_alpha(alpha)
    return target_mm


def default_strip_labels(n_steps: int, reference: bool):
    """``reference | closed | deploying ... | open``."""
    labels = ["closed"] + ["deploying"] * max(0, n_steps - 2) + (["open"] if n_steps > 1 else [])
    return (["reference"] if reference else []) + labels[:n_steps]


def render_deployment_strip(plan, flat_tiles, deployed, rgb, extent, target_unit, scale,
                            shift, outline_offset, psi, *, n_steps=4, panel_w=5.7,
                            panel_h=6.4, reference=False, ref_rgb=None, ref_extent=None,
                            labels=None, label_size=None, label_colour="#1A1A1A",
                            rasterize=False):
    """Closed to open in ``n_steps`` equal increments, all panels at one common scale.

    With ``reference``, the undeformed target is placed first, separated from the
    deployment sequence by a hairline. No titles or annotation: the panels share
    xlim/ylim spans, so the growth across the strip is the true areal expansion rather
    than the artefact of fitting each state to its own crop.

    Args:
        panel_w, panel_h: one panel's size in inches. Author these at the size the strip PRINTS,
            so ``label_size`` is a printed point size.
        rasterize: flatten each panel's per-tile rasters into one image, keeping the labels vector.
            This is what makes a PDF of this figure viable.
    """
    import matplotlib.pyplot as plt

    fractions = np.linspace(0.0, 1.0, n_steps)
    states = [np.asarray(deploy_tiles(plan, flat_tiles, float(f) * float(psi)))
              for f in fractions]

    clouds = [s.reshape(-1, 2) for s in states]
    if reference:
        clouds.append(target_polygon_mm(target_unit, scale, shift, outline_offset))

    # Bounding-box centring, not centroid: the reference is a solid silhouette while
    # the deployed state is a lace with boundary spikes, so their centroids sit
    # differently inside their own outlines and the two would not line up.
    centres = [0.5 * (c.min(axis=0) + c.max(axis=0)) for c in clouds]
    half = 1.03 * max(float(np.max(0.5 * (c.max(axis=0) - c.min(axis=0))))
                      for c in clouds)

    n_panels = n_steps + (1 if reference else 0)
    fig, axes = plt.subplots(1, n_panels, figsize=(panel_w * n_panels, panel_h))
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor("white")

    # Reference first, then the deployment sequence.
    deploy_axes = axes
    state_centres = centres
    if reference:
        render_reference(axes[0], rgb, extent, target_unit, scale, shift,
                         outline_offset, ref_rgb, ref_extent)
        _frame(axes[0], centres[-1], half)
        deploy_axes = axes[1:]
        state_centres = centres[:-1]

    for ax, state, f, centre in zip(deploy_axes, states, fractions, state_centres):
        render_texture_at(ax, state, deployed, rgb, extent, target_unit, scale, shift,
                          outline_offset, plan["sigma"], float(f) * float(psi), psi,
                          edge=None, fit=False)
        _frame(ax, centre, half)

    if labels is True:
        labels = default_strip_labels(n_steps, reference)

    size = label_size if label_size is not None else 0.055 * panel_h * 72.0
    # The band under the strip is sized from the type that goes in it, not from a fixed fraction of
    # the panel: as a fraction it held whatever the canvas happened to make it, which on a canvas
    # authored at printed size is less than one line.
    bottom = 0.004 if not labels else (1.7 * size + 4.0) / (panel_h * 72.0)
    fig.subplots_adjust(left=0.004, right=0.996, top=0.996, bottom=bottom, wspace=0.012)

    if labels:
        for ax, text in zip(axes, labels):
            ax.text(0.5, -0.012, text, transform=ax.transAxes, ha="center", va="top",
                    fontsize=size, color=label_colour, zorder=5)

    if rasterize:
        # Each panel is one clipped raster PER TILE -- several hundred of them -- and a PDF holding
        # them separately came out at 109 MB, which is why this figure has always shipped as a PNG.
        # The fix is not to abandon vector output but to flatten JUST the tile layer: below this
        # zorder everything collapses into ONE image per panel at save resolution, while the labels
        # above it stay real text.
        for ax in axes:
            ax.set_rasterization_zorder(2.5)

    if reference:
        _divider(fig, axes[0], axes[1])
    return fig


def _frame(ax, centre, half):
    """Fix the window on ``centre``. ``half`` is a scalar or a per-axis pair."""
    hx, hy = np.broadcast_to(np.asarray(half, dtype=float), (2,))
    ax.set_xlim(centre[0] - hx, centre[0] + hx)
    ax.set_ylim(centre[1] - hy, centre[1] + hy)
    ax.set_aspect("equal")
    ax.axis("off")


def _divider(fig, left_ax, right_ax, *, colour="#C7CBD1", lw=1.0, frac=0.78):
    """Hairline between the reference and the deployment sequence, in figure space."""
    from matplotlib.lines import Line2D

    lb = left_ax.get_position()
    rb = right_ax.get_position()
    x = 0.5 * (lb.x1 + rb.x0)
    mid = 0.5 * (lb.y0 + lb.y1)
    fig.add_artist(Line2D([x, x], [mid - frac * (lb.y1 - lb.y0) / 2,
                                   mid + frac * (lb.y1 - lb.y0) / 2],
                          color=colour, linewidth=lw, solid_capstyle="round"))


def render_frames(plan, flat_tiles, psi_star, colours, out_dir, *, n_frames=48,
                  dpi=110, figsize=(5.0, 6.5), make_gif=True):
    """Deployment sweep. One scalar drives it, so every frame is exact.

    Frames are centred on their own bounding box so the sheet grows in place
    instead of drifting off the spanning-tree root tile.
    """
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    fractions = np.linspace(0.0, 1.0, n_frames)

    extents = []
    states = []
    for f in fractions:
        tiles = np.asarray(deploy_tiles(plan, flat_tiles, float(f) * float(psi_star)))
        tiles = tiles - tiles.reshape(-1, 2).mean(axis=0)
        states.append(tiles)
        pts = tiles.reshape(-1, 2)
        extents.append(np.max(np.abs(pts)))
    half = float(np.max(extents)) * 1.04

    paths = []
    for i, tiles in enumerate(states):
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor("white")
        ax.add_collection(PolyCollection(_tile_polys(tiles), facecolors=colours,
                                         edgecolors="#1A1A1A", linewidths=0.12, zorder=2))
        ax.set_xlim(-half, half)
        ax.set_ylim(-half, half)
        ax.set_aspect("equal")
        ax.axis("off")
        p = os.path.join(out_dir, f"frame_{i:03d}.png")
        fig.savefig(p, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        paths.append(p)

    if make_gif:
        _write_gif(paths, os.path.join(out_dir, "deploy.gif"))
    return paths


def _write_gif(frame_paths, out_gif, duration_ms: float = 70, *, pingpong: bool = True,
               loop: int = 0):
    """GIF via PIL. Ping-pong by default, so the sheet opens and closes.

    Each frame is quantised to its own adaptive palette. That is invisible on flat tile
    colours and visible as flicker on a photographic texture, so a textured sequence
    wants few frames and a modest resolution.
    """
    from PIL import Image
    frames = [Image.open(p).convert("P", palette=Image.ADAPTIVE) for p in frame_paths]
    sequence = frames + frames[-2:0:-1] if pingpong else frames

    # PIL merges consecutive identical frames and loses part of their duration, which
    # shortens every hold. Collapse them here instead and carry the time explicitly.
    kept, durations = [], []
    for frame in sequence:
        if kept and frame.tobytes() == kept[-1].tobytes():
            durations[-1] += duration_ms
        else:
            kept.append(frame)
            durations.append(duration_ms)

    # GIF stores delays in centiseconds, so rounding each frame on its own drifts the
    # whole clip. Rounding the running total keeps the duration matched to the movie's.
    elapsed, durations = 0.0, list(durations)
    for i, d in enumerate(durations):
        start = round(elapsed / 10.0) * 10
        elapsed += d
        durations[i] = max(10, round(elapsed / 10.0) * 10 - start)

    kept[0].save(out_gif, save_all=True, append_images=kept[1:],
                 duration=durations, loop=loop, optimize=True)
    return out_gif


def _ease_fractions(n: int, ease: str = "smoothstep") -> Float[np.ndarray, "n"]:
    """Deployment fractions in [0, 1], eased or uniform.

    Smoothstep is exact at both ends, so the swept endpoints stay bit-identical to the
    closed sheet and to the reference deployed state.
    """
    if n < 1:
        return np.zeros(0)
    if n == 1:
        return np.ones(1)
    t = np.linspace(0.0, 1.0, n)
    if ease == "linear":
        return t
    if ease == "smoothstep":
        return t * t * (3.0 - 2.0 * t)
    raise ValueError(f"unknown ease {ease!r}; expected 'smoothstep' or 'linear'")


def motto_centre_mm(rgb, extent, target_unit, scale, shift, outline_offset,
                    threshold: int = 200):
    """Centre of the motto, in deployed millimetres.

    The book is the only white inside the seal: the upper band is black and the lower
    field is orange and black. Its two pages label as separate regions, each sitting off
    to one side, so the centroid is taken over all enclosed white at once rather than
    over the largest piece.
    """
    from matplotlib.path import Path as MplPath

    rgb = np.asarray(rgb)
    H, W = rgb.shape[:2]
    x_min, x_max, y_min, y_max = extent
    us = np.linspace(x_min, x_max, W)
    vs = np.linspace(y_max, y_min, H)           # rasters are drawn origin="upper"
    U, V = np.meshgrid(us, vs)

    seal_unit = to_artwork_frame(target_polygon_mm(target_unit, scale, shift, outline_offset),
                                 target_unit, scale, shift, outline_offset)
    inside = MplPath(seal_unit).contains_points(
        np.column_stack([U.ravel(), V.ravel()])).reshape(H, W)
    white = (rgb > threshold).all(axis=2) & inside
    if not white.any():
        raise ValueError("no enclosed white found -- is this the seal artwork?")

    rows, cols = np.where(white)
    unit_to_mm = _unit_to_mm(target_unit, scale, shift, outline_offset)
    return np.asarray(unit_to_mm.transform_point(
        (us[int(round(cols.mean()))], vs[int(round(rows.mean()))])), dtype=float)


def _animation_schedule(psi: float, *, n_frames: int = 120, wipe_frames: int = 36,
                        hold_closed: int = 6, hold_open: int = 12, hold_logo: int = 24,
                        ease: str = "smoothstep", style: str = "wipe",
                        clean_frames: int = 12):
    """One ``(kind, a, b)`` triple per frame.

    ``("tiles", psi_state, _)`` draws the sheet at that opening angle. The ending then
    differs by style. Under ``wipe`` the finished shield spreads outward from the motto as
    ``("wipe", progress, _)``: behind the front the voids are already filled and the sheet
    spilling past the silhouette is already gone, so nothing is left to tidy afterwards.
    Under ``fade`` the artwork instead fades in everywhere at once and the sheet fades out
    after it, which leaves that spill standing until a second, separate beat removes it.
    """
    psi = float(psi)
    schedule = [("tiles", 0.0, 1.0)] * int(hold_closed)
    schedule += [("tiles", float(f) * psi, 1.0) for f in _ease_fractions(n_frames, ease)]
    schedule += [("tiles", psi, 1.0)] * int(hold_open)

    if style == "wipe":
        schedule += [("wipe", float(a), 0.0) for a in _ease_fractions(wipe_frames, "linear")]
        schedule += [("wipe", 1.0, 0.0)] * int(hold_logo)
    elif style == "fade":
        schedule += [("fade", float(a), 1.0) for a in _ease_fractions(wipe_frames, ease)]
        schedule += [("fade", 1.0, float(1.0 - a)) for a in _ease_fractions(clean_frames, ease)]
        schedule += [("fade", 1.0, 0.0)] * int(hold_logo)
    else:
        raise ValueError(f"unknown style {style!r}; expected 'wipe' or 'fade'")
    return schedule


def _wipe_mask(dist_px, radius_px: float, feather_px: float):
    """Soft radial front: 1 behind it, 0 ahead of it."""
    t = np.clip((radius_px - np.asarray(dist_px)) / max(feather_px, 1e-6) + 0.5, 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t))[..., None]


def _pixel_of(point_mm, centre, half, width_px: int, height_px: int):
    """Data millimetres to pixel coordinates within the rendered canvas."""
    x = (float(point_mm[0]) - (centre[0] - half[0])) / (2.0 * half[0]) * width_px
    y = ((centre[1] + half[1]) - float(point_mm[1])) / (2.0 * half[1]) * height_px
    return np.array([x, y], dtype=float)


def _fixed_camera(deployed, target_mm, figsize, margin: float = 0.03):
    """Window covering the open state and the artwork, at the canvas aspect.

    The artwork outline reaches past the deployed tiles, so a window sized on the tiles
    alone clips the seal once it fades in. Every intermediate state sits inside the open
    state's own box, which makes this the union over the whole sweep.
    """
    pts = np.concatenate([np.asarray(deployed).reshape(-1, 2),
                          np.asarray(target_mm).reshape(-1, 2)], axis=0)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    centre = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo) * (1.0 + margin)

    aspect = float(figsize[0]) / float(figsize[1])
    if half[0] / half[1] < aspect:
        half[0] = half[1] * aspect
    else:
        half[1] = half[0] / aspect
    return centre, half


def _write_mp4(out_dir, out_mp4, *, fps: int = 30, crf: int = 18, stem: str = "frame"):
    """H.264 encode of the PNG sequence, playable outside a browser.

    ``yuv420p`` and even dimensions are what QuickTime, Keynote and PowerPoint require;
    the scale filter supplies the parity the canvas size does not guarantee.
    """
    import shutil
    import subprocess

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH -- `brew install ffmpeg`, "
                           "or ask for the GIF instead")
    cmd = ["ffmpeg", "-y", "-v", "error",
           "-framerate", str(fps), "-start_number", "0",
           "-i", os.path.join(out_dir, f"{stem}_%04d.png"),
           "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2:flags=lanczos,format=yuv420p",
           "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-r", str(fps),
           out_mp4]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg failed ({exc.returncode}):\n{exc.stderr[-2000:]}") from exc
    return out_mp4


def render_deploy_animation(plan, flat_tiles, deployed, rgb, extent, target_unit, scale,
                            shift, outline_offset, psi, out_dir, *, n_frames: int = 120,
                            wipe_frames: int = 36, hold_closed: int = 6,
                            hold_open: int = 12, hold_logo: int = 24,
                            clean_frames: int = 12, style: str = "wipe",
                            wipe_centre="motto", wipe_feather: float = 25.0,
                            wipe_rate: str = "area",
                            ease: str = "smoothstep", dpi: int = 150,
                            figsize=(5.4, 6.5), margin: float = 0.03, edge=None,
                            lw: float = 0.06, background: str = "white",
                            stem: str = "deploy", mp4: bool = True, gif: bool = False,
                            fps: int = 30, crf: int = 18, keep_frames: bool = True,
                            progress: bool = True) -> dict:
    """Textured deployment sweep that resolves into the undeformed artwork.

    The camera is fixed on the open state and the artwork together, so the sheet grows in
    place, the last frame registers with the artwork, and every PNG carries identical
    pixel dimensions -- which ffmpeg and ``animategraphics`` both require.

    Under the default ``wipe`` style the finished shield spreads outward from the motto.
    Behind the front the artwork is already whole and the sheet is already gone, so the
    tiles reaching past the silhouette are swallowed as the front passes them instead of
    being cleared afterwards. The front only chooses which of two fixed images wins at
    each pixel, so it costs two renders and a per-frame blend rather than a render a frame.
    """
    import shutil

    import matplotlib.pyplot as plt
    from PIL import Image

    os.makedirs(out_dir, exist_ok=True)
    schedule = _animation_schedule(float(psi), n_frames=n_frames, wipe_frames=wipe_frames,
                                   hold_closed=hold_closed, hold_open=hold_open,
                                   hold_logo=hold_logo, ease=ease, style=style,
                                   clean_frames=clean_frames)
    if len(schedule) > 9999:
        raise ValueError(f"{len(schedule)} frames exceeds the %04d naming; ask for fewer")

    target_mm = target_polygon_mm(target_unit, scale, shift, outline_offset)
    centre, half = _fixed_camera(deployed, target_mm, figsize, margin=margin)
    sigma = np.asarray(plan["sigma"])

    if isinstance(wipe_centre, str) and wipe_centre == "motto":
        origin_mm = motto_centre_mm(rgb, extent, target_unit, scale, shift, outline_offset)
    elif isinstance(wipe_centre, str) and wipe_centre == "shield":
        origin_mm = np.asarray(centre, dtype=float)
    else:
        origin_mm = np.asarray(wipe_centre, dtype=float)

    def _draw_tiles(psi_state, alpha=1.0):
        state = np.asarray(deploy_tiles(plan, flat_tiles, float(psi_state)))
        render_texture_at(ax, state, deployed, rgb, extent, target_unit, scale, shift,
                          outline_offset, sigma, float(psi_state), float(psi),
                          edge=edge, lw=lw, fit=False, alpha=alpha)

    paths, drawn = [], {}
    pattern_path, blend = None, None
    fig, ax = plt.subplots(figsize=figsize)
    try:
        fig.patch.set_facecolor(background)
        fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
        for i, entry in enumerate(schedule):
            path = os.path.join(out_dir, f"frame_{i:04d}.png")
            key = tuple(entry[:1]) + tuple(round(float(v), 12) for v in entry[1:])
            if key in drawn:                      # a hold repeats a frame already drawn
                shutil.copyfile(drawn[key], path)
                paths.append(path)
                continue
            if progress and (i % 20 == 0 or i == len(schedule) - 1):
                print(f"  frame {i + 1}/{len(schedule)}", flush=True)

            kind, a, b = entry
            if kind == "wipe":
                if blend is None:
                    blend = _prepare_wipe(fig, ax, out_dir, pattern_path, rgb, extent,
                                          target_unit, scale, shift, outline_offset,
                                          deployed, origin_mm, centre, half, wipe_feather,
                                          dpi, background)
                Image.fromarray(_blend_wipe(blend, a, wipe_rate)).save(path)
            else:
                ax.clear()
                if kind == "tiles" or b > 0.0:
                    _draw_tiles(psi if kind == "fade" else a, alpha=b)
                if kind == "fade" and a > 0.0:
                    render_reference(ax, rgb, extent, target_unit, scale, shift,
                                     outline_offset, alpha=a, zorder=5)
                _frame(ax, centre, half)
                fig.savefig(path, dpi=dpi, facecolor=background)
                if kind == "tiles":
                    pattern_path = path
            drawn[key] = path
            paths.append(path)
    finally:
        plt.close(fig)

    out_mp4 = out_gif = None
    if mp4:
        out_mp4 = _write_mp4(out_dir, os.path.join(out_dir, f"{stem}.mp4"),
                             fps=fps, crf=crf, stem="frame")
    if gif:
        out_gif = _write_gif(paths, os.path.join(out_dir, f"{stem}.gif"),
                             duration_ms=1000.0 / fps, pingpong=False, loop=0)
    if blend is not None and not keep_frames:
        os.remove(blend["path"])
    if not keep_frames:
        for p in paths:
            os.remove(p)
        paths = []

    return {"frames": paths, "mp4": out_mp4, "gif": out_gif, "centre": centre,
            "half": half, "schedule": schedule, "origin_mm": origin_mm}


def _prepare_wipe(fig, ax, out_dir, pattern_path, rgb, extent, target_unit, scale, shift,
                  outline_offset, deployed, origin_mm, centre, half, feather_mm, dpi,
                  background):
    """The two images the front selects between, and the distances that drive it.

    ``pattern`` is the open sheet, already on disk as the last frame drawn before the
    front starts; only the finished artwork has to be rendered here.
    """
    from PIL import Image

    ax.clear()
    render_reference(ax, rgb, extent, target_unit, scale, shift, outline_offset,
                     alpha=1.0, zorder=5)
    _frame(ax, centre, half)
    perfect_path = os.path.join(out_dir, "_perfect.png")
    fig.savefig(perfect_path, dpi=dpi, facecolor=background)

    pattern = np.asarray(Image.open(pattern_path).convert("RGB"), dtype=float)
    perfect = np.asarray(Image.open(perfect_path).convert("RGB"), dtype=float)
    height_px, width_px = perfect.shape[:2]

    origin_px = _pixel_of(origin_mm, centre, half, width_px, height_px)
    yy, xx = np.mgrid[0:height_px, 0:width_px]
    dist = np.hypot(xx - origin_px[0], yy - origin_px[1])

    mm_per_px = 2.0 * half[0] / width_px
    reach_mm = float(np.max(np.hypot(*(np.asarray(deployed).reshape(-1, 2) - origin_mm).T)))
    return {"pattern": pattern, "perfect": perfect, "dist": dist,
            "reach_px": reach_mm / mm_per_px, "feather_px": feather_mm / mm_per_px,
            "path": perfect_path}


def _blend_wipe(blend, fraction: float, rate: str = "area"):
    """The frame at one position of the front, composited in float to avoid banding.

    The front sweeps a radius but reveals an area, which grows as its square. Advancing
    the radius uniformly would therefore uncover the shield ever faster; ``area`` holds
    the revealed area to a steady rate instead, which matters here because the motto sits
    high in the seal and the outline is 3.6x further away at the bottom point than at the
    top.
    """
    fraction = float(fraction)
    if rate == "area":
        fraction = np.sqrt(fraction)
    elif rate == "smoothstep":
        fraction = fraction * fraction * (3.0 - 2.0 * fraction)
    elif rate != "linear":
        raise ValueError(f"unknown rate {rate!r}; expected 'area', 'linear' or 'smoothstep'")

    # The front must finish a clear feather PAST the outermost tile vertex: ink is
    # antialiased a little beyond it, and the mask only saturates behind the trailing
    # edge. Stopping exactly on the vertex leaves a few stray pixels alive forever.
    span = blend["reach_px"] + 2.0 * blend["feather_px"]
    radius = -0.5 * blend["feather_px"] + fraction * span
    mask = _wipe_mask(blend["dist"], radius, blend["feather_px"])
    out = mask * blend["perfect"] + (1.0 - mask) * blend["pattern"]
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


# ── Public names for the two helpers that carry no shield-specific state. ──────────────────
# Both are generic enough that a second animation (the training film) reuses them verbatim;
# exporting them here keeps that import off a private name.
write_mp4 = _write_mp4
ease_fractions = _ease_fractions
