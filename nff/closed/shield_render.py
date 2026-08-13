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
                      edge=None, lw=0.1, fit=True):
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
        if edge is not None:
            ax.add_patch(MplPolygon(state[k], closed=True, facecolor="none",
                                    edgecolor=edge, linewidth=lw, zorder=3))
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
                     ref_rgb=None, ref_extent=None):
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
                   interpolation="bilinear", zorder=2)
    im.set_clip_path(patch)
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
    ax.set_xlim(centre[0] - half, centre[0] + half)
    ax.set_ylim(centre[1] - half, centre[1] + half)
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


def _write_gif(frame_paths, out_gif, duration_ms: int = 70):
    """Ping-pong GIF via PIL, so the sheet opens and closes."""
    from PIL import Image
    frames = [Image.open(p).convert("P", palette=Image.ADAPTIVE) for p in frame_paths]
    sequence = frames + frames[-2:0:-1]
    frames[0].save(out_gif, save_all=True, append_images=sequence[1:],
                   duration=duration_ms, loop=0, optimize=True)
    return out_gif
