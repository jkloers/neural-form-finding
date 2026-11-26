"""Target extraction: an SVG logo into an outline and a colour raster.

Produces two things in one shared, normalised frame so that a point matched
against the outline can also be used to sample the artwork:

    outline  (n, 2)      ordered closed silhouette, centred, unit height
    raster   (H, W, 3)   RGB artwork with its extent in the same frame

The vector paths are parsed directly rather than going through a rasteriser --
macOS Quick Look stretches the artwork into a square canvas, which distorted the
shield from its true 803:1024 aspect to 0.936:1. Parsing also means the
silhouette carries no marching-squares error.

The silhouette is the union of the non-white paths, so interior artwork (the open
book, the chevrons) contributes colour but never geometry.
"""

import re
import xml.etree.ElementTree as ET

import numpy as np
from jaxtyping import Float
from matplotlib.path import Path as MplPath
from shapely.geometry import Polygon
from shapely.ops import unary_union

_NUM = re.compile(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?")
_CMD = re.compile(r"([MmLlCcZz])")


def _parse_path_data(d: str):
    """Parse an absolute ``M``/``L``/``C``/``Z`` path into matplotlib vertices and codes.

    Only the command set this artwork uses is supported; anything else raises
    rather than silently dropping geometry.
    """
    vertices, codes = [], []
    tokens = [t.strip() for t in _CMD.split(d) if t.strip()]

    i = 0
    while i < len(tokens):
        cmd = tokens[i]
        args = [float(v) for v in _NUM.findall(tokens[i + 1])] if i + 1 < len(tokens) else []
        i += 2 if (i + 1 < len(tokens) and not _CMD.fullmatch(tokens[i + 1])) else 1

        if cmd == "M":
            for k in range(0, len(args), 2):
                vertices.append((args[k], args[k + 1]))
                codes.append(MplPath.MOVETO if k == 0 else MplPath.LINETO)
        elif cmd == "L":
            for k in range(0, len(args), 2):
                vertices.append((args[k], args[k + 1]))
                codes.append(MplPath.LINETO)
        elif cmd == "C":
            for k in range(0, len(args), 6):
                vertices.extend([(args[k], args[k + 1]), (args[k + 2], args[k + 3]),
                                 (args[k + 4], args[k + 5])])
                codes.extend([MplPath.CURVE4] * 3)
        elif cmd in "Zz":
            vertices.append((0.0, 0.0))
            codes.append(MplPath.CLOSEPOLY)
        else:
            raise ValueError(f"unsupported SVG path command {cmd!r}")

    return np.array(vertices, dtype=float), np.array(codes, dtype=np.uint8)


def parse_svg_paths(svg_path: str):
    """Parse filled paths in document (paint) order, in a y-up world frame.

    Returns:
        paths: list of ``{"path": MplPath, "fill": (r, g, b) floats in [0,1]}``.
        viewbox: ``(x_min, y_min, width, height)`` of the SVG, y-up.
    """
    tree = ET.parse(svg_path)
    root = tree.getroot()
    vb = [float(v) for v in root.get("viewBox").split()]

    flip = np.array([1.0, -1.0])
    out = []
    for el in root.iter():
        if not el.tag.endswith("path"):
            continue
        d = el.get("d")
        if not d:
            continue
        fill = el.get("fill", "#000000")
        if fill.lower() in ("none",):
            continue
        verts, codes = _parse_path_data(d)
        out.append({"path": MplPath(verts * flip, codes), "fill": _hex_to_rgb(fill)})

    return out, (vb[0], -(vb[1] + vb[3]), vb[2], vb[3])


def _hex_to_rgb(s: str):
    s = s.strip().lstrip("#")
    return tuple(int(s[k:k + 2], 16) / 255.0 for k in (0, 2, 4))


def _silhouette_polygon(paths, white_tol: float = 0.95):
    """Union of the non-white filled paths -- the shield body."""
    polys = []
    for entry in paths:
        if min(entry["fill"]) > white_tol:
            continue
        for ring in entry["path"].to_polygons(closed_only=False):
            if len(ring) >= 4:
                p = Polygon(ring)
                if p.is_valid and p.area > 0:
                    polys.append(p)
                elif not p.is_valid:
                    polys.append(p.buffer(0))
    merged = unary_union(polys)
    if merged.geom_type == "MultiPolygon":
        merged = max(merged.geoms, key=lambda g: g.area)
    return merged


def _normaliser(bounds):
    """Map into the frame where the silhouette is centred with unit height."""
    x0, y0, x1, y1 = bounds
    h = y1 - y0
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    return lambda pts: (np.asarray(pts, dtype=float) - np.array([cx, cy])) / h, (cx, cy, h)


def load_shield_outline(svg_path: str, n_points: int = 480) -> Float[np.ndarray, "n 2"]:
    """Ordered outer silhouette, centred at the origin with unit height.

    Resampled at uniform arclength so the Chamfer target is not biased toward
    regions where the vector path happened to place dense control points.
    """
    paths, _ = parse_svg_paths(svg_path)
    body = _silhouette_polygon(paths)
    norm, _ = _normaliser(body.bounds)
    ring = norm(np.asarray(body.exterior.coords))
    return _resample_closed(ring, n_points)


def _resample_closed(pts, n_points: int) -> Float[np.ndarray, "n 2"]:
    """Uniform-arclength resampling of a closed polyline."""
    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    target = np.linspace(0.0, s[-1], n_points, endpoint=False)
    return np.stack([np.interp(target, s, pts[:, 0]), np.interp(target, s, pts[:, 1])], axis=1)


def _bleed_outward(rgb, inside_mask, erode_px: int = 2):
    """Extend the artwork past its silhouette with the nearest interior colour.

    The artwork is rendered on white, so a tile sitting on the silhouette edge picks
    up background white -- and the antialiased edge itself is part white. That paints
    a white rim onto real material. Replacing everything outside (and the antialiased
    fringe just inside) with the nearest strictly-interior colour removes the rim
    without touching the geometry or any fit metric.
    """
    from scipy.ndimage import binary_erosion, distance_transform_edt

    core = binary_erosion(inside_mask, iterations=max(1, erode_px))
    if not core.any():
        return rgb
    _, (ri, ci) = distance_transform_edt(~core, return_indices=True)
    return rgb[ri, ci]


def load_shield_raster(svg_path: str, px: int = 2400, bleed: bool = True):
    """Artwork rendered to RGB, with its extent in the outline's frame.

    Rendered from the parsed vector paths in paint order, so the raster and the
    outline are guaranteed to share a frame and an aspect ratio.

    Returns:
        rgb: (H, W, 3) uint8.
        extent: ``(x_min, x_max, y_min, y_max)`` for ``imshow``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import PathPatch
    from matplotlib.transforms import Affine2D

    paths, _ = parse_svg_paths(svg_path)
    body = _silhouette_polygon(paths)
    norm, (cx, cy, h) = _normaliser(body.bounds)
    to_norm = Affine2D().translate(-cx, -cy).scale(1.0 / h)

    x0, y0, x1, y1 = body.bounds
    pad = 0.02 * h
    ext_data = (x0 - pad, x1 + pad, y0 - pad, y1 + pad)
    aspect = (ext_data[1] - ext_data[0]) / (ext_data[3] - ext_data[2])

    W = max(8, int(round(px * aspect)))
    fig = plt.figure(figsize=(W / 100.0, px / 100.0), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(ext_data[0], ext_data[1])
    ax.set_ylim(ext_data[2], ext_data[3])
    ax.set_facecolor("white")
    ax.axis("off")
    for entry in paths:
        ax.add_patch(PathPatch(entry["path"], facecolor=entry["fill"], edgecolor="none",
                               antialiased=True))
    fig.canvas.draw()
    rgb = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()
    plt.close(fig)

    if bleed:
        H, W = rgb.shape[:2]
        xs = np.linspace(ext_data[0], ext_data[1], W)
        ys = np.linspace(ext_data[3], ext_data[2], H)          # row 0 is the top
        gx, gy = np.meshgrid(xs, ys)
        ring_data = np.asarray(body.exterior.coords)
        inside = MplPath(ring_data).contains_points(
            np.stack([gx.reshape(-1), gy.reshape(-1)], axis=1)).reshape(H, W)
        rgb = _bleed_outward(rgb, inside)

    corners = norm(np.array([[ext_data[0], ext_data[2]], [ext_data[1], ext_data[3]]]))
    extent = (float(corners[0, 0]), float(corners[1, 0]),
              float(corners[0, 1]), float(corners[1, 1]))
    return rgb, extent


def load_shield_sdf(svg_path: str, px: int = 900):
    """Signed distance to the silhouette in the outline's frame, positive OUTSIDE.

    Used to keep every tile inside the target. Without it the optimiser parks
    boundary tiles outside the shield, where they sample background white and
    disappear against a white page -- a silhouette that looks fitted but is not.

    Returns:
        sdf: (H, W) float, unit-frame distance, positive outside the shield.
        extent: ``(x_min, x_max, y_min, y_max)`` matching the raster convention.
    """
    from scipy.ndimage import distance_transform_edt

    paths, _ = parse_svg_paths(svg_path)
    body = _silhouette_polygon(paths)
    norm, (cx, cy, h) = _normaliser(body.bounds)

    ring = norm(np.asarray(body.exterior.coords))
    pad = 0.06
    x_min, x_max = ring[:, 0].min() - pad, ring[:, 0].max() + pad
    y_min, y_max = ring[:, 1].min() - pad, ring[:, 1].max() + pad

    W = max(8, int(round(px * (x_max - x_min) / (y_max - y_min))))
    xs = np.linspace(x_min, x_max, W)
    ys = np.linspace(y_max, y_min, px)                       # row 0 is y_max
    gx, gy = np.meshgrid(xs, ys)

    inside = MplPath(ring).contains_points(
        np.stack([gx.reshape(-1), gy.reshape(-1)], axis=1)).reshape(px, W)

    px_size = (y_max - y_min) / px
    d_out = distance_transform_edt(~inside) * px_size
    d_in = distance_transform_edt(inside) * px_size
    return (d_out - d_in).astype(float), (x_min, x_max, y_min, y_max)


def sample_artwork(rgb, extent, points: Float[np.ndarray, "n 2"],
                   background=(255, 255, 255)) -> Float[np.ndarray, "n 3"]:
    """Nearest-neighbour colour lookup at world-frame points, clamped to background."""
    x_min, x_max, y_min, y_max = extent
    H, W = rgb.shape[:2]
    pts = np.asarray(points, dtype=float)

    col = (pts[:, 0] - x_min) / (x_max - x_min) * W - 0.5
    row = (y_max - pts[:, 1]) / (y_max - y_min) * H - 0.5
    ci = np.rint(col).astype(int)
    ri = np.rint(row).astype(int)

    inside = (ci >= 0) & (ci < W) & (ri >= 0) & (ri < H)
    out = np.tile(np.asarray(background, dtype=np.uint8), (len(pts), 1))
    out[inside] = rgb[ri[inside], ci[inside]]
    return out
