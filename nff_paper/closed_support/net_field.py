"""Lay a continuous komon net over the sheet and let the tessellation deform it.

The control mesh is the grid of TILE CENTROIDS. Two reasons, one structural and one physical.

Structurally it is conforming: cell ``(i, j)`` spans ``c[i,j], c[i+1,j], c[i+1,j+1], c[i,j+1]``, so
two neighbouring cells share a whole edge and each computes it as the same interpolation of the same
two centroids. The panel tiles cannot do this -- they meet in T-junctions, which is why every motif
before this one had to stay inside a single tile.

Physically the centroid displacement field IS the sheet's deformation, so a net drawn regular in
reference coordinates and pushed through this mesh is warped by the mechanics and by nothing else.
"""

import numpy as np
from jaxtyping import Float

from nff_paper.topology_art.net import build_net_fn, warp_net_fn
from nff_paper.topology_art.quad_warp import path_points_fn


def control_mesh_fn(tiles: Float[np.ndarray, "n 4 2"], grid) -> Float[np.ndarray, "rows cols 2"]:
    """Tile centroids as a conforming quad mesh, and a check that its cells are usable.

    The bilinear warp needs convex cells. Measured across every field in the study these come out
    convex, but a large enough amplitude would eventually fold one, and a folded cell warps the net
    inside out rather than failing, so it is checked rather than assumed.
    """
    across, along = grid
    # Tile t is (i, j) with t = i*N + j, and it is i that runs along the sheet's x. The warper wants
    # the first mesh index to be y, so the grid is transposed once, here, rather than everywhere.
    mesh = np.asarray(tiles, float).mean(axis=1).reshape(across, along, 2).transpose(1, 0, 2)

    corners = np.stack([mesh[:-1, :-1], mesh[:-1, 1:], mesh[1:, 1:], mesh[1:, :-1]], axis=2)
    edges = np.roll(corners, -1, axis=2) - corners
    previous = np.roll(edges, 1, axis=2)
    cross = previous[..., 0] * edges[..., 1] - previous[..., 1] * edges[..., 0]
    folded = int((~((cross > 0).all(axis=2) | (cross < 0).all(axis=2))).sum())
    if folded:
        raise ValueError(f"{folded} control cells are not convex; the bilinear warp would fold")
    return mesh


def build_net_sheet_fn(tiles, grid, name, *, scale=2.0, **params):
    """The deformed net over one sheet.

    Args:
        tiles: (n, 4, 2) flat tiles.
        grid: ``(M, N)`` panel grid.
        name: which net; ``nff_paper.topology_art.net.NETS`` lists them.
        scale: control-mesh cells per motif repeat.

    Returns:
        ``(paths, report)`` with paths in sheet coordinates as exact Beziers.
    """
    mesh = control_mesh_fn(tiles, grid)
    rows, cols = mesh.shape[0] - 1, mesh.shape[1] - 1
    strokes = build_net_fn(name, cols, rows, scale=scale, **params)
    paths = warp_net_fn(strokes, mesh)

    flat = mesh.reshape(-1, 2)
    corners = np.stack([mesh[:-1, :-1], mesh[:-1, 1:], mesh[1:, 1:], mesh[1:, :-1]], axis=2)
    edges = np.roll(corners, -1, axis=2) - corners
    previous = np.roll(edges, 1, axis=2)
    areas = 0.5 * np.abs((previous[..., 0] * edges[..., 1]
                          - previous[..., 1] * edges[..., 0]).sum(axis=2))
    report = {
        "n_strokes": len(strokes),
        "n_pieces": len(paths),
        "cells": rows * cols,
        "cell_area_spread": float(areas.max() / max(areas.min(), 1e-12)),
        "mesh_bounds": (flat[:, 0].min(), flat[:, 1].min(), flat[:, 0].max(), flat[:, 1].max()),
    }
    return paths, report


def net_join_residual_fn(paths, *, tol=1e-6) -> float:
    """How far apart two pieces that should meet actually land.

    The whole construction rests on neighbouring cells agreeing on their shared edge, so this is the
    invariant worth measuring rather than trusting. Endpoints are matched to their nearest partner
    and the worst gap among genuinely coincident pairs is returned.
    """
    if not paths:
        return 0.0
    ends = np.array([[np.asarray(path.start, float), np.asarray(path.endpoint, float)]
                     for path in paths]).reshape(-1, 2)
    from scipy.spatial import cKDTree

    tree = cKDTree(ends)
    pairs = tree.query_pairs(tol, output_type="ndarray")
    if not len(pairs):
        return 0.0
    gaps = np.linalg.norm(ends[pairs[:, 0]] - ends[pairs[:, 1]], axis=1)
    return float(gaps.max())


def net_region_fn(paths, width, *, page=None, per_segment=10):
    """The net given a strut width: the region a blade would leave.

    A net of struts is a single connected piece by construction, which is the whole attraction of
    this construction over a field of separate elements -- the connectivity needs no tuning.
    """
    from shapely.geometry import LineString, box
    from shapely.ops import unary_union

    # Round caps, not flat. The net arrives as separate pieces, one per cell, and two flat-capped
    # pieces meeting at an angle leave a notch on the outside of the turn that pinches the strut to
    # well under its own width -- measured at half of it. A round cap fills the corner.
    lines = [LineString(path_points_fn(path, per_segment=per_segment)) for path in paths]
    region = unary_union([line.buffer(0.5 * width, cap_style=1, join_style=1) for line in lines])
    if page is not None:
        region = region.intersection(box(*page))
    return region


def render_net_preview_fn(paths, out_png, *, width_mm=1.6, crop=None, dpi=150,
                          fill="#F2EFE6", ground="#141414", figure_width_in=7.0, per_segment=10,
                          elements=None, specs=None):
    """Raster preview: the net as struts of constant width on the ground.

    Width is a stroke width, never a geometric offset, so it stays exactly constant across the sheet
    however the cells are stretched -- the same rule the rest of the artwork follows.
    """
    from nff.utils import figstyle

    figstyle.use_agg()
    import matplotlib.pyplot as plt

    figstyle.apply_charter(**{"axes.grid": False})
    sampled = [path_points_fn(path, per_segment=per_segment) for path in paths]
    if crop is None:
        stacked = np.vstack(sampled)
        crop = (stacked[:, 0].min(), stacked[:, 1].min(), stacked[:, 0].max(), stacked[:, 1].max())
    span_x, span_y = crop[2] - crop[0], crop[3] - crop[1]

    figure, axis = plt.subplots(
        figsize=(figure_width_in, figure_width_in * span_y / max(span_x, 1e-9)))
    figure.patch.set_facecolor(ground)
    axis.set_facecolor(ground)
    # Convert a millimetre width into points at this figure's own scale, so the strut prints the
    # width it claims.
    points_per_mm = 72.0 / 25.4
    linewidth = width_mm * points_per_mm * (figure_width_in * 25.4 / max(span_x, 1e-9))
    for points in sampled:
        axis.plot(points[:, 0], points[:, 1], color=fill, linewidth=linewidth,
                  solid_capstyle="round", solid_joinstyle="round")

    if elements:
        # The two-scale komon: the net is the fine ground and the figures sit on it. Painted after
        # the net, so a figure covers the ground rather than being cut up by it.
        from matplotlib.patches import Polygon as MplPolygon

        from nff_paper.topology_art.leaf import ElementSpec, element_layers_fn, leaf_polygon_fn

        ink = {"pale": fill, "dark": ground}
        default = ElementSpec()
        for element in elements:
            for leaf, tag in element_layers_fn(element.leaf,
                                               (specs or {}).get(element.population, default)):
                axis.add_patch(MplPolygon(leaf_polygon_fn(leaf, n=64), closed=True,
                                          facecolor=ink[tag], edgecolor="none"))
    axis.set_xlim(crop[0], crop[2])
    axis.set_ylim(crop[1], crop[3])
    axis.set_aspect("equal")
    axis.axis("off")
    figure.savefig(out_png, dpi=dpi, bbox_inches="tight", pad_inches=0.0, facecolor=ground)
    plt.close(figure)
    return out_png
