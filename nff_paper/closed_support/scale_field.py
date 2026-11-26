"""Turn the flat tiling into a flowing field of overlapping scales.

The tessellation stops carrying a motif per cell here. Instead it supplies three fields -- a
direction, a size and a packing density -- over which one element is scattered freely. Elements
straddle cells and ignore them, so the tiling governs the picture without ever being visible as
cells, which is how the reference stencils work.

The direction comes from each tile's principal axis. These tiles are the output of inverse design
under load, so their elongation encodes the mechanics; the flow the artwork follows is therefore the
sheet's own deformation field rather than an ornament laid over it.

Angles are smoothed as DOUBLED angles. A tile's long axis is a line, not an arrow, so 179 degrees and
1 degree are neighbours; averaging the raw angles would tear the field along an arbitrary seam.
"""

from typing import NamedTuple

import numpy as np
from jaxtyping import Float

from nff_paper.closed_support.katagami import shared_corners_fn, tile_areas_fn
from nff_paper.topology_art.quad_warp import bilinear_fn
from nff_paper.topology_art.leaf import (
    ElementSpec,
    Leaf,
    element_layers_fn,
    leaf_polygon_fn,
    transform_leaf_fn,
)


# Paper and hole, in that order. A palette substitutes into roles that already exist -- the 'pale'
# tag is always the paper and 'dark' always the hole -- so it composes with `invert` rather than
# competing with it. Kakishibu is what an Ise-katagami stencil actually is: mulberry paper cured in
# persimmon tannin, read against the light.
PALETTES = {
    "sumi": {"fill": "#F2EFE6", "ground": "#141414", "edge": "#141414"},
    "kakishibu": {"fill": "#8C5230", "ground": "#F7F1E4", "edge": "#F7F1E4"},
    "ai": {"fill": "#24384F", "ground": "#EDE6D6", "edge": "#EDE6D6"},
}


class Element(NamedTuple):
    """One placed element, which population it belongs to, and which tile it came from.

    The population decides only how the element is inked, never where it sits. A sheet may carry a
    main figure over a finer ground, and the renderer needs to tell them apart without the field
    having to know what ink either will be given. The tile index is what lets a connectivity
    measurement ask whether two fused elements are lattice neighbours or merely near each other.
    """

    leaf: Leaf
    population: str = "main"
    tile: int = -1


def transform_elements_fn(elements, *, scale: float, offset) -> list:
    """Apply one similarity to a whole field, at the parameter level."""
    return [Element(transform_leaf_fn(e.leaf, scale=scale, offset=offset), e.population, e.tile)
            for e in elements]


def _lattice_sites(tiles, grid, *, bond, regularity, edge_looseness, edge_width=1.5, rng=None):
    """One site per tile, alternate rows offset, disorder growing toward the sheet edge.

    The tiling is combinatorially a perfect grid whose metric alone was deformed, so the lattice is
    already present in the tile indexing and does not have to be constructed. Placing sites on it
    gives alignment that is regular in INCIDENCE -- every element meets the same neighbours -- while
    remaining irregular in shape, which is what carries the flow.

    The row offset is taken in the unit cell rather than as a distance. A half-cell in parameter
    space lands correctly however the tile is deformed; half a millimetre would not.

    Returns:
        ``(points, tile_of, local_regularity)``.
    """
    rows, cols = grid
    index = np.arange(len(tiles))
    row, column = index // cols, index % cols

    # Disorder is allowed to grow near the boundary: the body of the sheet stays calm and the edges
    # drift, so the eye reads control without the pattern becoming a stated rule.
    depth = np.minimum(np.minimum(row, rows - 1 - row), np.minimum(column, cols - 1 - column))
    local = np.clip(regularity - edge_looseness * np.exp(-depth / edge_width), 0.0, 1.0)

    offset = 0.5 * bond * (row % 2)
    uv = np.stack([np.full(len(tiles), 0.5) + offset, np.full(len(tiles), 0.5)], axis=1)
    points = np.array([bilinear_fn(tiles[t], uv[t : t + 1])[0] for t in index])

    if rng is not None:
        spacing = np.sqrt(tile_areas_fn(tiles))
        drift = (1.0 - local)[:, None] * spacing[:, None] * rng.normal(size=(len(tiles), 2))
        points = points + 0.5 * drift
    return points, index, local


def _covariance(tiles):
    centred = tiles - tiles.mean(axis=1, keepdims=True)
    return np.einsum("nki,nkj->nij", centred, centred) / 4.0


def tile_axes_fn(tiles: Float[np.ndarray, "n 4 2"]) -> Float[np.ndarray, "n"]:
    """Angle of each tile's long principal axis, from the covariance of its four corners."""
    covariance = _covariance(tiles)
    # Closed form for the principal direction of a symmetric 2x2, as a doubled angle.
    xx, yy, xy = covariance[:, 0, 0], covariance[:, 1, 1], covariance[:, 0, 1]
    return 0.5 * np.arctan2(2.0 * xy, xx - yy)


def tile_anisotropy_fn(tiles: Float[np.ndarray, "n 4 2"]) -> Float[np.ndarray, "n"]:
    """How elongated each tile is: the ratio of its principal extents, at least 1.

    The direction field already reads the tile's long axis. Its elongation is the other half of the
    same measurement, and it is thrown away today because the element's shape is held constant.
    Letting the shape read it too puts the sheet's deformation into what the elements ARE, not only
    into which way they lie.
    """
    covariance = _covariance(tiles)
    xx, yy, xy = covariance[:, 0, 0], covariance[:, 1, 1], covariance[:, 0, 1]
    half_trace = 0.5 * (xx + yy)
    gap = np.sqrt(np.maximum(0.25 * (xx - yy) ** 2 + xy**2, 0.0))
    return np.sqrt((half_trace + gap) / np.maximum(half_trace - gap, 1e-12))


def tile_adjacency_fn(tiles: Float[np.ndarray, "n 4 2"]) -> list:
    """Neighbour lists: tiles are adjacent when they share a cut vertex."""
    corners = shared_corners_fn(tiles)
    neighbours = [set() for _ in range(len(tiles))]
    for owners in corners["owners"]:
        for tile in owners:
            neighbours[tile].update(o for o in owners if o != tile)
    return [sorted(group) for group in neighbours]


def smooth_field_fn(values, neighbours, *, passes: int, weight: float = 0.5, doubled=False):
    """Laplacian smoothing over the tile graph.

    ``doubled`` smooths a direction field through ``exp(2i*theta)``, which is what makes an axis
    field flow instead of tearing where the raw angle wraps.
    """
    if doubled:
        state = np.exp(2j * np.asarray(values, float))
    else:
        state = np.asarray(values, float).astype(complex)
    for _ in range(passes):
        averaged = np.array([state[group].mean() if group else state[i]
                             for i, group in enumerate(neighbours)])
        state = (1.0 - weight) * state + weight * averaged
    if doubled:
        return 0.5 * np.angle(state)
    return state.real


def orient_field_fn(angles, neighbours, forward):
    """Turn an axis field into a vector field by PROPAGATING the sign across the sheet.

    A tile's principal axis is a line, so its direction has to be chosen. Choosing it per tile, by
    which way it leans relative to the flow, breaks down wherever a tile sits near-perpendicular to
    that flow: the sign then turns on noise and two neighbours end up pointing at each other, which
    is exactly the artifact this removes.

    Instead the sign is grown outward from a seed by breadth-first search, each tile agreeing with
    the neighbours already settled. A line field can carry topological defects that no orientation
    can remove, so this does not promise zero disagreements -- it concentrates the unavoidable ones
    into a few seams instead of scattering them everywhere, and reports how many are left.

    Returns:
        ``(oriented_angles, n_opposed_neighbour_pairs)``.
    """
    angles = np.asarray(angles, float).copy()
    n = len(angles)
    settled = np.zeros(n, dtype=bool)

    def vector(i):
        return np.array([np.cos(angles[i]), np.sin(angles[i])])

    order = np.argsort([-len(g) for g in neighbours])
    for seed in order:
        if settled[seed]:
            continue
        if float(vector(seed) @ forward) < 0.0:
            angles[seed] += np.pi
        settled[seed] = True
        queue = [seed]
        while queue:
            current = queue.pop(0)
            for other in neighbours[current]:
                if settled[other]:
                    continue
                agreement = sum(float(vector(other) @ np.array(
                    [np.cos(angles[k]), np.sin(angles[k])]))
                    for k in neighbours[other] if settled[k])
                if agreement < 0.0:
                    angles[other] += np.pi
                settled[other] = True
                queue.append(other)

    opposed = sum(1 for i, group in enumerate(neighbours) for j in group
                  if j > i and float(vector(i) @ vector(j)) < 0.0)
    return angles, opposed


def _poisson_disk(bounds, radius_at, *, seed, k=24, cell_scale=1.0):
    """Dart-throwing with a spatial hash and a spatially varying radius.

    Bridson's algorithm assumes one radius; the field here does not have one, so this uses plain
    rejection with a grid sized to the smallest radius. Slower, but correct for a varying radius and
    fast enough at the sheet sizes involved.
    """
    rng = np.random.default_rng(seed)
    min_x, min_y, max_x, max_y = bounds
    probe = np.stack(np.meshgrid(np.linspace(min_x, max_x, 24),
                                 np.linspace(min_y, max_y, 24)), axis=-1).reshape(-1, 2)
    smallest = float(np.min(radius_at(probe)))
    cell = max(smallest * cell_scale, 1e-9)
    grid = {}
    points = []

    target = int(4.0 * (max_x - min_x) * (max_y - min_y) / (np.pi * smallest**2))
    for _ in range(target * k):
        candidate = np.array([rng.uniform(min_x, max_x), rng.uniform(min_y, max_y)])
        radius = float(radius_at(candidate[None])[0])
        key = (int(candidate[0] / cell), int(candidate[1] / cell))
        reach = int(np.ceil(radius / cell)) + 1
        clash = False
        for dx in range(-reach, reach + 1):
            for dy in range(-reach, reach + 1):
                for other, other_radius in grid.get((key[0] + dx, key[1] + dy), ()):
                    if np.linalg.norm(other - candidate) < max(radius, other_radius):
                        clash = True
                        break
                if clash:
                    break
            if clash:
                break
        if not clash:
            grid.setdefault(key, []).append((candidate, radius))
            points.append(candidate)
    return np.asarray(points)


def build_scale_field_fn(
    tiles: Float[np.ndarray, "n 4 2"],
    *,
    smoothing: int = 8,
    size_smoothing: int | None = None,
    size_gamma: float = 1.0,
    density: float = 0.24,
    size_frac: float = 2.60,
    leaf_width: float = 0.42,
    leaf_bow: float = 0.10,
    angle_offset: float = 0.0,
    size_jitter: float = 0.10,
    angle_jitter: float = 0.07,
    depth_axis: float = -90.0,
    bleed: float = 0.7,
    hide_tips: bool = False,
    ground_scale: float = 0.0,
    ground_density: float = 0.34,
    aniso_power: float = 0.0,
    placement: str = "poisson",
    grid: tuple[int, int] | None = None,
    bond: float = 1.0,
    regularity: float = 1.0,
    edge_looseness: float = 0.0,
    seed: int = 0,
) -> tuple[list, dict]:
    """Scatter crescents over the tiling's direction and size fields, ordered back to front.

    Args:
        tiles: (n, 4, 2) convex CCW quads.
        smoothing: Laplacian passes on the DIRECTION field; more passes means calmer swirls.
        size_smoothing: passes on the SIZE field, defaulting to a third of ``smoothing``. Kept
            separate because the two want opposite things: the flow reads best when it is smooth,
            while the size range survives only when it is not.
        size_gamma: exponent on the size field about its median. The tiles span far more than the
            elements do, because averaging pulls both tails toward the middle; raising this puts the
            range back. 1 leaves the field as measured, 2 roughly squares the spread.
        density: packing radius as a fraction of local element length. Below 0.5 the elements
            overlap, which is what makes them read as scales rather than as scattered leaves.
        size_frac: element length as a multiple of the local tile scale.
        leaf_width, leaf_bow: element shape, constant across the whole sheet by design.
        angle_offset: rotate every element off the local axis [radians].
        size_jitter, angle_jitter: small per-element variation.
        depth_axis: direction, in degrees, along which elements are stacked back to front. It also
            resolves the element's heading: a tile's principal axis is a LINE, so it cannot say which
            way a leaf points, and pointing every tip down-field is what makes the shingling read.
        hide_tips: reverse the paint order so each blunt base covers its neighbour point. Tips
            read as texture when they sit on top and as nothing when they do not, so this is the
            dial between a spiky field and a smooth one.
        bleed: how far past the sheet edge to seed elements, in element lengths. Without it the
            border is a ragged band of bare ground, because a Poisson process cannot place a centre
            outside its own domain.
        ground_scale: size of a second, finer population as a fraction of the main element, or 0
            for none. A stencil sets a fine ground under its main figure, and the two scales are
            what give the sheet depth; both follow the same flow, so it stays one pattern.
        ground_density: packing radius for that finer population.
        aniso_power: how strongly the element's WIDTH follows the local tile's elongation. 0 holds
            the shape constant across the sheet, as today; 1 makes an elongated tile carry a
            slender leaf, so the deformation shows in the element as well as in its heading.
        placement: ``'poisson'`` scatters elements by dart-throwing, which leaves the connectivity
            of the fused sheet to chance. ``'lattice'`` puts one element per tile, so which elements
            fuse is decided rather than discovered. The ground population always stays Poisson: a
            fine texture wants no order of its own.
        grid: ``(M, N)`` panel grid; required by the lattice placement.
        bond: row offset for the lattice, 0 for a square grid and 1 for a half-cell shingle.
        regularity: 1 places every element exactly on its site; lower values let position, size and
            heading drift by that fraction of the local spacing.
        edge_looseness: how much regularity to give up at the sheet boundary, so the body of the
            pattern stays calm while its edges break up.
        seed: jitter and packing seed.

    Returns:
        ``(elements, report)`` with elements already in paint order, ground population first.
    """
    from scipy.spatial import cKDTree

    neighbours = tile_adjacency_fn(tiles)
    angles = smooth_field_fn(tile_axes_fn(tiles), neighbours, passes=smoothing, doubled=True)
    passes = smoothing // 3 if size_smoothing is None else size_smoothing
    scales = smooth_field_fn(np.sqrt(tile_areas_fn(tiles)), neighbours, passes=passes)
    if size_gamma != 1.0:
        middle = float(np.median(scales))
        scales = middle * (np.maximum(scales, 1e-12) / middle) ** size_gamma
    centroids = tiles.mean(axis=1)
    tree = cKDTree(centroids)

    def nearest(points):
        return tree.query(np.atleast_2d(points))[1]

    axis = np.deg2rad(depth_axis)
    forward = np.array([np.cos(axis), np.sin(axis)])
    angles, flipped = orient_field_fn(angles, neighbours, forward)

    widths = np.full(len(tiles), float(leaf_width))
    if aniso_power != 0.0:
        ratio = smooth_field_fn(tile_anisotropy_fn(tiles), neighbours, passes=passes)
        widths = leaf_width * (float(np.median(ratio)) / np.maximum(ratio, 1e-12)) ** aniso_power
        widths = np.clip(widths, 0.20, 0.92)

    flat = tiles.reshape(-1, 2)
    sheet = (flat[:, 0].min(), flat[:, 1].min(), flat[:, 0].max(), flat[:, 1].max())

    def place(fraction, packing, population, stream, on_lattice=False):
        """Scatter one population at ``fraction`` of the main element size."""
        rng = np.random.default_rng(seed + stream + 1)
        if on_lattice:
            points, index, local = _lattice_sites(
                tiles, grid, bond=bond, regularity=regularity,
                edge_looseness=edge_looseness, rng=rng)
            # Size and heading follow the same dial as position, or a lattice of identical sites
            # would still read as scattered.
            slack = 1.0 - local
        else:
            pad = bleed * fraction * size_frac * float(np.median(scales))
            grown = (sheet[0] - pad, sheet[1] - pad, sheet[2] + pad, sheet[3] + pad)
            points = _poisson_disk(
                grown, lambda p: packing * fraction * size_frac * scales[nearest(p)],
                seed=seed + stream)
            index = nearest(points)
            slack = np.ones(len(points))
        lengths = (fraction * size_frac * scales[index]
                   * (1.0 + slack * size_jitter * rng.normal(size=len(points))))
        headings = (angles[index] + angle_offset
                    + slack * angle_jitter * rng.normal(size=len(points)))
        # Order by where each TIP lands, not by the element's centre. The tip is the part that must
        # stay visible, so a leaf whose tip reaches furthest down-field is laid down first; every
        # leaf painted after it then puts its own tip on that leaf's BASE, not across its point.
        facing = np.stack([np.cos(headings), np.sin(headings)], axis=1)
        depth = (points + 0.5 * lengths[:, None] * facing) @ forward
        order = np.argsort(depth if hide_tips else -depth)
        placed = [Element(Leaf(centre=points[i], angle=float(headings[i]),
                               length=float(lengths[i]), width=float(widths[index[i]]),
                               bow=leaf_bow), population, int(index[i]))
                  for i in order]
        return placed, lengths

    lattice = placement == "lattice"
    if lattice and grid is None:
        raise ValueError("placement='lattice' needs the panel grid; pass grid=(M, N)")
    elements, lengths = place(1.0, density, "main", 0, on_lattice=lattice)
    ground = []
    if ground_scale > 0.0:
        # The whole fine layer goes down before the first figure, so the figures read as sitting ON
        # a ground rather than interleaved with it. Its own stream keeps the main packing identical
        # to a sheet with no ground at all, which is what makes the two comparable.
        ground, _ = place(ground_scale, ground_density, "ground", 977)
        elements = ground + elements

    spread = np.abs(np.diff(np.sort(angles)))
    report = {
        "n_elements": len(elements),
        "n_ground": len(ground),
        "length_min": float(lengths.min()),
        "length_median": float(np.median(lengths)),
        "length_max": float(lengths.max()),
        "length_spread": float(lengths.max() / max(lengths.min(), 1e-12)),
        "neighbour_angle_p95_deg": float(np.degrees(np.percentile(spread, 95))),
        "elements_per_tile": len(elements) / max(len(tiles), 1),
        "sheet_bounds": sheet,
        "opposed_neighbours": flipped,
        "width_min": float(widths.min()),
        "width_max": float(widths.max()),
    }
    return elements, report


def render_scale_preview_fn(elements, out_png, *, specs=None, web_mm=0.5, dpi=300, crop=None,
                            fill="#F2EFE6", ground="#141414", edge="#141414", n=64,
                            figure_width_in=11.0):
    """Raster preview in paint order: opaque pale fills over a dark ground.

    Occlusion is the construction, not a finishing touch -- painter's order alone produces the
    scale reading, and the dark outlines become the thin webs between shapes.

    Args:
        elements: ``Element`` records in paint order.
        specs: how to ink each population, as ``{population: ElementSpec}``. The decomposition
            itself comes from ``element_layers_fn``, so this preview and the vector output cannot
            disagree about what an element is.
    """
    from matplotlib.patches import Polygon as MplPolygon

    from nff.utils import figstyle

    figstyle.use_agg()
    import matplotlib.pyplot as plt

    figstyle.apply_charter(**{"axes.grid": False})

    specs = specs or {}
    default = ElementSpec()
    painted = [(element.population, element_layers_fn(element.leaf,
                                                      specs.get(element.population, default)))
               for element in elements]

    if crop is None:
        stacked = np.concatenate([leaf_polygon_fn(layers[0][0], n=n) for _, layers in painted])
        crop = (stacked[:, 0].min(), stacked[:, 1].min(), stacked[:, 0].max(), stacked[:, 1].max())
    width = crop[2] - crop[0]
    height = crop[3] - crop[1]

    ink = {"pale": fill, "dark": edge}
    figure, axis = plt.subplots(
        figsize=(figure_width_in, figure_width_in * height / max(width, 1e-9)))
    figure.patch.set_facecolor(ground)
    axis.set_facecolor(ground)
    for _, layers in painted:
        for depth, (leaf, tag) in enumerate(layers):
            # Only the silhouette can carry a web, and at zero width it carries none. Dropping it
            # is not a cosmetic choice: the outline is the only thing drawing a boundary between two
            # elements that, in paper, are one piece, so without it the picture shows the object.
            webbed = depth == 0 and web_mm > 0.0
            axis.add_patch(MplPolygon(
                leaf_polygon_fn(leaf, n=n), closed=True, facecolor=ink[tag],
                edgecolor=edge if webbed else "none",
                linewidth=web_mm * 2.0 if webbed else 0.0, joinstyle="round"))
    axis.set_xlim(crop[0], crop[2])
    axis.set_ylim(crop[1], crop[3])
    axis.set_aspect("equal")
    axis.axis("off")
    figure.savefig(out_png, dpi=dpi, bbox_inches="tight", pad_inches=0.0, facecolor=ground)
    plt.close(figure)
    return out_png
