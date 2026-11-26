"""Ise-katagami artwork from a flat closed-pipeline sheet.

The closed pipeline leaves behind a flat sheet partitioned into irregular convex quadrilaterals: the
topology is a perfectly regular grid, the metric is not, because ``z`` and ``bnd_logits`` were
optimised against a mechanical target. A traditional motif laid onto that tiling comes out different
in every cell, which is the irregular repeat this module exists to draw.

Two rules keep the tiling from becoming legible in the artwork:

- motif strokes reach the cell boundary only at corners, which are shared exactly between two tiles
  (``nff_paper.topology_art.motif``), so strokes join their neighbour to machine precision;
- the ornament sitting on each shared corner is drawn ONCE, in sheet coordinates, so there is no
  join to get wrong and the tangent mismatch between two tiles' strokes is absorbed by a designed
  feature rather than showing as a kink.

Only the flat, undeployed state is drawn. Nothing here is meant to be cut.
"""

import os
import pickle

import numpy as np
import yaml
from jaxtyping import Float

from nff.topology.closed_builder_jax import (
    boundary_flat_from_logits,
    build_boundary_edges,
    build_deploy_structure,
    solve_cut_vertices_jax,
)
from nff_paper.topology_art.motif import build_motif_fn, check_corner_ports_fn
from nff_paper.topology_art.quad_warp import (
    BezierPath,
    assert_ccw_convex_fn,
    path_points_fn,
    transform_path_fn,
    warp_polyline_fn,
)

NODE_CELL = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])


# ── sheet sources ────────────────────────────────────────────────────────────────────────────


def _tiles_from_params(z, bnd_logits, M, N, spacing, spacing_y):
    """Flat tile polygons from design parameters -- the linear cut-vertex solve, no physics."""
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    struct = build_deploy_structure(M, N)
    sliders = build_boundary_edges(struct, spacing=spacing, spacing_y=spacing_y)
    boundary_flat = boundary_flat_from_logits(sliders, jnp.asarray(bnd_logits))
    coords = np.asarray(solve_cut_vertices_jax(struct, boundary_flat, jax.nn.sigmoid(jnp.asarray(z))))
    return coords[struct["corner_pid"]]


R_FIELDS = ("smooth", "gradient", "radial", "banded", "ridged", "swirl", "patchy",
            "columns", "rows", "uniform", "shear", "saddle", "pinch", "vortex2", "lens",
            "swell", "terrace")


def build_r_field_fn(shape, *, kind="smooth", seed=0, amplitude=2.0, n_modes=3):
    """Latent field over the cut grid; ``r = sigmoid(z)`` then sets every void's aspect ratio.

    This is the deepest lever on the artwork. The element field reads each tile's principal axis, so
    changing how tile shapes vary across the sheet changes the FLOW, not merely the arrangement --
    a banded sheet gives banded flow, a swirled sheet gives eddies.

    Args:
        shape: ``(M+1, N+1)`` cut grid.
        kind: one of ``R_FIELDS``.
        seed: randomness for the stochastic kinds.
        amplitude: logit amplitude; larger means more extreme tile-size spread.
        n_modes: number of random modes for the noise-based kinds.

    Returns:
        ``(M+1, N+1)`` array of logits.
    """
    rng = np.random.default_rng(seed)
    rows, cols = shape
    yy, xx = np.meshgrid(np.linspace(0, 1, rows), np.linspace(0, 1, cols), indexing="ij")
    centred_x, centred_y = xx - 0.5, yy - 0.5
    radius = np.hypot(centred_x, centred_y) / 0.7071
    angle = np.arctan2(centred_y, centred_x)

    def noise(modes):
        out = np.zeros(shape)
        for _ in range(modes):
            kx, ky = rng.uniform(0.5, 2.5, size=2)
            phase = rng.uniform(0, 2 * np.pi, size=2)
            out += rng.normal() * np.sin(2 * np.pi * kx * xx + phase[0]) * np.sin(
                2 * np.pi * ky * yy + phase[1])
        return out

    if kind == "uniform":
        field = np.zeros(shape)
    elif kind == "smooth":
        field = noise(n_modes)
    elif kind == "gradient":
        field = 2.0 * xx - 1.0
    elif kind == "radial":
        field = np.cos(2.0 * np.pi * 1.6 * radius)
    elif kind == "columns":
        field = np.sin(2.0 * np.pi * 2.5 * yy)
    elif kind == "rows":
        field = np.sin(2.0 * np.pi * 2.5 * xx)
    elif kind == "banded":
        theta = rng.uniform(0, np.pi)
        field = np.sin(2.0 * np.pi * 2.5 * (xx * np.cos(theta) + yy * np.sin(theta)))
    elif kind == "ridged":
        rough = noise(n_modes)
        field = 2.0 * np.abs(rough / max(np.abs(rough).max(), 1e-12)) - 1.0
    elif kind == "swirl":
        field = np.sin(3.0 * angle + 7.0 * radius)
    elif kind == "swell":
        # One long undulation across the whole sheet: a gentle gradation of SIZE.
        # ⚠ Having no angular term does NOT make a field's flow calm. The heading comes from each
        # tile's principal axis, so a defect appears wherever a tile lands near-isotropic whatever
        # the field looks like: measured at square pitch, swell and terrace give 9 and 11 opposed
        # neighbour pairs against 7 for a two-vortex field. A still flow comes from the tile PITCH
        # (`aspect`) or from `banded`, not from the absence of a swirl in the r-field.
        field = np.sin(2.0 * np.pi * (0.6 * yy + 0.18 * xx) - 0.6)
    elif kind == "terrace":
        # Broad, smoothly separated steps: the size changes in bands. See the caveat above.
        field = np.tanh(3.0 * np.sin(2.0 * np.pi * 1.2 * yy))
    elif kind == "shear":
        # Bands whose spacing opens across the sheet, so the flow fans rather than combing.
        field = np.sin(2.0 * np.pi * (1.4 * xx + 2.6 * xx * yy))
    elif kind == "saddle":
        # A hyperbolic field: the flow runs together along one diagonal and apart along the other.
        field = centred_x**2 - centred_y**2
    elif kind == "pinch":
        # One tight knot at the centre that decays to calm ground, so the eye has a single subject.
        field = np.cos(2.0 * np.pi * 1.4 * radius) * np.exp(-4.0 * radius**2)
    elif kind == "vortex2":
        field = np.zeros(shape)
        for centre_x, centre_y, turn in ((0.30, 0.62, +1.0), (0.72, 0.36, -1.0)):
            offset_x, offset_y = xx - centre_x, yy - centre_y
            field += turn * np.sin(2.0 * np.arctan2(offset_y, offset_x)
                                   + 9.0 * np.hypot(offset_x, offset_y))
    elif kind == "lens":
        # A single elongated body of large tiles lying across the middle of the sheet.
        field = 2.0 * np.exp(-((centred_x / 0.46) ** 2 + (centred_y / 0.16) ** 2)) - 1.0
    elif kind == "patchy":
        # Sparse blobs: groups of like-sized tiles separated by calmer ground.
        field = np.zeros(shape)
        for _ in range(5):
            cx, cy = rng.uniform(0.15, 0.85, size=2)
            width = rng.uniform(0.10, 0.22)
            field += rng.choice([-1.0, 1.0]) * np.exp(
                -(((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * width**2)))
    else:
        raise ValueError(f"unknown r-field {kind!r}; available: {R_FIELDS}")

    peak = max(float(np.abs(field).max()), 1e-12)
    return amplitude * field / peak


def _smooth_field(shape, seed, amplitude, n_modes=3):
    """Backwards-compatible alias for the random-mode field."""
    return build_r_field_fn(shape, kind="smooth", seed=seed, amplitude=amplitude, n_modes=n_modes)


def load_flat_tiles_fn(
    *,
    run_dir: str | None = None,
    design_npz: str | None = None,
    synthetic: tuple[int, int] | None = None,
    seed: int = 0,
    amplitude: float = 2.0,
    field_kind: str = "smooth",
    aspect: float = 1.0,
) -> tuple[Float[np.ndarray, "n_tiles 4 2"], dict]:
    """Flat tile polygons plus provenance, from one of three sources.

    Args:
        run_dir: a trained run folder holding ``config.yaml`` and ``best_params.pkl``.
        design_npz: a cached shield design carrying ``flat_tiles``.
        synthetic: ``(M, N)`` for a sheet built from a smooth random ``z`` field, needing no run.
        seed: seed for the synthetic field.
        amplitude: logit amplitude of the synthetic field; larger means more irregular tiles.
        field_kind: which r-field to build; see ``R_FIELDS``.
        aspect: tile pitch in y over pitch in x. Above 1 the tiles are tall, so their principal
            axes -- and therefore the whole element field -- stand upright. This is the honest way
            to get a vertical pattern: the sheet is vertical, not just the drawing on it.

    Returns:
        ``(tiles, meta)`` with ``tiles`` of shape ``(n_tiles, 4, 2)`` in sheet units.
    """
    given = [x for x in (run_dir, design_npz, synthetic) if x is not None]
    if len(given) != 1:
        raise ValueError("give exactly one of run_dir, design_npz, synthetic")

    if design_npz is not None:
        data = np.load(design_npz)
        if "flat_tiles" not in data:
            raise ValueError(f"{design_npz} has no 'flat_tiles' array")
        tiles = np.asarray(data["flat_tiles"], float)
        meta = {"source": design_npz, "kind": "design_npz"}
    elif run_dir is not None:
        with open(os.path.join(run_dir, "config.yaml"), encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        topo = config["tessellation"]
        M, N = int(topo["M"]), int(topo["N"])
        spacing = float(topo.get("spacing", 1.0))
        spacing_y = float(topo.get("spacing_y", spacing))
        with open(os.path.join(run_dir, "best_params.pkl"), "rb") as handle:
            params = pickle.load(handle)
        tiles = np.asarray(_tiles_from_params(
            params["z"], params["bnd_logits"], M, N, spacing, spacing_y), float)
        meta = {"source": run_dir, "kind": "run_dir", "M": M, "N": N}
    else:
        M, N = (int(v) for v in synthetic)
        z = build_r_field_fn((M + 1, N + 1), kind=field_kind, seed=seed, amplitude=amplitude)
        tiles = np.asarray(_tiles_from_params(
            z, np.zeros(2 * M + 2 * N), M, N, 1.0, aspect), float)
        meta = {"source": f"synthetic {M}x{N} {field_kind} seed {seed}", "kind": "synthetic",
                "M": M, "N": N}

    assert_ccw_convex_fn(tiles, name=meta["kind"])
    meta["n_tiles"] = int(len(tiles))
    meta["areas"] = tile_areas_fn(tiles)
    return tiles, meta


# ── sheet structure ──────────────────────────────────────────────────────────────────────────


def tile_areas_fn(tiles: Float[np.ndarray, "n 4 2"]) -> Float[np.ndarray, "n"]:
    """Shoelace area of every tile."""
    x, y = tiles[:, :, 0], tiles[:, :, 1]
    return 0.5 * np.abs(np.sum(x * np.roll(y, -1, axis=1) - y * np.roll(x, -1, axis=1), axis=1))


def shared_corners_fn(
    tiles: Float[np.ndarray, "n 4 2"], *, rel_tol: float = 1e-9
) -> dict:
    """Group tile corners into the sheet's shared cut-vertices.

    On a closed-builder sheet each interior corner belongs to exactly two tiles, which is the only
    join available: tile edges are split differently on each side and almost never coincide.

    Returns:
        Dict with ``positions`` (K, 2), ``valence`` (K,), ``owners`` (list of tile-index lists) and
        ``scale`` (the sheet's bounding-box diagonal, used to set the merge tolerance).
    """
    flat = tiles.reshape(-1, 2)
    span = flat.max(axis=0) - flat.min(axis=0)
    scale = float(np.hypot(*span))
    tol = max(rel_tol * scale, 1e-12)
    keys = np.round(flat / tol).astype(np.int64)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.ravel()

    n_groups = counts.size
    positions = np.zeros((n_groups, 2))
    np.add.at(positions, inverse, flat)
    positions /= counts[:, None]

    owners = [[] for _ in range(n_groups)]
    for slot, group in enumerate(inverse):
        owners[group].append(slot // 4)
    return {"positions": positions, "valence": counts, "owners": owners,
            "inverse": inverse, "scale": scale}


def assign_detail_fn(
    areas: Float[np.ndarray, "n"], *, levels: tuple[int, ...] = (1, 2, 3)
) -> Float[np.ndarray, "n"]:
    """Level of detail per tile, by area rank.

    Corner ports are identical at every level, so detail may vary tile to tile without breaking the
    joins. Small tiles get the sparse figure and large tiles the dense one, which keeps ink density
    roughly even across a sheet whose tiles span more than an order of magnitude in area.
    """
    order = np.argsort(np.argsort(areas))
    bucket = (order * len(levels) // max(len(areas), 1)).clip(0, len(levels) - 1)
    return np.asarray(levels, int)[bucket]


# ── composition ──────────────────────────────────────────────────────────────────────────────


def compose_katagami_fn(
    tiles: Float[np.ndarray, "n 4 2"],
    *,
    motif_name: str = "shippo",
    node_name: str | None = None,
    node_scale: float = 0.17,
    lod: bool = True,
    motif_params: dict | None = None,
    node_params: dict | None = None,
) -> dict[str, list[BezierPath]]:
    """Warp a motif into every tile and drop one ornament on every shared corner.

    Args:
        tiles: (n, 4, 2) convex CCW quads, BL BR TR TL.
        motif_name: registry key for the cell figure.
        node_name: registry key for the corner ornament, or ``None`` to omit it.
        node_scale: ornament radius as a fraction of the local tile size.
        lod: vary motif detail with tile area.
        motif_params, node_params: motif-specific overrides.

    Returns:
        Layer mapping ``{"motif": [...], "nodes": [...], "tiles": [...]}``.
    """
    assert_ccw_convex_fn(tiles, name="tiles")
    areas = tile_areas_fn(tiles)
    details = assign_detail_fn(areas) if lod else np.full(len(tiles), 2)

    cache = {}
    for detail in np.unique(details):
        motif = build_motif_fn(motif_name, detail=int(detail), **(motif_params or {}))
        report = check_corner_ports_fn(motif)
        if not report["ok"]:
            raise ValueError(
                f"motif {motif_name!r} (detail {detail}) breaks the corner-port rule: "
                f"{len(report['mid_edge'])} mid-edge points, {len(report['outside'])} outside"
            )
        cache[int(detail)] = motif

    motif_paths = []
    for quad, detail in zip(tiles, details):
        for stroke in cache[int(detail)].strokes:
            motif_paths.append(warp_polyline_fn(quad, stroke.points, closed=stroke.closed))

    node_paths = []
    if node_name:
        node = build_motif_fn(node_name, **(node_params or {}))
        corners = shared_corners_fn(tiles)
        radius = np.sqrt(areas)
        # A node sized in proportion to its tiles would trace the tile-size field and hand the
        # tiling straight back to the viewer. Keep it near-constant, and only shrink it where a
        # constant node would swamp a small tile.
        uniform = node_scale * float(np.median(radius))
        for position, owners in zip(corners["positions"], corners["owners"]):
            local = min(uniform, 0.45 * float(np.min(radius[owners])))
            heading = np.mean(tiles[owners].reshape(-1, 2), axis=0) - position
            norm = float(np.linalg.norm(heading))
            angle = np.arctan2(heading[1], heading[0]) if norm > 1e-12 else 0.0
            rotation = np.array([[np.cos(angle), -np.sin(angle)],
                                 [np.sin(angle), np.cos(angle)]])
            quad = position + local * (NODE_CELL @ rotation.T)
            for stroke in node.strokes:
                node_paths.append(warp_polyline_fn(quad, stroke.points, closed=stroke.closed))

    tile_paths = [warp_polyline_fn(quad, np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
                                   closed=True) for quad in tiles]
    return {"motif": motif_paths, "nodes": node_paths, "tiles": tile_paths}


def transform_layers_fn(layers, matrix, offset):
    """Apply one affine map to every path in every layer."""
    return {name: [transform_path_fn(p, matrix, offset) for p in paths]
            for name, paths in layers.items()}


def layer_bounds_fn(layers, names=("motif", "nodes")) -> tuple[float, float, float, float]:
    """Bounding box of the named layers, sampled densely enough to include curve bulges."""
    pts = [path_points_fn(path, per_segment=8)
           for name in names for path in layers.get(name, [])]
    if not pts:
        raise ValueError("no geometry to bound")
    stacked = np.concatenate(pts, axis=0)
    return (float(stacked[:, 0].min()), float(stacked[:, 1].min()),
            float(stacked[:, 0].max()), float(stacked[:, 1].max()))


def corner_join_residual_fn(tiles, *, rel_tol: float = 1e-9) -> float:
    """Largest spread among the tile corners that are meant to be one sheet vertex.

    The seam test in one number: motif strokes terminate at corners, so this bounds how far apart
    two neighbouring strokes can land.
    """
    corners = shared_corners_fn(tiles, rel_tol=rel_tol)
    flat = tiles.reshape(-1, 2)
    offsets = flat - corners["positions"][corners["inverse"]]
    return float(np.linalg.norm(offsets, axis=1).max())


# ── chains: the structure that already exists in the tiling ──────────────────────────────────


def diagonal_chains_fn(tiles: Float[np.ndarray, "n 4 2"], *, rel_tol: float = 1e-9) -> dict:
    """Follow each tile's two diagonals through the corners they share into long chains.

    Nothing here is invented. Pair every tile's corners 0-2 and 1-3, and because each interior
    corner belongs to exactly two tiles, every diagonal end meets exactly one partner: the graph on
    corners is 2-regular and therefore falls apart into disjoint chains and cycles. On the 15x10
    sheet that yields a handful of chains, the longest of which crosses every tile.

    This is what a stroke should follow. A line that runs for twenty cells is read as a line; the
    grid it happens to cross is not read at all.

    Returns:
        Dict with ``chains`` (each a dict of ``nodes``, ``edges``, ``closed``), ``corners`` (the
        ``shared_corners_fn`` result) and ``groups`` (n_tiles, 4) corner-group index per tile corner.
    """
    corners = shared_corners_fn(tiles, rel_tol=rel_tol)
    groups = corners["inverse"].reshape(-1, 4)

    edges = []
    for tile in range(len(tiles)):
        edges.append((int(groups[tile, 0]), int(groups[tile, 2]), tile, (0, 2)))
        edges.append((int(groups[tile, 1]), int(groups[tile, 3]), tile, (1, 3)))

    incident = {}
    for index, (node_a, node_b, _, _) in enumerate(edges):
        incident.setdefault(node_a, []).append(index)
        incident.setdefault(node_b, []).append(index)

    used = [False] * len(edges)

    def walk(node, edge_index):
        nodes, walked = [node], []
        while edge_index is not None and not used[edge_index]:
            used[edge_index] = True
            node_a, node_b, _, _ = edges[edge_index]
            node = node_b if node_a == node else node_a
            walked.append(edge_index)
            nodes.append(node)
            spare = [e for e in incident[node] if not used[e]]
            edge_index = spare[0] if spare else None
        return {"nodes": nodes, "edges": walked, "closed": nodes[0] == nodes[-1] and len(nodes) > 2}

    chains = []
    for node in sorted(incident):  # open chains first, so cycles are what is left over
        if len(incident[node]) == 1:
            spare = [e for e in incident[node] if not used[e]]
            if spare:
                chains.append(walk(node, spare[0]))
    for index in range(len(edges)):
        if not used[index]:
            chains.append(walk(edges[index][0], index))

    return {"chains": chains, "corners": corners, "groups": groups, "edges": edges}


def _chain_frames(positions, closed):
    """Per-node world tangent and arclength fraction along one chain.

    The tangent is the Catmull-Rom secant through the neighbouring corners, so it belongs to the
    CHAIN rather than to either tile. Both tiles meeting at a corner therefore read the same
    direction, which is the whole reason the junction can be fluid at nonzero width.
    """
    n = len(positions)
    tangents = np.empty_like(positions)
    if closed:
        tangents = np.roll(positions, -1, axis=0) - np.roll(positions, 1, axis=0)
    else:
        tangents[1:-1] = positions[2:] - positions[:-2]
        tangents[0] = positions[1] - positions[0]
        tangents[-1] = positions[-1] - positions[-2]
    tangents /= np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-15)

    # One smoothing pass along the chain. Corner positions zigzag, and at a hairpin the raw secant
    # swings hard, which the stroke shows as a kink. Averaging with the neighbours is safe here
    # because every tangent on a chain already points the same way along the walk.
    if len(tangents) > 2:
        neighbours = np.empty_like(tangents)
        if closed:
            neighbours = np.roll(tangents, 1, axis=0) + np.roll(tangents, -1, axis=0)
        else:
            neighbours[1:-1] = tangents[:-2] + tangents[2:]
            neighbours[0] = 2.0 * tangents[0]
            neighbours[-1] = 2.0 * tangents[-1]
        tangents = tangents + 0.5 * neighbours
        tangents /= np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-15)

    steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(steps)])
    return tangents, arc / max(arc[-1], 1e-15)


def build_vine_fn(
    tiles: Float[np.ndarray, "n 4 2"],
    *,
    width_frac: float = 0.10,
    swell: float = 0.45,
    handle: float = 0.42,
    samples: int = 18,
    taper: float = 0.7,
    floor: float = 0.38,
    rel_tol: float = 1e-9,
) -> tuple[list, dict]:
    """One tapered brush stroke per chain, drawn as filled ribbons.

    Each chain becomes a single stroke that swells toward its middle and tapers to nothing only at
    its two free ends. Everywhere it crosses a shared corner the width is nonzero and identical on
    both sides, because both the direction and the width there belong to the chain rather than to
    either tile.

    Args:
        tiles: (n, 4, 2) convex CCW quads.
        width_frac: base half-width as a fraction of the median tile radius.
        swell: extra mid-chord swell, scaled per tile by how round the tile is.
        handle: cubic handle length as a fraction of the port separation.
        samples: centreline samples per chord.
        taper: exponent of the chain-length taper; lower is blunter.
        rel_tol: corner-merge tolerance.

    Returns:
        ``(paths, report)`` with filled closed ribbon outlines and chain statistics.
    """
    from nff_paper.topology_art.motif import CORNERS
    from nff_paper.topology_art.ribbon import build_chord_fn, build_ribbon_fn, width_profile_fn

    structure = diagonal_chains_fn(tiles, rel_tol=rel_tol)
    positions_all = structure["corners"]["positions"]
    edges = structure["edges"]

    areas = tile_areas_fn(tiles)
    base = width_frac * float(np.median(np.sqrt(areas)))
    perimeter = np.linalg.norm(np.roll(tiles, -1, axis=1) - tiles, axis=2).sum(axis=1)
    roundness = np.clip(4.0 * np.pi * areas / np.maximum(perimeter**2, 1e-15) / 0.785, 0.0, 1.0)

    paths, lengths = [], []
    for chain in structure["chains"]:
        nodes = chain["nodes"]
        lengths.append(len(chain["edges"]))
        positions = positions_all[nodes]
        tangents, arc = _chain_frames(positions, chain["closed"])
        # A chain taper that runs all the way to zero leaves long wispy stretches on a chain that
        # crosses a hundred cells, which reads as an accident rather than as a brush lifting.
        profile = (np.ones_like(arc) if chain["closed"]
                   else floor + (1.0 - floor) * np.sin(np.pi * arc) ** taper)
        half_widths = base * profile

        for step, edge_index in enumerate(chain["edges"]):
            node_a, _, tile, (corner_a, corner_b) = edges[edge_index]
            start, end = (corner_a, corner_b) if nodes[step] == node_a else (corner_b, corner_a)
            quad = tiles[tile]
            centre_uv = build_chord_fn(
                quad, CORNERS[start], CORNERS[end], tangents[step], tangents[step + 1],
                handle=handle, n_samples=samples)
            widths = width_profile_fn(samples + 1, half_widths[step], half_widths[step + 1],
                                      swell=swell * float(roundness[tile]))
            outline = build_ribbon_fn(quad, centre_uv, widths,
                                      end_directions=(tangents[step], tangents[step + 1]))
            paths.append(warp_polyline_fn(quad, outline, closed=True))

    lengths = np.asarray(lengths)
    loose = int((structure["corners"]["valence"] == 1).sum())
    report = {
        "n_chains": len(lengths),
        "chain_cells_min": int(lengths.min()),
        "chain_cells_median": float(np.median(lengths)),
        "chain_cells_max": int(lengths.max()),
        "n_closed_chains": int(sum(c["closed"] for c in structure["chains"])),
        "loose_ends": loose,
        "half_width": base,
    }
    return paths, report


# ── preview ──────────────────────────────────────────────────────────────────────────────────


def render_preview_fn(layers, out_png, *, stroke_mm=0.4, show_tiles=False, dpi=260,
                      page_mm=None, background="white", filled=False):
    """Raster preview drawn from the exact control points, not from a flattened copy.

    Matplotlib renders quadratics natively via ``CURVE3``, so what is previewed is what is written.
    """
    from matplotlib.path import Path

    from nff.utils import figstyle

    figstyle.use_agg()
    import matplotlib.pyplot as plt

    figstyle.apply_charter()

    def to_mpl(path):
        verts, codes = [np.asarray(path.start, float)], [Path.MOVETO]
        for kind, points in path.segments:
            if kind == "L":
                verts.append(points[0])
                codes.append(Path.LINETO)
            else:
                verts.extend([points[0], points[1]])
                codes.extend([Path.CURVE3, Path.CURVE3])
        if path.closed:
            verts.append(path.start)
            codes.append(Path.CLOSEPOLY)
        return Path(np.asarray(verts, float), codes)

    bounds = layer_bounds_fn(layers)
    width, height = bounds[2] - bounds[0], bounds[3] - bounds[1]
    figure, axis = plt.subplots(figsize=(9.0, 9.0 * height / max(width, 1e-9)))
    figure.patch.set_facecolor(background)
    axis.set_facecolor(background)

    styles = {
        "tiles": (figstyle.GREY, 0.3 * stroke_mm, 0.35),
        "motif": (figstyle.INK, stroke_mm, 1.0),
        "nodes": (figstyle.INK, stroke_mm, 1.0),
    }
    for name in ("tiles", "motif", "nodes"):
        if name == "tiles" and not show_tiles:
            continue
        colour, width_mm, alpha = styles[name]
        solid = filled and name in ("motif", "nodes")
        for path in layers.get(name, []):
            axis.add_patch(plt.matplotlib.patches.PathPatch(
                to_mpl(path), fill=solid,
                facecolor=colour if solid else "none",
                edgecolor="none" if solid else colour, alpha=alpha,
                linewidth=0.0 if solid else (
                    width_mm * (72.0 / 25.4) if page_mm else width_mm * 2.0),
                joinstyle="round", capstyle="round"))

    axis.set_xlim(bounds[0], bounds[2])
    axis.set_ylim(bounds[1], bounds[3])
    axis.set_aspect("equal")
    axis.axis("off")
    figure.savefig(out_png, dpi=dpi, bbox_inches="tight", facecolor=background)
    plt.close(figure)
    return out_png
