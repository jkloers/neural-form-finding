"""Statistics of a deployed sheet's load path -- what the force field looks like, as numbers.

:mod:`nff.analysis.load_paths` extracts the force field and :mod:`nff.analysis.load_path_viz` draws it.
This module is the third reading: the quantities a paragraph can cite. Every function takes an
extracted :class:`~nff.analysis.load_paths.ForceField` and returns plain numbers, so a claim in the
manuscript and a mark in the figure are traceable to one array.

Two measurement traps are handled here rather than left to the caller, because both silently invert
their answer:

* **The force field has two scales.** A smooth, domain-scale part set by the applied load, and the
  mechanism's own alternating background. Correlating the RAW tile field with the checkerboard mode
  reports the smooth part and hides the alternation. :func:`checkerboard_amplitude` detrends first.
* **A whole-lattice statistic hides a local motif.** A pattern confined to one region of the sheet
  does not survive averaging over the sheet. :func:`lattice_autocorrelation` takes a sub-block.
"""

from typing import Any, Dict, Optional, Sequence

import jax.numpy as jnp
import networkx as nx
import numpy as np

from nff.analysis.load_paths import ForceField, bulk_tiles, edge_modes
from nff.stages.physics.energy import ligament_energy, ligament_energy_linearized
from nff.utils.linalg import face_to_node_kinematics_fn


# ── Per-tile fields ───────────────────────────────────────────────────────────────────────────

def tile_pressure(field: ForceField) -> np.ndarray:
    """(n_faces,) mean normal stress per tile, ``tr(sigma)/2``, positive in TENSION.

    The NET state of a tile, unlike the one-sided compression channel of
    :func:`mode_intensity`. A tile squeezed at one hinge and pulled at three registers here as
    tension and there as compression, so the two disagree by design.
    """
    sigma = np.asarray(field.sigma)
    return 0.5 * (sigma[:, 0, 0] + sigma[:, 1, 1])


def tile_couple(field: ForceField) -> np.ndarray:
    """(n_faces,) skew part of the tile stress, the couple each tile carries."""
    sigma = np.asarray(field.sigma)
    return 0.5 * (sigma[:, 0, 1] - sigma[:, 1, 0])


def mode_intensity(field: ForceField, dec) -> tuple:
    """``(compression, shear)`` carried by each tile's own hinges, as a share of F0."""
    compression = np.zeros(field.n_faces)
    shear = np.zeros(field.n_faces)
    pairs = np.asarray(field.face_pairs)
    for column in (0, 1):
        np.add.at(compression, pairs[:, column], np.maximum(-dec['f_n'], 0.0))
        np.add.at(shear, pairs[:, column], np.abs(dec['f_t']))
    return compression / field.f0, shear / field.f0


def _pure_fraction(edges, sign, nodes) -> float:
    """Tile-weighted fraction of the graph's components carrying a single pressure sign."""
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    pure = sum(len(c) for c in nx.connected_components(graph) if len({sign[n] for n in c}) == 1)
    return pure / len(nodes)


def mode_boundary_shear(field: ForceField, dec, *, n_null: int = 4000,
                        seed: int = 0) -> Dict[str, Any]:
    """Whether shear-dominated interfaces sit on the boundary between tension and compression.

    A mode boundary is an interface whose two tiles carry :func:`tile_pressure` of opposite sign.
    Two readings are returned and they answer different questions. ``enrichment`` is local: how
    much more often a boundary interface is shear-dominated than the sheet average.
    ``cutset_purity`` is topological: whether deleting the shear-dominated interfaces partitions
    the tile graph into pressure-pure regions, which is what the word "separated" asserts.

    ⚠ THE TWO DISAGREE, so quote the one that matches the sentence being written. On the
    five-actuation matrix the enrichment is 1.17-1.42 at z = 2.3-4.8, while the cut-set purity is
    indistinguishable from deleting the same number of interfaces at random. Shear is
    over-represented on the boundary and does not partition the sheet. Both facts are needed:
    the second is why "tension paths are separated by shear interfaces" cannot be written.

    Four controls, each of which changes the answer when dropped:

    * **Bulk tiles only.** :func:`~nff.analysis.load_paths.virial_stress` sums internal interactions,
      so a clamped or loaded tile reports its reaction and not its state. Interfaces touching one
      are excluded.
    * **Stratified by cut family.** :func:`~nff.analysis.load_paths.edge_modes` is a branch-frame
      label, and :func:`~nff.analysis.load_paths.orientation_confound` measures how much of it is
      really hinge orientation. ``enrichment_stratified`` and ``per_family`` are the controlled
      readings, and the pooled ``enrichment`` sits beside them because the two differ. On the
      five-actuation matrix the whole excess lives in one family.
    * **The null permutes the pressure SIGN over tiles**, holding the graph and the interface
      labels fixed. Permuting interface labels instead destroys the spatial correlation of the
      pressure field and inflates every ``z``.
    * **The cut-set null deletes the same COUNT of interfaces**, so the comparison is against
      thinning the graph rather than against the intact one.
    """
    pressure, family = tile_pressure(field), orientation_families(field)
    bulk = bulk_tiles(field)
    pairs = np.asarray(field.face_pairs)
    keep = bulk[pairs[:, 0]] & bulk[pairs[:, 1]]

    left, right = pairs[keep, 0], pairs[keep, 1]
    shear = edge_modes(dec)[keep] == 'shear'
    sign = np.sign(pressure)
    boundary = sign[left] != sign[right]
    nodes = sorted(set(left.tolist()) | set(right.tolist()))
    nan = float('nan')

    out: Dict[str, Any] = {
        'n_interfaces': int(keep.sum()),
        'n_boundary': int(boundary.sum()),
        'p_shear': float(shear.mean()) if shear.size else nan,
        'p_shear_boundary': float(shear[boundary].mean()) if boundary.any() else nan,
    }
    out['coverage'] = out['p_shear_boundary']
    out['enrichment'] = (out['p_shear_boundary'] / out['p_shear']
                         if out['p_shear'] else nan)

    rng = np.random.default_rng(seed)
    bulk_ids = np.where(bulk)[0]
    out['z'] = nan
    if boundary.any():
        draws = np.empty(n_null)
        for k in range(n_null):
            shuffled = sign.copy()
            shuffled[bulk_ids] = rng.permutation(sign[bulk_ids])
            drawn = shuffled[left] != shuffled[right]
            draws[k] = shear[drawn].mean() if drawn.any() else nan
        spread = np.nanstd(draws)
        if spread:
            out['z'] = float((out['p_shear_boundary'] - np.nanmean(draws)) / spread)

    out['per_family'] = {}
    weighted, weight = 0.0, 0.0
    for name, selected in (('a', family[keep]), ('b', ~family[keep])):
        base = shear[selected].mean() if selected.any() else nan
        on_boundary = shear[selected & boundary].mean() if (selected & boundary).any() else nan
        ratio = on_boundary / base if base else nan
        out['per_family'][name] = {'n': int(selected.sum()),
                                   'n_boundary': int((selected & boundary).sum()),
                                   'p_shear': float(base), 'enrichment': float(ratio)}
        if np.isfinite(ratio):
            share = float((selected & boundary).sum())
            weighted, weight = weighted + share * ratio, weight + share
    out['enrichment_stratified'] = weighted / weight if weight else nan

    # The topological test. Delete the shear-dominated interfaces and ask whether what remains is
    # pressure-pure, against deleting the same number at random.
    if nodes:
        observed = _pure_fraction(list(zip(left[~shear], right[~shear])), sign, nodes)
        null = np.array([_pure_fraction(list(zip(left[~drawn], right[~drawn])), sign, nodes)
                         for drawn in (rng.permutation(shear) for _ in range(min(n_null, 500)))])
        out['cutset_purity'] = float(observed)
        out['cutset_purity_null'] = float(null.mean())
        out['cutset_z'] = float((observed - null.mean()) / null.std()) if null.std() else nan
    else:
        out['cutset_purity'] = out['cutset_purity_null'] = out['cutset_z'] = nan
    return out


# ── Where the deployment energy is stored ─────────────────────────────────────────────────────

def energy_split(state, displacement, reference_bond_vectors, *, k_stretch, k_shear, k_rot,
                 linearized: bool = True) -> Dict[str, Any]:
    """The stored hinge energy, split into the three springs of the reference model.

    ⚠ This answers a DIFFERENT question from :func:`nff.analysis.load_paths.force_budget`, and the two
    disagree completely on this sheet. The force budget says how load crosses an interface, and it
    sums forces only. This says where the work of deployment ends up. A rotating-tile sheet stores
    almost all of it in the rotational spring while transmitting almost none of it as a couple, and
    reporting only the first reads as a complete account of the mechanics when it is not.

    Exact rather than reconstructed. The three springs are independent quadratics, so evaluating the
    SOLVER'S OWN bond energy three times with the other two stiffnesses zeroed returns each term.
    Rebuilding the split from a strain measure of one's own choosing is the trap: a corotated
    reconstruction disagreed with the solve's stored energy by up to 7% on these runs, because they
    solve with ``linearized_strains``. Check the returned ``total`` against the ``energy`` array of
    the matching ``load_path.npz`` and do not report a split that fails it.

    Args:
        state: the ``CentroidalState`` the solve ran on.
        displacement: (n_faces, 3) final ``[dx, dy, dtheta]``.
        reference_bond_vectors: (n_hinges, 2) rest bond vectors, from the solve.
        linearized: match the run's ``physics.linearized_strains``. These configs set it true.
    """
    cnv = jnp.asarray(state.centroid_node_vectors)
    bond = np.asarray(state.bond_connectivity)
    n_faces, n_nodes, _ = cnv.shape
    # The gather Stage 2 itself uses, so these are the DOFs the bond energy saw.
    nodes = face_to_node_kinematics_fn(jnp.asarray(displacement), cnv).reshape((n_faces * n_nodes, 3))
    pair = (nodes[bond[:, 0]], nodes[bond[:, 1]])
    reference = jnp.asarray(reference_bond_vectors)
    energy_fn = ligament_energy_linearized if linearized else ligament_energy

    channels = {'axial': (k_stretch, 0.0, 0.0),
                'shear': (0.0, k_shear, 0.0),
                'rotation': (0.0, 0.0, k_rot)}
    per_hinge = {name: np.asarray(energy_fn(pair, reference_vector=reference,
                                            k_stretch=ka, k_shear=ks, k_rot=kr))
                 for name, (ka, ks, kr) in channels.items()}

    totals = {name: float(v.sum()) for name, v in per_hinge.items()}
    total = sum(totals.values()) or 1.0
    return {
        'total': sum(totals.values()),
        'totals': totals,
        'shares': {name: v / total for name, v in totals.items()},
        'per_hinge': per_hinge,
    }


def rotation_departure(field: ForceField, dec, per_hinge: Dict[str, np.ndarray],
                       n_bins: int = 4) -> Dict[str, Any]:
    """How far each hinge leaves pure rotation, against the force it carries.

    A rotating-tile sheet has one soft mode, and a hinge uses it wherever it can: a lightly loaded
    interface stores essentially all of its energy in the rotational spring. Where the load will not
    pass on rotation alone the hinge is pushed into its stiff directions, and the axial and shear
    springs start to take a share. This measures that departure and its dependence on load.

    ⚠ Report it as a graded crossover, never as a threshold. The rank correlation is essentially
    one, so the departure is monotone in the force with no knee anywhere to call a transition point.

    Args:
        per_hinge: the ``'per_hinge'`` mapping from :func:`energy_split`, so the components are the
            exact ones already checked against the solve rather than a second reconstruction.
        n_bins: quantile bins of interface force magnitude, lowest first.
    """
    rotation = np.asarray(per_hinge['rotation'], dtype=float)
    other = np.asarray(per_hinge['axial'], dtype=float) + np.asarray(per_hinge['shear'], dtype=float)
    total = rotation + other
    share = np.where(total > 0, other / np.where(total > 0, total, 1.0), 0.0)

    force = np.asarray(dec['f_mag'], dtype=float) / (field.f0 or 1.0)
    edges = np.quantile(force, np.linspace(0.0, 1.0, n_bins + 1))
    bins = []
    for k in range(n_bins):
        upper = force <= edges[k + 1] if k == n_bins - 1 else force < edges[k + 1]
        selected = (force >= edges[k]) & upper
        bins.append(float(share[selected].mean()) if selected.any() else float('nan'))

    order_f = np.argsort(np.argsort(force))          # Spearman without a scipy dependency here
    order_s = np.argsort(np.argsort(share))
    rho = (float(np.corrcoef(order_f, order_s)[0, 1])
           if force.size > 2 and order_f.std() and order_s.std() else float('nan'))

    return {
        'share_by_force_quantile': bins,
        'share_weakest': bins[0],
        'share_strongest': bins[-1],
        'ratio': (bins[-1] / bins[0]) if bins[0] else float('nan'),
        'spearman': rho,
        'share_overall': float(share.mean()),
    }


def force_clustering(field: ForceField, dec, quantile: float = 0.75) -> Dict[str, Any]:
    """Whether the strongly loaded interfaces form compact regions or scatter over the sheet.

    Measured on interfaces rather than tiles: among the interfaces with at least one strongly loaded
    tile, the share whose BOTH tiles are strongly loaded. Under a random placement of the same number
    of strong tiles that share is about ``1 - quantile``, so the ratio against it is the compactness.
    """
    force = scatter_to_tiles(field, np.asarray(dec['f_mag']))
    strong = force >= np.quantile(force, quantile)
    pairs = np.asarray(field.face_pairs)
    both = int((strong[pairs[:, 0]] & strong[pairs[:, 1]]).sum())
    either = int((strong[pairs[:, 0]] | strong[pairs[:, 1]]).sum())
    observed = both / either if either else float('nan')
    chance = 1.0 - quantile
    return {
        'n_strong_tiles': int(strong.sum()),
        'observed': observed,
        'chance': chance,
        'ratio': observed / chance if chance else float('nan'),
    }


def scatter_to_tiles(field: ForceField, per_hinge: np.ndarray) -> np.ndarray:
    """(n_faces,) sum of ``|per_hinge|`` over the hinges touching each tile."""
    out = np.zeros(field.n_faces)
    pairs = np.asarray(field.face_pairs)
    for column in (0, 1):
        np.add.at(out, pairs[:, column], np.abs(per_hinge))
    return out


# ── How concentrated each channel is ──────────────────────────────────────────────────────────

def _effective_count(values: np.ndarray) -> float:
    """Participation number ``(sum f)^2 / sum f^2``: how many hinges genuinely carry the channel."""
    values = np.abs(np.asarray(values, dtype=float))
    values = values[values > 0.0]
    if values.size == 0:
        return 0.0
    return float(values.sum() ** 2 / (values ** 2).sum())


def _top_decile_share(values: np.ndarray) -> float:
    values = np.sort(np.abs(np.asarray(values, dtype=float)))[::-1]
    total = values.sum()
    if total <= 0.0:
        return float('nan')
    return float(values[:max(1, values.size // 10)].sum() / total)


def channel_concentration(dec) -> Dict[str, Dict[str, float]]:
    """Per channel, how few interfaces carry it.

    A channel spread over many weak interfaces and one concentrated in a few strong ones can hold
    the same share of the force budget, so the budget alone does not say which is a path.
    """
    channels = {
        'tension': np.maximum(dec['f_n'], 0.0),
        'compression': np.maximum(-dec['f_n'], 0.0),
        'shear': np.abs(dec['f_t']),
    }
    return {
        name: {
            'n_eff': _effective_count(values),
            'top_decile_share': _top_decile_share(values),
            'n_active': int(np.count_nonzero(values > 0.0)),
        }
        for name, values in channels.items()
    }


# ── Tension routes ────────────────────────────────────────────────────────────────────────────

def tension_routes(field: ForceField, dec, keep, *, min_run: int = 3) -> list:
    """The end-to-end spine of every strong-tension run of at least ``min_run`` tiles.

    The component and spine selection are shared with :func:`nff.analysis.load_path_viz.
    draw_tension_routes`, so the routes a paragraph counts and the routes the figure draws are the
    same objects. ``directness`` is the spine's arc length over the straight line between its ends:
    one means the load travels in a straight line from where it is applied to where it is resisted.

    Args:
        field: the extracted force field.
        dec: its :func:`~nff.analysis.load_paths.interface_decomposition`.
        keep: (n_hinges,) bool, the hinges above the drawing threshold.
        min_run: shortest run, in tiles, that counts as a route rather than a fragment.

    Returns:
        ``{'routes': [...], 'graph_hinges': [...], 'spine_hinges': [...]}``. Each route carries
        ``node_path``, ``hinges``, ``n_tiles``, ``arc_length``, ``chord`` and ``directness``.
        ``graph_hinges`` is what survived into the graph, which is fewer than ``keep`` selects when
        one tile pair carries several hinges.
    """
    empty = {'routes': [], 'graph_hinges': [], 'spine_hinges': []}
    modes = edge_modes(dec)
    selected = np.flatnonzero(np.asarray(keep) & (modes == 'tension'))
    if selected.size == 0:
        return empty

    graph = nx.Graph()
    for h in selected:
        i, j = (int(v) for v in field.face_pairs[h])
        graph.add_edge(i, j, hinge=int(h))
    for _, _, data in graph.edges(data=True):
        data['distance'] = 1.0 / (dec['f_norm'][data['hinge']] + 1e-9)

    face_xy = np.asarray(field.face_xy)
    routes = []
    for component in nx.connected_components(graph):
        block = graph.subgraph(component)
        if block.number_of_nodes() < min_run:
            continue
        # Both ends found by WEIGHTED distance, matching the drawing: the spine follows the force,
        # not the geometry, so a strong detour is preferred to a weak short cut.
        start = next(iter(component))
        far = max(nx.single_source_dijkstra_path_length(block, start, weight='distance').items(),
                  key=lambda kv: kv[1])[0]
        other = max(nx.single_source_dijkstra_path_length(block, far, weight='distance').items(),
                    key=lambda kv: kv[1])[0]
        node_path = nx.shortest_path(block, far, other, weight='distance')

        points = face_xy[node_path]
        arc = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
        chord = float(np.linalg.norm(points[-1] - points[0]))
        routes.append({
            'node_path': [int(n) for n in node_path],
            'hinges': [int(block[a][b]['hinge']) for a, b in zip(node_path, node_path[1:])],
            'n_tiles': len(node_path),
            'arc_length': arc,
            'chord': chord,
            'directness': float(arc / chord) if chord > 1e-12 else float('nan'),
        })

    spine = sorted({h for route in routes for h in route['hinges']})
    graph_hinges = sorted(int(data['hinge']) for _, _, data in graph.edges(data=True))
    return {'routes': routes, 'graph_hinges': graph_hinges, 'spine_hinges': spine}


def route_directness(field: ForceField, dec, *, threshold: Optional[float] = None,
                     min_run: int = 3, loop_at: float = 1.5) -> Dict[str, Any]:
    """Summary of :func:`tension_routes` at the figure's own default threshold.

    ⚠ Report the MEDIAN and the principal route, not the plain mean. A few routes double back on
    themselves, and for those the straight line between the ends is a poor denominator: one such
    route reaches a ratio above four while carrying a third of the load its neighbours carry. The
    mean over routes is then dominated by the fragments rather than by the load path. ``n_looping``
    counts them so they are visible rather than averaged away.
    """
    if threshold is None:
        threshold = float(np.mean(dec['f_norm']))
    routes = tension_routes(field, dec, dec['f_norm'] >= threshold, min_run=min_run)['routes']
    for route in routes:
        route['carried'] = float(np.sum(dec['f_norm'][route['hinges']]))

    finite = [r for r in routes if np.isfinite(r['directness'])]
    values = np.array([r['directness'] for r in finite]) if finite else np.array([])
    carried = np.array([r['carried'] for r in finite]) if finite else np.array([])
    principal = max(finite, key=lambda r: r['carried']) if finite else None

    return {
        'threshold': threshold,
        'n_routes': len(routes),
        'route_sizes': sorted((r['n_tiles'] for r in routes), reverse=True),
        'tiles_on_routes': int(sum(r['n_tiles'] for r in routes)),
        'directness_median': float(np.median(values)) if values.size else float('nan'),
        'directness_weighted': (float(np.sum(values * carried) / carried.sum())
                                if carried.sum() > 0 else float('nan')),
        'directness_principal': float(principal['directness']) if principal else float('nan'),
        'principal_tiles': int(principal['n_tiles']) if principal else 0,
        'directness_best': float(values.min()) if values.size else float('nan'),
        'directness_worst': float(values.max()) if values.size else float('nan'),
        'n_looping': int((values > loop_at).sum()) if values.size else 0,
    }


# ── The two scales of the field ───────────────────────────────────────────────────────────────

def bipartite_parity(field: ForceField) -> tuple:
    """``(parity, is_bipartite)`` -- the checkerboard 2-colouring of the tile adjacency graph.

    In a rotating-tile sheet neighbouring tiles turn in opposite senses, so the hinge graph is
    bipartite and the checkerboard mode is exactly its ``+-1`` colouring rather than a fitted
    wavelength. ``is_bipartite`` is returned rather than asserted: a sheet whose graph is not
    bipartite has no checkerboard mode and the amplitude below is meaningless.
    """
    pairs = np.asarray(field.face_pairs)
    neighbours: Dict[int, list] = {}
    for i, j in pairs:
        neighbours.setdefault(int(i), []).append(int(j))
        neighbours.setdefault(int(j), []).append(int(i))

    colour: Dict[int, int] = {}
    is_bipartite = True
    for seed in range(field.n_faces):
        if seed in colour:
            continue
        colour[seed] = 0
        stack = [seed]
        while stack:
            node = stack.pop()
            for other in neighbours.get(node, ()):
                if other not in colour:
                    colour[other] = 1 - colour[node]
                    stack.append(other)
                elif colour[other] == colour[node]:
                    is_bipartite = False
    parity = np.array([1 - 2 * colour[i] for i in range(field.n_faces)], dtype=float)
    return parity, is_bipartite


def polynomial_detrend(values: np.ndarray, xy: np.ndarray, degree: int = 3) -> np.ndarray:
    """``values`` with a smooth polynomial trend in position removed.

    The trend is the load's own doing and the residual is the sheet's. Skipping this step is what
    makes an alternating field read as a correlated one.
    """
    xy = np.asarray(xy, dtype=float)
    x = (xy[:, 0] - xy[:, 0].mean()) / (xy[:, 0].std() + 1e-12)
    y = (xy[:, 1] - xy[:, 1].mean()) / (xy[:, 1].std() + 1e-12)
    basis = np.column_stack([x ** a * y ** b
                             for a in range(degree + 1) for b in range(degree + 1 - a)])
    coefficients, *_ = np.linalg.lstsq(basis, np.asarray(values, dtype=float), rcond=None)
    return np.asarray(values, dtype=float) - basis @ coefficients


def neighbour_correlation(field: ForceField, values: np.ndarray) -> float:
    """Correlation of ``values`` across hinge-connected tile pairs. Negative means alternating."""
    pairs = np.asarray(field.face_pairs)
    values = np.asarray(values, dtype=float)
    a, b = values[pairs[:, 0]], values[pairs[:, 1]]
    if a.size < 5 or a.std() < 1e-15 or b.std() < 1e-15:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def checkerboard_amplitude(field: ForceField, values: Optional[np.ndarray] = None,
                           degree: int = 3) -> Dict[str, Any]:
    """How much of a tile field sits in the mechanism's alternating mode, after detrending.

    ``variance_share`` is the fraction of the DETRENDED variance explained by the checkerboard, so
    it measures the background against the fluctuation it lives in, not against the applied load.
    """
    if values is None:
        values = tile_pressure(field)
    values = np.asarray(values, dtype=float)
    parity, is_bipartite = bipartite_parity(field)
    residual = polynomial_detrend(values, np.asarray(field.face_xy), degree=degree)

    def _corr(u, v):
        if u.std() < 1e-15 or v.std() < 1e-15:
            return float('nan')
        return float(np.corrcoef(u, v)[0, 1])

    detrended = _corr(residual, parity)
    return {
        'is_bipartite': bool(is_bipartite),
        'corr_raw': _corr(values, parity),
        'corr_detrended': detrended,
        'variance_share': float(detrended ** 2) if np.isfinite(detrended) else float('nan'),
        'neighbour_corr_raw': neighbour_correlation(field, values),
        'neighbour_corr_detrended': neighbour_correlation(field, residual),
    }


# ── Where the compression sits ────────────────────────────────────────────────────────────────

def support_band_pressure(field: ForceField, support_faces: Sequence[int], *,
                          along=(0.15, 0.85), across: float = 0.25) -> Dict[str, Any]:
    """Tile pressure inside the band spanning the supports, against the rest of the sheet.

    A sheet pulled away from two anchors slackens between them, as a cloth does. The mechanism
    cannot slacken, so it takes that region into compression instead. The band is the strip joining
    the two supports, trimmed at both ends so the supported tiles themselves are excluded.

    Args:
        field: the extracted force field.
        support_faces: the tiles carrying the supports. Exactly two are used, the farthest apart.
        along: fractional start and end along the support chord.
        across: half-width of the band, as a fraction of the chord length.
    """
    support_faces = [int(f) for f in dict.fromkeys(support_faces)]
    if len(support_faces) < 2:
        return {'n_band': 0, 'reason': 'fewer than two distinct support tiles'}

    xy = np.asarray(field.face_xy)
    anchors = xy[support_faces]
    separations = np.linalg.norm(anchors[:, None] - anchors[None], axis=-1)
    a, b = np.unravel_index(separations.argmax(), separations.shape)
    start, end = anchors[a], anchors[b]

    chord = end - start
    length = float(np.linalg.norm(chord))
    if length < 1e-12:
        return {'n_band': 0, 'reason': 'supports coincide'}
    unit = chord / length

    offset = xy - start
    projected = offset @ unit
    perpendicular = np.abs(offset[:, 0] * unit[1] - offset[:, 1] * unit[0])
    band = ((projected > along[0] * length) & (projected < along[1] * length)
            & (perpendicular < across * length))

    pressure = tile_pressure(field)
    scale = float(np.abs(pressure).mean()) or 1.0
    return {
        'support_faces': [support_faces[a], support_faces[b]],
        'n_band': int(band.sum()),
        'pressure_band': float(pressure[band].mean() / scale) if band.any() else float('nan'),
        'pressure_rest': float(pressure[~band].mean() / scale),
        'compressive_fraction_band': float((pressure[band] < 0).mean()) if band.any() else float('nan'),
        'compressive_fraction_rest': float((pressure[~band] < 0).mean()),
    }


def pressure_profile(field: ForceField, n_bins: int = 5, axis: int = 0) -> list:
    """Mean tile pressure in ``n_bins`` strips across the sheet, normalised by mean ``|p|``."""
    pressure = tile_pressure(field)
    scale = float(np.abs(pressure).mean()) or 1.0
    coordinate = np.asarray(field.face_xy)[:, axis]
    edges = np.linspace(coordinate.min(), coordinate.max(), n_bins + 1)
    profile = []
    for k in range(n_bins):
        upper = coordinate <= edges[k + 1] if k == n_bins - 1 else coordinate < edges[k + 1]
        inside = (coordinate >= edges[k]) & upper
        profile.append(float(pressure[inside].mean() / scale) if inside.any() else float('nan'))
    return profile


# ── The two cut families ──────────────────────────────────────────────────────────────────────

def orientation_families(field: ForceField) -> np.ndarray:
    """(n_hinges,) bool splitting the hinges into their two cut-orientation families.

    ⚠ The split is found FROM THE DATA, not at a fixed angle. An orientation is defined modulo
    180 degrees, so each axis is mapped to its doubled angle, where two families a quarter turn
    apart sit at opposite points, and the set is cut along its own principal direction.

    A fixed boundary at 0 and 90 degrees, which is the obvious implementation, is wrong on this
    sheet and wrong in a way that looks plausible: the measured families centre near 90 and 170
    degrees, so a 0/90 boundary bisects BOTH clusters instead of separating them. It still returns
    a near-even split, 140 against 135, which is why it survives inspection, yet it agrees with the
    correct split on only 58% of hinges and manufactures a difference between the halves.
    """
    axis = np.asarray(field.hinge_axis, dtype=float)
    angle = np.mod(np.arctan2(axis[:, 1], axis[:, 0]), np.pi)
    doubled = np.column_stack([np.cos(2.0 * angle), np.sin(2.0 * angle)])
    centred = doubled - doubled.mean(axis=0)
    _, vectors = np.linalg.eigh(centred.T @ centred)
    return (centred @ vectors[:, -1]) > 0.0


def cut_family_split(field: ForceField, dec, mask: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """The sheet's two cut orientations, and how differently they are loaded.

    A rotating-tile sheet cuts in two orientation families. When the load sits preferentially on
    one of them, the alternation a reader sees in the figure is between families, not between
    neighbouring tiles.

    Args:
        mask: optional (n_hinges,) bool restricting the split, e.g. to the weak interfaces.
    """
    family = orientation_families(field)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
    else:
        mask = np.ones(family.shape, dtype=bool)

    f_n = dec['f_n'] / dec['f0']
    f_t = np.abs(dec['f_t']) / dec['f0']
    out: Dict[str, Any] = {}
    for name, selected in (('a', family & mask), ('b', (~family) & mask)):
        out[name] = {
            'n': int(selected.sum()),
            'mean_f_n': float(f_n[selected].mean()) if selected.any() else float('nan'),
            'mean_f_t': float(f_t[selected].mean()) if selected.any() else float('nan'),
            'tension_fraction': float((f_n[selected] > 0).mean()) if selected.any() else float('nan'),
            'shear_dominated_fraction': (float(dec['shear_dominated'][selected].mean())
                                         if selected.any() else float('nan')),
        }
    means = [out['a']['mean_f_n'], out['b']['mean_f_n']]
    finite = [m for m in means if np.isfinite(m) and m > 0]
    out['normal_force_ratio'] = float(max(finite) / min(finite)) if len(finite) == 2 else float('nan')
    return out


# ── Tile shape ────────────────────────────────────────────────────────────────────────────────

def tile_shape(face_nodes: np.ndarray) -> tuple:
    """``(aspect_ratio, long_axis_deg)`` per tile, from the mean lengths of its opposite edges.

    Quadrilateral tiles only. Panels are rigid, so this is a property of the DESIGN and is
    unchanged by deployment; only the angle rotates with the tile.
    """
    nodes = np.asarray(face_nodes, dtype=float)
    if nodes.ndim != 3 or nodes.shape[1] != 4:
        raise ValueError(f"tile_shape expects quadrilateral tiles, got shape {nodes.shape}")
    edges = nodes - np.roll(nodes, -1, axis=1)
    lengths = np.linalg.norm(edges, axis=2)
    side_a = 0.5 * (lengths[:, 0] + lengths[:, 2])
    side_b = 0.5 * (lengths[:, 1] + lengths[:, 3])
    aspect = np.maximum(side_a, side_b) / np.maximum(np.minimum(side_a, side_b), 1e-12)
    long_edge = np.where((side_a >= side_b)[:, None], edges[:, 0, :], edges[:, 1, :])
    angle = np.degrees(np.mod(np.arctan2(long_edge[:, 1], long_edge[:, 0]), np.pi))
    return aspect, angle


def elongation_alternation(field: ForceField, *, region: Optional[np.ndarray] = None,
                           aspect_min: float = 1.4) -> Dict[str, Any]:
    """Whether the elongated tiles of a region alternate in orientation.

    A nearest-neighbour long-axis difference near ninety degrees is the rotating-RECTANGLE motif;
    near zero the tiles are simply aligned.
    """
    aspect, angle = tile_shape(field.face_nodes)
    if region is None:
        region = np.ones(aspect.shape, dtype=bool)
    region = np.asarray(region, dtype=bool)
    elongated = region & (aspect > aspect_min)
    result = {
        'n_region': int(region.sum()),
        'n_elongated': int(elongated.sum()),
        'aspect_median': float(np.median(aspect[region])) if region.any() else float('nan'),
        'aspect_max': float(aspect[region].max()) if region.any() else float('nan'),
        'fraction_above_1p5': float((aspect[region] > 1.5).mean()) if region.any() else float('nan'),
    }
    if elongated.sum() < 4:
        result['neighbour_angle_median'] = float('nan')
        return result

    points = np.asarray(field.face_xy)[elongated]
    separations = np.linalg.norm(points[:, None] - points[None], axis=-1)
    np.fill_diagonal(separations, np.inf)
    nearest = separations.argmin(axis=1)
    selected = angle[elongated]
    difference = np.abs(selected - selected[nearest])
    difference = np.minimum(difference, 180.0 - difference)
    result['neighbour_angle_median'] = float(np.median(difference))
    result['neighbour_angle_mean'] = float(difference.mean())
    return result


def region_mask(field: ForceField, fraction=(0.0, 1 / 3), axis: int = 0) -> np.ndarray:
    """(n_faces,) bool selecting a fractional slab of the sheet along ``axis``."""
    coordinate = np.asarray(field.face_xy)[:, axis]
    low, high = np.quantile(coordinate, [fraction[0], fraction[1]])
    upper = coordinate <= high if fraction[1] >= 1.0 else coordinate < high
    return (coordinate >= low) & upper


# ── Uniform patches ───────────────────────────────────────────────────────────────────────────

def motif_families(descriptors: np.ndarray, n_families: int, *,
                   max_iter: int = 200) -> tuple:
    """Group motifs by their FULL geometry, into ``n_families`` families of similar shape.

    Lloyd's algorithm on the standardised descriptors, so every component of the descriptor counts
    equally whatever its units. The initial centres are evenly spaced order statistics along the
    first principal direction, which makes the result deterministic: the figure that consumes this
    must redraw identically without carrying a seed.

    ⚠ The descriptor is deliberately SHORT. On a deployed kirigami sheet anything panel-derived is
    contaminated, because deployment counter-rotates neighbouring panels: measured over five
    sheets, appending mean panel area and aspect to a two-component void descriptor drops the
    spatial coherence of the families from 1.12--1.44 to 1.15--1.29 times chance, and a ten-
    component sorted-panel descriptor drops it to 0.80--1.38. Adding features here makes the
    families worse, not sharper.

    Args:
        descriptors: ``(n_motifs, n_features)``, one row per motif.
        n_families: how many families to split into, at least two.
        max_iter: Lloyd iterations before giving up on exact convergence.

    Returns:
        ``(labels, centroids)`` -- ``(n_motifs,)`` int, and ``(n_families, n_features)`` in the
        ORIGINAL units of ``descriptors``.
    """
    x = np.asarray(descriptors, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"motif_families expects (n_motifs, n_features), got shape {x.shape}")
    if n_families < 2:
        raise ValueError(f"n_families must be at least 2, got {n_families}")
    if n_families > x.shape[0]:
        raise ValueError(f"{n_families} families asked of {x.shape[0]} motifs")

    mean, spread = x.mean(axis=0), x.std(axis=0)
    z = (x - mean) / np.maximum(spread, 1e-12)

    # Order along the leading principal direction, then take evenly spaced quantiles of it as the
    # seeds. Any deterministic spread of seeds converges to the same partition on well-separated
    # data; this one also degrades gracefully when the data are not.
    _, _, vt = np.linalg.svd(z - z.mean(axis=0), full_matrices=False)
    order = np.argsort(z @ vt[0])
    seeds = order[np.linspace(0, len(order) - 1, n_families).round().astype(int)]
    centres = z[seeds].copy()

    labels = np.zeros(len(z), dtype=int)
    for _ in range(max_iter):
        labels = np.argmin(((z[:, None, :] - centres[None]) ** 2).sum(axis=2), axis=1)
        moved = np.stack([z[labels == c].mean(axis=0) if np.any(labels == c) else centres[c]
                          for c in range(n_families)])
        if np.allclose(moved, centres):
            break
        centres = moved
    return labels, centres * np.maximum(spread, 1e-12) + mean


def clustered_groups(labels: np.ndarray, *, min_size: int = 3) -> tuple:
    """Where like motifs sit TOGETHER: the connected groups of at least ``min_size`` same family.

    Similarity and proximity are two different claims, and this function tests the second one
    alone. A motif whose family has no like neighbours is left out of the mask, so a drawing built
    on it colours a motif only where both hold.

    Args:
        labels: ``(R, C)`` integer family per lattice site.
        min_size: smallest connected run of one family that counts as a group.

    Returns:
        ``(mask, groups)`` -- ``(R, C)`` bool, and ``(family, size)`` pairs sorted large first.
    """
    lab = np.asarray(labels, dtype=int)
    if lab.ndim != 2:
        raise ValueError(f"clustered_groups expects a 2-D lattice, got shape {lab.shape}")
    mask = np.zeros(lab.shape, dtype=bool)
    groups = []
    for family in np.unique(lab):
        components = patch_labels(lab == family)
        for c in range(1, int(components.max()) + 1):
            member = components == c
            if int(member.sum()) >= min_size:
                mask |= member
                groups.append((int(family), int(member.sum())))
    return mask, sorted(groups, key=lambda g: -g[1])


def family_contiguity(labels: np.ndarray) -> Dict[str, Any]:
    """Whether a lattice of family labels forms REGIONS or is scattered.

    Compares the fraction of orthogonally adjacent sites sharing a family against the fraction
    expected if the same labels were shuffled over the lattice. A ratio above one is clustering;
    a ratio BELOW one is alternation, which is what a period-two motif looks like and what a
    plain neighbour count would otherwise report as "no structure".

    Args:
        labels: ``(R, C)`` integer family per lattice site.

    Returns:
        ``observed``, ``chance``, ``ratio`` and ``n_regions`` (connected components, any family).
    """
    lab = np.asarray(labels, dtype=int)
    if lab.ndim != 2:
        raise ValueError(f"family_contiguity expects a 2-D lattice, got shape {lab.shape}")
    same = total = 0
    for a, b in ((lab[:-1, :], lab[1:, :]), (lab[:, :-1], lab[:, 1:])):
        same += int((a == b).sum())
        total += int(a.size)
    share = np.bincount(lab.ravel() - lab.min(), minlength=1) / lab.size
    chance = float((share ** 2).sum())
    observed = float(same / total) if total else float('nan')
    regions = 0
    for value in np.unique(lab):
        regions += int(patch_labels(lab == value).max())
    return {'observed': observed, 'chance': chance,
            'ratio': observed / chance if chance else float('nan'),
            'n_regions': regions}


def patch_labels(mask: np.ndarray) -> np.ndarray:
    """``(M, N)`` int labelling each connected patch from one, zero where ``mask`` is false.

    Four-connectivity: two panels meeting only at a corner are separate patches, matching how a
    reader sees them on the sheet.
    """
    mask = np.asarray(mask, dtype=bool)
    labels = np.zeros(mask.shape, dtype=int)
    if not mask.any():
        return labels
    lattice = nx.grid_2d_graph(*mask.shape)
    members = [tuple(ij) for ij in np.argwhere(mask)]
    for n, component in enumerate(nx.connected_components(lattice.subgraph(members)), start=1):
        for i, j in component:
            labels[i, j] = n
    return labels


# ── The cut field ─────────────────────────────────────────────────────────────────────────────

def lattice_autocorrelation(values: np.ndarray, lags=(1, 2, 3),
                            block: Optional[tuple] = None) -> Dict[str, Any]:
    """Autocorrelation of a lattice field along both axes, over the whole lattice or a sub-block.

    A motif confined to one region of the sheet does not survive averaging over the sheet, so the
    whole-lattice and sub-block answers can differ in sign. Both are reported.

    Args:
        values: the 2-D lattice field, e.g. ``sigmoid(z)`` on the cut lattice.
        lags: lattice spacings to report. A sign change between lag one and lag two is a period-two
            motif; alternating tiles show it and a smooth gradient does not.
        block: optional ``(rows, cols)`` slices selecting a sub-block.
    """
    field_2d = np.asarray(values, dtype=float)
    if block is not None:
        field_2d = field_2d[block[0], block[1]]
    field_2d = field_2d - field_2d.mean()

    def _at(axis: int, lag: int) -> float:
        if field_2d.shape[axis] <= lag:
            return float('nan')
        head = np.take(field_2d, range(0, field_2d.shape[axis] - lag), axis=axis).ravel()
        tail = np.take(field_2d, range(lag, field_2d.shape[axis]), axis=axis).ravel()
        if head.std() < 1e-15 or tail.std() < 1e-15:
            return float('nan')
        return float(np.corrcoef(head, tail)[0, 1])

    return {
        'shape': tuple(int(s) for s in field_2d.shape),
        'axis_0': {f'lag_{lag}': _at(0, lag) for lag in lags},
        'axis_1': {f'lag_{lag}': _at(1, lag) for lag in lags},
    }
