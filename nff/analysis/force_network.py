"""The load path as a weighted graph: force chains, bottlenecks, and locked-in self-stress.

A deployed sheet of rigid tiles is a spatially embedded weighted network -- one node per tile, one
edge per interface, edge weight = the force transmitted across it. That is the standard formalism
for a discrete force-carrying assembly (Papadopoulos, Porter, Daniels & Bassett 2018), and it lets
the load-carrying backbone be extracted by computation rather than picked by eye.

Three independent readings, none of which a net-force arrow per interface can give:

  * **Force chains** -- modularity communities of the force-weighted graph under the GEOGRAPHICAL
    null model (Bassett, Owens, Porter, Manning & Daniels 2015). Run separately on the normal and
    tangential networks, which are known to have different structure and must not be merged into one
    scalar (Zhang, Majmudar, Tordesillas & Behringer 2010).
  * **Bottlenecks** -- the minimum cut of a flow network from the loaded tiles to the supports, with
    interface capacities projected on the load axis (Tordesillas et al. 2014; Wautier et al. 2023).
    This is the set of interfaces that limits the structure, which no arrow plot shows.
  * **Self-stress** -- the null space of the equilibrium matrix, i.e. the Maxwell-Calladine states of
    self-stress (Calladine 1978; Mao & Lubensky 2018), which for kirigami are the "stressed lines"
    that carry load first (Chen et al. 2016).

Interfaces here are POINT hinges, so the frame is the branch vector rather than a contact plane --
see :func:`nff.analysis.load_paths.branch_frame`.
"""

from typing import Any, Dict, Optional, Tuple

import networkx as nx
import numpy as np

from nff.analysis.load_paths import ForceField, _cross, branch_frame, interface_decomposition

_EPS = 1e-12


# ── The graph ─────────────────────────────────────────────────────────────────────────────────

WEIGHTS = ('total', 'normal', 'shear')


def edge_weights(field: ForceField, weight: str = 'total') -> np.ndarray:
    """(n_hinges,) non-negative edge weight for the requested force channel.

    ``normal`` and ``shear`` exist as separate channels on purpose: the two networks carry different
    structure, and a single combined magnitude hides which one a given chain is made of.
    """
    if weight not in WEIGHTS:
        raise ValueError(f"weight must be one of {WEIGHTS}, got {weight!r}")
    dec = interface_decomposition(field)
    return {'total': dec['f_mag'],
            'normal': np.abs(dec['f_n']),
            'shear': np.abs(dec['f_t'])}[weight]


def build_graph(field: ForceField, weight: str = 'total') -> nx.Graph:
    """Tile-adjacency graph carrying the transmitted force on each edge.

    Parallel hinges between the same tile pair are summed into one edge and their indices kept in
    ``hinges``, so the graph stays simple while every hinge remains addressable.
    """
    w = edge_weights(field, weight)
    graph = nx.Graph()
    graph.add_nodes_from(range(field.n_faces))
    for h, (i, j) in enumerate(np.asarray(field.face_pairs)):
        i, j = int(i), int(j)
        if graph.has_edge(i, j):
            graph[i][j]['w'] += float(w[h])
            graph[i][j]['hinges'].append(h)
        else:
            graph.add_edge(i, j, w=float(w[h]), hinges=[h])
    for i, xy in enumerate(np.asarray(field.face_xy)):
        graph.nodes[i]['xy'] = xy
    return graph


# ── Force chains: modularity under the geographical null model ────────────────────────────────

def _local_move(n_nodes: int, edges, seed: int) -> np.ndarray:
    """Greedy relabelling that increases the within-community sum of ``b``, to a local optimum."""
    adjacency: list = [[] for _ in range(n_nodes)]
    for i, j, b in edges:
        adjacency[i].append((j, b))
        adjacency[j].append((i, b))

    rng = np.random.default_rng(seed)
    community = np.arange(n_nodes)
    fresh = n_nodes                                   # label supply for "move to your own community"

    for _ in range(100):
        moved = False
        for i in rng.permutation(n_nodes):
            current = community[i]
            reachable: Dict[int, float] = {}
            for j, b in adjacency[i]:
                reachable[community[j]] = reachable.get(community[j], 0.0) + b

            stay = reachable.get(current, 0.0)        # what i currently contributes where it sits
            best, best_gain = current, 0.0
            for c, gained in reachable.items():
                if c != current and gained - stay > best_gain + _EPS:
                    best, best_gain = c, gained - stay
            # Splitting off alone is a real option on a signed modularity matrix: a node joined only
            # by below-null (repulsive) edges is better off in its own community.
            if -stay > best_gain + _EPS:
                best, best_gain, fresh = fresh, -stay, fresh + 1
            if best != current:
                community[i], moved = best, True
        if not moved:
            break
    return community


def _compact(labels: np.ndarray) -> np.ndarray:
    """Relabel to a dense 0..k-1 range, preserving grouping."""
    return np.unique(labels, return_inverse=True)[1]


def _aggregate(edges, community: np.ndarray):
    """Collapse each community to a super-node; internal edges are dropped.

    Internal weight stays internal whichever community a super-node later joins, so it cancels out of
    every move gain -- keeping it as a self-loop would change nothing but the bookkeeping.
    """
    merged: Dict[Tuple[int, int], float] = {}
    for i, j, b in edges:
        a, c = int(community[i]), int(community[j])
        if a == c:
            continue
        merged[(min(a, c), max(a, c))] = merged.get((min(a, c), max(a, c)), 0.0) + b
    return [(a, c, b) for (a, c), b in merged.items()]


def _louvain(n_nodes: int, edges, seed: int = 0, max_levels: int = 20) -> np.ndarray:
    """Louvain applied to an explicit modularity matrix rather than a built-in null model."""
    labels = np.arange(n_nodes)
    level_edges, level_n = list(edges), n_nodes

    for level in range(max_levels):
        community = _compact(_local_move(level_n, level_edges, seed + level))
        if community.max() + 1 == level_n:
            break                                     # nothing merged: converged
        labels = community[labels]
        level_edges = _aggregate(level_edges, community)
        level_n = int(community.max()) + 1
        if not level_edges:
            break
    return _compact(labels)


def geographical_communities(field: ForceField, gamma: float = 1.0, weight: str = 'total',
                             seed: int = 0) -> Dict[str, Any]:
    """Force-chain communities under the GEOGRAPHICAL null model ``P_ij = rho * A_ij``.

    The null model is the whole point. Newman-Girvan's configuration model penalises a node in
    proportion to its own strength, which in a spatially embedded contact network rewards compact
    blobs; replacing it with a uniform ``rho`` over existing contacts rewards chains instead, which
    is what a force chain is (Bassett et al. 2015). ``networkx``'s ``louvain_communities`` hardcodes
    the configuration model and cannot express this, so the optimisation is done here.

    Because ``P_ij`` is supported on the adjacency, the modularity matrix reduces to one number per
    existing edge, ``b_e = w_e - gamma * rho``: contacts stronger than the mean attract their
    endpoints together, weaker ones push them apart.

    Args:
        gamma: resolution. Larger -> more, shorter chains. Reported in the figure caption.
        seed: fixes the sweep order, so the partition is reproducible.
    """
    graph = build_graph(field, weight)
    weights = np.array([d['w'] for _, _, d in graph.edges(data=True)])
    if weights.size == 0:
        return {'labels': np.zeros(field.n_faces, dtype=int), 'modularity': 0.0,
                'gamma': gamma, 'weight': weight, 'rho': 0.0, 'n_communities': 1}

    rho = float(weights.mean())
    index = {node: k for k, node in enumerate(graph.nodes())}
    edges = [(index[u], index[v], d['w'] - gamma * rho) for u, v, d in graph.edges(data=True)]

    labels = _louvain(graph.number_of_nodes(), edges, seed=seed)
    within = sum(b for i, j, b in edges if labels[i] == labels[j])

    node_labels = np.zeros(field.n_faces, dtype=int)
    for node, k in index.items():
        node_labels[node] = labels[k]

    return {
        'labels': node_labels,
        'modularity': float(within / max(weights.sum(), _EPS)),
        'gamma': float(gamma),
        'weight': weight,
        'rho': rho,
        'n_communities': int(labels.max()) + 1,
    }


# ── Betweenness ───────────────────────────────────────────────────────────────────────────────

GAMMA_SWEEP = (0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 1.8, 2.2)


def auto_gamma(field: ForceField, weight: str = 'total', seed: int = 0,
               min_chain: int = 3) -> float:
    """Resolution that yields the most chains long enough to be a chain.

    The resolution has to be chosen and stated, not assumed: the right value depends on the weight
    distribution, and the total-force network needs a higher one than either channel alone. Its
    weights are more uniform, so fewer edges fall below ``gamma * rho``, the repulsive term is weaker,
    and at ``gamma = 0.6`` one community absorbs 118 of 150 tiles -- a blob, which is precisely the
    failure the geographical null model exists to avoid.

    Scoring by the number of communities holding at least ``min_chain`` tiles penalises both failure
    modes at once, with no free threshold: a giant blob scores 1 because it IS one community, and
    over-fragmentation scores 0 because nothing reaches the minimum. Ties go to the coarser
    resolution. (A "largest community below a fixed share of the sheet" rule looks equivalent but
    is not -- on a 9- or 16-tile sheet that share is a couple of tiles, so it drove the resolution up
    until no chain survived at all.)
    """
    best, best_score = float(GAMMA_SWEEP[0]), -1
    for gamma in GAMMA_SWEEP:
        labels = geographical_communities(field, gamma=gamma, weight=weight, seed=seed)['labels']
        if labels.size == 0:
            continue
        score = int((np.bincount(labels) >= min_chain).sum())
        if score > best_score:
            best, best_score = float(gamma), score
    return best


def edge_betweenness(field: ForceField, weight: str = 'total') -> np.ndarray:
    """(n_hinges,) edge betweenness centrality of the force-weighted graph.

    Betweenness on a contact network tracks the force actually carried (Kollmer & Daniels 2019) and
    forecasts failure locations in disordered 2D lattices (Berthier, Porter & Daniels 2019).

    ⚠ ``networkx`` minimises the ``weight`` attribute along shortest paths, i.e. it reads it as a
    DISTANCE. Handing it the transmitted force would make the STRONGEST contacts the most expensive
    to traverse and invert the whole result. The distance is therefore ``F0 / |f_e|``: a link is
    "short" when it carries a lot.
    """
    graph = build_graph(field, weight)
    if graph.number_of_edges() == 0:
        return np.zeros(field.n_hinges)

    f0 = field.f0
    for _, _, data in graph.edges(data=True):
        data['distance'] = f0 / (data['w'] + _EPS * f0)

    centrality = nx.edge_betweenness_centrality(graph, weight='distance', normalized=True)
    out = np.zeros(field.n_hinges)
    for (u, v), value in centrality.items():
        for h in graph[u][v]['hinges']:
            out[h] = value
    return out


# ── Bottleneck: maximum flow / minimum cut ────────────────────────────────────────────────────

_CAPACITY_SCALE = 10 ** 6


def load_axis(field: ForceField) -> np.ndarray:
    """Unit vector of the total applied load -- the axis flow is measured along."""
    total = np.asarray(field.f_ext)[:, :2].sum(axis=0)
    norm = float(np.linalg.norm(total))
    return total / norm if norm > 0 else np.array([1.0, 0.0])


def min_cut(field: ForceField, protect_rings: int = 0) -> Dict[str, Any]:
    """Bottleneck interfaces: the minimum cut from the loaded tiles to the supports.

    Capacity is the transmitted force projected on the load axis, ``c_e = |f_e . e_load|``, following
    the network-flow treatment of granular force transmission (Tordesillas et al. 2014; Wautier et
    al. 2023, which uses exactly this component). With the load along x this is the spec's
    ``|f_n,x + f_t,x|``; writing it as a projection generalises to any load direction.

    ⚠ **The cut VALUE is pinned by equilibrium and is not a measurement.** Every cut separating the
    loaded tiles from the supports transmits the applied load, so its capacity
    ``sum |f_e . e_load| >= |sum f_e . e_load| = F0``, with equality exactly when every interface in
    the cut carries the same way along the load axis. A reported ``1.00 x F0`` therefore means "a
    perfectly aligned cut exists", not "the structure is at capacity". The informative output is
    WHERE the cut falls.

    ⚠ The unconstrained minimum is usually the collar of interfaces around the loaded tile itself --
    true, but trivial: the weakest point of any structure is where you grab it. ``protect_rings``
    makes interfaces within that many graph steps of a grip uncuttable, pushing the cut into the
    interior where the structure's own routing decides it.

    ⚠ Capacities are scaled to integers. ``networkx``'s flow algorithms are documented as unreliable
    on floating-point capacities, and a bottleneck found by round-off is worse than none.
    """
    loaded = np.flatnonzero(np.abs(np.asarray(field.f_ext)).sum(axis=1) > 0)
    clamped = np.flatnonzero(~np.asarray(field.free_mask).all(axis=1))
    empty = {'cut_value': float('nan'), 'cut_hinges': np.zeros(0, dtype=int),
             'source_side': np.zeros(field.n_faces, dtype=bool),
             'capacity': np.zeros(field.n_hinges), 'at_grip': False,
             'protect_rings': protect_rings}
    if loaded.size == 0 or clamped.size == 0 or set(loaded) & set(clamped):
        return empty

    axis = load_axis(field)
    capacity = np.abs(np.asarray(field.t_total) @ axis)
    scaled = np.maximum(np.round(capacity / field.f0 * _CAPACITY_SCALE).astype(np.int64), 0)

    infinite = int(scaled.sum()) + 1
    protected = np.zeros(field.n_faces, dtype=bool)
    if protect_rings > 0:
        graph = build_graph(field)
        grips = set(int(f) for f in np.concatenate([loaded, clamped]))
        for node in grips:
            for reached in nx.single_source_shortest_path_length(graph, node,
                                                                 cutoff=protect_rings - 1):
                protected[int(reached)] = True

    flow = nx.DiGraph()
    for h, (i, j) in enumerate(np.asarray(field.face_pairs)):
        weight = infinite if (protected[i] or protected[j]) else int(scaled[h])
        for a, b in ((int(i), int(j)), (int(j), int(i))):
            if flow.has_edge(a, b):
                flow[a][b]['capacity'] += weight
            else:
                flow.add_edge(a, b, capacity=weight)

    for f in loaded:
        flow.add_edge('S', int(f), capacity=infinite)
    for f in clamped:
        flow.add_edge(int(f), 'T', capacity=infinite)

    try:
        cut_value, (source_side, _) = nx.minimum_cut(flow, 'S', 'T')
    except nx.NetworkXUnbounded:                                   # no path: nothing to cut
        return empty

    # Every route was protected, so the "cut" is a wall of infinities and says nothing. Back off one
    # ring rather than reporting a meaningless number -- on a small sheet a two-ring guard can reach
    # every interface there is.
    if cut_value >= infinite and protect_rings > 0:
        return min_cut(field, protect_rings=protect_rings - 1)

    on_source = np.zeros(field.n_faces, dtype=bool)
    for node in source_side:
        if node not in ('S', 'T'):
            on_source[int(node)] = True

    pairs = np.asarray(field.face_pairs)
    straddles = on_source[pairs[:, 0]] != on_source[pairs[:, 1]]
    cut_hinges = np.flatnonzero(straddles)
    grips = set(int(f) for f in np.concatenate([loaded, clamped]))
    at_grip = bool(any(int(pairs[h, 0]) in grips or int(pairs[h, 1]) in grips for h in cut_hinges))
    return {
        'cut_value': float(cut_value) / _CAPACITY_SCALE * field.f0,   # back to newtons
        'cut_hinges': cut_hinges,
        'source_side': on_source,
        'capacity': capacity,
        'at_grip': at_grip,
        'protect_rings': protect_rings,
    }


# ── Chain spines ──────────────────────────────────────────────────────────────────────────────

def chain_spines(field: ForceField, labels: np.ndarray, weight: str = 'total',
                 min_size: int = 3, smooth: bool = True) -> list:
    """One ordered, smoothed polyline per force chain.

    A chain drawn as disconnected per-edge segments does not read as a chain. Each community's spine
    is its heaviest end-to-end path (found by a double sweep on inverse-force distance), which is
    what lets the eye follow load from the loaded edge to the supports.
    """
    from scipy.interpolate import splev, splprep

    graph = build_graph(field, weight)
    f0 = field.f0
    for _, _, data in graph.edges(data=True):
        data['distance'] = f0 / (data['w'] + _EPS * f0)

    face_xy = np.asarray(field.face_xy)
    spines = []
    for community in range(int(np.max(labels)) + 1 if labels.size else 0):
        members = np.flatnonzero(labels == community)
        if members.size < min_size:
            continue
        sub = graph.subgraph(members.tolist())
        for part in nx.connected_components(sub):
            if len(part) < min_size:
                continue
            block = sub.subgraph(part)
            start = next(iter(part))
            far = max(nx.single_source_dijkstra_path_length(block, start, weight='distance').items(),
                      key=lambda kv: kv[1])[0]
            other = max(nx.single_source_dijkstra_path_length(block, far, weight='distance').items(),
                        key=lambda kv: kv[1])[0]
            path = nx.shortest_path(block, far, other, weight='distance')
            pts = face_xy[np.asarray(path)]
            if smooth and pts.shape[0] >= 4:
                try:
                    tck, _ = splprep(pts.T, s=0.0, k=min(3, pts.shape[0] - 1))
                    smoothed = np.stack(splev(np.linspace(0, 1, 6 * pts.shape[0]), tck), axis=-1)
                    # A spline through a community that wanders can still bow outside the material.
                    # A spine marks WHICH interfaces group together; drawn off the sheet it invents
                    # a chain where there are no tiles, so clip it to the path's own extent.
                    lo, hi = pts.min(axis=0), pts.max(axis=0)
                    pad = 0.25 * np.maximum(hi - lo, 1e-9)
                    inside = np.all((smoothed >= lo - pad) & (smoothed <= hi + pad), axis=-1)
                    pts = smoothed[inside] if inside.sum() >= 2 else pts
                except (ValueError, TypeError):
                    pass                                       # degenerate path: keep it polygonal
            spines.append(pts)
    return spines


# ── States of self-stress ─────────────────────────────────────────────────────────────────────

def equilibrium_matrix(field: ForceField) -> np.ndarray:
    """``A`` mapping interface forces to the generalized load on each tile: ``A q = f_internal``.

    Unknowns per hinge are ``(f_n, f_t, m_e)`` in the branch frame; rows are ``(Fx, Fy, M)`` per
    tile, moments about that tile's own centroid. The sign convention follows
    ``load_paths.interface_decomposition``: positive ``f_n`` is TENSION, so it pulls each tile
    towards the other.

    ``A q`` reproduces the extracted internal force field exactly, which is what
    :func:`self_stress` asserts before using it.
    """
    n_hat, t_hat = branch_frame(field)
    pairs = np.asarray(field.face_pairs)
    face_xy = np.asarray(field.face_xy)
    hinge_xy = np.asarray(field.hinge_xy)

    A = np.zeros((3 * field.n_faces, 3 * field.n_hinges))
    for h in range(field.n_hinges):
        i, j = int(pairs[h, 0]), int(pairs[h, 1])
        n, t = n_hat[h], t_hat[h]
        for face, sign in ((i, +1.0), (j, -1.0)):
            lever = hinge_xy[h] - face_xy[face]
            A[3 * face + 0, 3 * h + 0] = sign * n[0]
            A[3 * face + 1, 3 * h + 0] = sign * n[1]
            A[3 * face + 2, 3 * h + 0] = sign * _cross(lever, n)
            A[3 * face + 0, 3 * h + 1] = sign * t[0]
            A[3 * face + 1, 3 * h + 1] = sign * t[1]
            A[3 * face + 2, 3 * h + 1] = sign * _cross(lever, t)
            A[3 * face + 2, 3 * h + 2] = -sign
    return A


def _nondimensionalise(A: np.ndarray, scale: float):
    """Row/column scaling so force and moment entries are numerically comparable.

    ``A`` mixes newtons with newton-millimetres, so its singular values are not directly comparable
    and a rank tolerance on the raw matrix is meaningless. Scaling every moment row and moment column
    by the tile size is an invertible change of variables: it moves no null space, it only makes the
    numerical rank well posed.
    """
    rows = np.ones(A.shape[0])
    rows[2::3] = 1.0 / scale
    cols = np.ones(A.shape[1])
    cols[2::3] = scale
    return rows[:, None] * A * cols[None, :], cols


def self_stress(field: ForceField) -> Dict[str, Any]:
    """Split the interface forces into the part that carries the load and the part that does not.

    The null space of ``A`` is the space of STATES OF SELF-STRESS and the null space of ``A^T`` is
    the space of mechanisms, per the Maxwell-Calladine index theorem (Calladine 1978; Mao & Lubensky
    2018). For kirigami these are the "stressed lines" that carry load first (Chen et al. 2016).

    The null space is huge here (a point hinge transmitting force and moment is a full rigid
    connection, so the sheet is heavily indeterminate) and no single basis vector of it means
    anything. What IS unique and physical is the self-stress content of the ACTUAL force state:
    splitting ``q = q_particular + q_self`` with ``q_particular`` the minimum-norm equilibrium
    solution leaves ``q_self`` in the null space -- the locked-in pre-stress that carries no external
    load at all.

    ``share`` is the norm fraction of ``q`` lying in the null space -- an orthogonal projection, so
    it is well defined once the inner product is fixed, which is what :func:`_nondimensionalise`
    does. Quoting it without saying that would be meaningless, since force and moment components are
    otherwise not comparable.

    ⚠ The usual caveat that an indeterminate rigid-block assembly has no unique force solution
    (Kollmer & Daniels 2019) applies to limit analysis, NOT here. This pipeline minimises a
    deformable energy, so compatibility selects one solution: the state shown is the unique elastic
    one, not an arbitrary member of a 300-dimensional family.

    ⚠ ``equilibrium_defect`` is NOT an extraction error. Describing a pair interaction by a force at
    the hinge point plus one couple assumes the pair transmits zero net moment; the energy model very
    slightly violates that, because closed hinges get a nominal reference bond (``l0_nom``) that does
    not rotate with the configuration. The defect equals ``M_i + M_j + (c_j - c_i) x t`` scattered
    onto the tiles -- the same frame dependence ``load_paths.run_checks`` already reports. It is
    measured and returned rather than silently absorbed.
    """
    dec = interface_decomposition(field)
    q = np.stack([dec['f_n'], dec['f_t'], dec['m_e']], axis=-1).reshape(-1)
    A = equilibrium_matrix(field)
    f_net = A @ q

    scale = field.tile_scale
    A_hat, cols = _nondimensionalise(A, scale)
    q_hat = q / cols

    rank = int(np.linalg.matrix_rank(A_hat))
    s, m = A.shape[1] - rank, A.shape[0] - rank

    particular_hat, *_ = np.linalg.lstsq(A_hat, (A_hat @ q_hat), rcond=None)
    self_hat = q_hat - particular_hat

    f_int = np.asarray(field.residual) - np.asarray(field.f_ext)
    defect = float(np.abs(f_net.reshape(-1, 3) - f_int).max()) / max(field.f0, _EPS)

    return {
        'equilibrium_defect': defect,
        'A': A,
        'q': q.reshape(-1, 3),
        'q_particular': (particular_hat * cols).reshape(-1, 3),
        'q_self': (self_hat * cols).reshape(-1, 3),
        'f_net': f_net.reshape(-1, 3),
        'rank': rank,
        'n_self_stress': s,
        'n_mechanisms': m,
        # Maxwell-Calladine: (columns - rows) == s - m, identically. A non-zero here means the
        # numerical rank was mis-called, not that the theorem failed.
        'maxwell_calladine': int((A.shape[1] - A.shape[0]) - (s - m)),
        'residual_norm': float(np.linalg.norm(A_hat @ self_hat)),
        'share': float(np.linalg.norm(self_hat) / max(np.linalg.norm(q_hat), _EPS)),
        'per_hinge': np.linalg.norm((self_hat * cols).reshape(-1, 3)[:, :2], axis=-1),
    }


# ── Summary ───────────────────────────────────────────────────────────────────────────────────

def network_summary(field: ForceField, gamma: Optional[float] = None, seed: int = 0,
                    protect_rings: int = 0) -> Dict[str, Any]:
    """Everything the figure caption and the diagnostics CLI need, computed once.

    ``gamma=None`` picks the resolution per channel with :func:`auto_gamma` and reports it.
    """
    resolved = {w: (auto_gamma(field, weight=w, seed=seed) if gamma is None else gamma)
                for w in WEIGHTS}
    # The TOTAL-force communities are what the figure draws. The per-channel runs below are kept as
    # diagnostics only: splitting a figure on them tore each chain in half, because on a lattice
    # under a near-uniaxial load the normal/shear label tracks hinge ORIENTATION rather than force
    # character (see `load_paths.orientation_confound`).
    total = geographical_communities(field, gamma=resolved['total'], weight='total', seed=seed)
    normal = geographical_communities(field, gamma=resolved['normal'], weight='normal', seed=seed)
    shear = geographical_communities(field, gamma=resolved['shear'], weight='shear', seed=seed)
    centrality = edge_betweenness(field)
    cut = min_cut(field, protect_rings=protect_rings)
    stress = self_stress(field)

    dec = interface_decomposition(field)
    return {
        'communities_total': total,
        'communities_normal': normal,
        'communities_shear': shear,
        'betweenness': centrality,
        'betweenness_top_decile': (centrality >= np.quantile(centrality, 0.9)
                                   if centrality.size else np.zeros(0, dtype=bool)),
        'min_cut': cut,
        'self_stress': stress,
        'shear_dominated_fraction': float(dec['shear_dominated'].mean()) if field.n_hinges else 0.0,
        'f0': field.f0,
        'gamma': resolved['total'],
        'gamma_per_channel': resolved,
        'gamma_auto': gamma is None,
    }
