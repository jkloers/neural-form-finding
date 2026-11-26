"""Graph-level load-path analysis: chains, bottlenecks, and self-stress.

The physics assertions here matter more than the graph ones. A force-chain picture is only worth
drawing if the equilibrium matrix behind it reproduces the extracted forces and the Maxwell-Calladine
index closes, so those are checked first and directly.
"""

import networkx as nx
import numpy as np
import pytest

from nff.analysis import force_network as fnet
from nff.analysis.load_paths import ForceField, transmitted_moment


# ── Fixtures ──────────────────────────────────────────────────────────────────────────────────

def _field(face_xy, face_pairs, t, m_i=None, m_j=None, hinge_xy=None, f_ext=None, clamped=()):
    """A minimal ForceField carrying only what the graph layer reads."""
    face_xy = np.asarray(face_xy, dtype=float)
    face_pairs = np.asarray(face_pairs, dtype=int)
    t = np.asarray(t, dtype=float).reshape(-1, 2)
    n_faces, n_hinges = face_xy.shape[0], face_pairs.shape[0]

    if hinge_xy is None:
        hinge_xy = 0.5 * (face_xy[face_pairs[:, 0]] + face_xy[face_pairs[:, 1]])
    hinge_xy = np.asarray(hinge_xy, dtype=float)

    span = face_xy[face_pairs[:, 1]] - face_xy[face_pairs[:, 0]]
    axis = span / np.maximum(np.linalg.norm(span, axis=-1, keepdims=True), 1e-30)
    axial = -np.sum(t * axis, axis=-1)
    transverse = t[:, 0] * axis[:, 1] - t[:, 1] * axis[:, 0]

    m_lig = np.stack([np.zeros(n_hinges) if m_i is None else np.asarray(m_i, dtype=float),
                      np.zeros(n_hinges) if m_j is None else np.asarray(m_j, dtype=float)], axis=-1)

    free_mask = np.ones((n_faces, 3), dtype=bool)
    for f in clamped:
        free_mask[f] = False
    external = np.zeros((n_faces, 3))
    if f_ext is not None:
        for face, vec in f_ext.items():
            external[face, :len(vec)] = vec

    # A unit square around each centroid: only the area (=1) is ever read, for the moment scale.
    corners = np.array([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]])
    return ForceField(
        face_xy=face_xy, face_nodes=face_xy[:, None, :] + corners[None], face_areas=np.ones(n_faces),
        hinge_xy=hinge_xy, face_pairs=face_pairs, t_lig=t, m_lig=m_lig,
        t_con=np.zeros_like(t), m_con=np.zeros_like(m_lig), hinge_axis=np.zeros_like(t),
        axial=axial, transverse=transverse, energy=np.zeros(n_hinges),
        sigma=np.zeros((n_faces, 2, 2)), f_ext=external,
        residual=np.zeros((n_faces, 3)), free_mask=free_mask,
        boundary_xy=np.zeros((0, 2)), checks={})


def _chain(n=8, spacing=1.0):
    """A straight chain of tiles, each pulling on the next."""
    face_xy = np.stack([np.arange(n) * spacing, np.zeros(n)], axis=-1)
    pairs = np.stack([np.arange(n - 1), np.arange(1, n)], axis=-1)
    return face_xy, pairs


# ── §0  The transmitted couple ────────────────────────────────────────────────────────────────

def test_transmitted_moment_is_the_couple_not_the_centroid_moment():
    """A pure force through the hinge, offset from both centroids, transmits NO couple.

    The generalized moment on each tile is non-zero (the force has a lever arm about each centroid),
    so a naive read of `m_lig` would report a couple that does not exist.
    """
    face_xy = np.array([[0.0, 0.0], [2.0, 0.0]])
    pairs = np.array([[0, 1]])
    hinge = np.array([[1.0, 1.0]])                       # offset perpendicular to the branch
    t = np.array([[3.0, 0.0]])                           # force on face j

    lever_i, lever_j = hinge[0] - face_xy[0], hinge[0] - face_xy[1]
    m_i = lever_i[0] * (-t[0, 1]) - lever_i[1] * (-t[0, 0])
    m_j = lever_j[0] * t[0, 1] - lever_j[1] * t[0, 0]

    field = _field(face_xy, pairs, t, m_i=[m_i], m_j=[m_j], hinge_xy=hinge)
    assert abs(field.m_lig[0, 1]) > 1e-9                 # the centroid moment IS non-zero
    assert transmitted_moment(field, side=1)[0] == pytest.approx(0.0, abs=1e-12)
    assert transmitted_moment(field, side=0)[0] == pytest.approx(0.0, abs=1e-12)


def test_transmitted_moment_balances_between_the_two_tiles():
    """A genuine couple must read as +m from one tile and -m from the other."""
    face_xy = np.array([[0.0, 0.0], [1.0, 0.5]])
    pairs = np.array([[0, 1]])
    hinge = np.array([[0.4, 0.1]])
    t = np.array([[1.7, -0.9]])
    couple = 2.5

    lever_i, lever_j = hinge[0] - face_xy[0], hinge[0] - face_xy[1]
    m_i = (lever_i[0] * (-t[0, 1]) - lever_i[1] * (-t[0, 0])) - couple
    m_j = (lever_j[0] * t[0, 1] - lever_j[1] * t[0, 0]) + couple

    field = _field(face_xy, pairs, t, m_i=[m_i], m_j=[m_j], hinge_xy=hinge)
    assert transmitted_moment(field, side=1)[0] == pytest.approx(couple, rel=1e-12)
    assert transmitted_moment(field, side=0)[0] == pytest.approx(-couple, rel=1e-12)


# ── §7  Equilibrium matrix and self-stress ────────────────────────────────────────────────────

def test_equilibrium_matrix_reproduces_the_internal_forces():
    """A @ q must regenerate the generalized force on every tile -- the sign-convention test."""
    face_xy, pairs = _chain(6)
    rng = np.random.default_rng(0)
    t = rng.normal(size=(pairs.shape[0], 2))
    couple = rng.normal(size=pairs.shape[0])

    lever_i = 0.5 * (face_xy[pairs[:, 1]] - face_xy[pairs[:, 0]])
    m_i = (lever_i[:, 0] * (-t[:, 1]) - lever_i[:, 1] * (-t[:, 0])) - couple
    m_j = (-lever_i[:, 0] * t[:, 1] + lever_i[:, 1] * t[:, 0]) + couple
    field = _field(face_xy, pairs, t, m_i=m_i, m_j=m_j)

    expected = np.zeros((field.n_faces, 3))
    np.add.at(expected, pairs[:, 0], np.column_stack([-t, m_i]))
    np.add.at(expected, pairs[:, 1], np.column_stack([t, m_j]))

    result = fnet.self_stress(field)
    assert np.abs(result['f_net'] - expected).max() < 1e-10


def test_maxwell_calladine_index_closes():
    """(columns - rows) == s - m, identically (Calladine 1978)."""
    face_xy, pairs = _chain(7)
    field = _field(face_xy, pairs, np.ones((pairs.shape[0], 2)))
    result = fnet.self_stress(field)
    assert result['maxwell_calladine'] == 0
    assert (3 * field.n_hinges - 3 * field.n_faces
            == result['n_self_stress'] - result['n_mechanisms'])


def test_free_chain_has_exactly_three_rigid_body_mechanisms():
    face_xy, pairs = _chain(7)
    field = _field(face_xy, pairs, np.ones((pairs.shape[0], 2)))
    assert fnet.self_stress(field)['n_mechanisms'] == 3


def test_self_stress_decomposition_is_exact_and_load_free():
    """q splits into a load-carrying part and a null-space part, with nothing left over."""
    face_xy, pairs = _chain(6)
    rng = np.random.default_rng(1)
    field = _field(face_xy, pairs, rng.normal(size=(pairs.shape[0], 2)),
                   m_i=rng.normal(size=pairs.shape[0]), m_j=rng.normal(size=pairs.shape[0]))
    result = fnet.self_stress(field)

    assert np.abs(result['q_particular'] + result['q_self'] - result['q']).max() < 1e-10
    assert result['residual_norm'] < 1e-9                # A @ q_self == 0: carries no external load


def test_statically_determinate_chain_has_no_self_stress():
    """A chain of tiles joined end to end is determinate, so nothing can be locked in."""
    face_xy, pairs = _chain(5)
    rng = np.random.default_rng(2)
    field = _field(face_xy, pairs, rng.normal(size=(pairs.shape[0], 2)))
    result = fnet.self_stress(field)
    assert result['n_self_stress'] == 0
    assert result['share'] == pytest.approx(0.0, abs=1e-9)


# ── §2.1  Communities under the geographical null model ───────────────────────────────────────

def test_geographical_null_model_finds_chains_where_configuration_finds_blobs():
    """The null model is the point: it must split a strong chain from the weak mesh around it.

    Two parallel chains of strong contacts embedded in a uniformly weak background. The geographical
    model (uniform rho) keeps each strong chain together and cuts the weak links; this asserts the
    strong chains are recovered, which a strength-proportional null model blurs.
    """
    n = 6
    face_xy, strong_a = _chain(n)
    face_xy = np.vstack([face_xy, face_xy + np.array([0.0, 3.0])])
    strong_b = strong_a + n
    weak = np.stack([np.arange(n), np.arange(n) + n], axis=-1)      # rungs between the chains

    pairs = np.vstack([strong_a, strong_b, weak])
    magnitude = np.concatenate([np.full(len(strong_a), 10.0), np.full(len(strong_b), 10.0),
                                np.full(len(weak), 0.2)])
    axis = face_xy[pairs[:, 1]] - face_xy[pairs[:, 0]]
    axis /= np.linalg.norm(axis, axis=-1, keepdims=True)
    field = _field(face_xy, pairs, -magnitude[:, None] * axis)

    labels = fnet.geographical_communities(field, gamma=1.0)['labels']
    assert len(set(labels[:n])) == 1                     # chain A is one community
    assert len(set(labels[n:])) == 1                     # chain B is one community
    assert labels[0] != labels[n]                        # and they are NOT merged into one blob


def test_communities_are_deterministic_under_a_fixed_seed():
    face_xy, pairs = _chain(10)
    rng = np.random.default_rng(3)
    field = _field(face_xy, pairs, rng.normal(size=(pairs.shape[0], 2)))
    first = fnet.geographical_communities(field, seed=7)['labels']
    second = fnet.geographical_communities(field, seed=7)['labels']
    assert np.array_equal(first, second)


def test_higher_gamma_fragments_more():
    """Resolution behaves monotonically, which is what makes a gamma sweep meaningful."""
    face_xy, pairs = _chain(12)
    axis = face_xy[pairs[:, 1]] - face_xy[pairs[:, 0]]
    axis /= np.linalg.norm(axis, axis=-1, keepdims=True)
    magnitude = np.linspace(1.0, 10.0, pairs.shape[0])
    field = _field(face_xy, pairs, -magnitude[:, None] * axis)

    coarse = fnet.geographical_communities(field, gamma=0.2)['n_communities']
    fine = fnet.geographical_communities(field, gamma=3.0)['n_communities']
    assert fine >= coarse


# ── §2.2  Betweenness ─────────────────────────────────────────────────────────────────────────

def test_betweenness_ranks_the_strong_route_above_the_weak_one():
    """Guards the distance inversion: networkx MINIMISES weight, so force must be inverted first.

    Two parallel routes between the same endpoints, one strong and one weak. The strong route must
    score higher. Passing force straight in as `weight` would rank the weak route above it.
    """
    face_xy = np.array([[0., 0.], [1., 1.], [2., 1.], [3., 0.], [1., -1.], [2., -1.]])
    pairs = np.array([[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 3]])
    magnitude = np.array([10.0, 10.0, 10.0, 0.1, 0.1, 0.1])       # top route strong, bottom weak
    axis = face_xy[pairs[:, 1]] - face_xy[pairs[:, 0]]
    axis /= np.linalg.norm(axis, axis=-1, keepdims=True)
    field = _field(face_xy, pairs, -magnitude[:, None] * axis)

    centrality = fnet.edge_betweenness(field)
    assert centrality[:3].min() > centrality[3:].max()


# ── §2.3  Bottleneck ──────────────────────────────────────────────────────────────────────────

def test_min_cut_finds_the_single_bottleneck():
    """Two wide bands joined by one narrow link: the cut must be exactly that link.

    Both loaded tiles and both supports sit on their own side of the throat, so severing the throat
    (0.5) genuinely beats severing either band (10).
    """
    face_xy = np.array([[0., 0.], [0., 1.], [1., .5], [2., .5], [3., 0.], [3., 1.]])
    pairs = np.array([[0, 2], [1, 2], [2, 3], [3, 4], [3, 5]])
    magnitude = np.array([5.0, 5.0, 0.5, 5.0, 5.0])                # edge 2 is the throat
    axis = face_xy[pairs[:, 1]] - face_xy[pairs[:, 0]]
    axis /= np.linalg.norm(axis, axis=-1, keepdims=True)
    field = _field(face_xy, pairs, -magnitude[:, None] * axis,
                   f_ext={0: (1.0, 0.0), 1: (1.0, 0.0)}, clamped=(4, 5))

    cut = fnet.min_cut(field)
    assert cut['cut_hinges'].tolist() == [2]
    assert cut['cut_value'] == pytest.approx(np.abs(field.t_total[2] @ fnet.load_axis(field)),
                                             rel=1e-3)


def test_min_cut_returns_empty_without_supports_or_loads():
    face_xy, pairs = _chain(4)
    field = _field(face_xy, pairs, np.ones((pairs.shape[0], 2)))
    assert np.isnan(fnet.min_cut(field)['cut_value'])


# ── §1.3  Chain spines ────────────────────────────────────────────────────────────────────────

def test_chain_spine_spans_the_whole_chain():
    """A spine must run end to end, otherwise the eye cannot follow the load through it."""
    face_xy, pairs = _chain(8)
    axis = np.tile([[1.0, 0.0]], (pairs.shape[0], 1))
    field = _field(face_xy, pairs, -5.0 * axis)

    spines = fnet.chain_spines(field, np.zeros(field.n_faces, dtype=int))
    assert len(spines) == 1
    assert spines[0][:, 0].min() < 0.5 and spines[0][:, 0].max() > 6.5


def test_graph_sums_parallel_hinges_but_keeps_them_addressable():
    face_xy = np.array([[0., 0.], [1., 0.]])
    pairs = np.array([[0, 1], [0, 1]])
    field = _field(face_xy, pairs, np.array([[1.0, 0.0], [2.0, 0.0]]),
                   hinge_xy=np.array([[.5, .4], [.5, -.4]]))
    graph = fnet.build_graph(field)
    assert graph.number_of_edges() == 1
    assert sorted(graph[0][1]['hinges']) == [0, 1]
    assert graph[0][1]['w'] == pytest.approx(3.0)


def test_edge_weights_separate_normal_from_shear():
    """The two channels must not be merged into one scalar (Zhang et al. 2010)."""
    face_xy = np.array([[0., 0.], [1., 0.]])
    pairs = np.array([[0, 1]])
    field = _field(face_xy, pairs, np.array([[0.0, 4.0]]))          # pure shear across the branch
    assert fnet.edge_weights(field, 'normal')[0] == pytest.approx(0.0, abs=1e-12)
    assert fnet.edge_weights(field, 'shear')[0] == pytest.approx(4.0)


def test_unknown_weight_channel_is_rejected():
    face_xy, pairs = _chain(3)
    field = _field(face_xy, pairs, np.ones((pairs.shape[0], 2)))
    with pytest.raises(ValueError, match='weight must be one of'):
        fnet.edge_weights(field, 'bending')
