"""Stage-0 chart ablation: is the closed-state parameterization a real inductive bias?

Compares three charts over the SAME closed RDPQK sheet, same target, same physics:

    A  ``closed_les``      per-cut aspect ratios + ordered boundary sliders
    B  ``direct_vertices`` free panel vertices, nothing enforced
    C  ``direct_vertices`` + a hinge-gap penalty

Everything is compared in one common geometric metric -- the flat-sheet panel-node
coordinate ``y``, whose Euclidean distance is RMS panel-vertex displacement in mm.
Raw-parameter condition numbers are inadmissible across charts: they carry different
units, ``theta -> A theta`` sends ``H -> A^-T H A^-1`` so any of them whitens to 1,
and the closed chart has structurally dead coordinates.

The hinge-gap operator ``B`` here is KINEMATIC -- it measures whether the flat cut
pattern is still one connected sheet. It is not the damage measure, which is physical
and accumulates during deployment.
"""

from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from nff.closed.setup import build_closed_initial_state, init_closed_les_params
from nff.stages.geometry import reconstruct_vertices
from nff.stages.mapping import apply_closed_les_mapping, apply_direct_mapping


# ── the common geometric coordinate ────────────────────────────────────────────────

def geom_coords(state) -> jnp.ndarray:
    """Flat panel-node coordinate ``y`` of a CentroidalState, as a flat (n_faces*max_nodes*2,) vector."""
    return reconstruct_vertices(state.face_centroids, state.centroid_node_vectors).reshape(-1)


def lift_geom(y, n_faces: int, max_nodes: int) -> dict:
    """The canonical section ``psi``: panel nodes -> direct-vertices design parameters.

    Identical to what ``apply_closed_les_mapping`` does with its solved cut vertices, so
    ``L(psi(y))`` is the one reference objective every chart shares.
    """
    nodes = jnp.reshape(y, (n_faces, max_nodes, 2))
    centroids = nodes.mean(axis=1)
    return {'face_centroids': centroids,
            'centroid_node_vectors': nodes - centroids[:, None, :]}


def geom_shape(state) -> tuple:
    """``(n_faces, max_nodes)`` of a CentroidalState."""
    cnv = np.asarray(state.centroid_node_vectors)
    return int(cnv.shape[0]), int(cnv.shape[1])


# ── the seam: arms B and C start where arm A starts ────────────────────────────────

def init_direct_from_closed(config, seed=None):
    """Free-vertex design parameters seeded from the closed chart's own starting geometry.

    Deliberately NOT ``init_direct_vertices_params``, which recenters and rescales the sheet
    to ``target.radius`` and would change its physical size. Pushing the closed init through
    the closed chart instead means every arm starts at bit-identical geometry for a given
    seed, so the arms differ only in what the optimizer may move -- and arms B/C receive a
    valid start they could not have found themselves, which is conservative in their favour.

    Returns:
        (params, state, tessellation) -- params keyed for ``map_type: direct_vertices``.
    """
    cfg = deepcopy(config)
    if seed is not None:
        cfg.topology = {**cfg.topology, 'init_seed': int(seed)}
    params_closed, static_features = init_closed_les_params(cfg)
    state, tessellation = build_closed_initial_state(cfg)
    mapped = apply_closed_les_mapping(state, params_closed, static_features)
    params = {'face_centroids': mapped.face_centroids,
              'centroid_node_vectors': mapped.centroid_node_vectors}
    return params, state, tessellation


def closed_start_geometry(config, seed=None):
    """``(y, params_closed, static_features, state, tessellation)`` at a given seed.

    The shared reference point. Seed ``None`` with ``init_noise: 0`` is the deterministic
    uniform sheet at which every Hessian and transect is evaluated.
    """
    cfg = deepcopy(config)
    if seed is not None:
        cfg.topology = {**cfg.topology, 'init_seed': int(seed)}
    params_closed, static_features = init_closed_les_params(cfg)
    state, tessellation = build_closed_initial_state(cfg)
    mapped = apply_closed_les_mapping(state, params_closed, static_features)
    return geom_coords(mapped), params_closed, static_features, state, tessellation


# ── the chart tangent map ──────────────────────────────────────────────────────────

def chart_jacobian(config, seed=None):
    """``J = dy/dtheta`` of the closed chart -- Stage 0 only, no physics.

    Returns:
        (J, theta_flat, unravel, state) with J of shape (n_faces*max_nodes*2, P).
    """
    _, params_closed, static_features, state, _ = closed_start_geometry(config, seed)
    theta_flat, unravel = ravel_pytree(params_closed)

    def y_of_theta(flat):
        mapped = apply_closed_les_mapping(state, unravel(flat), static_features)
        return geom_coords(mapped)

    return np.asarray(jax.jacfwd(y_of_theta)(theta_flat)), theta_flat, unravel, state


def chart_basis(J, rtol: float = 1e-8):
    """Orthonormal basis of the chart's tangent image, and its numerical rank."""
    U, s, _ = np.linalg.svd(np.asarray(J), full_matrices=False)
    if s.size == 0:
        return U[:, :0], 0, s
    keep = s > rtol * s[0]
    return U[:, keep], int(keep.sum()), s


# ── per-arm validity mechanism ─────────────────────────────────────────────────────

PROJ_ITERS = 100        # altproj residual ~1e-8 (solve_alternating_projections docstring)

# ⚠ The single letters are INDICES, not abbreviations -- in particular arm 'D' has nothing to do
# with the ductile damage measure ``D`` used everywhere else in this project. They are kept only
# because they index the ``arm_idx`` column of saved ensembles. Use ARM_NAME in anything a human
# reads: printouts, figure labels, captions, paper prose.
ARM_NAME = {'A': 'closed chart', 'B': 'naive', 'C': '+penalty',
            'D': '+projection', 'E': '+proj +overlap'}


def arm_validity_cfg(config, arm: str, n_iters: int = PROJ_ITERS):
    """Config with the validity mechanism that arm ``arm`` is entitled to.

    Arms A/B/C are returned unchanged. Arms D/E get Stage-1 alternating projections.

    The closed config ships ``validity_method: none``, which is correct for arm A -- the closed
    chart is valid by construction and Stage 1 is *designed* to be bypassed. It is NOT transferable
    to a free-vertex chart: carrying it over removes the chart AND the projection in one step, and
    then attributes the damage to the chart alone.

    ``n_iters`` is 100 rather than the repo-canonical 20 because 20 leaves a ~2e-3 residual while
    100 reaches ~1e-8, and the baseline should be given every advantage.
    """
    if arm not in ('D', 'E'):
        return config

    import equinox as eqx
    from nff.config.experiment import ValidityConfig
    return eqx.tree_at(
        lambda c: c.validity, config,
        ValidityConfig(weights=dict(config.validity.weights),
                       validity_method='alternating_projection', n_proj_iters=int(n_iters)))


def project_if_needed(state, config):
    """Apply Stage-1 alternating projections when the config asks for them, else pass through.

    The sheet that reaches Stage 2 -- and therefore the sheet that must be audited -- is the
    POST-projection one. Auditing the Stage-0 output instead reports a hinge gap that the physics
    never saw.
    """
    if config.validity.validity_method != 'alternating_projection':
        return state
    from nff.stages.projection import solve_alternating_projections
    return solve_alternating_projections(state, n_iters=config.validity.n_proj_iters)


# ── the global non-overlap penalty (arm E) ─────────────────────────────────────────

def overlap_penalty(nodes, margin: float = 0.0):
    """Differentiable panel-panel interpenetration penalty over a flat sheet.

    Panels are convex quadrilaterals, so for a vertex ``p`` and a panel ``P`` the signed distance
    to each directed edge is positive exactly when ``p`` lies inside; the minimum over edges is
    then the penetration depth. The penalty is ``sum relu(depth - margin)**2`` over every
    (vertex, other-panel) pair.

    This is the constraint no projection can supply: unlike hinge connectivity, which is a linear
    equality with a closed-form projection, non-self-intersection is global and non-convex.

    Vertex containment ALONE is not sufficient and must not be used alone: two convex panels can
    overlap in a "+" with no vertex of either inside the other. Measured -- an optimizer given only
    the containment term drove it to exactly 0.0 while the true overlap ROSE to 4.5%, i.e. it
    escaped through the blind spot. For convex panels, overlapping interiors imply either a
    contained vertex or crossing boundaries, so ``segment_crossing_penalty`` completes the cover and
    is added here. See ``penalty_detects_overlap`` for the empirical equivalence check.

    Args:
        nodes:  (n_faces, max_nodes, 2) flat panel vertices.
        margin: depth tolerated before the penalty engages, in pipeline length units.

    Returns:
        scalar penalty, in squared pipeline length units.
    """
    n_faces, max_nodes = nodes.shape[0], nodes.shape[1]
    a = nodes                                          # edge start   (F, K, 2)
    e = jnp.roll(nodes, -1, axis=1) - a                # edge vector  (F, K, 2)

    # Orientation-robust: a clockwise panel flips every cross product, so normalise by the sign of
    # its own signed area. Inverted panels are penalised separately by `flat_validity`.
    sgn = jnp.sign(0.5 * jnp.sum(a[:, :, 0] * jnp.roll(a[:, :, 1], -1, axis=1)
                                 - jnp.roll(a[:, :, 0], -1, axis=1) * a[:, :, 1], axis=1))
    sgn = jnp.where(sgn == 0, 1.0, sgn)                # (F,)

    pts = nodes.reshape(-1, 2)                         # (F*K, 2)
    d = pts[None, None, :, :] - a[:, :, None, :]       # (F, K, F*K, 2)
    cross = e[:, :, None, 0] * d[..., 1] - e[:, :, None, 1] * d[..., 0]
    sdist = cross * sgn[:, None, None] / (jnp.linalg.norm(e, axis=-1)[:, :, None] + 1e-12)

    depth = jnp.min(sdist, axis=1)                     # (F, F*K) -- >0 means strictly inside
    own = (jnp.arange(pts.shape[0])[None, :] // max_nodes) == jnp.arange(n_faces)[:, None]
    contained = jnp.sum(jnp.where(own, 0.0, jax.nn.relu(depth - margin) ** 2))
    return contained + segment_crossing_penalty(nodes)


def segment_crossing_penalty(nodes):
    """Penalty on panel EDGES that properly cross, in squared pipeline length units.

    Closes the blind spot in vertex containment: two convex panels overlapping in a "+" have no
    vertex of either inside the other, but their boundaries must cross.

    Segments ``p`` and ``q`` cross properly when the endpoints of each straddle the other's line,
    i.e. ``d1*d2 < 0`` and ``d3*d4 < 0`` for the four point-to-line signed distances. The two
    straddle amounts are combined harmonically so the term is zero unless BOTH are positive and
    keeps length-squared units, matching the containment term it is added to.
    """
    n_faces, max_nodes = nodes.shape[0], nodes.shape[1]
    a = nodes.reshape(-1, 2)                                   # segment starts (F*K, 2)
    b = jnp.roll(nodes, -1, axis=1).reshape(-1, 2)             # segment ends
    seg_face = jnp.arange(a.shape[0]) // max_nodes

    def straddle(p0, p1, q0, q1):
        """Signed distances of q0, q1 from the line through p0, p1, times the relu of their product."""
        e = p1 - p0
        n = jnp.stack([-e[..., 1], e[..., 0]], axis=-1)        # left normal
        n = n / (jnp.linalg.norm(n, axis=-1, keepdims=True) + 1e-12)
        d0 = jnp.sum(n * (q0 - p0), axis=-1)
        d1 = jnp.sum(n * (q1 - p0), axis=-1)
        return jax.nn.relu(-d0 * d1)                           # >0 exactly when q straddles line p

    A0, A1 = a[:, None, :], b[:, None, :]                      # (S,1,2)
    B0, B1 = a[None, :, :], b[None, :, :]                      # (1,S,2)
    s_pq = straddle(A0, A1, B0, B1)
    s_qp = straddle(B0, B1, A0, A1)
    cross = s_pq * s_qp / (s_pq + s_qp + 1e-30)                # zero unless BOTH straddle

    # Only distinct panels: edges of one panel meet at its own corners by construction, and two
    # hinge-adjacent panels touch at a shared corner without properly crossing (straddle = 0 there).
    same = seg_face[:, None] == seg_face[None, :]
    return 0.5 * jnp.sum(jnp.where(same, 0.0, cross))          # 0.5: each pair counted twice


def penalty_detects_overlap(nodes_batch, state, tol: float = 1e-12) -> dict:
    """Agreement between ``overlap_penalty`` and the shapely audit over a batch of sheets.

    The penalty is a hand-built approximation of a global non-convex constraint, so its adequacy
    is an empirical question, not an assumption. This reports the confusion counts -- in particular
    ``missed``, sheets shapely calls overlapping while the penalty reads zero, which are exactly the
    designs an optimizer can escape through.
    """
    n_faces, max_nodes = geom_shape(state)
    missed = spurious = both = neither = 0
    for y in nodes_batch:
        ov = flat_validity(np.asarray(y).reshape(-1), state)['overlap_fraction'] > 1e-9
        pen = float(overlap_penalty(jnp.asarray(y).reshape(n_faces, max_nodes, 2))) > tol
        if ov and not pen:
            missed += 1
        elif pen and not ov:
            spurious += 1
        elif ov:
            both += 1
        else:
            neither += 1
    return {'missed': missed, 'spurious': spurious, 'both': both, 'neither': neither,
            'n': len(nodes_batch)}


def tiling_penalty(nodes, stock_lo, stock_hi):
    """Differentiable penalty on the flat panels failing to partition the stock sheet.

    A cut pattern is a PARTITION of the stock: the panels are the pieces obtained by cutting a
    rectangle, so their total area equals the stock area and no vertex lies outside it. Two terms,
    both smooth:

      area   ``(sum_f |A_f| - A_stock)^2 / A_stock^2`` -- material is neither created nor lost;
      escape ``sum relu(outside)^2``                   -- no vertex leaves the rectangle.

    Together they are necessary for a partition and, with the non-overlap term, sufficient for the
    panels to cover the stock exactly. This is the constraint a free-vertex chart has no projection
    for: unlike hinge closure, which is a linear equality, "the pieces tile the sheet" is a global
    set equality with no closed-form projection operator.
    """
    a = nodes
    area = 0.5 * jnp.abs(jnp.sum(a[:, :, 0] * jnp.roll(a[:, :, 1], -1, axis=1)
                                 - jnp.roll(a[:, :, 0], -1, axis=1) * a[:, :, 1], axis=1))
    stock_area = (stock_hi[0] - stock_lo[0]) * (stock_hi[1] - stock_lo[1])
    e_area = ((jnp.sum(area) - stock_area) / stock_area) ** 2
    out = (jax.nn.relu(stock_lo[None, None, :] - a) + jax.nn.relu(a - stock_hi[None, None, :]))
    e_escape = jnp.sum(out ** 2) / stock_area
    return e_area + e_escape


def arm_extra_loss(config, arm: str, state, w_overlap: float = 0.0, w_tiling: float = 0.0):
    """Extra loss term for an arm, or ``None`` -- the hook consumed by ``create_train_step``.

    Only arm E defines one. It penalises the POST-projection sheet, i.e. exactly the geometry the
    validity audit scores, rather than the Stage-0 output the projection is about to move.

    ``w_tiling`` engages the partition constraint. Without it the baseline is being scored on
    something it was never asked to satisfy, which is not a comparison; with it, the baseline is
    given a mechanism for every constraint the closed chart satisfies by construction, and the
    question becomes whether it can attain them numerically while matching the target.
    """
    if arm != 'E' or (w_overlap == 0.0 and w_tiling == 0.0):
        return None

    cfg = arm_validity_cfg(config, arm)
    n_faces, max_nodes = geom_shape(state)
    y0 = np.asarray(geom_coords(state)).reshape(n_faces, max_nodes, 2)
    stock_lo = jnp.asarray(y0.reshape(-1, 2).min(axis=0))
    stock_hi = jnp.asarray(y0.reshape(-1, 2).max(axis=0))

    def extra_loss_fn(params):
        mapped = project_if_needed(apply_direct_mapping(state, params), cfg)
        nodes = reconstruct_vertices(mapped.face_centroids, mapped.centroid_node_vectors)
        return (w_overlap * overlap_penalty(nodes)
                + w_tiling * tiling_penalty(nodes, stock_lo, stock_hi))

    return extra_loss_fn


def r_interval(params) -> tuple:
    """``(r_min, r_max)`` of a closed-chart design -- the containment check on the admissible set.

    The chart's validity theorem lives on the open interval ``r in (0, 1)``; brute-force sampling
    that drives ``r`` to ~1e-3 probes a degenerate limit no optimizer visits. Logging the interval
    a training run actually occupies turns "the chart is exact" into a statement with a measured
    domain attached.
    """
    z = params.get('z') if isinstance(params, dict) else None
    if z is None:
        return (float('nan'), float('nan'))
    r = jax.nn.sigmoid(jnp.asarray(z))
    return (float(jnp.min(r)), float(jnp.max(r)))


# ── the hinge-gap operator (kinematic -- NOT damage) ───────────────────────────────

def hinge_gap_operator(state):
    """``B``: flat panel nodes -> the gap vector at every hinge.

    Row block ``h`` returns ``p1 - p2``, the two panel corners bonded at hinge ``h``. It is
    identically zero exactly when the cut pattern is still one connected sheet, and it is the
    operand of ``hinge_connectivity`` and of the ``hinge_gap`` loss term.

    Returns:
        (2*n_hinges, n_faces*max_nodes*2) dense numpy array.
    """
    n_faces, max_nodes = geom_shape(state)
    pairs = np.asarray(state.hinge_node_pairs)
    n_hinges = pairs.shape[0]
    B = np.zeros((2 * n_hinges, n_faces * max_nodes * 2))
    for h in range(n_hinges):
        (fi, lj), (fk, ll) = pairs[h, 0], pairs[h, 1]
        for d in range(2):
            B[2 * h + d, (int(fi) * max_nodes + int(lj)) * 2 + d] += 1.0
            B[2 * h + d, (int(fk) * max_nodes + int(ll)) * 2 + d] -= 1.0
    return B


# ── analytic dimension counting, for the scaling panel ─────────────────────────────

def chart_dims(M: int, N: int) -> dict:
    """Closed-form dimensions of the ablation on an ``M x N`` sheet.

    ``k_A`` counts the interior cuts that actually consume an aspect ratio -- boundary cuts
    are skipped by the LES assembly -- plus the boundary sliders, each edge's logits being
    defined only up to a shift.
    """
    n_faces = M * N
    n_geom = 8 * M * N                                  # 4 corners x 2 coordinates per panel
    n_hinge = 2 * M * N - M - N
    k_A = (M - 1) * (N - 1) + 2 * (M - 1) + 2 * (N - 1)
    return {'M': M, 'N': N, 'n_faces': n_faces, 'n_geom': n_geom, 'n_hinge': n_hinge,
            'k_A': k_A, 'compression': n_geom / k_A,
            # The fraction of a random direction that tears is the ORTHOGONAL PROJECTION onto
            # row(B), i.e. rank(B)/n_geom. Not E[||Bd||^2], which is twice as large because
            # each row e_a - e_b has norm sqrt(2) rather than 1.
            'hinge_gap_fraction': 2 * n_hinge / n_geom,
            'expected_BtB_ratio': 4 * n_hinge / n_geom}


# ── flat-sheet validity ────────────────────────────────────────────────────────────

def flat_validity(y, state, B=None) -> dict:
    """Geometric audit of a flat sheet: is it still a valid, non-overlapping cut pattern?

    Mirrors the post hoc audit the closed driver runs (``run_closed._overlap``) but takes the
    panel nodes directly, so it applies to any chart. Pass ``B`` to avoid rebuilding it.
    """
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    n_faces, max_nodes = geom_shape(state)
    nodes = np.asarray(y).reshape(n_faces, max_nodes, 2)
    if not np.all(np.isfinite(nodes)):
        return {'finite': False, 'n_twisted': n_faces, 'overlap_fraction': float('nan'),
                'min_signed_area': float('nan'), 'max_hinge_gap': float('nan'), 'valid': False}

    signed = 0.5 * np.sum(nodes[:, :, 0] * np.roll(nodes[:, :, 1], -1, axis=1)
                          - np.roll(nodes[:, :, 0], -1, axis=1) * nodes[:, :, 1], axis=1)

    polys = [Polygon(nodes[f]) for f in range(n_faces)]
    n_twisted = sum(0 if p.is_valid else 1 for p in polys)
    polys = [p if p.is_valid else p.buffer(0) for p in polys]
    total = sum(p.area for p in polys)
    overlap = (total - unary_union(polys).area) / max(total, 1e-12)

    if B is None:
        B = hinge_gap_operator(state)
    gaps = np.abs(np.asarray(B) @ np.asarray(y)).reshape(-1, 2) if B.shape[0] else np.zeros((0, 2))
    max_gap = float(np.linalg.norm(gaps, axis=1).max()) if gaps.size else 0.0

    # The three failure modes are reported SEPARATELY -- they have different scales, and
    # collapsing them into one flag hides which one the chart actually removes.
    return {'finite': True,
            'connected': bool(max_gap < 1e-9),          # still one sheet
            'non_overlapping': bool(overlap < 1e-6),    # panels do not interpenetrate
            'non_inverted': bool(signed.min() > 0 and n_twisted == 0),
            'n_twisted': int(n_twisted), 'overlap_fraction': float(overlap),
            'min_signed_area': float(signed.min()), 'max_hinge_gap': max_gap,
            'valid': bool(max_gap < 1e-9 and overlap < 1e-6
                          and signed.min() > 0 and n_twisted == 0)}


# ── the shared objective ───────────────────────────────────────────────────────────

def fixed_target(config):
    """The target every arm is scored against -- read from the config, not fitted.

    Two shapes are supported, each requiring its size to be stated EXPLICITLY in ``topology``: a
    rectangle via ``target_half_w`` / ``target_half_h``, or a circle via ``target_r``. The
    explicitness is the point -- an auto-fitted target would depend on an untrained deploy and so on
    the chart, which would confound the one variable the comparison is meant to isolate.
    """
    from nff.config.experiment import TargetConfig
    from nff.config.targets import get_target_points

    topo = config.topology
    cen = (float(topo.get('target_cx', config.target.center[0])),
           float(topo.get('target_cy', config.target.center[1])))

    if config.target.type == 'circle' and 'target_r' in topo:
        r = float(topo['target_r'])
        cloud = get_target_points({'type': 'circle', 'center': np.array(cen), 'radius': r},
                                  n_points=400)
        return TargetConfig(type='circle', center=cen, radius=r), jnp.asarray(cloud, dtype=float)

    if config.target.type == 'rectangle' and 'target_half_w' in topo:
        hw, hh = float(topo['target_half_w']), float(topo['target_half_h'])
        rot = float(topo.get('target_rot_deg', 0.0) or 0.0)
        cloud = get_target_points({'type': 'rectangle', 'center': np.array(cen),
                                   'radius': max(hw, hh), 'half_w': hw, 'half_h': hh,
                                   'rot_deg': rot},
                                  n_points=400)
        target = TargetConfig(type='circle', center=cen, radius=float(max(hw, hh)))
        return target, jnp.asarray(cloud, dtype=float)

    raise NotImplementedError(
        "a FIXED target is required: rectangle via target_half_w/target_half_h, or circle via "
        f"target_r; got target.type={config.target.type!r} with neither present. An auto-fitted "
        "target depends on the untrained deploy and therefore on the chart.")


def arm_training_cfg(config, hinge_gap: float = 0.0):
    """Training config with the chart-dependent regularizer removed.

    ``_term_regularization`` is ``sum(theta**2)`` over the RAW parameter tree, so its meaning
    differs between charts -- against free vertices it is a shrink-to-origin with no counterpart
    in the closed chart. Leaving it on would invalidate the ablation.
    """
    import equinox as eqx
    from nff.config.experiment import LossWeights

    # Copy EVERY field, then override the two this function sets. Enumerating the fields by hand
    # made a new weight silently vanish here -- it was reset to its default while the config still
    # read as if it were active, which is the same class of failure as a renamed loss term.
    w = config.training.loss_weights
    fields = {name: getattr(w, name) for name in w.__annotations__}
    new_w = LossWeights(**{**fields, 'regularization': 0.0, 'hinge_gap': float(hinge_gap)})
    return eqx.tree_at(lambda t: t.loss_weights, config.training, new_w)


def build_geom_loss_fn(config, hinge_gap: float = 0.0):
    """``L_geom(y)`` -- the one reference objective, over flat panel nodes in R^{8MN}.

    Every arm is compared through this function, so all curvature is measured in the same
    Euclidean space and the same units. The ROM hinge model is used throughout.

    Returns:
        (loss_fn, y0, state, tessellation)
    """
    from nff.training.loss import compute_end_to_end_loss

    y0, _, _, state, tess = closed_start_geometry(config)
    n_faces, max_nodes = geom_shape(state)
    target, cloud = fixed_target(config)
    training_cfg = arm_training_cfg(config, hinge_gap=hinge_gap)
    load_specs = config.topology.get('loads', [])

    def loss_fn(y):
        return compute_end_to_end_loss(
            lift_geom(y, n_faces, max_nodes), state, target, config.validity, config.physics,
            training_cfg, map_type='direct_vertices', static_features=None,
            load_specs=load_specs, target_cloud=cloud)[0]

    return loss_fn, y0, state, tess


def build_arm_loss_fn(config, arm: str, hinge_gap: float = 0.0, seed=None):
    """``loss_fn(params)`` in an arm's OWN parameters, plus that arm's starting point.

    ``arm`` is 'A' (closed_les), 'B' (free vertices) or 'C' (free vertices + hinge-gap penalty).
    Arms B and C start at the geometry arm A starts from, so only the chart differs.
    """
    from nff.training.loss import compute_end_to_end_loss

    target, cloud = fixed_target(config)
    load_specs = config.topology.get('loads', [])

    if arm == 'A':
        _, params0, static_features, state, tess = closed_start_geometry(config, seed)
        map_type, sf = 'closed_les', static_features
        training_cfg = arm_training_cfg(config, hinge_gap=0.0)
    elif arm in ('B', 'C'):
        params0, state, tess = init_direct_from_closed(config, seed)
        map_type, sf = 'direct_vertices', None
        training_cfg = arm_training_cfg(config, hinge_gap=hinge_gap if arm == 'C' else 0.0)
    else:
        raise ValueError(f"unknown arm {arm!r}")

    def loss_fn(params):
        return compute_end_to_end_loss(
            params, state, target, config.validity, config.physics, training_cfg,
            map_type=map_type, static_features=sf, load_specs=load_specs,
            target_cloud=cloud)[0]

    return loss_fn, params0, state, tess


def deploy_validity(config, params, state, map_type, static_features=None) -> dict:
    """Deploy a design through Stage-2 physics and audit the result.

    Stage 2 does not reject a separated sheet: ``ReferenceGeometry`` simply takes the flat hinge
    gap as the ligament rest length, so a torn design deploys as disconnected panels without
    complaint. The audit therefore has to be external.
    """
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    from nff.closed.deploy import _boundary_cloud
    from nff.stages.geometry import deformed_vertices
    from nff.stages.pipeline import forward_pipeline

    target, _ = fixed_target(config)
    try:
        res = forward_pipeline(state, target, config.validity, config.physics,
                               map_type=map_type, map_params=params,
                               static_features=static_features,
                               load_specs=config.topology.get('loads', []))
        disp = res['solution'].fields[-1]
        nodes = np.asarray(deformed_vertices(res['valid_state'], disp))
        # Returned alongside the nodes so a caller needing both the deployed panels and the
        # deployed outline pays for ONE forward solve rather than two.
        boundary = np.asarray(_boundary_cloud(res['valid_state'], disp))
    except Exception:
        return {'finite': False, 'non_overlapping': False, 'valid': False,
                'overlap_fraction': float('nan')}

    if not np.all(np.isfinite(nodes)):
        return {'finite': False, 'non_overlapping': False, 'valid': False,
                'overlap_fraction': float('nan')}

    polys = [Polygon(nodes[f]) for f in range(nodes.shape[0])]
    polys = [p if p.is_valid else p.buffer(0) for p in polys]
    total = sum(p.area for p in polys)
    overlap = (total - unary_union(polys).area) / max(total, 1e-12)
    return {'finite': True, 'non_overlapping': bool(overlap < 1e-3), 'nodes': nodes,
            'boundary': boundary,
            'overlap_fraction': float(overlap), 'valid': bool(overlap < 1e-3)}


def deployed_boundary(config, params, state, map_type, static_features=None):
    """Deployed boundary point cloud, for the multi-start dispersion panel."""
    from nff.closed.deploy import _boundary_cloud
    from nff.stages.pipeline import forward_pipeline

    target, _ = fixed_target(config)
    res = forward_pipeline(state, target, config.validity, config.physics,
                           map_type=map_type, map_params=params,
                           static_features=static_features,
                           load_specs=config.topology.get('loads', []))
    return np.asarray(_boundary_cloud(res['valid_state'], res['solution'].fields[-1]))


# ── geometric step size, the shared x-axis ─────────────────────────────────────────

def rms_displacement(y, y_ref, n_nodes: int, length_scale: float = 1.0) -> float:
    """RMS panel-vertex displacement in mm -- the unit every arm is compared in."""
    d = np.asarray(y) - np.asarray(y_ref)
    return float(np.linalg.norm(d) / np.sqrt(n_nodes) * length_scale)
