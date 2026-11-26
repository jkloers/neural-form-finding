"""End-to-end design loss for Neural Form-Finding.

The objective is a sum of independently-weighted TERMS, each a pure function of a shared
:class:`LossContext`. ``TERMS`` below is the registry: it is the whole objective, in evaluation
order, and a term contributes nothing unless its weight is non-zero.

Metric convention, uniform across terms:
    ``comp_<name>``  the WEIGHTED contribution (what the loss actually paid)
    ``<name>_raw``   the unweighted quantity (comparable across weight settings)
plus whatever extra diagnostics a term chooses to report.
"""

from typing import Any, NamedTuple, Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from nff.config.targets import get_target_points
from nff.config.experiment import TargetConfig, PhysicsConfig, TrainingConfig, ValidityConfig
from nff.stages.constraints import hinge_connectivity
from nff.stages.geometry import compute_total_area, compute_void_area
from nff.stages.pipeline import forward_pipeline
from nff.stages.state import CentroidalState

# Softmax temperature for the path reductions (relative units: D is D_scale-normalized, eta_a is
# a/w_lig). A hard jnp.max/min picks one load step and its gradient jumps to another as the design
# moves; the softmax spreads it over the steps actually near the extremum. Small enough that the
# value tracks the true extremum closely.
_PATH_TEMPERATURE = 0.05

# Void areas are clipped before log1p: Stage-1 divergence on a hard problem otherwise turns a
# reward into inf/nan and poisons the whole gradient.
_VOID_CLIP = 20.0


# ── shared context ────────────────────────────────────────────────────────────

class LossContext(NamedTuple):
    """Everything the terms read. Built once per loss evaluation, never mutated."""
    results: dict                  # forward_pipeline output
    initial_state: CentroidalState
    map_params: Any
    map_type: str
    training_cfg: TrainingConfig
    target_cloud: Float[Array, "n_target 2"]
    hinge_geometry: Any            # HingeGeometry, or None under the ROM
    hinge_probe: Optional[dict]    # per-step per-hinge surrogate readings, or None
    learn_global_scale: bool


def _softmax_max(x, axis, temperature: float = _PATH_TEMPERATURE):
    """Smooth maximum along ``axis`` (softmax-weighted mean, so it stays inside the data range)."""
    w = jax.nn.softmax(x / temperature, axis=axis)
    return jnp.sum(w * x, axis=axis)


def _softmax_min(x, axis, temperature: float = _PATH_TEMPERATURE):
    """Smooth minimum along ``axis``."""
    return -_softmax_max(-x, axis, temperature)


# Soft-extremum temperature as a FRACTION of each path's own range, rather than the absolute
# _PATH_TEMPERATURE above. A softmax-weighted mean is biased toward the mean by roughly the
# temperature, so an absolute temperature is only meaningful next to the spread of the data it
# reduces -- and on the 15x10 sheet it was not: temperature 0.05 against a mean per-path eta_a
# spread of 0.0198 biased the "minimum" by +0.0101, i.e. 0.51x the whole signal. The reduction
# reported a mean of +0.0024 (tension) for a population whose true path-minimum averaged -0.0077
# (compression), so relu(-worst) zeroed 68 of 196 compressive hinges before the penalty ever saw
# them. Scaling with the spread makes the bias a fixed small fraction of the signal instead.
_PATH_REL_TEMPERATURE = 0.05


def _softmax_extremum_relative(x, axis, sign: float,
                               rel_temperature: float = _PATH_REL_TEMPERATURE):
    """Smooth extremum with a per-path temperature set from that path's own range.

    Written on the NORMALIZED coordinate ``u = (x - min) / spread`` rather than on ``x`` with a
    temperature ``rel_temperature * spread``. The two are ALGEBRAICALLY IDENTICAL -- softmax is
    shift-invariant, so ``softmax(x / (r * spread))`` and ``softmax(u / r)`` differ only by the
    constant ``min / (r * spread)``, which cancels -- but only the second form is numerically safe.

    The first form divides by ``spread``, and its guard (``maximum(r * spread, 1e-9)``) did not
    guard: a path whose values barely move makes the temperature ``1e-9``, so the softmax Jacobian
    carries ``1 / t = 1e9`` and the reverse pass overflows. On the normalized coordinate ``u`` lies
    in [0, 1] and the softmax argument is at most ``1 / rel_temperature``, independent of the data
    scale, so no path can drive it. A path with zero spread gets ``u = 0`` everywhere, hence uniform
    weights and the constant value itself -- the right answer.

    ⚠ This is a latent-hazard fix, NOT the fix for the ``shear_path`` NaN. That one was ``jnp.abs``
    (see ``_ABS_EPS``), and it was still NaN at 80 deg after this rewrite. The hazard here is real
    and was measured to bite at 70 deg, but do not credit this rewrite with the whole bug.

    Values are unchanged: agreement with the old form is 2.7e-16 relative on non-degenerate data, so
    every ``compression`` result predating this rewrite stands.

    Args:
        x: values to reduce.
        axis: reduction axis.
        sign: ``+1`` for a smooth maximum, ``-1`` for a smooth minimum.
        rel_temperature: temperature as a fraction of the per-path peak-to-peak range.

    Returns:
        The soft extremum, with ``axis`` removed.
    """
    lo = jnp.min(x, axis=axis, keepdims=True)
    spread = jnp.max(x, axis=axis, keepdims=True) - lo
    # Double `where`: sanitizing only the OUTPUT still returns NaN from the backward pass, because
    # AD differentiates the discarded branch too. The denominator itself has to be finite.
    degenerate = spread <= 0.0
    safe_spread = jnp.where(degenerate, 1.0, spread)
    u = jnp.where(degenerate, 0.0, (x - lo) / safe_spread)
    w = jax.nn.softmax(sign * u / rel_temperature, axis=axis)
    return jnp.sum(w * x, axis=axis)


# Smoothing for |eta_s|, in squared eta units, so the kink at eta_s = 0 has radius sqrt(_ABS_EPS)
# = 1e-6. That is four orders of magnitude below the smallest per-hinge |eta_s| path spread measured
# on the 15x10 sheet (2.4e-4), so it changes the reported value nowhere that matters.
#
# It is a NUMERICAL fix, not a modelling choice. `jnp.abs` inside the path reduction returned a NaN
# gradient UNDER JIT ONLY -- eager was exact, so it passed a gradient check and then silently froze
# training, because the trainer's NaN-safe update zeroes a non-finite gradient rather than failing.
# Measured on sheet_15x10_rom_d80_shear at the untrained design: abs gave 176/226 NaN parameter
# gradients under jit and 0 eager, while this smooth form gives 0 NaN and reproduces BOTH the abs
# loss (7.72284e+01, 6 significant figures) and the abs EAGER gradient norm (4.971e+01). Independent
# of the penalty weight and of `backward_reg`, and non-monotonic in load steps (10 clean, 20 and 40
# NaN, 80 clean) -- the signature of a fused-kernel cancellation, not of the physics.
_ABS_EPS = 1e-12


def _smooth_abs(x, eps: float = _ABS_EPS):
    """``|x|``, differentiable at zero. See ``_ABS_EPS``: this exists to keep the path reduction's
    gradient finite under JIT, and is exact to 6 significant figures at the scales in use."""
    return jnp.sqrt(x * x + eps)


def _softmax_min_relative(x, axis, rel_temperature: float = _PATH_REL_TEMPERATURE):
    """Smooth minimum, temperature scaled by each path's own range."""
    return _softmax_extremum_relative(x, axis, -1.0, rel_temperature)


def _softmax_max_relative(x, axis, rel_temperature: float = _PATH_REL_TEMPERATURE):
    """Smooth maximum, temperature scaled by each path's own range."""
    return _softmax_extremum_relative(x, axis, +1.0, rel_temperature)


def _deformed_state(valid_state, displacements):
    """Rigid-tile deformed centroids, rotations and centroid-node vectors at a displacement field."""
    centroids = valid_state.face_centroids + displacements[:, :2]
    thetas = displacements[:, 2]
    cos_t, sin_t = jnp.cos(thetas), jnp.sin(thetas)
    cnv = valid_state.centroid_node_vectors
    deformed_cnv = jnp.stack([
        cos_t[:, None] * cnv[:, :, 0] - sin_t[:, None] * cnv[:, :, 1],
        sin_t[:, None] * cnv[:, :, 0] + cos_t[:, None] * cnv[:, :, 1],
    ], axis=-1)
    return centroids, thetas, deformed_cnv


# ── geometric terms ───────────────────────────────────────────────────────────

def _circularity_loss(pts: Float[Array, "n 2"]) -> Float[Array, ""]:
    """Scale/translation-invariant circularity loss for a boundary point cloud.

    Fits the best circle algebraically (Kåsa: solve for the center and radius that make the points
    equidistant from one unknown point), then returns the mean squared *relative* radial residual.
    Zero iff the points lie on a circle of any size/position -- so the target circle's size adapts
    to the tessellation instead of being fixed.

    Args:
        pts: (n, 2) deployed boundary vertices.

    Returns:
        Scalar circularity loss.
    """
    n = pts.shape[0]
    A = jnp.concatenate([2.0 * pts, jnp.ones((n, 1))], axis=1)         # (n, 3)
    rhs = jnp.sum(pts ** 2, axis=1)                                    # (n,)
    coeffs = jnp.linalg.solve(A.T @ A + 1e-8 * jnp.eye(3), A.T @ rhs)  # [cx, cy, c]
    center = coeffs[:2]
    radius = jnp.sqrt(jnp.clip(coeffs[2] + jnp.sum(center ** 2), 1e-12, None))
    dist = jnp.linalg.norm(pts - center[None, :], axis=1)
    return jnp.mean(((dist - radius) / radius) ** 2)


def _deployed_boundary_points(ctx: LossContext) -> Float[Array, "n_boundary 2"]:
    """The deployed geometry the shape term matches against the target."""
    valid_state = ctx.results['valid_state']
    centroids, thetas, _ = _deformed_state(valid_state, ctx.results['solution'].fields[-1])

    b_face_ids = valid_state.boundary_face_node_ids[:, 0]
    if ctx.training_cfg.geometric_loss_type not in ("boundary_vertices", "circle_fit"):
        return centroids[b_face_ids]   # centroid fallback: faster, more approximate

    # The actual exterior VERTICES of the boundary faces, carried by their face's rigid motion.
    b_local_node_ids = valid_state.boundary_face_node_ids[:, 1]
    b_vectors = valid_state.centroid_node_vectors[b_face_ids, b_local_node_ids]
    b_thetas = thetas[b_face_ids]
    cos_t, sin_t = jnp.cos(b_thetas), jnp.sin(b_thetas)
    rotated = jnp.stack([
        cos_t * b_vectors[:, 0] - sin_t * b_vectors[:, 1],
        sin_t * b_vectors[:, 0] + cos_t * b_vectors[:, 1],
    ], axis=-1)
    return centroids[b_face_ids] + rotated


def _term_chamfer(ctx: LossContext):
    """Shape matching: bidirectional Chamfer to the target cloud, or a size-free circularity fit.

    ``coverage`` scales only the recall half. It exists to stop a collapse to a single point, and it
    carries a hard floor set by how few boundary vertices there are -- on the 3x3 sheet, 12 vertices
    against a sampled perimeter, that floor can exceed the precision term and fight it.
    """
    points = _deployed_boundary_points(ctx)
    if ctx.training_cfg.geometric_loss_type == "circle_fit":
        value = _circularity_loss(points)
        return value, {'chamfer_precision': value}

    sq_dist = jnp.sum((points[:, None, :] - ctx.target_cloud[None, :, :]) ** 2, axis=-1)
    precision = jnp.mean(jnp.min(sq_dist, axis=1))   # each boundary vertex -> nearest target point
    coverage = jnp.mean(jnp.min(sq_dist, axis=0))    # each target point -> nearest boundary vertex
    value = precision + ctx.training_cfg.loss_weights.coverage * coverage
    return value, {'chamfer_precision': precision, 'chamfer_coverage': coverage}


def _term_material_area(ctx: LossContext):
    """Hold the mapped sheet's total area at the flat reference area.

    Only meaningful when the global scale is NOT learnable -- otherwise the scale absorbs the
    constraint. GNN maps move centroids without transforming the CNVs, so their Stage-0 area is
    unchanged by construction and the Stage-1 output is what carries the real distortion.
    """
    if ctx.learn_global_scale:
        return jnp.asarray(0.0, dtype=jnp.float64), {'area_deviation': 0.0}
    source = 'valid_state' if ctx.map_type.startswith('gnn_') else 'mapped_state'
    mapped_area = compute_total_area(ctx.results[source].centroid_node_vectors)
    deviation = mapped_area - jnp.sum(ctx.initial_state.initial_face_areas)
    return deviation ** 2, {'area_deviation': deviation}


def _term_hinge_gap(ctx: LossContext):
    """Penalize hinge vertex pairs that should coincide but don't, at the Stage-0 output.

    Evaluated BEFORE Stage 1 so the mapping gets a direct gradient toward connected tiles instead of
    leaving every gap for the validity solver to absorb.
    """
    ms = ctx.results['mapped_state']
    return hinge_connectivity(ms.face_centroids, ms.centroid_node_vectors, ms.hinge_node_pairs), {}


def _stage2_void_area(ctx: LossContext):
    vs = ctx.results['valid_state']
    centroids, _, deformed_cnv = _deformed_state(vs, ctx.results['solution'].fields[-1])
    area = compute_void_area(centroids, deformed_cnv, vs.boundary_face_node_ids)
    return jnp.clip(jnp.nan_to_num(area, nan=0.0, posinf=0.0, neginf=0.0), 0.0, _VOID_CLIP)


def _term_void_closure(ctx: LossContext):
    """Penalize void area REMAINING after loading.

    No Stage-1 reference, so it cannot be gamed by inflating the starting void.
    """
    void2 = _stage2_void_area(ctx)
    return jnp.log1p(void2), {'void_stage2': void2}


def _term_closure_delta(ctx: LossContext):
    """Reward the DECREASE in void area from Stage 1 to Stage 2.

    Rewards the loads actually doing the closing: a rigid-body swing leaves void area invariant, so
    the delta is 0 and it earns nothing.
    """
    vs = ctx.results['valid_state']
    void1 = compute_void_area(vs.face_centroids, vs.centroid_node_vectors, vs.boundary_face_node_ids)
    void1 = jnp.clip(jnp.nan_to_num(void1, nan=0.0, posinf=0.0, neginf=0.0), 0.0, _VOID_CLIP)
    delta = jnp.clip(void1 - _stage2_void_area(ctx), 0.0, _VOID_CLIP)
    return -jnp.log1p(delta), {'void_stage1': void1, 'void_delta': delta}


# ── energy terms ──────────────────────────────────────────────────────────────

def _energy_term(component: str):
    """Build a term reading one component of the Stage-2 energy decomposition at the last load step.

    ⚠ The decomposition is the ROM SPRING energy -- ``build_decompose_energy_fn`` has no surrogate
    hook -- so under ``hinge_model: surrogate`` these read linear-spring energies evaluated on
    surrogate-solved displacements. They are 0 in every live surrogate config; a non-zero weight
    there is a modelling error, not a physics term.
    """
    def term(ctx: LossContext):
        energies = ctx.results['solution'].energies
        if component not in energies:
            # Never substitute zeros. A weighted term that silently evaluates to 0 is
            # indistinguishable from a term that is doing nothing useful, and the weight in the
            # config reads as if the physics were active.
            raise KeyError(
                f"loss term '{component}' is weighted but the Stage-2 energy decomposition does not "
                f"provide it; available: {sorted(energies)}.")
        return energies[component][-1], {}
    return term


# ── hinge terms (surrogate only) ──────────────────────────────────────────────

def _term_damage(ctx: LossContext):
    """Push accumulated irreversibility down on EVERY hinge, threshold-free.

    Per hinge take the PATH MAX of ``D`` over the load steps, then ``mean(D_max**2)``. Path max
    rather than endpoint because the real material's PEEQ is monotone along any path while the
    surrogate's ``D`` is a STATE function: a hinge that swings out and comes back has done permanent
    damage that ``D`` at the final step no longer reports.

    ``D`` arrives already divided by the train-set RMS, so ``mean(D**2) ~ 1`` on the training
    distribution and the weight stays an O(1) knob.

    ``damage_path_gap`` (path max minus endpoint) is the size of that forgetting -- it measures how
    much path dependence the current design rides, comparable to the polyline-vs-straight-ray ratio
    from the path-dependence audit.
    """
    D = ctx.hinge_probe['D']                       # (n_steps, n_hinges)
    D_max = _softmax_max(D, axis=0)                # (n_hinges,)
    return jnp.mean(D_max ** 2), {
        'hinge_max_D': jnp.max(D_max),
        'hinge_mean_D': jnp.mean(D_max),
        'hinge_p90_D': jnp.percentile(D_max, 90.0),
        'damage_path_gap': jnp.max(D_max - D[-1]),
    }


def _term_compression(ctx: LossContext):
    """Penalize hinges driven into AXIAL COMPRESSION at any point during deployment.

    ``eta_a = a / w_lig`` is signed; ``eta_a < 0`` means the two tiles press together across the cut
    instead of opening. Two reasons to resist it. Physically, a 0.5 mm PET ligament in axial
    compression buckles out of plane immediately -- neither what the in-plane mechanism intends nor
    what a rigid-tile model represents. Epistemically, the oracle samples ``eta_a >= 0``, so
    compression is extrapolation.

    Path MIN, not endpoint: compression is transient, and a hinge compressed mid-deployment has
    already buckled whatever the final state says.

    ⚠ This becomes the ONLY handle on compression once the surrogate's trust box is itself
    compressive (``eta_a_min < 0``, which a checkpoint trained on compressive rows will report),
    because the OOD barrier then sees no violation at all. Metric names mirror
    ``nff/closed/hinge_paths.py::path_diagnostics`` so the differentiable term and the post-hoc
    NumPy diagnostic read against each other directly.
    """
    eta_a = ctx.hinge_probe['eta_a']                          # (n_steps, n_hinges)
    worst = _softmax_min_relative(eta_a, axis=0)              # (n_hinges,) most-compressive value
    violation = jax.nn.relu(-worst)                           # 0 in tension, |eta_a| in compression
    # LINEAR, not quadratic. Violations here are order 1e-2, so squaring shrinks the term and its
    # gradient by another ~76x and makes the penalty weakest exactly where it is still doing useful
    # work -- on the many mildly compressed hinges rather than the few extreme ones. A linear
    # penalty also matches how the result is reported (a population fraction, not a worst case).
    true_worst = jnp.min(eta_a, axis=0)
    return jnp.mean(violation), {
        'min_eta_a': jnp.min(eta_a),
        'n_hinges_compressive': jnp.sum(worst < -1e-6),
        'compression_sample_frac': jnp.mean(eta_a < -1e-6),
        # The reduction's own bias, so it can never again silently exceed the signal it reduces.
        'compression_softmin_bias': jnp.mean(worst - true_worst),
        'compression_path_spread': jnp.mean(jnp.max(eta_a, axis=0) - true_worst),
    }


def _term_shear_path(ctx: LossContext):
    """Penalize hinges driven into transverse SHEAR at any point during deployment.

    ``eta_s = s / w_lig`` is the ligament's transverse offset in units of its own width, signed
    either way, so the penalized quantity is ``|eta_s|``. A rotating-squares mechanism is supposed
    to open by ROTATION about a hinge that stays aligned; shear is the ligament being dragged
    sideways, which does no opening work and is the in-plane mode a slender neck resists least.

    The sibling of ``compression`` and deliberately built the same way: path extremum (a hinge
    sheared mid-deployment has been sheared whatever the final state says), soft-min temperature
    scaled to the path's own range, and a LINEAR penalty so the many mildly-sheared hinges are not
    discounted against the few extreme ones.

    ``compression`` and this term act on the two axes of the SAME plane -- one pushes the path cloud
    off ``eta_a < 0``, the other squeezes it toward ``eta_s = 0`` -- so weighting both asks whether
    those two moves are compatible for one design. ``compression_shear_corr`` reports it directly:
    the correlation, over hinges, between how compressed and how sheared each one is. Strongly
    negative means the two penalties are fighting.
    """
    eta_s = ctx.hinge_probe['eta_s']                          # (n_steps, n_hinges)
    abs_eta_s = _smooth_abs(eta_s)                            # |eta_s|, differentiable at 0
    worst = _softmax_max_relative(abs_eta_s, axis=0)          # (n_hinges,) most-sheared value
    true_worst = jnp.max(abs_eta_s, axis=0)

    metrics = {
        'max_abs_eta_s': jnp.max(abs_eta_s),
        'mean_abs_eta_s': jnp.mean(abs_eta_s),
        'shear_softmax_bias': jnp.mean(true_worst - worst),
        'shear_path_spread': jnp.mean(true_worst - jnp.min(abs_eta_s, axis=0)),
    }
    if 'eta_a' in ctx.hinge_probe:
        compression = jax.nn.relu(-_softmax_min_relative(ctx.hinge_probe['eta_a'], axis=0))
        metrics['compression_shear_corr'] = jnp.corrcoef(compression, worst)[0, 1]
    return jnp.mean(worst), metrics


def _term_ood(ctx: LossContext):
    """Penalize hinges leaving the surrogate's trustworthy training box, at the deployed state.

    This is the DESIGN-side barrier, distinct from ``hinge_model.barrier`` which adds the same shape
    to the Stage-2 ENERGY to keep the solve bounded. One shapes the design, the other the solve.
    """
    return jnp.sum(ctx.hinge_probe['ood'][-1]), {}


def _term_regularization(ctx: LossContext):
    """Ridge on the design parameters, to stop the map running away."""
    squared = jax.tree_util.tree_map(lambda x: jnp.sum(x ** 2), ctx.map_params)
    return jax.tree_util.tree_reduce(lambda a, b: a + b, squared, initializer=0.0), {}


# ── the registry ──────────────────────────────────────────────────────────────

# (name, LossWeights attribute, term fn, REQUIRED PROBE KEY or None).
# This tuple IS the objective: evaluation order, and nothing outside it enters the loss.
#
# The fourth field was a bare "needs the probe" boolean, which conflated two different things: does
# this term read the hinge probe, and does the probe in use actually supply what it reads. The ROM
# has a KINEMATIC probe (eta_a, eta_s, theta -- see setup.build_rom_probe_fn) but no D and no ood,
# so a boolean would either enable all three terms or none. Naming the key each term needs lets
# `compression` run under the linear-spring model while `damage` and `ood`, which are surrogate
# predictions, disable themselves because their key is simply not there.
TERMS = (
    ('chamfer',        'chamfer',        _term_chamfer,           None),
    ('material_area',  'material_area',  _term_material_area,     None),
    ('hinge_gap',      'hinge_gap',      _term_hinge_gap,         None),
    ('void_closure',   'void_closure',   _term_void_closure,      None),
    ('closure_delta',  'closure_delta',  _term_closure_delta,     None),
    ('stretching',     'stretching',     _energy_term('stretch'), None),
    ('shearing',       'shearing',       _energy_term('shear'),   None),
    ('bending',        'bending',        _energy_term('rot'),     None),
    ('contact',        'contact',        _energy_term('contact'), None),
    ('damage',         'damage',         _term_damage,            'D'),
    ('compression',    'compression',    _term_compression,       'eta_a'),
    ('shear_path',     'shear_path',     _term_shear_path,        'eta_s'),
    ('ood',            'ood',            _term_ood,               'ood'),
    ('regularization', 'regularization', _term_regularization,    None),
)


def _legacy_metric_aliases(metrics: dict, total) -> dict:
    """Keys the trainer, the closed driver and the plotter already read, kept stable.

    ``chamfer_total`` in particular is a checkpoint-selection criterion, so it has to survive the
    rename even though the term now reports itself as ``chamfer_raw``.
    """
    zero = jnp.asarray(0.0, dtype=jnp.float64)
    energy = sum((metrics.get(f'{k}_raw', zero)
                  for k in ('stretching', 'shearing', 'bending', 'contact')), zero)
    return {
        'total':                total,
        'loss_total':           total,
        'chamfer_total':        metrics.get('chamfer_raw', zero),
        'energy':               energy,
        'global_material_area': metrics.get('area_deviation', zero),
        'hinge_gap':            metrics.get('comp_hinge_gap', zero),
        'void_closure':         metrics.get('comp_void_closure', zero),
        'closure_delta':        metrics.get('comp_closure_delta', zero),
        'stab_damage':          metrics.get('comp_damage', zero),
        'stab_ood':             metrics.get('comp_ood', zero),
    }


def compute_end_to_end_loss(
        map_params: Any,
        initial_state: CentroidalState,
        target_cfg: TargetConfig,
        validity_cfg: ValidityConfig,
        physics_cfg: PhysicsConfig,
        training_cfg: TrainingConfig,
        map_type: str = 'conformal_polynomial',
        use_shirley_chiu: bool = True,
        strict_boundary_fit: bool = True,
        learn_global_scale: bool = False,
        static_features: Any = None,
        load_specs: Any = None,
        target_cloud: Optional[Float[Array, "n_target 2"]] = None,
        bond_energy_fn=None,
        hinge_probe_fn=None,
        hinge_geometry_fn=None,
) -> tuple[Float[Array, ""], dict]:
    """Run the forward pipeline and evaluate every weighted term. Input to ``jax.value_and_grad``.

    Args:
        learn_global_scale: When True a learnable log_scale is present in map_params and the area
            constraint is lifted; when False the mapped area is held at the flat reference area.
        target_cloud: Precomputed (n_target, 2) target boundary points. Pass this from outside the
            JIT boundary (e.g. from ``create_train_step``) to avoid rebaking the numpy->JAX
            conversion at every trace. If None, computed internally.
        hinge_probe_fn: Surrogate probe from ``build_hinge_probe_fn``; enables the damage,
            compression and ood terms. None -> ROM, and those terms are skipped whatever their
            weights say.
        hinge_geometry_fn: ``map_params -> HingeGeometry``. Recomputed each step and threaded into
            the solver as ``control_params`` (NOT closed over), so jaxopt's implicit diff carries
            d(loss)/d(design) through the hinge geometry as well.

    Returns:
        ``(total_loss, metrics)``.
    """
    if target_cloud is None:
        target_params = {'type': target_cfg.type, 'center': target_cfg.center,
                         'radius': target_cfg.radius}
        target_cloud = jnp.asarray(get_target_points(target_params, n_points=500),
                                   dtype=jnp.float64)

    hinge_geometry = hinge_geometry_fn(map_params) if hinge_geometry_fn is not None else None

    # The SOLVER only needs the hinge geometry when a surrogate bond energy reads it; the ROM's
    # ligament energy is a function of the corotated strains alone and ignores it. Keeping it out of
    # control_params on the ROM branch means a linear-spring solve is bit-identical whether or not a
    # kinematic probe was built alongside it -- the probe still gets the design-tracked geometry
    # below, so d(compression)/d(design) flows through sec_dir as it must.
    solver_hinge_geometry = hinge_geometry if bond_energy_fn is not None else None

    results = forward_pipeline(
        initial_state=initial_state,
        target_cfg=target_cfg,
        validity_cfg=validity_cfg,
        physics_cfg=physics_cfg,
        map_type=map_type,
        map_params=map_params,
        use_shirley_chiu=use_shirley_chiu,
        strict_boundary_fit=strict_boundary_fit,
        static_features=static_features,
        load_specs=load_specs,
        bond_energy_fn=bond_energy_fn,
        hinge_geometry=solver_hinge_geometry,
    )

    weights = training_cfg.loss_weights
    # The probe runs ONCE and is shared: damage, compression and ood all read the same per-step
    # readings, and the surrogate MLP should not sweep the load history three times.
    wants_probe = any(float(getattr(weights, attr)) != 0.0
                      for _, attr, _, key in TERMS if key is not None)
    hinge_probe = None
    if hinge_probe_fn is not None and wants_probe:
        hinge_probe = hinge_probe_fn(results['solution'].fields,
                                     results['valid_state'].centroid_node_vectors,
                                     hinge_geometry, results['reference_bond_vectors'])

    ctx = LossContext(
        results=results,
        initial_state=initial_state,
        map_params=map_params,
        map_type=map_type,
        training_cfg=training_cfg,
        target_cloud=target_cloud,
        hinge_geometry=hinge_geometry,
        hinge_probe=hinge_probe,
        learn_global_scale=learn_global_scale,
    )

    total = jnp.asarray(0.0, dtype=jnp.float64)
    metrics: dict = {}
    for name, attr, term_fn, probe_key in TERMS:
        weight = float(getattr(weights, attr))
        if weight == 0.0:
            continue
        if probe_key is not None and (hinge_probe is None or probe_key not in hinge_probe):
            continue
        raw, extra = term_fn(ctx)
        contribution = weight * raw
        total = total + contribution
        metrics[f'comp_{name}'] = contribution
        metrics[f'{name}_raw'] = raw
        metrics.update(extra)

    metrics.update(_legacy_metric_aliases(metrics, total))
    return total, metrics
