import dataclasses
import os
import warnings
import yaml
import numpy as np
from typing import Any, Dict, Tuple, Union

import jax.numpy as jnp
import equinox as eqx

from nff.topology.core import UnitPattern


# ── YAML → JAX parameter conversion ──────────────────────────────────────────

def parse_map_params(
        raw_params: Union[Dict, jnp.ndarray, list],
) -> Union[Dict, jnp.ndarray]:
    """Convert raw mapping parameters (from YAML/dict/list) into JAX-compatible format.

    Dictionaries are preserved but values are converted to jnp arrays.
    Lists/arrays are converted to jnp.float64 arrays.
    """
    if isinstance(raw_params, dict):
        return {
            k: v if isinstance(v, bool) else jnp.array(v, dtype=float)
            for k, v in raw_params.items()
        }
    return jnp.array(raw_params, dtype=float)


# ── Dataclasses ───────────────────────────────────────────────────────────────

class TargetConfig(eqx.Module):
    type: str
    center: Tuple[float, float]
    radius: float

    def __init__(self, type: str, center: Tuple[float, float], radius: float):
        self.type = type
        self.center = center
        self.radius = radius


class MappingConfig(eqx.Module):
    type: str
    params: Any
    use_shirley_chiu: bool
    strict_boundary_fit: bool
    domain_restriction: float
    learn_global_scale: bool

    def __init__(self, type: str, params: Any, use_shirley_chiu: bool,
                 strict_boundary_fit: bool, domain_restriction: float,
                 learn_global_scale: bool = False):
        self.type = type
        self.params = params
        self.use_shirley_chiu = use_shirley_chiu
        self.strict_boundary_fit = strict_boundary_fit
        self.domain_restriction = domain_restriction
        self.learn_global_scale = learn_global_scale


class ValidityConfig(eqx.Module):
    weights: Dict[str, float]
    validity_method: str   # 'lbfgs' | 'alternating_projection'
    n_proj_iters: int      # only used when validity_method == 'alternating_projection'

    def __init__(self, weights: Dict[str, float],
                 validity_method: str = 'lbfgs',
                 n_proj_iters: int = 20):
        self.weights = weights
        self.validity_method = validity_method
        self.n_proj_iters = n_proj_iters


class PhysicsConfig(eqx.Module):
    domain_restriction: float
    use_contact: bool
    k_contact: float
    # Radians HERE, but the YAML supplies DEGREES -- `_parse_physics_config` converts. Writing
    # radians in a config double-converts them (a "-0.0349 # rad" entry ran at -0.035 deg).
    min_angle: float
    cutoff_angle: float
    linearized_strains: bool
    incremental: bool
    num_load_steps: int
    solver_maxiter: int
    solver_tol: float
    updated_lagrangian: bool
    backward_reg: float
    prescribed_displacements: tuple

    def __init__(self, domain_restriction: float, use_contact: bool,
                 k_contact: float, min_angle: float, cutoff_angle: float,
                 linearized_strains: bool, incremental: bool,
                 num_load_steps: int, solver_maxiter: int = 1000,
                 solver_tol: float = 1e-5, updated_lagrangian: bool = False,
                 backward_reg: float = 0.0, prescribed_displacements: tuple = ()):
        self.domain_restriction = domain_restriction
        self.use_contact = use_contact
        self.k_contact = k_contact
        self.min_angle = min_angle
        self.cutoff_angle = cutoff_angle
        self.linearized_strains = linearized_strains
        self.incremental = incremental
        self.num_load_steps = num_load_steps
        self.solver_maxiter = solver_maxiter
        self.solver_tol = solver_tol
        self.updated_lagrangian = updated_lagrangian
        # Tikhonov ridge on the implicit-diff backward solve (0 = off). Lifts the near-singular/
        # indefinite tangent stiffness to PD so the IFT gradient is well-conditioned.
        self.backward_reg = backward_reg
        # Raw `displacement_control` specs — imposed Stage-2 motion instead of (or alongside)
        # applied force. Kept raw here because nff/config must not import from nff/stages;
        # nff.stages.physics.displacement parses and validates them.
        self.prescribed_displacements = tuple(prescribed_displacements or ())


class LossWeights(eqx.Module):
    """Every weight in the design loss, in one place.

    All ADDITIVE weights default to 0.0: a term contributes only when a config asks for it. The
    previous non-zero defaults (material_area 1.0, contact 1.0, stretching/shearing/bending 0.1)
    meant every closed config had to write eleven explicit zeros just to switch terms off, and a
    config that merely forgot the block silently trained against contact and spring energies.

    ``chamfer`` keeps its 1.0 default (a loss with no shape term is not an experiment), and
    ``coverage`` is a MULTIPLIER inside the chamfer term rather than an additive weight -- it
    scales the recall half of `precision + coverage * recall`.
    """
    chamfer: float
    material_area: float
    stretching: float
    shearing: float
    bending: float
    contact: float
    regularization: float
    coverage: float
    hinge_gap: float
    void_closure: float  # reward void closing between Stage 1 and Stage 2
    closure_delta: float # sharpness of the void-closure reward
    damage: float        # surrogate hinge damage D, path-max over the deployment
    compression: float   # hinge axial compression (eta_a < 0), path-min over the deployment
    ood: float           # hinges leaving the surrogate's trustworthy training box

    def __init__(self, chamfer: float = 1.0, material_area: float = 0.0,
                 stretching: float = 0.0, shearing: float = 0.0,
                 bending: float = 0.0, contact: float = 0.0,
                 regularization: float = 0.0, coverage: float = 1.0,
                 hinge_gap: float = 0.0,
                 void_closure: float = 0.0, closure_delta: float = 0.0,
                 damage: float = 0.0, compression: float = 0.0, ood: float = 0.0):
        self.chamfer = float(chamfer)
        self.material_area = float(material_area)
        self.stretching = float(stretching)
        self.shearing = float(shearing)
        self.bending = float(bending)
        self.contact = float(contact)
        self.regularization = float(regularization)
        self.coverage = float(coverage)
        self.hinge_gap = float(hinge_gap)
        self.void_closure = float(void_closure)
        self.closure_delta = float(closure_delta)
        self.damage = float(damage)
        self.compression = float(compression)
        self.ood = float(ood)


class TrainingConfig(eqx.Module):
    num_epochs: int
    learning_rate: float
    optimizer: str
    loss_weights: LossWeights
    geometric_loss_type: str
    grad_clip: float
    lr_schedule: str  # "constant" or "cosine"

    def __init__(self, num_epochs: int, learning_rate: float,
                 optimizer: str = "adam", loss_weights: LossWeights = None,
                 geometric_loss_type: str = "boundary_vertices",
                 grad_clip: float = 1.0, lr_schedule: str = "constant"):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.loss_weights = loss_weights if loss_weights is not None else LossWeights()
        self.geometric_loss_type = geometric_loss_type
        self.grad_clip = float(grad_clip)
        self.lr_schedule = str(lr_schedule)


class VisualizationConfig(eqx.Module):
    stage0: bool
    stage1: bool
    stage2: bool
    energy_plot: bool
    animation: bool
    show_plots: bool
    save_outputs: bool
    show_hinges: bool
    show_hinge_indices: bool
    show_face_indices: bool
    show_external_forces: bool
    show_kinematic_blocks: bool

    def __init__(self, stage0: bool, stage1: bool, stage2: bool,
                 energy_plot: bool, animation: bool, show_plots: bool,
                 save_outputs: bool, show_hinges: bool = True,
                 show_hinge_indices: bool = True, show_face_indices: bool = True,
                 show_external_forces: bool = False, show_kinematic_blocks: bool = False):
        self.stage0 = stage0
        self.stage1 = stage1
        self.stage2 = stage2
        self.energy_plot = energy_plot
        self.animation = animation
        self.show_plots = show_plots
        self.save_outputs = save_outputs
        self.show_hinges = show_hinges
        self.show_hinge_indices = show_hinge_indices
        self.show_face_indices = show_face_indices
        self.show_external_forces = show_external_forces
        self.show_kinematic_blocks = show_kinematic_blocks


class HingeModelConfig(eqx.Module):
    """Which hinge energy the Stage-2 solver uses (config-selectable), + its material context.

    ``type``: ``'rom'`` (linear-spring ligament energy — the default, so existing configs are
    unchanged) or ``'surrogate'`` (the learned condensed hinge energy). For the surrogate, the
    material / thickness / kerf are what it was TRAINED for — declarative oversight (and a guard):
    changing them in config does NOT retrain it. ``w_lig_mm`` IS a real manufacturing choice within
    the trained range [1, 10] mm. ``calibrate`` co-solves length/energy scale to the pipeline's
    k_stretch/k_rot (Gap 2); set ``calibrate: false`` to pin ``length_scale``/``energy_scale``.

    ``barrier`` stays here: it is part of the Stage-2 ENERGY (it makes W coercive so the solve
    cannot run out of the trusted box), not part of the design objective. The design-loss weights
    that used to live here -- ``w_damage``, ``w_ood`` -- moved to ``loss_weights:`` where the rest
    of the objective is declared; see ``_migrate_hinge_model_weights``.
    """
    type: str
    checkpoint: str
    material: str
    thickness_mm: float
    w_lig_mm: float
    calibrate: bool
    length_scale: float
    energy_scale: float
    barrier: float
    fail_line: float
    learn_w_lig: bool

    def __init__(self, type='rom', checkpoint='data/surrogates/hinge_surrogate.pkl', material='S235',
                 thickness_mm=1.0, w_lig_mm=5.0, calibrate=True, length_scale=0.0,
                 energy_scale=0.0, barrier=0.05,
                 fail_line=1.0, learn_w_lig=False):
        self.type = type
        self.checkpoint = checkpoint
        self.material = material
        self.thickness_mm = thickness_mm
        self.w_lig_mm = w_lig_mm
        self.calibrate = calibrate
        self.length_scale = length_scale
        self.energy_scale = energy_scale
        self.barrier = barrier
        # fail_line: REPORTING-only threshold (count of hinges above it); never enters the loss --
        #   set it to the campaign's calibrated Delta_tear, NOT to 1.
        self.fail_line = fail_line
        self.learn_w_lig = learn_w_lig    # per-hinge ligament width as a learnable design DOF


class ExperimentConfig(eqx.Module):
    topology: dict
    mapping: MappingConfig
    target: TargetConfig
    validity: ValidityConfig
    physics: PhysicsConfig
    training: TrainingConfig
    visualization: VisualizationConfig
    hinge_model: HingeModelConfig

    def __init__(self, topology: dict, mapping: MappingConfig, target: TargetConfig,
                 validity: ValidityConfig, physics: PhysicsConfig,
                 training: TrainingConfig, visualization: VisualizationConfig,
                 hinge_model: HingeModelConfig = None):
        self.topology = topology
        self.mapping = mapping
        self.target = target
        self.validity = validity
        self.physics = physics
        self.training = training
        self.visualization = visualization
        self.hinge_model = hinge_model if hinge_model is not None else HingeModelConfig()


# ── Private parsing helpers ───────────────────────────────────────────────────

def _load_pattern(topo_raw: dict, config_dir: str) -> UnitPattern:
    """Load the named pattern from the patterns library and build a UnitPattern."""
    patterns_path = "data/library/patterns.yaml"
    if not os.path.exists(patterns_path):
        patterns_path = os.path.join(config_dir, "../library/patterns.yaml")

    with open(patterns_path) as f:
        patterns_data = yaml.safe_load(f)

    pattern_name = topo_raw.get('pattern', "unit_RDQK_D")
    if pattern_name not in patterns_data:
        raise ValueError(f"Pattern '{pattern_name}' not found in {patterns_path}")

    pattern_raw = patterns_data[pattern_name]

    internal_hinges = []
    for h in pattern_raw.get('internal_hinges', []):
        h_copy = h.copy()
        if 'angle_factor' in h_copy:
            h_copy['angle'] = h_copy.pop('angle_factor') * jnp.pi
        internal_hinges.append(h_copy)

    return UnitPattern(
        vertices=np.array(pattern_raw['vertices']),
        faces=pattern_raw['faces'],
        internal_hinges=internal_hinges,
        external_hinges=pattern_raw.get('external_hinges', []),
        border_edges=pattern_raw.get('border_edges', {}),
    )


def _parse_mapping_config(mapping_raw: dict) -> MappingConfig:
    """Parse the [mapping] YAML section into a MappingConfig."""
    m_type = mapping_raw.get("map_type", "conformal_polynomial")
    m_use_sc = bool(mapping_raw.get('use_shirley_chiu', True))

    params_raw = mapping_raw.get("map_params", mapping_raw.get("params", []))

    # If use_shirley_chiu was nested inside map_params, extract and strip it.
    if isinstance(params_raw, dict):
        m_use_sc = bool(params_raw.get('use_shirley_chiu', m_use_sc))
        params_raw = {k: v for k, v in params_raw.items()
                      if k not in ('use_shirley_chiu', 's_val')}

    # For GNN types, map_params holds the initialization config (hidden_dim, seed, ...), not
    # trainable weights, so it is kept raw.
    if m_type.startswith('gnn_'):
        parsed_params = params_raw if isinstance(params_raw, dict) else {}
    else:
        parsed_params = parse_map_params(params_raw)

    return MappingConfig(
        type=m_type,
        params=parsed_params,
        use_shirley_chiu=m_use_sc,
        strict_boundary_fit=bool(mapping_raw.get('strict_boundary_fit', True)),
        domain_restriction=float(mapping_raw.get("domain_restriction", 0.8)),
        learn_global_scale=bool(mapping_raw.get('learn_global_scale', False)),
    )


def _parse_validity_config(weights_raw: dict) -> ValidityConfig:
    """Parse the [optimization_weights] YAML section.

    Recognises two special keys that are not penalty weights:
      validity_method : 'lbfgs' (default) | 'alternating_projection'
      n_proj_iters    : int, only used when validity_method='alternating_projection'
    All other keys are treated as penalty weights and passed to the L-BFGS solver.
    """
    raw = dict(weights_raw)  # copy so we don't mutate the caller's dict
    validity_method = str(raw.pop('validity_method', 'lbfgs'))
    n_proj_iters    = int(raw.pop('n_proj_iters', 20))
    return ValidityConfig(
        weights=raw,
        validity_method=validity_method,
        n_proj_iters=n_proj_iters,
    )


def _parse_physics_config(physics_raw: dict, domain_restriction: float,
                          displacement_raw: list | None = None) -> PhysicsConfig:
    """Parse the [physics] YAML section. Angles are converted from degrees to radians.

    ``displacement_raw`` is the top-level [displacement_control] list — a sibling of [loads],
    carried here because both are Stage-2 boundary conditions.
    """
    deg_to_rad = float(jnp.pi / 180.0)
    if displacement_raw is not None and not isinstance(displacement_raw, list):
        raise TypeError("[displacement_control] must be a list of {face, dof, value} entries, "
                        f"got {type(displacement_raw).__name__}.")
    return PhysicsConfig(
        domain_restriction=domain_restriction,
        use_contact=bool(physics_raw.get("use_contact", True)),
        k_contact=float(physics_raw.get("k_contact", 1.0)),
        min_angle=float(physics_raw.get("min_angle", 0.1)) * deg_to_rad,
        cutoff_angle=float(physics_raw.get("cutoff_angle", 5.0)) * deg_to_rad,
        linearized_strains=bool(physics_raw.get("linearized_strains", True)),
        incremental=bool(physics_raw.get("incremental", False)),
        num_load_steps=int(physics_raw.get("num_load_steps", 10)),
        solver_maxiter=int(physics_raw.get("solver_maxiter", 1000)),
        solver_tol=float(physics_raw.get("solver_tol", 1e-5)),
        updated_lagrangian=bool(physics_raw.get("updated_lagrangian", False)),
        backward_reg=float(physics_raw.get("backward_reg", 0.0)),
        prescribed_displacements=tuple(displacement_raw or ()),
    )


# hinge_model key -> the loss_weights key it became. These are DESIGN-LOSS weights; they belong
# with the rest of the objective, not in the block that identifies which hinge model to load.
_MIGRATED_HINGE_WEIGHTS = {"w_damage": "damage", "w_ood": "ood"}


def _migrate_hinge_model_weights(hinge_raw: dict, loss_weights_raw: dict) -> dict:
    """Fold legacy ``hinge_model.w_damage`` / ``w_ood`` into the loss_weights dict.

    An explicit ``loss_weights:`` entry always wins -- the migration only fills a weight the new
    block does not mention, so a config already updated is never overridden by a stale key.
    """
    merged = dict(loss_weights_raw or {})
    for old, new in _MIGRATED_HINGE_WEIGHTS.items():
        if old not in (hinge_raw or {}):
            continue
        if new in merged:
            warnings.warn(f"hinge_model.{old} and loss_weights.{new} are both set; "
                          f"using loss_weights.{new}={merged[new]}.", stacklevel=3)
            continue
        warnings.warn(f"hinge_model.{old} moved to loss_weights.{new} (design-loss weights now live "
                      f"in one block). Reading it from hinge_model for now.", stacklevel=3)
        merged[new] = hinge_raw[old]
    return merged


def _parse_hinge_model_config(raw: dict) -> HingeModelConfig:
    # w_fail/m_safe were the second damage criterion (a softplus break barrier). Removed 2026-07-28:
    # there is now ONE damage term. Warn rather than ignore, so an old config that actually relied
    # on the barrier cannot change meaning silently.
    if float(raw.get("w_fail", 0.0)) != 0.0:
        warnings.warn("hinge_model.w_fail is removed (the break barrier was a second damage "
                      "criterion); the single term is loss_weights.damage. Ignoring it.",
                      stacklevel=2)
    return HingeModelConfig(
        type=str(raw.get("type", "rom")),
        checkpoint=str(raw.get("checkpoint", "data/surrogates/hinge_surrogate.pkl")),
        material=str(raw.get("material", "S235")),
        thickness_mm=float(raw.get("thickness_mm", 1.0)),
        w_lig_mm=float(raw.get("w_lig_mm", 5.0)),
        calibrate=bool(raw.get("calibrate", True)),
        length_scale=float(raw.get("length_scale", 0.0)),
        energy_scale=float(raw.get("energy_scale", 0.0)),
        barrier=float(raw.get("barrier", 0.05)),
        fail_line=float(raw.get("fail_line", 1.0)),
        learn_w_lig=bool(raw.get("learn_w_lig", False)),
    )


def _parse_target_config(target_raw: dict) -> TargetConfig:
    """Parse the [target] YAML section."""
    return TargetConfig(
        type=target_raw.get("type", "circle"),
        center=tuple(target_raw.get("center", (0.0, 0.0))),
        radius=float(target_raw.get("radius", 1.0)),
    )


# Weights deleted from LossWeights, mapped to why. A config may still carry them; they are dropped
# with a warning instead of raising, so an old experiment file still loads.
_REMOVED_LOSS_WEIGHTS = {
    "openness": "rewarded Stage-1 void area; superseded by void_closure/closure_delta",
    "deformation": "rewarded raw Stage-2 displacement, which a rigid-body swing maximises for free",
}


def _parse_loss_weights(raw: dict) -> LossWeights:
    """Build LossWeights from the [loss_weights] YAML section, tolerating stale keys.

    Unknown keys WARN and are dropped rather than raising TypeError. The strict splat this replaces
    made any config carrying a since-removed weight fail to load at all -- which is how
    ``architectures/hinge_closing.yaml`` and the legacy asymmetric_roots problems became unloadable.
    """
    known = {f.name for f in dataclasses.fields(LossWeights)}
    clean, stale = {}, []
    for key, value in (raw or {}).items():
        if key in known:
            clean[key] = value
        elif key in _REMOVED_LOSS_WEIGHTS:
            if float(value or 0.0) != 0.0:
                warnings.warn(f"loss_weights.{key} is removed ({_REMOVED_LOSS_WEIGHTS[key]}) and "
                              f"was set to {value}. Ignoring it.", stacklevel=3)
        else:
            stale.append(key)
    if stale:
        warnings.warn(f"unknown loss_weights ignored: {sorted(stale)}. "
                      f"Known weights: {sorted(known)}.", stacklevel=3)
    return LossWeights(**clean)


def _parse_training_config(training_raw: dict, loss_weights_raw: dict) -> TrainingConfig:
    """Parse the [training] and [loss_weights] YAML sections."""
    return TrainingConfig(
        num_epochs=int(training_raw.get("num_epochs", 500)),
        learning_rate=float(training_raw.get("learning_rate", 0.01)),
        optimizer=str(training_raw.get("optimizer", "adam") or "adam"),
        loss_weights=_parse_loss_weights(loss_weights_raw),
        geometric_loss_type=str(training_raw.get("geometric_loss_type", "boundary_vertices")),
        grad_clip=float(training_raw.get("grad_clip", 1.0)),
        lr_schedule=str(training_raw.get("lr_schedule", "constant")),
    )


def _parse_visualization_config(vis_raw: dict) -> VisualizationConfig:
    """Parse the [visualization] YAML section."""
    return VisualizationConfig(
        stage0=bool(vis_raw.get("stage0", False)),
        stage1=bool(vis_raw.get("stage1", False)),
        stage2=bool(vis_raw.get("stage2", True)),
        energy_plot=bool(vis_raw.get("energy_plot", True)),
        animation=bool(vis_raw.get("animation", True)),
        show_plots=bool(vis_raw.get("show_plots", True)),
        save_outputs=bool(vis_raw.get("save_outputs", True)),
        show_hinges=bool(vis_raw.get("show_hinges", True)),
        show_hinge_indices=bool(vis_raw.get("show_hinge_indices", True)),
        show_face_indices=bool(vis_raw.get("show_face_indices", True)),
        show_external_forces=bool(vis_raw.get("show_external_forces", False)),
        show_kinematic_blocks=bool(vis_raw.get("show_kinematic_blocks", False)),
    )


# ── Public entry points ───────────────────────────────────────────────────────

def load_arch_config(arch_path: str) -> dict:
    """Load an architecture YAML (no BCs/loads/physics/material) as a raw dict."""
    with open(arch_path) as f:
        return yaml.safe_load(f)


def load_problem_suite(suite_path: str) -> list[dict]:
    """Load a problem suite YAML and return a list of fully-resolved problem dicts.

    Each returned dict is ready to be merged with an arch dict via merge_arch_problem().
    physics and material keys are resolved from per-problem overrides + suite defaults.
    """
    with open(suite_path) as f:
        raw = yaml.safe_load(f)

    physics_defaults = raw.get('physics_defaults', {})
    material_defaults = raw.get('material_defaults', {})

    problems = []
    for p in raw.get('problems', []):
        resolved = dict(p)
        resolved['physics'] = {**physics_defaults, **p.get('physics', {})}
        resolved['material'] = {**material_defaults, **p.get('material', {})}
        problems.append(resolved)
    return problems


def merge_arch_problem(arch_raw: dict, problem: dict) -> dict:
    """Merge an architecture dict and a resolved problem dict into a single raw dict.

    The problem supplies: boundary_conditions, loads, physics, material.
    The arch supplies everything else (tessellation, mapping, training, etc.).
    Problem keys always win on overlap.
    """
    merged = dict(arch_raw)
    merged['boundary_conditions'] = problem.get('boundary_conditions', {})
    merged['loads'] = problem.get('loads', [])
    merged['displacement_control'] = problem.get('displacement_control', [])
    merged['physics'] = problem.get('physics', {})
    merged['material'] = problem.get('material', {})
    return merged


def load_combined_config(arch_path: str, problem: dict) -> 'ExperimentConfig':
    """Build an ExperimentConfig from an architecture file + a resolved problem dict."""
    arch_raw = load_arch_config(arch_path)
    merged = merge_arch_problem(arch_raw, problem)
    config_dir = os.path.dirname(arch_path)
    return _parse_full_raw(merged, config_dir)


def _parse_full_raw(raw: dict, config_dir: str) -> 'ExperimentConfig':
    """Parse a fully-merged raw dict into an ExperimentConfig."""
    topo_raw = raw.get("tessellation", {})
    mapping_raw = raw.get("mapping", {})
    mat_raw = raw.get("material", {})
    bc_raw = raw.get("boundary_conditions", {})
    loads_raw = raw.get("loads", [])

    pattern_obj = _load_pattern(topo_raw, config_dir)
    mapping_cfg = _parse_mapping_config(mapping_raw)
    validity_cfg = _parse_validity_config(raw.get("optimization_weights", {}))
    physics_cfg = _parse_physics_config(raw.get("physics", {}), mapping_cfg.domain_restriction,
                                        raw.get("displacement_control"))
    target_cfg = _parse_target_config(raw.get("target", {}))
    hinge_model_raw = raw.get("hinge_model", {})
    # hinge_model is read BEFORE training: its legacy w_damage/w_ood are folded into loss_weights.
    loss_weights_raw = _migrate_hinge_model_weights(hinge_model_raw, raw.get("loss_weights", {}))
    training_cfg = _parse_training_config(raw.get("training", {}), loss_weights_raw)
    vis_cfg = _parse_visualization_config(raw.get("visualization", {}))
    hinge_model_cfg = _parse_hinge_model_config(hinge_model_raw)

    topo_combined = {
        **topo_raw,
        **mapping_raw,
        **mat_raw,
        'pattern': pattern_obj,
        'bc_clamped': bc_raw.get('clamped_faces', "boundary"),
        'loads': loads_raw,
    }

    return ExperimentConfig(
        topology=topo_combined,
        mapping=mapping_cfg,
        target=target_cfg,
        validity=validity_cfg,
        physics=physics_cfg,
        training=training_cfg,
        visualization=vis_cfg,
        hinge_model=hinge_model_cfg,
    )


def load_and_parse_config(yaml_path: str) -> ExperimentConfig:
    """Read a single YAML experiment file and return an immutable ExperimentConfig."""
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)
    return _parse_full_raw(raw, os.path.dirname(yaml_path))
