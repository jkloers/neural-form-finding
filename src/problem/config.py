import os
import yaml
import numpy as np
from typing import Any, Dict, Tuple

import jax.numpy as jnp
import equinox as eqx

from src.topology.core import UnitPattern
from jax_backend.initial_map import parse_map_params


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

    def __init__(self, weights: Dict[str, float]):
        self.weights = weights


class PhysicsConfig(eqx.Module):
    domain_restriction: float
    use_contact: bool
    k_contact: float
    min_angle: float   # radians
    cutoff_angle: float
    linearized_strains: bool
    incremental: bool
    num_load_steps: int
    solver_maxiter: int
    solver_tol: float
    updated_lagrangian: bool

    def __init__(self, domain_restriction: float, use_contact: bool,
                 k_contact: float, min_angle: float, cutoff_angle: float,
                 linearized_strains: bool, incremental: bool,
                 num_load_steps: int, solver_maxiter: int = 1000,
                 solver_tol: float = 1e-5, updated_lagrangian: bool = False):
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


class LossWeights(eqx.Module):
    chamfer: float
    material_area: float
    stretching: float
    shearing: float
    bending: float
    contact: float
    regularization: float
    coverage: float
    hinge_gap: float
    openness: float    # reward large void area at Stage 1 (open initial state)
    deformation: float # reward large Stage 2 displacement (significant closing)

    def __init__(self, chamfer: float = 1.0, material_area: float = 1.0,
                 stretching: float = 0.1, shearing: float = 0.1,
                 bending: float = 0.1, contact: float = 1.0,
                 regularization: float = 0.001, coverage: float = 1.0,
                 hinge_gap: float = 0.0,
                 openness: float = 0.0, deformation: float = 0.0):
        self.chamfer = float(chamfer)
        self.material_area = float(material_area)
        self.stretching = float(stretching)
        self.shearing = float(shearing)
        self.bending = float(bending)
        self.contact = float(contact)
        self.regularization = float(regularization)
        self.coverage = float(coverage)
        self.hinge_gap = float(hinge_gap)
        self.openness = float(openness)
        self.deformation = float(deformation)


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


class ExperimentConfig(eqx.Module):
    topology: dict
    mapping: MappingConfig
    target: TargetConfig
    validity: ValidityConfig
    physics: PhysicsConfig
    training: TrainingConfig
    visualization: VisualizationConfig

    def __init__(self, topology: dict, mapping: MappingConfig, target: TargetConfig,
                 validity: ValidityConfig, physics: PhysicsConfig,
                 training: TrainingConfig, visualization: VisualizationConfig):
        self.topology = topology
        self.mapping = mapping
        self.target = target
        self.validity = validity
        self.physics = physics
        self.training = training
        self.visualization = visualization


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

    # Pour les types GNN, map_params contient la config d'initialisation
    # (hidden_dim, seed…), pas des poids entraînables. On la garde brute.
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
    """Parse the [optimization_weights] YAML section."""
    return ValidityConfig(weights=weights_raw)


def _parse_physics_config(physics_raw: dict, domain_restriction: float) -> PhysicsConfig:
    """Parse the [physics] YAML section. Angles are converted from degrees to radians."""
    deg_to_rad = float(jnp.pi / 180.0)
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
    )


def _parse_target_config(target_raw: dict) -> TargetConfig:
    """Parse the [target] YAML section."""
    return TargetConfig(
        type=target_raw.get("type", "circle"),
        center=tuple(target_raw.get("center", (0.0, 0.0))),
        radius=float(target_raw.get("radius", 1.0)),
    )


def _parse_training_config(training_raw: dict, loss_weights_raw: dict) -> TrainingConfig:
    """Parse the [training] and [loss_weights] YAML sections."""
    return TrainingConfig(
        num_epochs=int(training_raw.get("num_epochs", 500)),
        learning_rate=float(training_raw.get("learning_rate", 0.01)),
        optimizer=str(training_raw.get("optimizer", "adam") or "adam"),
        loss_weights=LossWeights(**loss_weights_raw),
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
    physics_cfg = _parse_physics_config(raw.get("physics", {}), mapping_cfg.domain_restriction)
    target_cfg = _parse_target_config(raw.get("target", {}))
    training_cfg = _parse_training_config(raw.get("training", {}), raw.get("loss_weights", {}))
    vis_cfg = _parse_visualization_config(raw.get("visualization", {}))

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
    )


def load_and_parse_config(yaml_path: str) -> ExperimentConfig:
    """Read a single YAML experiment file and return an immutable ExperimentConfig."""
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)
    return _parse_full_raw(raw, os.path.dirname(yaml_path))
