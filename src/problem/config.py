import yaml
import os
import numpy as np
from typing import Dict, Tuple, Any

import jax.numpy as jnp
import equinox as eqx

from jax_backend.initial_map import parse_map_params


class TargetConfig(eqx.Module):
    type: str
    center: Tuple[float, float]
    radius: float
    enforce_global_material_area: bool

    def __init__(self, type: str, center: Tuple[float, float], radius: float, 
                 enforce_global_material_area: bool = False):
        self.type = type
        self.center = center
        self.radius = radius
        self.enforce_global_material_area = enforce_global_material_area

class MappingConfig(eqx.Module):
    type: str
    params: Any
    use_shirley_chiu: bool
    strict_boundary_fit: bool
    domain_restriction: float

    def __init__(self, type: str, params: Any, use_shirley_chiu: bool, 
                 strict_boundary_fit: bool, 
                 domain_restriction: float):
        self.type = type
        self.params = params
        self.use_shirley_chiu = use_shirley_chiu
        self.strict_boundary_fit = strict_boundary_fit
        self.domain_restriction = domain_restriction


class ValidityConfig(eqx.Module):
    """Configuration for Stage 1 — Geometric Validity."""
    weights: Dict[str, float]
    preserve_face_area: bool
    face_area_weight: float

    def __init__(self, weights: Dict[str, float], 
                 preserve_face_area: bool = False, 
                 face_area_weight: float = 1.0):
        self.weights = weights
        self.preserve_face_area = preserve_face_area
        self.face_area_weight = face_area_weight


class PhysicsConfig(eqx.Module):
    domain_restriction: float
    use_contact: bool
    k_contact: float
    min_angle: float  # radians
    cutoff_angle: float
    linearized_strains: bool
    incremental: bool
    num_load_steps: int
    solver_maxiter: int = 1000
    solver_tol: float = 1e-5

    def __init__(self,
                 domain_restriction: float,
                 use_contact: bool,
                 k_contact: float,
                 min_angle: float,
                 cutoff_angle: float,
                 linearized_strains: bool,
                 incremental: bool,
                 num_load_steps: int,
                 solver_maxiter: int = 1000,
                 solver_tol: float = 1e-5):
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


class LossWeights(eqx.Module):
    """Flat structure for all loss components (Stage 3)."""
    # Geometric
    chamfer: float = 1.0
    material_area: float = 1.0
    
    # Physics (Energies)
    stretching: float = 0.1
    shearing: float = 0.1
    bending: float = 0.1
    contact: float = 1.0
    
    # Regularization
    regularization: float = 0.001
    
    # Chamfer breakdown
    coverage: float = 1.0

    def __init__(self, **kwargs):
        # Default values if not provided
        self.chamfer = float(kwargs.get('chamfer', 1.0))
        self.material_area = float(kwargs.get('material_area', 1.0))
        self.stretching = float(kwargs.get('stretching', 0.1))
        self.shearing = float(kwargs.get('shearing', 0.1))
        self.bending = float(kwargs.get('bending', 0.1))
        self.contact = float(kwargs.get('contact', 1.0))
        self.regularization = float(kwargs.get('regularization', 0.001))
        self.coverage = float(kwargs.get('coverage', 1.0))

class TrainingConfig(eqx.Module):
    num_epochs: int
    learning_rate: float
    optimizer: str = "adam"
    loss_weights: LossWeights
    geometric_loss_type: str = "boundary_vertices"

    def __init__(self, num_epochs: int, learning_rate: float, optimizer: str = "adam", loss_weights: LossWeights = None, geometric_loss_type: str = "boundary_vertices"):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.loss_weights = loss_weights if loss_weights is not None else LossWeights()
        self.geometric_loss_type = geometric_loss_type


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

    def __init__(self, stage0: bool, stage1: bool, stage2: bool, energy_plot: bool, animation: bool, show_plots: bool, save_outputs: bool,
                 show_hinges: bool = True, show_hinge_indices: bool = True, show_face_indices: bool = True, show_external_forces: bool = False, show_kinematic_blocks: bool = False):
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


def load_and_parse_config(yaml_path: str) -> ExperimentConfig:
    """Read a YAML file and instantiate an immutable ExperimentConfig.
    Angles are converted from degrees to radians.
    """
    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f)

    deg_to_rad = jnp.pi / 180.0

    # 1. Load patterns library
    patterns_path = "data/library/patterns.yaml"
    if not os.path.exists(patterns_path):
        patterns_path = os.path.join(os.path.dirname(yaml_path), "../library/patterns.yaml")
        
    with open(patterns_path, 'r') as f:
        patterns_data = yaml.safe_load(f)

    # 1. Topology / Tessellation / Mapping
    topo_raw = raw.get("tessellation", {})
    mapping_raw = raw.get("mapping", {})
    
    pattern_name = topo_raw.get('pattern', "unit_RDQK_D")
    if pattern_name in patterns_data:
        pattern_raw = patterns_data[pattern_name]
        
        # Instantiate UnitPattern
        internal_hinges = []
        for h in pattern_raw.get('internal_hinges', []):
            h_copy = h.copy()
            if 'angle_factor' in h_copy:
                h_copy['angle'] = h_copy.pop('angle_factor') * jnp.pi
            internal_hinges.append(h_copy)
            
        from src.topology.core import UnitPattern
        pattern_obj = UnitPattern(
            vertices=np.array(pattern_raw['vertices']),
            faces=pattern_raw['faces'],
            internal_hinges=internal_hinges,
            external_hinges=pattern_raw.get('external_hinges', []),
            border_edges=pattern_raw.get('border_edges', {})
        )
    else:
        raise ValueError(f"Pattern '{pattern_name}' not found in {patterns_path}")

    # Material properties
    mat_raw = raw.get("material", {})
    
    # BCs and Loads
    bc_raw = raw.get("boundary_conditions", {})
    loads_raw = raw.get("loads", [])
    
    # Merge everything for configure_tessellation()
    topo_combined = {
        **topo_raw, 
        **mapping_raw, 
        **mat_raw,
        'pattern': pattern_obj,
        'bc_clamped': bc_raw.get('clamped_faces', "boundary"),
        'loads': loads_raw
    }
    
    # 2. Mapping
    m_type = mapping_raw.get("map_type", "conformal_polynomial")
    m_use_sc = bool(mapping_raw.get('use_shirley_chiu', True))
    
    # Handle 'params' vs 'map_params' alias for initial mapping
    m_params_raw = mapping_raw.get("map_params", mapping_raw.get("params", []))
    
    # If the user put 'use_shirley_chiu' inside map_params in YAML, we extract it
    # but we STRIP it from the trainable parameters to avoid JAX/Optax errors.
    m_params_trainable = m_params_raw
    if isinstance(m_params_raw, dict):
        m_use_sc = bool(m_params_raw.get('use_shirley_chiu', m_use_sc))
        m_params_trainable = {k: v for k, v in m_params_raw.items() 
                              if k not in ['use_shirley_chiu', 's_val']}
        
    # Standardize to JAX PyTree (dict/array)
    m_params = parse_map_params(m_params_trainable)
        
    mapping_cfg = MappingConfig(
        type=m_type,
        params=m_params,
        use_shirley_chiu=m_use_sc,
        strict_boundary_fit=bool(mapping_raw.get('strict_boundary_fit', True)),
        domain_restriction=mapping_raw.get("domain_restriction", 0.8)
    )

    # 3. Geometric Validity & Optimization Weights
    validity_raw = raw.get("validity", {})
    weights_raw = raw.get("optimization_weights", {})
    validity_cfg = ValidityConfig(
        weights=weights_raw,
        preserve_face_area=bool(validity_raw.get("preserve_face_area", False)),
        face_area_weight=float(weights_raw.get("face_area", 1.0))
    )

    # 4. Physics
    phys_raw = raw.get("physics", {})
    physics_cfg = PhysicsConfig(
        domain_restriction=mapping_cfg.domain_restriction,
        use_contact=phys_raw.get("use_contact", True),
        k_contact=phys_raw.get("k_contact", 1.0),
        min_angle=phys_raw.get("min_angle", 0.1) * deg_to_rad,
        cutoff_angle=phys_raw.get("cutoff_angle", 5.0) * deg_to_rad,
        linearized_strains=phys_raw.get("linearized_strains", True),
        incremental=phys_raw.get("incremental", False),
        num_load_steps=phys_raw.get("num_load_steps", 10),
        solver_maxiter=int(phys_raw.get("solver_maxiter", 1000)),
        solver_tol=float(phys_raw.get("solver_tol", 1e-5)),
    )
    
    # 5. Target
    target_raw = raw.get("target", {})
    target_cfg = TargetConfig(
        type=target_raw.get("type", "circle"),
        center=tuple(target_raw.get("center", (0.0, 0.0))),
        radius=float(target_raw.get("radius", 1.0)),
        enforce_global_material_area=bool(target_raw.get("enforce_global_material_area", False))
    )
    
    # 4. Training (with defaults if missing)
    train_raw = raw.get("training", {})
    weights_raw = raw.get("loss_weights", {})
    l_weights = LossWeights(**weights_raw)

    training_cfg = TrainingConfig(
        num_epochs=int(train_raw.get("num_epochs", 500)),
        learning_rate=float(train_raw.get("learning_rate", 0.01)),
        optimizer=str(train_raw.get("optimizer", "adam") or "adam"),
        loss_weights=l_weights,
        geometric_loss_type=str(train_raw.get("geometric_loss_type", "boundary_vertices"))
    )
    
    # 5. Visualization
    vis_raw = raw.get("visualization", {})
    vis_cfg = VisualizationConfig(
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
        show_kinematic_blocks=bool(vis_raw.get("show_kinematic_blocks", False))
    )
    
    return ExperimentConfig(
        topology=topo_combined,
        mapping=mapping_cfg,
        target=target_cfg,
        validity=validity_cfg,
        physics=physics_cfg,
        training=training_cfg,
        visualization=vis_cfg,
    )
