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

    def __init__(self, type: str, center: Tuple[float, float], radius: float):
        self.type = type
        self.center = center
        self.radius = radius


class MappingConfig(eqx.Module):
    type: str
    params: Any
    use_shirley_chiu: bool
    scale_factor: float
    domain_restriction: float

    def __init__(self, type: str, params: Any, use_shirley_chiu: bool, scale_factor: float, domain_restriction: float):
        self.type = type
        self.params = params
        self.use_shirley_chiu = use_shirley_chiu
        self.scale_factor = scale_factor
        self.domain_restriction = domain_restriction


class PhysicsConfig(eqx.Module):
    scale_factor: float
    domain_restriction: float
    use_contact: bool
    k_contact: float
    min_angle: float  # radians
    cutoff_angle: float
    linearized_strains: bool
    incremental: bool
    num_load_steps: int
    geom_weights: Dict[str, float]
    solver_maxiter: int = 1000
    solver_tol: float = 1e-5

    def __init__(self,
                 scale_factor: float,
                 domain_restriction: float,
                 use_contact: bool,
                 k_contact: float,
                 min_angle: float,
                 cutoff_angle: float,
                 linearized_strains: bool,
                 incremental: bool,
                 num_load_steps: int,
                 geom_weights: Dict[str, float],
                 solver_maxiter: int = 1000,
                 solver_tol: float = 1e-5):
        self.scale_factor = scale_factor
        self.domain_restriction = domain_restriction
        self.use_contact = use_contact
        self.k_contact = k_contact
        self.min_angle = min_angle
        self.cutoff_angle = cutoff_angle
        self.linearized_strains = linearized_strains
        self.incremental = incremental
        self.num_load_steps = num_load_steps
        self.geom_weights = geom_weights
        self.solver_maxiter = solver_maxiter
        self.solver_tol = solver_tol


class TrainingConfig(eqx.Module):
    num_epochs: int
    learning_rate: float
    optimizer: str = "adam"
    loss_weights: dict
    geometric_loss_type: str = "boundary_vertices"

    def __init__(self, num_epochs: int, learning_rate: float, optimizer: str = "adam", loss_weights: dict = None, geometric_loss_type: str = "boundary_vertices"):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.loss_weights = loss_weights if loss_weights is not None else {"geometric": 1.0, "physics": 0.1, "regularization": 1e-3}
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
    physics: PhysicsConfig
    training: TrainingConfig
    visualization: VisualizationConfig

    def __init__(self, topology: dict, mapping: MappingConfig, target: TargetConfig, physics: PhysicsConfig, training: TrainingConfig, visualization: VisualizationConfig):
        self.topology = topology
        self.mapping = mapping
        self.target = target
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
        m_params_trainable = {k: v for k, v in m_params_raw.items() if k != 'use_shirley_chiu'}
        
    # Standardize to JAX PyTree (dict/array)
    m_params = parse_map_params(m_params_trainable)
        
    mapping_cfg = MappingConfig(
        type=m_type,
        params=m_params,
        use_shirley_chiu=m_use_sc,
        scale_factor=mapping_raw.get("scale_factor") if mapping_raw.get("scale_factor") is not None else 1.0,
        domain_restriction=mapping_raw.get("domain_restriction", 0.8)
    )

    # 3. Physics & Weights
    phys_raw = raw.get("physics", {})
    weights_raw = raw.get("optimization_weights", {})
    
    physics_cfg = PhysicsConfig(
        scale_factor=mapping_cfg.scale_factor,
        domain_restriction=mapping_cfg.domain_restriction,
        use_contact=phys_raw.get("use_contact", True),
        k_contact=phys_raw.get("k_contact", 1.0),
        min_angle=phys_raw.get("min_angle", 0.1) * deg_to_rad,
        cutoff_angle=phys_raw.get("cutoff_angle", 5.0) * deg_to_rad,
        linearized_strains=phys_raw.get("linearized_strains", True),
        incremental=phys_raw.get("incremental", False),
        num_load_steps=phys_raw.get("num_load_steps", 10),
        geom_weights=weights_raw,
        solver_maxiter=int(phys_raw.get("solver_maxiter", 1000)),
        solver_tol=float(phys_raw.get("solver_tol", 1e-5)),
    )
    
    # 3. Target
    target_raw = raw.get("target", {})
    target_cfg = TargetConfig(
        type=target_raw.get("type", "circle"),
        center=tuple(target_raw.get("center", (0.0, 0.0))),
        radius=float(target_raw.get("radius", 1.0))
    )
    
    # 4. Training (with defaults if missing)
    train_raw = raw.get("training", {})
    training_cfg = TrainingConfig(
        num_epochs=int(train_raw.get("num_epochs", 500)),
        learning_rate=float(train_raw.get("learning_rate", 0.01)),
        optimizer=str(train_raw.get("optimizer", "adam") or "adam"),
        loss_weights=raw.get("loss_weights", train_raw.get("loss_weights", {"geometric": 1.0, "physics": 0.1, "regularization": 1e-3})),
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
        physics=physics_cfg,
        training=training_cfg,
        visualization=vis_cfg,
    )
