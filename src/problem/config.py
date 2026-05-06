import yaml
import os
import numpy as np
from typing import Dict, Tuple

import equinox as eqx
import jax.numpy as jnp


class TargetConfig(eqx.Module):
    type: str
    center: Tuple[float, float]
    radius: float

    def __init__(self, type: str, center: Tuple[float, float], radius: float):
        self.type = type
        self.center = center
        self.radius = radius


class PhysicsConfig(eqx.Module):
    scale_factor: float
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

    def __init__(self, num_epochs: int, learning_rate: float, optimizer: str = "adam"):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.optimizer = optimizer


class VisualizationConfig(eqx.Module):
    show_stage0: bool
    show_stage1: bool
    show_stage2: bool
    save_plots: bool
    save_animation: bool

    def __init__(self, show_stage0: bool, show_stage1: bool, show_stage2: bool, save_plots: bool, save_animation: bool):
        self.show_stage0 = show_stage0
        self.show_stage1 = show_stage1
        self.show_stage2 = show_stage2
        self.save_plots = save_plots
        self.save_animation = save_animation


class ExperimentConfig(eqx.Module):
    topology: dict
    target: TargetConfig
    physics: PhysicsConfig
    training: TrainingConfig
    visualization: VisualizationConfig

    def __init__(self, topology: dict, target: TargetConfig, physics: PhysicsConfig, training: TrainingConfig, visualization: VisualizationConfig):
        self.topology = topology
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
    
    # Handle 'params' vs 'map_params' alias for initial mapping
    m_params = mapping_raw.get("map_params", mapping_raw.get("params", []))
    topo_combined["map_params"] = m_params
    
    # 2. Physics & Weights
    phys_raw = raw.get("physics", {})
    weights_raw = raw.get("optimization_weights", {})
    
    physics_cfg = PhysicsConfig(
        scale_factor=mapping_raw.get("scale_factor", 1.0),
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
        optimizer=str(train_raw.get("optimizer", "adam") or "adam")
    )
    
    # 5. Visualization
    vis_raw = raw.get("visualization", {})
    vis_cfg = VisualizationConfig(
        show_stage0=bool(vis_raw.get("show_stage0", False)),
        show_stage1=bool(vis_raw.get("show_stage1", False)),
        show_stage2=bool(vis_raw.get("show_stage2", True)),
        save_plots=bool(vis_raw.get("save_plots", True)),
        save_animation=bool(vis_raw.get("save_animation", True))
    )
    
    return ExperimentConfig(
        topology=topo_combined,
        target=target_cfg,
        physics=physics_cfg,
        training=training_cfg,
        visualization=vis_cfg,
    )
