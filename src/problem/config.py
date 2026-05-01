import yaml
import os
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Callable, Any
from src.problem.targets import DEFAULT_TARGET
from src.topology.core import UnitPattern

# ─────────────────────────────────────────────────────────────────────────────
# 1. Schema Definition
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CentroidalConfig:
    # Tessellation
    width: int
    height: int
    pattern: UnitPattern  # Now holds the instantiated object

    # Target shape
    target_type: str
    target_center: Tuple[float, float]
    target_radius: float

    # Material properties
    k_stretch: float
    k_shear: float
    k_rot: float
    density: float

    # Initial mapping
    map_type: str
    scale_factor: float

    # Geometric validity weights
    w_connectivity: float
    w_non_intersection: float
    w_target: float
    w_arm_symmetry: float
    w_void_length: float
    w_void_collinear: float

    # Physics
    use_contact: bool
    linearized_strains: bool
    k_contact: float
    min_angle: float
    cutoff_angle: float
    incremental: bool
    num_load_steps: int

    # Boundary Conditions & Loading
    bc_clamped: Any
    loads: list

    # Visualization
    show_stage0: bool
    show_stage1: bool
    show_stage2: bool
    save_plots: bool
    save_animation: bool


# ─────────────────────────────────────────────────────────────────────────────
# 2. Loading Logic
# ─────────────────────────────────────────────────────────────────────────────

def _instantiate_pattern(name, data):
    """Converts raw YAML data into a UnitPattern object."""
    # Process internal hinges to handle np.pi
    internal_hinges = []
    for h in data.get('internal_hinges', []):
        h_copy = h.copy()
        if 'angle_factor' in h_copy:
            h_copy['angle'] = h_copy.pop('angle_factor') * np.pi
        internal_hinges.append(h_copy)
        
    return UnitPattern(
        vertices=data['vertices'],
        faces=data['faces'],
        internal_hinges=internal_hinges,
        external_hinges=data.get('external_hinges', []),
        border_edges=data.get('border_edges', {})
    )

def load_config(yaml_path: str) -> CentroidalConfig:
    """Loads a CentroidalConfig from a YAML file.
    
    If the file doesn't exist or is empty, returns a config with built-in defaults.
    """
    # 1. Load patterns library
    patterns_path = os.path.join(os.path.dirname(yaml_path), "../library/patterns.yaml")
    if not os.path.exists(patterns_path):
        # Fallback for when running from root
        patterns_path = "data/library/patterns.yaml"
        
    with open(patterns_path, 'r') as f:
        patterns_data = yaml.safe_load(f)

    # 2. Load problem config
    data = {}
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f) or {}
    else:
        print(f"Warning: Configuration file {yaml_path} not found. Using built-in defaults.")

    config_dict = {}
    
    # Tessellation
    tess = data.get('tessellation', {})
    config_dict['width'] = tess.get('width', 2)
    config_dict['height'] = tess.get('height', 2)
    
    pattern_name = tess.get('pattern', "unit_RDQK_D")
    if pattern_name in patterns_data:
        config_dict['pattern'] = _instantiate_pattern(pattern_name, patterns_data[pattern_name])
    else:
        # Fallback to first pattern if requested name not found
        first_name = list(patterns_data.keys())[0]
        config_dict['pattern'] = _instantiate_pattern(first_name, patterns_data[first_name])

    # Target
    target = data.get('target', {})
    config_dict['target_type'] = target.get('type', DEFAULT_TARGET['type'])
    config_dict['target_center'] = tuple(target.get('center', [0.0, 0.0]))
    config_dict['target_radius'] = target.get('radius', DEFAULT_TARGET['radius'])

    # Material
    mat = data.get('material', {})
    config_dict['k_stretch'] = mat.get('k_stretch', 10.0)
    config_dict['k_shear'] = mat.get('k_shear', 5.0)
    config_dict['k_rot'] = mat.get('k_rot', 1.0)
    config_dict['density'] = mat.get('density', 1.0)

    # Mapping
    mapping = data.get('mapping', {})
    config_dict['map_type'] = mapping.get('type', 'elliptical_grip')
    config_dict['scale_factor'] = mapping.get('scale_factor', 1.0)

    # Weights
    w = data.get('optimization_weights', {})
    config_dict['w_connectivity'] = w.get('connectivity', 700.0)
    config_dict['w_non_intersection'] = w.get('non_intersection', 1000.0)
    config_dict['w_target'] = w.get('target', 1.0)
    config_dict['w_arm_symmetry'] = w.get('arm_symmetry', 1.0)
    config_dict['w_void_length'] = w.get('void_length', 1000.0)
    config_dict['w_void_collinear'] = w.get('void_collinear', 1000.0)

    # Physics
    phys = data.get('physics', {})
    config_dict['use_contact'] = phys.get('use_contact', True)
    config_dict['linearized_strains'] = phys.get('linearized_strains', True)
    config_dict['k_contact'] = float(phys.get('k_contact', 1.0))
    config_dict['min_angle'] = float(phys.get('min_angle', 0.0))
    config_dict['cutoff_angle'] = float(phys.get('cutoff_angle', 5.0))
    config_dict['incremental'] = phys.get('incremental', False)
    config_dict['num_load_steps'] = int(phys.get('num_load_steps', 10))

    # BCs & Loads
    bc = data.get('boundary_conditions', {})
    config_dict['bc_clamped'] = bc.get('clamped_faces', "boundary")
    config_dict['loads'] = data.get('loads', [{'face': 'central', 'dof': 1, 'value': -1.0}])

    # Visualization
    vis = data.get('visualization', {})
    config_dict['show_stage0'] = vis.get('show_stage0', data.get('show_stage0', False))
    config_dict['show_stage1'] = vis.get('show_stage1', data.get('show_stage1', False))
    config_dict['show_stage2'] = vis.get('show_stage2', data.get('show_stage2', True))
    config_dict['save_plots'] = vis.get('save_plots', data.get('save_plots', False))
    config_dict['save_animation'] = vis.get('save_animation', data.get('save_animation', True))

    return CentroidalConfig(**config_dict)
