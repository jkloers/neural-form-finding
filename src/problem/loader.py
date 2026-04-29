import yaml
import os
from problem.config import CentroidalConfig
from unit_patterns import unit_RDQK_D, unit_RDQK_0

# Map string names from YAML to actual function objects
PATTERN_MAP = {
    "unit_RDQK_D": unit_RDQK_D,
    "unit_RDQK_0": unit_RDQK_0,
}

def load_config(yaml_path: str) -> CentroidalConfig:
    """Loads a CentroidalConfig from a YAML file.
    
    If the file doesn't exist, returns default configuration.
    """
    if not os.path.exists(yaml_path):
        print(f"Warning: Configuration file {yaml_path} not found. Using defaults.")
        return CentroidalConfig()

    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    # Flatten nested YAML structure to match flat dataclass if needed
    # Or map sections one by one
    config_dict = {}
    
    # Tessellation
    tess = data.get('tessellation', {})
    config_dict['width'] = tess.get('width', 2)
    config_dict['height'] = tess.get('height', 2)
    pattern_name = tess.get('pattern', "unit_RDQK_D")
    config_dict['pattern'] = PATTERN_MAP.get(pattern_name, unit_RDQK_D)

    # Target
    target = data.get('target', {})
    config_dict['target_type'] = target.get('type', 'circle')
    config_dict['target_center'] = tuple(target.get('center', [0.0, 0.0]))
    config_dict['target_radius'] = target.get('radius', 1.0)

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

    # BCs & Loads
    bc = data.get('boundary_conditions', {})
    config_dict['bc_clamped'] = bc.get('clamped_faces', "boundary")
    config_dict['loads'] = data.get('loads', [])

    return CentroidalConfig(**config_dict)
