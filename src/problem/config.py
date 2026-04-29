from dataclasses import dataclass, field
from typing import Tuple, Callable, Any
from problem.targets import DEFAULT_TARGET
from unit_patterns import unit_RDQK_D

@dataclass
class CentroidalConfig:
    # Tessellation
    width: int
    height: int
    pattern: Callable

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

    # Boundary Conditions & Loading
    # bc_clamped can be "boundary" or a list of IDs
    bc_clamped: Any
    # loads is a list of dicts: [{'face': 'central', 'dof': 1, 'value': -1.0}, ...]
    loads: list
