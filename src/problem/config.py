from dataclasses import dataclass, field
from typing import Tuple, Callable, Any
from problem.targets import DEFAULT_TARGET
from unit_patterns import unit_RDQK_D

@dataclass
class CentroidalConfig:
    # Tessellation
    width: int = 2
    height: int = 2
    pattern: Callable = unit_RDQK_D

    # Target shape
    target_type: str = DEFAULT_TARGET['type']
    target_center: Tuple[float, float] = (0.0, 0.0)
    target_radius: float = DEFAULT_TARGET['radius']

    # Material properties
    k_stretch: float = 10.0
    k_shear: float = 5.0
    k_rot: float = 1.0
    density: float = 1.0

    # Initial mapping
    map_type: str = 'elliptical_grip'
    scale_factor: float = 1.0

    # Geometric validity weights
    w_connectivity: float = 700.0
    w_non_intersection: float = 1000.0
    w_target: float = 1.0
    w_arm_symmetry: float = 1.0
    w_void_length: float = 1000.0
    w_void_collinear: float = 1000.0

    # Physics
    use_contact: bool = True
    linearized_strains: bool = True
    k_contact: float = 1.0
    min_angle: float = 0.0 # degrees
    cutoff_angle: float = 5.0 # degrees

    # Boundary Conditions & Loading
    # bc_clamped can be "boundary" or a list of IDs
    bc_clamped: Any = "boundary"
    # loads is a list of dicts: [{'face': 'central', 'dof': 1, 'value': -1.0}, ...]
    loads: list = field(default_factory=lambda: [{'face': 'central', 'dof': 1, 'value': -1.0}])
