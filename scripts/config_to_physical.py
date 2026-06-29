"""
nff/sofa/config_to_physical.py
================================
Derive physical geometry and stiffness values from a sofa config dict.

All physical quantities are in SI (metres, Pa, N, N·m).

Usage
-----
    import yaml
    from nff.sofa.config_to_physical import physical_scale_from_config

    with open("data/configs/sofa/moment_1x1.yaml") as f:
        raw = yaml.safe_load(f)

    p = physical_scale_from_config(raw)
    print(p.face_size, p.k_rot)   # 0.1 m,  0.787 N·m/rad
"""

from __future__ import annotations

import math
from typing import NamedTuple


class PhysicalParams(NamedTuple):
    """Derived physical quantities for a sofa config."""
    # Geometry (metres)
    face_size:        float   # side of each square face panel
    arm_width:        float   # gap between face panels (hinge span)
    fold_length:      float   # thin hinge dimension (bending occurs here)
    sheet_thickness:  float   # z-height of 3D FEM mesh

    # JAX coordinate scale
    jax_scale:        float   # metres per JAX normalised unit

    # Stiffnesses derived from material + geometry
    k_rot:            float   # N·m/rad — rotational hinge stiffness (physical)
    k_stretch:        float   # JAX normalised — rigid-face in-plane stretch
    k_shear:          float   # JAX normalised — rigid-face in-plane shear

    # Material
    young_modulus:    float   # Pa
    poisson_ratio:    float
    yield_strength:   float   # Pa
    sheet_thickness_m: float  # m  (alias kept for SOFA callers)


def physical_scale_from_config(raw: dict) -> PhysicalParams:
    """Compute physical dimensions and stiffnesses from a raw YAML config dict.

    Scaling formula
    ---------------
    jax_scale   = (target_diameter_m / 2) / target.radius
    face_size   = jax_scale × sqrt(total_area / n_faces)
    arm_width   = arm_width_frac  × face_size
    fold_length = fold_length_frac × face_size

    k_rot = E × t × fold_length³ / (12 × arm_width)   [in-plane bending of hinge strip]
    """
    sofa_raw  = raw.get('sofa', {})
    mat_raw   = raw.get('material', {})
    target    = raw.get('target', {})
    tess      = raw.get('tessellation', {})

    # Material
    E    = float(mat_raw.get('young_modulus',  3.5e9))
    nu   = float(mat_raw.get('poisson_ratio',  0.36))
    sy   = float(mat_raw.get('yield_strength', 50.0e6))
    t    = float(sofa_raw.get('sheet_thickness_m', 0.001))

    # Geometry fractions
    arm_frac  = float(sofa_raw.get('arm_width_frac',   0.10))
    fold_frac = float(sofa_raw.get('fold_length_frac', 0.03))

    # Physical scale from target diameter
    target_diam   = float(sofa_raw.get('target_diameter_m', 0.20))
    target_r_norm = float(target.get('radius', 1.0))
    total_area    = float(tess.get('total_area', 4.0))
    nx            = int(tess.get('width',  1))
    ny            = int(tess.get('height', 1))
    n_faces       = nx * ny * 4   # 4 faces per unit_RDQK cell

    jax_scale   = (target_diam / 2.0) / target_r_norm
    face_size   = jax_scale * math.sqrt(total_area / n_faces)
    arm_width   = arm_frac  * face_size
    fold_length = fold_frac * face_size

    # Rotational stiffness: in-plane bending of thin corner strip about z
    # Strip cross-section: arm_width (x) × fold_length (y) × sheet_thickness (z)
    # For in-plane bending: I_z = arm_width × fold_length³ / 12
    # k_rot = E × I_z / arm_width  (cantilever beam formula, length = arm_width)
    k_rot = E * t * fold_length ** 3 / (12.0 * arm_width)

    # JAX normalised stiffnesses (large → rigid face panels; not physically calibrated)
    k_stretch = float(mat_raw.get('k_stretch', 1000.0))
    k_shear   = float(mat_raw.get('k_shear',   1000.0))

    return PhysicalParams(
        face_size        = face_size,
        arm_width        = arm_width,
        fold_length      = fold_length,
        sheet_thickness  = t,
        jax_scale        = jax_scale,
        k_rot            = k_rot,
        k_stretch        = k_stretch,
        k_shear          = k_shear,
        young_modulus    = E,
        poisson_ratio    = nu,
        yield_strength   = sy,
        sheet_thickness_m = t,
    )


def loads_from_config(raw: dict) -> list[dict]:
    """Return the loads list from the config, unmodified."""
    return raw.get('loads', [])


def clamped_faces_from_config(raw: dict) -> list[int]:
    """Return the list of clamped face indices from boundary_conditions."""
    bc = raw.get('boundary_conditions', {})
    return list(bc.get('clamped_faces', []))
