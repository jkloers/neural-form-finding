"""
Problem-level configuration for a tessellation.

This module handles the "physical setup" stage: reading a YAML config and
applying boundary conditions, external loads, and material properties to a
Tessellation object *before* it is exported to a CentroidalState.

This is distinct from physics_solver/loading.py, which operates at a lower
level and builds JAX-differentiable loading functions for the solver itself.

Architecture:
    config (YAML)
        ↓ configure_tessellation()
    Tessellation  (NumPy, mutated in-place)
        ↓ to_centroidal_state()
    CentroidalState → JAX physics solver
"""

import numpy as np


# ── Individual setters ────────────────────────────────────────────────────────

def set_material_properties(tessellation, config):
    """Sets stiffness and density for all hinges and faces from config."""
    tessellation.set_hinge_properties(
        k_stretch=config.k_stretch,
        k_shear=config.k_shear,
        k_rot=config.k_rot,
    )
    tessellation.set_all_faces_properties(density=config.density)


def apply_boundary_conditions(tessellation, config):
    """Applies Dirichlet boundary conditions (clamped faces) from config.

    config.bc_clamped can be:
        - "boundary": clamp all faces on the tessellation boundary.
        - list[int]: clamp only the specified face IDs.
        - None / False: no constraints.
    """
    mode = config.bc_clamped

    if mode == "boundary":
        clamped_ids = tessellation.clamp_boundary_faces()
    elif isinstance(mode, list):
        clamped_ids = mode
        for fid in clamped_ids:
            tessellation.set_face_dofs(fid, [0, 1, 2])
    else:
        clamped_ids = []

    return clamped_ids


def apply_loads(tessellation, config):
    """Applies Neumann loads (external forces) from config.

    config.loads is a list of dicts:
        [{'face': 'central' | int | list[int], 'dof': 0/1/2, 'value': float}, ...]
    """
    applied_loads = []

    for load_info in config.loads:
        face_spec = load_info.get('face')
        dof = load_info.get('dof', 1)
        value = load_info.get('value', 0.0)

        if face_spec == "central":
            # Apply to the middle interior face (heuristic)
            boundary_ids = set(tessellation.get_boundary_face_ids())
            interior_ids = sorted(set(range(len(tessellation.faces))) - boundary_ids)
            target_faces = [interior_ids[len(interior_ids) // 2]] if interior_ids else []
        elif isinstance(face_spec, list):
            target_faces = face_spec
        elif isinstance(face_spec, int):
            target_faces = [face_spec]
        else:
            target_faces = []

        for fid in target_faces:
            tessellation.set_face_load(fid, dof_id=dof, value=value)
            applied_loads.append((fid, dof, value))

    return applied_loads


# ── Unified entry point ───────────────────────────────────────────────────────

def configure_tessellation(tessellation, config):
    """Full physical setup of a tessellation from a config object.

    Applies material properties, boundary conditions, and external loads in
    the correct order. This is the single function to call before exporting
    to a CentroidalState.
    """
    set_material_properties(tessellation, config)
    apply_boundary_conditions(tessellation, config)
    apply_loads(tessellation, config)
