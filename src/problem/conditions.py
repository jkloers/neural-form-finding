import numpy as np

def apply_boundary_conditions(tessellation, config):
    """Applies boundary conditions (clamped faces) based on config.
    
    config.bc_clamped can be:
        - "boundary": all faces on the boundary are clamped.
        - list of int: specific face IDs to clamp.
        - None/False: no faces clamped.
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
    """Applies external loads based on config.
    
    config.loads is a list of dicts:
        [{'face': 'central' or ID, 'dof': 0/1/2, 'value': float}, ...]
    """
    applied_loads = []
    
    for load_info in config.loads:
        face_spec = load_info.get('face')
        dof = load_info.get('dof', 1)
        value = load_info.get('value', 0.0)
        
        target_face = None
        
        if face_spec == "central":
            # Heuristic for central interior face
            clamped_ids = tessellation.get_boundary_face_ids()
            all_ids = set(range(len(tessellation.faces)))
            interior_ids = sorted(all_ids - set(clamped_ids))
            if interior_ids:
                target_face = interior_ids[len(interior_ids) // 2]
        elif isinstance(face_spec, int):
            target_face = face_spec
            
        if target_face is not None:
            tessellation.set_face_load(target_face, dof_id=dof, value=value)
            applied_loads.append((target_face, dof, value))
            
    return applied_loads

def set_material_properties(tessellation, config):
    """Sets material properties for all hinges and faces from config."""
    tessellation.set_hinge_properties(
        k_stretch=config.k_stretch,
        k_shear=config.k_shear,
        k_rot=config.k_rot
    )
    tessellation.set_all_faces_properties(density=config.density)

# Deprecated / Legacy helpers (kept for compatibility if needed temporarily)
def apply_standard_boundary_conditions(tessellation, clamp_boundary=True):
    if clamp_boundary: return tessellation.clamp_boundary_faces()
    return []

def apply_central_load(tessellation, force_value=-1.0, dof_id=1):
    clamped_ids = tessellation.get_boundary_face_ids()
    interior_ids = sorted(set(range(len(tessellation.faces))) - set(clamped_ids))
    if not interior_ids: return None
    loaded_face = interior_ids[len(interior_ids) // 2]
    tessellation.set_face_load(loaded_face, dof_id=dof_id, value=force_value)
    return loaded_face

def apply_face_moment(tessellation, face_id, moment_value):
    tessellation.set_face_load(face_id, dof_id=2, value=moment_value)
