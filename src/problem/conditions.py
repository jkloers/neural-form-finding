import numpy as np

def apply_standard_boundary_conditions(tessellation, clamp_boundary=True):
    """Applies common boundary conditions to a tessellation.
    
    Args:
        tessellation: Tessellation object to modify.
        clamp_boundary: If True, all boundary faces will be clamped (all DOFs fixed).
        
    Returns:
        list: IDs of clamped faces.
    """
    clamped_ids = []
    if clamp_boundary:
        clamped_ids = tessellation.clamp_boundary_faces()
    
    return clamped_ids

def apply_central_load(tessellation, force_value=-1.0, dof_id=1):
    """Applies a point load to the most central interior face.
    
    Args:
        tessellation: Tessellation object to modify.
        force_value: Magnitude of the force.
        dof_id: DOF index (0=x, 1=y, 2=theta).
        
    Returns:
        int: ID of the loaded face, or None if no interior face found.
    """
    clamped_ids = tessellation.get_boundary_face_ids()
    all_ids = set(range(len(tessellation.faces)))
    interior_ids = sorted(all_ids - set(clamped_ids))
    
    if not interior_ids:
        return None
        
    # Simple heuristic: pick the middle one in the list of interior IDs
    loaded_face = interior_ids[len(interior_ids) // 2]
    tessellation.set_face_load(loaded_face, dof_id=dof_id, value=force_value)
    
    return loaded_face

def set_material_properties(tessellation, k_stretch=10.0, k_shear=5.0, k_rot=1.0, density=1.0):
    """Sets material properties for all hinges and faces."""
    tessellation.set_hinge_properties(
        k_stretch=k_stretch,
        k_shear=k_shear,
        k_rot=k_rot
    )
    tessellation.set_all_faces_properties(density=density)
