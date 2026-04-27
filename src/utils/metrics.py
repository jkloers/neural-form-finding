def compute_tessellation_differences(tess1, tess2):
    """
    Computes the difference in area and height-to-width ratio for each face
    between two tessellations that share the same topology but have different vertex positions.
    
    Returns:
        diff_areas (list of float): The absolute difference in area for each face.
        diff_ratios (list of float): The absolute difference in height-to-width ratio for each face.
    """
    if len(tess1.faces) != len(tess2.faces):
        raise ValueError("Tessellations must have the same number of faces.")
        
    diff_areas = []
    diff_ratios = []
    
    for i in range(len(tess1.faces)):
        area1 = tess1.get_area(i)
        ratio1 = tess1.compute_ratio(i)
        
        area2 = tess2.get_area(i)
        ratio2 = tess2.compute_ratio(i)
        
        pct_area = (abs(area1 - area2) / area1 * 100.0) if area1 > 1e-9 else 0.0
        pct_ratio = (abs(ratio1 - ratio2) / ratio1 * 100.0) if ratio1 > 1e-9 and ratio1 != float('inf') else 0.0
        
        diff_areas.append(pct_area)
        diff_ratios.append(pct_ratio)
        
    return diff_areas, diff_ratios
