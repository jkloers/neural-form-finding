import numpy as np

def atan2_angle(center, p1, p2):
    """
    Computes the signed angle from vector (p1 - center) to vector (p2 - center).
    p1 and p2 are expected to be 1D arrays or lists like [x, y].
    """
    if np.isscalar(center) and center == 0:
        center = np.array([0.0, 0.0])
    
    v1 = np.asarray(p1).flatten() - center
    v2 = np.asarray(p2).flatten() - center
    
    ang1 = np.arctan2(v1[1], v1[0])
    ang2 = np.arctan2(v2[1], v2[0])
    
    # Angle in range [-pi, pi]
    angle = ang2 - ang1
    # Adjust to keep within [-pi, pi] if needed
    if angle > np.pi:
        angle -= 2 * np.pi
    elif angle < -np.pi:
        angle += 2 * np.pi
        
    return angle

def twoD_rotation(theta):
    """Returns a 2D rotation matrix for a given angle theta."""
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])

def get_contracted_shape(solved_pointsD, face_setsD, Dto0):
    """
    Get the contracted shape of a deployed pattern.
    Works for quad and hexagon kirigami tessellations.
    Idea:
    - start from any face
    - find a common edge with another face
    - glue and then remove it from the face list
    - continue until all faces are glued
    
    Note: Assumes inputs (face_setsD, Dto0) use 0-based indexing.
    """
    # initialize pattern points
    solved_points0 = np.zeros((np.max(Dto0) + 1, 2))
    
    # Assuming face_setsD is a list of arrays. face_setsD[0] corresponds to face_setsD{1} in MATLAB.
    faces = face_setsD[0]
    num_faces, num_vertices_per_face = faces.shape
    
    # start from the central face
    face_centroid = np.zeros((num_faces, 2))
    for j in range(num_faces):
        # average over vertices for each face
        face_centroid[j, :] = np.sum(solved_pointsD[faces[j, :], :], axis=0) / num_vertices_per_face
        
    # Find face closest to centroid of all face centroids
    mean_centroid = np.mean(face_centroid, axis=0)
    dist_sq = np.sum((face_centroid - mean_centroid)**2, axis=1)
    faceid1 = np.argmin(dist_sq)
    
    face1 = faces[faceid1, :]
    
    solved_points0[Dto0[face1], :] = solved_pointsD[face1, :]
    
    current_indices0 = list(np.unique(Dto0[face1]))
    current_indicesD = list(np.unique(face1))
    current_faces = [faceid1]
    
    # form edges from faces (using 0-based indexing)
    # in matlab: [reshape(face', ...), reshape(face(:,[2:end, 1])', ...)]
    edges = []
    for j in range(num_faces):
        for k in range(num_vertices_per_face):
            v1 = faces[j, k]
            v2 = faces[j, (k + 1) % num_vertices_per_face]
            edges.append([v1, v2])
    edges = np.array(edges)
    
    # count occurrence in Dto0
    unique_vals, counts = np.unique(Dto0, return_counts=True)
    split_points = list(unique_vals[counts > 1])
    
    face_handled = [faceid1]
    
    while len(split_points) > 0:
        # find points that are both split points and currently in our glued set
        points_available_for_gluing = [p for p in split_points if p in current_indices0]
        if not points_available_for_gluing:
            break # Just in case it gets stuck
            
        p0 = points_available_for_gluing[0]
        # Find exactly where p0 is in Dto0
        pD_all = np.where(Dto0 == p0)[0]
        
        # pD_old is the one we already have handled
        pD_old_candidates = [p for p in pD_all if p in current_indicesD]
        pD_old = pD_old_candidates[0]
        
        # pD_new is the one we haven't glued yet
        pD_new_candidates = [p for p in pD_all if p != pD_old]
        pD_new = pD_new_candidates[0]
        
        check = True
        
        # Check for some index flip
        while check:
            # find the common edge and hence the common point
            # row1: edges containing pD(1) which here are [pD_old, pD_new]
            # Since MATLAB used pD(1) and pD(2), we use pD_old and pD_new
            pD_pair = [pD_old, pD_new]
            
            row1 = np.where((edges == pD_pair[0]).any(axis=1))[0]
            row2 = np.where((edges == pD_pair[1]).any(axis=1))[0]
            
            common_edges = np.intersect1d(edges[row1, :].flatten(), edges[row2, :].flatten())
            # Exclude the current point we are trying to glue if it happens to be present? 
            # In MATLAB: fixed_pointD = intersect(edges(row1,:), edges(row2,:));
            fixed_pointD = common_edges[0]
            
            # find the old face containing the split point and the common point
            faceid_candidates = np.where((faces == fixed_pointD).any(axis=1))[0]
            faceid1_cands = np.intersect1d(faceid_candidates, current_faces)
            
            if len(faceid1_cands) == 0:
                # handle some exceptional case
                if len(common_edges) > 1:
                    fixed_pointD = common_edges[1]
                faceid_candidates = np.where((faces == fixed_pointD).any(axis=1))[0]
                faceid1_cands = np.intersect1d(faceid_candidates, current_faces)
                
            faceid_old = faceid1_cands[0]
            
            # find the new face containing the split point and the common point
            faceid_new_cands = np.setdiff1d(faceid_candidates, faceid1_cands)
            faceid_new = faceid_new_cands[0]
            
            face1 = faces[faceid_old, :]
            face2 = faces[faceid_new, :]
            
            # check flip
            if pD_new not in face2 or not np.any(Dto0[face1] == Dto0[pD_old]):
                # flipped pD_old and pD_new
                pD_old, pD_new = pD_new, pD_old
            else:
                check = False
                
        # Translation vector to match the common point
        translation_vector = solved_points0[Dto0[fixed_pointD], :] - solved_pointsD[fixed_pointD, :]
        points0_face2 = solved_pointsD[face2, :] + translation_vector
        
        # Rotate with respect to the common point to match the edge
        shift_to_origin = solved_points0[Dto0[fixed_pointD], :]
        points0_face1 = solved_points0[Dto0[face1], :] - shift_to_origin
        points0_face2 = points0_face2 - shift_to_origin
        
        # Rotate 
        mask_old = (Dto0[face1] == Dto0[pD_old])
        mask_new = (face2 == pD_new)
        
        rot_angle = -atan2_angle(0.0, points0_face1[mask_old, :], points0_face2[mask_new, :])
        R = twoD_rotation(rot_angle)
        
        points0_face2 = (R @ points0_face2.T).T
        points0_face2 += shift_to_origin
        
        # Merge operation / adding points
        count = np.ones(solved_points0.shape[0])
        for t in range(len(face2)):
            idx_0 = Dto0[face2[t]]
            if solved_points0[idx_0, 0] == 0.0 and solved_points0[idx_0, 1] == 0.0:
                # new point, can safely set
                solved_points0[idx_0, :] = points0_face2[t, :]
            else:
                # avoid small discrepancy by taking average
                solved_points0[idx_0, :] = (solved_points0[idx_0, :] * count[idx_0] + points0_face2[t, :]) / (count[idx_0] + 1)
                count[idx_0] += 1
                
        new_indices0 = list(np.unique(Dto0[face2]))
        new_indicesD = list(np.unique(face2))
        
        current_indices0 = list(np.unique(current_indices0 + new_indices0))
        current_indicesD = list(np.unique(current_indicesD + new_indicesD))
        current_faces.append(faceid_new)
        
        face_handled.append(faceid_new)
        split_points.remove(p0)
        
    return solved_points0
