import numpy as np
from .core import Tessellation, UnitPattern
from dataclasses import dataclass


def build_tessellation(pattern: UnitPattern, nx: int, ny: int) -> Tessellation:
    """
    Assemble a unit pattern into a periodic tessellation with specified repetitions and hinge connections.
    """
    tessellation = Tessellation(dim=pattern.vertices.shape[1])
    vec_x = pattern.shift_vectors[0]
    vec_y = pattern.shift_vectors[1]

    num_v_per_cell = pattern.num_vertices()
    num_f_per_cell = pattern.num_faces()
    
    # ---------------------------------------------------------
    # Adding internal hinges
    # ---------------------------------------------------------
    for j in range(ny):
        for i in range(nx):

            translation = np.array([i * vec_x, j * vec_y])
            
            # Add vertices and faces for this cell
            for v in pattern.vertices:
                tessellation.add_vertex(v + translation)
            
            # Calculate the offset for vertex and face indices based on the current cell position
            cell_offset_index = j * nx + i
            vertex_offset = cell_offset_index * num_v_per_cell
            face_offset = cell_offset_index * num_f_per_cell
            
            # Add faces for this cell
            for face_indices in pattern.faces:
                shifted_indices = [idx + vertex_offset for idx in face_indices]
                tessellation.add_face(shifted_indices) #make sure it adds a face

            # Add internal hinges (defined in the unit pattern)
            for h_i in pattern.internal_hinges:
                # Check trigonometric orientation of hinge, else reverse hinge
                if np.cross(tessellation.vertices[h_i['vertex1'] + vertex_offset] - tessellation.vertices[h_i['vertex_adjacent1'] + vertex_offset], tessellation.vertices[h_i['vertex2'] + vertex_offset] - tessellation.vertices[h_i['vertex_adjacent2'] + vertex_offset]) < 0:
                    face1, face2 = face2, face1
                    vertex1, vertex2 = vertex2, vertex1
                    vertex_adjacent1, vertex_adjacent2 = vertex_adjacent2, vertex_adjacent1

                tessellation.add_hinge(
                    face1=h_i['face1'] + face_offset,
                    face2=h_i['face2'] + face_offset,
                    vertex1=h_i['vertex1'] + vertex_offset,
                    vertex2=h_i['vertex2'] + vertex_offset,
                    vertex_adjacent1=h_i['vertex_adjacent1'] + vertex_offset,
                    vertex_adjacent2=h_i['vertex_adjacent2'] + vertex_offset,
                    angle=h_i.get('angle', np.pi/4),
                    properties=h_i.get('properties', {})
                )

    # ---------------------------------------------------------
    # Adding external hinges
    # ---------------------------------------------------------


    for j in range(ny):
        for i in range(nx):
            cell_offset_index = j * nx + i
            vertex_offset = cell_offset_index * num_v_per_cell
            face_offset = cell_offset_index * num_f_per_cell
            
            if j < ny - 1:  # Connect to cell above
                for h_e in pattern.external_hinges:
                    if h_e.get('type') == 'y':  # Only add if it's a vertical hinge
                        face1 = h_e['face1'] + face_offset
                        face2 = h_e['face1'] + face_offset + h_e['face2_offset']
                        vertex1 = h_e['vertex1'] + vertex_offset
                        vertex_adjacent1 = h_e['vertex_adjacent1'] + vertex_offset
                        vertex2 = (vertex_offset + num_v_per_cell * nx) + (h_e['vertex1'] + h_e['vertex2_offset'])
                        vertex_adjacent2 = (vertex_offset + num_v_per_cell * nx) + (h_e['vertex1'] + h_e['vertex_adjacent2_offset'])

                        # Check trigonometric orientation of hinge, else reverse hinge
                        if np.cross(tessellation.vertices[vertex1] - tessellation.vertices[vertex_adjacent1], tessellation.vertices[vertex2] - tessellation.vertices[vertex_adjacent2]) < 0:
                            face1, face2 = face2, face1
                            vertex1, vertex2 = vertex2, vertex1
                            vertex_adjacent1, vertex_adjacent2 = vertex_adjacent2, vertex_adjacent1

                        tessellation.add_hinge(
                            face1=face1,
                            face2=face2,
                            vertex1=vertex1,
                            vertex2=vertex2,
                            vertex_adjacent1=vertex_adjacent1,
                            vertex_adjacent2=vertex_adjacent2,
                            angle=h_e.get('angle', np.pi/4),
                            properties=h_e.get('properties', {})
                        )

            if i < nx - 1:  # Connect to cell to the right
                for h_e in pattern.external_hinges:
                    if h_e.get('type') == 'x':  # Only add if it's a horizontal hinge
                        face1 = h_e['face1'] + face_offset
                        face2 = h_e['face1'] + face_offset + h_e['face2_offset']
                        vertex1 = h_e['vertex1'] + vertex_offset
                        vertex_adjacent1 = h_e['vertex_adjacent1'] + vertex_offset
                        vertex2 = (vertex_offset + num_v_per_cell) + (h_e['vertex1'] + h_e['vertex2_offset'])
                        vertex_adjacent2 = (vertex_offset + num_v_per_cell) + (h_e['vertex1'] + h_e['vertex_adjacent2_offset'])

                        # Check trigonometric orientation of hinge, else reverse hinge
                        if np.cross(tessellation.vertices[vertex1] - tessellation.vertices[vertex_adjacent1], tessellation.vertices[vertex2] - tessellation.vertices[vertex_adjacent2]) < 0:
                            face1, face2 = face2, face1
                            vertex1, vertex2 = vertex2, vertex1
                            vertex_adjacent1, vertex_adjacent2 = vertex_adjacent2, vertex_adjacent1

                        tessellation.add_hinge(
                            face1=face1,
                            face2=face2,
                            vertex1=vertex1,
                            vertex2=vertex2,
                            vertex_adjacent1=vertex_adjacent1,
                            vertex_adjacent2=vertex_adjacent2,
                            angle=h_e.get('angle', np.pi/4),
                            properties=h_e.get('properties', {})
                        )

    # ---------------------------------------------------------
    # Identify voids based on pairs of opposite hinges (assuming RDQK-like patterns)
    # ---------------------------------------------------------
    
    unvisited_hinges = set(range(len(tessellation.hinges)))
    primary_to_hinges = tessellation.build_primary_to_hinges()
    adjacents_to_hinge = tessellation.build_adjacents_to_hinge()

    def find_voids_recursive(h_id, unvisited, discovered_voids):
        if h_id not in unvisited:
            return discovered_voids
        
        unvisited.remove(h_id)
        h0 = tessellation.hinges[h_id]
        a1, a2 = h0.vertex_adjacent1, h0.vertex_adjacent2
        
        for hs1_id in primary_to_hinges.get(a1, []):
            if hs1_id == h_id: continue
            hs1 = tessellation.hinges[hs1_id]
            p1 = hs1.vertex2 if hs1.vertex1 == a1 else hs1.vertex1
            
            for hs2_id in primary_to_hinges.get(a2, []):
                if hs2_id == h_id: continue
                hs2 = tessellation.hinges[hs2_id]
                p2 = hs2.vertex2 if hs2.vertex1 == a2 else hs2.vertex1
                
                target_pair = frozenset([p1, p2])
                if target_pair in adjacents_to_hinge:
                    h_opp_id = adjacents_to_hinge[target_pair]
                    if h_opp_id != h_id:

                        void_sig = tuple(sorted([h_id, h_opp_id]))
                        if void_sig not in discovered_voids:
                            discovered_voids.add(void_sig)
                            tessellation.add_void(h_id, h_opp_id)   
                        break

        for v in [h0.vertex1, h0.vertex2, h0.vertex_adjacent1, h0.vertex_adjacent2]:
            for next_h_id in primary_to_hinges.get(v, []):
                find_voids_recursive(next_h_id, unvisited, discovered_voids)
        
        return discovered_voids

    # Launching the search (looping to handle disconnected components)
    all_voids = set()
    while unvisited_hinges:
        start_id = next(iter(unvisited_hinges))
        find_voids_recursive(start_id, unvisited_hinges, all_voids)
    # ---------------------------------------------------------
    # Extract border edges from the global tessellation
    # ---------------------------------------------------------
    if hasattr(pattern, 'border_edges'):
        global_borders = {'bottom': [], 'top': [], 'left': [], 'right': []}
        for j in range(ny):
            for i in range(nx):
                cell_offset_index = j * nx + i
                vertex_offset = cell_offset_index * num_v_per_cell
                
                # Bottom border (j == 0)
                if j == 0 and 'bottom' in pattern.border_edges:
                    for edge in pattern.border_edges['bottom']:
                        global_borders['bottom'].append([edge[0] + vertex_offset, edge[1] + vertex_offset])
                
                # Top border (j == ny - 1)
                if j == ny - 1 and 'top' in pattern.border_edges:
                    for edge in pattern.border_edges['top']:
                        global_borders['top'].append([edge[0] + vertex_offset, edge[1] + vertex_offset])
                        
                # Left border (i == 0)
                if i == 0 and 'left' in pattern.border_edges:
                    for edge in pattern.border_edges['left']:
                        global_borders['left'].append([edge[0] + vertex_offset, edge[1] + vertex_offset])
                        
                # Right border (i == nx - 1)
                if i == nx - 1 and 'right' in pattern.border_edges:
                    for edge in pattern.border_edges['right']:
                        global_borders['right'].append([edge[0] + vertex_offset, edge[1] + vertex_offset])
                        
        tessellation.border_edges = global_borders

    return tessellation