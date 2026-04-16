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
                tessellation.add_hinge(
                    face1=h_i['face1'] + face_offset,
                    face2=h_i['face2'] + face_offset,
                    vertex1=h_i['vertex1'] + vertex_offset,
                    vertex2=h_i['vertex2'] + vertex_offset,
                    vertex_adjacent1=h_i['vertex_adjacent1'] + vertex_offset,
                    vertex_adjacent2=h_i['vertex_adjacent2'] + vertex_offset,
                    angle=h_i.get('angle', 0.0),
                    stiffness=h_i.get('stiffness', 1.0)
                )

    # Add external hinges to connect this cell to its neighbors
    for j in range(ny):
        for i in range(nx):
            cell_offset_index = j * nx + i
            vertex_offset = cell_offset_index * num_v_per_cell
            face_offset = cell_offset_index * num_f_per_cell
            
            if j < ny - 1:  # Connect to cell above
                for h_e in pattern.external_hinges:
                    if h_e.get('type') == 'y':  # Only add if it's a vertical hinge
                        face1 = h_e['face1'] + face_offset
                        face_offset_adj = face_offset + num_f_per_cell * nx
                        face2 = face_offset_adj + h_e['face2_offset']
                        vertex1 = h_e['vertex1'] + vertex_offset
                        vertex_adjacent1 = h_e['vertex_adjacent1'] + vertex_offset
                        vertex2 = (vertex_offset + num_v_per_cell * nx) + (h_e['vertex1'] + h_e['vertex2_offset'])
                        vertex_adjacent2 = (vertex_offset + num_v_per_cell * nx) + (h_e['vertex1'] + h_e['vertex_adjacent2_offset'])

                        tessellation.add_hinge(
                            face1=face1,
                            face2=face2,
                            vertex1=vertex1,
                            vertex2=vertex2,
                            vertex_adjacent1=vertex_adjacent1,
                            vertex_adjacent2=vertex_adjacent2,
                            angle=h_e.get('angle', 0.0),
                            stiffness=h_e.get('stiffness', 1.0)
                        )

            if i < nx - 1:  # Connect to cell to the right
                for h_e in pattern.external_hinges:
                    if h_e.get('type') == 'x':  # Only add if it's a horizontal hinge
                        face1 = h_e['face1'] + face_offset
                        face_offset_adj = face_offset + num_f_per_cell
                        face2 = face_offset_adj + h_e['face2_offset']
                        vertex1 = h_e['vertex1'] + vertex_offset
                        vertex_adjacent1 = h_e['vertex_adjacent1'] + vertex_offset
                        vertex2 = (vertex_offset + num_v_per_cell) + (h_e['vertex1'] + h_e['vertex2_offset'])
                        vertex_adjacent2 = (vertex_offset + num_v_per_cell) + (h_e['vertex1'] + h_e['vertex_adjacent2_offset'])

                        tessellation.add_hinge(
                            face1=face1,
                            face2=face2,
                            vertex1=vertex1,
                            vertex2=vertex2,
                            vertex_adjacent1=vertex_adjacent1,
                            vertex_adjacent2=vertex_adjacent2,
                            angle=h_e.get('angle', 0.0),
                            stiffness=h_e.get('stiffness', 1.0)
                        )

    return tessellation




