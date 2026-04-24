import numpy as np
from collections import defaultdict
from dataclasses import dataclass

@dataclass
class IndexedFace:
    """A face represented by global vertex indices."""
    vertex_indices: np.ndarray
    id: any = None
    
    def centroid(self, vertices):
        return np.mean(vertices[self.vertex_indices], axis=0)

    def area(self, vertices):
        coords = vertices[self.vertex_indices]
        if coords.shape[1] == 2:
            x = coords[:, 0]
            y = coords[:, 1]
            return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        if coords.shape[1] == 3:
            x = coords[:, 0]
            y = coords[:, 1]
            return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        raise ValueError("Face must be defined in 2D or 3D")

    def __repr__(self):
        return f"IndexedFace(id={self.id}, vertices={self.vertex_indices.tolist()})"

class Hinge:
    """A hinge defined by two faces and two vertex indices."""

    def __init__(self, face1: int, face2: int, vertex1: int, vertex2: int, vertex_adjacent1: int, vertex_adjacent2: int, angle: float = 0.0, properties: dict = None, id: any = None):
        self.face1 = face1
        self.face2 = face2
        self.vertex1 = vertex1
        self.vertex2 = vertex2
        self.vertex_adjacent1 = vertex_adjacent1 # The vertex adjacent to vertex1 in face1 along the hinge closing edge
        self.vertex_adjacent2 = vertex_adjacent2 # The vertex adjacent to vertex2 in face2 along the hinge closing edge
        self.angle = angle
        self.properties = properties if properties is not None else {}
        self.id = id

    def __repr__(self):
        return (
            f"Hinge(id={self.id}, face1={self.face1}, face2={self.face2}, "
            f"vertex1={self.vertex1}, vertex2={self.vertex2}, "
            f"vertex_adj1={self.vertex_adjacent1}, vertex_adj2={self.vertex_adjacent2}, "
            f"angle={self.angle}, properties={self.properties})"
        )
    

class UnitPattern:
    def __init__(self, vertices, faces, internal_hinges, external_hinges, shift_vectors=None, border_edges=None):
        """
        Defines a unit pattern for tessellation, including vertices, faces, internal hinges, and optional shift vectors for tiling.
        """
        self.vertices = np.asarray(vertices, dtype=float)
        self.faces = faces
        self.internal_hinges = internal_hinges
        if shift_vectors is None:
            self.shift_vectors = self._get_shift_vectors()
        else:
            self.shift_vectors = np.asarray(shift_vectors, dtype=float)
        self.external_hinges = external_hinges  # Connections between unit cells
        self.border_edges = border_edges if border_edges is not None else {}

    def _get_shift_vectors(self):
        """Calculate shift vectors based on the bounding box of the vertices, assuming a rectangular tiling pattern."""
        vec_x = max(self.vertices[:, 0]) - min(self.vertices[:, 0])
        vec_y = max(self.vertices[:, 1]) - min(self.vertices[:, 1])
        return np.array([vec_x, vec_y])
    
    def num_vertices(self):
        return self.vertices.shape[0]
    
    def num_faces(self):
        return len(self.faces)



class Tessellation:
    """Indexed tessellation builder for JAX-ready kirigami topology."""

    def __init__(self, vertices=None, faces=None, hinges=None, voids=None, border_edges=None, dim=2, tolerance=1e-6):
        if vertices is None:
            self.vertices = np.zeros((0, dim), dtype=float)
        else:
            self.vertices = np.asarray(vertices, dtype=float)
            if self.vertices.ndim == 1:
                self.vertices = self.vertices.reshape(1, -1)
            if self.vertices.shape[1] != dim:
                raise ValueError("vertices must have shape (N, dim)")
        self.faces = []
        self.hinges = []
        self.tolerance = float(tolerance)
        self.voids = []  # List of opposite hinge pair indices that define voids in the tessellation (assuming RDQK-like patterns)
        if voids is not None:
            self.voids = list(voids)
        self.border_edges = border_edges if border_edges is not None else {}

        if faces is not None:
            for face in faces:
                self.add_face(face)

        if hinges is not None:
            for hinge in hinges:
                self.add_hinge(
                    face1=hinge.face1,
                    face2=hinge.face2,
                    vertex1=hinge.vertex1,
                    vertex2=hinge.vertex2,
                    vertex_adjacent1=hinge.vertex_adjacent1,
                    vertex_adjacent2=hinge.vertex_adjacent2,
                    angle=hinge.angle,
                    properties=hinge.properties,
                    id=hinge.id,
                )
    
    def copy(self):
        """Create a deep copy of the tessellation."""
        new_tess = Tessellation(
            vertices=self.vertices.copy(),
            faces=[IndexedFace(f.vertex_indices.copy(), f.id) for f in self.faces],
            hinges=[
                Hinge(
                    h.face1, h.face2, h.vertex1, h.vertex2,
                    h.vertex_adjacent1, h.vertex_adjacent2,
                    h.angle, h.properties.copy(), h.id
                ) for h in self.hinges
            ],
            voids=self.voids.copy(),
            border_edges={k: [e.copy() for e in v] for k, v in self.border_edges.items()},
            tolerance=self.tolerance
        )
        return new_tess
    
    def update_vertices(self, new_vertices):
        """Update the vertex positions in the tessellation."""
        self.vertices = np.asarray(new_vertices, dtype=float)

    def add_vertex(self, vertex):
        self.vertices = np.vstack([self.vertices, vertex])

    def add_face(self, vertex_indices, id=None):
        if isinstance(vertex_indices, IndexedFace):
            face = vertex_indices
        else:
            face = IndexedFace(vertex_indices, id=id)
        self.faces.append(face)
        return face

    def add_hinge(self, face1, face2, vertex1, vertex2, vertex_adjacent1, vertex_adjacent2, angle=0.0, properties=None, id=None):
        if id is None:
            id = len(self.hinges)
        hinge = Hinge(face1, face2, vertex1, vertex2, vertex_adjacent1, vertex_adjacent2, angle, properties, id)
        self.hinges.append(hinge)
        return hinge

    def build_primary_to_hinges(self):
        """Build a map from primary vertices to their incident hinges."""
        primary_to_hinges = {}
        for hinge in self.hinges:
            primary_to_hinges.setdefault(hinge.vertex1, []).append(hinge.id)
            primary_to_hinges.setdefault(hinge.vertex2, []).append(hinge.id)
        return primary_to_hinges

    def build_adjacent_to_hinge(self):
        """Build a map from hinge side vertices to the hinge id."""
        adjacent_to_hinge = {}
        for hinge in self.hinges:
            adjacent_to_hinge[hinge.vertex_adjacent1] = hinge.id
            adjacent_to_hinge[hinge.vertex_adjacent2] = hinge.id
        return adjacent_to_hinge
    
    def build_adjacents_to_hinge(self):
        """Build a map from hinge side vertices to the hinge id."""
        adjacents_to_hinge = {}
        for hinge in self.hinges:
            pair = frozenset([hinge.vertex_adjacent1, hinge.vertex_adjacent2])
            adjacents_to_hinge[pair] = hinge.id
        return adjacents_to_hinge
    
    def add_void(self, hinge1_id, hinge2_id):
        """Add a void defined by a pair of opposite hinges."""
        self.voids.append((hinge1_id, hinge2_id))

    def build_void_opposite_edges(self):
        """Build the opposite edges of the voids."""
        void_opposite_edges = []
        for h1_id, h2_id in self.voids:
            e1 = [self.hinges[h1_id].vertex1, self.hinges[h1_id].vertex_adjacent1]
            e2 = [self.hinges[h2_id].vertex1, self.hinges[h2_id].vertex_adjacent1]
            void_opposite_edges.append([e1, e2])

            e3 = [self.hinges[h1_id].vertex2, self.hinges[h1_id].vertex_adjacent2]
            e4 = [self.hinges[h2_id].vertex2, self.hinges[h2_id].vertex_adjacent2]
            void_opposite_edges.append([e3, e4])
            
        return void_opposite_edges

    def update_vertices(self, vertices):
        vertices = np.asarray(vertices, dtype=float)
        if vertices.shape != self.vertices.shape:
            raise ValueError("New vertices must match the existing vertex shape")
        self.vertices = vertices
        return self
    
    def get_rectangular_bounds(self):
        if self.vertices.shape[0] == 0:
            return np.zeros((2, self.dim), dtype=float)
        min_coords = np.min(self.vertices, axis=0)
        max_coords = np.max(self.vertices, axis=0)
        return np.array([min_coords, max_coords])

    def face_centroids(self):
        if not self.faces:
            return np.zeros((0, self.dim), dtype=float)
        return np.vstack([face.centroid(self.vertices) for face in self.faces])

    def face_areas(self):
        if not self.faces:
            return np.zeros((0,), dtype=float)
        return np.array([face.area(self.vertices) for face in self.faces], dtype=float)

    def face_features(self):
        centroids = self.face_centroids()
        areas = self.face_areas().reshape(-1, 1)
        if centroids.size == 0:
            return np.zeros((0, self.dim + 1), dtype=float)
        return np.concatenate([centroids, areas], axis=1)
    
    def boundary_points(self):
        """Returns the points not involved in any hinge connections, which can be used as boundarys for optimization."""
        vertices = set(range(len(self.vertices)))
        hinge_vertices = self.build_primary_to_hinges().keys()
        return list(vertices - hinge_vertices)
        
    def compute_border_edges_lengths_sq(self, alpha=1.0):
        """
        Compute the squared lengths of the border edges based on the current vertices of this Tessellation.
        This should be called on the unmapped Tessellation to get the true rest lengths.
        The alpha parameter allows scaling these rest lengths to account for mapping transformations.
        """
        rest_lengths_sq = {}
        for group, edges in self.border_edges.items():
            if not edges:
                continue
            edges_arr = np.array(edges)
            p0 = self.vertices[edges_arr[:, 0]]
            p1 = self.vertices[edges_arr[:, 1]]
            lengths_sq = np.sum((p1 - p0)**2, axis=-1)
            # Scale by alpha squared because these are squared lengths
            rest_lengths_sq[group] = lengths_sq * (alpha ** 2)
        return rest_lengths_sq

    def boundary_pivot_indices(self):
        """Return the set of hinge pivot indices that lie on the boundary."""
        boundary_indices = set(self.boundary_points())
        
        # Build map from adjacent vertex to the actual pivot vertex
        adjacent_to_pivot = {}
        for hinge in self.hinges:
            adjacent_to_pivot[hinge.vertex_adjacent1] = hinge.vertex1
            adjacent_to_pivot[hinge.vertex_adjacent2] = hinge.vertex2
        
        pivots_on_boundary = set()
        for p in boundary_indices:
            if p in adjacent_to_pivot:
                pivots_on_boundary.add(adjacent_to_pivot[p])
                
        return list(pivots_on_boundary)
        

    def to_jax_state(self):

        # Convert vertices (N, dim)
        X = np.asarray(self.vertices, dtype=float)

        # Convert faces (N_faces, max_vertices_per_face) - assuming quads for now, with padding if necessary
        F_idx = np.array([face.vertex_indices for face in self.faces], dtype=np.int32) 

        # Adjacent edge vertices (N_hinges, 2, 2) - vertex indices of the adjacent vertices along the hinge closing edge
        E_adjacent = np.array([[[hinge.vertex1, hinge.vertex_adjacent1], [hinge.vertex2, hinge.vertex_adjacent2]] for hinge in self.hinges], dtype=np.int32)

        # Rest angles for hinges (N_hinges,)
        A_rest = np.array([hinge.angle for hinge in self.hinges], dtype=float)

        # Hinge properties (N_hinges,) - stiffness values for each hinge
        H_angular_stiffness = np.array([hinge.properties.get('angular_stiffness', 1.0) for hinge in self.hinges], dtype=float)
        H_linear_stiffness = np.array([hinge.properties.get('linear_stiffness', 1.0) for hinge in self.hinges], dtype=float)

        # Topological vertex connectivity (N_hinges, 2) - vertex indices of the hinge connections
        V_hinge = np.array([[hinge.vertex1, hinge.vertex2] for hinge in self.hinges], dtype=np.int32)

        # boundary indices (N_boundarys,) - vertex indices of boundary points not involved in any hinge connections
        Boundary_indices = np.array(self.boundary_points(), dtype=np.int32)
                    
        Boundary_pivot_indices = np.array(self.boundary_pivot_indices(), dtype=np.int32)

        # Opposite edges (2*N_voids, 2, 2) - vertex indices of the opposite edges
        E_opp = np.array(self.build_void_opposite_edges(), dtype=np.int32)

        # Border edges
        # We can pass them as a list or dict of arrays. Since JAX works with dicts of arrays, we can do that.
        Border_edges = {group: np.array(edges, dtype=np.int32) for group, edges in self.border_edges.items()}

        return {
            'vertices': X,
            'faces': F_idx,
            'hinge_adjacent_edges': E_adjacent,
            'void_opposite_edges': E_opp,
            'angles_rest': A_rest,
            'hinge_angular_stiffness': H_angular_stiffness,
            'hinge_linear_stiffness': H_linear_stiffness,
            'hinge_vertex_connections': V_hinge,
            'boundary_indices': Boundary_indices,
            'boundary_pivot_indices': Boundary_pivot_indices,
            'border_edges': Border_edges,
        }

