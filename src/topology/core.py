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

@dataclass
class Hinge:
    """A hinge defined by two faces and two vertex indices."""

    face1: int
    face2: int
    vertex1: int # Index of the vertex in face1 that defines the hinge
    vertex2: int
    rest_angle: float = 0.0
    stiffness: float = 1.0
    id: any = None

    def __repr__(self):
        return (
            f"Hinge(id={self.id}, face1={self.face1}, face2={self.face2}, "
            f"vertex1={self.vertex1}, vertex2={self.vertex2}, "
            f"rest_angle={self.rest_angle}, stiffness={self.stiffness})"
        )
    

class UnitPattern:
    def __init__(self, vertices, faces, internal_hinges, external_hinges, shift_vectors=None):
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

    def __init__(self, vertices=None, faces=None, hinges=None, dim=2, tolerance=1e-6):
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
                    rest_angle=hinge.rest_angle,
                    stiffness=hinge.stiffness,
                    id=hinge.id,

                )

    def add_vertex(self, vertex):
        vertex = np.asarray(vertex, dtype=float)
        if vertex.shape != (self.vertices.shape[1],):
            raise ValueError(f"Vertex must have shape ({self.vertices.shape[1]},)")
        self.vertices = np.vstack([self.vertices, vertex])
        return len(self.vertices) - 1  # Return the index of the new vertex

    def add_face(self, vertex_indices, id=None):
        if isinstance(vertex_indices, IndexedFace):
            face = vertex_indices
        else:
            face = IndexedFace(vertex_indices, id=id)
        self.faces.append(face)
        return face

    def add_hinge(self, face1, face2, vertex1, vertex2, rest_angle=0.0, stiffness=1.0, id=None):
        hinge = Hinge(face1, face2, vertex1, vertex2, rest_angle, stiffness, id)
        self.hinges.append(hinge)
        return hinge

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

    def to_jax_state(self, pad_faces=False, pad_value=-1):
        X = np.asarray(self.vertices, dtype=float)
        num_faces = len(self.faces)
        face_vertex_counts = np.array([face.num_vertices for face in self.faces], dtype=int)

        if pad_faces:
            max_length = int(np.max(face_vertex_counts)) if num_faces > 0 else 0
            F_idx = np.full((num_faces, max_length), pad_value, dtype=int)
            for face_idx, face in enumerate(self.faces):
                F_idx[face_idx, : face.num_vertices] = face.vertex_indices
            face_ptrs = None
        else:
            face_ptrs = np.concatenate([[0], np.cumsum(face_vertex_counts, dtype=int)]) if num_faces > 0 else np.array([0], dtype=int)
            F_idx = np.concatenate([face.vertex_indices for face in self.faces], dtype=int) if num_faces > 0 else np.array([], dtype=int)

        if self.hinges:
            hinge_face1 = np.array([hinge.face1 for hinge in self.hinges], dtype=int)
            hinge_face2 = np.array([hinge.face2 for hinge in self.hinges], dtype=int)
            hinge_vertex1 = np.array([hinge.vertex1 for hinge in self.hinges], dtype=int)
            hinge_vertex2 = np.array([hinge.vertex2 for hinge in self.hinges], dtype=int)
            hinge_rest_angle = np.array([hinge.rest_angle for hinge in self.hinges], dtype=float)
            hinge_stiffness = np.array([hinge.stiffness for hinge in self.hinges], dtype=float)
            senders = hinge_face1
            receivers = hinge_face2
        else:
            hinge_face1 = np.zeros((0,), dtype=int)
            hinge_face2 = np.zeros((0,), dtype=int)
            hinge_vertex1 = np.zeros((0,), dtype=int)
            hinge_vertex2 = np.zeros((0,), dtype=int)
            hinge_rest_angle = np.zeros((0,), dtype=float)
            hinge_stiffness = np.zeros((0,), dtype=float)
            senders = np.zeros((0,), dtype=int)
            receivers = np.zeros((0,), dtype=int)

        return {
            "vertices": X,
            "face_ptrs": face_ptrs,
            "face_vertex_indices": F_idx,
            "face_vertex_counts": face_vertex_counts,
            "hinge_face1": hinge_face1,
            "hinge_face2": hinge_face2,
            "hinge_vertex1": hinge_vertex1,
            "hinge_vertex2": hinge_vertex2,
            "hinge_rest_angle": hinge_rest_angle,
            "hinge_stiffness": hinge_stiffness,
            "graph_senders": senders,
            "graph_receivers": receivers,
            "face_features": self.face_features(),
        }

    def __repr__(self):
        return (
            f"Tessellation(vertices={len(self.vertices)}, faces={len(self.faces)}, "
            f"hinges={len(self.hinges)}, dim={self.dim})"
        )