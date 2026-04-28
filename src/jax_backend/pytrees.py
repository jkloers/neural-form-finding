
import jax
import jax.numpy as jnp
from typing import NamedTuple, Callable, Tuple
from jax import vmap


class TessellationState(NamedTuple):
    X: jnp.ndarray
    
    # Topology
    F_idx: jnp.ndarray
    F_rest_lengths_sq: jnp.ndarray # (N_faces, 6): 4 edges + 2 diagonals
    E_adjacent: jnp.ndarray
    E_opp: jnp.ndarray
    A_rest: jnp.ndarray
    H_angular_stiffness: jnp.ndarray
    H_linear_stiffness: jnp.ndarray
    V_connect: jnp.ndarray
    Boundary_indices: jnp.ndarray
    Border_edges: dict
    Border_edges_rest_lengths_sq: dict

jax.tree_util.register_pytree_node(
    TessellationState,
    lambda state: ((state.X,), (state.F_idx, state.F_rest_lengths_sq, state.E_adjacent, state.E_opp, state.A_rest, state.H_angular_stiffness, state.H_linear_stiffness, state.V_connect, state.Boundary_indices, state.Border_edges, state.Border_edges_rest_lengths_sq)),
    lambda aux, dynamic: TessellationState(dynamic[0], *aux)
)

def compute_face_lengths_sq(X, F_idx):
    """Calculates the squared lengths of the 4 edges and 2 diagonals for each face."""
    p0, p1, p2, p3 = X[F_idx[:, 0]], X[F_idx[:, 1]], X[F_idx[:, 2]], X[F_idx[:, 3]]
    edges = [p0-p1, p1-p2, p2-p3, p3-p0, p0-p2, p1-p3]
    return jnp.stack([jnp.sum(e**2, axis=-1) for e in edges], axis=1)

def create_jax_state(tess_dict):
    """Converts a tessellation dictionary to a JAX-compatible state representation."""
    X_init = jnp.array(tess_dict['vertices'])
    F_idx = jnp.array(tess_dict['faces'])
    
    # Pre-calculate initial face lengths for rigidity constraint
    rest_lengths_sq = compute_face_lengths_sq(X_init, F_idx)

    return TessellationState(
        X=X_init,
        F_idx=F_idx,
        F_rest_lengths_sq=rest_lengths_sq,
        E_adjacent=jnp.array(tess_dict['hinge_adjacent_edges']),
        E_opp=jnp.array(tess_dict['void_opposite_edges']),
        A_rest=jnp.array(tess_dict['angles_rest']),
        H_angular_stiffness=jnp.array(tess_dict['hinge_angular_stiffness']),
        H_linear_stiffness=jnp.array(tess_dict['hinge_linear_stiffness']),
        V_connect=jnp.array(tess_dict['hinge_vertex_connections']),
        Boundary_indices=jnp.array(tess_dict['boundary_indices']),
        Border_edges={k: jnp.array(v) for k, v in tess_dict.get('border_edges', {}).items()},
        Border_edges_rest_lengths_sq={
            k: jnp.array(v)
            for k, v in tess_dict.get('border_edges_rest_lengths_sq', {}).items()
        }
    )


class Geometry:
    """
    Template class for defining geometric data for rigid-face assemblies.
    """

    n_faces: int
    n_nodes: int
    face_centroids: Callable
    centroid_node_vectors: Callable
    bond_connectivity: Callable
    reference_bond_vectors: Callable

    @property
    def n_blocks(self):
        """Alias for n_faces (backward compatibility)."""
        return self.n_faces

    def compute_geometry(self):
        """Any geometric class must implement the definition of the following data structures:
        - `face_centroids`: (ndarray): array of shape (n_faces, 2) defining the centroid of each face.
        - `centroid_node_vectors` (ndarray): array of shape (n_faces, n_nodes_per_face, 2) defining the vectors connecting the centroid of the face to each node.
        - `bond_connectivity` (ndarray): array of shape (n_bonds, 2) defining the pair of nodes connected by bonds i.e. each row is of the form [node1, node2].
        - `reference_bond_vectors` (ndarray): array of shape (n_bonds, 2) defining the reference configuration of the bonds.

        Raises:
            NotImplementedError: `compute_geometry` must define `centroid_node_vectors`, `bond_connectivity`, and `reference_bond_vectors`.
        """
        raise NotImplementedError("Child classes should implement this method.")

    def get_reference_geometry(self, *args):
        """
        Computes reference configuration of all the nodes.
        """

        try:
            centroid_node_vectors = self.centroid_node_vectors(*args)
        except AttributeError as err:
            self.compute_geometry()
            centroid_node_vectors = self.centroid_node_vectors(*args)

        centroids = self.face_centroids(*args)

        return vmap(lambda face_nodes, centroid: face_nodes + centroid, in_axes=(0, 0))(centroid_node_vectors, centroids)

    def get_xy_limits(self, *args):
        """
        Computes reference configuration xy limits.
        """

        vertices = self.get_reference_geometry(*args).reshape((self.n_nodes, 2))
        return compute_xy_limits(vertices)

    def get_parametrization(self) -> Tuple[Callable, Callable, Callable, Callable]:
        """Returns the set of functions parameterizing the geometry.

        Returns:
            Tuple[Callable, Callable, Callable, Callable]: parameterizing functions: face_centroids, centroid_node_vectors, bond_connectivity, reference_bond_vectors.
        """

        self.compute_geometry()

        return self.face_centroids, self.centroid_node_vectors, self.bond_connectivity, self.reference_bond_vectors


class TessellationGeometry(Geometry):
    """Concrete Geometry built from the dict returned by Tessellation.to_jax_state_centroidal().

    Unlike the parametric Geometry base class (whose attributes are Callables),
    this class wraps fixed numpy/jnp arrays and exposes them through no-arg lambdas
    so that the rest of the physics solver pipeline (which calls ``face_centroids()``)
    works without changes.

    Usage::

        tess_state = tessellation.to_jax_state_centroidal()
        geom = TessellationGeometry.from_dict(tess_state)
        # geom.n_faces, geom.face_centroids(), geom.bond_connectivity(), etc.
    """

    def __init__(self, tess_dict: dict):
        _fc = jnp.array(tess_dict['face_centroids'])
        _cnv = jnp.array(tess_dict['centroid_node_vectors'])
        _bc = jnp.array(tess_dict['bond_connectivity'])
        _rbv = jnp.array(tess_dict['reference_bond_vectors'])

        self.n_faces = int(_fc.shape[0])
        self.n_nodes = int(self.n_faces * _cnv.shape[1])

        # Expose as no-arg callables (consistent with Geometry interface)
        self.face_centroids = lambda: _fc
        self.centroid_node_vectors = lambda: _cnv
        self.bond_connectivity = lambda: _bc
        self.reference_bond_vectors = lambda: _rbv

        # Store extra data from the tessellation export
        self._tess_dict = tess_dict

    def compute_geometry(self):
        """No-op: geometry is already computed from the tessellation export."""
        pass

    @classmethod
    def from_dict(cls, tess_dict: dict) -> 'TessellationGeometry':
        """Alternative constructor for readability."""
        return cls(tess_dict)
