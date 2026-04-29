
import jax
import jax.numpy as jnp
from typing import NamedTuple, Callable, Tuple, Any
from jax import vmap

# ─────────────────────────────────────────────────────────────────────────────
# MODERN: Centroidal Geometry (Pytree)
# ─────────────────────────────────────────────────────────────────────────────

@jax.tree_util.register_pytree_node_class
class Geometry:
    """Base class for JAX-compatible geometry representations."""
    
    def tree_flatten(self):
        # Default implementation for stateless geometries
        return (), ()

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

    def face_centroids(self): raise NotImplementedError()
    def centroid_node_vectors(self): raise NotImplementedError()
    def bond_connectivity(self): raise NotImplementedError()
    def reference_bond_vectors(self): raise NotImplementedError()

    def get_reference_geometry(self):
        """Computes reference configuration of all the nodes."""
        c = self.face_centroids()
        s = self.centroid_node_vectors()
        return vmap(lambda face_nodes, centroid: face_nodes + centroid)(s, c)

    @property
    def n_faces(self):
        return self.face_centroids().shape[0]

    @property
    def n_nodes(self):
        c_shape = self.centroid_node_vectors().shape
        return c_shape[0] * c_shape[1]


@jax.tree_util.register_pytree_node_class
class TessellationGeometry(Geometry):
    """Concrete Geometry representation for the centroidal pipeline.
    
    This class is a proper JAX Pytree, meaning it can be passed to JIT-compiled
    functions and supports gradients.
    """

    def __init__(self, face_centroids, centroid_node_vectors, 
                 bond_connectivity, reference_bond_vectors):
        self._fc = jnp.array(face_centroids)
        self._cnv = jnp.array(centroid_node_vectors)
        self._bc = jnp.array(bond_connectivity)
        self._rbv = jnp.array(reference_bond_vectors)

    def tree_flatten(self):
        # Children are the JAX arrays that carry data/gradients
        children = (self._fc, self._cnv, self._bc, self._rbv)
        # Aux data is empty here as everything is in children
        aux_data = {}
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    # Physics solver interface (called as no-arg methods)
    def face_centroids(self): return self._fc
    def centroid_node_vectors(self): return self._cnv
    def bond_connectivity(self): return self._bc
    def reference_bond_vectors(self): return self._rbv

    @classmethod
    def from_dict(cls, d: dict):
        """Build from the dictionary returned by Tessellation.to_centroidal_state()."""
        return cls(
            face_centroids=d['face_centroids'],
            centroid_node_vectors=d['centroid_node_vectors'],
            bond_connectivity=d['bond_connectivity'],
            reference_bond_vectors=d['reference_bond_vectors']
        )
