
from jax_backend.utils.utils import ControlParams
import jax
import jax.numpy as jnp
from typing import Any, TYPE_CHECKING
if TYPE_CHECKING:
    from jax_backend.state import CentroidalState

from jax import vmap

# ─────────────────────────────────────────────────────────────────────────────
# MODERN: Centroidal Geometry (Pytree)
# ─────────────────────────────────────────────────────────────────────────────

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
    
    This class is a proper JAX Pytree. 
    """

    def __init__(self, face_centroids, centroid_node_vectors, 
                 bond_connectivity, reference_bond_vectors):
        self._fc = jnp.array(face_centroids)
        self._cnv = jnp.array(centroid_node_vectors)
        self._bc = bond_connectivity 
        self._rbv = jnp.array(reference_bond_vectors)

    def tree_flatten(self):
        # We keep face_centroids, centroid_node_vectors and reference_bond_vectors as dynamic
        # Bond connectivity is static topology
        children = (self._fc, self._cnv, self._rbv)
        aux_data = {'bond_connectivity': self._bc}
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(children[0], children[1], aux_data['bond_connectivity'], children[2])

    # Physics solver interface (called as no-arg methods)
    def face_centroids(self): return self._fc
    def centroid_node_vectors(self): return self._cnv
    def bond_connectivity(self): return self._bc
    def reference_bond_vectors(self): return self._rbv

    @classmethod
    def from_centroidal_state(cls, state: 'CentroidalState') -> 'TessellationGeometry':
        """Factory pattern: builds physical geometry from a centroidal state.
        
        All topology (bond_connectivity) is read from the state where it is
        pre-computed as a static NumPy array — never a JAX Tracer.
        """
        from jax_backend.geometry import build_reference_bond_vectors
        
        return cls(
            face_centroids=state.face_centroids,
            centroid_node_vectors=state.centroid_node_vectors,
            bond_connectivity=state.bond_connectivity,
            reference_bond_vectors=build_reference_bond_vectors(state)
        )

    def build_control_params(self, state: 'CentroidalState',
                              k_contact: float = 1.0,
                              min_angle: float = 0.0,
                              cutoff_angle: float = 0.1,
                              use_contact: bool = True) -> 'ControlParams':
        """Assembles the ControlParams required by the physics solver.
        
        Encapsulates the mapping from (geometry, state) → ControlParams so
        that pipeline.py does not need to know the internal structure of
        either object.
        """
        from jax_backend.utils.utils import (
            ControlParams, GeometricalParams, MechanicalParams,
            LigamentParams, ContactParams
        )
        return ControlParams(
            geometrical_params=GeometricalParams(
                face_centroids=self.face_centroids(),
                centroid_node_vectors=self.centroid_node_vectors(),
                bond_connectivity=self.bond_connectivity(),
                reference_bond_vectors=self.reference_bond_vectors(),
            ),
            mechanical_params=MechanicalParams(
                bond_params=LigamentParams(
                    k_stretch=state.k_stretch,
                    k_shear=state.k_shear,
                    k_rot=state.k_rot,
                    reference_vector=self.reference_bond_vectors(),
                ),
                density=state.density,
                contact_params=ContactParams(
                    k_contact=k_contact,
                    min_angle=min_angle,
                    cutoff_angle=cutoff_angle,
                ) if use_contact else None,
            ),
            constraint_params=dict(),
        )