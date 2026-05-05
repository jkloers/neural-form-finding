
import jax
import jax.numpy as jnp
from typing import Any, Dict, NamedTuple, Optional, Union

# Registering all parameter classes as JAX Pytrees ensures they can carry Tracers 
# through JIT boundaries without "not a valid JAX type" errors.

@jax.tree_util.register_pytree_node_class
class SolutionData(NamedTuple):
    face_centroids: Any
    centroid_node_vectors: Any
    bond_connectivity: Any
    fields: Any
    energies: Any = None

    def tree_flatten(self):
        return (self.face_centroids, self.centroid_node_vectors, self.bond_connectivity, self.fields, self.energies), ()

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


@jax.tree_util.register_pytree_node_class
class GeometricalParams(NamedTuple):
    face_centroids: Any
    centroid_node_vectors: Any
    bond_connectivity: Any = None
    reference_bond_vectors: Any = None

    def tree_flatten(self):
        return (self.face_centroids, self.centroid_node_vectors, self.bond_connectivity, self.reference_bond_vectors), ()

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


@jax.tree_util.register_pytree_node_class
class LigamentParams(NamedTuple):
    k_stretch: Any
    k_shear: Any
    k_rot: Any
    reference_vector: Any

    def tree_flatten(self):
        return (self.k_stretch, self.k_shear, self.k_rot, self.reference_vector), ()

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)

BondParams = LigamentParams


@jax.tree_util.register_pytree_node_class
class ContactParams(NamedTuple):
    """Contact parameters for the simplified contact model.

    See `energy.contact_energy` for details.
    Note: If distance-based contact is used the min_angle and cutoff_angle are interpreted as distances.

    Attrs:
        min_angle (jnp.ndarray, optional): Lower bound for the angle between the blocks.
        cutoff_angle (jnp.ndarray, optional): Cutoff for the contact energy.
        k_contact (float, optional): Initial stiffness of the contact.
    """

    min_angle: Any
    cutoff_angle: Any
    k_contact: Any

    def tree_flatten(self):
        return (self.min_angle, self.cutoff_angle, self.k_contact), ()

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


@jax.tree_util.register_pytree_node_class
class MechanicalParams(NamedTuple):
    """Mechanical parameters of the system.

    Attrs:
        bond_params (BondParams): NamedTuple defining the bond parameters.
        density (jnp.ndarray): Density of the blocks, either a scalar or an array of shape (n_blocks,).
        contact_params (ContactParams, optional): NamedTuple defining the contact parameters. Defaults to None.
    """

    bond_params: BondParams
    density: Any
    contact_params: Optional[ContactParams] = None

    def tree_flatten(self):
        return (self.bond_params, self.density, self.contact_params), ()

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


@jax.tree_util.register_pytree_node_class
class ControlParams(NamedTuple):
    """Control parameters for the static solver.
    The control parameters are used to define the geometry, the mechanical properties, loading parameters, etc.
    This data structure is meant to help with the construction of the mapping: design variables -> geometry, mechanical properties, etc. -> static solver.

    Attrs:
        geometrical_params (GeometricalParams): NamedTuple defining the geometrical parameters.
        mechanical_params (MechanicalParams): NamedTuple defining the mechanical parameters.
        loading_params (Dict[str, Any]): Loading parameters to be passed to loading functions. Default: {}.
        constraint_params (Dict[str, Any]): Constraint parameters to be passed to constraint_DOFs_fn. Default: {}.
    """

    geometrical_params: GeometricalParams  # centroids and centroid_node_vectors
    mechanical_params: MechanicalParams  # bond params, mass density, damping
    loading_params: Dict = dict()
    constraint_params: Dict = dict()

    def tree_flatten(self):
        # We treat Dicts as auxiliary data if they don't contain tracers,
        # but here we keep them as children to be safe if they ever do.
        return (self.geometrical_params, self.mechanical_params, self.loading_params, self.constraint_params), ()

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)
