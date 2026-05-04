from typing import Any, Dict, NamedTuple, Optional, Union

import jax.numpy as jnp
import numpy as np


class SolutionData(NamedTuple):
    """Solution data for the static problem.

    Attrs:
        face_centroids (jnp.ndarray): shape (n_faces, 2) — reference centroids of the faces.
        centroid_node_vectors (jnp.ndarray): shape (n_faces, n_nodes_per_face, 2).
        bond_connectivity (jnp.ndarray): shape (n_bonds, 2).
        fields (jnp.ndarray): shape (n_faces, 3) for statics.
    """

    face_centroids: Any
    centroid_node_vectors: Any
    bond_connectivity: Any
    fields: Any
    energies: Any = None


class GeometricalParams(NamedTuple):
    """Geometrical parameters of the system.

    Attrs:
        face_centroids (jnp.ndarray): shape (n_faces, 2) — centroid coordinates.
        centroid_node_vectors (jnp.ndarray): shape (n_faces, n_nodes_per_face, 2).
        bond_connectivity (jnp.ndarray): shape (n_bonds, 2). Optional, used for energy computation.
        reference_bond_vectors (jnp.ndarray): shape (n_bonds, 2). Optional.
    """

    face_centroids: Any
    centroid_node_vectors: Any
    bond_connectivity: Any = None
    reference_bond_vectors: Any = None


class LigamentParams(NamedTuple):
    """Parameters for the bonds modeled as finite-length ligaments.

    Attrs:
        k_stretch (jnp.ndarray): Either a scalar or an array of shape (n_bonds,) representing the stretch stiffness of each bond.
        k_shear (jnp.ndarray): Either a scalar or an array of shape (n_bonds,) representing the shear stiffness of each bond.
        k_rot (jnp.ndarray): Either a scalar or an array of shape (n_bonds,) representing the rotational stiffness of each bond.
        reference_bond_vectors (jnp.ndarray): Array of shape (n_bonds, 2) representing the reference configuration of the bond (length matters). These are typically computed from a given geometry class.
    """

    k_stretch: Any
    k_shear: Any
    k_rot: Any
    reference_vector: Any


BondParams = LigamentParams


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






