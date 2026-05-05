"""
The `energy` module implements the energy functional for the whole structure.
"""

from typing import Callable, Tuple

import jax.numpy as jnp
from jax import vmap
from jax_md import smap

from jax_backend.physics_solver.kinematics import face_to_node_kinematics
from jax_backend.physics_solver.params import ControlParams
from jax_backend.utils.linalg import vdot, void_angles, build_void_edge_distance



def ligament_strains_linearized(DOFs1: jnp.ndarray, DOFs2: jnp.ndarray, reference_vector: jnp.ndarray = jnp.array([1., 0.])):
    """Computes linearized strain measures of an elastic ligament i.e. axial, shear, and flexural strains.

    The axial strain is defined as dU.v0/v0^2.
    The shear strain is defined as (theta1+theta2)/2 - v0✕dU/v0^2.
    The rotational strain is defined as theta2-theta1.

    Note: These strains are based on the linearized beam theory.

    Args:
        DOFs1 (jnp.ndarray): array of shape (Any, 3) representing the DOFs of the first node connected by the ligament.
        DOFs2 (jnp.ndarray): array of shape (Any, 3) representing the DOFs of the second node connected by the ligament.
        reference_vector (jnp.ndarray, optional): array of shape (2,) or (Any, 2) representing the reference configuration of the ligament. Defaults to jnp.array([1., 0.]).

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]: axial, shear, and rotational strains.
    """

    dU = DOFs2[:, :2] - DOFs1[:, :2]
    dRot = DOFs2[:, 2] - DOFs1[:, 2]

    axial_strain = vdot(dU, reference_vector) / jnp.linalg.norm(reference_vector, axis=-1)**2
    shear_strain = jnp.cross(reference_vector, dU, axis=-1) / jnp.linalg.norm(reference_vector, axis=-1)**2 - (DOFs2[:, 2] + DOFs1[:, 2])/2

    return axial_strain, shear_strain, dRot


def ligament_energy_linearized(nodal_DOFs: Tuple[jnp.ndarray, jnp.ndarray], reference_vector: jnp.ndarray = jnp.array([1., 0.]), k_stretch=1., k_shear=1., k_rot=1.):
    """Computes the strain energy of an elastic ligament using linearized strain measures (suitable for moderate global rotations).

    Args:
        nodal_DOFs (Tuple[ndarray, ndarray]): tuple of arrays of shape (Any, 3) representing the DOFs of the nodes connected by the ligament.
        reference_vector (ndarray, optional): array of shape (2,) or (Any, 2) representing the reference bond geometry (length matters). Defaults to jnp.array([1., 0.]).
        k_stretch (float, optional): linear stretching stiffness. Defaults to 1..
        k_shear (float, optional): linear shearing stiffness. Defaults to 1..
        k_rot (float, optional): linear rotational stiffness. Defaults to 1..

    Returns:
        float: strain energy.
    """

    axial_strain, shear_strain, dRot = ligament_strains_linearized(
        *nodal_DOFs, reference_vector=reference_vector)
    l0 = jnp.linalg.norm(reference_vector, axis=-1)

    return k_stretch * (axial_strain*l0)**2 / 2 + k_shear * (shear_strain*l0)**2 / 2 + k_rot * dRot**2 / 2


def ligament_strains(DOFs1: jnp.ndarray, DOFs2: jnp.ndarray, reference_vector: jnp.ndarray = jnp.array([1., 0.])):
    """Computes the nonlinear strain measures of an elastic ligament i.e. axial, shear, and flexural strains.

    The axial strain is defined as (L-L0)/L0.
    The shear strain is defined as current_bond_angle-reference_bond_pushed_angle where reference_bond_pushed_angle is the reference rotated by (theta1+theta2)/2.
    Note: the shear strain is assumed to be between -pi and pi.
    The rotational strain is defined as theta2-theta1.

    Note: These strains are based on beam theory (e.g. see https://static-content.springer.com/esm/art%3A10.1038%2Fnphys4269/MediaObjects/41567_2018_BFnphys4269_MOESM1_ESM.pdf).

    Args:
        DOFs1 (jnp.ndarray): array of shape (Any, 3) representing the DOFs of the first node connected by the ligament.
        DOFs2 (jnp.ndarray): array of shape (Any, 3) representing the DOFs of the second node connected by the ligament.
        reference_vector (jnp.ndarray, optional): array of shape (2,) or (Any, 2) representing the reference configuration of the ligament. Defaults to jnp.array([1., 0.]).

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]: axial, shear, and rotational strain measures.
    """
    eps = 1e-12
    dU = DOFs2[:, :2] - DOFs1[:, :2]
    dRot = DOFs2[:, 2] - DOFs1[:, 2]
    mean_rot = (DOFs2[:, 2] + DOFs1[:, 2]) / 2.0

    current_bond_vector = dU + reference_vector
    # Add eps to ensure current_bond_vector is never exactly (0,0)
    # This prevents arctan2 gradient explosion
    curr_x = current_bond_vector[:, 0] + eps
    curr_y = current_bond_vector[:, 1]
    current_bond_angle = jnp.arctan2(curr_y, curr_x)
    
    # Reference geometry "pushed" (rotated) by the mean rotation of the faces
    # Use vectorized rotation instead of nested vmap for stability
    c = jnp.cos(mean_rot)
    s = jnp.sin(mean_rot)
    
    # reference_vector can be (2,) or (N, 2)
    ref_x = reference_vector[..., 0]
    ref_y = reference_vector[..., 1]
    
    ref_pushed_x = c * ref_x - s * ref_y
    ref_pushed_y = s * ref_x + c * ref_y
    
    reference_bond_pushed_angle = jnp.arctan2(ref_pushed_y, ref_pushed_x + eps)
    
    # 1. Axial Strain: (L - L0) / L0
    l0_sq = vdot(reference_vector, reference_vector) + eps
    l_sq = vdot(current_bond_vector, current_bond_vector)
    axial_strain = jnp.sqrt(l_sq / l0_sq) - 1.0
    
    # 2. Shear Strain: angle difference (modulo 2pi)
    shear_strain = jnp.mod(current_bond_angle - reference_bond_pushed_angle + jnp.pi, 2*jnp.pi) - jnp.pi
    
    return axial_strain, shear_strain, dRot


def ligament_energy(nodal_DOFs: Tuple[jnp.ndarray, jnp.ndarray], reference_vector: jnp.ndarray = jnp.array([1., 0.]), k_stretch=1., k_shear=1., k_rot=1.):
    """Computes the strain energy of an elastic ligament using nonlinear strain measures (suitable for arbitrarily large rotations).

    Args:
        nodal_DOFs (Tuple[ndarray, ndarray]): tuple of arrays of shape (Any, 3) representing the DOFs of the nodes connected by the ligament.
        reference_vector (ndarray, optional): array of shape (2,) or (Any, 2) representing the reference configuration of the bond (length matters). Defaults to jnp.array([1., 0.]).
        k_stretch (float, optional): linear stretching stiffness. Defaults to 1..
        k_shear (float, optional): linear shearing stiffness. Defaults to 1..
        k_rot (float, optional): linear rotational stiffness. Defaults to 1..

    Returns:
        float: strain energy.
    """

    axial_strain, shear_strain, dRot = ligament_strains(
        *nodal_DOFs, reference_vector=reference_vector)
    l0 = jnp.linalg.norm(reference_vector, axis=-1)

    return k_stretch * (axial_strain*l0)**2 / 2 + k_shear * (shear_strain*l0)**2 / 2 + k_rot * dRot**2 / 2


def strain_energy_bond(bond_connectivity: jnp.ndarray, bond_energy_fn: Callable = ligament_energy_linearized):
    """Maps energy functional of a single bond to a set of bonds defined by `bond_connectivity`.

    Args:
        bond_connectivity (ndarray): array of shape (n_bonds, 2) where each row [n1, n2] defines a bond connecting nodes n1 and n2.
        bond_energy_fn (Callable): energy functional of a single bond. Defaults to `energy.ligament_energy_linearized`.

    Returns:
        Callable: strain energy vectorized over the set of bonds defined by `bond_connectivity`.
    """

    return smap.bond(
        bond_energy_fn,  # Single bond energy
        # This pattern is needed because smap.bond is not vmapping kwargs in the strain function (workaround: strain measures are computed inside bond energy).
        lambda Ua, Ub, **kwargs: (Ua, Ub),
        static_bonds=bond_connectivity,
        static_bond_types=None
        # It can take any additional parameters to be passed to the single bond energy function
    )


# Contact energy between adjacent edges
# NOTE: This is a simplified way to handle contact. The energy is just based on the angle between faces connected by a bond.
# NOTE: This is also not based on general data structures for defining edges (see geometry.compute_edge_angles).


def contact_energy(current_void_angles: jnp.ndarray, min_angle: jnp.ndarray = jnp.array(0.), cutoff_angle: jnp.ndarray = jnp.array(2.0*jnp.pi/180), k_contact=1.0):
    """Computes the contact energy between connected faces.

    This is a simplified way to handle contact. The energy is just based on the angle between faces connected by a bond.

    Args:
        current_void_angles (jnp.ndarray): array of shape (2*n_bonds,) defining the angles between connected faces.
        min_angle (jnp.ndarray, optional): lower bound for the angle between the faces. Defaults to jnp.array(0.).
        cutoff_angle (jnp.ndarray, optional): cutoff for the contact energy. Defaults to jnp.array(2.0*jnp.pi/180).
        k_contact (float, optional): initial stiffness of the contact. Defaults to 1.0.

    Returns:
        float: contact energy
    """
    # Current contact energy is of the kind ~1/x with a C^1 cutoff.
    # min_angle is an asymptote for the energy. This is to make sure that min_angle cannot be overcome.
    x = (current_void_angles-cutoff_angle)/(cutoff_angle-min_angle)
    energy = jnp.where(
        # This means that the faces are not in contact as we assume that min_angle is the minimum angle between the faces
        current_void_angles < min_angle,
        0,
        jnp.where(
            current_void_angles < cutoff_angle,
            k_contact/4 * (cutoff_angle-min_angle)**2 * \
            ((x+1)**-1 - (x-1)**-1 - 2),
            0
        )
    )
    return energy


def build_contact_energy(bond_connectivity: jnp.ndarray, angle_based=True):
    """Defines the energy functional for simulating contact between connected faces.

    Args:
        bond_connectivity (jnp.ndarray): array of shape (n_bonds, 2) where each row [n1, n2] defines a bond connecting nodes n1 and n2.
        angle_based (bool, optional): whether to use the angle-based contact energy or the distance-based one. Defaults to True (angle-based). Angle-based is more cheaper but less accurate for complex geometries.

    Returns:
        Callable: contact energy functional as a function of the DOFs of the faces and the `control_params`.
    """

    void_edge_distance_fn = build_void_edge_distance(bond_connectivity)

    def void_angle_fn(current_face_nodes): return void_angles(
        current_face_nodes, bond_connectivity)
    distance_fn = void_angle_fn if angle_based else void_edge_distance_fn

    def contact_energy_fn(face_displacement: jnp.ndarray, control_params: ControlParams):
        """Computes the contact energy between connected faces.

        Args:
            face_displacement (jnp.ndarray): array of shape (n_faces, 3) collecting the displacements (first two positions) and rotations (last position) of all the faces.
            centroid_node_vectors (ndarray): array of shape (n_faces, n_nodes_per_face, 2) representing the vectors connecting the centroid of the faces to the nodes.
            control_params (ControlParams): contains the contact params in control_params.mechanical_params.contact_params.

        Returns:
            float: Total contact energy.
        """

        face_centroids = control_params.geometrical_params.face_centroids
        centroid_node_vectors = control_params.geometrical_params.centroid_node_vectors
        contact_params = control_params.mechanical_params.contact_params

        node_displacements = jnp.array(
            face_to_node_kinematics(
                face_displacement,
                centroid_node_vectors
            )
        )[:, :, :2]
        current_face_nodes = face_centroids[:, None] + \
            centroid_node_vectors + node_displacements
        return jnp.sum(contact_energy(current_void_angles=distance_fn(current_face_nodes), **contact_params._asdict()))

    return contact_energy_fn


def build_strain_energy(bond_connectivity: jnp.ndarray, bond_energy_fn: Callable = ligament_energy_linearized) -> Callable:
    """Defines the strain energy functional of the system.

    Args:
        bond_connectivity (ndarray): array of shape (n_bonds, 2) where each row [n1, n2] defines a bond connecting nodes n1 and n2.
        bond_energy_fn (Callable): energy functional of a single bond. Defaults to `energy.ligament_energy_linearized`.

    Returns:
        Callable: function evaluating the strain energy of the system from the DOFs of the faces and the `control_params`.
    """

    # Build vectorized bond energy using smap.bond
    strain_energy_bonds = strain_energy_bond(
        bond_connectivity=bond_connectivity, bond_energy_fn=bond_energy_fn)

    def strain_energy_fn(face_displacement: jnp.ndarray, control_params: ControlParams) -> float:
        """Computes total strain energy by summing over all bonds.

        Args:
            face_displacement (ndarray): array of shape (n_faces, 3) collecting the displacements (first two positions) and rotations (last position) of all the faces.
            control_params (ControlParams): contains the geometrical params in control_params.geometrical_params, as well as the bond params in control_params.mechanical_params.bond_params.

        Returns:
            float: Total strain energy.
        """

        centroid_node_vectors = control_params.geometrical_params.centroid_node_vectors
        bond_params = control_params.mechanical_params.bond_params

        n_faces, n_nodes_per_face, _ = centroid_node_vectors.shape
        node_displacements = face_to_node_kinematics(
            face_displacement,
            centroid_node_vectors
        )
        node_displacements = node_displacements.reshape(
            (n_faces * n_nodes_per_face, 3))

        return strain_energy_bonds(node_displacements, **bond_params._asdict())

    return strain_energy_fn


def combine_face_energies(*energy_fns: Callable):
    """Combines multiple energy functions into a single function with signature (face_displacement, control_params) -> energy.

    Args:
        *energy_fns (Callable): energy functions with signature (face_displacement, control_params) -> energy.

    Returns:
        Callable: energy function with signature (face_displacement, control_params) -> energy.
    """

    def combined_energy_fn(face_displacement: jnp.ndarray, control_params: ControlParams):
        # NOTE: Maybe there is a better way of doing this using a scan/loop. See https://github.com/google/jax/issues/673#issuecomment-894955037.
        # But, a for loop should be fine as the number of energy functions is small, so unrolling the loop should not be a problem.
        energy = jnp.array(0.)
        for energy_fn in energy_fns:
            energy += energy_fn(face_displacement, control_params)
        return energy

    return combined_energy_fn


def constrain_energy(energy_fn: Callable, constrained_kinematics: Callable):
    """Defines a constrained version of `energy_fn` according to `constrained_kinematics`.

    Args:
        energy_fn (Callable): Energy functional to be constrained.
        constrained_kinematics (Callable): Constraint function mapping the free DOFs and time to the displacement of all the faces. Normally, this is the output of `kineamtics.build_constrained_kinematics`.

    Returns:
        Callable: Constrained energy functional with signature (free_dofs, time, control_params) -> energy.
    """

    def constrained_energy_fn(free_DOFs, t, control_params: ControlParams):
        return energy_fn(
            constrained_kinematics(
                free_DOFs, t, control_params.constraint_params),
            control_params
        )

    return constrained_energy_fn


def compute_ligament_strains(face_displacement, centroid_node_vectors, bond_connectivity, reference_bond_vectors):
    node_displacements = face_to_node_kinematics(
        face_displacement,
        centroid_node_vectors
    ).reshape(-1, 3)
    return ligament_strains(node_displacements[bond_connectivity[:, 0]],
                            node_displacements[bond_connectivity[:, 1]],
                            reference_vector=reference_bond_vectors)


compute_ligament_strains_history = vmap(
    compute_ligament_strains, in_axes=(0, None, None, None)
)


def build_decompose_energy_fn(control_params: ControlParams, linearized_strains: bool = True, use_contact: bool = True, angle_based: bool = True) -> Callable:
    """Builds a function to decompose the total energy into its components.
    
    Args:
        control_params (ControlParams): physics control parameters.
        linearized_strains (bool, optional): whether to use linearized strains. Defaults to True.
        use_contact (bool, optional): whether to include contact energy. Defaults to True.
        angle_based (bool, optional): whether contact is angle-based. Defaults to True.
        
    Returns:
        Callable: A function `decompose_energy_fn(face_displacement)` that returns 
                  an array [E_stretch, E_shear, E_rot, E_contact].
    """
    face_centroids = control_params.geometrical_params.face_centroids
    centroid_node_vectors = control_params.geometrical_params.centroid_node_vectors
    bond_connectivity = control_params.geometrical_params.bond_connectivity
    bond_params = control_params.mechanical_params.bond_params
    
    n_faces, n_nodes_per_face, _ = centroid_node_vectors.shape
    ref_vecs = bond_params.reference_vector
    
    if use_contact:
        contact_params = control_params.mechanical_params.contact_params
        if angle_based:
            distance_fn = lambda x: void_angles(x, bond_connectivity)
        else:
            distance_fn = build_void_edge_distance(bond_connectivity)
            
    def decompose_energy_fn(face_displacement: jnp.ndarray) -> jnp.ndarray:
        node_displacements = face_to_node_kinematics(face_displacement, centroid_node_vectors).reshape(-1, 3)
        DOFs1 = node_displacements[bond_connectivity[:, 0]]
        DOFs2 = node_displacements[bond_connectivity[:, 1]]
        
        if linearized_strains:
            axial, shear, rot = ligament_strains_linearized(DOFs1, DOFs2, reference_vector=ref_vecs)
        else:
            axial, shear, rot = ligament_strains(DOFs1, DOFs2, reference_vector=ref_vecs)
            
        l0 = jnp.linalg.norm(ref_vecs, axis=-1)
        
        E_stretch = jnp.sum(bond_params.k_stretch * (axial * l0)**2 / 2)
        E_shear = jnp.sum(bond_params.k_shear * (shear * l0)**2 / 2)
        E_rot = jnp.sum(bond_params.k_rot * rot**2 / 2)
        
        E_contact = jnp.array(0.0)
        if use_contact:
            current_face_nodes = face_centroids[:, None] + centroid_node_vectors + node_displacements.reshape(n_faces, n_nodes_per_face, 3)[..., :2]
            E_contact = jnp.sum(contact_energy(
                current_void_angles=distance_fn(current_face_nodes), 
                **contact_params._asdict()
            ))
            
        return jnp.array([E_stretch, E_shear, E_rot, E_contact])

    return decompose_energy_fn


def build_energy_history(solution, control_params: ControlParams,
                          linearized_strains: bool = True,
                          use_contact: bool = True) -> dict:
    """Computes and packages the energy history across all load steps.

    Encapsulates the vmap + indexing logic so that pipeline.py can simply
    call this function and receive a ready-to-use dictionary.

    Args:
        solution: SolutionData with `.fields` (n_steps, n_faces, 3) and `.energies`.
        control_params (ControlParams): physics control parameters.
        linearized_strains (bool): whether to use linearized strains.
        use_contact (bool): whether to include contact energy.

    Returns:
        dict: {
            'total': total energy per step,
            'stretch': axial strain energy per step,
            'shear': shear strain energy per step,
            'rot': rotational strain energy per step,
            'contact': contact energy per step,
            'work': external work per step,
        }
    """
    import jax

    # Build the per-step decomposition function
    decompose_fn = build_decompose_energy_fn(
        control_params=control_params,
        linearized_strains=linearized_strains,
        use_contact=use_contact,
        angle_based=True,
    )

    # Vectorize over load steps: (n_steps, n_faces, 3) → (n_steps, 4)
    components = jax.vmap(decompose_fn)(solution.fields)

    # Internal energy = sum of all components at each step
    u_int = jnp.sum(components, axis=1)

    # External work derived from energy balance: W_ext = U_int - E_total
    w_ext = u_int - solution.energies

    return {
        'total':   solution.energies,
        'stretch': components[:, 0],
        'shear':   components[:, 1],
        'rot':     components[:, 2],
        'contact': components[:, 3],
        'work':    w_ext,
    }
