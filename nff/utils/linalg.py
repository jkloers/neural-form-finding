import jax.numpy as jnp
from jax import vmap
from typing import Tuple


def vdot(v1, v2):
    """Vectorized dot product based on *.

    Args:
        v1 (jnp.ndarray): Array of shape (Any, Any).
        v2 (jnp.ndarray): Array having the same shape as v1 or (v1.shape[1],).

    Returns:
        jnp.ndarray: row-wise dot product between v1 and v2
    """

    return jnp.sum(v1 * v2, axis=-1)


def rotation_matrix(angle):
    """Compute rotation matrix for a given angle"""

    return jnp.array([[jnp.cos(angle), -jnp.sin(angle)],
                      [jnp.sin(angle), jnp.cos(angle)]])


def compute_edge_unit_vectors(current_face_nodes: jnp.ndarray, node_id: int):
    """Computes unit vectors from bond node to the two closest nodes of the same face.

    Args:
        current_face_nodes (jnp.ndarray): array of shape (n_faces, n_nodes_per_face, 2) defining the position of all the faces' vertices.
        node_id (int): global node index.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: unit vectors.
    """

    _, n_sides, _ = current_face_nodes.shape

    node = current_face_nodes[node_id // n_sides, node_id % n_sides]

    unit_vector_1 = current_face_nodes[node_id // n_sides, (node_id+1) % n_sides] - node
    unit_vector_1 = unit_vector_1/jnp.linalg.norm(unit_vector_1)

    unit_vector_2 = current_face_nodes[node_id // n_sides, (node_id-1) % n_sides] - node
    unit_vector_2 = unit_vector_2/jnp.linalg.norm(unit_vector_2)

    return unit_vector_1, unit_vector_2


def angle_between_unit_vectors(u1, u2):
    """Computes the signed angle between two unit vectors using arctan2.

    Args:
        u1 (jnp.ndarray): array of shape (2, ) defining the first unit vector.
        u2 (jnp.ndarray): array of shape (2, ) defining the second unit vector.

    Returns:
        float: Signed angle measured from u1 to u2 (positive counter-clockwise). Result is in the range [-pi, pi].
    """
    return jnp.arctan2(u1[0] * u2[1] - u1[1] * u2[0], u1[0] * u2[0] + u1[1] * u2[1])


def compute_edge_angles(current_face_nodes: jnp.ndarray, nodes: Tuple[int, int]):
    """Computes the two face and two void angles.

    Args:
        current_face_nodes (jnp.ndarray): array of shape (n_faces, n_nodes_per_face, 2) defining the position of all the faces' vertices.
        nodes (Tuple[int, int]): tuple of node indices connected by a bond.

    Returns:
        Tuple[float, float, float, float]: void and face angles.
    """

    face_1_node_1, face_1_node_2 = compute_edge_unit_vectors(current_face_nodes, nodes[0])
    face_2_node_1, face_2_node_2 = compute_edge_unit_vectors(current_face_nodes, nodes[1])

    void_angle_1 = angle_between_unit_vectors(face_2_node_2, face_1_node_1)
    void_angle_2 = angle_between_unit_vectors(face_1_node_2, face_2_node_1)
    face_angle_1 = angle_between_unit_vectors(face_1_node_1, face_1_node_2)
    face_angle_2 = angle_between_unit_vectors(face_2_node_1, face_2_node_2)

    return void_angle_1, void_angle_2, face_angle_1, face_angle_2

def void_angles(current_face_nodes: jnp.ndarray, bond_connectivity: jnp.ndarray):
    """Computes angles between faces connected by the bonds.

    Args:
        current_face_nodes (jnp.ndarray): array of shape (n_faces, n_nodes_per_face, 2) defining the current position of the faces.
        bond_connectivity (jnp.ndarray): array of shape (n_bonds, 2) where each row [n1, n2] defines a bond connecting nodes n1 and n2.

    Returns:
        jnp.ndarray: array of shape (2*n_bonds,) defining the void angles.
    """

    angles = vmap(lambda bond: compute_edge_angles(
        current_face_nodes, bond))(bond_connectivity)
    void_angles = jnp.array(angles)[:2].ravel()

    return void_angles


def point_to_edge_distance(point: jnp.ndarray, edge: jnp.ndarray):
    """Computes the distance between a point and an edge.

    Args:
        point (jnp.ndarray): array of shape (2,) defining the point.
        edge (jnp.ndarray): array of shape (2, 2) defining the edge.

    Returns:
        jnp.ndarray: distance between the point and the edge.
    """

    x0 = edge[0]
    x1 = edge[1]
    t = jnp.dot(point-x0, x1-x0)/jnp.dot(x1-x0, x1-x0)
    x_distance_to_e = jnp.where(
        (t >= 0) & (t <= 1),
        # Projected point is on the edge
        jnp.sum((point-x0)**2 - (t*(x1-x0))**2)**0.5,
        jnp.where(
            # Projected point is outside the edge
            t < 0,
            # Distance to first point
            jnp.sum((point-x0)**2)**0.5,
            # Distance to second point
            jnp.sum((point-x1)**2)**0.5
        )
    )
    return x_distance_to_e


# Contact model based edge-to-edge distances
def edges_distance(edge_1: jnp.ndarray, edge_2: jnp.ndarray):
    """Computes the distance between two edges.

    Args:
        edge_1 (jnp.ndarray): array of shape (2, 2) defining the first edge.
        edge_2 (jnp.ndarray): array of shape (2, 2) defining the second edge.

    Returns:
        jnp.ndarray: scalar distance between the two edges.
    """

    # Compute the distance projecting second edge on the first edge
    e2_onto_e1_distance = vmap(
        point_to_edge_distance, in_axes=(0, None))(edge_2, edge_1)
    # Compute the distance projecting first edge on the second edge
    e1_onto_e2_distance = vmap(
        point_to_edge_distance, in_axes=(0, None))(edge_1, edge_2)
    # Return the minimum distance
    distances = jnp.concatenate((e2_onto_e1_distance, e1_onto_e2_distance))

    return jnp.min(distances)

# Vectorized version of edges_distance (vectorized over arrays of edges)
edges_distance_mapped = vmap(edges_distance, in_axes=(0, 0))


def build_void_edge_distance(bond_connectivity: jnp.ndarray):
    """Builds a function that computes the distance between edges connected by the bonds.

    Args:
        bond_connectivity (jnp.ndarray): array of shape (n_bonds, 2) where each row [n1, n2] defines a bond connecting nodes n1 and n2.

    Returns:
        Callable: function that computes all the pairwise distances between edges connected by the bonds.
    """

    def void_edge_distance(current_face_nodes: jnp.ndarray):
        """Computes the distance between edges connected by the bonds.

        Args:
            current_face_nodes (jnp.ndarray): array of shape (n_faces, n_nodes_per_face, 2) defining the current position of the faces.

        Returns:
            jnp.ndarray: array of shape (2*n_bonds,) defining the distances between edges connected by the bonds.
        """

        _, n_nodes_per_face, _ = current_face_nodes.shape
        nodes_1_id = bond_connectivity[:, 0]
        nodes_2_id = bond_connectivity[:, 1]
        pts1 = current_face_nodes[nodes_1_id //
                                   n_nodes_per_face, nodes_1_id % n_nodes_per_face]
        pts1_prev = current_face_nodes[nodes_1_id //
                                        n_nodes_per_face, (nodes_1_id-1) % n_nodes_per_face]
        pts1_next = current_face_nodes[nodes_1_id //
                                        n_nodes_per_face, (nodes_1_id+1) % n_nodes_per_face]

        pts2 = current_face_nodes[nodes_2_id //
                                   n_nodes_per_face, nodes_2_id % n_nodes_per_face]
        pts2_prev = current_face_nodes[nodes_2_id //
                                        n_nodes_per_face, (nodes_2_id-1) % n_nodes_per_face]
        pts2_next = current_face_nodes[nodes_2_id //
                                        n_nodes_per_face, (nodes_2_id+1) % n_nodes_per_face]

        # Distance between edges on one side of the bond
        void_distances1 = edges_distance_mapped(
            jnp.concatenate((pts1[:, None], pts1_next[:, None]), axis=1),
            jnp.concatenate((pts2[:, None], pts2_prev[:, None]), axis=1)
        )
        # Distance between edges on the other side of the bond
        void_distances2 = edges_distance_mapped(
            jnp.concatenate((pts1[:, None], pts1_prev[:, None]), axis=1),
            jnp.concatenate((pts2[:, None], pts2_next[:, None]), axis=1)
        )

        return jnp.concatenate((void_distances1, void_distances2))

    return void_edge_distance
