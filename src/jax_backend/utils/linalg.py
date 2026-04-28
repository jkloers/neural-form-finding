import jax.numpy as jnp
from jax import vmap, jit
from typing import Tuple, Union


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


def current_coordinates(vertices, centroids, angles, displacements):
    """
    Computes the deformed configuration coordinates.
    """

    def _current_coordinates(v, Q, c, d):
        return (Q @ v.T).T + c + d

    rotations = vmap(rotation_matrix)(angles)
    current_coordinates_v = vmap(_current_coordinates, in_axes=(0, 0, 0, 0))  # Vectorize over faces
    return current_coordinates_v(vertices, rotations, centroids, displacements)


def get_point_ids_in_bounding_box(points: jnp.ndarray, bounding_box: jnp.ndarray):
    """Returns the indices of the points that lie within the bounding box.

    Args:
        points (jnp.ndarray): array of shape (n_points, 2) collecting the coordinates of the points.
        bounding_box (jnp.ndarray): array of shape (2, 2) collecting the coordinates of the bounding box. The first row collects the coordinates of the bottom-left corner and the second row collects the coordinates of the top-right corner.

    Returns:
        jnp.ndarray: array of shape (n_points_in_bounding_box,) collecting the indices of the points that lie within the bounding box.
    """

    return jnp.where(
        (points[:, 0] >= bounding_box[0, 0]) & (points[:, 0] <= bounding_box[1, 0]) &
        (points[:, 1] >= bounding_box[0, 1]) & (points[:, 1] <= bounding_box[1, 1])
    )[0]


def get_point_ids_in_circle(points: jnp.ndarray, center: jnp.ndarray, radius: float):
    """Returns the indices of the points that lie within the circle.

    Args:
        points (jnp.ndarray): array of shape (n_points, 2) collecting the coordinates of the points.
        center (jnp.ndarray): array of shape (2,) collecting the coordinates of the center of the circle.
        radius (float): radius of the circle.

    Returns:
        jnp.ndarray: array of shape (n_points_in_circle,) collecting the indices of the points that lie within the circle.
    """

    return jnp.where(jnp.linalg.norm(points - center, axis=1) <= radius)[0]


def polygon_area(vertices: jnp.ndarray):
    """Computes area of a polygon with `vertices` ordered counter-clockwise.

    Args:
        vertices (jnp.ndarray): array of shape (n_vertices, 2).

    Returns:
        float: Area of the polygon.
    """

    v1 = jnp.roll(vertices, shift=1, axis=0)
    v2 = vertices

    return jnp.abs(jnp.sum(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]) / 2)


def polygon_centroid(vertices: jnp.ndarray):
    """Computes centroid of a polygon with `vertices` ordered counter-clockwise.

    Args:
        vertices (jnp.ndarray): array of shape (n_vertices, 2).

    Returns:
        jnp.ndarray: Centroid of the polygon.
    """

    area = polygon_area(vertices)
    v1 = jnp.roll(vertices, shift=1, axis=0)
    v2 = vertices
    x_plus_y = v1 + v2
    v_cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]

    return jnp.array([
        jnp.sum(x_plus_y[:, 0] * v_cross),
        jnp.sum(x_plus_y[:, 1] * v_cross)
    ]) / (6 * area)


@vmap
def polygons_geometric_properties(vertices: jnp.ndarray):
    """Computes area, centroid, and polar moment of area of an array of polygons defined by `vertices`.

    Args:
        vertices (jnp.ndarray): array of shape (n_faces, n_nodes_per_face, 2).

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: centroid and area of the polygons.
    """

    return polygon_centroid(vertices), polygon_area(vertices)



def compute_edge_unit_vectors(current_face_nodes: jnp.ndarray, node_id: int):
    """Computes unit vectors from bond node to the two closest nodes of the same face.

    Args:
        current_face_coordinates (jnp.ndarray): array of shape (n_faces, n_nodes_per_face, 2) defining the position of all the faces' vertices.
        node_id (int): global node index.

    Returns:
        Tuple[jnp.array, jnp.array]: void and face angles.
    """

    _, n_sides, _ = current_face_nodes.shape

    node = current_face_nodes[node_id // n_sides, node_id % n_sides]

    unit_vector_1 = current_face_nodes[node_id // n_sides, (node_id+1) % n_sides] - node
    unit_vector_1 = unit_vector_1/jnp.linalg.norm(unit_vector_1)

    unit_vector_2 = current_face_nodes[node_id // n_sides, (node_id-1) % n_sides] - node
    unit_vector_2 = unit_vector_2/jnp.linalg.norm(unit_vector_2)

    return unit_vector_1, unit_vector_2


def compute_edge_lengths(centroid_node_vectors: jnp.ndarray):
    """Computes edge lengths of the faces.

    Args:
        centroid_node_vectors (jnp.ndarray): array of shape (n_faces, n_nodes_per_face, 2) defining the position of all the faces' vertices relative to the centroids.

    Returns:
        jnp.ndarray: array of shape (n_faces, n_nodes_per_face) collecting the edge lengths of the faces.
    """

    return jnp.linalg.norm(
        jnp.roll(centroid_node_vectors, 1, axis=1) - centroid_node_vectors,
        axis=2
    )


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
        current_face_coordinates (jnp.ndarray): array of shape (n_faces, n_nodes_per_face, 2) defining the position of all the faces' vertices.
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


def compute_xy_limits(points: jnp.ndarray):
    """Computes the the pair xlim, ylim for the given set of points.

    Args:
        points (jnp.ndarray): array of shape (n, 2)

    Returns:
        jnp.ndarray: array of xlim, ylim
    """

    return jnp.array([points.min(axis=0), points.max(axis=0)]).T
