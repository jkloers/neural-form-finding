"""Hinge-connectivity residual in centroidal coordinates (the loss's `hinge_gap` term).

The other open-pattern validity penalties and the L-BFGS objective built from them are archived
in legacy/open_pipeline/constraints.py.
"""

import jax.numpy as jnp

from nff.stages.geometry import hinge_vertex_positions


def hinge_connectivity(face_centroids, cnv, hinge_node_pairs):
    """Penalizes gap between vertices that should coincide at hinges.

    For each hinge vertex pair: c[i] + s[i][j] should equal c[k] + s[k][l].

    Args:
        face_centroids: (n_faces, 2)
        cnv: (n_faces, max_nodes, 2)
        hinge_node_pairs: (n_pairs, 2, 2)

    Returns:
        scalar — sum of squared distances between paired vertices.
    """
    p1, p2 = hinge_vertex_positions(face_centroids, cnv, hinge_node_pairs)
    return jnp.sum((p1 - p2)**2)

