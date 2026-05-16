"""
Conversion de CentroidalState vers jraph.GraphsTuple.

Deux fonctions distinctes :
  - build_static_graph_features : calcule UNE SEULE FOIS les données
    topologiques fixes (conditions aux limites, chargements, aires initiales).
    Ne contient aucun JAX Tracer — sûr à utiliser en dehors de JIT.

  - state_to_graph : appelée à chaque forward pass, construit le GraphsTuple
    complet en combinant les features statiques et les positions courantes
    (live JAX arrays). C'est la seule fonction à appeler dans le pipeline JIT.

Interface du graphe (convention EGNN) :
  nodes = { 'h': (n_faces, 7),   features statiques invariants
            'x': (n_faces, 2) }  positions courantes (différentiables)
  edges = (n_edges, 1)           distance euclidienne (scalaire équivariant)
  senders / receivers            arêtes bidirectionnelles depuis hinge_face_pairs
"""

import numpy as np
import jax.numpy as jnp
import jraph

from jax_backend.state import CentroidalState


# ── Layout des features de nœuds ─────────────────────────────────────────────
# Index  Signification
#   0    density
#   1    initial_face_area
#   2    is_boundary  (1 si le nœud est sur la frontière libre)
#   3    is_clamped   (1 si Dirichlet actif)
#   4    load_x       (force imposée en x)
#   5    load_y       (force imposée en y)
#   6    load_theta   (moment imposé)
NODE_FEAT_DIM = 7


def build_static_graph_features(state: CentroidalState) -> dict:
    """Pré-calcule les features topologiques fixes à partir d'un CentroidalState.

    Appeler AVANT la boucle d'entraînement. Le résultat est capturé en closure
    dans forward_pipeline et ne doit jamais contenir de JAX Tracers.

    Returns:
        dict avec les clés :
            'h_static'      — (n_faces, NODE_FEAT_DIM) NumPy float64
            'senders'       — (2*n_hinges,) NumPy int32  (arêtes bidirectionnelles)
            'receivers'     — (2*n_hinges,) NumPy int32
            'n_nodes'       — int
            'n_edges'       — int
            'node_feat_dim' — int  (== NODE_FEAT_DIM)
            'edge_feat_dim' — int  (== 1, distance scalaire)
    """
    n_faces = int(state.face_centroids.shape[0])

    # ── Features scalaires statiques ─────────────────────────────────────────
    is_boundary = np.zeros(n_faces, dtype=np.float64)
    is_clamped  = np.zeros(n_faces, dtype=np.float64)
    load_vec    = np.zeros((n_faces, 3), dtype=np.float64)

    if len(state.boundary_face_node_ids) > 0:
        boundary_ids = np.unique(
            np.array(state.boundary_face_node_ids, dtype=np.int32)[:, 0]
        )
        is_boundary[boundary_ids] = 1.0

    if len(state.constrained_face_DOF_pairs) > 0:
        clamped_ids = np.unique(
            np.array(state.constrained_face_DOF_pairs, dtype=np.int32)[:, 0]
        )
        is_clamped[clamped_ids] = 1.0

    if len(state.loaded_face_DOF_pairs) > 0:
        load_vals_np = np.array(state.load_values, dtype=np.float64)
        for i, (face_id, dof_id) in enumerate(
                np.array(state.loaded_face_DOF_pairs, dtype=np.int32)):
            load_vec[face_id, dof_id] = load_vals_np[i]

    density_np      = np.array(state.density, dtype=np.float64).reshape(-1, 1)
    init_areas_np   = np.array(state.initial_face_areas, dtype=np.float64).reshape(-1, 1)

    h_static = np.concatenate([
        density_np,            # col 0
        init_areas_np,         # col 1
        is_boundary[:, None],  # col 2
        is_clamped[:, None],   # col 3
        load_vec,              # cols 4-6
    ], axis=-1).astype(np.float64)  # (n_faces, NODE_FEAT_DIM)

    # ── Arêtes bidirectionnelles depuis hinge_face_pairs ─────────────────────
    hfp = np.array(state.hinge_face_pairs, dtype=np.int32)  # (n_hinges, 2)
    senders   = np.concatenate([hfp[:, 0], hfp[:, 1]]).astype(np.int32)
    receivers = np.concatenate([hfp[:, 1], hfp[:, 0]]).astype(np.int32)

    return {
        'h_static':      h_static,
        'senders':       senders,
        'receivers':     receivers,
        'n_nodes':       n_faces,
        'n_edges':       len(senders),
        'node_feat_dim': NODE_FEAT_DIM,
        'edge_feat_dim': 1,
    }


def state_to_graph(
        state: CentroidalState,
        static_features: dict,
) -> jraph.GraphsTuple:
    """Construit un jraph.GraphsTuple à partir de l'état courant.

    Appelée à chaque forward pass. Les positions `x = state.face_centroids`
    sont des JAX arrays (potentiellement des Tracers dans JIT), ce qui permet
    de calculer des distances d'arêtes différentiables.

    Le GraphsTuple résultant est l'interface formelle attendue par l'EGNN.

    Args:
        state:           CentroidalState courant (face_centroids peut être Tracer).
        static_features: Dict renvoyé par build_static_graph_features.

    Returns:
        jraph.GraphsTuple avec :
            nodes['h'] — (n_faces, NODE_FEAT_DIM)  features statiques (cast JAX)
            nodes['x'] — (n_faces, 2)              positions courantes
            edges       — (n_edges, 1)              distances euclidiennes
    """
    x         = state.face_centroids                          # JAX Tracer ou concret
    senders   = static_features['senders']                    # NumPy — indices statiques
    receivers = static_features['receivers']

    # Feature d'arête : distance (scalaire SO(2)-équivariant)
    diff = x[receivers] - x[senders]                         # (n_edges, 2)
    dist = jnp.linalg.norm(diff, axis=-1, keepdims=True)     # (n_edges, 1)

    return jraph.GraphsTuple(
        nodes={
            'h': jnp.asarray(static_features['h_static']),   # constante baked-in JIT
            'x': x,
        },
        edges=dist,
        senders=jnp.array(senders),
        receivers=jnp.array(receivers),
        globals=None,
        n_node=jnp.array([static_features['n_nodes']]),
        n_edge=jnp.array([static_features['n_edges']]),
    )
