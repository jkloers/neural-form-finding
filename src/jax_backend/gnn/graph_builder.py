"""
Conversion de CentroidalState vers jraph.GraphsTuple.

Three public functions:
  - build_static_graph_features      : base features for EGNN (7-dim node features,
                                        distance-only edge features).
  - build_static_graph_features_mpnn : extends the base with 2 normalised initial
                                        centroid positions → 9-dim node features for
                                        the non-equivariant MPNN.
  - build_static_features            : dispatcher keyed on map_type — always call this
                                        when you don't want to hard-code the architecture.

  - state_to_graph : called at every forward pass; combines static features with live
                     JAX centroid positions (may be Tracers inside JIT).

Node feature layout (base, shared by both architectures):
  Index  Field
    0    density
    1    initial_face_area
    2    is_boundary  (1 if face is on the free boundary)
    3    is_clamped   (1 if Dirichlet BC active)
    4    load_x
    5    load_y
    6    load_theta
MPNN adds two more columns:
    7    x_norm  (normalised initial centroid x ∈ [-1, 1])
    8    y_norm  (normalised initial centroid y ∈ [-1, 1])
"""

import numpy as np
import jax.numpy as jnp
import jraph

from jax_backend.state import CentroidalState


# ── Node feature dimensions ───────────────────────────────────────────────────
NODE_FEAT_DIM      = 7   # EGNN: invariant features only
NODE_FEAT_DIM_MPNN = 9   # MPNN: base + normalised initial positions


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


def build_static_graph_features_mpnn(state: CentroidalState) -> dict:
    """Like build_static_graph_features, but adds normalised initial centroid
    positions as extra node features (columns 7-8).

    The normalisation maps the flat-tessellation bounding box to [-1, 1],
    giving the MPNN absolute positional awareness without needing equivariance.

    Returns the same dict layout as build_static_graph_features, but with:
        'h_static'      — (n_faces, NODE_FEAT_DIM_MPNN)  i.e. shape (n, 9)
        'node_feat_dim' — NODE_FEAT_DIM_MPNN  (9)
    All other keys are identical.
    """
    base = build_static_graph_features(state)

    centroids = np.array(state.face_centroids, dtype=np.float64)   # (n_faces, 2)
    center = centroids.mean(axis=0)
    half_range = np.abs(centroids - center).max()
    half_range = max(half_range, 1e-6)
    pos_norm = (centroids - center) / half_range                    # (n_faces, 2) ∈ [-1,1]

    h_mpnn = np.concatenate([base['h_static'], pos_norm], axis=-1)  # (n_faces, 9)

    return {
        **base,
        'h_static':      h_mpnn,
        'node_feat_dim': NODE_FEAT_DIM_MPNN,
    }


def build_static_features_hinge(state: CentroidalState, tessellation) -> dict:
    """Precompute static features for the gnn_hinge parametrization.

    Extends the MPNN base features with vertex-level topology needed to
    reconstruct a CentroidalState from per-face vertex displacement predictions.

    Args:
        state:       Flat CentroidalState (concrete, not inside JIT).
        tessellation: Tessellation object with .vertices, .faces, .hinges.

    Returns:
        Dict with all MPNN base keys plus hinge/boundary vertex arrays.
    """
    base = build_static_graph_features_mpnn(state)

    verts = np.array(tessellation.vertices, dtype=np.float64)  # (n_total_verts, 2)
    n_faces   = len(tessellation.faces)
    max_nodes = max(len(f.vertex_indices) for f in tessellation.faces)

    # ── face_vertex_indices + node_mask (same as build_direct_vertex_features) ─
    fvi  = np.zeros((n_faces, max_nodes), dtype=np.int32)
    mask = np.zeros((n_faces, max_nodes), dtype=np.float64)
    for i, face in enumerate(tessellation.faces):
        n = len(face.vertex_indices)
        fvi[i, :n]  = face.vertex_indices
        mask[i, :n] = 1.0

    # ── Hinge vertex arrays ───────────────────────────────────────────────────
    # hinge_node_pairs: (n_hinges, 2, 2) = [[face_i, local_j], [face_k, local_l]]
    hnp = np.array(state.hinge_node_pairs, dtype=np.int32)  # (n_hinges, 2, 2)
    h_face1  = hnp[:, 0, 0]   # face owning vertex1
    h_local1 = hnp[:, 0, 1]   # local node index of vertex1 in face1
    h_face2  = hnp[:, 1, 0]   # face owning vertex2
    h_local2 = hnp[:, 1, 1]   # local node index of vertex2 in face2

    h_v1 = np.array([h.vertex1 for h in tessellation.hinges], dtype=np.int32)
    h_v2 = np.array([h.vertex2 for h in tessellation.hinges], dtype=np.int32)

    # ── Boundary vertex arrays ────────────────────────────────────────────────
    hinge_verts = set(h_v1.tolist()) | set(h_v2.tolist())
    n_verts = len(verts)
    # v_to_fn: vertex_global_id -> [face_id, local_node_id]
    v_to_fn = np.full((n_verts, 2), -1, dtype=np.int32)
    for i, face in enumerate(tessellation.faces):
        for j, v in enumerate(face.vertex_indices):
            v_to_fn[v] = [i, j]

    bv_global_list = [v for v in range(n_verts)
                      if v not in hinge_verts and v_to_fn[v, 0] != -1]
    bv_global = np.array(bv_global_list, dtype=np.int32)
    bv_face   = v_to_fn[bv_global, 0]
    bv_local  = v_to_fn[bv_global, 1]

    # ── Per-(face, local_node) partner lookup for JIT-safe averaging ──────────
    # For hinge nodes: partner_{face,local} points to the other face's copy.
    # For non-hinge nodes: self-reference (partner == self → no effect when masked).
    # is_hinge_node flags which (face, local) entries need averaging.
    partner_face  = np.zeros((n_faces, max_nodes), dtype=np.int32)
    partner_local = np.zeros((n_faces, max_nodes), dtype=np.int32)
    is_hinge_node = np.zeros((n_faces, max_nodes), dtype=np.float64)
    for f in range(n_faces):
        for n in range(max_nodes):
            partner_face[f, n]  = f
            partner_local[f, n] = n

    for i in range(len(h_face1)):
        f1, l1, f2, l2 = int(h_face1[i]), int(h_local1[i]), int(h_face2[i]), int(h_local2[i])
        partner_face[f1, l1]  = f2;  partner_local[f1, l1] = l2;  is_hinge_node[f1, l1] = 1.0
        partner_face[f2, l2]  = f1;  partner_local[f2, l2] = l1;  is_hinge_node[f2, l2] = 1.0

    return {
        **base,
        'face_vertex_indices':      fvi,
        'node_mask':                mask,
        'initial_vertex_positions': verts,
        'h_v1':    h_v1,
        'h_v2':    h_v2,
        'h_face1': h_face1,
        'h_local1': h_local1,
        'h_face2': h_face2,
        'h_local2': h_local2,
        'bv_global': bv_global,
        'bv_face':   bv_face,
        'bv_local':  bv_local,
        'partner_face':  partner_face,
        'partner_local': partner_local,
        'is_hinge_node': is_hinge_node,
        'max_nodes':    max_nodes,
        'n_total_verts': n_verts,
    }


def build_static_features(state: CentroidalState, map_type: str,
                           tessellation=None) -> dict:
    """Dispatcher: return the correct static features dict for a given map_type.

    Use this instead of calling the individual builders directly so that
    train.py and trainer.py stay in sync automatically.

    Args:
        state:       Flat CentroidalState (must be concrete, not inside JIT).
        map_type:    The mapping type string (e.g. 'gnn_egnn', 'gnn_mpnn').
        tessellation: Required for 'gnn_hinge'; ignored otherwise.

    Returns:
        Static features dict ready for apply_gnn_mapping / init_*_params.
    """
    if map_type == 'gnn_hinge':
        if tessellation is None:
            raise ValueError("tessellation is required for map_type='gnn_hinge'")
        return build_static_features_hinge(state, tessellation)
    if map_type == 'gnn_mpnn':
        return build_static_graph_features_mpnn(state)
    return build_static_graph_features(state)


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
