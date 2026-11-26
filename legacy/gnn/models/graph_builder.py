"""
CentroidalState → jraph.GraphsTuple conversion.

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

from nff.stages.state import CentroidalState


# ── Node feature dimensions ───────────────────────────────────────────────────
NODE_FEAT_DIM      = 7   # EGNN: invariant features only
NODE_FEAT_DIM_MPNN = 9   # MPNN: base + normalised initial positions


def build_static_graph_features(state: CentroidalState) -> dict:
    """Precompute fixed topology features from a CentroidalState.

    Call BEFORE the training loop. The result is closed over in forward_pipeline
    and must never contain JAX Tracers.

    Returns:
        dict with keys:
            'h_static'      — (n_faces, NODE_FEAT_DIM) NumPy float64
            'senders'       — (2*n_hinges,) NumPy int32  (bidirectional edges)
            'receivers'     — (2*n_hinges,) NumPy int32
            'n_nodes'       — int
            'n_edges'       — int
            'node_feat_dim' — int  (== NODE_FEAT_DIM)
            'edge_feat_dim' — int  (== 1, scalar distance)
    """
    n_faces = int(state.face_centroids.shape[0])

    # ── Static scalar node features ──────────────────────────────────────────
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

    # ── Bidirectional edges from hinge_face_pairs ────────────────────────────
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


def build_static_features(state: CentroidalState, map_type: str) -> dict:
    """Dispatcher: return the correct static features dict for a given map_type.

    Use this instead of calling the individual builders directly so that
    train.py and trainer.py stay in sync automatically.

    Args:
        state:    Flat CentroidalState (must be concrete, not inside JIT).
        map_type: The mapping type string (e.g. 'gnn_egnn', 'gnn_mpnn').

    Returns:
        Static features dict ready for apply_gnn_mapping / init_*_params.
    """
    if map_type == 'gnn_mpnn':
        return build_static_graph_features_mpnn(state)
    return build_static_graph_features(state)


def state_to_graph(
        state: CentroidalState,
        static_features: dict,
) -> jraph.GraphsTuple:
    """Build a jraph.GraphsTuple from the current state.

    Called at every forward pass. face_centroids may be a JAX Tracer inside JIT,
    enabling differentiable edge distance computation.

    Args:
        state:           Current CentroidalState (face_centroids may be a Tracer).
        static_features: Dict returned by build_static_graph_features.

    Returns:
        jraph.GraphsTuple with:
            nodes['h'] — (n_faces, NODE_FEAT_DIM)  static features (JAX constant)
            nodes['x'] — (n_faces, 2)              current centroid positions
            edges       — (n_edges, 1)              Euclidean distances
    """
    x         = state.face_centroids
    senders   = static_features['senders']    # NumPy — static indices
    receivers = static_features['receivers']

    # Edge feature: SO(2)-invariant distance scalar.
    diff = x[receivers] - x[senders]                         # (n_edges, 2)
    dist = jnp.linalg.norm(diff, axis=-1, keepdims=True)     # (n_edges, 1)

    return jraph.GraphsTuple(
        nodes={
            'h': jnp.asarray(static_features['h_static']),   # XLA compile-time constant
            'x': x,
        },
        edges=dist,
        senders=jnp.array(senders),
        receivers=jnp.array(receivers),
        globals=None,
        n_node=jnp.array([static_features['n_nodes']]),
        n_edge=jnp.array([static_features['n_edges']]),
    )
