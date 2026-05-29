"""
Hinge MPNN — message-passing GNN that predicts vertex displacements directly.

Instead of predicting centroid positions + a 2×2 face-deformation matrix
(as the standard MPNN does), this network predicts, for each face, the
displacement of each of its local vertices from their initial positions.

For hinge vertices (shared between two adjacent faces) the displacements
from both faces are averaged, guaranteeing that the two copies of the hinge
vertex end up at the same geometric position — hinge gap = 0 by construction.

Architecture (identical message-passing to mpnn.py):
    Embedding : h_i = tanh(h_raw_i @ W_emb + b_emb)

    Per layer l:
        1. Edge features : e_ij = [dx, dy, dist]         (static from initial positions)
        2. Message       : m_ij = phi_e([h_i | h_j | e_ij])
        3. Aggregation   : agg_i = (1/|N(i)|) Σ_j m_ij
        4. Node update   : h_i  = phi_h([h_i | agg_i])

    Output head (vertex displacements):
        delta_v = h_i @ vertex_W + vertex_b    → (max_nodes * 2,) per face
        reshaped → (max_nodes, 2) per-local-node displacement

    Bias initialized to zero → zero displacement at epoch 0.

Parameter layout:
    'emb_W'           : (node_feat_dim, hidden_dim)
    'emb_b'           : (hidden_dim,)
    Per layer l:
        'l{l}_phi_e_W1' : (2*hidden_dim + 3, hidden_dim)
        'l{l}_phi_e_b1' : (hidden_dim,)
        'l{l}_phi_e_W2' : (hidden_dim, hidden_dim)
        'l{l}_phi_e_b2' : (hidden_dim,)
        'l{l}_phi_h_W1' : (2*hidden_dim, hidden_dim)
        'l{l}_phi_h_b1' : (hidden_dim,)
        'l{l}_phi_h_W2' : (hidden_dim, hidden_dim)
        'l{l}_phi_h_b2' : (hidden_dim,)
    'vertex_W'        : (hidden_dim, max_nodes * 2)
    'vertex_b'        : (max_nodes * 2,)   — all zeros → identity at init
"""

import jax
import jax.numpy as jnp
import numpy as np

EDGE_FEAT_DIM = 3  # (dx, dy, dist)


def init_hinge_mpnn(
        key: jax.Array,
        node_feat_dim: int,
        hidden_dim: int = 32,
        num_layers: int = 2,
        max_nodes: int = 4,
        max_disp: float = 1.0,
) -> dict:
    """Initialise hinge-MPNN parameters.

    Args:
        key:           PRNG key.
        node_feat_dim: Node feature dimension (9 for gnn_hinge via MPNN base features).
        hidden_dim:    Width of all hidden layers.
        num_layers:    Number of message-passing layers.
        max_nodes:     Maximum number of local vertices per face (padded dimension).

    Returns:
        Flat dict PyTree compatible with optax / value_and_grad.
    """
    scale = 0.01
    n_keys = 1 + num_layers * 4 + 1  # emb + (phi_e x2, phi_h x2) per layer + vertex_head
    keys = jax.random.split(key, n_keys)
    ki = iter(keys)

    params = {}

    # ── Embedding ────────────────────────────────────────────────────────────
    params['emb_W'] = jax.random.normal(next(ki), (node_feat_dim, hidden_dim), dtype=jnp.float64) * scale
    params['emb_b'] = jnp.zeros(hidden_dim, dtype=jnp.float64)

    # ── Message-passing layers ────────────────────────────────────────────────
    edge_in_dim = 2 * hidden_dim + EDGE_FEAT_DIM

    for l in range(num_layers):
        params[f'l{l}_phi_e_W1'] = jax.random.normal(next(ki), (edge_in_dim, hidden_dim), dtype=jnp.float64) * scale
        params[f'l{l}_phi_e_b1'] = jnp.zeros(hidden_dim, dtype=jnp.float64)
        params[f'l{l}_phi_e_W2'] = jax.random.normal(next(ki), (hidden_dim, hidden_dim), dtype=jnp.float64) * scale
        params[f'l{l}_phi_e_b2'] = jnp.zeros(hidden_dim, dtype=jnp.float64)

        params[f'l{l}_phi_h_W1'] = jax.random.normal(next(ki), (2 * hidden_dim, hidden_dim), dtype=jnp.float64) * scale
        params[f'l{l}_phi_h_b1'] = jnp.zeros(hidden_dim, dtype=jnp.float64)
        params[f'l{l}_phi_h_W2'] = jax.random.normal(next(ki), (hidden_dim, hidden_dim), dtype=jnp.float64) * scale
        params[f'l{l}_phi_h_b2'] = jnp.zeros(hidden_dim, dtype=jnp.float64)

    # ── Vertex displacement head ──────────────────────────────────────────────
    # bias = 0 → zero displacement at initialization (tessellation unchanged)
    params['vertex_W'] = jax.random.normal(next(ki), (hidden_dim, max_nodes * 2), dtype=jnp.float64) * scale
    params['vertex_b'] = jnp.zeros(max_nodes * 2, dtype=jnp.float64)
    return params


def _mlp2(x, W1, b1, W2, b2):
    """Two-layer MLP with tanh activations."""
    return jnp.tanh(jnp.tanh(x @ W1 + b1) @ W2 + b2)


def apply_hinge_mpnn(
        params: dict,
        h_raw: jnp.ndarray,
        x_init: jnp.ndarray,
        senders_np: np.ndarray,
        receivers_np: np.ndarray,
        n_faces: int,
        max_nodes: int,
) -> jnp.ndarray:
    """Forward pass → per-face, per-local-node vertex displacements.

    Args:
        params:       Dict PyTree from init_hinge_mpnn.
        h_raw:        (n_faces, node_feat_dim) static node features.
        x_init:       (n_faces, 2) initial flat-tessellation centroid positions.
        senders_np:   (n_edges,) NumPy int32 — static sender indices.
        receivers_np: (n_edges,) NumPy int32 — static receiver indices.
        n_faces:      Python int.
        max_nodes:    Python int — padded number of local vertices per face.

    Returns:
        vertex_deltas: (n_faces, max_nodes, 2) — displacement of each local vertex
                       from its initial position. Zero at initialization.
    """
    num_layers = sum(1 for k in params if k.endswith('_phi_e_W1'))

    # ── Embedding ────────────────────────────────────────────────────────────
    h = jnp.tanh(h_raw @ params['emb_W'] + params['emb_b'])  # (n_faces, hidden_dim)

    # ── Static edge features ──────────────────────────────────────────────────
    diff = x_init[receivers_np] - x_init[senders_np]          # (n_edges, 2)
    dist = jnp.linalg.norm(diff, axis=-1, keepdims=True)      # (n_edges, 1)
    edge_feat = jnp.concatenate([diff, dist], axis=-1)        # (n_edges, 3)

    degree = jnp.zeros(n_faces, dtype=h.dtype).at[receivers_np].add(1.0)
    degree = jnp.maximum(degree, 1.0)[:, None]                # (n_faces, 1)

    # ── Message-passing layers ────────────────────────────────────────────────
    for l in range(num_layers):
        edge_input = jnp.concatenate(
            [h[senders_np], h[receivers_np], edge_feat], axis=-1)
        msg = _mlp2(
            edge_input,
            params[f'l{l}_phi_e_W1'], params[f'l{l}_phi_e_b1'],
            params[f'l{l}_phi_e_W2'], params[f'l{l}_phi_e_b2'],
        )
        msg_agg = (
            jnp.zeros((n_faces, msg.shape[-1]), dtype=msg.dtype)
            .at[receivers_np].add(msg)
        ) / degree

        h = _mlp2(
            jnp.concatenate([h, msg_agg], axis=-1),
            params[f'l{l}_phi_h_W1'], params[f'l{l}_phi_h_b1'],
            params[f'l{l}_phi_h_W2'], params[f'l{l}_phi_h_b2'],
        )

    # ── Vertex displacement head ──────────────────────────────────────────────
    delta_flat = h @ params['vertex_W'] + params['vertex_b']  # (n_faces, max_nodes*2)
    vertex_deltas = delta_flat.reshape((n_faces, max_nodes, 2))

    return vertex_deltas
