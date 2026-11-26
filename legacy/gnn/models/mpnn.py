"""
MPNN — Non-equivariant Message Passing Neural Network for neural form-finding.

Unlike the EGNN, this network does NOT enforce E(2) equivariance.  It uses
full directional edge features (dx, dy, dist) computed from the initial flat
tessellation positions, and absolute positional encodings in node features.
This gives each tile awareness of *where* it sits in the tessellation and
*which direction* its neighbours lie — information the EGNN cannot use.

Architecture (L layers):
    Embedding : h_i = tanh(h_raw_i @ W_emb + b_emb)

    Per layer l:
        1. Edge features : e_ij = [x_i-x_j, ||x_i-x_j||]     (dx, dy, dist) — static
        2. Message       : m_ij = phi_e([h_i | h_j | e_ij])   (2-layer MLP, tanh)
        3. Aggregation   : agg_i = (1/|N(i)|) Σ_j m_ij
        4. Node update   : h_i  = phi_h([h_i | agg_i])        (2-layer MLP, tanh)

Output heads (both init to identity / zero):
    Coordinate : x_new_i = x_init_i + h_i @ coord_W + coord_b
    Transform  : T_i = reshape(h_i @ transform_W + transform_b, 2, 2)

Key differences from EGNN:
    - Edge features include direction (dx, dy), not just distance²
    - Node features include normalized initial centroid position (x_norm, y_norm)
    - Coordinate output is a direct displacement, not an equivariant weighted sum
    - No E(2) equivariance — full positional and directional awareness

Parameter layout (flat dict PyTree for optax / value_and_grad):
    'emb_W'           : (node_feat_dim, hidden_dim)
    'emb_b'           : (hidden_dim,)
    Per layer l:
        'l{l}_phi_e_W1' : (2*hidden_dim + EDGE_FEAT_DIM, hidden_dim)
        'l{l}_phi_e_b1' : (hidden_dim,)
        'l{l}_phi_e_W2' : (hidden_dim, hidden_dim)
        'l{l}_phi_e_b2' : (hidden_dim,)
        'l{l}_phi_h_W1' : (2*hidden_dim, hidden_dim)
        'l{l}_phi_h_b1' : (hidden_dim,)
        'l{l}_phi_h_W2' : (hidden_dim, hidden_dim)
        'l{l}_phi_h_b2' : (hidden_dim,)
    'coord_W'         : (hidden_dim, 2)   — init near zero → x_new ≈ x_init
    'coord_b'         : (2,)
    'transform_W'     : (hidden_dim, 4)   — init near zero
    'transform_b'     : (4,)              — init [1, 0, 0, 1] → identity matrix
"""

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int

EDGE_FEAT_DIM = 3  # (dx, dy, dist) — direction + magnitude of initial relative position


def init_mpnn(
        key: jax.Array,
        node_feat_dim: int,
        hidden_dim: int = 32,
        num_layers: int = 2,
        inner_depth: int = 2,
) -> dict:
    """Initialise MPNN parameters.

    Args:
        key:           PRNG key.
        node_feat_dim: Node feature dimension (9 when built by build_static_graph_features_mpnn).
        hidden_dim:    Width of all hidden layers.
        num_layers:    Number of message-passing layers.

    Returns:
        Flat dict PyTree compatible with optax / value_and_grad.
    """
    scale = 0.01
    keys_per_layer = 2 if inner_depth == 1 else 4   # (phi_e, phi_h) each have 1 or 2 weight matrices
    n_keys = 1 + num_layers * keys_per_layer + 2
    keys = jax.random.split(key, n_keys)
    ki = iter(keys)

    params = {}

    # ── Embedding ────────────────────────────────────────────────────────────────
    params['emb_W'] = jax.random.normal(next(ki), (node_feat_dim, hidden_dim), dtype=jnp.float64) * scale
    params['emb_b'] = jnp.zeros(hidden_dim, dtype=jnp.float64)

    # ── Message-passing layers ────────────────────────────────────────────────────
    edge_in_dim = 2 * hidden_dim + EDGE_FEAT_DIM

    for l in range(num_layers):
        # phi_e: [h_i | h_j | e_ij] → message
        params[f'l{l}_phi_e_W1'] = jax.random.normal(next(ki), (edge_in_dim, hidden_dim), dtype=jnp.float64) * scale
        params[f'l{l}_phi_e_b1'] = jnp.zeros(hidden_dim, dtype=jnp.float64)
        if inner_depth == 2:
            params[f'l{l}_phi_e_W2'] = jax.random.normal(next(ki), (hidden_dim, hidden_dim), dtype=jnp.float64) * scale
            params[f'l{l}_phi_e_b2'] = jnp.zeros(hidden_dim, dtype=jnp.float64)

        # phi_h: [h_i | agg_i] → h_i next
        params[f'l{l}_phi_h_W1'] = jax.random.normal(next(ki), (2 * hidden_dim, hidden_dim), dtype=jnp.float64) * scale
        params[f'l{l}_phi_h_b1'] = jnp.zeros(hidden_dim, dtype=jnp.float64)
        if inner_depth == 2:
            params[f'l{l}_phi_h_W2'] = jax.random.normal(next(ki), (hidden_dim, hidden_dim), dtype=jnp.float64) * scale
            params[f'l{l}_phi_h_b2'] = jnp.zeros(hidden_dim, dtype=jnp.float64)

    # ── Output heads ─────────────────────────────────────────────────────────────
    # Coordinate: predicts displacement from initial position; bias = 0 → identity at init
    params['coord_W'] = jax.random.normal(next(ki), (hidden_dim, 2), dtype=jnp.float64) * scale
    params['coord_b'] = jnp.zeros(2, dtype=jnp.float64)

    # Transform: 2×2 tile deformation matrix; bias = [1,0,0,1] → identity at init
    params['transform_W'] = jax.random.normal(next(ki), (hidden_dim, 4), dtype=jnp.float64) * scale
    params['transform_b'] = jnp.array([1.0, 0.0, 0.0, 1.0], dtype=jnp.float64)

    return params


def _mlp1(x, W1, b1):
    """Single-layer MLP with tanh activation."""
    return jnp.tanh(x @ W1 + b1)


def _mlp2(x, W1, b1, W2, b2):
    """Two-layer MLP with tanh activations."""
    return jnp.tanh(jnp.tanh(x @ W1 + b1) @ W2 + b2)


def apply_mpnn(
        params: dict,
        h_raw: Float[Array, "n_faces node_feat_dim"],
        x_init: Float[Array, "n_faces 2"],
        senders_np: Int[np.ndarray, "n_edges"],
        receivers_np: Int[np.ndarray, "n_edges"],
        n_faces: int,
        num_layers: int,
        inner_depth: int = 2,
) -> tuple[Float[Array, "n_faces 2"], Float[Array, "n_faces hidden_dim"], Float[Array, "n_faces 2 2"]]:
    """Forward pass MPNN → new centroid positions and tile deformation matrices.

    Edge features are computed once from x_init (static throughout the forward
    pass).  The coordinate output is a direct linear displacement from x_init,
    so the network predicts *where* each face goes rather than nudging them
    incrementally as the EGNN does.

    Args:
        params:       Dict PyTree from init_mpnn.
        h_raw:        (n_faces, node_feat_dim) node features incl. positional encoding.
        x_init:       (n_faces, 2) initial flat-tessellation centroid positions.
        senders_np:   (n_edges,) NumPy int32 — static sender indices.
        receivers_np: (n_edges,) NumPy int32 — static receiver indices.
        n_faces:      Python int.
        num_layers:   Python int — number of message-passing layers.

    Returns:
        (x_new, h_new, local_transform) where:
          x_new           — (n_faces, 2)         new centroid positions
          h_new           — (n_faces, hidden_dim) final node features
          local_transform — (n_faces, 2, 2)       tile deformation (identity at init)
    """

    # ── Embedding ────────────────────────────────────────────────────────────────
    h = jnp.tanh(h_raw @ params['emb_W'] + params['emb_b'])   # (n_faces, hidden_dim)

    # ── Static edge features from initial positions ───────────────────────────────
    # (dx, dy) encodes direction — the key non-equivariant signal
    diff = x_init[receivers_np] - x_init[senders_np]           # (n_edges, 2)
    dist = jnp.linalg.norm(diff, axis=-1, keepdims=True)       # (n_edges, 1)
    edge_feat = jnp.concatenate([diff, dist], axis=-1)         # (n_edges, 3)

    # Normalised incoming degree for mean aggregation
    degree = jnp.zeros(n_faces, dtype=h.dtype).at[receivers_np].add(1.0)
    degree = jnp.maximum(degree, 1.0)[:, None]                 # (n_faces, 1)

    # ── Message-passing layers ────────────────────────────────────────────────────
    for l in range(num_layers):
        # Compute messages from sender/receiver features + edge geometry
        edge_input = jnp.concatenate(
            [h[senders_np], h[receivers_np], edge_feat], axis=-1)  # (n_edges, 2*hid+3)
        if inner_depth == 1:
            msg = _mlp1(edge_input, params[f'l{l}_phi_e_W1'], params[f'l{l}_phi_e_b1'])
        else:
            msg = _mlp2(edge_input,
                        params[f'l{l}_phi_e_W1'], params[f'l{l}_phi_e_b1'],
                        params[f'l{l}_phi_e_W2'], params[f'l{l}_phi_e_b2'])

        # Mean-aggregate incoming messages
        msg_agg = (
            jnp.zeros((n_faces, msg.shape[-1]), dtype=msg.dtype)
            .at[receivers_np].add(msg)
        ) / degree                                                  # (n_faces, hidden_dim)

        # Update node features
        node_input = jnp.concatenate([h, msg_agg], axis=-1)
        if inner_depth == 1:
            h = _mlp1(node_input, params[f'l{l}_phi_h_W1'], params[f'l{l}_phi_h_b1'])
        else:
            h = _mlp2(node_input,
                      params[f'l{l}_phi_h_W1'], params[f'l{l}_phi_h_b1'],
                      params[f'l{l}_phi_h_W2'], params[f'l{l}_phi_h_b2'])

    # ── Output heads ─────────────────────────────────────────────────────────────
    # Coordinate: predict displacement; near-zero init → x_new ≈ x_init at epoch 0
    delta_x = h @ params['coord_W'] + params['coord_b']            # (n_faces, 2)
    x_new = x_init + delta_x

    # Tile deformation matrix: identity at init
    transform_flat = h @ params['transform_W'] + params['transform_b']
    local_transform = transform_flat.reshape((n_faces, 2, 2))

    return x_new, h, local_transform
