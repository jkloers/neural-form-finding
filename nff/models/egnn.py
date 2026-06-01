"""
EGNN — E(2)-Equivariant Graph Neural Network for neural form-finding.

Architecture (L layers):
    Embedding : h_i = tanh(x_raw_i @ W_emb + b_emb)

    Per layer l:
        1. Edge invariant  : d²_ij = ||x_i - x_j||²
        2. Message         : m_ij  = φ_e([h_i | h_j | d²_ij])     (2-layer MLP, tanh)
        3. Coord update    : x_i  += (1/|N(i)|) Σ_j (x_i - x_j) * φ_x(m_ij)  (equivariant)
        4. Message aggreg  : agg_i = Σ_j m_ij
        5. Node update     : h_i   = φ_h([h_i | agg_i])            (2-layer MLP, tanh)

E(2) equivariance:
    - d²_ij is invariant under rotation/translation → φ_e produces invariant messages.
    - (x_i - x_j) is covariant (relative vector); scaled by scalar φ_x(m_ij) → equivariant δx.
    - h_i depends only on invariant quantities → h_out is invariant.

Parameter layout (flat dict, JAX PyTree):
    'emb_W'  : (node_feat_dim, hidden_dim)
    'emb_b'  : (hidden_dim,)
    Per layer l (0-indexed):
        'l{l}_phi_e_W1' : (2*hidden_dim + 1, hidden_dim)   (+1 for d²_ij)
        'l{l}_phi_e_b1' : (hidden_dim,)
        'l{l}_phi_e_W2' : (hidden_dim, hidden_dim)
        'l{l}_phi_e_b2' : (hidden_dim,)
        'l{l}_phi_x_W'  : (hidden_dim, 1)
        'l{l}_phi_x_b'  : (1,)
        'l{l}_phi_h_W1' : (2*hidden_dim, hidden_dim)
        'l{l}_phi_h_b1' : (hidden_dim,)
        'l{l}_phi_h_W2' : (hidden_dim, hidden_dim)
        'l{l}_phi_h_b2' : (hidden_dim,)
"""

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int


def init_egnn(
        key: jax.Array,
        node_feat_dim: int,
        hidden_dim: int = 32,
        num_layers: int = 2,
) -> dict:
    """Initialise an L-layer EGNN parameter dict.

    Args:
        key:           JAX PRNG key.
        node_feat_dim: Static node feature dimension (e.g. 7).
        hidden_dim:    Width of all hidden layers.
        num_layers:    Number of EGNN message-passing layers.

    Returns:
        Flat dict PyTree compatible with optax / value_and_grad.
    """
    scale = 0.01  # near-identity init — non-zero gradient from epoch 0
    # phi_x controls coordinate displacement; larger init gives better gradient signal
    # at the start of training (faces need to move ~O(1) but default scale gives ~0.01)
    scale_phi_x = 0.01
    keys = jax.random.split(key, 3 + num_layers * 5)
    ki = iter(keys)

    params = {}

    # Embedding : raw node features → hidden_dim
    params['emb_W'] = jax.random.normal(next(ki), (node_feat_dim, hidden_dim), dtype=jnp.float64) * scale
    params['emb_b'] = jnp.zeros(hidden_dim, dtype=jnp.float64)

    for l in range(num_layers):
        # φ_e : [h_i | h_j | d²_ij] → message  (dim: 2*hidden_dim + 1 → hidden_dim)
        params[f'l{l}_phi_e_W1'] = jax.random.normal(next(ki), (2 * hidden_dim + 1, hidden_dim), dtype=jnp.float64) * scale
        params[f'l{l}_phi_e_b1'] = jnp.zeros(hidden_dim, dtype=jnp.float64)
        params[f'l{l}_phi_e_W2'] = jax.random.normal(next(ki), (hidden_dim, hidden_dim), dtype=jnp.float64) * scale
        params[f'l{l}_phi_e_b2'] = jnp.zeros(hidden_dim, dtype=jnp.float64)

        # φ_x : message → coord weighting scalar  (dim: hidden_dim → 1)
        params[f'l{l}_phi_x_W'] = jax.random.normal(next(ki), (hidden_dim, 1), dtype=jnp.float64) * scale_phi_x
        params[f'l{l}_phi_x_b'] = jnp.zeros(1, dtype=jnp.float64)

        # φ_h : [h_i | agg_i] → next h_i  (dim: 2*hidden_dim → hidden_dim)
        params[f'l{l}_phi_h_W1'] = jax.random.normal(next(ki), (2 * hidden_dim, hidden_dim), dtype=jnp.float64) * scale
        params[f'l{l}_phi_h_b1'] = jnp.zeros(hidden_dim, dtype=jnp.float64)
        params[f'l{l}_phi_h_W2'] = jax.random.normal(next(ki), (hidden_dim, hidden_dim), dtype=jnp.float64) * scale
        params[f'l{l}_phi_h_b2'] = jnp.zeros(hidden_dim, dtype=jnp.float64)

    # Output head: local 2×2 transformation matrix (invariant).
    # transform_b initialised to identity [1, 0, 0, 1].
    params['transform_W'] = jax.random.normal(next(ki), (hidden_dim, 4), dtype=jnp.float64) * scale
    params['transform_b'] = jnp.array([1.0, 0.0, 0.0, 1.0], dtype=jnp.float64)

    return params


def _mlp2(x, W1, b1, W2, b2):
    """Two-layer MLP with tanh activations."""
    return jnp.tanh(jnp.tanh(x @ W1 + b1) @ W2 + b2)


def apply_egnn(
        params: dict,
        h_raw: Float[Array, "n_faces node_feat_dim"],
        x: Float[Array, "n_faces 2"],
        senders_np: Int[np.ndarray, "n_edges"],
        receivers_np: Int[np.ndarray, "n_edges"],
        n_faces: int,
        num_layers: int,
) -> tuple[Float[Array, "n_faces 2"], Float[Array, "n_faces hidden_dim"], Float[Array, "n_faces 2 2"]]:
    """Forward pass EGNN → updated positions and node features.

    Args:
        params:       Dict PyTree from init_egnn.
        h_raw:        (n_faces, node_feat_dim) static node features.
        x:            (n_faces, 2) current positions (JAX Tracer inside JIT).
        senders_np:   (n_edges,) NumPy int32 — static sender indices.
        receivers_np: (n_edges,) NumPy int32 — static receiver indices.
        n_faces:      Python int — required for scatter operations.
        num_layers:   Python int — number of message-passing layers.

    Returns:
        (x_new, h_new, local_transform) where:
          x_new           — (n_faces, 2)         updated positions (equivariant)
          h_new           — (n_faces, hidden_dim) final node features (invariant)
          local_transform — (n_faces, 2, 2)       tile deformation matrix (identity at init)

    E(2) equivariance:
        apply_egnn(params, h, R@x+t, ...) = (R @ x_out + t, h_out, transform_out)
        for any rotation R ∈ SO(2) and translation t ∈ R².
        Face clamping (Dirichlet BCs) belongs to Stage 2 physics, not Stage 0 mapping.
    """

    # Embed static features into hidden space.
    h = jnp.tanh(h_raw @ params['emb_W'] + params['emb_b'])  # (n_faces, hidden_dim)

    for l in range(num_layers):
        # ── Relative vectors and squared distances (covariant / invariant) ───────
        # diff_{ij} = x_i - x_j  (from sender j toward receiver i)
        diff = x[receivers_np] - x[senders_np]        # (n_edges, 2)  covariant
        dist_sq = jnp.sum(diff ** 2, axis=-1, keepdims=True)  # (n_edges, 1)  invariant

        # ── φ_e : invariant edge message ─────────────────────────────────────────
        h_senders   = h[senders_np]    # (n_edges, hidden_dim)
        h_receivers = h[receivers_np]  # (n_edges, hidden_dim)
        edge_input = jnp.concatenate([h_senders, h_receivers, dist_sq], axis=-1)

        msg = _mlp2(
            edge_input,
            params[f'l{l}_phi_e_W1'], params[f'l{l}_phi_e_b1'],
            params[f'l{l}_phi_e_W2'], params[f'l{l}_phi_e_b2'],
        )  # (n_edges, hidden_dim)

        # ── φ_x : equivariant coordinate update ──────────────────────────────────
        phi_x_scalar = msg @ params[f'l{l}_phi_x_W'] + params[f'l{l}_phi_x_b']  # (n_edges, 1)
        coord_msg = diff * phi_x_scalar  # (n_edges, 2)  covariant

        # Incoming degree for mean aggregation (avoids div-by-zero).
        degree = jnp.zeros(n_faces, dtype=x.dtype).at[receivers_np].add(1.0)
        degree = jnp.maximum(degree, 1.0)[:, None]

        coord_agg = jnp.zeros((n_faces, 2), dtype=x.dtype).at[receivers_np].add(coord_msg)
        x = x + coord_agg / degree  # (n_faces, 2)

        # ── φ_h : invariant node feature update ──────────────────────────────────
        msg_agg = jnp.zeros((n_faces, msg.shape[-1]), dtype=msg.dtype).at[receivers_np].add(msg)
        node_input = jnp.concatenate([h, msg_agg], axis=-1)  # (n_faces, 2*hidden_dim)

        h = _mlp2(
            node_input,
            params[f'l{l}_phi_h_W1'], params[f'l{l}_phi_h_b1'],
            params[f'l{l}_phi_h_W2'], params[f'l{l}_phi_h_b2'],
        )  # (n_faces, hidden_dim)

    # Predict local 2×2 transformation matrix.
    # At init (weights ≈ 0): h @ W ≈ 0, so output ≈ bias b = [1, 0, 0, 1] (identity).
    transform_flat = h @ params['transform_W'] + params['transform_b']
    local_transform = transform_flat.reshape((n_faces, 2, 2))

    return x, h, local_transform
