"""
EGNN — E(2)-Equivariant Graph Neural Network pour le neural form-finding.

Architecture (L couches) :
    Embedding : h_i = tanh(x_raw_i @ W_emb + b_emb)

    Pour chaque couche l :
        1. Invariant d'arête  : d²_ij = ||x_i - x_j||²
        2. Message            : m_ij  = φ_e([h_i | h_j | d²_ij])     (2-layer MLP, tanh)
        3. Mise à jour coords : x_i  += (1/|N(i)|) Σ_j (x_i - x_j) * φ_x(m_ij)  (équivariant)
        4. Agrégation msgs    : agg_i = Σ_j m_ij
        5. Mise à jour nœuds  : h_i   = φ_h([h_i | agg_i])            (2-layer MLP, tanh)

Équivariance E(2) :
    - d²_ij est invariant par rotation/translation → φ_e produit des messages invariants.
    - (x_i - x_j) est covariant (vecteur relatif) ; scalé par φ_x(m_ij) scalaire → δx équivariant.
    - h_i ne dépend que de grandeurs invariantes → h_out invariant.

Structure des paramètres (dict plat, PyTree JAX) :
    'emb_W'  : (node_feat_dim, hidden_dim)
    'emb_b'  : (hidden_dim,)
    Pour chaque couche l (0-indexed) :
        'l{l}_phi_e_W1' : (2*hidden_dim + 1, hidden_dim)   +1 pour d²_ij
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
    """Initialise les paramètres d'un EGNN L-couches.

    Args:
        key:           Clé PRNG JAX.
        node_feat_dim: Dimension des features statiques (ex. 7).
        hidden_dim:    Largeur de toutes les couches cachées.
        num_layers:    Nombre de couches de message-passing EGNN.

    Returns:
        Dict PyTree plat compatible optax / value_and_grad.
    """
    scale = 0.01  # near-identity init — gradient non nul dès l'epoch 0
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

        # φ_x : message → scalaire de pondération coord  (dim: hidden_dim → 1)
        params[f'l{l}_phi_x_W'] = jax.random.normal(next(ki), (hidden_dim, 1), dtype=jnp.float64) * scale_phi_x
        params[f'l{l}_phi_x_b'] = jnp.zeros(1, dtype=jnp.float64)

        # φ_h : [h_i | agg_i] → h_i suivant  (dim: 2*hidden_dim → hidden_dim)
        params[f'l{l}_phi_h_W1'] = jax.random.normal(next(ki), (2 * hidden_dim, hidden_dim), dtype=jnp.float64) * scale
        params[f'l{l}_phi_h_b1'] = jnp.zeros(hidden_dim, dtype=jnp.float64)
        params[f'l{l}_phi_h_W2'] = jax.random.normal(next(ki), (hidden_dim, hidden_dim), dtype=jnp.float64) * scale
        params[f'l{l}_phi_h_b2'] = jnp.zeros(hidden_dim, dtype=jnp.float64)

    # Tête de prédiction de la matrice de transformation 2x2 locale (invariante)
    # transform_W → (hidden_dim, 4)
    # transform_b → initialisée à l'identité [1, 0, 0, 1]
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
    """Forward pass EGNN → nouvelles positions et features de nœuds.

    Args:
        params:       Dict PyTree issu de init_egnn.
        h_raw:        (n_faces, node_feat_dim) features statiques brutes.
        x:            (n_faces, 2) positions courantes (Tracer JAX).
        senders_np:   (n_edges,) NumPy int32 — indices statiques.
        receivers_np: (n_edges,) NumPy int32 — indices statiques.
        n_faces:      int Python — nécessaire pour les opérations scatter.
        num_layers:   int Python — nombre de couches de message-passing.

    Returns:
        (x_new, h_new, local_transform) where:
          x_new           — (n_faces, 2)         nouvelles positions (équivariant)
          h_new           — (n_faces, hidden_dim) features finales (invariant)
          local_transform — (n_faces, 2, 2)       matrice de transformation (identité à l'init)

    Equivariance E(2) :
        apply_egnn(params, h, R@x+t, ...) = (R @ x_out + t, h_out, transform_out)
        transform est invariant (calculé depuis h invariant).
        pour toute rotation R ∈ SO(2) et translation t ∈ R².
        Le clamping des faces (BCs Dirichlet) appartient au solveur physique (Stage 2),
        pas au mapping géométrique (Stage 0) — toutes les faces bougent librement ici.
    """

    # Embedding des features statiques → espace caché
    h = jnp.tanh(h_raw @ params['emb_W'] + params['emb_b'])  # (n_faces, hidden_dim)

    for l in range(num_layers):
        # ── Vecteurs relatifs et distance² (invariants/covariants) ─────────────
        # Convention papier : diff_{ij} = x_i - x_j  (de sender j vers receiver i)
        diff = x[receivers_np] - x[senders_np]        # (n_edges, 2)  covariant
        dist_sq = jnp.sum(diff ** 2, axis=-1, keepdims=True)  # (n_edges, 1)  invariant

        # ── φ_e : message invariant ─────────────────────────────────────────────
        h_senders   = h[senders_np]    # (n_edges, hidden_dim)
        h_receivers = h[receivers_np]  # (n_edges, hidden_dim)
        edge_input = jnp.concatenate([h_senders, h_receivers, dist_sq], axis=-1)

        msg = _mlp2(
            edge_input,
            params[f'l{l}_phi_e_W1'], params[f'l{l}_phi_e_b1'],
            params[f'l{l}_phi_e_W2'], params[f'l{l}_phi_e_b2'],
        )  # (n_edges, hidden_dim)

        # ── φ_x : mise à jour équivariante des coordonnées ─────────────────────
        phi_x_scalar = msg @ params[f'l{l}_phi_x_W'] + params[f'l{l}_phi_x_b']  # (n_edges, 1)
        coord_msg = diff * phi_x_scalar  # (n_edges, 2)  covariant

        # Compter le degré entrant pour normalisation
        degree = jnp.zeros(n_faces, dtype=x.dtype).at[receivers_np].add(1.0)
        degree = jnp.maximum(degree, 1.0)[:, None]  # évite div/0

        coord_agg = jnp.zeros((n_faces, 2), dtype=x.dtype).at[receivers_np].add(coord_msg)
        x = x + coord_agg / degree  # (n_faces, 2)

        # ── φ_h : mise à jour invariante des features ──────────────────────────
        msg_agg = jnp.zeros((n_faces, msg.shape[-1]), dtype=msg.dtype).at[receivers_np].add(msg)
        node_input = jnp.concatenate([h, msg_agg], axis=-1)  # (n_faces, 2*hidden_dim)

        h = _mlp2(
            node_input,
            params[f'l{l}_phi_h_W1'], params[f'l{l}_phi_h_b1'],
            params[f'l{l}_phi_h_W2'], params[f'l{l}_phi_h_b2'],
        )  # (n_faces, hidden_dim)

    # Prédiction de la matrice de transformation 2x2
    # À l'init (poids ~0), h @ W ≈ 0, donc on renvoie le biais b = [1, 0, 0, 1] (Identité)
    transform_flat = h @ params['transform_W'] + params['transform_b']
    local_transform = transform_flat.reshape((n_faces, 2, 2))

    return x, h, local_transform
