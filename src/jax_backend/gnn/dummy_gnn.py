"""
Dummy GNN pour la validation du flux de gradients.

Un seul round de message-passing implémenté comme un dict JAX brut
(pas d'équinox / flax), directement compatible avec le pipeline optax
existant sans modification.

Architecture (une seule couche) :
    1. Encode nœuds  : h_i  @ W_enc → hidden_i              (tanh)
    2. Encode arêtes : e_ij @ W_msg → msg_ij                (tanh)
    3. Agrège        : Σ_{j→i} msg_ij → agg_i               (sum pooling)
    4. Met à jour    : [hidden_i + agg_i | x_i] @ W_out → Δx_i
    5. Retourne      : x_i + Δx_i                            (near-identity init)

Initialisation à petite variance (scale=0.01) : le réseau est quasiment
identité à l'epoch 0, ce qui garantit un signal de gradient non nul dès
le départ.

Le dict de poids est un PyTree JAX plat — optax, value_and_grad, et
tree_map fonctionnent dessus sans modification.
"""

import jax
import jax.numpy as jnp
import numpy as np


def init_dummy_gnn(
        key: jax.Array,
        node_feat_dim: int,
        hidden_dim: int = 16,
) -> dict:
    """Initialise les paramètres du dummy GNN.

    Args:
        key:           Clé PRNG JAX.
        node_feat_dim: Dimension des features statiques de nœud (e.g. 7).
        hidden_dim:    Largeur de la couche cachée.

    Returns:
        dict PyTree avec les clés :
            'W_enc'  — (node_feat_dim, hidden_dim)
            'W_msg'  — (1, hidden_dim)                  edge_feat_dim == 1
            'W_out'  — (hidden_dim + 2, 2)              +2 pour concat x_i
            'b_out'  — (2,)
    """
    k1, k2, k3 = jax.random.split(key, 3)
    scale = 0.01  # Init petite : near-identity à l'epoch 0

    return {
        'W_enc': jax.random.normal(k1, (node_feat_dim, hidden_dim), dtype=jnp.float64) * scale,
        'W_msg': jax.random.normal(k2, (1, hidden_dim), dtype=jnp.float64) * scale,
        'W_out': jax.random.normal(k3, (hidden_dim + 2, 2), dtype=jnp.float64) * scale,
        'b_out': jnp.zeros(2, dtype=jnp.float64),
    }


def apply_dummy_gnn(
        params: dict,
        h: jnp.ndarray,
        x: jnp.ndarray,
        edges: jnp.ndarray,
        senders_np: np.ndarray,
        receivers_np: np.ndarray,
        n_faces: int,
) -> jnp.ndarray:
    """Forward pass du dummy GNN → nouvelles positions des centroïdes.

    Args:
        params:       Dict PyTree des poids (issu de init_dummy_gnn).
        h:            (n_faces, node_feat_dim)  features statiques de nœud.
        x:            (n_faces, 2)             positions courantes (Tracer JAX).
        edges:        (n_edges, 1)             features d'arêtes (distances).
        senders_np:   (n_edges,) NumPy int32  — indices statiques pour indexation.
        receivers_np: (n_edges,) NumPy int32  — indices statiques pour segment_sum.
        n_faces:      int Python — nécessaire comme num_segments (statique dans JIT).

    Returns:
        (n_faces, 2) — nouvelles positions des centroïdes.

    Note sur le flux de gradients :
        x est concret (issu de initial_state en closure JIT).
        params est un Tracer. Le gradient ∂loss/∂params remonte via delta_x.
    """
    # 1. Encode les features statiques de chaque nœud
    node_enc = jnp.tanh(h @ params['W_enc'])          # (n_faces, hidden_dim)

    # 2. Encode les features d'arêtes → messages
    msgs = jnp.tanh(edges @ params['W_msg'])           # (n_edges, hidden_dim)

    # 3. Agrège les messages entrants par somme (sum pooling)
    #    on utilise scatter-add (.at[].add) plutôt que segment_sum :
    #    même sémantique mais code XLA différent, plus stable sur Metal.
    agg = jnp.zeros((n_faces, msgs.shape[-1]), dtype=msgs.dtype).at[receivers_np].add(msgs)  # (n_faces, hidden_dim)

    # 4. Combine état de nœud et position courante → prédiction de Δx
    node_state   = node_enc + agg                      # (n_faces, hidden_dim)
    node_with_x  = jnp.concatenate([node_state, x], axis=-1)  # (n_faces, hidden_dim+2)
    delta_x      = node_with_x @ params['W_out'] + params['b_out']  # (n_faces, 2)

    # 5. Mise à jour de position (résiduelle)
    return x + delta_x
