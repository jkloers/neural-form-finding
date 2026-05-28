"""
test_egnn_equivariance.py — Validation mathématique de l'équivariance E(2) de l'EGNN.

Test A : Equivariance sous rotation + translation.
    Pour toute rotation R ∈ SO(2) et translation t ∈ R² :
        x_out(R@x + t) ≈ R @ x_out(x) + t       (positions équivariantes)
        h_out(R@x + t) ≈ h_out(x)               (features invariantes)

Test B : Overfit sur un exemple unique (gradient flow + convergence).

Usage :
    cd /path/to/neural-form-finding
    python notebooks/test_egnn_equivariance.py
"""

import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
import sys
sys.path.append(os.path.abspath('src'))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from jax_backend.gnn.egnn import init_egnn, apply_egnn


# ── Fixtures ─────────────────────────────────────────────────────────────────

def make_test_graph(n_faces: int = 8, node_feat_dim: int = 7, seed: int = 42):
    """Génère un graphe aléatoire pour les tests."""
    rng = np.random.default_rng(seed)
    h = jnp.array(rng.standard_normal((n_faces, node_feat_dim)), dtype=jnp.float64)
    x = jnp.array(rng.standard_normal((n_faces, 2)), dtype=jnp.float64)

    # Graphe complet non-orienté (bidirectionnel)
    senders_list, receivers_list = [], []
    for i in range(n_faces):
        for j in range(n_faces):
            if i != j:
                senders_list.append(i)
                receivers_list.append(j)
    senders   = np.array(senders_list,   dtype=np.int32)
    receivers = np.array(receivers_list, dtype=np.int32)
    return h, x, senders, receivers, n_faces


def rotation_matrix_2d(theta: float) -> jnp.ndarray:
    c, s = jnp.cos(theta), jnp.sin(theta)
    return jnp.array([[c, -s], [s, c]], dtype=jnp.float64)


# ── Test A : Equivariance E(2) ────────────────────────────────────────────────

def test_equivariance(hidden_dim: int = 16, num_layers: int = 2, atol: float = 1e-8):
    print("=" * 60)
    print("Test A — Equivariance E(2)")
    print("=" * 60)

    node_feat_dim = 7
    h, x, senders, receivers, n_faces = make_test_graph(node_feat_dim=node_feat_dim)

    key = jax.random.PRNGKey(0)
    params = init_egnn(key, node_feat_dim, hidden_dim=hidden_dim, num_layers=num_layers)

    # Référence : sortie sur x original
    x_out_ref, h_out_ref, _, _ = apply_egnn(params, h, x, senders, receivers, n_faces)

    all_passed = True
    for trial, (theta, t) in enumerate([
        (jnp.pi / 4,  jnp.array([1.0, -2.0])),
        (jnp.pi,      jnp.array([0.0,  0.0])),
        (jnp.pi / 6,  jnp.array([-3.5, 2.1])),
        (0.0,         jnp.array([5.0,  5.0])),
    ]):
        R = rotation_matrix_2d(theta)
        t_vec = t[None, :]  # (1, 2) pour broadcast

        # Applique la transformation rigide aux positions d'entrée
        x_transformed = (R @ x.T).T + t_vec

        # Forward pass sur la configuration transformée
        x_out_transf, h_out_transf, _, _ = apply_egnn(
            params, h, x_transformed, senders, receivers, n_faces)

        # Ce que l'on devrait obtenir si le réseau est équivariant :
        x_out_expected = (R @ x_out_ref.T).T + t_vec

        eq_x = jnp.allclose(x_out_transf, x_out_expected, atol=atol)
        eq_h = jnp.allclose(h_out_transf, h_out_ref, atol=atol)

        max_err_x = float(jnp.max(jnp.abs(x_out_transf - x_out_expected)))
        max_err_h = float(jnp.max(jnp.abs(h_out_transf - h_out_ref)))

        status_x = "PASS" if eq_x else "FAIL"
        status_h = "PASS" if eq_h else "FAIL"
        t0, t1 = float(t[0]), float(t[1])
        print(f"  Trial {trial+1} (θ={float(theta):.2f} rad, t=[{t0:.1f}, {t1:.1f}])")
        print(f"    x equivariance : [{status_x}]  max_err = {max_err_x:.2e}")
        print(f"    h invariance   : [{status_h}]  max_err = {max_err_h:.2e}")
        all_passed = all_passed and bool(eq_x) and bool(eq_h)

    print()
    if all_passed:
        print("  => Test A PASSED — EGNN est E(2)-équivariant.")
    else:
        print("  => Test A FAILED — vérifier l'implémentation de apply_egnn.")
    return all_passed


# ── Test B : Gradient flow + overfit ─────────────────────────────────────────

def test_gradient_flow(hidden_dim: int = 16, num_layers: int = 2, num_steps: int = 10):
    print()
    print("=" * 60)
    print("Test B — Gradient flow & overfit (1 exemple)")
    print("=" * 60)

    import optax

    node_feat_dim = 7
    h, x, senders, receivers, n_faces = make_test_graph(node_feat_dim=node_feat_dim)

    key = jax.random.PRNGKey(1)
    params = init_egnn(key, node_feat_dim, hidden_dim=hidden_dim, num_layers=num_layers)

    # Cible simple : les positions d'origine décalées de +1 sur x
    x_target = x + 1.0

    def loss_fn(p):
        x_out, _, _, _ = apply_egnn(p, h, x, senders, receivers, n_faces)
        return jnp.mean((x_out - x_target) ** 2)

    optimizer = optax.adam(learning_rate=1e-2)
    opt_state = optimizer.init(params)

    losses = []
    for step in range(num_steps):
        loss_val, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        losses.append(float(loss_val))
        if step % 2 == 0 or step == num_steps - 1:
            # Vérifie que tous les gradients sont non-nuls et non-NaN
            leaves = jax.tree_util.tree_leaves(grads)
            has_nan  = any(bool(jnp.any(jnp.isnan(g))) for g in leaves)
            has_zero = all(float(jnp.max(jnp.abs(g))) == 0.0 for g in leaves)
            print(f"  Step {step:03d} | Loss: {loss_val:.4e} | NaN: {has_nan} | AllZero: {has_zero}")

    passed = losses[-1] < losses[0]
    print()
    if passed:
        print(f"  => Test B PASSED — loss décroit de {losses[0]:.4e} à {losses[-1]:.4e}.")
    else:
        print(f"  => Test B FAILED — loss ne décroit pas.")
    return passed


# ── Point d'entrée ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print()
    ok_a = test_equivariance(hidden_dim=32, num_layers=3)
    ok_b = test_gradient_flow(hidden_dim=32, num_layers=3, num_steps=20)

    print()
    print("=" * 60)
    overall = "ALL TESTS PASSED" if (ok_a and ok_b) else "SOME TESTS FAILED"
    print(f" {overall}")
    print("=" * 60)
    print()
    sys.exit(0 if (ok_a and ok_b) else 1)
