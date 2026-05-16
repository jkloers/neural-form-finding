"""
End-to-End Training Loop for Neural Form-Finding.

This module provides the training logic to optimize mapping parameters 
(or future GNN weights) using the differentiable physics pipeline.
"""

import jax
import jax.numpy as jnp
import optax

from jax_backend.training.loss import compute_end_to_end_loss


def _format_grad_norms(norms, prefix: str = '') -> list[str]:
    """Formate récursivement un PyTree de normes en paires 'clé=valeur'.

    Fonctionne pour les dicts plats (map_params classiques) et les dicts
    imbriqués (poids de réseaux de neurones multi-couches).
    """
    if isinstance(norms, dict):
        parts = []
        for k, v in norms.items():
            parts.extend(_format_grad_norms(v, f"{prefix}{k}"))
        return parts
    return [f"{prefix}={float(norms):.2e}"]

from problem.config import TargetConfig, PhysicsConfig, TrainingConfig, ValidityConfig

def create_train_step(initial_state, target_cfg: TargetConfig, validity_cfg: ValidityConfig, physics_cfg: PhysicsConfig, training_cfg: TrainingConfig, map_type: str = 'conformal_polynomial', use_shirley_chiu: bool = True, strict_boundary_fit: bool = True, learn_global_scale: bool = False, use_jit: bool = True):
    """Creates a compiled training step for optimizing map_params.

    Args:
        initial_state: The flat tessellation CentroidalState.
        target_cfg: Structured TargetConfig.
        physics_cfg: Structured PhysicsConfig.
        training_cfg: Structured TrainingConfig.
        learn_global_scale: See compute_end_to_end_loss.

    Returns:
        optimizer, train_step_fn
    """

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=training_cfg.learning_rate),
    )

    # Pour les types GNN, on précompute les features statiques du graphe ICI,
    # avant la frontière @jax.jit. Appeler build_static_graph_features() à
    # l'intérieur du trace JIT déclenche un bug du backend Metal sur Mac.
    # En capturant le résultat dans la closure de loss_fn, il devient une
    # constante XLA (jamais retracée) — solution correcte et efficace.
    if map_type.startswith('gnn_'):
        from jax_backend.gnn.graph_builder import build_static_graph_features
        _static_features = build_static_graph_features(initial_state)
    else:
        _static_features = None

    # The loss function closed over the constants
    def loss_fn(map_params):
        return compute_end_to_end_loss(
            map_params, initial_state, target_cfg, validity_cfg, physics_cfg, training_cfg,
            map_type=map_type, use_shirley_chiu=use_shirley_chiu,
            strict_boundary_fit=strict_boundary_fit,
            learn_global_scale=learn_global_scale,
            static_features=_static_features,
        )
        
    def _step_body(map_params, opt_state):
        (loss_val, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(map_params)

        leaves = jax.tree_util.tree_leaves(grads)
        global_grad_norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in leaves))
        grad_norms = jax.tree_util.tree_map(lambda g: jnp.sqrt(jnp.sum(g ** 2)), grads)
        aux['grad_norm'] = global_grad_norm
        aux['grad_norms'] = grad_norms

        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_map_params = optax.apply_updates(map_params, updates)

        return new_map_params, new_opt_state, loss_val, aux

    # GNN types désactivent JIT par défaut : le backend XLA/Metal sur Mac
    # crashe lors de la compilation de programmes impliquant scatter-add
    # + LBFGS imbriqués. Le code est correctement différentiable (validé
    # par test_gnn_nojit.py) — JIT sera réactivé quand le bug Metal sera corrigé.
    train_step_fn = jax.jit(_step_body) if use_jit else _step_body

    return optimizer, train_step_fn

def train_pipeline(initial_map_params, initial_state, target_cfg: TargetConfig,
                   validity_cfg: ValidityConfig,
                   physics_cfg: PhysicsConfig, training_cfg: TrainingConfig,
                   map_type: str = 'conformal_polynomial',
                   use_shirley_chiu: bool = True,
                   strict_boundary_fit: bool = True,
                   learn_global_scale: bool = False,
                   use_jit: bool = True):
    """Run the training loop to find optimal mapping parameters."""

    optimizer, train_step_fn = create_train_step(
        initial_state, target_cfg, validity_cfg, physics_cfg, training_cfg,
        map_type, use_shirley_chiu, strict_boundary_fit,
        learn_global_scale=learn_global_scale,
        use_jit=use_jit,
    )
    
    opt_state = optimizer.init(initial_map_params)
    current_params = initial_map_params
    
    history_loss = []
    
    for epoch in range(training_cfg.num_epochs):
        current_params, opt_state, loss, aux = train_step_fn(current_params, opt_state)
        history_loss.append(aux)
        
        if epoch % 5 == 0 or epoch == training_cfg.num_epochs - 1:
            grad_norm = float(aux['grad_norm'])
            flag = "  [VANISHING]" if grad_norm < 1e-6 else ("  [EXPLODING]" if grad_norm > 1e3 else "")
            per_param = " ".join(_format_grad_norms(aux['grad_norms']))
            area_dev = f" | ΔArea: {aux['global_material_area']:.2e}" if not learn_global_scale else ""
            print(
                f"Epoch {epoch:03d} | Loss: {aux['total']:.4e} | "
                f"Chamfer: {aux['chamfer_total']:.4e} | Energy: {aux['energy']:.4e}{area_dev} | "
                f"‖grad‖: {grad_norm:.2e}{flag}"
            )
            print(f"         grads: {per_param}")
            
    return current_params, history_loss
