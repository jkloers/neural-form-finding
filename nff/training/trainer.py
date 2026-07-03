"""
End-to-End Training Loop for Neural Form-Finding.

This module provides the training logic to optimize mapping parameters
(or future GNN weights) using the differentiable physics pipeline.
"""

import math
import time
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax

from jaxtyping import Array, Float
from nff.config.experiment import TargetConfig, PhysicsConfig, TrainingConfig, ValidityConfig
from nff.config.targets import get_target_points
from nff.training.loss import compute_end_to_end_loss


# ── Training state ────────────────────────────────────────────────────────────

class TrainState(NamedTuple):
    """Immutable PyTree bundling all mutable training state.

    Passing a single TrainState in/out of jax.jit makes the functional
    contract explicit: train_step is a pure function TrainState → TrainState.
    """
    params:    Any         # map_params PyTree (GNN weights or analytical coefficients)
    opt_state: Any         # optax optimizer state
    rng:       jax.Array   # PRNG key — split each step; ready for future stochastic ops


# ── Logging helpers ───────────────────────────────────────────────────────────

def _format_grad_norms(norms, prefix: str = '') -> list[str]:
    """Recursively format a PyTree of norms as 'key=value' pairs.

    Works for flat dicts (analytical map_params) and nested dicts
    (multi-layer GNN weights).
    """
    if isinstance(norms, dict):
        parts = []
        for k, v in norms.items():
            parts.extend(_format_grad_norms(v, f"{prefix}{k}"))
        return parts
    return [f"{prefix}={float(norms):.2e}"]


# ── Step factory ──────────────────────────────────────────────────────────────

def create_train_step(
        initial_state,
        target_cfg: TargetConfig,
        validity_cfg: ValidityConfig,
        physics_cfg: PhysicsConfig,
        training_cfg: TrainingConfig,
        map_type: str = 'conformal_polynomial',
        use_shirley_chiu: bool = True,
        strict_boundary_fit: bool = True,
        learn_global_scale: bool = False,
        use_jit: bool = True,
        load_specs=None,
        static_features=None,
        target_cloud=None,
        bond_energy_fn=None,
):
    """Creates a compiled training step for optimizing map_params.

    Returns:
        (optimizer, train_step_fn) where train_step_fn : TrainState → (TrainState, float, dict)
    """

    lr_schedule = getattr(training_cfg, 'lr_schedule', 'constant')
    if lr_schedule == 'cosine':
        lr = optax.cosine_decay_schedule(
            init_value=training_cfg.learning_rate,
            decay_steps=training_cfg.num_epochs,
            alpha=0.1,
        )
    elif lr_schedule == 'warmup_cosine':
        # Linear warm-up for first 5% of epochs (min 20, max 100), then cosine decay.
        # Prevents the large initial gradient steps from pushing the tessellation into
        # contact-singularity or near-mechanism configurations on the very first epochs.
        warmup_steps = max(20, min(100, training_cfg.num_epochs // 20))
        decay_steps  = max(1, training_cfg.num_epochs - warmup_steps)
        lr = optax.join_schedules(
            schedules=[
                optax.linear_schedule(
                    init_value=training_cfg.learning_rate * 0.01,
                    end_value=training_cfg.learning_rate,
                    transition_steps=warmup_steps,
                ),
                optax.cosine_decay_schedule(
                    init_value=training_cfg.learning_rate,
                    decay_steps=decay_steps,
                    alpha=0.1,
                ),
            ],
            boundaries=[warmup_steps],
        )
    else:
        lr = training_cfg.learning_rate

    optimizer = optax.chain(
        optax.clip_by_global_norm(training_cfg.grad_clip),
        optax.adam(learning_rate=lr),
    )

    # Static features precomputed BEFORE the JIT boundary.
    # For GNN types: build_static_features() calls non-JAX numpy ops that would
    # crash the Metal backend if traced. Capturing the result here makes it an
    # XLA compile-time constant — correct and efficient.
    # If caller provides static_features (e.g. with inner_depth, num_layers already
    # set), use it directly — this preserves GNN-config metadata like inner_depth
    # that is not recoverable from the initial_state alone.
    if static_features is not None:
        _static_features = static_features
    elif map_type.startswith('gnn_'):
        from nff.models.graph_builder import build_static_features
        _static_features = build_static_features(initial_state, map_type)
    else:
        _static_features = None

    # Target cloud precomputed BEFORE the JIT boundary.
    # Without this, get_target_points() would run at every JAX trace and its
    # result would be rebaked into the XLA program on each recompilation.
    _target_params = {
        'type': target_cfg.type,
        'center': target_cfg.center,
        'radius': target_cfg.radius,
    }
    # Allow a caller-supplied target cloud (e.g. a rectangle the closed-state
    # driver fits to the deployment) to override the shape from target_cfg.
    if target_cloud is not None:
        _target_cloud = jnp.asarray(target_cloud, dtype=jnp.float64)
    else:
        _target_cloud = jnp.asarray(
            get_target_points(_target_params, n_points=500), dtype=jnp.float64)

    # Pure loss function closed over all compile-time constants.
    def loss_fn(params: Any) -> tuple[Float[Array, ""], dict]:
        return compute_end_to_end_loss(
            params, initial_state, target_cfg, validity_cfg, physics_cfg, training_cfg,
            map_type=map_type,
            use_shirley_chiu=use_shirley_chiu,
            strict_boundary_fit=strict_boundary_fit,
            learn_global_scale=learn_global_scale,
            static_features=_static_features,
            load_specs=load_specs,
            target_cloud=_target_cloud,
            bond_energy_fn=bond_energy_fn,
        )

    def _step_body(state: TrainState) -> tuple[TrainState, Float[Array, ""], dict]:
        # Split key first so every step consumes a fresh subkey — ready for
        # dropout or noise if added later; overhead is negligible without them.
        rng, subkey = jax.random.split(state.rng)

        (loss_val, base_aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

        leaves = jax.tree_util.tree_leaves(grads)
        global_grad_norm = jnp.sqrt(
            jnp.sum(jnp.stack([jnp.sum(g ** 2) for g in leaves])))
        grad_norms = jax.tree_util.tree_map(
            lambda g: jnp.sqrt(jnp.sum(g ** 2)), grads)

        # NaN/Inf-safe update: replace any non-finite gradient entry with zero
        # before clipping and applying. This makes individual contact-singularity
        # spikes non-destructive — the optimizer step is skipped for the affected
        # parameters rather than corrupting them with inf/nan values.
        safe_grads = jax.tree_util.tree_map(
            lambda g: jnp.where(jnp.isfinite(g), g, jnp.zeros_like(g)),
            grads,
        )

        updates, new_opt_state = optimizer.update(safe_grads, state.opt_state)
        new_params = optax.apply_updates(state.params, updates)

        new_state = TrainState(params=new_params, opt_state=new_opt_state, rng=rng)
        aux = {**base_aux, 'grad_norm': global_grad_norm, 'grad_norms': grad_norms}

        return new_state, loss_val, aux

    # GNN types désactivent JIT par défaut : le backend XLA/Metal sur Mac
    # crashe lors de la compilation de programmes impliquant scatter-add
    # + LBFGS imbriqués. Le code est correctement différentiable (validé
    # par test_gnn_nojit.py) — JIT sera réactivé quand le bug Metal sera corrigé.
    train_step_fn = jax.jit(_step_body) if use_jit else _step_body

    return optimizer, train_step_fn


# ── Training loop ─────────────────────────────────────────────────────────────

def train_pipeline(
        initial_map_params,
        initial_state,
        target_cfg: TargetConfig,
        validity_cfg: ValidityConfig,
        physics_cfg: PhysicsConfig,
        training_cfg: TrainingConfig,
        map_type: str = 'conformal_polynomial',
        use_shirley_chiu: bool = True,
        strict_boundary_fit: bool = True,
        learn_global_scale: bool = False,
        use_jit: bool = True,
        load_specs=None,
        rng_seed: int = 0,
        static_features=None,
) -> tuple[Any, list[dict]]:
    """Run the training loop to find optimal mapping parameters.

    Returns:
        (optimized_params, history_loss) — history_loss is a list of metric dicts,
        one per epoch (keys: 'total', 'chamfer_total', 'energy', 'grad_norm', …).
    """
    optimizer, train_step_fn = create_train_step(
        initial_state, target_cfg, validity_cfg, physics_cfg, training_cfg,
        map_type, use_shirley_chiu, strict_boundary_fit,
        learn_global_scale=learn_global_scale,
        use_jit=use_jit,
        load_specs=load_specs,
        static_features=static_features,
    )

    state = TrainState(
        params=initial_map_params,
        opt_state=optimizer.init(initial_map_params),
        rng=jax.random.PRNGKey(rng_seed),
    )

    history_loss: list[dict] = []
    t_train_start = time.time()
    t_epoch_start = time.time()

    # Best-checkpoint: track the params that achieved the lowest valid chamfer
    # loss during the entire training run. Gradient explosions can corrupt the
    # final state — returning the best-seen params prevents that.
    best_chamfer  = float('inf')
    best_params   = initial_map_params

    for epoch in range(training_cfg.num_epochs):
        state, loss, aux = train_step_fn(state)
        history_loss.append(aux)

        chamfer_val = float(aux.get('chamfer_total', float('nan')))
        if math.isfinite(chamfer_val) and chamfer_val < best_chamfer:
            best_chamfer = chamfer_val
            best_params  = state.params

        if epoch % 5 == 0 or epoch == training_cfg.num_epochs - 1:
            t_now = time.time()
            elapsed_total = t_now - t_train_start
            elapsed_block = t_now - t_epoch_start
            epochs_in_block = 5 if epoch > 0 else 1
            s_per_epoch = elapsed_block / epochs_in_block
            remaining_epochs = training_cfg.num_epochs - epoch - 1
            eta_s = remaining_epochs * s_per_epoch
            eta_str = f"{eta_s/60:.1f}min" if eta_s >= 60 else f"{eta_s:.0f}s"
            t_epoch_start = t_now

            grad_norm = float(aux['grad_norm'])
            flag = ("  [VANISHING]" if grad_norm < 1e-6
                    else ("  [EXPLODING]" if grad_norm > 1e3 else ""))
            per_param = " ".join(_format_grad_norms(aux['grad_norms']))
            area_dev = (f" | ΔArea: {aux['global_material_area']:.2e}"
                        if not learn_global_scale else "")
            hinge_str = (f" | HingeGap: {aux.get('hinge_gap', 0.0):.4e}"
                         if aux.get('hinge_gap', 0.0) > 0 else "")
            open_val    = aux.get('openness', 0.0)
            deform_val  = aux.get('deformation', 0.0)
            vc_val      = aux.get('void_closure', 0.0)
            cd_val      = aux.get('closure_delta', 0.0)
            closing_str = ""
            if open_val != 0.0 or deform_val != 0.0 or vc_val != 0.0 or cd_val != 0.0:
                closing_str = (f" | Open: {float(open_val):.3e}"
                               f" | Deform: {float(deform_val):.3e}"
                               f" | VoidS2: {float(vc_val):.3e}"
                               f" | CloseDelta: {float(cd_val):.3e}")
            print(
                f"Epoch {epoch:03d} | Loss: {aux['total']:.4e} | "
                f"Chamfer: {aux['chamfer_total']:.4e} | Energy: {aux['energy']:.4e}"
                f"{area_dev}{hinge_str}{closing_str} | "
                f"‖grad‖: {grad_norm:.2e}{flag} | "
                f"{s_per_epoch:.1f}s/ep | elapsed {elapsed_total/60:.1f}min | ETA {eta_str}"
            )
            print(f"         grads: {per_param}")

    total_time = time.time() - t_train_start
    final_chamfer = float(history_loss[-1].get('chamfer_total', float('nan'))) if history_loss else float('nan')
    print(f"\nTraining complete: {training_cfg.num_epochs} epochs in {total_time/60:.1f}min "
          f"({total_time/training_cfg.num_epochs:.1f}s/epoch avg)")
    if best_chamfer < final_chamfer - 1e-6:
        print(f"  ✓ Returning best checkpoint: chamfer={best_chamfer:.4f} "
              f"(final was {final_chamfer:.4f} — checkpoint saved {final_chamfer/best_chamfer:.1f}× improvement)")
    else:
        print(f"  ✓ Final state is best: chamfer={final_chamfer:.4f}")

    return best_params, history_loss
