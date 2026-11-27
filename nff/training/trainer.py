"""
End-to-End Training Loop for Neural Form-Finding.

This module provides the training logic to optimize mapping parameters
(or future GNN weights) using the differentiable physics pipeline.
"""

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
        hinge_probe_fn=None,
        hinge_geometry_fn=None,
        extra_loss_fn=None,
):
    """Creates a compiled training step for optimizing map_params.

    Args:
        extra_loss_fn: optional ``params -> scalar`` added to the total loss inside
            ``value_and_grad``, so it contributes gradients. Exists for design-space penalties
            that are not part of the shared loss registry — a chart-specific term belongs to the
            experiment that needs it, not to ``LossWeights``, which every run shares. Its value is
            reported as ``aux['extra_loss']``.

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

    # Static features are precomputed by the caller BEFORE the JIT boundary (closed_les passes
    # its chart topology here), so they are closed over as an XLA compile-time constant.
    _static_features = static_features

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
        total, aux = compute_end_to_end_loss(
            params, initial_state, target_cfg, validity_cfg, physics_cfg, training_cfg,
            map_type=map_type,
            use_shirley_chiu=use_shirley_chiu,
            strict_boundary_fit=strict_boundary_fit,
            learn_global_scale=learn_global_scale,
            static_features=_static_features,
            load_specs=load_specs,
            target_cloud=_target_cloud,
            bond_energy_fn=bond_energy_fn,
            hinge_probe_fn=hinge_probe_fn,
            hinge_geometry_fn=hinge_geometry_fn,
        )
        if extra_loss_fn is not None:
            extra = extra_loss_fn(params)
            total = total + extra
            aux = {**aux, 'extra_loss': extra}
        return total, aux

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

    train_step_fn = jax.jit(_step_body) if use_jit else _step_body

    return optimizer, train_step_fn
