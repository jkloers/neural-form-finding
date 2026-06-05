"""
Full training run on c001 (mpnn_best_2x2, 1000 epochs).

Produces:
  • All standard pipeline outputs (stage 0/1/2 plots, energy plot,
    closing animation, loss history) — same as a regular train.py run.
  • Training evolution animation: kirigami at physical equilibrium
    synchronized with the stacked loss chart, one frame every 20 epochs.

Usage:
    python scripts/test_training_animation.py
"""

import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import math
import datetime
import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

from nff.config.experiment import (
    load_arch_config, load_problem_suite, merge_arch_problem, _parse_full_raw,
)
from nff.scripts.train import _build_initial_state, _init_map_params
from nff.stages.pipeline import forward_pipeline
from nff.training.trainer import create_train_step, TrainState
from nff.utils.pipeline_viz import visualize_pipeline_results, plot_loss_history
from nff.utils.training_animation import animate_training_evolution


ARCH_PATH         = "data/configs/architectures/mpnn_best_2x2.yaml"
SUITE_PATH        = "data/configs/problems/suite_compressive_t187.yaml"
PROBLEM_ID        = "c001"
NUM_EPOCHS        = 1000
SNAPSHOT_INTERVAL = 20      # every 20 epochs → 50 frames + last frame
TARGET_STAGE      = 2       # 0=mapping | 1=validity | 2=equilibrium
FPS               = 12


def main():
    # ── Config ─────────────────────────────────────────────────────────────
    arch_raw     = load_arch_config(ARCH_PATH)
    all_problems = load_problem_suite(SUITE_PATH)
    problem      = next(p for p in all_problems if p['id'] == PROBLEM_ID)
    merged       = merge_arch_problem(arch_raw, problem)
    config       = _parse_full_raw(merged, os.path.dirname(ARCH_PATH))

    print(f"Arch   : {ARCH_PATH}")
    print(f"Problem: {PROBLEM_ID}")
    print(f"Epochs : {NUM_EPOCHS} | snapshot every {SNAPSHOT_INTERVAL} | stage {TARGET_STAGE}")

    # ── Initial state ───────────────────────────────────────────────────────
    initial_state, tessellation = _build_initial_state(config)
    map_params, static_features = _init_map_params(config, initial_state)
    load_specs = config.topology.get('loads', []) or []

    # ── Training step factory ───────────────────────────────────────────────
    _use_jit = jax.default_backend() == 'cpu' or not config.mapping.type.startswith('gnn_')

    optimizer, train_step_fn = create_train_step(
        initial_state,
        config.target, config.validity, config.physics, config.training,
        map_type=config.mapping.type,
        use_shirley_chiu=config.mapping.use_shirley_chiu,
        strict_boundary_fit=config.mapping.strict_boundary_fit,
        learn_global_scale=config.mapping.learn_global_scale,
        use_jit=_use_jit,
        load_specs=load_specs,
        static_features=static_features,
    )

    state = TrainState(
        params=map_params,
        opt_state=optimizer.init(map_params),
        rng=jax.random.PRNGKey(0),
    )

    # ── Training loop with snapshot collection ──────────────────────────────
    params_history  = []
    snapshot_epochs = []
    history_loss    = []
    best_chamfer    = float('inf')
    best_params     = map_params

    print("\nTraining ...\n")
    for epoch in range(NUM_EPOCHS):
        state, _loss, aux = train_step_fn(state)

        chamfer_val = float(aux.get('chamfer_total', float('nan')))
        if math.isfinite(chamfer_val) and chamfer_val < best_chamfer:
            best_chamfer = chamfer_val
            best_params  = state.params

        history_loss.append({k: float(aux.get(k, 0.0)) for k in (
            'total', 'chamfer_total', 'energy',
            'comp_geom_chamfer', 'comp_geom_material_area',
            'comp_phys_stretching', 'comp_phys_shearing',
            'comp_phys_bending', 'comp_phys_contact',
            'hinge_gap', 'void_closure', 'closure_delta',
            'openness', 'deformation',
        )})

        if epoch % SNAPSHOT_INTERVAL == 0 or epoch == NUM_EPOCHS - 1:
            params_history.append(jax.tree_util.tree_map(np.array, state.params))
            snapshot_epochs.append(epoch)

        if epoch % 200 == 0 or epoch == NUM_EPOCHS - 1:
            print(f"  Epoch {epoch:04d} | chamfer={chamfer_val:.4f} | "
                  f"loss={float(aux['total']):.4e}")

    print(f"\nCollected {len(params_history)} snapshots | best chamfer = {best_chamfer:.4f}")

    # ── Output directory ────────────────────────────────────────────────────
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir   = os.path.join("data", "outputs", f"run_{timestamp}_{PROBLEM_ID}")
    os.makedirs(out_dir, exist_ok=True)

    target_params = {
        'type':   config.target.type,
        'center': config.target.center,
        'radius': config.target.radius,
    }

    # ── Standard pipeline outputs ───────────────────────────────────────────
    print("\nGenerating standard pipeline outputs ...")
    plot_loss_history(history_loss, config, run_dir=out_dir)

    result = forward_pipeline(
        initial_state,
        config.target, config.validity, config.physics,
        map_type=config.mapping.type,
        map_params=best_params,
        use_shirley_chiu=config.mapping.use_shirley_chiu,
        strict_boundary_fit=config.mapping.strict_boundary_fit,
        static_features=static_features,
        load_specs=load_specs,
    )
    visualize_pipeline_results(
        result, tessellation, config, target_params,
        f"{PROBLEM_ID}_trained", run_dir=out_dir,
        load_specs=load_specs,
    )

    # ── Training evolution animation ────────────────────────────────────────
    stage_suffix = {0: 'stage0', 1: 'stage1', 2: 'stage2'}
    ani_path = os.path.join(out_dir, f"training_evolution_{stage_suffix[TARGET_STAGE]}.gif")

    animate_training_evolution(
        params_history=params_history,
        initial_state=initial_state,
        tessellation=tessellation,
        target_cfg=config.target,
        validity_cfg=config.validity,
        physics_cfg=config.physics,
        target_params=target_params,
        map_type=config.mapping.type,
        target_stage=TARGET_STAGE,
        filepath=ani_path,
        fps=FPS,
        use_shirley_chiu=config.mapping.use_shirley_chiu,
        strict_boundary_fit=config.mapping.strict_boundary_fit,
        load_specs=load_specs,
        static_features=static_features,
        snapshot_epochs=snapshot_epochs,
        history_loss=history_loss,
        show_hinges=True,
        show_hinge_indices=False,
        show_face_indices=False,
        show_external_forces=False,
        show_target=True,
    )

    print(f"\nAll outputs saved to: {out_dir}")


if __name__ == '__main__':
    main()
