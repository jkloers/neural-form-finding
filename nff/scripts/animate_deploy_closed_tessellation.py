"""Animate the deployment of a flat closed kirigami sheet under Stage-2 physics.

Clamps the whole bottom row and pulls the whole top row upward, then animates
the incremental load-step history (flat -> deployed) as a GIF. Reuses the
Stage-2 solver via ``_execute_stage2_physics`` (no physics code modified).

Run:
    JAX_PLATFORMS=cpu conda run -n kgnn_mac \
        python nff/scripts/animate_deploy_closed_tessellation.py
"""

import os

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp

from nff.topology.closed_builder import build_closed_tessellation
from nff.stages.state import CentroidalState
from nff.stages.pipeline import _execute_stage2_physics
from nff.config.experiment import PhysicsConfig
from nff.utils.visualization import animate_tessellation
from nff.scripts.deploy_closed_tessellation import deformed_vertices

OUTPUT_DIR = os.path.join("data", "outputs")

M, N = 4, 4
R_ASPECT = 0.45
FORCE_Y = 25.0          # upward force per top-row face
K_STRETCH = 1000.0      # stiff ligaments keep hinges connected
K_SHEAR = 1000.0
K_ROT = 0.5             # soft rotational hinge spring -> easy deployment
NUM_STEPS = 20


def global_vertices(tess, node_positions) -> np.ndarray:
    """Pack per-face node positions into a flat (V, 2) global vertex array."""
    verts = np.array(tess.vertices, dtype=float)
    for f_id, face in enumerate(tess.faces):
        for local, gv in enumerate(face.vertex_indices):
            verts[gv] = node_positions[f_id, local]
    return verts


def main():
    tess = build_closed_tessellation(M, N, r=R_ASPECT,
                                     k_stretch=K_STRETCH, k_shear=K_SHEAR, k_rot=K_ROT)

    bottom_row = [i * N + 0 for i in range(M)]          # row 0 = bottom
    top_row = [i * N + (N - 1) for i in range(M)]
    for f in bottom_row:
        tess.set_face_dofs(f, [0, 1, 2])                 # clamp whole bottom edge
    for f in top_row:
        tess.set_face_load(f, 1, FORCE_Y)                # pull whole top edge up
    print(f"panels={len(tess.faces)} clamp={bottom_row} load={top_row} Fy={FORCE_Y}")

    state = CentroidalState.from_tessellation(tess)
    physics_cfg = PhysicsConfig(
        domain_restriction=0.0, use_contact=False, k_contact=0.0,
        min_angle=1.0, cutoff_angle=5.0, linearized_strains=True,
        incremental=True, num_load_steps=NUM_STEPS,
        solver_maxiter=1000, solver_tol=1e-5, updated_lagrangian=False,
    )

    solution, _ = _execute_stage2_physics(state, physics_cfg, load_specs=None)
    history = solution.fields                            # (NUM_STEPS, n_faces, 3)
    max_rot = float(jnp.max(jnp.abs(history[-1, :, 2])) * 180.0 / jnp.pi)
    print(f"final max |rotation|={max_rot:.2f} deg")

    # Frame 0 = flat; then one frame per load step.
    state_history = [np.array(tess.vertices, dtype=float)]
    for k in range(history.shape[0]):
        state_history.append(global_vertices(tess, deformed_vertices(state, history[k])))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "deploy_closed_tessellation.gif")
    animate_tessellation(
        tess, state_history, filepath=out_path, fps=12,
        show_hinges=False, show_face_indices=False, show_hinge_indices=False,
        show_target=False, color_faces="#F58025",
    )

    # Static frame strip for quick inspection.
    import matplotlib.pyplot as plt
    from nff.utils.visualization import plot_tessellation
    n = len(state_history)
    picks = [0, n // 3, 2 * n // 3, n - 1]
    fig, axes = plt.subplots(1, len(picks), figsize=(5 * len(picks), 5), facecolor="white")
    for ax, k in zip(axes, picks):
        snap = tess.copy()
        snap.update_vertices(state_history[k])
        plot_tessellation(snap, ax=ax, show_hinges=False, show_face_indices=False,
                          show_hinge_indices=False, show_target=False, color_faces="#F58025")
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"step {k}/{n - 1}")
    strip_path = os.path.join(OUTPUT_DIR, "deploy_closed_tessellation_strip.png")
    fig.savefig(strip_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {strip_path}")


if __name__ == "__main__":
    main()
