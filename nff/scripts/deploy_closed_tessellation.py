"""Deploy a flat closed kirigami sheet with the Stage-2 physics solver.

Wires the closed-state builder directly into the static physics solver,
bypassing Stage 0 (mapping) and Stage 1 (validity) — the flat sheet is already
geometrically valid. A face on the bottom row is clamped and an upward force is
applied to a face on the top row; the solver minimises potential energy and the
voids open (the sheet deploys).

Reuses ``_execute_stage2_physics`` from the main pipeline, so no physics code is
modified.

Run:
    JAX_PLATFORMS=cpu conda run -n kgnn_mac \
        python nff/scripts/deploy_closed_tessellation.py
"""

import os

import jax
jax.config.update("jax_enable_x64", True)        # match train.py physics precision

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

from nff.topology.closed_builder import build_closed_tessellation
from nff.stages.state import CentroidalState
from nff.stages.pipeline import _execute_stage2_physics
from nff.config.experiment import PhysicsConfig
from nff.utils.linalg import rotation_matrix
from nff.utils.visualization import plot_tessellation

OUTPUT_DIR = os.path.join("data", "outputs")

# ── Problem definition ────────────────────────────────────────────────────────
M, N = 4, 4
R_ASPECT = 0.45
CLAMP_FACE = 4      # bottom of column 1 (face index = col*N + row; row 0 = bottom)
LOAD_FACE = 7       # top of column 1
FORCE_Y = 60.0      # upward force magnitude
K_STRETCH = 1000.0
K_SHEAR = 1000.0
K_ROT = 0.5         # soft rotational hinge spring -> easy deployment


def deformed_vertices(state: CentroidalState, displacement: jnp.ndarray) -> np.ndarray:
    """Apply a (n_faces, 3) rigid-body displacement field to the panel vertices.

    Args:
        state: reference CentroidalState.
        displacement: (n_faces, 3) = [dx, dy, dtheta] per face.

    Returns:
        (n_faces, max_nodes, 2) deformed node positions.
    """
    centroids = state.face_centroids + displacement[:, :2]
    thetas = displacement[:, 2]
    rot = jax.vmap(rotation_matrix)(thetas)                       # (n_faces, 2, 2)
    rotated = jnp.einsum("fij,fnj->fni", rot, state.centroid_node_vectors)
    return np.asarray(centroids[:, None, :] + rotated)


def write_deformed_into(tess, node_positions):
    """Return a copy of ``tess`` with vertices set to the deformed positions."""
    deformed = tess.copy()
    new_vertices = np.array(deformed.vertices, dtype=float)
    for f_id, face in enumerate(deformed.faces):
        for local, gv in enumerate(face.vertex_indices):
            new_vertices[gv] = node_positions[f_id, local]
    deformed.update_vertices(new_vertices)
    return deformed


def main():
    # Build the flat closed sheet and attach boundary conditions.
    tess = build_closed_tessellation(M, N, r=R_ASPECT,
                                     k_stretch=K_STRETCH, k_shear=K_SHEAR, k_rot=K_ROT)
    tess.set_face_dofs(CLAMP_FACE, [0, 1, 2])         # clamp (Dirichlet)
    tess.set_face_load(LOAD_FACE, 1, FORCE_Y)          # upward force (Neumann, dof 1 = y)
    print(f"panels={len(tess.faces)} hinges={len(tess.hinges)} "
          f"clamp={CLAMP_FACE} load={LOAD_FACE} Fy={FORCE_Y}")

    state = CentroidalState.from_tessellation(tess)

    physics_cfg = PhysicsConfig(
        domain_restriction=0.0,
        use_contact=False,
        k_contact=0.0,
        min_angle=1.0,
        cutoff_angle=5.0,
        linearized_strains=True,
        incremental=True,
        num_load_steps=10,
        solver_maxiter=1000,
        solver_tol=1e-5,
        updated_lagrangian=False,
    )

    solution, _ = _execute_stage2_physics(state, physics_cfg, load_specs=None)
    final_disp = solution.fields[-1]                   # (n_faces, 3)
    max_disp = float(jnp.max(jnp.abs(final_disp[:, :2])))
    max_rot_deg = float(jnp.max(jnp.abs(final_disp[:, 2])) * 180.0 / jnp.pi)
    print(f"max |translation|={max_disp:.4f}  max |rotation|={max_rot_deg:.2f} deg")

    deployed = write_deformed_into(tess, deformed_vertices(state, final_disp))

    # ── Figure: flat (reference) vs deployed ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), facecolor="white")
    for ax, t, title in ((axes[0], tess, "Flat (closed)"),
                         (axes[1], deployed, "Deployed")):
        plot_tessellation(t, ax=ax, show_target=False, show_hinges=False,
                          show_face_indices=False, show_hinge_indices=False,
                          color_faces="#F58025")
        # Clamp marker (gray) and load arrow (red).
        clamp_c = np.asarray(t.faces[CLAMP_FACE].centroid(t.vertices))
        ax.scatter(*clamp_c, s=140, marker="s", color="#6C757D", zorder=10)
        load_c = np.asarray(t.faces[LOAD_FACE].centroid(t.vertices))
        ax.annotate("", xy=(load_c[0], load_c[1] + 0.6), xytext=(load_c[0], load_c[1]),
                    arrowprops=dict(arrowstyle="-|>", color="#D62828", lw=2.5), zorder=10)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "deploy_closed_tessellation.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
