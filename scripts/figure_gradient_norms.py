#!/usr/bin/env python3
"""
Figure: measured gradient norms at each stage boundary of the differentiable pipeline.

Uses jax.vjp to propagate the gradient backward stage by stage and measures
the Frobenius norm  ||∂L/∂y_i||  at each stage output boundary.

Produces: data/outputs/notebook_figures/gradient_norms.png / .pdf
"""

import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
import sys
_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.abspath(_root))
sys.path.insert(0, os.path.abspath(os.path.join(_root, 'src')))

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from types import SimpleNamespace

# ── Project imports ────────────────────────────────────────────────────────
from problem.config import (
    load_arch_config, load_problem_suite, merge_arch_problem, _parse_full_raw
)
from topology.builder import build_tessellation
from problem.conditions import configure_tessellation
from jax_backend.state import CentroidalState
from jax_backend.initial_map import apply_gnn_mapping
from jax_backend.validity_solver import solve_geometric_validity
from jax_backend.physics_solver.energy import build_potential_energy
from jax_backend.physics_solver.statics import setup_static_solver
from jax_backend.physics_solver.params import ReferenceGeometry, build_control_params
from jax_backend.gnn.graph_builder import build_static_graph_features
from jax_backend.gnn.egnn import init_egnn
from problem.targets import get_target_points


# ═══════════════════════════════════════════════════════════════════════════
# 0.  Setup — egnn_optimal arch + p001 problem (anchoring=500, face_area=10)
# ═══════════════════════════════════════════════════════════════════════════

ARCH_PATH  = "data/configs/architectures/egnn_optimal.yaml"
SUITE_PATH = "data/configs/problems/suite_2x2_rdqk.yaml"
PROBLEM_ID = "p001"

print(f"Loading arch:  {ARCH_PATH}")
print(f"Loading suite: {SUITE_PATH}  (problem: {PROBLEM_ID})")

arch_raw     = load_arch_config(ARCH_PATH)
all_problems = load_problem_suite(SUITE_PATH)
problem      = next(p for p in all_problems if p['id'] == PROBLEM_ID)
merged       = merge_arch_problem(arch_raw, problem)
config       = _parse_full_raw(merged, os.path.dirname(ARCH_PATH))

topo = config.topology
tessellation = build_tessellation(
    topo.get('pattern'), topo.get('width', 2), topo.get('height', 2))

requested_area = topo.get('total_area')
if requested_area:
    current_area = tessellation.compute_total_area()
    scale = np.sqrt(requested_area / current_area)
    tessellation.update_vertices(tessellation.vertices * scale)

topo_obj = SimpleNamespace(**topo)
configure_tessellation(tessellation, topo_obj)

target_center = np.array(getattr(config.target, 'center', [0.0, 0.0]), dtype=float)
tess_centroid = np.mean(tessellation.get_face_centroids(), axis=0)
tessellation.update_vertices(tessellation.vertices - tess_centroid + target_center)

initial_state = CentroidalState.from_tessellation(tessellation, target_cfg=config.target)

# GNN params
static_features = build_static_graph_features(initial_state)
node_feat_dim = static_features['node_feat_dim']
gnn_cfg = config.mapping.params if isinstance(config.mapping.params, dict) else {}
hidden_dim = int(gnn_cfg.get('hidden_dim', 32))
num_layers  = int(gnn_cfg.get('num_layers', 4))
seed = int(gnn_cfg.get('seed', 2))
key = jax.random.PRNGKey(seed)
map_params = init_egnn(key, node_feat_dim, hidden_dim, num_layers)

target_params = {
    'type': config.target.type,
    'center': config.target.center,
    'radius': config.target.radius,
}
target_cloud = jnp.array(get_target_points(target_params, n_points=200))

print(f"EGNN: hidden={hidden_dim}, layers={num_layers}")
print(f"n_faces={initial_state.face_centroids.shape[0]}")


# ═══════════════════════════════════════════════════════════════════════════
# Direct Chamfer loss — avoids solution.energies dict issues inside jax.grad
# ═══════════════════════════════════════════════════════════════════════════

def _chamfer_loss(boundary_pts, target_cloud_):
    dist2 = jnp.sum((boundary_pts[:, None, :] - target_cloud_[None, :, :])**2, axis=-1)
    precision = jnp.mean(jnp.min(dist2, axis=1))
    coverage  = jnp.mean(jnp.min(dist2, axis=0))
    return precision + coverage


# ═══════════════════════════════════════════════════════════════════════════
# Helper: build the Stage 2 + Loss function, parameterised by valid_state
# ═══════════════════════════════════════════════════════════════════════════

def _stage2_and_loss(face_centroids_valid, centroid_node_vectors_valid):
    """Runs Stage 2 (physics) + Chamfer loss given a valid_state geometry."""
    vs = initial_state._replace(
        face_centroids=face_centroids_valid,
        centroid_node_vectors=centroid_node_vectors_valid,
        loaded_face_DOF_pairs=_valid_state_ref.loaded_face_DOF_pairs,
        boundary_face_node_ids=_valid_state_ref.boundary_face_node_ids,
        constrained_face_DOF_pairs=_valid_state_ref.constrained_face_DOF_pairs,
        k_stretch=_valid_state_ref.k_stretch,
        k_shear=_valid_state_ref.k_shear,
        k_rot=_valid_state_ref.k_rot,
        density=_valid_state_ref.density,
    )

    geometry = ReferenceGeometry.from_centroidal_state(vs)
    potential_energy_fn = build_potential_energy(
        bond_connectivity=geometry.bond_connectivity,
        linearized_strains=config.physics.linearized_strains,
        use_contact=config.physics.use_contact,
    )
    loading_fn = vs.get_loading_function()
    solve_statics_fn = setup_static_solver(
        geometry=geometry,
        energy_fn=potential_energy_fn,
        loaded_face_DOF_pairs=vs.loaded_face_DOF_pairs if loading_fn else None,
        loading_fn=loading_fn,
        constrained_face_DOF_pairs=vs.constrained_face_DOF_pairs,
        incremental=config.physics.incremental,
        num_steps=config.physics.num_load_steps,
        solver_maxiter=config.physics.solver_maxiter,
        solver_tol=config.physics.solver_tol,
    )
    control_params = build_control_params(
        geometry=geometry,
        k_stretch=vs.k_stretch,
        k_shear=vs.k_shear,
        k_rot=vs.k_rot,
        density=vs.density,
        k_contact=config.physics.k_contact,
        min_angle=config.physics.min_angle,
        cutoff_angle=config.physics.cutoff_angle,
        use_contact=config.physics.use_contact,
    )
    initial_displacements = jnp.zeros((geometry.n_faces, 3), dtype=float)
    solution = solve_statics_fn(initial_displacements=initial_displacements,
                                control_params=control_params)

    # Reconstruct boundary vertex positions after physics
    final_u = solution.fields[-1]                        # (n_faces, 3)
    final_centroids = vs.face_centroids + final_u[:, :2]
    final_thetas    = final_u[:, 2]

    b_face_ids      = vs.boundary_face_node_ids[:, 0]
    b_local_ids     = vs.boundary_face_node_ids[:, 1]
    b_centroids     = final_centroids[b_face_ids]
    b_vecs          = vs.centroid_node_vectors[b_face_ids, b_local_ids]
    b_thetas        = final_thetas[b_face_ids]
    cos_t, sin_t    = jnp.cos(b_thetas), jnp.sin(b_thetas)
    rotated_vecs    = jnp.stack([
        cos_t * b_vecs[:, 0] - sin_t * b_vecs[:, 1],
        sin_t * b_vecs[:, 0] + cos_t * b_vecs[:, 1],
    ], axis=-1)
    boundary_pts    = b_centroids + rotated_vecs         # (n_boundary, 2)

    return _chamfer_loss(boundary_pts, target_cloud)


def _stage1_2_and_loss(face_centroids_mapped, centroid_node_vectors_mapped):
    """Runs Stage 1 (validity) + Stage 2 (physics) + Loss given Stage 0 output."""
    ms = initial_state._replace(
        face_centroids=face_centroids_mapped,
        centroid_node_vectors=centroid_node_vectors_mapped,
        loaded_face_DOF_pairs=_valid_state_ref.loaded_face_DOF_pairs,
        boundary_face_node_ids=_valid_state_ref.boundary_face_node_ids,
        constrained_face_DOF_pairs=_valid_state_ref.constrained_face_DOF_pairs,
        k_stretch=_valid_state_ref.k_stretch,
        k_shear=_valid_state_ref.k_shear,
        k_rot=_valid_state_ref.k_rot,
        density=_valid_state_ref.density,
    )
    vs = solve_geometric_validity(ms, target_cloud, validity_cfg=config.validity)
    return _stage2_and_loss(vs.face_centroids, vs.centroid_node_vectors)


def _full_loss(theta):
    """Full pipeline: Stage 0 + Stage 1 + Stage 2 + Chamfer loss."""
    ms = apply_gnn_mapping(initial_state, theta, static_features, map_type='gnn_egnn')
    vs = solve_geometric_validity(ms, target_cloud, validity_cfg=config.validity)
    return _stage2_and_loss(vs.face_centroids, vs.centroid_node_vectors)


# ═══════════════════════════════════════════════════════════════════════════
# 1.  Forward pass — get reference intermediate states
# ═══════════════════════════════════════════════════════════════════════════
print("\n── Forward pass ──")
mapped_state = apply_gnn_mapping(
    initial_state, map_params, static_features, map_type='gnn_egnn')
_valid_state_ref = solve_geometric_validity(
    mapped_state, target_cloud, validity_cfg=config.validity)

# ═══════════════════════════════════════════════════════════════════════════
# 2.  Gradient norms at each stage boundary  (backward pass per boundary)
# ═══════════════════════════════════════════════════════════════════════════

def _tree_norm(pytree):
    leaves = jax.tree_util.tree_leaves(pytree)
    return float(jnp.sqrt(sum(jnp.sum(g**2) for g in leaves)))

def _n_params(pytree):
    leaves = jax.tree_util.tree_leaves(pytree)
    return sum(int(np.prod(g.shape)) for g in leaves)


print("── Computing grad at Stage 1 output (entering Stage 2) ──")
g_fc_valid, g_cnv_valid = jax.grad(_stage2_and_loss, argnums=(0, 1))(
    _valid_state_ref.face_centroids, _valid_state_ref.centroid_node_vectors)
norm_stage1_out = float(jnp.sqrt(jnp.sum(g_fc_valid**2) + jnp.sum(g_cnv_valid**2)))
n_stage1_out = int(g_fc_valid.size + g_cnv_valid.size)
print(f"  ||∂L/∂y₁||_F = {norm_stage1_out:.4e}  ({n_stage1_out} DOFs)")

print("── Computing grad at Stage 0 output (entering Stage 1) ──")
g_fc_mapped, g_cnv_mapped = jax.grad(_stage1_2_and_loss, argnums=(0, 1))(
    mapped_state.face_centroids, mapped_state.centroid_node_vectors)
norm_stage0_out = float(jnp.sqrt(jnp.sum(g_fc_mapped**2) + jnp.sum(g_cnv_mapped**2)))
n_stage0_out = int(g_fc_mapped.size + g_cnv_mapped.size)
print(f"  ||∂L/∂y₀||_F = {norm_stage0_out:.4e}  ({n_stage0_out} DOFs)")

print("── Computing grad at θ (full pipeline) ──")
g_theta = jax.grad(_full_loss)(map_params)
norm_theta = _tree_norm(g_theta)
n_theta = _n_params(g_theta)
print(f"  ||∂L/∂θ||_F  = {norm_theta:.4e}  ({n_theta} params)")


# ═══════════════════════════════════════════════════════════════════════════
# 3.  Attenuation factors
# ═══════════════════════════════════════════════════════════════════════════
attn_stage2 = norm_stage1_out / norm_stage0_out if norm_stage0_out > 0 else 0.0
attn_stage1 = norm_stage0_out / norm_theta       if norm_theta > 0      else 0.0

print(f"\nAttenuation Stage 1→2  = {attn_stage2:.3f}  "
      f"(loss from {norm_stage1_out:.2e} → {norm_stage0_out:.2e})")
print(f"Attenuation Stage 0→1  = {attn_stage1:.3f}  "
      f"(loss from {norm_stage0_out:.2e} → {norm_theta:.2e})")

# RMS gradients (per-DOF, more scale-invariant)
rms_stage1 = norm_stage1_out / np.sqrt(n_stage1_out)
rms_stage0 = norm_stage0_out / np.sqrt(n_stage0_out)
rms_theta  = norm_theta      / np.sqrt(n_theta)

print(f"\nRMS gradient (per DOF):")
print(f"  Stage 1 output : {rms_stage1:.4e}")
print(f"  Stage 0 output : {rms_stage0:.4e}")
print(f"  θ (EGNN params): {rms_theta:.4e}")


# ═══════════════════════════════════════════════════════════════════════════
# 4.  Figure — single clean panel, RMS per DOF on log scale
# ═══════════════════════════════════════════════════════════════════════════
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import FancyArrowPatch

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

P_ORANGE = "#F58025"
P_DARK   = "#2B2B2B"
P_GREY   = "#AAAAAA"
P_LGREY  = "#EEEEEE"
P_BG     = "#FFFFFF"

# ── Data  (left-to-right = backward gradient direction: Loss → Params) ────
labels = [
    f"Stage 1 output  $\\mathbf{{y}}_1$\n({n_stage1_out} DOFs)",
    f"Stage 0 output  $\\mathbf{{y}}_0$\n({n_stage0_out} DOFs)",
    f"GNN parameters  $\\boldsymbol{{\\theta}}$\n({n_theta:,} params)",
]
rms_vals = [rms_stage1, rms_stage0, rms_theta]
n_vals   = [n_stage1_out, n_stage0_out, n_theta]
raw_vals = [norm_stage1_out, norm_stage0_out, norm_theta]
colors   = [P_DARK, P_DARK, P_ORANGE]

xs    = np.arange(len(labels))
bar_w = 0.48

# ── Figure ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.5, 4.4), facecolor=P_BG)
ax.set_facecolor(P_BG)

bars = ax.bar(xs, rms_vals, width=bar_w, color=colors,
              edgecolor=P_DARK, linewidth=0.9, zorder=3)

# Value labels: inside tall bars, above short bar
y_top = max(rms_vals) * 12   # top of y-axis (set below)
for bar, rms, raw, n, col in zip(bars, rms_vals, raw_vals, n_vals, colors):
    cx   = bar.get_x() + bar.get_width() / 2
    tall = (rms > min(rms_vals) * 5)  # True for both dark bars, False for orange
    if tall:
        # Inside the bar, white text
        ax.text(cx, rms * 0.35,
                f"RMS\n{rms:.2e}",
                ha='center', va='center', fontsize=8, color='white',
                fontweight='bold', multialignment='center', zorder=5)
        ax.text(cx, rms * 0.06,
                r"$\|\nabla\|_F=" + f"{raw:.2f}$",
                ha='center', va='center', fontsize=7.5, color='white',
                alpha=0.85, zorder=5)
    else:
        # Above the bar, dark text
        ax.text(cx, rms * 1.6,
                f"RMS = {rms:.2e}\n"
                r"$\|\nabla\|_F$" + f" = {raw:.2f}",
                ha='center', va='bottom', fontsize=7.8, color=P_DARK,
                multialignment='center')

# Attenuation annotations between bars
def _attn_label(ax, x0, x1, v0, v1):
    factor = v1 / v0
    y_mid  = np.exp((np.log(v0) + np.log(v1)) / 2)
    xm = (x0 + x1) / 2
    # Connector line
    ax.plot([x0 + bar_w/2, x1 - bar_w/2], [v0, v1],
            color=P_GREY, lw=1.3, ls='--', zorder=2)
    ax.text(xm, y_mid * 0.82,
            f"$\\times${factor:.2f}",
            ha='center', va='top', fontsize=9.5, color=P_GREY,
            fontweight='bold')

_attn_label(ax, 0, 1, rms_stage1, rms_stage0)
_attn_label(ax, 1, 2, rms_stage0, rms_theta)

# Axes
ax.set_yscale('log')
ax.set_xticks(xs)
ax.set_xticklabels(labels, fontsize=9, multialignment='center', linespacing=1.4)
ax.set_ylabel(r"$\|\partial\mathcal{L}/\partial\cdot\|_F\;/\;\sqrt{n}$"
              "\n(RMS gradient per DOF)", fontsize=9.5, labelpad=8)
ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())
ax.grid(axis='y', color=P_LGREY, linestyle='-', linewidth=0.7, zorder=0)
ax.set_axisbelow(True)

# Direction indicator under x-axis
ax.annotate(
    "Backward gradient propagation: Loss $\\rightarrow\\;\\mathbf{y}_1\\rightarrow\\;\\mathbf{y}_0\\rightarrow\\;\\boldsymbol{\\theta}$"
    "   (left = close to loss, right = parameters)",
    xy=(0.5, -0.18), xycoords='axes fraction',
    ha='center', va='top', fontsize=8.2, color="#555555",
    style='italic', annotation_clip=False,
)

ax.set_title(
    "Gradient Norm at Each Stage Boundary — Three-Stage Differentiable Pipeline\n"
    r"(EGNN, hidden=" + f"{hidden_dim}" + r", layers=" + f"{num_layers}" +
    r";  $\lambda_{\rm anchoring}=500$, problem p001)",
    fontsize=10, pad=10,
)

plt.tight_layout(rect=[0, 0.07, 1, 1])

# ── Save ─────────────────────────────────────────────────────────────────
out_dir = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "data", "outputs", "notebook_figures")
)
os.makedirs(out_dir, exist_ok=True)

for ext in ("png", "pdf"):
    out_path = os.path.join(out_dir, f"gradient_norms.{ext}")
    fig.savefig(out_path, dpi=250, bbox_inches="tight", facecolor=P_BG)
    print(f"Saved → {out_path}")
