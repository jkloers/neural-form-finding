#!/usr/bin/env python3
"""
Figure: ||∂L/∂θ||_F vs anchoring weight.

For each anchoring value the full pipeline runs once (Stage 0 → Stage 1 → Stage 2 → Loss)
and jax.grad gives the gradient norm at the EGNN parameters.

Produces: data/outputs/notebook_figures/anchoring_gradient.{png,pdf}
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

from problem.config import (
    ValidityConfig,
    load_arch_config, load_problem_suite, merge_arch_problem, _parse_full_raw,
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
# Setup — fixed across all anchoring values
# ═══════════════════════════════════════════════════════════════════════════

ARCH_PATH  = "data/configs/architectures/egnn_optimal.yaml"
SUITE_PATH = "data/configs/problems/suite_2x2_rdqk.yaml"
PROBLEM_ID = "p001"

arch_raw     = load_arch_config(ARCH_PATH)
all_problems = load_problem_suite(SUITE_PATH)
problem      = next(p for p in all_problems if p['id'] == PROBLEM_ID)
merged       = merge_arch_problem(arch_raw, problem)
config       = _parse_full_raw(merged, os.path.dirname(ARCH_PATH))

topo = config.topology
tessellation = build_tessellation(topo.get('pattern'), topo.get('width', 2), topo.get('height', 2))

requested_area = topo.get('total_area')
if requested_area:
    scale = np.sqrt(requested_area / tessellation.compute_total_area())
    tessellation.update_vertices(tessellation.vertices * scale)

configure_tessellation(tessellation, SimpleNamespace(**topo))

target_center = np.array(getattr(config.target, 'center', [0.0, 0.0]), dtype=float)
tessellation.update_vertices(
    tessellation.vertices - np.mean(tessellation.get_face_centroids(), axis=0) + target_center)

initial_state   = CentroidalState.from_tessellation(tessellation, target_cfg=config.target)
static_features = build_static_graph_features(initial_state)
node_feat_dim   = static_features['node_feat_dim']

gnn_cfg    = config.mapping.params if isinstance(config.mapping.params, dict) else {}
hidden_dim = int(gnn_cfg.get('hidden_dim', 32))
num_layers = int(gnn_cfg.get('num_layers', 4))
seed       = int(gnn_cfg.get('seed', 2))

map_params  = init_egnn(jax.random.PRNGKey(seed), node_feat_dim, hidden_dim, num_layers)
target_cloud = jnp.array(get_target_points(
    {'type': config.target.type, 'center': config.target.center, 'radius': config.target.radius},
    n_points=200))

print(f"EGNN: hidden={hidden_dim}, layers={num_layers}, n_params={sum(int(np.prod(g.shape)) for g in jax.tree_util.tree_leaves(map_params))}")

# ── Chamfer loss (avoids solution.energies string-indexing issue) ─────────
def _chamfer(pts, cloud):
    d2 = jnp.sum((pts[:, None] - cloud[None]) ** 2, axis=-1)
    return jnp.mean(jnp.min(d2, axis=1)) + jnp.mean(jnp.min(d2, axis=0))


# ── Full pipeline loss, parameterised by validity_cfg ────────────────────
def _full_loss(theta, validity_cfg):
    ms = apply_gnn_mapping(initial_state, theta, static_features, map_type='gnn_egnn')
    vs = solve_geometric_validity(ms, target_cloud, validity_cfg=validity_cfg)

    geometry   = ReferenceGeometry.from_centroidal_state(vs)
    energy_fn  = build_potential_energy(
        bond_connectivity=geometry.bond_connectivity,
        linearized_strains=config.physics.linearized_strains,
        use_contact=config.physics.use_contact,
    )
    loading_fn = vs.get_loading_function()
    solve_fn   = setup_static_solver(
        geometry=geometry,
        energy_fn=energy_fn,
        loaded_face_DOF_pairs=vs.loaded_face_DOF_pairs if loading_fn else None,
        loading_fn=loading_fn,
        constrained_face_DOF_pairs=vs.constrained_face_DOF_pairs,
        incremental=config.physics.incremental,
        num_steps=config.physics.num_load_steps,
        solver_maxiter=config.physics.solver_maxiter,
        solver_tol=config.physics.solver_tol,
    )
    ctrl = build_control_params(
        geometry=geometry,
        k_stretch=vs.k_stretch, k_shear=vs.k_shear,
        k_rot=vs.k_rot, density=vs.density,
        k_contact=config.physics.k_contact,
        min_angle=config.physics.min_angle,
        cutoff_angle=config.physics.cutoff_angle,
        use_contact=config.physics.use_contact,
    )
    sol   = solve_fn(jnp.zeros((geometry.n_faces, 3), dtype=float), ctrl)
    u_fin = sol.fields[-1]

    fc_fin  = vs.face_centroids + u_fin[:, :2]
    theta_f = u_fin[:, 2]
    b_fi    = vs.boundary_face_node_ids[:, 0]
    b_li    = vs.boundary_face_node_ids[:, 1]
    b_c     = fc_fin[b_fi]
    b_v     = vs.centroid_node_vectors[b_fi, b_li]
    b_t     = theta_f[b_fi]
    rot_v   = jnp.stack([
        jnp.cos(b_t) * b_v[:, 0] - jnp.sin(b_t) * b_v[:, 1],
        jnp.sin(b_t) * b_v[:, 0] + jnp.cos(b_t) * b_v[:, 1],
    ], axis=-1)
    return _chamfer(b_c + rot_v, target_cloud)


# ═══════════════════════════════════════════════════════════════════════════
# Sweep over anchoring values
# ═══════════════════════════════════════════════════════════════════════════

ANCHORING_VALUES = [0, 50, 200, 500, 2000]

results = []
base_weights = config.validity.weights.copy()

for anch in ANCHORING_VALUES:
    print(f"\n── anchoring = {anch} ──")
    w     = {**base_weights, 'anchoring': float(anch)}
    vcfg  = ValidityConfig(weights=w)
    grad  = jax.grad(_full_loss)(map_params, vcfg)
    leaves = jax.tree_util.tree_leaves(grad)
    norm  = float(jnp.sqrt(sum(jnp.sum(g**2) for g in leaves)))
    n_p   = sum(int(np.prod(g.shape)) for g in leaves)
    rms   = norm / np.sqrt(n_p)
    print(f"  ||∂L/∂θ||_F = {norm:.4e}   RMS = {rms:.4e}")
    results.append({'anchoring': anch, 'norm': norm, 'rms': rms, 'n': n_p})


# ═══════════════════════════════════════════════════════════════════════════
# Figure
# ═══════════════════════════════════════════════════════════════════════════
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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

anchors = [r['anchoring'] for r in results]
norms   = [r['norm']      for r in results]
rms_v   = [r['rms']       for r in results]

# ── The data shows binary behaviour: anchoring=0 → exact zero, else ~constant.
# Use linear scale; replace 0 with sentinel for bar height, annotate separately.

import matplotlib.patches as mpatches

norms_nonzero = [v for v in norms if v > 0]
y_max  = max(norms_nonzero) * 1.35
y_zero = y_max * 0.04   # small visible stub for the zero bar

plot_norms = [y_zero if v == 0.0 else v for v in norms]
colors_b   = [P_GREY  if a == 0 else P_DARK for a in anchors]

xs    = np.arange(len(anchors))
bar_w = 0.55

fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor=P_BG)
ax.set_facecolor(P_BG)

bars = ax.bar(xs, plot_norms, width=bar_w, color=colors_b,
              edgecolor=P_DARK, linewidth=0.9, zorder=3)

# Labels
for bar, val, plot_val, a in zip(bars, norms, plot_norms, anchors):
    cx = bar.get_x() + bar.get_width() / 2
    if val == 0.0:
        ax.text(cx, plot_val + y_max * 0.02,
                "= 0  (exact)",
                ha='center', va='bottom', fontsize=9, color=P_GREY,
                fontweight='bold')
    else:
        ax.text(cx, val * 0.45,
                f"{val:.3f}",
                ha='center', va='center', fontsize=9, color='white',
                fontweight='bold', zorder=5)

# Horizontal dashed line at the "saturated" level
sat = np.mean(norms_nonzero)
ax.axhline(sat, color=P_ORANGE, lw=1.5, ls='--', zorder=2)
ax.text(len(anchors) - 0.42, sat * 1.04,
        f"saturation  {sat:.3f}",
        ha='right', va='bottom', fontsize=8.5, color=P_ORANGE)

# Axes
ax.set_ylim(0, y_max)
ax.set_xticks(xs)
ax.set_xticklabels([str(a) for a in anchors], fontsize=10)
ax.set_xlabel("Anchoring weight  $\\lambda_{\\rm anch}$", fontsize=10)
ax.set_ylabel(r"$\|\partial\mathcal{L}/\partial\boldsymbol{\theta}\|_F$", fontsize=10)
ax.grid(axis='y', color=P_LGREY, linestyle='-', linewidth=0.7, zorder=0)
ax.set_axisbelow(True)

ax.set_title(
    r"Effect of Anchoring Weight on $\|\nabla_\theta\mathcal{L}\|_F$  —  "
    "EGNN Gradient Signal at Initialization\n"
    r"(egnn\_optimal + p001, measured via $\mathtt{jax.grad}$)",
    fontsize=10, pad=10,
)

legend_handles = [
    mpatches.Patch(fc=P_GREY, ec=P_DARK, lw=0.8,
                   label=r"$\lambda=0$ : gradient path blocked — $\|\nabla_\theta\mathcal{L}\|_F = 0$ exactly"),
    mpatches.Patch(fc=P_DARK, ec=P_DARK, lw=0.8,
                   label=r"$\lambda>0$ : gradient path open — signal saturates immediately"),
    mpatches.Patch(fc=P_ORANGE, ec='none',
                   label=r"Saturation level (nearly constant for any $\lambda>0$)"),
]
ax.legend(handles=legend_handles, fontsize=8.2, frameon=True,
          framealpha=0.95, edgecolor=P_GREY, loc='lower right',
          borderpad=0.8, handlelength=1.5)

plt.tight_layout()

# ── Save ─────────────────────────────────────────────────────────────────
out_dir = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "data", "outputs", "notebook_figures"))
os.makedirs(out_dir, exist_ok=True)

for ext in ("png", "pdf"):
    p = os.path.join(out_dir, f"anchoring_gradient.{ext}")
    fig.savefig(p, dpi=250, bbox_inches="tight", facecolor=P_BG)
    print(f"Saved → {p}")
