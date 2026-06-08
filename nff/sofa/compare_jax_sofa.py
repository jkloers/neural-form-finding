"""
nff/sofa/compare_jax_sofa.py
=============================
Qualitative comparison: JAX Stage-2 rigid-face model vs SOFA 3D FEM.

Default config: data/configs/sofa/c001_mpnn_2x2.yaml
  — MPNN Stage-0 mapping, 2×2 RDQK_D tessellation, 16 faces.
  — Stage-1 (alternating projections) closes hinge gaps.
  — Stage-2 (L-BFGS) solves static equilibrium.
  — SOFA 3D FEM runs on the Stage-1 hex mesh.

Workflow
--------
  # Step 1 — run JAX pipeline + build CS mesh (creates a run directory)
  conda run -n kgnn_mac python nff/sofa/compare_jax_sofa.py \\
      --config data/configs/sofa/c001_mpnn_2x2.yaml

  # Optional: load a pre-trained MPNN checkpoint
  conda run -n kgnn_mac python nff/sofa/compare_jax_sofa.py \\
      --config data/configs/sofa/c001_mpnn_2x2.yaml \\
      --checkpoint data/outputs/runs/<run>/best_params.pkl

  # Step 2 — SOFA simulation
  ./sofa/run_sofa.sh sofa/dump_results.py \\
      --config data/configs/sofa/c001_mpnn_2x2.yaml \\
      --out-dir data/outputs/runs/run_<...>_c001_mpnn_2x2/

  # Step 3 — comparison plot (re-run with existing run dir)
  conda run -n kgnn_mac python nff/sofa/compare_jax_sofa.py \\
      --run-dir data/outputs/runs/run_<...>_c001_mpnn_2x2/

Outputs (all in the run directory)
-----------------------------------
  config.yaml                      — copy of the driving config
  stage_0__initial_mapping.png     — JAX Stage-0 (mapping)
  stage_1_deformation.png          — JAX Stage-1 (validity)
  stage_2__static_equilibrium.png  — JAX Stage-2 (physics equilibrium)
  cs_mesh.npz                      — CS-derived hex mesh for SOFA
  sofa_result.npz                  — SOFA simulation output
  compare_jax_sofa.png             — side-by-side JAX vs SOFA comparison
"""

from __future__ import annotations

import argparse
import datetime
import os
import pickle
import shutil
import sys
import pathlib

# ── JAX CPU + x64 must come before any jax import ──────────────────────────
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
import jax
jax.config.update('jax_enable_x64', True)

import numpy as np
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from matplotlib.collections import PolyCollection
import yaml

# ── Path setup ───────────────────────────────────────────────────────────────
REPO = pathlib.Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO))

from types import SimpleNamespace

from nff.topology.builder import build_tessellation
from nff.stages.state import CentroidalState
from nff.sofa.mesh_builder import build_mesh_from_centroidal_state
from nff.sofa.config_to_physical import physical_scale_from_config
from nff.config.experiment import load_and_parse_config
from nff.config.conditions import configure_tessellation
from nff.stages.pipeline import forward_pipeline
from nff.utils.pipeline_viz import visualize_pipeline_results

DEFAULT_CONFIG = REPO / 'data' / 'configs' / 'sofa' / 'c001_mpnn_2x2.yaml'
RUNS_DIR       = REPO / 'data' / 'outputs' / 'runs'


# ─────────────────────────────────────────────────────────────────────────────
# Color helpers for N faces
# ─────────────────────────────────────────────────────────────────────────────

_BASE_COLORS = ['#4472C4', '#ED7D31', '#70AD47', '#FFC000',
                '#5A9BD5', '#F15A29', '#48A463', '#E2A829',
                '#9B59B6', '#1ABC9C', '#E74C3C', '#3498DB',
                '#F39C12', '#2ECC71', '#E91E63', '#00BCD4']


def _face_colors(n_faces: int) -> list:
    if n_faces <= len(_BASE_COLORS):
        return _BASE_COLORS[:n_faces]
    cmap = cm.get_cmap('tab20', n_faces)
    return [cmap(i) for i in range(n_faces)]


# ─────────────────────────────────────────────────────────────────────────────
# Run directory management
# ─────────────────────────────────────────────────────────────────────────────

def _make_run_dir(config_path: pathlib.Path) -> pathlib.Path:
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = RUNS_DIR / f"run_{timestamp}_{config_path.stem}"
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path, run_dir / 'config.yaml')
    return run_dir


def _resolve_run_dir_and_config(args) -> tuple[pathlib.Path, pathlib.Path, dict]:
    if args.run_dir is not None:
        run_dir     = pathlib.Path(args.run_dir).resolve()
        config_path = run_dir / 'config.yaml'
        if not config_path.exists():
            if args.config is not None:
                config_path = pathlib.Path(args.config)
                shutil.copy(config_path, run_dir / 'config.yaml')
            else:
                raise FileNotFoundError(
                    f"No config.yaml in {run_dir}. Pass --config explicitly.")
    else:
        config_path = pathlib.Path(args.config if args.config else DEFAULT_CONFIG)
        run_dir     = _make_run_dir(config_path)

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    return run_dir, config_path, raw


# ─────────────────────────────────────────────────────────────────────────────
# JAX full pipeline (Stage 0 / 1 / 2): MPNN mapping + validity + physics
# ─────────────────────────────────────────────────────────────────────────────

def _build_cs_from_config(config):
    """Build CentroidalState + tessellation in JAX normalized coordinates."""
    topo     = config.topology
    topo_obj = SimpleNamespace(**topo)
    tess     = build_tessellation(
        topo.get('pattern'),
        topo.get('width', 1),
        topo.get('height', 1),
    )
    requested_area = topo.get('total_area')
    if requested_area:
        scale = np.sqrt(requested_area / tess.compute_total_area())
        tess.update_vertices(tess.vertices * scale)
    configure_tessellation(tess, topo_obj)
    return CentroidalState.from_tessellation(tess, target_cfg=config.target), tess


def _init_gnn_params(config, initial_state, checkpoint_path: str = None):
    """Initialise MPNN/EGNN params: load checkpoint if given, else random init.

    Returns
    -------
    params          : JAX PyTree of GNN weights
    static_features : precomputed graph features (closed over before JIT)
    """
    from nff.models.graph_builder import build_static_features

    map_type  = config.mapping.type
    gnn_cfg   = config.mapping.params if isinstance(config.mapping.params, dict) else {}
    hidden_dim = int(gnn_cfg.get('hidden_dim', 32))
    num_layers = int(gnn_cfg.get('num_layers', 3))
    seed       = int(gnn_cfg.get('seed', 0))

    static_features = build_static_features(initial_state, map_type)
    static_features = {**static_features, 'num_layers': num_layers}

    if checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'rb') as f:
            params = pickle.load(f)
        print(f"  Loaded checkpoint: {checkpoint_path}")
        return params, static_features

    if checkpoint_path:
        print(f"  Checkpoint not found ({checkpoint_path}) — using random init (seed={seed})")

    key           = jax.random.PRNGKey(seed)
    node_feat_dim = static_features['node_feat_dim']

    if map_type == 'gnn_mpnn':
        from nff.models.mpnn import init_mpnn
        params = init_mpnn(key, node_feat_dim, hidden_dim, num_layers)
    elif map_type == 'gnn_egnn':
        from nff.models.egnn import init_egnn
        params = init_egnn(key, node_feat_dim, hidden_dim, num_layers)
    else:
        raise ValueError(f"Unsupported GNN map_type: {map_type!r}. "
                         "Only gnn_mpnn and gnn_egnn are supported.")

    return params, static_features


def run_jax_pipeline_and_viz(config_path: pathlib.Path, raw: dict,
                              run_dir: pathlib.Path,
                              checkpoint_path: str = None):
    """Run the full JAX pipeline (Stage 0 → 1 → 2) and save stage plots.

    Stage 0: MPNN mapping (exclusive — no direct_transform or conformal_polynomial).
             Loads pre-trained weights from checkpoint_path if provided; otherwise
             initialises from the seed in config.mapping.params.
    Stage 1: alternating projections → closes hinge gaps → deployed CentroidalState.
    Stage 2: L-BFGS physics equilibrium.

    Returns
    -------
    result       : forward_pipeline output dict
    tessellation : Tessellation (normalized coords)
    config       : parsed ExperimentConfig
    """
    config          = load_and_parse_config(str(config_path))
    initial_state, tessellation = _build_cs_from_config(config)

    # ── Initialise mapping parameters ────────────────────────────────────────
    map_type = config.mapping.type
    if map_type.startswith('gnn_'):
        map_params, static_features = _init_gnn_params(
            config, initial_state, checkpoint_path)
    else:
        # Non-GNN paths (direct_transform, conformal_polynomial) are supported
        # for backward compatibility but not the canonical MPNN path.
        map_params      = config.mapping.params if isinstance(config.mapping.params, dict) else {}
        static_features = None

    load_specs = config.topology.get('loads', []) or []

    print(f"  Stage 0 ({map_type}) → Stage 1 (altproj) → Stage 2 (L-BFGS) ...")
    result = forward_pipeline(
        initial_state,
        target_cfg          = config.target,
        validity_cfg        = config.validity,
        physics_cfg         = config.physics,
        map_type            = map_type,
        map_params          = map_params,
        use_shirley_chiu    = config.mapping.use_shirley_chiu,
        strict_boundary_fit = config.mapping.strict_boundary_fit,
        static_features     = static_features,
        load_specs          = load_specs,
    )

    equil_jax = np.array(result['solution'].fields[-1])
    n_faces   = len(equil_jax)
    print(f"  Stage 2 equilibrium (JAX normalized units):")
    for f in range(n_faces):
        dx, dy, dth = equil_jax[f]
        print(f"    Face {f}: dx={dx:+.3f}  dy={dy:+.3f}  dθ={np.degrees(dth):+.1f}°")

    target_params = {
        'type':   config.target.type,
        'center': config.target.center,
        'radius': config.target.radius,
    }
    print("  Saving stage-0 / stage-1 / stage-2 plots ...")
    visualize_pipeline_results(
        result, tessellation, config, target_params,
        config_name = config_path.stem,
        run_dir     = str(run_dir),
        load_specs  = load_specs,
    )
    return result, tessellation, config


# ─────────────────────────────────────────────────────────────────────────────
# SOFA result loader
# ─────────────────────────────────────────────────────────────────────────────

def load_sofa_result(npz_path: pathlib.Path):
    if not npz_path.exists():
        return None
    data   = np.load(npz_path)
    n_f    = int(data.get('n_faces', 4))
    is_mm  = bool(data.get('is_moment_mode', np.array(False)))
    if is_mm:
        print(f"  SOFA: {len(data['nodes_nat'])} nodes  "
              f"moment={float(data['applied_moment']):.3f} N·m  "
              f"({n_f} faces, moment mode)")
    else:
        print(f"  SOFA: {len(data['nodes_nat'])} nodes  "
              f"angle={float(data['rotation_angle_deg']):.0f}°  "
              f"({n_f} faces, rotation mode)")
    print(f"  Strain energy        : {float(data['strain_energy']):.4e} J")
    print(f"  Max in-plane XY disp : {float(data['max_xy_displacement'])*1e3:.2f} mm")
    print(f"  Max |z| buckling     : {float(data['max_z_displacement'])*1e3:.2f} mm")
    return data


def _n_faces_in_sofa(data) -> int:
    n = int(data.get('n_faces', 0))
    if n == 0:
        n = sum(1 for k in data.files
                if k.startswith('f') and k.endswith('_mask') and k[1:-5].isdigit())
    return n


# ─────────────────────────────────────────────────────────────────────────────
# Face kinematics from SOFA node positions
# ─────────────────────────────────────────────────────────────────────────────

def estimate_sofa_face_rotation(data, face_key: str) -> float:
    mask    = data[face_key].astype(bool)
    pts_nat = data['nodes_nat'][mask, :2]
    pts_cur = data['nodes_cur'][mask, :2]
    cen_nat = pts_nat.mean(axis=0)
    cen_cur = pts_cur.mean(axis=0)
    H       = (pts_nat - cen_nat).T @ (pts_cur - cen_cur)
    U, _, Vt = np.linalg.svd(H)
    R       = Vt.T @ U.T
    return float(np.arctan2(R[1, 0], R[0, 0]))


def sofa_face_centroid_displacement(data, face_key: str) -> np.ndarray:
    mask   = data[face_key].astype(bool)
    z_nat  = data['nodes_nat'][mask, 2]
    top    = z_nat > (z_nat.max() - 1e-9)
    nat_xy = data['nodes_nat'][mask][top, :2].mean(axis=0)
    cur_xy = data['nodes_cur'][mask][top, :2].mean(axis=0)
    return cur_xy - nat_xy


# ─────────────────────────────────────────────────────────────────────────────
# Drawing helpers (N-face aware)
# ─────────────────────────────────────────────────────────────────────────────

def _jax_nat_poly_mm(face_idx, fc, cnv) -> np.ndarray:
    return (fc[face_idx] + cnv[face_idx, :4]) * 1e3


def jax_face_polygon(face_idx, fc, cnv, equil) -> np.ndarray:
    dx, dy, dth = equil[face_idx]
    c, s = np.cos(dth), np.sin(dth)
    R    = np.array([[c, -s], [s, c]])
    new_centroid_mm = (fc[face_idx] + np.array([dx, dy])) * 1e3
    rotated_cnv_mm  = (R @ cnv[face_idx, :4].T).T * 1e3
    return new_centroid_mm + rotated_cnv_mm


def _set_ax_limits(ax, polygons, pad_frac=0.15):
    all_v = np.vstack(polygons)
    span  = max(all_v[:, 0].max() - all_v[:, 0].min(),
                all_v[:, 1].max() - all_v[:, 1].min())
    pad   = span * pad_frac
    ax.set_xlim(all_v[:, 0].min() - pad, all_v[:, 0].max() + pad)
    ax.set_ylim(all_v[:, 1].min() - pad, all_v[:, 1].max() + pad)


def draw_natural_panel(ax, fc, cnv, phys, clamped_faces: list, loaded_faces: list):
    n_faces = len(fc)
    colors  = _face_colors(n_faces)
    ax.set_aspect('equal')
    ax.set_title('Deployed natural state\n(Stage-1 output, hinges closed)', fontsize=10)
    polys = []
    for f in range(n_faces):
        poly = _jax_nat_poly_mm(f, fc, cnv)
        polys.append(poly)
        suffix = (' ◼' if f in clamped_faces else
                  ' ▲' if f in loaded_faces  else '')
        ax.add_patch(plt.Polygon(poly, closed=True, zorder=2,
                                 edgecolor='k', linewidth=0.8,
                                 facecolor=colors[f], alpha=0.75))
        cx, cy = poly.mean(axis=0)
        ax.text(cx, cy, f'F{f}{suffix}', ha='center', va='center',
                fontsize=7 if n_faces > 8 else 9, fontweight='bold')
    _set_ax_limits(ax, polys)
    ax.set_xlabel(f'arm_width={phys.arm_width*1e3:.1f}mm  '
                  f'fold_length={phys.fold_length*1e3:.1f}mm', fontsize=8)
    ax.set_ylabel('mm', fontsize=8)


def draw_jax_panel(ax, equil, fc, cnv, phys, applied_moment,
                   clamped_faces: list, loaded_faces: list):
    n_faces = len(fc)
    colors  = _face_colors(n_faces)
    ax.set_aspect('equal')
    ax.set_title(f'JAX Stage-2\nM={applied_moment:.2f} N·m  '
                 f'k_rot={phys.k_rot:.3f} N·m/rad', fontsize=10)
    nat_polys = [_jax_nat_poly_mm(f, fc, cnv) for f in range(n_faces)]
    def_polys = [jax_face_polygon(f, fc, cnv, equil) for f in range(n_faces)]
    for f in range(n_faces):
        ax.add_patch(plt.Polygon(nat_polys[f], closed=True,
                                 linewidth=0.4, linestyle='--',
                                 edgecolor='gray', facecolor='none', zorder=1))
        ax.add_patch(plt.Polygon(def_polys[f], closed=True, zorder=2,
                                 edgecolor='k', linewidth=0.8,
                                 facecolor=colors[f], alpha=0.75))
        cx, cy = def_polys[f].mean(axis=0)
        ax.text(cx, cy, f'F{f}', ha='center', va='center',
                fontsize=7 if n_faces > 8 else 9, fontweight='bold')
    _set_ax_limits(ax, nat_polys + def_polys)
    rot_vals = [f'F{f}:{np.degrees(equil[f, 2]):+.1f}°' for f in range(min(n_faces, 8))]
    if n_faces > 8:
        rot_vals.append('...')
    ax.set_xlabel('  '.join(rot_vals), fontsize=7)
    ax.set_ylabel('mm', fontsize=8)


def draw_sofa_panel(ax, data, applied_moment):
    n_faces  = _n_faces_in_sofa(data)
    colors   = _face_colors(n_faces)
    ax.set_aspect('equal')
    is_mm    = bool(data.get('is_moment_mode', np.array(False)))
    if is_mm:
        ax.set_title(f'SOFA 3D FEM\nM={applied_moment:.2f} N·m  '
                     f'E=3.5 GPa  t=1 mm', fontsize=10)
    else:
        ax.set_title(f'SOFA 3D FEM\nθ={float(data["rotation_angle_deg"]):.0f}°',
                     fontsize=10)

    nodes_cur_mm = data['nodes_cur'] * 1e3
    hexes        = data['hexes']
    top_quads    = hexes[:, [4, 5, 6, 7]]
    quad_xy      = nodes_cur_mm[:, :2][top_quads]

    quad_colors = np.full(len(hexes), '#cccccc', dtype=object)
    for fi in range(n_faces):
        fkey = f'f{fi}_mask'
        if fkey not in data:
            continue
        mask = data[fkey].astype(bool)
        for qi, quad in enumerate(top_quads):
            if mask[quad].any():
                quad_colors[qi] = colors[fi]

    coll = PolyCollection(quad_xy, facecolors=list(quad_colors), edgecolors='k',
                          linewidths=0.3, zorder=2)
    ax.add_collection(coll)

    xy_mm = nodes_cur_mm[:, :2]
    span  = max(xy_mm[:, 0].max() - xy_mm[:, 0].min(),
                xy_mm[:, 1].max() - xy_mm[:, 1].min()) * 0.15
    ax.set_xlim(xy_mm[:, 0].min() - span, xy_mm[:, 0].max() + span)
    ax.set_ylim(xy_mm[:, 1].min() - span, xy_mm[:, 1].max() + span)

    rot_vals = [f'F{fi}:{np.degrees(estimate_sofa_face_rotation(data, f"f{fi}_mask")):+.1f}°'
                for fi in range(min(n_faces, 8)) if f'f{fi}_mask' in data]
    if n_faces > 8:
        rot_vals.append('...')
    ax.set_xlabel('  '.join(rot_vals), fontsize=7)
    ax.set_ylabel('mm', fontsize=8)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='JAX Stage-2 vs SOFA comparison (MPNN 2×2 pipeline).')
    parser.add_argument('--config',     default=None,
                        help='Path to sofa YAML config. Creates a new run dir.')
    parser.add_argument('--run-dir',    default=None,
                        help='Existing run directory. Reads config.yaml from it.')
    parser.add_argument('--checkpoint', default=None,
                        help='Path to best_params.pkl from a training run. '
                             'If not given, MPNN weights are randomly initialised.')
    args = parser.parse_args()

    run_dir, config_path, raw = _resolve_run_dir_and_config(args)

    phys           = physical_scale_from_config(raw)
    applied_moment = float(raw.get('sofa', {}).get('applied_moment',
                           raw.get('loads', [{}])[0].get('value', 1.0)))

    sofa_npz_path = run_dir / 'sofa_result.npz'
    cs_mesh_path  = run_dir / 'cs_mesh.npz'
    out_png_path  = run_dir / 'compare_jax_sofa.png'

    print("=" * 60)
    print(f"JAX Stage-2 vs SOFA — config: {config_path.name}")
    print(f"  Run dir: {run_dir}")
    print(f"  face_size={phys.face_size*1e3:.1f}mm  "
          f"arm_width={phys.arm_width*1e3:.2f}mm  "
          f"fold_length={phys.fold_length*1e3:.2f}mm")
    print(f"  E={phys.young_modulus/1e9:.1f}GPa  "
          f"ν={phys.poisson_ratio}  "
          f"k_rot={phys.k_rot:.4f}N·m/rad")
    if args.checkpoint:
        print(f"  Checkpoint: {args.checkpoint}")
    print("=" * 60)

    # ── JAX pipeline — Stage 0 (MPNN) → 1 (altproj) → 2 (L-BFGS) ───────────
    print(f"\n[1/3] Running JAX pipeline ...")
    result, tessellation, jax_config = run_jax_pipeline_and_viz(
        config_path, raw, run_dir, checkpoint_path=args.checkpoint)

    # ── Scale Stage-1 (valid_state) to physical units ─────────────────────────
    valid_state = result['valid_state']
    scaled_cs = valid_state._replace(
        face_centroids        = valid_state.face_centroids        * phys.jax_scale,
        centroid_node_vectors = valid_state.centroid_node_vectors * phys.jax_scale,
    )

    n_faces = len(np.array(scaled_cs.face_centroids))
    fc  = np.array(scaled_cs.face_centroids)                    # (n_faces, 2) [m]
    cnv = np.array(scaled_cs.centroid_node_vectors)[:, :4, :]   # (n_faces, 4, 2) [m]

    # Stage-2 equilibrium: translate x,y to metres; dθ stays in radians.
    equil_jax = np.array(result['solution'].fields[-1])          # (n_faces, 3)
    equil = equil_jax.copy()
    equil[:, :2] *= phys.jax_scale                               # [m]

    print(f"  Equilibrium ({n_faces} faces, face_size={phys.face_size*1e3:.1f}mm):")
    for f in range(n_faces):
        dx, dy, dth = equil[f]
        print(f"    F{f}: dx={dx*1e3:+.1f}mm  dy={dy*1e3:+.1f}mm  dθ={np.degrees(dth):+.1f}°")

    # ── Identify clamped / loaded faces from config ────────────────────────────
    bc_cfg      = raw.get('boundary_conditions', {})
    loads_cfg   = raw.get('loads', [])
    clamped_faces = list(bc_cfg.get('clamped_faces', [0]))
    loaded_faces  = [int(l['face']) for l in loads_cfg if 'face' in l]

    # ── Build CS mesh from Stage-1 output ─────────────────────────────────────
    print(f"\n[2/3] Building SOFA mesh from Stage-1 CentroidalState ...")
    cs_nodes, cs_hexes, cs_bc = build_mesh_from_centroidal_state(
        scaled_cs,
        fold_length        = phys.fold_length,
        sheet_thickness    = phys.sheet_thickness,
        arm_width_physical = phys.arm_width,
    )

    # Typed loads (global_frame, tile_to_tile) are handled by force_types.py at
    # pipeline time and are never written to the tessellation, so
    # cs.loaded_face_DOF_pairs is empty for those load specs.  Override
    # clamped/loaded masks here from the config so SOFA always has valid BCs.
    if clamped_faces:
        clamped_union = np.zeros(len(cs_nodes), dtype=bool)
        for f in clamped_faces:
            clamped_union |= cs_bc[f'f{f}']
        cs_bc['clamped'] = clamped_union
        print(f"  Clamped mask: faces {clamped_faces} → "
              f"{int(clamped_union.sum())} nodes")

    if loaded_faces:
        loaded_union = np.zeros(len(cs_nodes), dtype=bool)
        for f in loaded_faces:
            loaded_union |= cs_bc[f'f{f}']
        cs_bc['loaded'] = loaded_union
        print(f"  Loaded  mask: faces {loaded_faces} → "
              f"{int(loaded_union.sum())} nodes")

    # Save cs_mesh.npz with all n_faces masks.
    face_mask_dict = {f'f{i}_mask': cs_bc[f'f{i}'] for i in range(n_faces)}
    np.savez(
        cs_mesh_path,
        nodes        = cs_nodes,
        hexes        = cs_hexes,
        **face_mask_dict,
        clamped_mask = cs_bc['clamped'],
        loaded_mask  = cs_bc['loaded'],
        face_size    = np.float64(phys.face_size),
        n_faces      = np.int32(n_faces),
    )
    print(f"  Mesh: {len(cs_nodes)} nodes, {len(cs_hexes)} hexes  → {cs_mesh_path}")

    # ── SOFA ─────────────────────────────────────────────────────────────────
    print(f"\n[3/3] Loading SOFA result  ({sofa_npz_path.name}) ...")
    sofa_data = load_sofa_result(sofa_npz_path)
    if sofa_data is None:
        print(f"  Not found — run SOFA then re-run this script with --run-dir:")
        print(f"\n  ./sofa/run_sofa.sh sofa/dump_results.py "
              f"--config {config_path} "
              f"--mesh-npz {cs_mesh_path} "
              f"--out-dir {run_dir}")
        print(f"\n  conda run -n kgnn_mac python nff/sofa/compare_jax_sofa.py "
              f"--run-dir {run_dir}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    print("\nPlotting JAX vs SOFA comparison ...")
    n_panels = 3 if sofa_data is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 6))
    if n_panels == 2:
        axes = list(axes)

    draw_natural_panel(axes[0], fc, cnv, phys, clamped_faces, loaded_faces)
    draw_jax_panel(axes[1], equil, fc, cnv, phys, applied_moment,
                   clamped_faces, loaded_faces)
    if sofa_data is not None:
        draw_sofa_panel(axes[2], sofa_data, applied_moment)

    # Legend: show clamped / loaded markers + first few face colors
    legend_handles = []
    colors = _face_colors(n_faces)
    for f in range(min(n_faces, 8)):
        suffix = (' (clamped)' if f in clamped_faces else
                  ' (loaded)'  if f in loaded_faces  else '')
        legend_handles.append(
            mpatches.Patch(color=colors[f], label=f'F{f}{suffix}'))
    if n_faces > 8:
        legend_handles.append(mpatches.Patch(color='white', label=f'... F7–F{n_faces-1}'))

    fig.legend(handles=legend_handles, loc='lower center',
               ncol=min(n_faces, 8), fontsize=8, framealpha=0.8)

    fig.suptitle(
        f'JAX Stage-2 vs SOFA  |  {n_faces} faces  |  '
        f'Config: {config_path.name}\n'
        f'E={phys.young_modulus/1e9:.1f}GPa  '
        f'ν={phys.poisson_ratio}  '
        f't={phys.sheet_thickness*1e3:.0f}mm  '
        f'k_rot={phys.k_rot:.4f}N·m/rad',
        fontsize=9, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_png_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved → {out_png_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n── Comparison summary ──────────────────────────────────────────")
    print(f"  JAX Stage-2:  M={applied_moment:.3f} N·m  k_rot={phys.k_rot:.4f} N·m/rad")
    for f in range(n_faces):
        dx, dy, dth = equil[f]
        print(f"    F{f}: dx={dx*1e3:+.1f}mm  dy={dy*1e3:+.1f}mm  dθ={np.degrees(dth):+.1f}°")

    if sofa_data is not None:
        n_sofa = _n_faces_in_sofa(sofa_data)
        is_mm  = bool(sofa_data.get('is_moment_mode', np.array(False)))
        M_label = (f"M={float(sofa_data['applied_moment']):.3f}N·m"
                   if is_mm else
                   f"θ={float(sofa_data['rotation_angle_deg']):.0f}°")
        print(f"\n  SOFA: {M_label}  ({n_sofa} faces)")
        for fi in range(n_sofa):
            fkey = f'f{fi}_mask'
            if fkey not in sofa_data:
                continue
            dxy = sofa_face_centroid_displacement(sofa_data, fkey)
            dth = np.degrees(estimate_sofa_face_rotation(sofa_data, fkey))
            print(f"    F{fi}: dx={dxy[0]*1e3:+.1f}mm  dy={dxy[1]*1e3:+.1f}mm  dθ={dth:+.1f}°")

    print("──────────────────────────────────────────────────────────────────")
    print(f"\nRun directory: {run_dir}")
    if sofa_data is None:
        print("SOFA result missing — run:")
        print(f"  ./sofa/run_sofa.sh sofa/dump_results.py "
              f"--config {config_path} "
              f"--mesh-npz {cs_mesh_path} "
              f"--out-dir {run_dir}")
        print(f"  conda run -n kgnn_mac python nff/sofa/compare_jax_sofa.py "
              f"--run-dir {run_dir}")


if __name__ == '__main__':
    main()
