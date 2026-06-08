"""
nff/sofa/compare_jax_sofa.py
=============================
Qualitative comparison: JAX Stage-2 rigid-face model vs SOFA 3D FEM.

All outputs for one run are co-located in a single run directory
(same convention as the JAX training pipeline):
  data/outputs/runs/run_<YYYYMMDD>_<HHMMSS>_<config-name>/
      config.yaml            — copy of the driving config
      cs_mesh.npz            — CS-derived hex mesh for SOFA
      sofa_result.npz        — SOFA simulation output
      compare_jax_sofa.png   — side-by-side comparison plot

Usage
-----
  # Step 1 — create run dir, run JAX Stage-2, build CS mesh
  conda run -n kgnn_mac python nff/sofa/compare_jax_sofa.py \\
      --config data/configs/sofa/moment_1x1.yaml
  # Prints the exact SOFA command to run next.

  # Step 2 — SOFA (Homebrew Python 3.12 + SOFA env)
  ./sofa/run_sofa.sh sofa/dump_results.py \\
      --config data/configs/sofa/moment_1x1.yaml \\
      --out-dir data/outputs/runs/run_<...>_moment_1x1/

  # Step 3 — generate comparison plot (re-run with existing run dir)
  conda run -n kgnn_mac python nff/sofa/compare_jax_sofa.py \\
      --run-dir data/outputs/runs/run_<...>_moment_1x1/

  # To add SOFA output to an existing JAX training run:
  conda run -n kgnn_mac python nff/sofa/compare_jax_sofa.py \\
      --run-dir data/outputs/runs/run_<training-run>/
"""

from __future__ import annotations

import argparse
import datetime
import os
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
from matplotlib.collections import PolyCollection
import yaml

# ── Path setup ───────────────────────────────────────────────────────────────
REPO = pathlib.Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO))

from nff.topology.core import UnitPattern
from nff.topology.builder import build_tessellation
from nff.stages.state import CentroidalState
from nff.stages.physics.energy import build_potential_energy
from nff.stages.physics.statics import setup_static_solver
from nff.stages.physics.params import ReferenceGeometry, build_control_params
from nff.sofa.mesh_builder import build_mesh_from_centroidal_state
from nff.sofa.config_to_physical import physical_scale_from_config

DEFAULT_CONFIG  = REPO / 'data' / 'configs' / 'sofa' / 'moment_1x1.yaml'
RUNS_DIR        = REPO / 'data' / 'outputs' / 'runs'

FACE_COLORS = ['#4472C4', '#ED7D31', '#70AD47', '#FFC000']


def _make_run_dir(config_path: pathlib.Path) -> pathlib.Path:
    """Create a timestamped run directory under data/outputs/runs/."""
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = RUNS_DIR / f"run_{timestamp}_{config_path.stem}"
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path, run_dir / 'config.yaml')
    return run_dir


def _resolve_run_dir_and_config(args) -> tuple[pathlib.Path, pathlib.Path, dict]:
    """Return (run_dir, config_path, raw_dict).

    If --run-dir is given, read config.yaml from it.
    If --config is given (no --run-dir), create a fresh run dir.
    """
    if args.run_dir is not None:
        run_dir     = pathlib.Path(args.run_dir).resolve()
        config_path = run_dir / 'config.yaml'
        if not config_path.exists():
            # fall back to explicit --config if provided
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
# JAX Stage-2 solver
# ─────────────────────────────────────────────────────────────────────────────

def run_jax_stage2(raw: dict, phys):
    """Build deployed 1×1 tessellation from config, apply moment, solve Stage-2.

    Uses physical k_rot (N·m/rad) and physical moment (N·m) so that JAX
    displacements (in JAX units × jax_scale) are directly comparable to SOFA.

    Deployed state: ARM_WIDTH gap shifts per face → non-zero hinge bond vectors.
      F0 vertices [0-3]:   shift ( 0,         0        )
      F1 vertices [4-7]:   shift (+ARM_WIDTH,  0        )
      F2 vertices [8-11]:  shift (+ARM_WIDTH, +ARM_WIDTH)
      F3 vertices [12-15]: shift ( 0,         +ARM_WIDTH)
    """
    patterns_path = REPO / 'data' / 'library' / 'patterns.yaml'
    with open(patterns_path) as f:
        lib = yaml.safe_load(f)

    tess_cfg = raw.get('tessellation', {})
    p_cfg    = lib[tess_cfg.get('pattern', 'unit_RDQK_0')]

    # Scale to physical units then deploy by shifting per-face vertex groups
    verts = np.array(p_cfg['vertices'], dtype=float) * phys.face_size
    face_shifts = {
        0: np.array([0.0,             0.0            ]),
        1: np.array([phys.arm_width,  0.0            ]),
        2: np.array([phys.arm_width,  phys.arm_width ]),
        3: np.array([0.0,             phys.arm_width ]),
    }
    face_vertex_ranges = {0: range(0,4), 1: range(4,8), 2: range(8,12), 3: range(12,16)}
    for fi, vi_range in face_vertex_ranges.items():
        verts[list(vi_range)] += face_shifts[fi]

    pattern = UnitPattern(
        vertices        = verts,
        faces           = p_cfg['faces'],
        internal_hinges = p_cfg['internal_hinges'],
        external_hinges = p_cfg.get('external_hinges', []),
    )

    tess = build_tessellation(pattern, nx=1, ny=1)
    tess.set_hinge_properties(
        k_stretch = phys.k_stretch,
        k_shear   = phys.k_shear,
        k_rot     = phys.k_rot,        # physical N·m/rad
    )
    tess.set_all_faces_properties(density=1.0)

    # BCs and loads from config
    bc_raw    = raw.get('boundary_conditions', {})
    loads_raw = raw.get('loads', [])

    for face_idx in bc_raw.get('clamped_faces', []):
        tess.set_face_dofs(int(face_idx), [0, 1, 2])

    for load in loads_raw:
        tess.set_face_load(int(load['face']), int(load['dof']), float(load['value']))

    cs       = CentroidalState.from_tessellation(tess)
    geometry = ReferenceGeometry.from_centroidal_state(cs)

    print(f"  n_faces={geometry.n_faces}, n_hinges={geometry.bond_connectivity.shape[0]}")
    print(f"  Bond vectors (should be ±{phys.arm_width*1e3:.0f}mm):")
    for i, bv in enumerate(np.array(geometry.reference_bond_vectors)):
        print(f"    H{i}: [{bv[0]*1e3:.1f}, {bv[1]*1e3:.1f}] mm")

    energy_fn = build_potential_energy(
        bond_connectivity  = geometry.bond_connectivity,
        linearized_strains = True,
        use_contact        = False,
    )

    physics_cfg = raw.get('physics', {})
    solve_statics = setup_static_solver(
        geometry                   = geometry,
        energy_fn                  = energy_fn,
        loaded_face_DOF_pairs      = cs.loaded_face_DOF_pairs,
        loading_fn                 = cs.get_loading_function(),
        constrained_face_DOF_pairs = cs.constrained_face_DOF_pairs,
        incremental                = bool(physics_cfg.get('incremental', True)),
        num_steps                  = int(physics_cfg.get('num_load_steps', 10)),
        solver_maxiter             = int(physics_cfg.get('solver_maxiter', 2000)),
        solver_tol                 = float(physics_cfg.get('solver_tol', 1e-6)),
    )

    control_params = build_control_params(
        geometry    = geometry,
        k_stretch   = cs.k_stretch,
        k_shear     = cs.k_shear,
        k_rot       = cs.k_rot,
        density     = cs.density,
        use_contact = False,
    )

    applied_moment = float(loads_raw[0]['value']) if loads_raw else 1.0
    print(f"  Solving JAX Stage-2 (M={applied_moment:.3f} N·m, "
          f"k_rot={phys.k_rot:.3f} N·m/rad) ...")
    initial_disp = jnp.zeros((geometry.n_faces, 3))
    solution     = solve_statics(initial_disp, control_params)
    equil        = np.array(solution.fields[-1])   # (n_faces, 3) in JAX units

    fc  = np.array(cs.face_centroids)
    cnv = np.array(cs.centroid_node_vectors)

    # fc and cnv are in metres (physical coords); equil displacements are also metres.
    print(f"  Equilibrium (face_size={phys.face_size*1e3:.0f}mm):")
    for f in range(geometry.n_faces):
        dx, dy, dth = equil[f]
        print(f"    Face {f}: dx={dx*1e3:+.1f} mm  dy={dy*1e3:+.1f} mm  dθ={np.degrees(dth):+.1f}°")

    return equil, fc, cnv, cs, phys


# ─────────────────────────────────────────────────────────────────────────────
# SOFA result loader
# ─────────────────────────────────────────────────────────────────────────────

def load_sofa_result(npz_path: pathlib.Path):
    if not npz_path.exists():
        return None
    data  = np.load(npz_path)
    is_mm = bool(data.get('is_moment_mode', np.array(False)))
    if is_mm:
        print(f"  SOFA: {len(data['nodes_nat'])} nodes  "
              f"moment={float(data['applied_moment']):.3f} N·m  (moment mode)")
    else:
        print(f"  SOFA: {len(data['nodes_nat'])} nodes  "
              f"angle={float(data['rotation_angle_deg']):.0f}°  (rotation mode)")
    print(f"  Strain energy        : {float(data['strain_energy']):.4e} J")
    print(f"  Max in-plane XY disp : {float(data['max_xy_displacement'])*1e3:.2f} mm")
    print(f"  Max |z| buckling     : {float(data['max_z_displacement'])*1e3:.2f} mm")
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Face rotation estimate from SOFA node positions (SVD best-fit)
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
    """Mean XY centroid displacement for a face (top layer only)."""
    mask   = data[face_key].astype(bool)
    z_nat  = data['nodes_nat'][mask, 2]
    top    = z_nat > (z_nat.max() - 1e-9)
    nat_xy = data['nodes_nat'][mask][top, :2].mean(axis=0)
    cur_xy = data['nodes_cur'][mask][top, :2].mean(axis=0)
    return cur_xy - nat_xy


# ─────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────────────────────────────────────

def jax_face_polygon(face_idx, fc, cnv, equil, phys) -> np.ndarray:
    """Deformed face polygon (4,2) in mm. fc/cnv/equil are in metres."""
    dx, dy, dth = equil[face_idx]
    c, s = np.cos(dth), np.sin(dth)
    R = np.array([[c, -s], [s, c]])
    new_centroid_mm = (fc[face_idx] + np.array([dx, dy])) * 1e3
    rotated_cnv_mm  = (R @ cnv[face_idx, :4].T).T * 1e3
    return new_centroid_mm + rotated_cnv_mm


def _jax_nat_poly_mm(face_idx, fc, cnv, phys) -> np.ndarray:
    """Natural-state face polygon in mm. fc/cnv are in metres."""
    return (fc[face_idx] + cnv[face_idx, :4]) * 1e3


def _set_ax_limits(ax, polygons, pad_frac=0.15):
    all_v = np.vstack(polygons)
    span  = max(all_v[:, 0].max() - all_v[:, 0].min(),
                all_v[:, 1].max() - all_v[:, 1].min())
    pad   = span * pad_frac
    ax.set_xlim(all_v[:, 0].min() - pad, all_v[:, 0].max() + pad)
    ax.set_ylim(all_v[:, 1].min() - pad, all_v[:, 1].max() + pad)


def draw_natural_panel(ax, fc, cnv, phys):
    ax.set_aspect('equal')
    ax.set_title('Deployed natural state\n(Stage-1 output, voids open)', fontsize=10)
    polys = []
    for f in range(4):
        poly = _jax_nat_poly_mm(f, fc, cnv, phys)
        polys.append(poly)
        ax.add_patch(plt.Polygon(poly, closed=True, zorder=2,
                                 edgecolor='k', linewidth=1.0,
                                 facecolor=FACE_COLORS[f], alpha=0.75))
        cx, cy = poly.mean(axis=0)
        ax.text(cx, cy, f'F{f}', ha='center', va='center', fontsize=9, fontweight='bold')
    _set_ax_limits(ax, polys)
    ax.set_xlabel(f'arm_width={phys.arm_width*1e3:.0f}mm  '
                  f'face_size={phys.face_size*1e3:.0f}mm  '
                  f'fold_length={phys.fold_length*1e3:.0f}mm', fontsize=8)
    ax.set_ylabel('mm', fontsize=8)


def draw_jax_panel(ax, equil, fc, cnv, phys, applied_moment):
    ax.set_aspect('equal')
    ax.set_title(f'JAX Stage-2\nM={applied_moment:.2f} N·m  '
                 f'k_rot={phys.k_rot:.3f} N·m/rad', fontsize=10)
    nat_polys = [_jax_nat_poly_mm(f, fc, cnv, phys) for f in range(4)]
    def_polys  = [jax_face_polygon(f, fc, cnv, equil, phys) for f in range(4)]

    for f in range(4):
        ax.add_patch(plt.Polygon(nat_polys[f], closed=True,
                                 linewidth=0.6, linestyle='--',
                                 edgecolor='gray', facecolor='none', zorder=1))
        ax.add_patch(plt.Polygon(def_polys[f], closed=True, zorder=2,
                                 edgecolor='k', linewidth=1.0,
                                 facecolor=FACE_COLORS[f], alpha=0.75))
        cx, cy = def_polys[f].mean(axis=0)
        ax.text(cx, cy, f'F{f}', ha='center', va='center', fontsize=9, fontweight='bold')

    _set_ax_limits(ax, nat_polys + def_polys)

    rot_strs = '  '.join(
        f'F{f}:{np.degrees(equil[f,2]):+.1f}°' for f in range(4))
    ax.set_xlabel(rot_strs, fontsize=8)
    ax.set_ylabel('mm', fontsize=8)


def draw_sofa_panel(ax, data, applied_moment):
    ax.set_aspect('equal')
    is_mm = bool(data.get('is_moment_mode', np.array(False)))
    if is_mm:
        ax.set_title(f'SOFA 3D FEM\nM={applied_moment:.2f} N·m  '
                     f'E=3.5 GPa  t=1 mm', fontsize=10)
    else:
        ax.set_title(f'SOFA 3D FEM\nθ₁={float(data["rotation_angle_deg"]):.0f}° (rotation)',
                     fontsize=10)

    nodes_cur_mm = data['nodes_cur'] * 1e3   # m → mm
    hexes        = data['hexes']
    top_quads    = hexes[:, [4, 5, 6, 7]]
    quad_xy      = nodes_cur_mm[:, :2][top_quads]

    face_keys   = ['f0_mask', 'f1_mask', 'f2_mask', 'f3_mask']
    quad_colors = np.full(len(hexes), '#cccccc')
    for fi, fkey in enumerate(face_keys):
        mask = data[fkey].astype(bool)
        for qi, quad in enumerate(top_quads):
            if mask[quad].any():
                quad_colors[qi] = FACE_COLORS[fi]

    coll = PolyCollection(quad_xy, facecolors=quad_colors, edgecolors='k',
                          linewidths=0.3, zorder=2)
    ax.add_collection(coll)

    xy_mm = nodes_cur_mm[:, :2]
    pad   = float(data['face_size']) * 1e3 * 0.15
    ax.set_xlim(xy_mm[:, 0].min() - pad, xy_mm[:, 0].max() + pad)
    ax.set_ylim(xy_mm[:, 1].min() - pad, xy_mm[:, 1].max() + pad)

    rot_strs = '  '.join(
        f'F{fi}:{np.degrees(estimate_sofa_face_rotation(data, fkey)):+.1f}°'
        for fi, fkey in enumerate(face_keys)
    )
    ax.set_xlabel(rot_strs, fontsize=8)
    ax.set_ylabel('mm', fontsize=8)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',  default=None,
                        help='Path to sofa YAML config. Creates a new run dir.')
    parser.add_argument('--run-dir', default=None,
                        help='Existing run directory. Reads config.yaml from it; '
                             'also used to find sofa_result.npz and write the plot.')
    args = parser.parse_args()

    run_dir, config_path, raw = _resolve_run_dir_and_config(args)

    phys           = physical_scale_from_config(raw)
    applied_moment = float(raw.get('loads', [{}])[0].get('value', 1.0))

    sofa_npz_path = run_dir / 'sofa_result.npz'
    cs_mesh_path  = run_dir / 'cs_mesh.npz'
    out_png_path  = run_dir / 'compare_jax_sofa.png'

    print("=" * 60)
    print(f"JAX Stage-2 vs SOFA — config: {config_path.name}")
    print(f"  Run dir: {run_dir}")
    print(f"  face_size={phys.face_size*1e3:.0f}mm  "
          f"arm_width={phys.arm_width*1e3:.0f}mm  "
          f"fold_length={phys.fold_length*1e3:.0f}mm")
    print(f"  E={phys.young_modulus/1e9:.1f}GPa  "
          f"ν={phys.poisson_ratio}  "
          f"k_rot={phys.k_rot:.3f}N·m/rad")
    print("=" * 60)

    # ── JAX ──────────────────────────────────────────────────────────────────
    print(f"\n[1/3] Running JAX Stage-2  (M={applied_moment:.3f} N·m) ...")
    equil, fc, cnv, cs, phys = run_jax_stage2(raw, phys)

    # ── Build CS mesh ─────────────────────────────────────────────────────────
    print(f"\n[1b] Building SOFA mesh from CentroidalState ...")
    cs_nodes, cs_hexes, cs_bc = build_mesh_from_centroidal_state(
        cs,
        fold_length     = phys.fold_length,
        sheet_thickness = phys.sheet_thickness,
    )
    np.savez(
        cs_mesh_path,
        nodes        = cs_nodes,
        hexes        = cs_hexes,
        f0_mask      = cs_bc['f0'],
        f1_mask      = cs_bc['f1'],
        f2_mask      = cs_bc['f2'],
        f3_mask      = cs_bc['f3'],
        clamped_mask = cs_bc['clamped'],
        loaded_mask  = cs_bc['loaded'],
        face_size    = np.float64(phys.face_size),
    )
    print(f"  Mesh: {len(cs_nodes)} nodes, {len(cs_hexes)} hexes  → {cs_mesh_path}")

    # ── SOFA ─────────────────────────────────────────────────────────────────
    print(f"\n[2/3] Loading SOFA result  ({sofa_npz_path.name}) ...")
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
    print("\n[3/3] Plotting ...")
    n_panels = 3 if sofa_data is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5.5))
    if n_panels == 2:
        axes = list(axes)

    draw_natural_panel(axes[0], fc, cnv, phys)
    draw_jax_panel(axes[1], equil, fc, cnv, phys, applied_moment)
    if sofa_data is not None:
        draw_sofa_panel(axes[2], sofa_data, applied_moment)

    legend_patches = [
        mpatches.Patch(color=FACE_COLORS[f],
                       label=f'Face {f}' + (' (clamped)' if f == 0 else ''))
        for f in range(4)
    ]
    fig.legend(handles=legend_patches, loc='lower center', ncol=4,
               fontsize=9, framealpha=0.8)

    fig.suptitle(
        f'Same experiment: CCW moment on F1  |  F0 clamped  |  deployed starting state\n'
        f'Config: {config_path.name}  |  '
        f'E={phys.young_modulus/1e9:.1f}GPa  '
        f'ν={phys.poisson_ratio}  '
        f't={phys.sheet_thickness*1e3:.0f}mm  '
        f'k_rot={phys.k_rot:.3f}N·m/rad',
        fontsize=9, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_png_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved → {out_png_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n── Comparison summary ──────────────────────────────────────────")
    print(f"  JAX:  M={applied_moment:.3f} N·m  k_rot={phys.k_rot:.3f} N·m/rad")
    for f in range(4):
        dx, dy, dth = equil[f]
        print(f"    F{f}: dx={dx*1e3:+.1f}mm  dy={dy*1e3:+.1f}mm  dθ={np.degrees(dth):+.1f}°")

    if sofa_data is not None:
        is_mm = bool(sofa_data.get('is_moment_mode', np.array(False)))
        M_label = (f"M={float(sofa_data['applied_moment']):.3f}N·m"
                   if is_mm else
                   f"θ₁={float(sofa_data['rotation_angle_deg']):.0f}°")
        print(f"\n  SOFA: {M_label}")
        for fi, fkey in enumerate(['f0_mask', 'f1_mask', 'f2_mask', 'f3_mask']):
            dxy = sofa_face_centroid_displacement(sofa_data, fkey)
            dth = np.degrees(estimate_sofa_face_rotation(sofa_data, fkey))
            print(f"    F{fi}: dx={dxy[0]*1e3:+.1f}mm  dy={dxy[1]*1e3:+.1f}mm  dθ={dth:+.1f}°")

    print("──────────────────────────────────────────────────────────────────")
    print(f"\nRun directory: {run_dir}")
    if sofa_data is not None:
        print(f"All outputs saved to run dir.")
    else:
        print(f"SOFA result missing — run:")
        print(f"  ./sofa/run_sofa.sh sofa/dump_results.py "
              f"--config {config_path} "
              f"--mesh-npz {cs_mesh_path} "
              f"--out-dir {run_dir}")
        print(f"  conda run -n kgnn_mac python nff/sofa/compare_jax_sofa.py "
              f"--run-dir {run_dir}")


if __name__ == '__main__':
    main()
