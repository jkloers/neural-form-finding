"""
nff/sofa/compare_jax_sofa.py
=============================
Qualitative comparison: JAX Stage-2 rigid-face model vs SOFA 3D FEM.

Same physical experiment in both solvers
-----------------------------------------
1×1 unit_RDQK_0 cell, deployed starting state (arm_width=10mm voids open).
  Face 0 : clamped (all DOFs fixed).
  Face 1 : CCW moment M applied — free to find its own equilibrium.
  Faces 2, 3 : free.

Both solvers use FORCE CONTROL (applied moment), not displacement control.
The moment magnitudes are not yet calibrated to each other (Phase 3 task);
this comparison is qualitative — checking the kinematic response pattern.

SOFA moment mode applies tangential forces to face-1 nodes about the face-1
centroid → pure torque, no net force — identical in kind to JAX's DOF-2
generalised force (pure moment on the face, no imposed translation).

Deployed starting state
-----------------------
Both start from the DEPLOYED configuration (stage-1 output): faces separated
by arm_width gaps so hinge bond vectors are non-zero (≈ arm_width in length).
JAX builds this by shifting per-face vertex groups before building the
CentroidalState. SOFA builds it via build_unified_mesh (same gap geometry).

Usage
-----
  # Step 1 — SOFA run (in SOFA environment)
  ./sofa/run_sofa.sh sofa/dump_results.py \\
      --mode moment --moment 1.0 \\
      --out sofa/output/sofa_moment.npz

  # Step 2 — JAX + comparison plot (kgnn_mac conda env)
  conda run -n kgnn_mac python nff/sofa/compare_jax_sofa.py

Output
------
  nff/sofa/output/compare_jax_sofa.png
"""

from __future__ import annotations

import os
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
sys.path.insert(0, str(REPO / 'sofa'))

from nff.topology.core import UnitPattern
from nff.topology.builder import build_tessellation
from nff.stages.state import CentroidalState
from nff.stages.physics.energy import build_potential_energy
from nff.stages.physics.statics import setup_static_solver
from nff.stages.physics.params import ReferenceGeometry, build_control_params
from nff.sofa.mesh_builder import build_mesh_from_centroidal_state

# ── Physical constants (match simulate_cell.py) ───────────────────────────────
FACE_SIZE       = 0.100   # m — 100 mm
ARM_WIDTH       = 0.010   # m — 10 mm
FOLD_LENGTH     = 0.003   # m — 3 mm
SHEET_THICKNESS = 0.001   # m — 1 mm

# ── JAX material params (normalised) ─────────────────────────────────────────
K_STRETCH = 1000.0
K_SHEAR   = 1000.0
K_ROT     = 0.5

# Applied moment on face 1. JAX units are not yet calibrated to N·m (Phase 3).
# Current value gives face-1 rotation of ≈20°.
JAX_MOMENT  = 0.6

# SOFA moment in N·m — physically meaningful (3D FEM, PLA E=3.5 GPa).
# Rough estimate: k_rot_sofa ≈ 0.79 N·m/rad per hinge → for ~20° response,
# effective system moment ≈ 1.0 N·m.  Adjust after seeing results.
SOFA_MOMENT = 1.0

SOFA_NPZ   = REPO / 'sofa' / 'output' / 'sofa_moment.npz'
CS_MESH    = REPO / 'sofa' / 'output' / 'cs_mesh.npz'
OUT_PNG    = pathlib.Path(__file__).parent / 'output' / 'compare_jax_sofa.png'

FACE_COLORS = ['#4472C4', '#ED7D31', '#70AD47', '#FFC000']


# ─────────────────────────────────────────────────────────────────────────────
# JAX Stage-2 solver
# ─────────────────────────────────────────────────────────────────────────────

def run_jax_stage2(moment: float = JAX_MOMENT):
    """Build deployed 1×1 tessellation, apply CCW moment on face 1, solve Stage-2.

    Deployed state: ARM_WIDTH gap shifts per face → non-zero hinge bond vectors.
      F0 vertices [0-3]:   shift ( 0,         0        )
      F1 vertices [4-7]:   shift (+ARM_WIDTH,  0        )
      F2 vertices [8-11]:  shift (+ARM_WIDTH, +ARM_WIDTH)
      F3 vertices [12-15]: shift ( 0,         +ARM_WIDTH)
    """
    patterns_path = REPO / 'data' / 'library' / 'patterns.yaml'
    with open(patterns_path) as f:
        lib = yaml.safe_load(f)
    p_cfg = lib['unit_RDQK_0']

    # Scale to physical units then deploy by shifting per-face vertex groups
    verts = np.array(p_cfg['vertices'], dtype=float) * FACE_SIZE
    face_shifts = {
        0: np.array([0.0,       0.0      ]),
        1: np.array([ARM_WIDTH, 0.0      ]),
        2: np.array([ARM_WIDTH, ARM_WIDTH]),
        3: np.array([0.0,       ARM_WIDTH]),
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
    tess.set_hinge_properties(k_stretch=K_STRETCH, k_shear=K_SHEAR, k_rot=K_ROT)
    tess.set_all_faces_properties(density=1.0)

    # BCs: clamp face 0, pure CCW moment on face 1 DOF 2
    tess.set_face_dofs(0, [0, 1, 2])
    tess.set_face_load(1, 2, moment)

    cs       = CentroidalState.from_tessellation(tess)
    geometry = ReferenceGeometry.from_centroidal_state(cs)

    print(f"  n_faces={geometry.n_faces}, n_hinges={geometry.bond_connectivity.shape[0]}")
    print(f"  Bond vectors (should be ±{ARM_WIDTH*1e3:.0f}mm):")
    for i, bv in enumerate(np.array(geometry.reference_bond_vectors)):
        print(f"    H{i}: [{bv[0]*1e3:.1f}, {bv[1]*1e3:.1f}] mm")

    energy_fn = build_potential_energy(
        bond_connectivity  = geometry.bond_connectivity,
        linearized_strains = True,
        use_contact        = False,
    )

    solve_statics = setup_static_solver(
        geometry                   = geometry,
        energy_fn                  = energy_fn,
        loaded_face_DOF_pairs      = cs.loaded_face_DOF_pairs,
        loading_fn                 = cs.get_loading_function(),
        constrained_face_DOF_pairs = cs.constrained_face_DOF_pairs,
        incremental                = True,
        num_steps                  = 10,
        solver_maxiter             = 2000,
        solver_tol                 = 1e-6,
    )

    control_params = build_control_params(
        geometry    = geometry,
        k_stretch   = cs.k_stretch,
        k_shear     = cs.k_shear,
        k_rot       = cs.k_rot,
        density     = cs.density,
        use_contact = False,
    )

    print(f"  Solving JAX Stage-2 (M={moment}) ...")
    initial_disp = jnp.zeros((geometry.n_faces, 3))
    solution     = solve_statics(initial_disp, control_params)
    equil        = np.array(solution.fields[-1])   # (n_faces, 3)

    fc  = np.array(cs.face_centroids)
    cnv = np.array(cs.centroid_node_vectors)

    print(f"  Equilibrium (deployed reference + delta):")
    for f in range(geometry.n_faces):
        dx, dy, dth = equil[f]
        print(f"    Face {f}: dx={dx*1e3:+.1f} mm  dy={dy*1e3:+.1f} mm  dθ={np.degrees(dth):+.1f}°")

    return equil, fc, cnv, cs


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
              f"moment={float(data['applied_moment']):.2f} N·m  (moment mode)")
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

def jax_face_polygon(face_idx, fc, cnv, equil) -> np.ndarray:
    """Deformed face polygon (4,2) from JAX equilibrium displacement."""
    dx, dy, dth = equil[face_idx]
    c, s = np.cos(dth), np.sin(dth)
    R = np.array([[c, -s], [s, c]])
    new_centroid = fc[face_idx] + np.array([dx, dy])
    return new_centroid + (R @ cnv[face_idx, :4].T).T   # (4, 2)


def _set_ax_limits(ax, polygons, pad_frac=0.15):
    all_v = np.vstack(polygons)
    span  = max(all_v[:, 0].max() - all_v[:, 0].min(),
                all_v[:, 1].max() - all_v[:, 1].min())
    pad   = span * pad_frac
    ax.set_xlim(all_v[:, 0].min() - pad, all_v[:, 0].max() + pad)
    ax.set_ylim(all_v[:, 1].min() - pad, all_v[:, 1].max() + pad)


def draw_natural_panel(ax, fc, cnv):
    ax.set_aspect('equal')
    ax.set_title('Deployed natural state\n(Stage-1 output, voids open)', fontsize=10)
    polys = []
    for f in range(4):
        poly = fc[f] + cnv[f, :4]
        polys.append(poly)
        ax.add_patch(plt.Polygon(poly, closed=True, zorder=2,
                                 edgecolor='k', linewidth=1.0,
                                 facecolor=FACE_COLORS[f], alpha=0.75))
        cx, cy = poly.mean(axis=0)
        ax.text(cx, cy, f'F{f}', ha='center', va='center', fontsize=9, fontweight='bold')
    _set_ax_limits(ax, polys)
    ax.set_xlabel(f'arm_width = {ARM_WIDTH*1e3:.0f} mm  |  face_size = {FACE_SIZE*1e3:.0f} mm',
                  fontsize=8)


def draw_jax_panel(ax, equil, fc, cnv, moment):
    ax.set_aspect('equal')
    ax.set_title(f'JAX Stage-2\nM={moment:.2f} (normalised), k_rot={K_ROT}', fontsize=10)
    nat_polys = [fc[f] + cnv[f, :4] for f in range(4)]
    def_polys  = [jax_face_polygon(f, fc, cnv, equil) for f in range(4)]

    for f in range(4):
        # natural outline (ghost)
        ax.add_patch(plt.Polygon(nat_polys[f], closed=True,
                                 linewidth=0.6, linestyle='--',
                                 edgecolor='gray', facecolor='none', zorder=1))
        # deformed fill
        ax.add_patch(plt.Polygon(def_polys[f], closed=True, zorder=2,
                                 edgecolor='k', linewidth=1.0,
                                 facecolor=FACE_COLORS[f], alpha=0.75))
        cx, cy = def_polys[f].mean(axis=0)
        ax.text(cx, cy, f'F{f}', ha='center', va='center', fontsize=9, fontweight='bold')

    _set_ax_limits(ax, nat_polys + def_polys)

    rot_strs = '  '.join(f'F{f}:{np.degrees(equil[f,2]):+.1f}°' for f in range(4))
    ax.set_xlabel(rot_strs, fontsize=8)


def draw_sofa_panel(ax, data, moment):
    ax.set_aspect('equal')
    is_mm = bool(data.get('is_moment_mode', np.array(False)))
    if is_mm:
        ax.set_title(f'SOFA 3D FEM\nM={moment:.2f} N·m (moment mode)', fontsize=10)
    else:
        ax.set_title(f'SOFA 3D FEM\nθ₁={float(data["rotation_angle_deg"]):.0f}° (rotation mode)',
                     fontsize=10)

    nodes_cur = data['nodes_cur']
    hexes     = data['hexes']
    top_quads = hexes[:, [4, 5, 6, 7]]
    quad_xy   = nodes_cur[:, :2][top_quads]

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

    xy  = nodes_cur[:, :2]
    pad = float(data['face_size']) * 0.15
    ax.set_xlim(xy[:, 0].min() - pad, xy[:, 0].max() + pad)
    ax.set_ylim(xy[:, 1].min() - pad, xy[:, 1].max() + pad)

    rot_strs = '  '.join(
        f'F{fi}:{np.degrees(estimate_sofa_face_rotation(data, fkey)):+.1f}°'
        for fi, fkey in enumerate(face_keys)
    )
    ax.set_xlabel(rot_strs, fontsize=8)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("JAX Stage-2 vs SOFA — same experiment: moment on face 1")
    print("=" * 60)

    # ── JAX ──────────────────────────────────────────────────────────────────
    print(f"\n[1/3] Running JAX Stage-2  (M={JAX_MOMENT}) ...")
    equil, fc, cnv, cs = run_jax_stage2(moment=JAX_MOMENT)

    # ── Build CS mesh and save (for use by SOFA's --mesh-npz flag) ───────────
    print(f"\n[1b] Building SOFA mesh from CentroidalState ...")
    cs_nodes, cs_hexes, cs_bc = build_mesh_from_centroidal_state(
        cs,
        fold_length     = FOLD_LENGTH,
        sheet_thickness = SHEET_THICKNESS,
    )
    CS_MESH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        CS_MESH,
        nodes       = cs_nodes,
        hexes       = cs_hexes,
        f0_mask     = cs_bc['f0'],
        f1_mask     = cs_bc['f1'],
        f2_mask     = cs_bc['f2'],
        f3_mask     = cs_bc['f3'],
        clamped_mask= cs_bc['clamped'],
        loaded_mask = cs_bc['loaded'],
        face_size   = np.float64(FACE_SIZE),
    )
    print(f"  Mesh from CS: {len(cs_nodes)} nodes, {len(cs_hexes)} hexes")
    print(f"  Saved → {CS_MESH}")

    # ── SOFA ─────────────────────────────────────────────────────────────────
    print(f"\n[2/3] Loading SOFA result  ({SOFA_NPZ.name}) ...")
    sofa_data = load_sofa_result(SOFA_NPZ)
    if sofa_data is None:
        print(f"  WARNING: not found.  Run with the CS-derived mesh:")
        print(f"  ./sofa/run_sofa.sh sofa/dump_results.py "
              f"--mesh-npz sofa/output/cs_mesh.npz "
              f"--mode moment --moment {SOFA_MOMENT} "
              f"--out sofa/output/sofa_moment.npz")

    # ── Plot ─────────────────────────────────────────────────────────────────
    print("\n[3/3] Plotting ...")
    n_panels = 3 if sofa_data is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5.5))
    if n_panels == 2:
        axes = list(axes)

    draw_natural_panel(axes[0], fc, cnv)
    draw_jax_panel(axes[1], equil, fc, cnv, JAX_MOMENT)
    if sofa_data is not None:
        draw_sofa_panel(axes[2], sofa_data, SOFA_MOMENT)

    legend_patches = [
        mpatches.Patch(color=FACE_COLORS[f],
                       label=f'Face {f}' + (' (clamped)' if f == 0 else ''))
        for f in range(4)
    ]
    fig.legend(handles=legend_patches, loc='lower center', ncol=4,
               fontsize=9, framealpha=0.8)

    fig.suptitle(
        'Same experiment: CCW moment on face 1  |  face 0 clamped  |  deployed starting state\n'
        f'JAX: k_stretch={K_STRETCH:.0f}, k_shear={K_SHEAR:.0f}, k_rot={K_ROT}  |  '
        f'SOFA: E=3.5 GPa, ν=0.36, t=1 mm  (units not yet calibrated)',
        fontsize=9, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150, bbox_inches='tight')
    print(f"\nSaved → {OUT_PNG}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n── Comparison summary ──────────────────────────────────────────")
    print(f"  JAX:  M = {JAX_MOMENT:.2f}  (normalised, k_rot={K_ROT})")
    for f in range(4):
        dx, dy, dth = equil[f]
        print(f"    F{f}: dx={dx*1e3:+.1f} mm  dy={dy*1e3:+.1f} mm  dθ={np.degrees(dth):+.1f}°")

    if sofa_data is not None:
        is_mm = bool(sofa_data.get('is_moment_mode', np.array(False)))
        M_label = (f"M={float(sofa_data['applied_moment']):.2f} N·m  (moment mode)"
                   if is_mm else
                   f"θ₁={float(sofa_data['rotation_angle_deg']):.0f}°  (rotation mode)")
        print(f"\n  SOFA: {M_label}")
        for fi, fkey in enumerate(['f0_mask', 'f1_mask', 'f2_mask', 'f3_mask']):
            dxy = sofa_face_centroid_displacement(sofa_data, fkey)
            dth = np.degrees(estimate_sofa_face_rotation(sofa_data, fkey))
            print(f"    F{fi}: dx={dxy[0]*1e3:+.1f} mm  dy={dxy[1]*1e3:+.1f} mm  dθ={dth:+.1f}°")
    print("──────────────────────────────────────────────────────────────────")
    print("\nNext step: run SOFA with the CS-derived mesh, then re-run this script.")
    print(f"  ./sofa/run_sofa.sh sofa/dump_results.py "
          f"--mesh-npz sofa/output/cs_mesh.npz "
          f"--mode moment --moment {SOFA_MOMENT} "
          f"--out sofa/output/sofa_moment.npz")


if __name__ == '__main__':
    main()
