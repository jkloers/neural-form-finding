"""
sofa/dump_results.py — Run the SOFA simulation and save results to .npz.

NO matplotlib import — avoids the SOFA+Qt/Cocoa crash on macOS.
Results are loaded by sofa/visualize.py in the kgnn_mac conda env.

Usage (via run_sofa.sh):
    # Config-driven (recommended — single source of truth):
    ./sofa/run_sofa.sh sofa/dump_results.py \\
        --config data/configs/sofa/moment_1x1.yaml

    # Config + pre-built CS mesh:
    ./sofa/run_sofa.sh sofa/dump_results.py \\
        --config data/configs/sofa/moment_1x1.yaml \\
        --mesh-npz sofa/output/cs_mesh.npz

    # Legacy — explicit geometry/material flags:
    ./sofa/run_sofa.sh sofa/dump_results.py --arm-width 0.010 --fold-length 0.020 --angle 90
    ./sofa/run_sofa.sh sofa/dump_results.py --out /tmp/sofa_result.npz
    ./sofa/run_sofa.sh sofa/dump_results.py \\
        --mesh-npz sofa/output/cs_mesh.npz --mode moment --moment 1.0
"""

import sys
import os
import argparse
import math
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, REPO_ROOT)

try:
    import Sofa
    import Sofa.Core
    import Sofa.Simulation
except ImportError as e:
    sys.exit(f"Cannot import SOFA: {e}\nRun via ./sofa/run_sofa.sh")

from simulate_cell import (
    evaluate_unit_cell,
    FACE_SIZE, SHEET_THICKNESS, YOUNG_MODULUS, POISSON_RATIO, YIELD_STRENGTH,
)
from materials import vm_stress_per_hex
from nff.sofa.config_to_physical import physical_scale_from_config

DEFAULT_OUT = os.path.join(os.path.dirname(__file__), 'output', 'sofa_result.npz')


def run_and_dump(
    hinge_arm_width      = 0.010,
    hinge_fold_length    = 0.003,
    rotation_angle_deg   = -45.0,
    applied_moment       = 0.0,
    loading_mode         = 'rotation',
    face_size            = FACE_SIZE,
    sheet_thickness      = SHEET_THICKNESS,
    young_modulus        = YOUNG_MODULUS,
    poisson_ratio        = POISSON_RATIO,
    yield_strength       = YIELD_STRENGTH,
    out_path             = DEFAULT_OUT,
    mesh_npz             = None,
):
    # Pre-built mesh from CentroidalState (correct mesh builder).
    mesh_data = None
    if mesh_npz is not None:
        print(f"  Loading pre-built mesh from {mesh_npz} ...")
        d = np.load(mesh_npz)
        nodes = d['nodes']
        hexes = d['hexes']
        n = len(nodes)
        bc_masks = {
            'f0': d['f0_mask'].astype(bool), 'f1': d['f1_mask'].astype(bool),
            'f2': d['f2_mask'].astype(bool), 'f3': d['f3_mask'].astype(bool),
            'face_0': d['f0_mask'].astype(bool), 'face_1': d['f1_mask'].astype(bool),
            'face_2': d['f2_mask'].astype(bool), 'face_3': d['f3_mask'].astype(bool),
            'clamped': d.get('clamped_mask', d['f0_mask']).astype(bool),
            'loaded':  d.get('loaded_mask',  d['f1_mask']).astype(bool),
        }
        mesh_data = (nodes, hexes, bc_masks)
        face_size = float(d.get('face_size', face_size))
        print(f"  Mesh: {len(nodes)} nodes, {len(hexes)} hexes")

    print("  Running SOFA simulation ...")
    r = evaluate_unit_cell(
        hinge_arm_width    = hinge_arm_width,
        hinge_fold_length  = hinge_fold_length,
        rotation_angle_deg = rotation_angle_deg,
        applied_moment     = applied_moment,
        loading_mode       = loading_mode,
        face_size          = face_size,
        sheet_thickness    = sheet_thickness,
        young_modulus      = young_modulus,
        poisson_ratio      = poisson_ratio,
        yield_strength     = yield_strength,
        mesh_data          = mesh_data,
    )

    bc = r['bc_masks']
    # Per-hex von Mises stress for field visualization
    vm_hex = vm_stress_per_hex(r['nodes_nat'], r['nodes_cur'], r['hexes'],
                                young_modulus, poisson_ratio)
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    np.savez(
        out_path,
        nodes_nat            = r['nodes_nat'],
        nodes_cur            = r['nodes_cur'],
        hexes                = r['hexes'],
        f0_mask              = bc['f0'],
        f1_mask              = bc['f1'],
        f2_mask              = bc['f2'],
        f3_mask              = bc['f3'],
        vm_per_hex           = vm_hex,
        strain_energy        = np.array(r['strain_energy']),
        max_von_mises_stress = np.array(r['max_von_mises_stress']),
        max_xy_displacement  = np.array(r['max_xy_displacement']),
        max_z_displacement   = np.array(r['max_z_displacement']),
        first_yield_fraction = np.array(r['first_yield_fraction']),
        hinge_arm_width      = np.array(hinge_arm_width),
        hinge_fold_length    = np.array(hinge_fold_length),
        rotation_angle_deg   = np.array(rotation_angle_deg),
        applied_moment       = np.array(applied_moment),
        is_moment_mode       = np.array(loading_mode == 'moment'),
        face_size            = np.array(face_size),
        sheet_thickness      = np.array(sheet_thickness),
        young_modulus        = np.array(young_modulus),
        poisson_ratio        = np.array(poisson_ratio),
        yield_strength       = np.array(yield_strength),
    )
    print(f"  Saved -> {out_path}")
    print(f"  Strain energy        : {r['strain_energy']:.4e} J")
    print(f"  Max von Mises stress : {r['max_von_mises_stress']/1e6:.1f} MPa")
    print(f"  Max in-plane XY disp : {r['max_xy_displacement']*1e3:.2f} mm")
    print(f"  Max |z| buckling     : {r['max_z_displacement']*1e3:.2f} mm")
    print(f"  sigma / sigma_y      : {r['first_yield_fraction']:.2f}")


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument('--config',      type=str,   default=None,
                   help='Path to sofa YAML config (single source of truth). '
                        'Overrides all geometry/material flags below.')
    p.add_argument('--arm-width',   type=float, default=0.010,
                   help='Gap between panels [m] (default 10mm)')
    p.add_argument('--fold-length', type=float, default=0.003,
                   help='Hinge strip length along face edge at corner [m] (default 3mm)')
    p.add_argument('--angle',       type=float, default=-45.0,
                   help='In-plane rotation angle for F1 [degrees] (negative=CW, default -45)')
    p.add_argument('--moment',      type=float, default=0.0,
                   help='Applied bending moment on F1 [N.m] (moment mode only)')
    p.add_argument('--mode',        type=str,   default='rotation',
                   choices=['rotation', 'moment'])
    p.add_argument('--face-size',   type=float, default=FACE_SIZE)
    p.add_argument('--thickness',   type=float, default=SHEET_THICKNESS)
    p.add_argument('--young',       type=float, default=YOUNG_MODULUS)
    p.add_argument('--poisson',     type=float, default=POISSON_RATIO)
    p.add_argument('--yield-str',   type=float, default=YIELD_STRENGTH)
    p.add_argument('--out',         type=str,   default=DEFAULT_OUT)
    p.add_argument('--out-dir',     type=str,   default=None,
                   help='Run directory (from compare_jax_sofa.py). '
                        'Saves sofa_result.npz there; overrides --out.')
    p.add_argument('--mesh-npz',    type=str,   default=None,
                   help='Pre-built mesh .npz from build_mesh_from_centroidal_state. '
                        'When provided, skips build_unified_mesh.')
    return p.parse_args()


if __name__ == '__main__':
    import yaml

    args = _parse()

    if args.config is not None:
        # Config-driven path — single source of truth
        config_path = os.path.abspath(args.config)
        with open(config_path) as f:
            raw = yaml.safe_load(f)
        phys     = physical_scale_from_config(raw)
        sofa_raw = raw.get('sofa', {})
        loads    = raw.get('loads', [])

        arm_width       = phys.arm_width
        fold_length     = phys.fold_length
        face_size       = phys.face_size
        sheet_thickness = phys.sheet_thickness
        young_modulus   = phys.young_modulus
        poisson_ratio   = phys.poisson_ratio
        yield_strength  = phys.yield_strength
        loading_mode    = sofa_raw.get('loading_mode', 'moment')
        applied_moment  = float(loads[0]['value']) if loads else 0.0
        rotation_angle  = -45.0   # not used in moment mode

        # --out-dir puts sofa_result.npz in the run directory
        if args.out_dir is not None:
            out_path = os.path.join(args.out_dir, 'sofa_result.npz')
            # auto-detect cs_mesh.npz in run dir if --mesh-npz not given
            if args.mesh_npz is None:
                candidate = os.path.join(args.out_dir, 'cs_mesh.npz')
                if os.path.exists(candidate):
                    args.mesh_npz = candidate
        elif args.out != DEFAULT_OUT:
            out_path = args.out
        else:
            out_path = os.path.join(os.path.dirname(__file__), 'output', 'sofa_moment.npz')

        label = (f'config={os.path.basename(config_path)}  '
                 f'mode={loading_mode}  M={applied_moment:.3f}N·m  '
                 f'w={arm_width*1e3:.1f}mm  L={fold_length*1e3:.1f}mm  '
                 f'E={young_modulus/1e9:.1f}GPa  t={sheet_thickness*1e3:.1f}mm')
    else:
        # Legacy explicit-flag path
        arm_width       = args.arm_width
        fold_length     = args.fold_length
        face_size       = args.face_size
        sheet_thickness = args.thickness
        young_modulus   = args.young
        poisson_ratio   = args.poisson
        yield_strength  = args.yield_str
        loading_mode    = args.mode
        applied_moment  = args.moment
        rotation_angle  = args.angle
        out_path        = (os.path.join(args.out_dir, 'sofa_result.npz')
                           if args.out_dir is not None else args.out)
        label = (f'mode={args.mode}  w={args.arm_width*1e3:.1f}mm  '
                 f'L={args.fold_length*1e3:.1f}mm  angle={args.angle:.0f}deg  '
                 f't={args.thickness*1e3:.1f}mm')

    print(f'\nSOFA dump  ({label})')
    run_and_dump(
        hinge_arm_width    = arm_width,
        hinge_fold_length  = fold_length,
        rotation_angle_deg = rotation_angle,
        applied_moment     = applied_moment,
        loading_mode       = loading_mode,
        face_size          = face_size,
        sheet_thickness    = sheet_thickness,
        young_modulus      = young_modulus,
        poisson_ratio      = poisson_ratio,
        yield_strength     = yield_strength,
        out_path           = out_path,
        mesh_npz           = args.mesh_npz,
    )
