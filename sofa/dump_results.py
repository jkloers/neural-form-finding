"""
sofa/dump_results.py — Run the SOFA simulation and save results to .npz.

NO matplotlib import — avoids the SOFA+Qt/Cocoa crash on macOS.
Results are loaded by sofa/visualize.py in the kgnn_mac conda env.

Usage (via run_sofa.sh):
    ./sofa/run_sofa.sh sofa/dump_results.py
    ./sofa/run_sofa.sh sofa/dump_results.py --fold-length 0.010 --displacement 0.020
    ./sofa/run_sofa.sh sofa/dump_results.py --out /tmp/sofa_result.npz
"""

import sys
import os
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

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

DEFAULT_OUT = os.path.join(os.path.dirname(__file__), 'output', 'sofa_result.npz')


def run_and_dump(
    hinge_arm_width      = 0.005,
    hinge_fold_length    = 0.020,
    applied_displacement = 0.010,
    applied_moment       = 0.0,
    loading_mode         = 'rotation',
    face_size            = FACE_SIZE,
    sheet_thickness      = SHEET_THICKNESS,
    young_modulus        = YOUNG_MODULUS,
    poisson_ratio        = POISSON_RATIO,
    yield_strength       = YIELD_STRENGTH,
    out_path             = DEFAULT_OUT,
):
    print("  Running SOFA simulation ...")
    r = evaluate_unit_cell(
        hinge_arm_width      = hinge_arm_width,
        hinge_fold_length    = hinge_fold_length,
        applied_displacement = applied_displacement,
        applied_moment       = applied_moment,
        loading_mode         = loading_mode,
        face_size            = face_size,
        sheet_thickness      = sheet_thickness,
        young_modulus        = young_modulus,
        poisson_ratio        = poisson_ratio,
        yield_strength       = yield_strength,
    )

    bc = r['bc_masks']
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
        strain_energy        = np.array(r['strain_energy']),
        max_von_mises_stress = np.array(r['max_von_mises_stress']),
        max_z_displacement   = np.array(r['max_z_displacement']),
        first_yield_fraction = np.array(r['first_yield_fraction']),
        hinge_arm_width      = np.array(hinge_arm_width),
        hinge_fold_length    = np.array(hinge_fold_length),
        applied_displacement = np.array(applied_displacement),
        applied_moment       = np.array(applied_moment),
        is_moment_mode       = np.array(loading_mode == 'moment'),
        face_size            = np.array(face_size),
        sheet_thickness      = np.array(sheet_thickness),
        young_modulus        = np.array(young_modulus),
        poisson_ratio        = np.array(poisson_ratio),
        yield_strength       = np.array(yield_strength),
    )
    print(f"  Saved → {out_path}")
    print(f"  Strain energy        : {r['strain_energy']:.4e} J")
    print(f"  Max von Mises stress : {r['max_von_mises_stress']/1e6:.1f} MPa")
    print(f"  Max |z| displacement : {r['max_z_displacement']*1e3:.4f} mm")
    print(f"  σ / σ_y              : {r['first_yield_fraction']:.3f}")


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument('--arm-width',    type=float, default=0.005)
    p.add_argument('--fold-length',  type=float, default=0.020)
    p.add_argument('--displacement', type=float, default=0.010,
                   help='Peak z-displacement of F1 [m] (rotation mode)')
    p.add_argument('--moment',       type=float, default=0.0,
                   help='Applied bending moment on F1 [N·m] (moment mode)')
    p.add_argument('--mode',         type=str,   default='rotation',
                   choices=['rotation', 'moment'],
                   help='Loading mode: rotation (displacement-ctrl) or moment (force-ctrl)')
    p.add_argument('--face-size',    type=float, default=FACE_SIZE)
    p.add_argument('--thickness',    type=float, default=SHEET_THICKNESS)
    p.add_argument('--young',        type=float, default=YOUNG_MODULUS)
    p.add_argument('--poisson',      type=float, default=POISSON_RATIO)
    p.add_argument('--yield-str',    type=float, default=YIELD_STRENGTH)
    p.add_argument('--out',          type=str,   default=DEFAULT_OUT)
    return p.parse_args()


if __name__ == '__main__':
    args = _parse()
    label = (f'mode={args.mode}  w={args.arm_width*1e3:.1f}mm  '
             f'L={args.fold_length*1e3:.1f}mm  t={args.thickness*1e3:.1f}mm')
    print(f'\nSOFA dump  ({label})')
    run_and_dump(
        hinge_arm_width      = args.arm_width,
        hinge_fold_length    = args.fold_length,
        applied_displacement = args.displacement,
        applied_moment       = args.moment,
        loading_mode         = args.mode,
        face_size            = args.face_size,
        sheet_thickness      = args.thickness,
        young_modulus        = args.young,
        poisson_ratio        = args.poisson,
        yield_strength       = args.yield_str,
        out_path             = args.out,
    )
