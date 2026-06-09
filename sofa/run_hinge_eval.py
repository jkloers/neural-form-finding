"""
sofa/run_hinge_eval.py — SOFA runner for hinge optimization.

Loads a pre-built CS mesh (nodes + hexes + bc_masks), runs one SOFA static
simulation, and saves a full result NPZ compatible with sofa/visualize.py.

Must be run via ./sofa/run_sofa.sh (requires Homebrew Python 3.12 + SOFA):

    ./sofa/run_sofa.sh sofa/run_hinge_eval.py \\
        --mesh-npz      /tmp/cs_mesh.npz \\
        --out-file      /tmp/sofa_result.npz \\
        --arm-width     0.010 \\
        --fold-length   0.003 \\
        --rotation-angle-deg -5.0 \\
        --young-modulus 3.5e9 --poisson-ratio 0.36 \\
        --yield-strength 50e6 --sheet-thickness 0.001
"""

import sys
import os
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import Sofa  # noqa: F401
except ImportError as e:
    sys.exit(f"Cannot import SOFA: {e}\nRun via ./sofa/run_sofa.sh")

from simulate_cell import evaluate_unit_cell
from materials     import vm_stress_per_hex


def _load_mesh_npz(path: str):
    d = np.load(path)
    nodes   = d['nodes']
    hexes   = d['hexes']
    n_faces = int(d['n_faces'])

    bc_masks: dict = {}
    for i in range(n_faces):
        for prefix in (f'f{i}', f'face_{i}'):
            key = f'{prefix}_mask'
            if key in d:
                bc_masks[prefix] = d[key].astype(bool)

    for label in ('clamped', 'loaded'):
        key = f'{label}_mask'
        if key in d:
            bc_masks[label] = d[key].astype(bool)

    return nodes, hexes, bc_masks, n_faces


def main():
    p = argparse.ArgumentParser(description='Run one SOFA unit-cell simulation.')
    p.add_argument('--mesh-npz',           required=True,  help='Pre-built CS mesh NPZ.')
    p.add_argument('--out-file',           required=True,  help='Output full-result NPZ.')
    p.add_argument('--arm-width',          type=float, default=0.0,
                   help='arm_width_physical [m] — saved as metadata for visualize.py.')
    p.add_argument('--fold-length',        type=float, default=0.0,
                   help='fold_length [m] — saved as metadata for visualize.py.')
    p.add_argument('--rotation-angle-deg', type=float, default=-5.0)
    p.add_argument('--young-modulus',      type=float, default=3.5e9)
    p.add_argument('--poisson-ratio',      type=float, default=0.36)
    p.add_argument('--yield-strength',     type=float, default=50e6)
    p.add_argument('--sheet-thickness',    type=float, default=0.001)
    args = p.parse_args()

    nodes, hexes, bc_masks, n_faces = _load_mesh_npz(args.mesh_npz)

    result = evaluate_unit_cell(
        nodes, hexes, bc_masks,
        rotation_angle_deg = args.rotation_angle_deg,
        applied_moment     = 0.0,
        loading_mode       = 'rotation',
        sheet_thickness    = args.sheet_thickness,
        young_modulus      = args.young_modulus,
        poisson_ratio      = args.poisson_ratio,
        yield_strength     = args.yield_strength,
    )

    vm_hex = vm_stress_per_hex(
        nodes, result['nodes_cur'], hexes,
        args.young_modulus, args.poisson_ratio,
    )

    fmasks = {
        f'f{i}_mask': bc_masks[f'f{i}']
        for i in range(n_faces) if f'f{i}' in bc_masks
    }

    np.savez(
        args.out_file,
        nodes_nat            = nodes,
        nodes_cur            = result['nodes_cur'],
        hexes                = hexes,
        vm_per_hex           = vm_hex,
        n_faces              = np.int32(n_faces),
        **fmasks,
        clamped_mask         = bc_masks.get('clamped',
                               bc_masks.get('f0', np.zeros(len(nodes), dtype=bool))),
        loaded_mask          = bc_masks.get('loaded',
                               bc_masks.get(f'f{n_faces-1}', np.zeros(len(nodes), dtype=bool))),
        strain_energy        = np.array(result['strain_energy']),
        max_von_mises_stress = np.array(result['max_von_mises_stress']),
        max_xy_displacement  = np.array(result['max_xy_displacement']),
        max_z_displacement   = np.array(result['max_z_displacement']),
        first_yield_fraction = np.array(result['first_yield_fraction']),
        rotation_angle_deg   = np.array(args.rotation_angle_deg),
        applied_moment       = np.array(0.0),
        is_moment_mode       = np.array(False),
        arm_width            = np.array(args.arm_width),
        fold_length          = np.array(args.fold_length),
        sheet_thickness      = np.array(args.sheet_thickness),
        young_modulus        = np.array(args.young_modulus),
        poisson_ratio        = np.array(args.poisson_ratio),
        yield_strength       = np.array(args.yield_strength),
    )

    print(f"max_von_mises_stress = {result['max_von_mises_stress']:.4e} Pa")
    print(f"strain_energy        = {result['strain_energy']:.4e} J")
    print(f"first_yield_fraction = {result['first_yield_fraction']:.4f}")
    print(f"Saved → {args.out_file}")


if __name__ == '__main__':
    main()
