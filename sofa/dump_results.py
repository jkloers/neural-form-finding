"""
sofa/dump_results.py — Run the SOFA simulation and save results to .npz.

Both --config and --mesh-npz are required. The mesh encodes the CS geometry
and was built by nff.sofa.mesh_builder.build_mesh_from_centroidal_state.

NO matplotlib import — avoids the SOFA+Qt/Cocoa crash on macOS.
Results are loaded by sofa/visualize.py in the kgnn_mac conda env.

Usage (via run_sofa.sh):
    ./sofa/run_sofa.sh sofa/dump_results.py \\
        --config  data/configs/sofa/c001_mpnn_2x2.yaml \\
        --mesh-npz data/outputs/runs/<run>/cs_mesh.npz \\
        [--out-dir data/outputs/runs/<run>/]
"""

import sys
import os
import argparse
import yaml
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

from simulate_cell import evaluate_unit_cell
from materials import vm_stress_per_hex
from nff.sofa.config_to_physical import physical_scale_from_config

DEFAULT_OUT = os.path.join(os.path.dirname(__file__), 'output', 'sofa_result.npz')


def _load_mesh(mesh_npz):
    d = np.load(mesh_npz)
    nodes = d['nodes']
    hexes = d['hexes']
    n_faces = int(d['n_faces']) if 'n_faces' in d else sum(
        1 for k in d.files
        if k.startswith('f') and k.endswith('_mask') and k[1:-5].isdigit())
    bc_masks = {}
    for i in range(n_faces):
        m = d[f'f{i}_mask'].astype(bool)
        bc_masks[f'f{i}']     = m
        bc_masks[f'face_{i}'] = m
    bc_masks['clamped'] = d.get('clamped_mask', d['f0_mask']).astype(bool)
    bc_masks['loaded']  = d.get('loaded_mask',  d['f1_mask']).astype(bool)
    return nodes, hexes, bc_masks, n_faces


def main():
    p = argparse.ArgumentParser(
        description='Run SOFA simulation from a pre-built CS mesh and save results.')
    p.add_argument('--config',   required=True,
                   help='Path to YAML config (sofa.* keys + material params).')
    p.add_argument('--mesh-npz', required=True,
                   help='Pre-built CS hex mesh .npz (from build_mesh_from_centroidal_state).')
    p.add_argument('--out',      default=DEFAULT_OUT,
                   help='Output .npz path.')
    p.add_argument('--out-dir',  default=None,
                   help='Run directory; saves sofa_result.npz there (overrides --out).')
    args = p.parse_args()

    with open(args.config) as f:
        raw = yaml.safe_load(f)
    phys     = physical_scale_from_config(raw)
    sofa_raw = raw.get('sofa', {})
    loads    = raw.get('loads', [])

    rotation_angle = float(sofa_raw.get('rotation_angle_deg', -45.0))
    applied_moment = float(sofa_raw.get('applied_moment',
                                        loads[0]['value'] if loads else 0.0))
    loading_mode   = sofa_raw.get('loading_mode', 'rotation')

    out_path = (os.path.join(args.out_dir, 'sofa_result.npz')
                if args.out_dir is not None else args.out)

    print(f"\nSOFA dump  (config={os.path.basename(args.config)}  "
          f"mode={loading_mode}  angle={rotation_angle:.0f}deg  M={applied_moment:.4f}N·m  "
          f"w={phys.arm_width*1e3:.1f}mm  L={phys.fold_length*1e3:.1f}mm  "
          f"E={phys.young_modulus/1e9:.1f}GPa  t={phys.sheet_thickness*1e3:.1f}mm)")

    print(f"  Loading mesh from {args.mesh_npz} ...")
    nodes, hexes, bc_masks, n_faces = _load_mesh(args.mesh_npz)
    print(f"  Mesh: {len(nodes)} nodes, {len(hexes)} hexes, {n_faces} faces")

    print(f"  Running SOFA [{loading_mode}] ...", flush=True)
    r = evaluate_unit_cell(
        nodes, hexes, bc_masks,
        rotation_angle_deg = rotation_angle,
        applied_moment     = applied_moment,
        loading_mode       = loading_mode,
        sheet_thickness    = phys.sheet_thickness,
        young_modulus      = phys.young_modulus,
        poisson_ratio      = phys.poisson_ratio,
        yield_strength     = phys.yield_strength,
    )

    bc    = r['bc_masks']
    n_f   = sum(1 for k in bc if k.startswith('f') and k[1:].isdigit())
    vm_hex = vm_stress_per_hex(r['nodes_nat'], r['nodes_cur'], r['hexes'],
                                phys.young_modulus, phys.poisson_ratio)
    fmasks = {f'f{i}_mask': bc[f'f{i}'] for i in range(n_f)}

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    np.savez(
        out_path,
        nodes_nat            = r['nodes_nat'],
        nodes_cur            = r['nodes_cur'],
        hexes                = r['hexes'],
        **fmasks,
        vm_per_hex           = vm_hex,
        n_faces              = np.int32(n_f),
        strain_energy        = np.array(r['strain_energy']),
        max_von_mises_stress = np.array(r['max_von_mises_stress']),
        max_xy_displacement  = np.array(r['max_xy_displacement']),
        max_z_displacement   = np.array(r['max_z_displacement']),
        first_yield_fraction = np.array(r['first_yield_fraction']),
        rotation_angle_deg   = np.array(rotation_angle),
        applied_moment       = np.array(applied_moment),
        is_moment_mode       = np.array(loading_mode == 'moment'),
        arm_width            = np.array(phys.arm_width),
        fold_length          = np.array(phys.fold_length),
        sheet_thickness      = np.array(phys.sheet_thickness),
        young_modulus        = np.array(phys.young_modulus),
        poisson_ratio        = np.array(phys.poisson_ratio),
        yield_strength       = np.array(phys.yield_strength),
    )
    print(f"  Saved -> {out_path}")
    print(f"  Strain energy        : {r['strain_energy']:.4e} J")
    print(f"  Max von Mises stress : {r['max_von_mises_stress']/1e6:.1f} MPa")
    print(f"  Max in-plane XY disp : {r['max_xy_displacement']*1e3:.2f} mm")
    print(f"  Max |z| buckling     : {r['max_z_displacement']*1e3:.2f} mm")
    print(f"  sigma / sigma_y      : {r['first_yield_fraction']:.2f}")


if __name__ == '__main__':
    main()
