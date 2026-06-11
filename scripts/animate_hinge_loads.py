#!/usr/bin/env python3
import sys
import os
import argparse
import pathlib
import yaml
import numpy as np
import subprocess

REPO = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from sofa.mesh_builder import build_mesh_from_centroidal_state
from sofa.hinge_optimizer import build_physical_cs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--mode", default="rotation",
                        help="Load mode to animate: rotation, shear, tension (default: rotation)")
    args = parser.parse_args()

    run_dir = pathlib.Path(args.run_dir)
    cfg_path = run_dir / "config.yaml"
    npz_path = run_dir / "convergence.npz"

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    data = np.load(npz_path)
    best_idx = int(np.argmin(data['energy_rot']))

    def _get(key, default=None):
        return float(data[key][best_idx]) if key in data.files else default

    phys = {
        'arm_width': _get('arm_width'),
        'fold_top':  _get('fold_top'),
        'fold_bot':  _get('fold_bot'),
        'bc1_x':     _get('bc1_x'),
        'bc1_y':     _get('bc1_y'),
        'bc2_x':     _get('bc2_x'),
        'bc2_y':     _get('bc2_y'),
        'bc1l_x':    _get('bc1l_x'),
        'bc1l_y':    _get('bc1l_y'),
        'bc2l_x':    _get('bc2l_x'),
        'bc2l_y':    _get('bc2l_y'),
    }

    print(f"Animating best parameters (Epoch {best_idx+1}):")
    print(f"  arm_width = {phys['arm_width']*1e3:.2f} mm")
    print(f"  fold_top  = {phys['fold_top']*1e3:.2f} mm")
    print(f"  fold_bot  = {phys['fold_bot']*1e6:.2f} μm")
    print(f"  bc1_up = ({phys['bc1_x']*1e3:.2f}, {phys['bc1_y']*1e3:.2f}) mm")
    print(f"  bc2_up = ({phys['bc2_x']*1e3:.2f}, {phys['bc2_y']*1e3:.2f}) mm")
    if phys['bc1l_x'] is not None:
        print(f"  bc1_lo = ({phys['bc1l_x']*1e3:.2f}, {phys['bc1l_y']*1e3:.2f}) mm")
        print(f"  bc2_lo = ({phys['bc2l_x']*1e3:.2f}, {phys['bc2l_y']*1e3:.2f}) mm")

    cs = build_physical_cs(cfg)
    sofa_cfg = cfg.get('sofa', {})

    bezier_params = {
        'bc1_upper_xy': [phys['bc1_x'],  phys['bc1_y']],
        'bc2_upper_xy': [phys['bc2_x'],  phys['bc2_y']],
        'bc1_lower_xy': [phys['bc1l_x'], phys['bc1l_y']] if phys['bc1l_x'] is not None else None,
        'bc2_lower_xy': [phys['bc2l_x'], phys['bc2l_y']] if phys['bc2l_x'] is not None else None,
    }

    nodes, hexes, bc_masks = build_mesh_from_centroidal_state(
        cs,
        fold_top           = phys['fold_top'],
        fold_bot           = phys['fold_bot'],
        arm_width_physical = phys['arm_width'],
        sheet_thickness    = float(sofa_cfg.get('sheet_thickness', 0.001)),
        n_face             = int(sofa_cfg.get('n_face', 4)),
        n_hinge            = int(sofa_cfg.get('n_hinge', 4)),
        n_z                = int(sofa_cfg.get('n_z', 2)),
        bezier_params      = bezier_params,
    )

    # Rotation pivot: hinge corner vertex (rotation_pivot_auto logic)
    hnp = cs.hinge_node_pairs
    rotation_pivot = None
    if len(hnp) > 0:
        fi = int(hnp[0, 0, 0])
        lj = int(hnp[0, 0, 1])
        corner_xy = cs.face_centroids[fi] + cs.centroid_node_vectors[fi, lj]
        rotation_pivot = (float(corner_xy[0]), float(corner_xy[1]))
        print(f"  rotation_pivot = ({rotation_pivot[0]*1e3:.2f}, {rotation_pivot[1]*1e3:.2f}) mm")

    rotation_angle = float(sofa_cfg.get('rotation_angle_deg', 10.0))
    shear_disp     = float(sofa_cfg.get('shear_displacement_m', 0.0002))
    tension_disp   = float(sofa_cfg.get('tension_displacement_m', 0.0002))
    fem_method     = str(sofa_cfg.get('fem_method', 'small'))
    n_steps        = int(sofa_cfg.get('n_steps', 100))

    # Only run the requested mode
    if args.mode == 'rotation':
        modes = [('rotation', rotation_angle, 0.0)]
    elif args.mode == 'shear':
        modes = [('shear', 0.0, shear_disp)]
    elif args.mode == 'tension':
        modes = [('tension', 0.0, tension_disp)]
    else:
        modes = [
            ('rotation', rotation_angle, 0.0),
            ('shear',    0.0, shear_disp),
            ('tension',  0.0, tension_disp),
        ]

    mesh_out = run_dir / "mesh_input.npz"
    np.savez(mesh_out,
             nodes=np.array(nodes), hexes=np.array(hexes),
             bc_masks=bc_masks,
             modes=modes,
             sofa_cfg={
                 'sheet_thickness': sofa_cfg.get('sheet_thickness', 0.001),
                 'fem_method':      fem_method,
                 'n_steps':         n_steps,
                 'rotation_pivot':  rotation_pivot,
             })

    print("\nCalling SOFA worker...")
    subprocess.run(
        ["./sofa/run_sofa.sh", "scripts/simulate_loads_sofa.py", str(run_dir)],
        check=True)

    print("\nCalling visualizer...")
    subprocess.run(
        ["/opt/miniconda3/envs/kgnn_mac/bin/python",
         "scripts/visualize_hinge_run.py", str(run_dir),
         "--mode", args.mode],
        check=True)

if __name__ == "__main__":
    main()
