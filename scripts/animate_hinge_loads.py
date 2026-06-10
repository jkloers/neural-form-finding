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
    args = parser.parse_args()

    run_dir = pathlib.Path(args.run_dir)
    cfg_path = run_dir / "config.yaml"
    npz_path = run_dir / "convergence.npz"

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
        
    data = np.load(npz_path)
    best_idx = int(np.argmin(data['max_vm_rot']))
    phys = {
        'arm_width': float(data['arm_width'][best_idx]),
        'fold_top':  float(data['fold_top'][best_idx]),
        'fold_bot':  float(data['fold_bot'][best_idx]),
        'waist_top': float(data['waist_top'][best_idx]),
        'waist_bot': float(data['waist_bot'][best_idx]),
    }
    
    print(f"Animating best parameters (Epoch {best_idx+1}):")
    print(f"  arm_width = {phys['arm_width']*1e3:.2f} mm")
    print(f"  fold_top  = {phys['fold_top']*1e3:.2f} mm")

    cs_static = build_physical_cs(cfg)
    clamped_faces = sorted({int(f) for f in cs_static.constrained_face_DOF_pairs[:, 0]})
    loaded_faces  = sorted({int(f) for f in cs_static.loaded_face_DOF_pairs[:, 0]})

    sofa_cfg = cfg.get('sofa', {})
    
    nodes, hexes, bc_masks = build_mesh_from_centroidal_state(
        cs_static,
        fold_length        = float(phys['fold_top']),
        fold_bot           = float(phys['fold_bot']),
        bezier_params      = {'waist_top': phys['waist_top'], 'waist_bot': phys['waist_bot'], 'n_ctrl': int(sofa_cfg.get('n_ctrl', 3))},
        sheet_thickness    = float(sofa_cfg.get('sheet_thickness', 0.001)),
        n_face             = int(sofa_cfg.get('n_face', 4)),
        n_hinge            = int(sofa_cfg.get('n_hinge', 2)),
        n_z                = int(sofa_cfg.get('n_z', 2)),
        arm_width_physical = float(phys['arm_width']),
    )

    for k in ['clamped', 'loaded']:
        bc_masks[k] = np.zeros(len(nodes), dtype=bool)
    for fi in clamped_faces:
        if f'f{fi}' in bc_masks: bc_masks['clamped'] |= bc_masks[f'f{fi}']
    for fi in loaded_faces:
        if f'f{fi}' in bc_masks: bc_masks['loaded'] |= bc_masks[f'f{fi}']

    modes = [
        ('rotation', sofa_cfg.get('rotation_angle_deg', -5.0), 0.0),
        ('shear', 0.0, sofa_cfg.get('shear_displacement_m', 0.005)),
        ('tension', 0.0, sofa_cfg.get('tension_displacement_m', 0.005)),
    ]

    mesh_out = run_dir / "mesh_input.npz"
    np.savez(mesh_out, nodes=nodes, hexes=hexes, bc_masks=bc_masks, modes=modes, sofa_cfg=sofa_cfg)
    
    print("\nCalling SOFA worker...")
    subprocess.run(["./sofa/run_sofa.sh", "scripts/simulate_loads_sofa.py", str(run_dir)], check=True)
    
    print("\nCalling renderer...")
    subprocess.run(["conda", "run", "-n", "kgnn_mac", "python", "scripts/render_hinge_animations.py", str(run_dir)], check=True)

if __name__ == "__main__":
    main()
