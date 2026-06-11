#!/usr/bin/env python3
import sys
import argparse
import pathlib
import numpy as np

try:
    import Sofa
    import Sofa.Core
    import Sofa.Simulation
except ImportError as e:
    sys.exit(f"Cannot import SOFA: {e}\nPlease run via run_sofa.sh")

# Ensure we can import local modules
REPO = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(REPO))
from sofa.scene_builder import build_scene, N_STEPS_DEFAULT as N_STEPS, DT

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    args = parser.parse_args()
    
    run_dir = pathlib.Path(args.run_dir)
    mesh_path = run_dir / "mesh_input.npz"
    data = np.load(mesh_path, allow_pickle=True)
    
    nodes = data['nodes']
    hexes = data['hexes']
    bc_masks = data['bc_masks'].item()
    modes = data['modes'] # list of [mode, angle, disp]
    sofa_cfg = data['sofa_cfg'].item()
    
    n_steps        = int(sofa_cfg.get('n_steps', N_STEPS))
    rotation_pivot = sofa_cfg.get('rotation_pivot', None)
    fem_method     = str(sofa_cfg.get('fem_method', 'small'))

    for mode, angle, disp in modes:
        print(f"Simulating {mode}...")
        frames = []
        root = Sofa.Core.Node("root")
        try:
            mstate = build_scene(
                root, nodes, hexes, bc_masks,
                rotation_angle_deg     = float(angle) if mode == 'rotation' else 0.0,
                applied_moment         = 0.0,
                loading_mode           = str(mode),
                shear_displacement_m   = float(disp) if mode == 'shear' else 0.0,
                tension_displacement_m = float(disp) if mode == 'tension' else 0.0,
                young                  = 3.5e9,
                nu                     = 0.36,
                sheet_thickness        = float(sofa_cfg.get('sheet_thickness', 0.001)),
                rotation_pivot         = tuple(rotation_pivot) if rotation_pivot is not None else None,
                fem_method             = fem_method,
            )
            Sofa.Simulation.init(root)
            frames.append(np.array(mstate.position.value, dtype=np.float64))

            record_every = max(1, n_steps // 30)
            for step in range(1, n_steps + 1):
                Sofa.Simulation.animate(root, DT)
                if step % record_every == 0 or step == n_steps:
                    frames.append(np.array(mstate.position.value, dtype=np.float64))
        finally:
            Sofa.Simulation.unload(root)
            
        frames_np = np.stack(frames)
        np.savez(run_dir / f"frames_{mode}.npz", frames=frames_np, clamped=bc_masks['clamped'], loaded=bc_masks['loaded'])
        print(f"Saved frames for {mode}")

if __name__ == "__main__":
    main()
