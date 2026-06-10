#!/usr/bin/env python3
import sys
import argparse
import pathlib
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import PolyCollection

P_ORANGE  = "#F58025"
P_CLAMP   = "#2B2B2B"
P_LOAD    = "#C0392B"
P_BG      = "#FFFFFF"
P_EDGE    = "#D0D0D0"

def get_top_quads(nodes, hexes, bc_masks):
    top_z = nodes[:, 2].max()
    top_node_mask = nodes[:, 2] > (top_z - 1e-4)
    
    quads = []
    colors = []
    
    for h in hexes:
        top_h = [n for n in h if top_node_mask[n]]
        if len(top_h) >= 4:
            top_h = top_h[:4] # Take 4 just in case
            pts = nodes[top_h, :2]
            c = pts.mean(axis=0)
            angles = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
            sorted_h = np.array(top_h)[np.argsort(angles)]
            quads.append(sorted_h)
            
            is_clamped = all(bc_masks['clamped'][n] for n in sorted_h)
            is_loaded  = all(bc_masks['loaded'][n] for n in sorted_h)
            if is_clamped:
                colors.append(P_CLAMP)
            elif is_loaded:
                colors.append(P_LOAD)
            else:
                colors.append(P_ORANGE)
                
    return np.array(quads), colors

def _animate_mode(frames: np.ndarray, hexes: np.ndarray, bc_masks: dict, out_path: str, title: str):
    fig, ax = plt.subplots(figsize=(6, 6), facecolor=P_BG)
    fig.patch.set_facecolor(P_BG)
    
    ax.set_title(title, fontweight='bold', fontsize=12)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Calculate bounds
    all_x = np.concatenate([f[:, 0] for f in frames])
    all_y = np.concatenate([f[:, 1] for f in frames])
    pad = (all_x.max() - all_x.min()) * 0.1
    ax.set_xlim(all_x.min() - pad, all_x.max() + pad)
    ax.set_ylim(all_y.min() - pad, all_y.max() + pad)

    # Get quad topologies and colors from the initial frame
    quads, colors = get_top_quads(frames[0], hexes, bc_masks)
    
    # Create PolyCollection
    poly = PolyCollection([], facecolors=colors, edgecolors=P_EDGE, linewidths=0.5, alpha=0.9)
    ax.add_collection(poly)

    def init():
        poly.set_verts([])
        return poly,

    def update(frame_idx):
        pos = frames[frame_idx]
        verts = pos[quads, :2]
        poly.set_verts(verts)
        return poly,

    ani = animation.FuncAnimation(fig, update, frames=len(frames),
                                  init_func=init, blit=True, interval=100)
    
    writer = animation.PillowWriter(fps=15)
    ani.save(out_path, writer=writer, dpi=150)
    plt.close(fig)
    print(f"Saved animation → {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    args = parser.parse_args()

    run_dir = pathlib.Path(args.run_dir)
    mesh_path = run_dir / "mesh_input.npz"
    if not mesh_path.exists():
        sys.exit("Error: mesh_input.npz not found.")
        
    mesh_data = np.load(mesh_path, allow_pickle=True)
    hexes = mesh_data['hexes']
    
    for mode in ['rotation', 'shear', 'tension']:
        npz = run_dir / f"frames_{mode}.npz"
        if not npz.exists(): continue
        
        data = np.load(npz)
        bc_masks = {'clamped': data['clamped'], 'loaded': data['loaded']}
        out_path = str(run_dir / f"animation_{mode}.gif")
        
        print(f"Rendering {mode} (solid mesh)...")
        _animate_mode(data['frames'], hexes, bc_masks, out_path, title=f"{mode.capitalize()} Load Case")

if __name__ == "__main__":
    main()
