#!/usr/bin/env python3
"""
scripts/plot_hinge_convergence.py — Standardized convergence plotting.

Reads the `convergence.npz` file from a hinge optimization run and generates
a publication-quality 6-panel convergence plot respecting project aesthetics.

Usage:
    python scripts/plot_hinge_convergence.py <run_dir>
"""

import sys
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════════════════
# Style — matches Princeton palette used throughout the project
# ═══════════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.color": "#CCCCCC",
    "axes.edgecolor": "#1A1A1A",
    "axes.linewidth": 1.2,
})

P_ORANGE  = "#F58025"   # Princeton orange
P_RED     = "#C0392B"   # Deep crimson (Shear)
P_BLUE    = "#154360"   # Dark navy (Rotation)
P_GREEN   = "#27AE60"   # Emerald green (Tension)
P_DARK    = "#1A1A1A"   # Near black (Loss)
P_MAGENTA = "#8E44AD"   # Purple (Stress)
P_BG      = "#FFFFFF"

def _plot_convergence(history: dict, out_path: str) -> None:
    epochs = range(1, len(history['total_loss']) + 1)
    
    fig, axes = plt.subplots(6, 1, figsize=(9.5, 14), sharex=True, facecolor=P_BG)
    fig.patch.set_facecolor(P_BG)
    
    fig.suptitle('SOFA Hinge Optimization — Bézier 5-Param Convergence', 
                 fontsize=14, fontweight='bold', color=P_DARK, y=0.98)

    # 1. Total Loss
    axes[0].plot(epochs, history['total_loss'], color=P_DARK, marker='o', markersize=3, linewidth=1.5)
    axes[0].set_ylabel('Total loss', fontweight='bold', color=P_DARK)
    
    # 2. Rotation Energy
    axes[1].semilogy(epochs, history['energy_rot'], color=P_BLUE, marker='o', markersize=3, linewidth=1.5,
                     label='E_rot (↓ compliant)')
    axes[1].set_ylabel('Rotation energy [J]', fontweight='bold', color=P_DARK)
    axes[1].legend(fontsize=8, loc='upper right')

    # 3. Max Von Mises Stress
    vm_mpa = [v / 1e6 for v in history['max_vm_rot']]
    axes[2].plot(epochs, vm_mpa, color=P_MAGENTA, marker='o', markersize=3, linewidth=1.5, label='σ_max (rot)')
    axes[2].set_ylabel('Max von Mises [MPa]', fontweight='bold', color=P_DARK)
    axes[2].legend(fontsize=8, loc='upper right')

    # 4. Shear & Tension Energy
    axes[3].semilogy(epochs, history['energy_shear'], color=P_RED, marker='o', markersize=3, linewidth=1.5,
                     label='E_shear (↑ stiff)')
    axes[3].semilogy(epochs, history['energy_tension'], color=P_GREEN, marker='^', markersize=3, linewidth=1.5,
                     label='E_tension (↑ stiff)')
    axes[3].set_ylabel('Stiffness energy [J]', fontweight='bold', color=P_DARK)
    axes[3].legend(fontsize=8, loc='upper right')

    # 5. Fold Geometry
    arm_mm = [v * 1e3 for v in history['arm_width']]
    ft_mm  = [v * 1e3 for v in history['fold_top']]
    fb_mm  = [v * 1e3 for v in history['fold_bot']]
    axes[4].plot(epochs, arm_mm, color=P_RED, marker='o', markersize=3, linewidth=1.5, label='arm_width')
    axes[4].plot(epochs, ft_mm, color=P_BLUE, marker='s', markersize=3, linewidth=1.5, label='fold_top')
    axes[4].plot(epochs, fb_mm, color=P_BLUE, marker='^', linestyle='--', markersize=3, linewidth=1.0, alpha=0.7, label='fold_bot')
    axes[4].set_ylabel('Fold geometry [mm]', fontweight='bold', color=P_DARK)
    axes[4].legend(fontsize=8, loc='upper left')

    # 6. Bézier Waist Geometry
    wt_mm = [v * 1e3 for v in history['waist_top']]
    wb_mm = [v * 1e3 for v in history['waist_bot']]
    axes[5].plot(epochs, wt_mm, color=P_GREEN, marker='o', markersize=3, linewidth=1.5, label='waist_top')
    axes[5].plot(epochs, wb_mm, color=P_GREEN, marker='^', linestyle='--', markersize=3, linewidth=1.0, alpha=0.7, label='waist_bot')
    axes[5].set_ylabel('Bézier waist [mm]', fontweight='bold', color=P_DARK)
    axes[5].set_xlabel('Epoch', fontweight='bold', color=P_DARK)
    axes[5].legend(fontsize=8, loc='upper left')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=250, bbox_inches='tight', facecolor=P_BG)
    plt.close(fig)
    print(f"Standardized convergence plot saved → {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Plot convergence history of hinge optimization.")
    parser.add_argument("run_dir", help="Path to the hinge_opt output directory containing convergence.npz")
    args = parser.parse_args()

    npz_path = os.path.join(args.run_dir, "convergence.npz")
    if not os.path.exists(npz_path):
        sys.exit(f"Error: {npz_path} not found.")

    out_path = os.path.join(args.run_dir, "convergence_standard.png")
    
    # Load history
    data = np.load(npz_path)
    history = {k: data[k] for k in data.files}
    
    _plot_convergence(history, out_path)

if __name__ == "__main__":
    main()
