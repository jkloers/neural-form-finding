#!/usr/bin/env python3
"""
scripts/visualize_hinge_run.py — Princeton-themed SOFA hinge run visualization.

Generates two outputs from a hinge_opt run directory:
  1. convergence.png  — optimizer convergence: loss, geometric params, CP trajectories, summary
  2. animation_{mode}.gif — Princeton-styled animated deformation under each load case

Usage:
    python scripts/visualize_hinge_run.py [run_dir] [--mode rotation|shear|tension|all]
    python scripts/visualize_hinge_run.py           # uses latest hinge_opt run

Options:
    --convergence-only   Only generate convergence.png, skip animations.
    --animation-only     Only generate animation GIFs, skip convergence plot.
    --out-convergence    Override output path for convergence.png.

Called automatically from animate_hinge_loads.py after SOFA simulation.
"""

import sys
import os
import argparse
import pathlib

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as mpl_animation
import matplotlib.patches as mpatches
from matplotlib.collections import PolyCollection

# ── Princeton palette ──────────────────────────────────────────────────────────
P_ORANGE  = "#F58025"
P_BLUE    = "#154360"
P_GREEN   = "#27AE60"
P_RED     = "#C0392B"
P_DARK    = "#1A1A1A"
P_BG      = "#FFFFFF"
P_GRID    = "#E8E8E8"
P_EDGE    = "#C8C8C8"
P_CLAMP   = "#154360"
P_LOAD    = "#C0392B"

# Nominal corner for unit_2face_D at face_size_m=0.1 m
_CORNER_X_MM = 141.42
_CORNER_Y_MM = 70.71


# ── Helpers ────────────────────────────────────────────────────────────────────

def _latest_run_dir() -> pathlib.Path:
    base = pathlib.Path(__file__).parent.parent / 'data' / 'outputs' / 'hinge_opt'
    if not base.exists():
        sys.exit(f'No hinge_opt output directory found at {base}')
    runs = sorted(
        d for d in os.listdir(base)
        if (base / d / 'convergence.npz').exists()
        or any((base / d / f'frames_{m}.npz').exists()
               for m in ('rotation', 'shear', 'tension'))
    )
    if not runs:
        sys.exit(f'No hinge_opt runs found in {base}')
    return base / runs[-1]


def _style_ax(ax, title='', xlabel='Epoch', ylabel=''):
    ax.set_title(title, fontsize=9, fontweight='bold', color=P_DARK, pad=4)
    ax.set_xlabel(xlabel, fontsize=8, color=P_DARK)
    ax.set_ylabel(ylabel, fontsize=8, color=P_DARK)
    ax.tick_params(labelsize=7, colors=P_DARK)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#BBBBBB')
    ax.spines['bottom'].set_color('#BBBBBB')
    ax.grid(True, alpha=0.25, color=P_GRID, linewidth=0.7)


# ── Convergence plot ───────────────────────────────────────────────────────────

def plot_convergence(run_dir: pathlib.Path, out_path: str = '') -> None:
    npz_path = run_dir / 'convergence.npz'
    if not npz_path.exists():
        print(f'  No convergence.npz in {run_dir} — skipping convergence plot.')
        return

    d = np.load(npz_path)
    n = len(d['total_loss'])
    epochs = np.arange(1, n + 1)

    best_e    = int(np.argmin(d['energy_rot']))
    has_bc    = 'bc1_x'  in d.files
    has_bc_lo = 'bc1l_x' in d.files

    fig, axes = plt.subplots(
        3, 2, figsize=(13, 11),
        gridspec_kw={'hspace': 0.52, 'wspace': 0.38},
        facecolor=P_BG,
    )
    fig.suptitle(
        f'Corner-hinge optimizer — {run_dir.name}',
        fontsize=11, fontweight='bold', color=P_DARK, y=0.98,
    )
    for ax in axes.flat:
        ax.set_facecolor(P_BG)

    # (0,0) Loss + rotation energy
    ax = axes[0, 0]
    ax.semilogy(epochs, np.abs(d['total_loss']), color=P_DARK, lw=1.8,
                marker='o', ms=2.5, label='|loss|')
    ax.semilogy(epochs, d['energy_rot'], color=P_BLUE, lw=1.5,
                marker='s', ms=2.5, linestyle='--', label='E_rot [J]')
    ax.axvline(best_e + 1, color=P_RED, linestyle=':', lw=1.2,
               label=f'best (ep {best_e + 1})')
    ax.legend(fontsize=7, framealpha=0.75, edgecolor=P_EDGE)
    _style_ax(ax, 'Loss + rotation energy', ylabel='Energy [J]')

    # (0,1) Geometric params
    ax = axes[0, 1]
    ax.plot(epochs, d['arm_width'] * 1e3, color=P_RED, lw=1.5, ms=2.5,
            marker='o', label='arm_width [mm]')
    ax.plot(epochs, d['fold_top']  * 1e3, color=P_BLUE, lw=1.5, ms=2.5,
            marker='s', label='fold_top [mm]')
    ax.plot(epochs, d['fold_bot']  * 1e3 * 100, color=P_GREEN, lw=1.0, ms=2.5,
            marker='^', linestyle='--', label='fold_bot×100 [mm]')
    ax.legend(fontsize=7, framealpha=0.75, edgecolor=P_EDGE)
    _style_ax(ax, 'Geometric params (arm + fold)', ylabel='[mm]')

    # (1,0) Upper wing CP trajectories
    ax = axes[1, 0]
    if has_bc:
        for key_x, key_y, cmap, lbl in [
            ('bc1_x', 'bc1_y', 'Blues',  'bc1_up'),
            ('bc2_x', 'bc2_y', 'Greens', 'bc2_up'),
        ]:
            ax.scatter(d[key_x] * 1e3, d[key_y] * 1e3, c=epochs,
                       cmap=cmap, s=18, zorder=3, alpha=0.85)
            ax.plot(d[key_x] * 1e3, d[key_y] * 1e3, lw=0.6, alpha=0.30, color=P_DARK)
            ax.plot(d[key_x][0]  * 1e3, d[key_y][0]  * 1e3,
                    's', color=P_GREEN, ms=7, zorder=5)
            ax.plot(d[key_x][-1] * 1e3, d[key_y][-1] * 1e3,
                    '*', color=P_RED, ms=10, zorder=5, label=f'{lbl} end')
    ax.axhline(_CORNER_Y_MM, color=P_DARK, lw=0.8, linestyle='--', alpha=0.4)
    ax.axvline(_CORNER_X_MM, color=P_DARK, lw=0.8, linestyle=':', alpha=0.4)
    ax.legend(fontsize=7, framealpha=0.75, edgecolor=P_EDGE)
    _style_ax(ax, 'Upper wing CPs (bc1_up, bc2_up)',
              xlabel='x [mm]', ylabel='y [mm]')

    # (1,1) Lower wing CP trajectories
    ax = axes[1, 1]
    if has_bc_lo:
        for key_x, key_y, cmap, lbl in [
            ('bc1l_x', 'bc1l_y', 'Oranges', 'bc1_lo'),
            ('bc2l_x', 'bc2l_y', 'Reds',    'bc2_lo'),
        ]:
            ax.scatter(d[key_x] * 1e3, d[key_y] * 1e3, c=epochs,
                       cmap=cmap, s=18, zorder=3, alpha=0.85)
            ax.plot(d[key_x] * 1e3, d[key_y] * 1e3, lw=0.6, alpha=0.30, color=P_DARK)
            ax.plot(d[key_x][0]  * 1e3, d[key_y][0]  * 1e3,
                    's', color=P_GREEN, ms=7, zorder=5)
            ax.plot(d[key_x][-1] * 1e3, d[key_y][-1] * 1e3,
                    '*', color=P_RED, ms=10, zorder=5, label=f'{lbl} end')
    ax.axhline(_CORNER_Y_MM, color=P_DARK, lw=0.8, linestyle='--', alpha=0.4)
    ax.axvline(_CORNER_X_MM, color=P_DARK, lw=0.8, linestyle=':', alpha=0.4)
    ax.legend(fontsize=7, framealpha=0.75, edgecolor=P_EDGE)
    _style_ax(ax, 'Lower wing CPs (bc1_lo, bc2_lo)',
              xlabel='x [mm]', ylabel='y [mm]')

    # (2,0) Initial vs best CP positions
    ax = axes[2, 0]
    if has_bc:
        for key_x, key_y, color, lbl in [
            ('bc1_x',  'bc1_y',  '#1565C0', 'bc1_up'),
            ('bc2_x',  'bc2_y',  '#2196F3', 'bc2_up'),
        ]:
            ax.scatter(d[key_x][0]      * 1e3, d[key_y][0]      * 1e3,
                       marker='s', color=P_GREEN, s=60, zorder=4)
            ax.scatter(d[key_x][best_e] * 1e3, d[key_y][best_e] * 1e3,
                       marker='*', color=color, s=140, zorder=5, label=lbl)
            ax.annotate(lbl, (d[key_x][best_e] * 1e3, d[key_y][best_e] * 1e3),
                        fontsize=7, xytext=(3, 3), textcoords='offset points',
                        color=P_DARK)
    if has_bc_lo:
        for key_x, key_y, color, lbl in [
            ('bc1l_x', 'bc1l_y', '#E65100', 'bc1_lo'),
            ('bc2l_x', 'bc2l_y', '#FF8F00', 'bc2_lo'),
        ]:
            ax.scatter(d[key_x][0]      * 1e3, d[key_y][0]      * 1e3,
                       marker='s', color=P_GREEN, s=60, zorder=4)
            ax.scatter(d[key_x][best_e] * 1e3, d[key_y][best_e] * 1e3,
                       marker='*', color=color, s=140, zorder=5, label=lbl)
            ax.annotate(lbl, (d[key_x][best_e] * 1e3, d[key_y][best_e] * 1e3),
                        fontsize=7, xytext=(3, 3), textcoords='offset points',
                        color=P_DARK)
    ax.axhline(_CORNER_Y_MM, color=P_DARK, lw=0.8, linestyle='--',
               alpha=0.4, label='corner y')
    ax.axvline(_CORNER_X_MM, color=P_DARK, lw=0.8, linestyle=':',
               alpha=0.4, label='corner x')
    ax.legend(fontsize=7, framealpha=0.75, edgecolor=P_EDGE)
    _style_ax(ax, 'Initial (■ green) vs best (★) CPs',
              xlabel='x [mm]', ylabel='y [mm]')

    # (2,1) Text summary
    ax = axes[2, 1]
    bc_str = ''
    if has_bc:
        bc_str += (
            f'bc1_up = ({d["bc1_x"][best_e]*1e3:.2f}, {d["bc1_y"][best_e]*1e3:.2f}) mm\n'
            f'bc2_up = ({d["bc2_x"][best_e]*1e3:.2f}, {d["bc2_y"][best_e]*1e3:.2f}) mm\n'
        )
    if has_bc_lo:
        bc_str += (
            f'bc1_lo = ({d["bc1l_x"][best_e]*1e3:.2f}, {d["bc1l_y"][best_e]*1e3:.2f}) mm\n'
            f'bc2_lo = ({d["bc2l_x"][best_e]*1e3:.2f}, {d["bc2l_y"][best_e]*1e3:.2f}) mm\n'
        )
    e0   = float(d['energy_rot'][0])
    ebst = float(d['energy_rot'][best_e])
    reduction = e0 / ebst if ebst > 1e-12 else float('inf')
    ax.text(
        0.06, 0.94,
        f'Epochs:     {n}\n'
        f'Best epoch: {best_e + 1}  (min E_rot)\n\n'
        f'E_rot_init  = {e0:.4f} J\n'
        f'E_rot_best  = {ebst:.4f} J\n'
        f'E_rot_final = {d["energy_rot"][-1]:.4f} J\n'
        f'Reduction   = {reduction:.1f}×\n\n'
        f'arm_width = {d["arm_width"][best_e]*1e3:.3f} mm\n'
        f'fold_top  = {d["fold_top"][best_e]*1e3:.3f} mm\n'
        f'fold_bot  = {d["fold_bot"][best_e]*1e3:.4f} mm\n\n'
        f'{bc_str}',
        transform=ax.transAxes, fontsize=9, va='top', family='monospace',
        color=P_DARK,
        bbox=dict(
            boxstyle='round,pad=0.6',
            facecolor='#FFF8F0',
            edgecolor=P_ORANGE,
            linewidth=1.2,
            alpha=0.92,
        ),
    )
    ax.axis('off')
    ax.set_title('Best configuration', fontsize=9, fontweight='bold', color=P_DARK, pad=4)

    out = out_path or str(run_dir / 'convergence.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=P_BG)
    plt.close(fig)
    print(f'  Convergence → {out}')


# ── Animation ──────────────────────────────────────────────────────────────────

def _get_top_quads(nodes: np.ndarray, hexes: np.ndarray, bc_masks: dict):
    """Extract top-z quads from hex mesh and assign Princeton BC colors."""
    top_z    = nodes[:, 2].max()
    top_mask = nodes[:, 2] > (top_z - 1e-4)

    quads  = []
    colors = []
    for h in hexes:
        top_h = [n for n in h if top_mask[n]]
        if len(top_h) < 4:
            continue
        top_h = top_h[:4]
        pts = nodes[top_h, :2]
        c   = pts.mean(axis=0)
        ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
        sorted_h = np.array(top_h)[np.argsort(ang)]
        quads.append(sorted_h)

        is_clamped = all(bc_masks['clamped'][n] for n in sorted_h)
        is_loaded  = all(bc_masks['loaded'][n]  for n in sorted_h)
        if is_clamped:
            colors.append(P_CLAMP)
        elif is_loaded:
            colors.append(P_LOAD)
        else:
            colors.append(P_ORANGE)

    return np.array(quads), colors


def _render_animation(
    frames:   np.ndarray,
    hexes:    np.ndarray,
    bc_masks: dict,
    out_path: str,
    mode:     str,
) -> None:
    """Princeton-styled animated GIF of SOFA hinge deformation."""
    fig, ax = plt.subplots(figsize=(7, 7), facecolor=P_BG)
    fig.patch.set_facecolor(P_BG)
    ax.set_facecolor(P_BG)

    ax.set_title(
        f'Hinge deformation — {mode.capitalize()} load case',
        fontsize=11, fontweight='bold', color=P_DARK, pad=8,
    )
    ax.set_aspect('equal')
    ax.axis('off')

    all_xy = np.concatenate([f[:, :2] for f in frames], axis=0)
    rx = all_xy[:, 0].max() - all_xy[:, 0].min()
    ry = all_xy[:, 1].max() - all_xy[:, 1].min()
    px = max(rx * 0.12, ry * 0.05)
    py = max(ry * 0.12, rx * 0.05)
    ax.set_xlim(all_xy[:, 0].min() - px, all_xy[:, 0].max() + px)
    ax.set_ylim(all_xy[:, 1].min() - py, all_xy[:, 1].max() + py)

    quads, face_colors = _get_top_quads(frames[0], hexes, bc_masks)

    poly = PolyCollection(
        [],
        facecolors=face_colors,
        edgecolors=P_EDGE,
        linewidths=0.5,
        alpha=0.92,
    )
    ax.add_collection(poly)

    frame_txt = ax.text(
        0.97, 0.03, '',
        transform=ax.transAxes,
        fontsize=8, ha='right', va='bottom',
        color=P_DARK, family='monospace',
        bbox=dict(boxstyle='round,pad=0.3', facecolor=P_BG, alpha=0.75, edgecolor='none'),
    )

    legend_patches = [
        mpatches.Patch(facecolor=P_ORANGE, edgecolor=P_EDGE, label='Panel'),
        mpatches.Patch(facecolor=P_CLAMP,  edgecolor=P_EDGE, label='Clamped'),
        mpatches.Patch(facecolor=P_LOAD,   edgecolor=P_EDGE, label='Loaded'),
    ]
    ax.legend(handles=legend_patches, loc='upper left', fontsize=7,
              framealpha=0.85, edgecolor=P_EDGE)

    n_frames = len(frames)

    def _init():
        poly.set_verts([])
        frame_txt.set_text('')
        return poly, frame_txt

    def _update(i):
        pos  = frames[i]
        verts = pos[quads, :2]
        poly.set_verts(verts)
        frame_txt.set_text(f'step {i + 1}/{n_frames}')
        return poly, frame_txt

    ani = mpl_animation.FuncAnimation(
        fig, _update, frames=n_frames,
        init_func=_init, blit=True, interval=80,
    )
    writer = mpl_animation.PillowWriter(fps=15)
    ani.save(out_path, writer=writer, dpi=150)
    plt.close(fig)
    print(f'  Animation → {out_path}')


def render_animations(run_dir: pathlib.Path, modes: list) -> None:
    mesh_path = run_dir / 'mesh_input.npz'
    if not mesh_path.exists():
        print(f'  No mesh_input.npz in {run_dir} — skipping animations.')
        return

    mesh_data = np.load(mesh_path, allow_pickle=True)
    hexes = mesh_data['hexes']

    for mode in modes:
        npz = run_dir / f'frames_{mode}.npz'
        if not npz.exists():
            continue
        data     = np.load(npz)
        bc_masks = {'clamped': data['clamped'], 'loaded': data['loaded']}
        out_path = str(run_dir / f'animation_{mode}.gif')
        print(f'  Rendering {mode} ({len(data["frames"])} frames)...')
        _render_animation(data['frames'], hexes, bc_masks, out_path, mode)


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description='Princeton-themed SOFA hinge run visualization.')
    ap.add_argument('run_dir', nargs='?', default=None,
                    help='Hinge opt run directory. Default: latest run in data/outputs/hinge_opt/.')
    ap.add_argument('--mode', default='rotation',
                    help='Load mode(s) to animate: rotation, shear, tension, or all.')
    ap.add_argument('--convergence-only', action='store_true',
                    help='Only generate convergence.png.')
    ap.add_argument('--animation-only', action='store_true',
                    help='Only generate animation GIFs.')
    ap.add_argument('--out-convergence', default='',
                    help='Override output path for convergence.png.')
    args = ap.parse_args()

    run_dir = pathlib.Path(args.run_dir) if args.run_dir else _latest_run_dir()
    print(f'Run: {run_dir}')

    if not args.animation_only:
        plot_convergence(run_dir, out_path=args.out_convergence)

    if not args.convergence_only:
        modes = (['rotation', 'shear', 'tension'] if args.mode == 'all'
                 else [args.mode])
        render_animations(run_dir, modes)


if __name__ == '__main__':
    main()
