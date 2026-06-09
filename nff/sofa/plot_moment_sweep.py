"""
nff/sofa/plot_moment_sweep.py — Plot moment-sweep results from sweep_summary.npz.

Usage:
    conda run -n kgnn_mac python nff/sofa/plot_moment_sweep.py \\
        --sweep-dir data/outputs/runs/<run>/moment_sweep/

Outputs (in --sweep-dir):
    moment_sweep_response.png  — moment vs per-face rotation curves
    moment_sweep_energy.png    — moment vs strain energy (log-log linearity = linear elasticity)
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

REPO = pathlib.Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO))


_BASE_COLORS = [
    '#4472C4', '#ED7D31', '#70AD47', '#FFC000',
    '#5A9BD5', '#F15A29', '#48A463', '#E2A829',
    '#9B59B6', '#1ABC9C', '#E74C3C', '#3498DB',
    '#F39C12', '#2ECC71', '#E91E63', '#00BCD4',
]


def _face_colors(n):
    if n <= len(_BASE_COLORS):
        return _BASE_COLORS[:n]
    cmap = cm.get_cmap('tab20', n)
    return [cmap(i) for i in range(n)]


def _plot_response(d, sweep_dir: pathlib.Path):
    moments        = d['moments']               # (N,)
    dtheta         = d['dtheta']               # (N, n_faces)
    n_faces        = int(d['n_faces'])
    clamped_faces  = list(d['clamped_faces'])
    loaded_faces   = list(d['loaded_faces'])

    colors = _face_colors(n_faces)
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.axhline(0, color='k', linewidth=0.5, linestyle='--', alpha=0.4)

    for fi in range(n_faces):
        vals = dtheta[:, fi]
        if np.all(np.abs(vals) < 0.01):
            continue   # skip faces that don't move
        label = f'F{fi}'
        if fi in clamped_faces:
            label += ' (clamped)'
        elif fi in loaded_faces:
            label += ' (loaded)'
        lw = 2.0 if (fi in clamped_faces or fi in loaded_faces) else 1.2
        ls = '--' if fi in clamped_faces else '-'
        ax.plot(moments * 1e3, vals, color=colors[fi], linewidth=lw,
                linestyle=ls, marker='o', markersize=4, label=label)

    ax.set_xlabel('Applied moment [mN·m]', fontsize=12)
    ax.set_ylabel('Face rotation dθ [°]', fontsize=12)
    ax.set_title(f'SOFA moment sweep — {n_faces}-face RDQK mesh\n'
                 f'clamped: F{clamped_faces}  loaded (moment): F{loaded_faces}',
                 fontsize=11)
    ax.legend(fontsize=8, ncol=2, loc='best')
    ax.grid(True, alpha=0.3)

    out = sweep_dir / 'moment_sweep_response.png'
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved → {out}')
    return out


def _plot_energy(d, sweep_dir: pathlib.Path):
    moments  = d['moments']
    energies = d['energies']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(moments * 1e3, energies * 1e6, 'o-', color='#4472C4', linewidth=2, markersize=5)
    ax.set_xlabel('Applied moment [mN·m]', fontsize=11)
    ax.set_ylabel('Strain energy [μJ]', fontsize=11)
    ax.set_title('Strain energy vs moment', fontsize=10)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    if np.all(moments > 0) and np.all(energies > 0):
        ax.loglog(moments * 1e3, energies * 1e6, 'o-', color='#ED7D31', linewidth=2, markersize=5)
        # Reference slope-2 line (linear elasticity: E ∝ M²)
        m_ref = np.array([moments.min(), moments.max()]) * 1e3
        e_ref = energies[0] * 1e6 * (m_ref / (moments[0] * 1e3)) ** 2
        ax.loglog(m_ref, e_ref, 'k--', linewidth=1, alpha=0.6, label='slope 2 (elastic)')
        ax.legend(fontsize=9)
        ax.set_xlabel('Applied moment [mN·m]', fontsize=11)
        ax.set_ylabel('Strain energy [μJ]', fontsize=11)
        ax.set_title('Log-log: slope≈2 ↔ linear-elastic regime', fontsize=10)
        ax.grid(True, which='both', alpha=0.2)

    out = sweep_dir / 'moment_sweep_energy.png'
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved → {out}')


def _print_table(d):
    moments       = d['moments']
    dtheta        = d['dtheta']
    energies      = d['energies']
    n_faces       = int(d['n_faces'])
    clamped       = list(d['clamped_faces'])
    loaded        = list(d['loaded_faces'])

    moving = [fi for fi in range(n_faces)
              if np.any(np.abs(dtheta[:, fi]) > 0.01)]

    print("\n── Moment sweep summary ─────────────────────────────────────────")
    header = (f"{'M [mN·m]':>10}  {'E [μJ]':>9}  "
              + "  ".join(f"F{fi:2d} dθ[°]" for fi in moving))
    print(header)
    print("─" * len(header))
    for i, M in enumerate(moments):
        row = (f"{M*1e3:10.2f}  {energies[i]*1e6:9.3f}  "
               + "  ".join(f"{dtheta[i, fi]:10.2f}" for fi in moving))
        print(row)
    print("──────────────────────────────────────────────────────────────────")
    print(f"Clamped faces: F{clamped}   Loaded faces: F{loaded}")

    # Linearity check
    if len(moments) >= 2:
        ratios = dtheta[:, loaded[0]] / moments if len(loaded) > 0 else None
        if ratios is not None:
            cv = np.std(ratios) / np.abs(np.mean(ratios))
            print(f"\nLoaded face F{loaded[0]} dθ/M linearity: cv = {cv:.3f} "
                  f"({'linear' if cv < 0.05 else 'nonlinear'})")


def main():
    parser = argparse.ArgumentParser(description='Plot SOFA moment sweep results.')
    parser.add_argument('--sweep-dir', required=True,
                        help='Directory containing sweep_summary.npz')
    args = parser.parse_args()

    sweep_dir  = pathlib.Path(args.sweep_dir)
    summary    = sweep_dir / 'sweep_summary.npz'
    if not summary.exists():
        parser.error(f'Not found: {summary}')

    d = np.load(summary)
    _print_table(d)
    _plot_response(d, sweep_dir)
    _plot_energy(d, sweep_dir)


if __name__ == '__main__':
    main()
