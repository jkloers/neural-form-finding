#!/usr/bin/env python3
"""
scripts/visualize_hinge_run.py — clean Princeton-themed hinge-run visuals.

Produces four focused PNGs from a hinge_opt run directory:
  1. initial_state.png — undeformed initial design + boundary conditions
  2. loss.png          — training loss (stacked component contributions)
  3. final_hinge.png   — optimal hinge close-up with the two Bézier arcs
  4. von_mises.png     — deformed tessellation coloured by von Mises stress

Panels 1 & 3 rebuild the mesh locally (gmsh) from the optimised parameters.
Panel 4 uses the deformed nodes + per-tet field saved by the optimizer in
final_state.npz (the Tesseract oracle returns it at the best design).

Usage:
    python scripts/visualize_hinge_run.py [run_dir]      # default: latest run
"""
import os
import sys
import types
import pathlib
import argparse

import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PolyCollection, LineCollection

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from nff.sofa.mesh_builder_gmsh import build_mesh_gmsh, compute_hinge_geometry

# ── Princeton palette ──────────────────────────────────────────────────────────
P_ORANGE = "#F58025"   # free panels
P_CLAMP  = "#2B2B2B"   # clamped face
P_GREEN  = "#27AE60"   # loaded face
P_EDGE   = "#1A1A1A"   # mesh edges
P_DARK   = "#1A1A1A"
P_BG     = "#FFFFFF"
P_HINGE  = "#F4C542"   # hinge strip fill
ARC_UP   = "#E8743B"   # upper Bézier arc
ARC_LO   = "#7E57C2"   # lower Bézier arc
# Loss-component colours — identical to the main pipeline (training_animation.py)
LOSS_ROT, LOSS_SH, LOSS_TEN, LOSS_TOT = '#E07B39', '#D32F2F', '#1976D2', '#111111'

plt.rcParams.update({'font.family': 'serif', 'axes.linewidth': 1.0,
                     'font.size': 11, 'figure.facecolor': P_BG})

_PARAMS = ['gap', 's0_top', 's0_bot', 's1_top', 's1_bot',
           'bc1u_x', 'bc1u_y', 'bc2u_x', 'bc2u_y',
           'bc1l_x', 'bc1l_y', 'bc2l_x', 'bc2l_y']


# ── run-dir + data loading ──────────────────────────────────────────────────────

def _latest_run_dir() -> pathlib.Path:
    base = pathlib.Path(__file__).parent.parent / 'data' / 'outputs' / 'hinge_opt'
    runs = sorted(d for d in (base.iterdir() if base.exists() else [])
                  if (d / 'convergence.npz').exists())
    if not runs:
        sys.exit(f'No hinge_opt runs with convergence.npz under {base}')
    return runs[-1]


def _cs_from_final(fs) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        face_centroids             = fs['face_centroids'],
        centroid_node_vectors      = fs['centroid_node_vectors'],
        hinge_node_pairs           = fs['hinge_node_pairs'],
        hinge_adj_info             = fs['hinge_adj_info'],
        constrained_face_DOF_pairs = fs['constrained_face_DOF_pairs'],
        loaded_face_DOF_pairs      = fs['loaded_face_DOF_pairs'],
    )


def _bezier_params(conv, idx) -> tuple:
    g  = float(conv['gap'][idx])
    bp = {
        's0_top': float(conv['s0_top'][idx]), 's0_bot': float(conv['s0_bot'][idx]),
        's1_top': float(conv['s1_top'][idx]), 's1_bot': float(conv['s1_bot'][idx]),
        'bc1_up_xy': [float(conv['bc1u_x'][idx]), float(conv['bc1u_y'][idx])],
        'bc2_up_xy': [float(conv['bc2u_x'][idx]), float(conv['bc2u_y'][idx])],
        'bc1_lo_xy': [float(conv['bc1l_x'][idx]), float(conv['bc1l_y'][idx])],
        'bc2_lo_xy': [float(conv['bc2l_x'][idx]), float(conv['bc2l_y'][idx])],
    }
    return g, bp


# ── mesh-rendering helpers ──────────────────────────────────────────────────────

def _bottom_tris(nodes, tets):
    """Boundary triangles on the bottom (z≈min) layer + their owning tet index."""
    faces = {}
    for ti, t in enumerate(tets):
        for c in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)):
            f = tuple(sorted((int(t[c[0]]), int(t[c[1]]), int(t[c[2]]))))
            faces.setdefault(f, []).append(ti)
    zmin = nodes[:, 2].min()
    bot = nodes[:, 2] < zmin + 1e-6
    tri, owner = [], []
    for f, tis in faces.items():
        if len(tis) == 1 and bot[f[0]] and bot[f[1]] and bot[f[2]]:
            tri.append(f); owner.append(tis[0])
    return np.array(tri), np.array(owner)


def _edges(tri):
    e = np.sort(np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [0, 2]]]), axis=1)
    return np.unique(e, axis=0)


def _bez(p0, c1, c2, p3, n=200):
    t = np.linspace(0, 1, n)[:, None]
    return (1-t)**3*p0 + 3*(1-t)**2*t*c1 + 3*(1-t)*t**2*c2 + t**3*p3


def _draw_arcs(ax, geo, scale=1000.0, anchors=True):
    for hd in geo['hinge_data']:
        up = _bez(hd['p0_top'], hd['bc1_up'], hd['bc2_up'], hd['p1_top']) * scale
        lo = _bez(hd['p0_bot'], hd['bc1_lo'], hd['bc2_lo'], hd['p1_bot']) * scale
        ax.plot(up[:, 0], up[:, 1], '-', color=ARC_UP, lw=2.5, zorder=9)
        ax.plot(lo[:, 0], lo[:, 1], '-', color=ARC_LO, lw=2.5, zorder=9)
        if anchors:
            for key, col in [('p0_top', ARC_UP), ('p1_top', ARC_UP),
                             ('p0_bot', ARC_LO), ('p1_bot', ARC_LO)]:
                ax.plot(*(hd[key] * scale), 's', color=col, ms=7, zorder=11,
                        markeredgecolor='white', markeredgewidth=1)


def _clean(ax):
    ax.set_aspect('equal')
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.spines['left'].set_color('#BBBBBB')
    ax.spines['bottom'].set_color('#BBBBBB')
    ax.tick_params(labelsize=8, colors=P_DARK)
    ax.set_xlabel('x [mm]', fontsize=9); ax.set_ylabel('y [mm]', fontsize=9)


# ── Panel 1 — initial state + boundary conditions ──────────────────────────────

def plot_initial_state(run_dir, cs, conv, out):
    g, bp = _bezier_params(conv, 0)
    nodes, tets, bc = build_mesh_gmsh(cs, gap=g, bezier_params=bp, n_z_layers=1)
    geo = compute_hinge_geometry(cs, gap=g, bezier_params=bp)
    tri, _ = _bottom_tris(nodes, tets)
    xy = nodes[:, :2] * 1000

    region = np.where(bc['clamped'], 0, np.where(bc['loaded'], 2, 1))
    rc = {0: P_CLAMP, 1: P_ORANGE, 2: P_GREEN}
    fc = [rc[int(round(region[t].mean()))] for t in tri]

    fig, ax = plt.subplots(figsize=(8, 6.2))
    ax.add_collection(PolyCollection(xy[tri], facecolors=fc, alpha=0.78,
                                     edgecolors='none', zorder=1))
    ax.add_collection(LineCollection(xy[_edges(tri)], colors='white', lw=0.35,
                                     alpha=0.55, zorder=2))
    _draw_arcs(ax, geo, anchors=False)

    # Boundary-condition labels
    fcx = cs.face_centroids * 1000
    cl = sorted({int(r[0]) for r in np.asarray(cs.constrained_face_DOF_pairs)})
    ld = sorted({int(r[0]) for r in np.asarray(cs.loaded_face_DOF_pairs)})
    for f in cl:
        ax.annotate('CLAMPED', fcx[f], ha='center', va='center', fontsize=11,
                    fontweight='bold', color='white', zorder=12)
    for f in ld:
        ax.annotate('LOADED', fcx[f], ha='center', va='center', fontsize=11,
                    fontweight='bold', color='white', zorder=12)
        ax.annotate('rotation', (fcx[f][0], fcx[f][1] - 9), ha='center', va='center',
                    fontsize=8.5, style='italic', color='white', zorder=12)

    ax.autoscale(); _clean(ax)
    ax.set_title(f'Initial design — boundary conditions   (gap = {g*1e3:.1f} mm)',
                 fontsize=12, fontweight='bold', color=P_DARK, pad=8)
    ax.legend(handles=[mpatches.Patch(color=P_CLAMP, label='clamped face'),
                       mpatches.Patch(color=P_GREEN, label='loaded face'),
                       mpatches.Patch(color=P_ORANGE, label='free panel')],
              loc='upper right', fontsize=8.5, frameon=True, edgecolor='#DDDDDD')
    fig.tight_layout(); fig.savefig(out, dpi=160, bbox_inches='tight', facecolor=P_BG)
    plt.close(fig); print(f'  → {out.name}')


# ── Panel 2 — training loss (stacked, main-pipeline style) ──────────────────────

def plot_loss(run_dir, conv, out):
    n = len(conv['total_loss'])
    ep = np.arange(1, n + 1)
    e_rot = np.asarray(conv['energy_rot'], float)
    e_sh  = np.asarray(conv['energy_shear'], float)
    e_ten = np.asarray(conv['energy_tension'], float)
    total = np.asarray(conv['total_loss'], float)
    best  = int(np.argmin(conv['max_vm_rot']))

    def active(a): return bool(np.any(np.abs(a) > 1e-12))
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.stackplot(ep, e_rot, colors=[LOSS_ROT], labels=['Rotation energy  (min)'],
                 alpha=0.75, zorder=2)
    cum = np.zeros(n)
    for arr, lbl, col in [(-e_sh, 'Shear stiffness  (max ↓)', LOSS_SH),
                          (-e_ten, 'Tension stiffness  (max ↓)', LOSS_TEN)]:
        if active(arr):
            ax.fill_between(ep, cum, cum + arr, color=col, alpha=0.55, label=lbl, zorder=2)
            cum = cum + arr
    ax.plot(ep, total, color=LOSS_TOT, lw=2.5, label='Total loss', zorder=10)
    ax.axhline(0, color='#333333', lw=0.8, alpha=0.4)
    ax.axvline(best + 1, color=P_ORANGE, ls=':', lw=1.4, label=f'best (ep {best+1})', zorder=8)

    ax.set_xlim(1, max(n, 2))
    ax.set_xlabel('Optimizer epoch', fontsize=10, fontweight='bold')
    ax.set_ylabel('Weighted loss contribution', fontsize=10, fontweight='bold')
    ax.set_title('Training loss — component contributions', fontsize=12,
                 fontweight='bold', color=P_DARK, pad=8)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.grid(True, ls=':', alpha=0.3)
    ax.legend(loc='upper right', fontsize=8.5, frameon=True, edgecolor='#DDDDDD', ncol=1)
    fig.tight_layout(); fig.savefig(out, dpi=160, bbox_inches='tight', facecolor=P_BG)
    plt.close(fig); print(f'  → {out.name}')


# ── Panel 3 — optimal hinge close-up ───────────────────────────────────────────

def plot_final_hinge(run_dir, cs, conv, best, out):
    g, bp = _bezier_params(conv, best)
    nodes, tets, bc = build_mesh_gmsh(cs, gap=g, bezier_params=bp, n_z_layers=1)
    geo = compute_hinge_geometry(cs, gap=g, bezier_params=bp)
    tri, _ = _bottom_tris(nodes, tets)
    xy = nodes[:, :2] * 1000
    hinge = ~bc['clamped'] & ~bc['loaded']
    fc = [P_HINGE if hinge[t].mean() > 0.5 else '#E8E8E8' for t in tri]

    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    ax.add_collection(PolyCollection(xy[tri], facecolors=fc, alpha=0.55,
                                     edgecolors='none', zorder=1))
    ax.add_collection(LineCollection(xy[_edges(tri)], colors='#9AA7B0', lw=0.5,
                                     alpha=0.8, zorder=2))
    _draw_arcs(ax, geo, anchors=True)

    hd = geo['hinge_data'][0]
    cx, cy = hd['corner'] * 1000
    pad = max(g * 1000 * 3.5, 8.0)
    ax.set_xlim(cx - pad, cx + pad); ax.set_ylim(cy - pad, cy + pad)
    _clean(ax)
    ax.set_title(f'Optimal hinge — Bézier arcs   (epoch {best+1}, gap = {g*1e3:.2f} mm)',
                 fontsize=12, fontweight='bold', color=P_DARK, pad=8)
    # Label arcs by their actual vertical position (internal up/lo follows the
    # cell's perpendicular axis, which may sit either way round in world XY).
    top_y = 0.5 * (hd['p0_top'][1] + hd['p1_top'][1])
    bot_y = 0.5 * (hd['p0_bot'][1] + hd['p1_bot'][1])
    up_lbl, lo_lbl = (('upper arc', 'lower arc') if top_y >= bot_y
                      else ('lower arc', 'upper arc'))
    ax.legend(handles=[plt.Line2D([0], [0], color=ARC_UP, lw=2.5, label=up_lbl),
                       plt.Line2D([0], [0], color=ARC_LO, lw=2.5, label=lo_lbl)],
              loc='upper right', fontsize=8.5, frameon=True, edgecolor='#DDDDDD')
    fig.tight_layout(); fig.savefig(out, dpi=160, bbox_inches='tight', facecolor=P_BG)
    plt.close(fig); print(f'  → {out.name}')


# ── Panel 4 — von Mises over the deformed tessellation ─────────────────────────

def plot_von_mises(run_dir, fs, out):
    nodes = np.asarray(fs['deformed_nodes'], float)
    tets  = np.asarray(fs['mesh_tets'], int)
    vm    = np.asarray(fs['von_mises_field'], float)
    if nodes.size == 0 or tets.size == 0:
        print('  (no field in final_state.npz — skipping von_mises.png)')
        return
    tri, owner = _bottom_tris(nodes, tets)
    xy = nodes[:, :2] * 1000
    vm_mpa = vm[owner] / 1e6

    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    pc = PolyCollection(xy[tri], array=vm_mpa, cmap='magma',
                        edgecolors='none', zorder=1)
    ax.add_collection(pc)
    ax.add_collection(LineCollection(xy[_edges(tri)], colors='white', lw=0.18,
                                     alpha=0.3, zorder=2))
    cb = fig.colorbar(pc, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label('von Mises stress [MPa]', fontsize=9)
    cb.ax.tick_params(labelsize=8)

    ax.autoscale(); _clean(ax)
    ax.set_title(f'Final state — von Mises stress   (σ_max = {vm_mpa.max():.0f} MPa)',
                 fontsize=12, fontweight='bold', color=P_DARK, pad=8)
    fig.tight_layout(); fig.savefig(out, dpi=160, bbox_inches='tight', facecolor=P_BG)
    plt.close(fig); print(f'  → {out.name}')


# ── Entry point ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='Clean Princeton-themed hinge-run visuals.')
    ap.add_argument('run_dir', nargs='?', default=None,
                    help='hinge_opt run directory (default: latest).')
    args = ap.parse_args()

    run_dir = pathlib.Path(args.run_dir) if args.run_dir else _latest_run_dir()
    print(f'Visualizing {run_dir.name}')

    conv = np.load(run_dir / 'convergence.npz')
    best = int(np.argmin(conv['max_vm_rot']))

    fs_path = run_dir / 'final_state.npz'
    fs = np.load(fs_path) if fs_path.exists() else None
    if fs is not None:
        cs = _cs_from_final(fs)
        plot_initial_state(run_dir, cs, conv, run_dir / 'initial_state.png')
        plot_final_hinge(run_dir, cs, conv, best, run_dir / 'final_hinge.png')
        plot_von_mises(run_dir, fs, run_dir / 'von_mises.png')
    else:
        print('  (no final_state.npz — initial/hinge/von_mises panels need it; '
              'run the optimizer with the field-returning oracle)')
    plot_loss(run_dir, conv, run_dir / 'loss.png')
    print('Done.')


if __name__ == '__main__':
    main()
