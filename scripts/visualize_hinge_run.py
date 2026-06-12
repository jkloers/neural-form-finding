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
import sys
import types
import pathlib
import argparse

import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as manim
from matplotlib.collections import PolyCollection, LineCollection

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from nff.sofa.mesh_builder_gmsh import build_mesh_gmsh, compute_hinge_geometry

# ── Princeton palette (mirrors nff/utils/visualization.py) ─────────────────────
P_ORANGE = "#F58025"   # free / loaded panels AND hinge mesh fill
P_GRAY   = "#6C757D"   # clamped face
P_EDGE   = "#1A1A1A"   # mesh edges
P_DARK   = "#1A1A1A"
P_BG     = "#FFFFFF"
ARROW_RED = "#D62828"  # positive-moment arrow
GREEN_UP  = "#1B6B3A"  # upper strip (dark Princeton green)
GREEN_LO  = "#5CB87F"  # lower strip (light Princeton green)
CP_COL    = "#16324A"  # Bézier control points / polygons
# Loss colour — peak-stress band (orange, matching the main pipeline positive term)
LOSS_STRESS, LOSS_TOT = '#E07B39', '#111111'

plt.rcParams.update({'font.family': 'serif', 'axes.linewidth': 1.0,
                     'font.size': 11, 'figure.facecolor': P_BG})

_PARAMS = ['gap', 's0_top', 's0_bot', 's1_top', 's1_bot',
           'bcu_x', 'bcu_y', 'bcl_x', 'bcl_y']


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
        'bc_up_xy': [float(conv['bcu_x'][idx]), float(conv['bcu_y'][idx])],
        'bc_lo_xy': [float(conv['bcl_x'][idx]), float(conv['bcl_y'][idx])],
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


def _bez(p0, c, p2, n=200):
    t = np.linspace(0, 1, n)[:, None]
    return (1-t)**2*p0 + 2*(1-t)*t*c + t**2*p2


def _draw_arcs(ax, geo, scale=1000.0, control_points=False):
    """Draw the two quadratic Bézier strips (upper=dark green, lower=light green).

    control_points=True also shows the endpoints + the single interior CP per arc
    with thin control polygons.
    """
    for hd in geo['hinge_data']:
        top_keys = ('p0_top', 'bc_up', 'p1_top')
        bot_keys = ('p0_bot', 'bc_lo', 'p1_bot')
        # Dark green = the visually-upper strip (internal up/lo follows the cell's
        # perpendicular axis, which may sit either way round in world XY).
        top_y = 0.5 * (hd['p0_top'][1] + hd['p1_top'][1])
        bot_y = 0.5 * (hd['p0_bot'][1] + hd['p1_bot'][1])
        order = ([(top_keys, GREEN_UP), (bot_keys, GREEN_LO)] if top_y >= bot_y
                 else [(bot_keys, GREEN_UP), (top_keys, GREEN_LO)])
        for keys, col in order:
            p0, c, p2 = (hd[k] for k in keys)
            arc = _bez(p0, c, p2) * scale
            ax.plot(arc[:, 0], arc[:, 1], '-', color=col, lw=2.8, zorder=9)
            if control_points:
                poly = np.array([p0, c, p2]) * scale
                ax.plot(poly[:, 0], poly[:, 1], '--', color=CP_COL, lw=0.9,
                        alpha=0.55, zorder=10)
                # endpoints (squares) + the single interior control point (circle)
                ax.plot(poly[[0, 2], 0], poly[[0, 2], 1], 's', color=CP_COL,
                        ms=7, zorder=12, markeredgecolor='white', markeredgewidth=1)
                ax.plot(poly[1, 0], poly[1, 1], 'o', color=CP_COL,
                        ms=7, zorder=12, markeredgecolor='white', markeredgewidth=1)


def _noaxis(ax):
    ax.set_aspect('equal')
    ax.axis('off')


# ── Panel 1 — initial state + boundary conditions ──────────────────────────────

def plot_initial_state(run_dir, cs, conv, out):
    g, bp = _bezier_params(conv, 0)
    nodes, tets, bc = build_mesh_gmsh(cs, gap=g, bezier_params=bp, n_z_layers=1)
    tri, _ = _bottom_tris(nodes, tets)
    xy = nodes[:, :2] * 1000

    # Clamped face gray; loaded face + hinge orange (matches main pipeline).
    region = np.where(bc['clamped'], 0, 1)        # 0 = clamped, 1 = active/orange
    fc = [P_GRAY if region[t].mean() < 0.5 else P_ORANGE for t in tri]

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ax.add_collection(PolyCollection(xy[tri], facecolors=fc, alpha=0.95,
                                     edgecolors=P_EDGE, linewidths=0.25, zorder=1))

    fcx = cs.face_centroids * 1000
    cl = sorted({int(r[0]) for r in np.asarray(cs.constrained_face_DOF_pairs)})
    ld = sorted({int(r[0]) for r in np.asarray(cs.loaded_face_DOF_pairs)})
    for f in cl:
        ax.annotate('clamped', fcx[f], ha='center', va='center', fontsize=11,
                    fontweight='bold', color='white', zorder=12)

    # Positive-moment arrow on the loaded face (red curved arrow, main-pipeline style).
    for f in ld:
        cxy = fcx[f]
        r = 0.30 * float(np.ptp(xy[:, 0]))   # arc radius ~ scene scale
        a0, a1 = np.deg2rad(-50), np.deg2rad(140)   # CCW = positive = closing
        start = cxy + r * np.array([np.cos(a0), np.sin(a0)])
        end   = cxy + r * np.array([np.cos(a1), np.sin(a1)])
        ax.add_patch(mpatches.FancyArrowPatch(
            start, end, connectionstyle="arc3,rad=0.42", color=ARROW_RED,
            arrowstyle="Simple,tail_width=1.6,head_width=8,head_length=10", zorder=13))
        ax.annotate('M', cxy, ha='center', va='center', fontsize=12,
                    fontweight='bold', color=ARROW_RED, zorder=14)

    ax.autoscale(); _noaxis(ax)
    ax.set_title('Initial state', fontsize=13, fontweight='bold', color=P_DARK, pad=6)
    fig.tight_layout(); fig.savefig(out, dpi=160, bbox_inches='tight', facecolor=P_BG)
    plt.close(fig); print(f'  → {out.name}')


# ── Panel 2 — training loss (stacked, main-pipeline style) ──────────────────────

def plot_loss(run_dir, conv, out):
    n = len(conv['total_loss'])
    ep = np.arange(1, n + 1)
    total = np.asarray(conv['total_loss'], float)
    best  = int(np.argmin(total))
    # Stacked component contributions (fracture / material / gap).
    comps, labels, cols = [], [], []
    for key, lbl, col in [('loss_fatigue', 'Plastic strain (fatigue)', ARROW_RED),
                          ('loss_mat',     'Material',                 LOSS_STRESS),
                          ('loss_gap',     'Gap penalty',              '#1976D2')]:
        if key in conv.files:
            comps.append(np.asarray(conv[key], float)); labels.append(lbl); cols.append(col)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    if comps:
        ax.stackplot(ep, *comps, colors=cols, labels=labels, alpha=0.8, zorder=2)
    ax.plot(ep, total, color=LOSS_TOT, lw=2.5, label='Total loss', zorder=10)
    ax.axvline(best + 1, color=GREEN_UP, ls=':', lw=1.4, label='best', zorder=8)

    ax.set_xlim(1, max(n, 2)); ax.set_ylim(0, float(total.max()) * 1.12 + 1e-6)
    ax.set_xlabel('Optimizer epoch', fontsize=10, fontweight='bold')
    ax.set_ylabel('Loss contribution', fontsize=10, fontweight='bold')
    ax.set_title('Training loss', fontsize=13, fontweight='bold', color=P_DARK, pad=8)
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
    # Hinge mesh in Princeton orange; flanking panels a faint gray.
    fc = [P_ORANGE if hinge[t].mean() > 0.5 else '#ECECEC' for t in tri]

    fig, ax = plt.subplots(figsize=(7.2, 6.6))
    ax.add_collection(PolyCollection(xy[tri], facecolors=fc, alpha=0.85,
                                     edgecolors='none', zorder=1))
    ax.add_collection(LineCollection(xy[_edges(tri)], colors='white', lw=0.45,
                                     alpha=0.7, zorder=2))
    _draw_arcs(ax, geo, control_points=True)

    hd = geo['hinge_data'][0]
    cx, cy = hd['corner'] * 1000
    pad = max(g * 1000 * 3.2, 7.0)
    ax.set_xlim(cx - pad, cx + pad); ax.set_ylim(cy - pad, cy + pad)
    _noaxis(ax)
    ax.set_title('Optimal hinge', fontsize=13, fontweight='bold', color=P_DARK, pad=6)
    ax.legend(handles=[
        plt.Line2D([0], [0], color=GREEN_UP, lw=2.8, label='upper strip'),
        plt.Line2D([0], [0], color=GREEN_LO, lw=2.8, label='lower strip'),
        plt.Line2D([0], [0], color=CP_COL, lw=0, marker='o', ms=6, label='control points')],
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

    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    pc = PolyCollection(xy[tri], array=vm_mpa, cmap='magma',
                        edgecolors='none', zorder=1)
    ax.add_collection(pc)
    ax.add_collection(LineCollection(xy[_edges(tri)], colors='white', lw=0.2,
                                     alpha=0.35, zorder=2))
    cb = fig.colorbar(pc, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label('von Mises stress [MPa]', fontsize=9)
    cb.ax.tick_params(labelsize=8)

    # Zoom onto the stressed hinge region (where the field is meaningful).
    hi = vm_mpa > 0.05 * vm_mpa.max()
    pts = xy[tri[hi]].reshape(-1, 2) if hi.any() else xy[tri].reshape(-1, 2)
    c = 0.5 * (pts.min(0) + pts.max(0))
    half = 0.6 * float(np.ptp(pts, axis=0).max()) + 3.0
    ax.set_xlim(c[0] - half, c[0] + half); ax.set_ylim(c[1] - half, c[1] + half)
    _noaxis(ax)
    ax.set_title('Final state — von Mises stress', fontsize=13, fontweight='bold',
                 color=P_DARK, pad=6)
    fig.tight_layout(); fig.savefig(out, dpi=160, bbox_inches='tight', facecolor=P_BG)
    plt.close(fig); print(f'  → {out.name}')


# ── Training animation — Bézier curves learning + loss ──────────────────────────

def animate_training(run_dir, cs, conv, out, fps=3):
    """Animate the hinge morphing over epochs (left) alongside the stacked loss (right).

    Uses convergence.npz only (per-epoch params) — no simulation. Mirrors the main
    pipeline's training animation: geometry learns on the left, loss accrues on the right.
    """
    n = len(conv['total_loss'])
    if n < 2:
        print('  (need ≥2 epochs for the training animation — skipping)')
        return

    # Pre-build per-epoch geometry (rebuilding gmsh inside the anim loop is slow/fragile).
    frames = []
    for e in range(n):
        g, bp = _bezier_params(conv, e)
        nodes, tets, bc = build_mesh_gmsh(cs, gap=g, bezier_params=bp, n_z_layers=1)
        geo = compute_hinge_geometry(cs, gap=g, bezier_params=bp)
        tri, _ = _bottom_tris(nodes, tets)
        hinge = ~bc['clamped'] & ~bc['loaded']
        frames.append((nodes[:, :2] * 1000, tri, hinge, geo))
    # Fixed, hinge-centred view across all frames so it doesn't jump. Size it from
    # the HINGE tris only (not the big flanking faces).
    cx, cy = frames[0][3]['hinge_data'][0]['corner'] * 1000
    span = 10.0
    for xy, tri, hinge, _ in frames:
        htris = tri[hinge[tri].mean(axis=1) > 0.5]
        if len(htris):
            pts = xy[htris].reshape(-1, 2)
            span = max(span, float(np.abs(pts - [cx, cy]).max()))
    span += 5.0

    # Loss components for the right panel.
    ep = np.arange(1, n + 1)
    total = np.asarray(conv['total_loss'], float)
    comps, cols, labels = [], [], []
    for key, lbl, col in [('loss_fatigue', 'Plastic strain (fatigue)', ARROW_RED),
                          ('loss_mat', 'Material', LOSS_STRESS),
                          ('loss_gap', 'Gap penalty', '#1976D2')]:
        if key in conv.files:
            comps.append(np.asarray(conv[key], float)); cols.append(col); labels.append(lbl)

    fig, (axh, axl) = plt.subplots(1, 2, figsize=(13.5, 6), facecolor=P_BG,
                                   gridspec_kw={'width_ratios': [1.05, 1.25], 'wspace': 0.18})
    fig.suptitle('Hinge optimization — learning the Bézier flexure',
                 fontsize=13, fontweight='bold', color=P_DARK, y=0.98)
    # Static loss panel (drawn once); a marker sweeps across it.
    if comps:
        axl.stackplot(ep, *comps, colors=cols, labels=labels, alpha=0.8, zorder=2)
    axl.plot(ep, total, color=LOSS_TOT, lw=2.2, label='Total loss', zorder=10)
    axl.set_xlim(1, max(n, 2)); axl.set_ylim(0, float(total.max()) * 1.12 + 1e-6)
    axl.set_xlabel('Optimizer epoch', fontsize=10, fontweight='bold')
    axl.set_ylabel('Loss contribution', fontsize=10, fontweight='bold')
    for s in ('top', 'right'):
        axl.spines[s].set_visible(False)
    axl.grid(True, ls=':', alpha=0.3)
    axl.legend(loc='upper right', fontsize=8.5, frameon=True, edgecolor='#DDDDDD')
    vline = axl.axvline(1, color=GREEN_UP, lw=1.6, zorder=11)
    vdot, = axl.plot([1], [total[0]], 'o', color=GREEN_UP, ms=8, zorder=12)

    def update(e):
        xy, tri, hinge, geo = frames[e]
        axh.clear()
        fc = [P_ORANGE if hinge[t].mean() > 0.5 else '#ECECEC' for t in tri]
        axh.add_collection(PolyCollection(xy[tri], facecolors=fc, alpha=0.85,
                                          edgecolors='none', zorder=1))
        axh.add_collection(LineCollection(xy[_edges(tri)], colors='white', lw=0.4,
                                          alpha=0.7, zorder=2))
        _draw_arcs(axh, geo, control_points=True)
        axh.set_xlim(cx - span, cx + span); axh.set_ylim(cy - span, cy + span)
        _noaxis(axh)
        axh.set_title('Hinge shape', fontsize=11, fontweight='bold', color=P_DARK, pad=4)
        axh.text(0.03, 0.04, f'epoch {e+1}/{n}', transform=axh.transAxes,
                 fontsize=10, color=P_DARK, fontweight='bold')
        vline.set_xdata([e + 1, e + 1]); vdot.set_data([e + 1], [total[e]])
        return ()

    anim = manim.FuncAnimation(fig, update, frames=n, blit=False)
    anim.save(str(out), writer=manim.PillowWriter(fps=fps), dpi=110)
    plt.close(fig); print(f'  → {out.name}')


# ── Entry point ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='Clean Princeton-themed hinge-run visuals.')
    ap.add_argument('run_dir', nargs='?', default=None,
                    help='hinge_opt run directory (default: latest).')
    ap.add_argument('--animate', action='store_true',
                    help='Also render training.gif (hinge morph + loss).')
    args = ap.parse_args()

    run_dir = pathlib.Path(args.run_dir) if args.run_dir else _latest_run_dir()
    print(f'Visualizing {run_dir.name}')

    conv = np.load(run_dir / 'convergence.npz')
    best = int(np.argmin(conv['total_loss']))

    fs_path = run_dir / 'final_state.npz'
    fs = np.load(fs_path) if fs_path.exists() else None
    if fs is not None:
        cs = _cs_from_final(fs)
        plot_final_hinge(run_dir, cs, conv, best, run_dir / 'final_hinge.png')
        plot_von_mises(run_dir, fs, run_dir / 'von_mises.png')
        if args.animate:
            animate_training(run_dir, cs, conv, run_dir / 'training.gif')
    else:
        print('  (no final_state.npz — initial/hinge/von_mises panels need it; '
              'run the optimizer with the field-returning oracle)')
    plot_loss(run_dir, conv, run_dir / 'loss.png')
    print('Done.')


if __name__ == '__main__':
    main()
