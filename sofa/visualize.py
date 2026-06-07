"""
sofa/visualize.py — Render SOFA kirigami simulation results from .npz.

NO Sofa import — runs in kgnn_mac conda env (matplotlib only).
matplotlib.use('Agg') is set before any other import to prevent Qt crash.

Layout (two-panel "opening" view):
  Left  : Natural flat state — top-down XY plan showing kirigami cut pattern
          (void cuts shown in grey, faces colored, hinge strips highlighted)
  Right : Deformed state — top-down XY view showing in-plane mechanism activation
          (face polygons projected onto XY; flat reference wireframe as dashes;
          void gaps open/close as faces rotate in-plane)

Usage:
    conda run -n kgnn_mac python sofa/visualize.py --npz sofa/output/sofa_result.npz --save
"""

import matplotlib
matplotlib.use('Agg')

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ── Visual style ───────────────────────────────────────────────────────────────
BG       = '#FAFAFA'
C_ORANGE = '#E07020'
C_BLUE   = '#2B6CB0'
C_GREEN  = '#276749'
C_RED    = '#C53030'
C_GREY   = '#718096'
C_VOID   = '#CBD5E0'   # kirigami cut regions
C_HINGE  = '#A0AEC0'  # hinge strips

FACE_KEYS   = ['f0', 'f1', 'f2', 'f3']
FACE_LABELS = ['F0 (clamped)', 'F1 (driven)', 'F2 (free)', 'F3 (free)']
FACE_COLORS = [C_ORANGE, C_BLUE, C_GREEN, C_RED]

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'figure.facecolor': BG,
    'axes.facecolor': BG,
    'text.color': '#1A202C',
})


# ── Mesh helpers ───────────────────────────────────────────────────────────────

_HEX_FACE_LOCAL = [
    [0, 1, 2, 3],  # bottom
    [4, 5, 6, 7],  # top
    [0, 1, 5, 4],  # front
    [1, 2, 6, 5],  # right
    [2, 3, 7, 6],  # back
    [3, 0, 4, 7],  # left
]


def _boundary_quads(hexes):
    """Return (M,4) boundary quad node indices and (M,) owner hex indices."""
    from collections import defaultdict
    face_owner = defaultdict(list)
    for h_idx, h in enumerate(hexes):
        for local in _HEX_FACE_LOCAL:
            key = tuple(sorted(h[n] for n in local))
            face_owner[key].append((h_idx, local))
    quads, owners = [], []
    for key, lst in face_owner.items():
        if len(lst) == 1:
            h_idx, local = lst[0]
            quads.append([hexes[h_idx][n] for n in local])
            owners.append(h_idx)
    return np.array(quads, dtype=np.int64), np.array(owners, dtype=np.int64)


def _node_face_labels(nodes, bc_masks):
    """Per-node label: 0=F0 1=F1 2=F2 3=F3 -1=hinge."""
    lab = np.full(len(nodes), -1, dtype=int)
    for i, k in enumerate(FACE_KEYS):
        lab[bc_masks[k]] = i
    return lab


def _quad_hex_labels(hexes, node_labels):
    """Per-hex label from the first top-node."""
    return node_labels[hexes[:, 4]]


def load_npz(path):
    d = np.load(path, allow_pickle=False)
    bc = {k: d[f'{k}_mask'] for k in FACE_KEYS}
    params = {k: float(d[k]) if k in d else 0.0 for k in (
        'hinge_arm_width', 'hinge_fold_length', 'applied_displacement',
        'applied_moment', 'face_size', 'sheet_thickness',
        'young_modulus', 'poisson_ratio', 'yield_strength')}
    qois = {
        'strain_energy':        float(d['strain_energy']),
        'max_von_mises_stress': float(d['max_von_mises_stress']),
        'max_xy_displacement':  float(d['max_xy_displacement']) if 'max_xy_displacement' in d else 0.0,
        'max_z_displacement':   float(d['max_z_displacement']),
        'first_yield_fraction': float(d['first_yield_fraction']),
    }
    return {
        'nodes_nat':  d['nodes_nat'],
        'nodes_cur':  d['nodes_cur'],
        'hexes':      d['hexes'],
        'bc_masks':   bc,
        'params':     params,
        'qois':       qois,
        'vm_per_hex': d['vm_per_hex'] if 'vm_per_hex' in d else None,
        'is_moment_mode':     bool(d['is_moment_mode'])     if 'is_moment_mode'     in d else False,
        'rotation_angle_deg': float(d['rotation_angle_deg']) if 'rotation_angle_deg' in d else 0.0,
    }


# ── Natural-state flat pattern panel ──────────────────────────────────────────

def _draw_flat_pattern(ax, nodes_nat, hexes, bc_masks, params):
    """Left panel: top-down view of the natural flat kirigami cut pattern."""
    a = params['face_size']
    w = params['hinge_arm_width']
    L = params['hinge_fold_length']
    total = 2 * a + w

    ax.set_aspect('equal')
    ax.set_xlim(-0.01 * total, 1.01 * total)
    ax.set_ylim(-0.01 * total, 1.01 * total)
    ax.set_title('Natural flat state\n(kirigami cut pattern)', fontsize=11, pad=8)
    ax.set_xlabel('x [m]', fontsize=9)
    ax.set_ylabel('y [m]', fontsize=9)
    ax.tick_params(labelsize=8)

    # Draw full bounding sheet in void color (the base material)
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((0, 0), total, total,
                            facecolor=C_VOID, edgecolor='none', zorder=0))

    # Draw face panels (solid colored)
    face_rects = [
        (0,       0,       a,     a,     C_ORANGE),  # F0
        (a + w,   0,       a,     a,     C_BLUE),    # F1
        (a + w,   a + w,   a,     a,     C_GREEN),   # F2
        (0,       a + w,   a,     a,     C_RED),     # F3
    ]
    for x0, y0, dx, dy, c in face_rects:
        ax.add_patch(Rectangle((x0, y0), dx, dy,
                               facecolor=c, alpha=0.75, edgecolor='#333', linewidth=0.8, zorder=2))

    # Draw hinge strips (slightly lighter than faces)
    hinge_rects = [
        (a,       0,       w,     L,     C_HINGE, 'H0'),          # H0
        (a + w,   a,       L,     w,     C_HINGE, 'H1'),          # H1
        (a,       total-L, w,     L,     C_HINGE, 'H2'),          # H2
        (a - L,   a,       L,     w,     C_HINGE, 'H3'),          # H3
    ]
    for x0, y0, dx, dy, c, lbl in hinge_rects:
        ax.add_patch(Rectangle((x0, y0), dx, dy,
                               facecolor=c, edgecolor='#555', linewidth=0.6, zorder=3))
        ax.text(x0 + dx / 2, y0 + dy / 2, lbl,
                ha='center', va='center', fontsize=7, color='#333', zorder=4)

    # Void regions (kirigami cuts) — hatched in a neutral tone
    # Vertical gap (minus hinge strips): x∈[a,a+w], y∈[L, total-L]
    void_vertical = (a, L, w, total - 2 * L)
    # Horizontal gap (minus hinge strips): y∈[a,a+w], x∈[L, total-L]
    void_h_left   = (0, a, a - L, w)              # left of H3
    void_h_right  = (a + w + L, a, a - L, w)     # right of H1
    # Corner void (center): x∈[a,a+w], y∈[a,a+w]
    void_center   = (a, a, w, w)

    for x0, y0, dx, dy in (void_vertical, void_h_left, void_h_right, void_center):
        if dx > 0 and dy > 0:
            ax.add_patch(Rectangle((x0, y0), dx, dy,
                                   facecolor='white', edgecolor='#888',
                                   linewidth=0.5, hatch='////', zorder=1))

    # Face labels
    cx = [(a/2, a/2), (a+w+a/2, a/2), (a+w+a/2, a+w+a/2), (a/2, a+w+a/2)]
    for (px, py), lbl, c in zip(cx, FACE_LABELS, FACE_COLORS):
        ax.text(px, py, lbl, ha='center', va='center',
                fontsize=8.5, color='white', fontweight='bold', zorder=5,
                path_effects=[pe.withStroke(linewidth=2, foreground=c)])

    # Loading indicator: curved arrow showing in-plane CCW rotation of F1
    import matplotlib.patches as mpa
    arc = mpa.Arc((a + w + a * 0.5, a * 0.5), a * 0.3, a * 0.3,
                  angle=0, theta1=20, theta2=160, color=C_BLUE, lw=2.0, zorder=6)
    ax.add_patch(arc)
    ax.annotate('', xy=(a + w + a * 0.37, a * 0.5 + a * 0.15),
                xytext=(a + w + a * 0.37, a * 0.5 + a * 0.11),
                arrowprops=dict(arrowstyle='->', color=C_BLUE, lw=1.5), zorder=6)
    ax.text(a + w + a * 0.5, a * 0.75, 'M (in-plane)',
            ha='center', fontsize=7, color=C_BLUE, zorder=6)

    # Fixed indicator on F0
    ax.annotate('', xy=(a * 0.2, a * 0.2),
                xytext=(a * 0.2, a * 0.5),
                arrowprops=dict(arrowstyle='-|>', color='#333', lw=1.2), zorder=6)
    ax.text(a * 0.2, a * 0.15, 'fixed', ha='center',
            fontsize=7, color='#333', zorder=6)

    # Scale bar
    bar_len = a
    ax.plot([0.02 * total, 0.02 * total + bar_len], [1.03 * total, 1.03 * total],
            'k-', lw=2, clip_on=False)
    ax.text(0.02 * total + bar_len / 2, 1.05 * total,
            f'a = {a*1e3:.0f} mm', ha='center', fontsize=8, clip_on=False)


# ── Shared helpers for field panels ───────────────────────────────────────────

def _top_quads(hexes):
    """Return (H,4) top-face node indices for each hex (z-max face [4,5,6,7])."""
    return hexes[:, [4, 5, 6, 7]]


def _nat_outline(ax, params):
    """Draw natural-state face outlines as dashed grey reference."""
    a, w = params['face_size'], params['hinge_arm_width']
    total = 2 * a + w
    for x0, y0, x1, y1 in [
        (0,     0,     a,     a),
        (a + w, 0,     total, a),
        (a + w, a + w, total, total),
        (0,     a + w, a,     total),
    ]:
        ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0],
                '--', color='#888', lw=0.7, alpha=0.45, zorder=1)


def _field_panel(ax, nodes_nat, nodes_cur, hexes, field_per_hex,
                 cmap_name, vmin, vmax, title, cbar_label, params):
    """
    Top-down XY panel coloring each hex top-face by a scalar field value.

    field_per_hex : (H,) float array — one value per hex element.
    """
    cmap   = plt.get_cmap(cmap_name)
    norm   = plt.Normalize(vmin=vmin, vmax=vmax)
    top_qi = _top_quads(hexes)       # (H, 4)

    ax.set_aspect('equal')
    ax.set_xlabel('x [m]', fontsize=9)
    ax.set_ylabel('y [m]', fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_title(title, fontsize=11, pad=8)

    _nat_outline(ax, params)

    colors = cmap(norm(field_per_hex))   # (H, 4) RGBA
    quads_xy = nodes_cur[top_qi][:, :, :2]   # (H, 4, 2)
    coll = PolyCollection(quads_xy, facecolors=colors,
                          linewidths=0.15, edgecolors='none', zorder=2)
    ax.add_collection(coll)

    xd = nodes_cur[:, 0]
    yd = nodes_cur[:, 1]
    a, w = params['face_size'], params['hinge_arm_width']
    pad = 0.06 * (2 * a + w)
    ax.set_xlim(xd.min() - pad, xd.max() + pad)
    ax.set_ylim(yd.min() - pad, yd.max() + pad)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, shrink=0.75, pad=0.02,
                 label=cbar_label, format='%.2g')


# ── Main figure ───────────────────────────────────────────────────────────────

def make_figure(data, save_path=None):
    nn  = data['nodes_nat']
    nc  = data['nodes_cur']
    hx  = data['hexes']
    bc  = data['bc_masks']
    p   = data['params']
    q   = data['qois']
    is_moment   = data.get('is_moment_mode', False)
    angle_deg   = data.get('rotation_angle_deg', 0.0)

    load_label = (f"M = {p['applied_moment']*1e3:.1f} mN·m  (moment-ctrl)"
                  if is_moment else
                  f"theta = {angle_deg:.0f} deg  (rotation-ctrl)")

    fig = plt.figure(figsize=(22, 8), facecolor=BG)
    fig.suptitle(
        f"RDQK kirigami unit cell   |   {load_label}   |   "
        f"a = {p['face_size']*1e3:.0f} mm   "
        f"w = {p['hinge_arm_width']*1e3:.1f} mm   "
        f"L = {p['hinge_fold_length']*1e3:.1f} mm   "
        f"t = {p['sheet_thickness']*1e3:.1f} mm",
        fontsize=11, y=0.99,
    )

    vm_per_hex = data.get('vm_per_hex')

    # Grid: 3 panels — natural | z-displacement | stress
    gs = fig.add_gridspec(1, 3, left=0.04, right=0.97, top=0.91, bottom=0.09,
                          wspace=0.28, width_ratios=[1, 1, 1])

    # Left: flat natural pattern (reference)
    ax_flat = fig.add_subplot(gs[0])
    _draw_flat_pattern(ax_flat, nn, hx, bc, p)

    # Centre: out-of-plane z-displacement field
    z_disp_per_node = nc[:, 2] - nn[:, 2]           # signed z-change per node
    top_qi = _top_quads(hx)
    z_disp_per_hex = z_disp_per_node[top_qi].mean(axis=1)   # (H,)
    z_abs_max = np.abs(z_disp_per_hex).max() or 1e-9
    ax_z = fig.add_subplot(gs[1])
    _field_panel(
        ax_z, nn, nc, hx, z_disp_per_hex,
        cmap_name='RdBu_r',
        vmin=-z_abs_max, vmax=z_abs_max,
        title=f'Out-of-plane displacement (z)\nmax |z| = {q["max_z_displacement"]*1e3:.2f} mm',
        cbar_label='z-displacement [m]',
        params=p,
    )

    # Right: von Mises stress field
    ax_s = fig.add_subplot(gs[2])
    if vm_per_hex is not None:
        # Cap colorscale at 99th percentile to suppress stress-concentration spikes
        vm_cap = float(np.percentile(vm_per_hex, 99))
        _field_panel(
            ax_s, nn, nc, hx, vm_per_hex,
            cmap_name='plasma',
            vmin=0.0, vmax=max(vm_cap, 1.0),
            title=f'Von Mises stress\nmax = {q["max_von_mises_stress"]/1e6:.1f} MPa',
            cbar_label='σ_vm [Pa]',
            params=p,
        )
    else:
        ax_s.text(0.5, 0.5, 'stress not saved\n(re-run dump_results.py)',
                  ha='center', va='center', transform=ax_s.transAxes, fontsize=10)
        ax_s.set_title('Von Mises stress', fontsize=11)

    # QoI strip at bottom of figure
    fig.text(0.5, 0.01,
             f"Strain energy = {q['strain_energy']:.3e} J   |   "
             f"Max σ_vm = {q['max_von_mises_stress']/1e6:.0f} MPa   |   "
             f"σ/σ_yield = {q['first_yield_fraction']:.1f}   |   "
             f"E = {p['young_modulus']/1e9:.1f} GPa   ν = {p['poisson_ratio']:.2f}",
             ha='center', fontsize=9, color='#4A5568')

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=BG)
        print(f"Saved -> {save_path}")
    else:
        plt.show()
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--npz',  default='sofa/output/sofa_result.npz')
    p.add_argument('--save', action='store_true')
    args = p.parse_args()
    data = load_npz(args.npz)
    save_path = args.npz.replace('.npz', '.png') if args.save else None
    make_figure(data, save_path)


if __name__ == '__main__':
    main()
