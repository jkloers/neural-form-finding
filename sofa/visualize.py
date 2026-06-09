"""
sofa/visualize.py — Render SOFA kirigami simulation results from .npz.

NO Sofa import — runs in kgnn_mac conda env (matplotlib only).
matplotlib.use('Agg') is set before any other import to prevent Qt crash.

Layout (3-panel view):
  Left   : Natural flat state — top-down XY hex mesh colored by face ID
  Centre : Out-of-plane z-displacement field (RdBu colormap)
  Right  : Von Mises stress field (plasma colormap)

Works for any number of faces (1×1, 2×2, etc.) — face count is read from .npz.

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
from matplotlib.collections import PolyCollection

# ── Visual style ───────────────────────────────────────────────────────────────
BG = '#FAFAFA'

_BASE_COLORS = [
    '#E07020',  # 0 orange  (F0 clamped)
    '#2B6CB0',  # 1 blue    (F1 driven)
    '#276749',  # 2 green
    '#C53030',  # 3 red
    '#805AD5',  # 4 purple
    '#2C7A7B',  # 5 teal
    '#D69E2E',  # 6 yellow
    '#E53E3E',  # 7 bright red
    '#3182CE',  # 8 mid blue
    '#38A169',  # 9 mid green
]
C_HINGE = '#A0AEC0'

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'figure.facecolor': BG,
    'axes.facecolor': BG,
    'text.color': '#1A202C',
})


def _face_colors(n_faces):
    """Return dict {face_id: hex color} for n_faces faces."""
    return {i: _BASE_COLORS[i % len(_BASE_COLORS)] for i in range(n_faces)}


# ── Mesh helpers ───────────────────────────────────────────────────────────────

def _top_quads(hexes):
    """(H, 4) top-face node indices for each hex (z-max face [4,5,6,7])."""
    return hexes[:, [4, 5, 6, 7]]


# ── Load .npz ─────────────────────────────────────────────────────────────────

def load_npz(path):
    d = np.load(path, allow_pickle=False)
    nodes_nat = d['nodes_nat']
    nodes_cur = d['nodes_cur']
    hexes     = d['hexes']

    n_faces = int(d['n_faces']) if 'n_faces' in d else sum(
        1 for k in d.files
        if k.startswith('f') and k.endswith('_mask') and k[1:-5].isdigit())

    bc = {}
    for i in range(n_faces):
        key = f'f{i}_mask'
        if key in d:
            bc[f'f{i}'] = d[key].astype(bool)
    bc['clamped'] = (d['clamped_mask'] if 'clamped_mask' in d else d['f0_mask']).astype(bool)
    bc['loaded']  = (d['loaded_mask']  if 'loaded_mask'  in d else d['f1_mask']).astype(bool)

    # Backward-compatible param loading (old keys: hinge_arm_width, etc.)
    params = {}
    _aliases = {'arm_width': 'hinge_arm_width', 'fold_length': 'hinge_fold_length'}
    for key in ('arm_width', 'fold_length', 'sheet_thickness',
                'young_modulus', 'poisson_ratio', 'yield_strength', 'applied_moment'):
        if key in d:
            params[key] = float(d[key])
        elif key in _aliases and _aliases[key] in d:
            params[key] = float(d[_aliases[key]])
        else:
            params[key] = 0.0

    qois = {
        'strain_energy':        float(d['strain_energy']),
        'max_von_mises_stress': float(d['max_von_mises_stress']),
        'max_xy_displacement':  float(d['max_xy_displacement']) if 'max_xy_displacement' in d else 0.0,
        'max_z_displacement':   float(d['max_z_displacement']),
        'first_yield_fraction': float(d['first_yield_fraction']),
    }
    return {
        'nodes_nat':          nodes_nat,
        'nodes_cur':          nodes_cur,
        'hexes':              hexes,
        'bc_masks':           bc,
        'n_faces':            n_faces,
        'params':             params,
        'qois':               qois,
        'vm_per_hex':         d['vm_per_hex'] if 'vm_per_hex' in d else None,
        'is_moment_mode':     bool(d['is_moment_mode'])     if 'is_moment_mode'     in d else False,
        'rotation_angle_deg': float(d['rotation_angle_deg']) if 'rotation_angle_deg' in d else 0.0,
    }


# ── Natural-state panel ────────────────────────────────────────────────────────

def _draw_natural_mesh(ax, nodes_nat, hexes, bc_masks, n_faces):
    """Left panel: top-down hex mesh colored by face ID."""
    top_qi    = _top_quads(hexes)
    fcolors   = _face_colors(n_faces)

    # Per-node face label (−1 = hinge)
    lab_node = np.full(len(nodes_nat), -1, dtype=int)
    for i in range(n_faces):
        key = f'f{i}'
        if key in bc_masks:
            lab_node[bc_masks[key]] = i

    # Per-hex label from majority vote of top-face nodes
    lab_hex = np.array([
        int(np.bincount(lab_node[top_qi[h]] + 1,
                        minlength=n_faces + 1).argmax()) - 1
        for h in range(len(hexes))
    ], dtype=int)

    hex_colors = [fcolors.get(lb, C_HINGE) for lb in lab_hex]
    quads_xy   = nodes_nat[top_qi][:, :, :2]
    coll = PolyCollection(quads_xy, facecolors=hex_colors,
                          linewidths=0.3, edgecolors='#555555', zorder=2)
    ax.add_collection(coll)

    xn, yn = nodes_nat[:, 0], nodes_nat[:, 1]
    span   = max(xn.max() - xn.min(), yn.max() - yn.min(), 1e-9)
    pad    = 0.05 * span
    ax.set_xlim(xn.min() - pad, xn.max() + pad)
    ax.set_ylim(yn.min() - pad, yn.max() + pad)
    ax.set_aspect('equal')
    ax.set_title('Natural state (CS mesh)', fontsize=11, pad=8)
    ax.set_xlabel('x [m]', fontsize=9)
    ax.set_ylabel('y [m]', fontsize=9)
    ax.tick_params(labelsize=8)

    handles = [mpatches.Patch(color=fcolors[i], label=f'F{i}') for i in range(n_faces)]
    handles.append(mpatches.Patch(color=C_HINGE, label='hinge'))
    ax.legend(handles=handles, fontsize=7, loc='upper right', framealpha=0.8)


# ── Field panel ────────────────────────────────────────────────────────────────

def _field_panel(ax, nodes_nat, nodes_cur, hexes, field_per_hex,
                 cmap_name, vmin, vmax, title, cbar_label):
    """
    Top-down XY panel coloring each hex top-face by a scalar field.

    Draws natural hex outlines as a dashed grey reference behind the
    deformed field.
    """
    cmap   = plt.get_cmap(cmap_name)
    norm   = plt.Normalize(vmin=vmin, vmax=vmax)
    top_qi = _top_quads(hexes)

    ax.set_aspect('equal')
    ax.set_xlabel('x [m]', fontsize=9)
    ax.set_ylabel('y [m]', fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_title(title, fontsize=11, pad=8)

    # Natural hex outlines as dashed reference
    nat_quads = nodes_nat[top_qi][:, :, :2]
    nat_coll  = PolyCollection(nat_quads, facecolors='none', linewidths=0.4,
                               edgecolors='#BBBBBB', linestyle='--', zorder=1)
    ax.add_collection(nat_coll)

    # Deformed hex field
    colors   = cmap(norm(field_per_hex))
    quads_xy = nodes_cur[top_qi][:, :, :2]
    coll     = PolyCollection(quads_xy, facecolors=colors,
                              linewidths=0.15, edgecolors='none', zorder=2)
    ax.add_collection(coll)

    xd   = nodes_cur[:, 0]
    yd   = nodes_cur[:, 1]
    span = max(xd.max() - xd.min(), yd.max() - yd.min(), 1e-9)
    pad  = 0.06 * span
    ax.set_xlim(xd.min() - pad, xd.max() + pad)
    ax.set_ylim(yd.min() - pad, yd.max() + pad)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, shrink=0.75, pad=0.02,
                 label=cbar_label, format='%.2g')


# ── Main figure ────────────────────────────────────────────────────────────────

def make_figure(data, save_path=None):
    nn      = data['nodes_nat']
    nc      = data['nodes_cur']
    hx      = data['hexes']
    bc      = data['bc_masks']
    n_faces = data['n_faces']
    p       = data['params']
    q       = data['qois']
    is_moment   = data.get('is_moment_mode', False)
    angle_deg   = data.get('rotation_angle_deg', 0.0)

    load_label = (f"M = {p['applied_moment']*1e3:.1f} mN·m  (moment-ctrl)"
                  if is_moment else
                  f"theta = {angle_deg:.0f} deg  (rotation-ctrl)")

    fig = plt.figure(figsize=(22, 8), facecolor=BG)
    fig.suptitle(
        f"Kirigami unit cell ({n_faces} faces)   |   {load_label}   |   "
        f"w = {p['arm_width']*1e3:.1f} mm   "
        f"L = {p['fold_length']*1e3:.1f} mm   "
        f"t = {p['sheet_thickness']*1e3:.1f} mm",
        fontsize=11, y=0.99,
    )

    vm_per_hex = data.get('vm_per_hex')

    gs = fig.add_gridspec(1, 3, left=0.04, right=0.97, top=0.91, bottom=0.09,
                          wspace=0.28, width_ratios=[1, 1, 1])

    # Left: natural state mesh colored by face ID
    ax_flat = fig.add_subplot(gs[0])
    _draw_natural_mesh(ax_flat, nn, hx, bc, n_faces)

    # Centre: out-of-plane z-displacement field
    top_qi          = _top_quads(hx)
    z_disp_per_node = nc[:, 2] - nn[:, 2]
    z_disp_per_hex  = z_disp_per_node[top_qi].mean(axis=1)
    z_abs_max       = np.abs(z_disp_per_hex).max() or 1e-9
    ax_z = fig.add_subplot(gs[1])
    _field_panel(
        ax_z, nn, nc, hx, z_disp_per_hex,
        cmap_name='RdBu_r',
        vmin=-z_abs_max, vmax=z_abs_max,
        title=f'Out-of-plane displacement (z)\nmax |z| = {q["max_z_displacement"]*1e3:.2f} mm',
        cbar_label='z-displacement [m]',
    )

    # Right: von Mises stress field
    ax_s = fig.add_subplot(gs[2])
    if vm_per_hex is not None:
        vm_cap = float(np.percentile(vm_per_hex, 99))
        _field_panel(
            ax_s, nn, nc, hx, vm_per_hex,
            cmap_name='plasma',
            vmin=0.0, vmax=max(vm_cap, 1.0),
            title=f'Von Mises stress\nmax = {q["max_von_mises_stress"]/1e6:.1f} MPa',
            cbar_label='σ_vm [Pa]',
        )
    else:
        ax_s.text(0.5, 0.5, 'stress not saved\n(re-run dump_results.py)',
                  ha='center', va='center', transform=ax_s.transAxes, fontsize=10)
        ax_s.set_title('Von Mises stress', fontsize=11)

    # QoI strip at bottom
    fig.text(
        0.5, 0.01,
        f"Strain energy = {q['strain_energy']:.3e} J   |   "
        f"Max σ_vm = {q['max_von_mises_stress']/1e6:.0f} MPa   |   "
        f"σ/σ_yield = {q['first_yield_fraction']:.1f}   |   "
        f"E = {p['young_modulus']/1e9:.1f} GPa   ν = {p['poisson_ratio']:.2f}",
        ha='center', fontsize=9, color='#4A5568',
    )

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
    data      = load_npz(args.npz)
    save_path = args.npz.replace('.npz', '.png') if args.save else None
    make_figure(data, save_path)


if __name__ == '__main__':
    main()
