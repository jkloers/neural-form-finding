"""
sofa/visualize.py — Render SOFA kirigami simulation results from .npz.

NO Sofa import — runs in kgnn_mac conda env (matplotlib only).
matplotlib.use('Agg') is set before any other import to prevent Qt crash.

Layout:
  Top row   : 3D perspective view of deformed mesh (boundary surface, full width)
  Bottom row: z-displacement top-down | von Mises top-down | QoI + loading legend

Usage:
    conda run -n kgnn_mac python sofa/visualize.py --npz sofa/output/sofa_result.npz --save
"""

import matplotlib
matplotlib.use('Agg')

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ── Project visual style ───────────────────────────────────────────────────────
BG       = '#FFFFFF'
C_ORANGE = '#F58025'
C_RED    = '#D62828'
C_BLUE   = '#457B9D'
C_GREEN  = '#2D6A4F'
C_GREY   = '#AAAAAA'

FACE_NAMES  = ['F0 (clamped)', 'F1 (driven, δz)', 'F2 (free)', 'F3 (free)']
FACE_COLORS = [C_ORANGE, C_BLUE, C_GREEN, C_RED]

plt.rcParams.update({
    'font.family': 'serif',
    'figure.facecolor': BG,
    'axes.facecolor': BG,
    'text.color': '#222222',
})


# ── Mesh helpers ───────────────────────────────────────────────────────────────

# VTK hex face definitions (outward-pointing normal order)
_HEX_FACE_NODE_INDICES = [
    [0, 1, 2, 3],  # bottom
    [4, 5, 6, 7],  # top
    [0, 1, 5, 4],  # front  (+y normal)
    [1, 2, 6, 5],  # right  (+x normal)
    [2, 3, 7, 6],  # back   (-y normal)
    [3, 0, 4, 7],  # left   (-x normal)
]


def _boundary_quads(hexes):
    """Return (M, 4) index array of boundary (outer-surface) quad faces.

    A face is on the boundary iff it belongs to exactly one hex element.
    Interior shared faces are excluded, so the result is a clean closed surface.
    Returns face_node_indices (M, 4) and parallel hex_indices (M,) for label lookup.
    """
    from collections import defaultdict
    face_owner: dict = defaultdict(list)
    for h_idx, h in enumerate(hexes):
        for fi, local in enumerate(_HEX_FACE_NODE_INDICES):
            key = tuple(sorted(h[n] for n in local))
            face_owner[key].append((h_idx, local))
    quads, owners = [], []
    for key, lst in face_owner.items():
        if len(lst) == 1:
            h_idx, local = lst[0]
            quads.append([hexes[h_idx][n] for n in local])
            owners.append(h_idx)
    return np.array(quads, dtype=np.int64), np.array(owners, dtype=np.int64)


def _top_quads(hexes):
    """VTK hex nodes 4-7 = top face (highest z). Returns (H,4) index array."""
    return hexes[:, [4, 5, 6, 7]]


def _bot_quads(hexes):
    return hexes[:, [0, 1, 2, 3]]


def _node_face_labels(nodes, bc_masks):
    """(N,) int: 0=F0 1=F1 2=F2 3=F3 -1=hinge/free."""
    lab = np.full(len(nodes), -1, dtype=int)
    for i, k in enumerate(['f0', 'f1', 'f2', 'f3']):
        lab[bc_masks[k]] = i
    return lab


def _quad_labels(hexes, node_labels):
    """(H,) label per hex: majority label of top-face nodes."""
    top = hexes[:, [4, 5, 6, 7]]
    # take the label of the first top node
    return node_labels[top[:, 0]]


def _hex_avg_vm(nn, nc, hexes, young, nu):
    lam = young * nu / ((1 + nu) * (1 - 2*nu))
    mu  = young / (2 * (1 + nu))
    vm = np.zeros(len(hexes))
    for i, h in enumerate(hexes):
        dX = (nn[h[1:4]] - nn[h[0]]).T
        dx = (nc[h[1:4]] - nc[h[0]]).T
        det_dX = np.linalg.det(dX)
        if abs(det_dX) < 1e-30:
            continue
        F = dx @ np.linalg.inv(dX)
        J = np.linalg.det(F)
        if abs(J) < 1e-10:
            continue
        E = 0.5 * (F.T @ F - np.eye(3))
        S = lam * np.trace(E) * np.eye(3) + 2*mu * E
        sig = F @ S @ F.T / J
        s = sig - np.trace(sig)/3 * np.eye(3)
        vm[i] = np.sqrt(1.5 * np.sum(s**2))
    return vm


def load_npz(path):
    d = np.load(path, allow_pickle=False)
    bc = {k: d[f'{k}_mask'] for k in ('f0', 'f1', 'f2', 'f3')}
    return {
        'nodes_nat': d['nodes_nat'],
        'nodes_cur': d['nodes_cur'],
        'hexes':     d['hexes'],
        'bc_masks':  bc,
        'qois': {k: float(d[k]) for k in (
            'strain_energy', 'max_von_mises_stress',
            'max_z_displacement', 'first_yield_fraction')},
        'params': {k: float(d[k]) if k in d else 0.0 for k in (
            'hinge_arm_width', 'hinge_fold_length', 'applied_displacement',
            'applied_moment', 'face_size', 'sheet_thickness', 'young_modulus',
            'poisson_ratio', 'yield_strength')},
        'is_moment_mode': bool(d['is_moment_mode']) if 'is_moment_mode' in d else False,
    }


def make_figure(data, save_path=None):
    nn  = data['nodes_nat']
    nc  = data['nodes_cur']
    hx  = data['hexes']
    bc  = data['bc_masks']
    q   = data['qois']
    p   = data['params']
    a   = p['face_size']
    w   = p['hinge_arm_width']
    L   = p['hinge_fold_length']

    nlabels  = _node_face_labels(nc, bc)
    quad_lab = _quad_labels(hx, nlabels)
    top_idx  = _top_quads(hx)

    # Boundary surface: all outer-facing quads (top + bottom + sides)
    bnd_quad_idx, bnd_hex_idx = _boundary_quads(hx)
    bnd_lab = quad_lab[bnd_hex_idx]   # face label per boundary quad

    # Top-down projections (still use only top face for 2D panels)
    top_2d  = nc[top_idx][:, :, :2]   # (H,4,2) — xy only
    z_vals  = nc[top_idx, 2].mean(axis=1)
    vm_vals = _hex_avg_vm(nn, nc, hx, p['young_modulus'], p['poisson_ratio'])

    # ── Figure layout ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 11), facecolor=BG)
    is_moment = data.get('is_moment_mode', False)
    load_label = (f"M = {p['applied_moment']*1e3:.1f} mN·m"
                  if is_moment else
                  f"δz = {p['applied_displacement']*1e3:.1f} mm")
    fig.suptitle(
        f"Kirigami unit cell  (RDQK 1×1)   "
        f"{load_label}  │  "
        f"L = {p['hinge_fold_length']*1e3:.1f} mm  │  "
        f"w = {p['hinge_arm_width']*1e3:.1f} mm  │  "
        f"t = {p['sheet_thickness']*1e3:.1f} mm",
        fontsize=12, y=0.99,
    )

    # Top row: 3D view (full width)
    ax3 = fig.add_subplot(2, 1, 1, projection='3d')
    ax3.set_facecolor(BG)

    # ── 3D deformed mesh — full boundary surface (top + bottom + sides) ───────
    # Gather all boundary quads per face region and render as one collection
    for fi, (nm, fc) in enumerate(zip(FACE_NAMES, FACE_COLORS)):
        mask = bnd_lab == fi
        if not mask.any():
            continue
        polys = nc[bnd_quad_idx[mask]]    # (M, 4, 3)
        coll = Poly3DCollection(polys, facecolors=fc, alpha=0.85,
                                linewidths=0.2, edgecolors='#333')
        ax3.add_collection3d(coll)

    # Hinge strips (boundary quads labelled -1)
    hmask = bnd_lab == -1
    if hmask.any():
        polys_h = nc[bnd_quad_idx[hmask]]
        ax3.add_collection3d(Poly3DCollection(polys_h, facecolors=C_GREY,
                                              alpha=0.85, linewidths=0.2, edgecolors='#444'))

    # Natural state wireframe (dashed lines in the z=0 plane for reference)
    def _draw_rect(ax, x0, x1, y0, y1, **kw):
        xs = [x0, x1, x1, x0, x0]
        ys = [y0, y0, y1, y1, y0]
        ax.plot(xs, ys, [0]*5, **kw)

    _wire_kw = dict(color='#AAAAAA', lw=0.8, ls='--', alpha=0.6)
    _draw_rect(ax3, 0,   a,     0,   a,     **_wire_kw)
    _draw_rect(ax3, a+w, 2*a+w, 0,   a,     **_wire_kw)
    _draw_rect(ax3, a+w, 2*a+w, a+w, 2*a+w, **_wire_kw)
    _draw_rect(ax3, 0,   a,     a+w, 2*a+w, **_wire_kw)

    # Annotate F0 (fixed) and F1 (driven)
    f0c = nn[bc['f0']].mean(axis=0)
    f1c = nn[bc['f1']].mean(axis=0)
    ax3.text(f0c[0], f0c[1], 0.005, 'F0\n(clamped)', fontsize=8,
             ha='center', va='bottom', color=C_ORANGE, fontweight='bold')
    ax3.text(f1c[0], f1c[1], p['applied_displacement']*0.55,
             f'F1 (driven)\ndz={p["applied_displacement"]*1e3:.0f}mm', fontsize=8,
             ha='center', va='bottom', color=C_BLUE, fontweight='bold')

    # Axis limits
    xmax = 2*a + w
    zmax = max(nc[:, 2].max(), abs(nc[:, 2].min())) * 1.4
    ax3.set_xlim(0, xmax)
    ax3.set_ylim(0, xmax)
    ax3.set_zlim(-zmax, zmax)
    # Exaggerate z-scale visually (deformation is ~10% of face size)
    ax3.set_box_aspect([1, 1, 0.5])
    ax3.set_xlabel('x [m]', labelpad=6, fontsize=9)
    ax3.set_ylabel('y [m]', labelpad=6, fontsize=9)
    ax3.set_zlabel('z [m]', labelpad=6, fontsize=9)
    ax3.set_title('Deformed shape (3D)  —  dashed wireframe = natural flat state', fontsize=10)
    ax3.view_init(elev=28, azim=35)

    # Legend for face colours
    patches = ([mpatches.Patch(color=fc, label=nm, alpha=0.85)
                for nm, fc in zip(FACE_NAMES, FACE_COLORS)] +
               [mpatches.Patch(color=C_GREY, label='Hinges')])
    ax3.legend(handles=patches, loc='upper left', fontsize=8, framealpha=0.7)

    # ── Bottom row: 3 small panels ─────────────────────────────────────────────
    gs_bot = fig.add_gridspec(1, 3, left=0.04, right=0.97,
                               top=0.40, bottom=0.05,
                               wspace=0.30)

    # Panel B1: z-displacement top-down
    ax_z = fig.add_subplot(gs_bot[0])
    ax_z.set_aspect('equal')
    ax_z.set_title('z-displacement (top view)', fontsize=10)
    ax_z.set_xlabel('x [m]', fontsize=8); ax_z.set_ylabel('y [m]', fontsize=8)
    vext = max(abs(z_vals).max(), 1e-9)
    norm_z = mcolors.TwoSlopeNorm(vcenter=0, vmin=-vext, vmax=vext)
    coll_z = PolyCollection(top_2d, array=z_vals, cmap='RdYlBu_r',
                             norm=norm_z, linewidths=0.2, edgecolors='k')
    ax_z.add_collection(coll_z)
    ax_z.autoscale_view()
    fig.colorbar(coll_z, ax=ax_z, label='z [m]', shrink=0.85)

    # Panel B2: von Mises
    ax_vm = fig.add_subplot(gs_bot[1])
    ax_vm.set_aspect('equal')
    ax_vm.set_title('von Mises stress (top view)', fontsize=10)
    ax_vm.set_xlabel('x [m]', fontsize=8); ax_vm.set_ylabel('y [m]', fontsize=8)
    norm_vm = mcolors.Normalize(vmin=0, vmax=max(vm_vals.max(), 1e3))
    coll_vm = PolyCollection(top_2d, array=vm_vals, cmap='Reds',
                              norm=norm_vm, linewidths=0.2, edgecolors='k')
    ax_vm.add_collection(coll_vm)
    ax_vm.autoscale_view()
    fig.colorbar(coll_vm, ax=ax_vm, label='σ_vm [Pa]', shrink=0.85)

    # Panel B3: QoI table + loading description
    ax_t = fig.add_subplot(gs_bot[2])
    ax_t.axis('off')

    rows = [
        ['Loading', ''],
        ['  F0', 'Fully clamped (z = 0)'],
        ['  F1', ('Moment M=%.1f mN·m' % (p['applied_moment']*1e3))
                  if is_moment else 'Rotation about H0 axis'],
        ['  ',   ('  (force-ctrl)' if is_moment
                  else f'  max δz = {p["applied_displacement"]*1e3:.1f} mm')],
        ['  F2, F3', 'Free (kinematic)'],
        ['─'*10, '─'*18],
        ['Strain energy',   f"{q['strain_energy']:.3e} J"],
        ['Max σ_vm',        f"{q['max_von_mises_stress']/1e6:.1f} MPa"],
        ['Max |z| (free)',  f"{q['max_z_displacement']*1e3:.2f} mm"],
        ['σ / σ_yield',     f"{q['first_yield_fraction']:.2f}"],
        ['─'*10, '─'*18],
        ['Face size a',     f"{p['face_size']*1e3:.0f} mm"],
        ['Arm width w',     f"{p['hinge_arm_width']*1e3:.1f} mm"],
        ['Fold length L',   f"{p['hinge_fold_length']*1e3:.1f} mm"],
        ['Thickness t',     f"{p['sheet_thickness']*1e3:.1f} mm"],
        ['E / ν',           f"{p['young_modulus']/1e9:.1f} GPa / {p['poisson_ratio']:.2f}"],
    ]
    tbl = ax_t.table(cellText=rows, colLabels=['', ''],
                     loc='center', cellLoc='left')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.25)
    # remove header row border
    for col in (0, 1):
        tbl[0, col].set_visible(False)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=BG)
        print(f"Saved → {save_path}")
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
