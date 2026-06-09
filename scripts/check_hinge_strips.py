"""
Visualize individual hinge strip quad shapes to detect truly-flipped strips.
A strip is 'flipped' if the quad (hs0, hs1, hs2, hs3) crosses itself.
"""

import os, sys
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
jax.config.update('jax_enable_x64', True)

from nff.topology.builder import build_tessellation
from nff.stages.state import CentroidalState
from nff.config.experiment import load_and_parse_config
from nff.config.conditions import configure_tessellation
from nff.sofa.mesh_builder import _extract_geometry, CLOSED_GAP_TOL
from nff.sofa.config_to_physical import physical_scale_from_config
from types import SimpleNamespace
import yaml

RUN    = 'data/outputs/runs/run_20260608_232554_c001_mpnn_2x2'
CONFIG = 'data/configs/sofa/c001_mpnn_2x2.yaml'

config = load_and_parse_config(CONFIG)
topo   = config.topology
topo_obj = SimpleNamespace(**topo)
tess   = build_tessellation(topo.get('pattern'), topo.get('width', 1), topo.get('height', 1))
area   = topo.get('total_area')
if area:
    scale = np.sqrt(area / tess.compute_total_area())
    tess.update_vertices(tess.vertices * scale)
configure_tessellation(tess, topo_obj)
cs = CentroidalState.from_tessellation(tess, target_cfg=config.target)

with open(CONFIG) as f:
    raw = yaml.safe_load(f)
phys = physical_scale_from_config(raw)

scaled_cs = cs._replace(
    face_centroids        = cs.face_centroids        * phys.jax_scale,
    centroid_node_vectors = cs.centroid_node_vectors * phys.jax_scale,
)

face_verts, hinge_strips = _extract_geometry(scaled_cs, phys.fold_length, phys.arm_width)
n_faces  = len(face_verts)
n_hinges = len(hinge_strips)

def _cross2d(a, b):
    """2D cross product."""
    return a[0]*b[1] - a[1]*b[0]

def strip_winding(hs0, hs1, hs2, hs3):
    """
    Determine strip quad winding (CCW=+1, CW=-1) and whether it self-intersects.
    Quad vertex order: hs0→hs1→hs2→hs3 (one face side → other face side).
    """
    v01 = hs1 - hs0
    v12 = hs2 - hs1
    v23 = hs3 - hs2
    v30 = hs0 - hs3
    crosses = [_cross2d(v01, v12), _cross2d(v12, v23),
               _cross2d(v23, v30), _cross2d(v30, v01)]
    # If signs differ, quad is non-convex or self-intersecting
    signs = [1 if c > 0 else -1 for c in crosses]
    same_sign = all(s == signs[0] for s in signs)
    return same_sign, signs[0]  # (convex, winding_sign)

print(f"{'H':>3}  {'fi[lj]':>8}  {'fk[ll]':>8}  {'convex':>7}  {'wind':>5}  {'status'}")
print('-' * 60)

flipped = []
for h, hs in enumerate(hinge_strips):
    p1  = hs['p1']
    p2  = hs['p2']
    fdi = hs['fold_dir_i']
    fdk = hs['fold_dir_k']
    fl  = hs['arm_width'] if hs['arm_width'] > CLOSED_GAP_TOL else phys.arm_width
    fold_l = phys.fold_length

    hs0 = p1
    hs1 = p2
    hs2 = p2 + fdk * fold_l
    hs3 = p1 + fdi * fold_l

    convex, winding = strip_winding(hs0, hs1, hs2, hs3)
    status = 'OK' if convex else 'FLIPPED'
    if not convex:
        flipped.append(h)
    fi, lj = hs['fi'], hs['lj']
    fk, ll = hs['fk'], hs['ll']
    wind_str = 'CCW' if winding > 0 else 'CW'
    print(f"{h:3d}  F{fi}[{lj}]→F{fk}[{ll}]  {str(convex):>7}  {wind_str:>5}  {status}")

print()
print(f'Flipped (non-convex) hinge strips: {len(flipped)}')
if flipped:
    print(f'  Hinges: {flipped}')

# ── Plot strip quads, highlight CW-wound ones ──────────────────────────────────
COLORS = ['#4472C4','#ED7D31','#70AD47','#FFC000','#5A9BD5','#F15A29','#48A463',
          '#E2A829','#9B59B6','#1ABC9C','#E74C3C','#3498DB','#F39C12','#2ECC71',
          '#E91E63','#00BCD4']

fig, axes = plt.subplots(1, 2, figsize=(20, 10))

for ax_idx, (ax, title_sfx) in enumerate([(axes[0], 'full mesh'), (axes[1], 'zoom hinge strips')]):
    fv_mm = face_verts * 1e3
    for fi in range(n_faces):
        poly = fv_mm[fi]
        ax.add_patch(plt.Polygon(poly, closed=True, facecolor=COLORS[fi], alpha=0.25,
                                  edgecolor='k', linewidth=1.2 if ax_idx == 0 else 0.5))
        cx, cy = poly.mean(axis=0)
        ax.text(cx, cy, str(fi), ha='center', va='center',
                fontsize=9 if ax_idx == 0 else 6, fontweight='bold')

    winding_counts = {'CCW': 0, 'CW': 0}
    for h, hs in enumerate(hinge_strips):
        p1  = hs['p1'] * 1e3
        p2  = hs['p2'] * 1e3
        fdi = hs['fold_dir_i']
        fdk = hs['fold_dir_k']
        fold_l = phys.fold_length * 1e3

        hs0 = p1
        hs1 = p2
        hs2 = p2 + fdk * fold_l
        hs3 = p1 + fdi * fold_l

        convex, winding = strip_winding(hs0, hs1, hs2, hs3)
        color = 'green' if winding > 0 else 'orange'
        alpha = 0.6 if convex else 0.9
        edgecol = 'blue' if h in flipped else 'gray'
        lw = 2.0 if h in flipped else 0.8

        ax.add_patch(plt.Polygon([hs0, hs1, hs2, hs3], closed=True,
                                  facecolor=color, alpha=alpha,
                                  edgecolor=edgecol, linewidth=lw, zorder=5))
        mid = (hs0 + hs1 + hs2 + hs3) / 4
        ax.text(mid[0], mid[1], str(h), ha='center', va='center',
                fontsize=5, color='white', fontweight='bold', zorder=6)
        winding_counts['CCW' if winding > 0 else 'CW'] += 1

    all_verts = fv_mm.reshape(-1, 2)
    pad = phys.face_size * 1e3 * 0.15
    ax.set_xlim(all_verts[:,0].min() - pad, all_verts[:,0].max() + pad)
    ax.set_ylim(all_verts[:,1].min() - pad, all_verts[:,1].max() + pad)
    ax.set_aspect('equal')
    ax.set_xlabel('x [mm]'); ax.set_ylabel('y [mm]')
    ax.set_title(f'Hinge strips ({title_sfx})\n'
                 f'Green=CCW-wound ({winding_counts["CCW"]}), '
                 f'Orange=CW-wound ({winding_counts["CW"]}), '
                 f'Blue-border=non-convex ({len(flipped)})\n'
                 f'fold_length={phys.fold_length*1e3:.1f}mm  arm_width={phys.arm_width*1e3:.1f}mm')

if ax_idx == 1:
    # Zoom into center region
    axes[1].set_xlim(70, 180)
    axes[1].set_ylim(70, 180)

out = f'{RUN}/hinge_strips.png'
fig.tight_layout()
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f'\nSaved: {out}')
