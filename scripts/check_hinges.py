"""
Visualize hinge fold directions from the CS mesh to check for flipped hinges.
Uses only the cs_mesh.npz (no JAX) — geometry extracted from face node masks.
"""

import os
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import sys
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
from nff.sofa.mesh_builder import _extract_geometry
from nff.sofa.config_to_physical import physical_scale_from_config
from types import SimpleNamespace
import yaml

RUN = 'data/outputs/runs/run_20260608_232554_c001_mpnn_2x2'
CONFIG = 'data/configs/sofa/c001_mpnn_2x2.yaml'

# ── Rebuild initial CentroidalState (no GNN, just initial tessellation) ────────
config = load_and_parse_config(CONFIG)
topo = config.topology
topo_obj = SimpleNamespace(**topo)
tess = build_tessellation(topo.get('pattern'), topo.get('width', 1), topo.get('height', 1))
requested_area = topo.get('total_area')
if requested_area:
    scale = np.sqrt(requested_area / tess.compute_total_area())
    tess.update_vertices(tess.vertices * scale)
configure_tessellation(tess, topo_obj)
cs = CentroidalState.from_tessellation(tess, target_cfg=config.target)

with open(CONFIG) as f:
    raw = yaml.safe_load(f)
from nff.sofa.config_to_physical import physical_scale_from_config
phys = physical_scale_from_config(raw)

# Scale to physical units
scaled_cs = cs._replace(
    face_centroids        = cs.face_centroids        * phys.jax_scale,
    centroid_node_vectors = cs.centroid_node_vectors * phys.jax_scale,
)

# ── Extract geometry ────────────────────────────────────────────────────────────
face_verts, hinge_strips = _extract_geometry(scaled_cs, phys.fold_length, phys.arm_width)
n_faces = len(face_verts)

fc  = np.array(scaled_cs.face_centroids)
cnv = np.array(scaled_cs.centroid_node_vectors)
hnp = np.array(cs.hinge_node_pairs)
adj = np.array(cs.hinge_adj_info)

print(f'n_faces:  {n_faces}')
print(f'n_hinges: {len(hinge_strips)}')
print()

# ── Analyse fold directions ─────────────────────────────────────────────────────
_PC = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
print(f"{'H':>3} {'fi[lj]':>8} {'fk[ll]':>8} {'adj_li':>7} {'adj_lk':>7} {'fdi·fdk':>8}  status")
print('-' * 65)
problem_hinges = []
for h, hs in enumerate(hinge_strips):
    fi, lj = hs['fi'], hs['lj']
    fk, ll = hs['fk'], hs['ll']
    adj_li, adj_lk = hs['adj_li'], hs['adj_lk']
    fdi = hs['fold_dir_i']
    fdk = hs['fold_dir_k']
    dot = float(np.dot(fdi, fdk))
    p1  = hs['p1']
    p2  = hs['p2']
    # For a correct hinge, both fold dirs should point INTO the gap
    # i.e., fdi should point from p1 AWAY from face i interior
    face_center_i = fc[fi]
    face_center_k = fc[fk]
    # dot(fdi, p1 - face_center_i) > 0 means fold_dir_i points outward (away from face centre)
    outward_i = float(np.dot(fdi, p1 - face_center_i)) > 0
    outward_k = float(np.dot(fdk, p2 - face_center_k)) > 0
    status = 'OK'
    if not outward_i:
        status = 'FDI_INWARD'
        problem_hinges.append((h, 'fdi'))
    if not outward_k:
        status += '+FDK_INWARD' if status != 'OK' else 'FDK_INWARD'
        problem_hinges.append((h, 'fdk'))
    print(f'{h:3d}  F{fi}[{lj}]→F{fk}[{ll}]  li={adj_li}  lk={adj_lk}  {dot:+.3f}  {status}')

print()
print(f'Total problematic fold dirs: {len(problem_hinges)}')
if problem_hinges:
    print('  ' + ', '.join(f'H{h}({side})' for h, side in problem_hinges))
else:
    print('  None found — all fold dirs point outward.')

# ── Plot ────────────────────────────────────────────────────────────────────────
COLORS = ['#4472C4','#ED7D31','#70AD47','#FFC000','#5A9BD5','#F15A29','#48A463',
          '#E2A829','#9B59B6','#1ABC9C','#E74C3C','#3498DB','#F39C12','#2ECC71',
          '#E91E63','#00BCD4']

fig, ax = plt.subplots(figsize=(14, 14))
fv_mm = face_verts * 1e3

for fi in range(n_faces):
    poly = fv_mm[fi]
    ax.add_patch(plt.Polygon(poly, closed=True, facecolor=COLORS[fi], alpha=0.35, edgecolor='k', linewidth=1.2))
    cx, cy = poly.mean(axis=0)
    ax.text(cx, cy, str(fi), ha='center', va='center', fontsize=11, fontweight='bold')

fold_arrow = phys.face_size * 1e3 * 0.2  # 20% of face_size in mm
for h, hs in enumerate(hinge_strips):
    p1 = hs['p1'] * 1e3
    p2 = hs['p2'] * 1e3
    fdi = hs['fold_dir_i']
    fdk = hs['fold_dir_k']
    fi, fk = hs['fi'], hs['fk']

    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', linewidth=2, zorder=5)

    # fold_dir_i arrow (blue = face_i side)
    ax.annotate('', xy=(p1 + fdi*fold_arrow), xytext=p1,
                arrowprops=dict(arrowstyle='->', color='blue', lw=2.0))
    # fold_dir_k arrow (red = face_k side)
    ax.annotate('', xy=(p2 + fdk*fold_arrow), xytext=p2,
                arrowprops=dict(arrowstyle='->', color='red', lw=2.0))

    mid = (p1 + p2) * 0.5
    ax.text(mid[0], mid[1], f'H{h}', ha='center', va='center',
            fontsize=7, color='purple', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.8))

# Mark problem hinges
for h, side in problem_hinges:
    hs = hinge_strips[h]
    p = (hs['p1' if side == 'fdi' else 'p2']) * 1e3
    ax.plot(p[0], p[1], 'X', markersize=14, color='magenta', zorder=10)

all_verts = fv_mm.reshape(-1, 2)
pad = phys.face_size * 1e3 * 0.3
ax.set_xlim(all_verts[:,0].min() - pad, all_verts[:,0].max() + pad)
ax.set_ylim(all_verts[:,1].min() - pad, all_verts[:,1].max() + pad)
ax.set_aspect('equal')
ax.set_xlabel('x [mm]'); ax.set_ylabel('y [mm]')
ax.set_title('Hinge fold directions (initial RDQK_D tessellation)\n'
             'Blue arrow = fold_dir_i (face_i side), Red arrow = fold_dir_k (face_k side)\n'
             'Both should point OUT OF face into the hinge gap.\n'
             'Magenta X = inward (wrong direction)',
             fontsize=11)

out = f'{RUN}/hinge_fold_dirs.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f'\nSaved: {out}')
