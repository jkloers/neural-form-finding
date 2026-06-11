"""
visualize_gmsh_mesh.py — gmsh lens-hinge mesh, rendered from REAL connectivity.

Hinge = strip between an upper and a lower cubic Bézier arc, each with DISTINCT
endpoints that slide on the face edges.  Parameters: gap (face separation),
s0/s1 reaches, and the four interior control points (bc1/bc2 upper + lower).

Unlike a naive matplotlib re-triangulation, this script draws the actual tet
boundary triangles so the mesh shown is exactly the mesh SOFA receives.  It also
runs a validity audit (volumes, conformity, manifoldness) and prints a verdict.

Run:
    python scripts/visualize_gmsh_mesh.py
"""
import sys, types
import matplotlib; matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection, PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ── 2-face geometry (unit_2face_D, face_size_m=0.1 m) ─────────────────────────
F0 = np.array([[70.71,0],[141.42,70.71],[70.71,141.42],[0,70.71]]) * 1e-3
F1 = np.array([[141.42,70.71],[212.13,0],[282.84,70.71],[212.13,141.42]]) * 1e-3

arm    = 0.003

# ── build mesh ────────────────────────────────────────────────────────────────
sys.path.insert(0, 'sofa')
from nff.sofa.mesh_builder_gmsh import build_mesh_gmsh, compute_hinge_geometry

cs = types.SimpleNamespace(
    face_centroids             = np.array([[70.71e-3,70.71e-3],[212.13e-3,70.71e-3]]),
    centroid_node_vectors      = np.array([F0-[[70.71e-3,70.71e-3]],
                                           F1-[[212.13e-3,70.71e-3]]]),
    hinge_node_pairs           = np.array([[[0,1],[1,0]]], dtype=np.int32),
    hinge_adj_info             = np.array([[0,0,0,2,3]],  dtype=np.int32),
    constrained_face_DOF_pairs = np.array([[0,0],[0,1],[0,2]], dtype=np.int32),
    loaded_face_DOF_pairs      = np.array([[1,2]], dtype=np.int32),
)

nodes, tets, bc = build_mesh_gmsh(
    cs, gap=arm, sheet_thickness=0.001,
    bezier_params=None,   # use symmetric default CPs
    n_z_layers=1,
)
N, M = len(nodes), len(tets)
hinge_mask = ~bc['clamped'] & ~bc['loaded']
print(f'Mesh: {N} nodes, {M} tets')
print(f'Clamped: {bc["clamped"].sum()}, Loaded: {bc["loaded"].sum()}, '
      f'Hinge: {hinge_mask.sum()}')


# ── validity audit (from real connectivity) ──────────────────────────────────
def _audit(nodes, tets):
    p = nodes[tets]
    vol = np.einsum('mi,mi->m', np.cross(p[:,1]-p[:,0], p[:,2]-p[:,0]), p[:,3]-p[:,0]) / 6.0
    av = np.abs(vol)
    es = np.sum((p[:,[0,0,0,1,1,2]] - p[:,[1,2,3,2,3,3]])**2, axis=(1,2))
    q = 12*(3*av)**(2/3) / es
    ref = np.zeros(len(nodes), bool); ref[tets.ravel()] = True
    from collections import Counter
    fc = Counter(map(tuple, np.sort(np.concatenate(
        [tets[:,[0,1,2]],tets[:,[0,1,3]],tets[:,[0,2,3]],tets[:,[1,2,3]]]), axis=1)))
    nonmanifold = sum(1 for c in fc.values() if c > 2)
    bnd = np.array([f for f, c in fc.items() if c == 1])
    return dict(inverted=int((vol <= 0).sum()), unref=int((~ref).sum()),
                nonmanifold=nonmanifold, qmin=float(q.min()),
                volmin=float(av.min()), boundary_tris=bnd)

a = _audit(nodes, tets)
ok = (a['inverted'] == 0 and a['unref'] == 0 and a['nonmanifold'] == 0 and a['qmin'] > 0.05)
verdict = 'VALID ✓' if ok else 'INVALID ✗'
print(f'Audit: inverted={a["inverted"]} unreferenced={a["unref"]} '
      f'non-manifold={a["nonmanifold"]} qmin={a["qmin"]:.3f} → {verdict}')


# ── real hinge Bézier geometry (4 DISTINCT anchors per hinge) ─────────────────
# Pulled straight from the builder so the drawing can never drift from the mesh.
# Upper arc:  p0_top (face fi upper edge) → p1_top (face fk upper edge)
# Lower arc:  p0_bot (face fi lower edge) → p1_bot (face fk lower edge)
geo = compute_hinge_geometry(cs, gap=arm, bezier_params=None)

def _bez(p0, c1, c2, p3, n=300):
    t = np.linspace(0,1,n+1)[:,None]
    return (1-t)**3*p0 + 3*(1-t)**2*t*c1 + 3*(1-t)*t**2*c2 + t**3*p3

arcs = []
for hd in geo['hinge_data']:
    arcs.append(dict(
        up=_bez(hd['p0_top'], hd['bc1_up'], hd['bc2_up'], hd['p1_top']) * 1000,
        lo=_bez(hd['p0_bot'], hd['bc1_lo'], hd['bc2_lo'], hd['p1_bot']) * 1000,
        uc=np.array([hd['p0_top'], hd['bc1_up'], hd['bc2_up'], hd['p1_top']]) * 1000,
        lc=np.array([hd['p0_bot'], hd['bc1_lo'], hd['bc2_lo'], hd['p1_bot']]) * 1000,
        anchors=np.array([hd['p0_top'], hd['p1_top'],
                          hd['p0_bot'], hd['p1_bot']]) * 1000,
        center=hd['corner'] * 1000, arm=hd['gap'] * 1000,
    ))

# ── real bottom-layer mesh edges (z ≈ 0 triangles from tets) ──────────────────
zmin = nodes[:, 2].min()
bot = nodes[:, 2] < zmin + 1e-6
# Boundary triangles whose 3 vertices all sit on the bottom layer == the 2D mesh.
bnd = a['boundary_tris']
bot_tris = bnd[bot[bnd].all(axis=1)]
xy = nodes[:, :2] * 1000

# unique mesh edges from the bottom triangles
e = np.sort(np.concatenate([bot_tris[:,[0,1]], bot_tris[:,[1,2]], bot_tris[:,[0,2]]]), axis=1)
e = np.unique(e, axis=0)
segs = xy[e]

# per-triangle fill colour by region (majority vote of its 3 nodes)
reg = np.where(bc['clamped'], 0, np.where(bc['loaded'], 2, 1))
tri_reg = np.round(reg[bot_tris].mean(axis=1)).astype(int)
REGC = {0: '#66BB6A', 1: '#FFD54F', 2: '#42A5F5'}
tri_fc = [REGC[r] for r in tri_reg]

# ── figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 7))
fig.suptitle(f'gmsh tet mesh (real connectivity) — 2-face kirigami  |  '
             f'{N} nodes, {M} tets  |  {verdict}',
             fontsize=13, fontweight='bold')
ax1 = fig.add_subplot(1, 3, 1)
ax2 = fig.add_subplot(1, 3, 2)
ax3 = fig.add_subplot(1, 3, 3, projection='3d')

for ax, zoom in [(ax1, False), (ax2, True)]:
    ax.add_collection(PolyCollection(xy[bot_tris], facecolors=tri_fc,
                                     alpha=0.35, edgecolors='none', zorder=1))
    ax.add_collection(LineCollection(segs, colors='steelblue', lw=0.5,
                                     alpha=0.7, zorder=2))

    for fv, c in [(F0,'#1a5276'), (F1,'#1e8449')]:
        ax.add_patch(plt.Polygon(np.vstack([fv*1000, fv[0]*1000]),
                                  fill=False, edgecolor=c, lw=2, linestyle='--', zorder=4))

    for arc in arcs:
        ax.plot(arc['up'][:,0], arc['up'][:,1], '-', color='tomato',       lw=2.5, zorder=8)
        ax.plot(arc['lo'][:,0], arc['lo'][:,1], '-', color='mediumpurple', lw=2.5, zorder=8)
        ax.plot(arc['uc'][:,0], arc['uc'][:,1], 'o--', color='tomato',       ms=6, lw=1, alpha=0.8, zorder=7)
        ax.plot(arc['lc'][:,0], arc['lc'][:,1], 'o--', color='mediumpurple', ms=6, lw=1, alpha=0.8, zorder=7)
        # Four DISTINCT anchor points: top pair (red), bottom pair (purple)
        ax.plot(arc['anchors'][:2,0], arc['anchors'][:2,1], 's', color='tomato',
                ms=9, zorder=10, markeredgecolor='white', markeredgewidth=1)
        ax.plot(arc['anchors'][2:,0], arc['anchors'][2:,1], 's', color='mediumpurple',
                ms=9, zorder=10, markeredgecolor='white', markeredgewidth=1)

    if zoom:
        arc = arcs[0]
        cx, cy = arc['center']
        pad = arc['arm'] * 3.2
        ax.set_xlim(cx-pad, cx+pad); ax.set_ylim(cy-pad, cy+pad)
        ax.set_title('Hinge zoom — 4 distinct anchors, real tet edges', fontsize=11)
        for pt, lbl, c in [(arc['anchors'][0],'p0_top','tomato'),
                           (arc['anchors'][1],'p1_top','tomato'),
                           (arc['anchors'][2],'p0_bot','mediumpurple'),
                           (arc['anchors'][3],'p1_bot','mediumpurple')]:
            ax.annotate(lbl, pt, xytext=(4,5), textcoords='offset points',
                        fontsize=8, color=c, fontweight='bold')
    else:
        ax.set_xlim(xy[:,0].min()-5, xy[:,0].max()+5)
        ax.set_ylim(xy[:,1].min()-5, xy[:,1].max()+5)
        ax.set_title('Full mesh — actual boundary triangulation', fontsize=11)
        ax.legend(handles=[
            mpatches.Patch(color=REGC[0], alpha=0.6, label=f'face 0 clamped ({bc["clamped"].sum()})'),
            mpatches.Patch(color=REGC[1], alpha=0.6, label=f'hinge ({hinge_mask.sum()})'),
            mpatches.Patch(color=REGC[2], alpha=0.6, label=f'face 1 loaded ({bc["loaded"].sum()})'),
            plt.Line2D([0],[0],color='tomato',lw=2.5,label='upper arc'),
            plt.Line2D([0],[0],color='mediumpurple',lw=2.5,label='lower arc'),
        ], fontsize=8, loc='upper left', ncol=2)

    ax.set_aspect('equal')
    ax.set_xlabel('x [mm]', fontsize=10); ax.set_ylabel('y [mm]', fontsize=10)

# ── 3D view: full tet boundary surface ────────────────────────────────────────
tris3d = nodes[a['boundary_tris']] * 1000
fcol = [REGC[r] for r in np.round(reg[a['boundary_tris']].mean(axis=1)).astype(int)]
ax3.add_collection3d(Poly3DCollection(tris3d, facecolors=fcol, edgecolors='steelblue',
                                      linewidths=0.25, alpha=0.85))
lim = np.array([xy[:,0].min(), xy[:,0].max(), xy[:,1].min(), xy[:,1].max()])
ax3.set_xlim(lim[0], lim[1]); ax3.set_ylim(lim[2], lim[3]); ax3.set_zlim(-15, 15)
ax3.set_box_aspect((lim[1]-lim[0], lim[3]-lim[2], 8))
ax3.view_init(elev=55, azim=-60)
ax3.set_title('3D tet boundary surface (extruded sheet)', fontsize=11)
ax3.set_xlabel('x [mm]'); ax3.set_ylabel('y [mm]'); ax3.set_zlabel('z [mm]')

plt.tight_layout()
out = 'scripts/gmsh_mesh_overview.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved → {out}')
