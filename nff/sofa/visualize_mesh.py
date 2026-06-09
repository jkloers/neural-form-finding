"""
nff/sofa/visualize_mesh.py — Visualize hex mesh geometry before FEM.

Shows the top-face (z=max) layer of quads, color-coded per face group,
with close-up panels on each hinge to diagnose twisted hinge strips.

Usage
-----
  # From an existing cs_mesh.npz
  conda run -n kgnn_mac python nff/sofa/visualize_mesh.py \
      --mesh data/outputs/runs/<run>/cs_mesh.npz

  # Or build fresh from a config
  conda run -n kgnn_mac python nff/sofa/visualize_mesh.py \
      --config data/configs/sofa/c001_mpnn_2x2.yaml
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

os.environ['JAX_PLATFORM_NAME'] = 'cpu'
import jax
jax.config.update('jax_enable_x64', True)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PolyCollection
import matplotlib.cm as cm

REPO = pathlib.Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO))


# ── color helpers ─────────────────────────────────────────────────────────────

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


# ── mesh loading ──────────────────────────────────────────────────────────────

def _load_or_build_mesh(args) -> tuple[np.ndarray, np.ndarray, dict, int]:
    """Return (nodes, hexes, masks, n_faces)."""
    if args.mesh:
        npz = np.load(args.mesh)
        nodes  = npz['nodes']
        hexes  = npz['hexes']
        n_faces = int(npz.get('n_faces', 4))
        masks  = {f'f{i}': npz.get(f'f{i}_mask', None) for i in range(n_faces)}
        return nodes, hexes, masks, n_faces

    # Build from config
    from types import SimpleNamespace
    import yaml

    from nff.topology.builder import build_tessellation
    from nff.stages.state import CentroidalState
    from nff.config.experiment import load_and_parse_config
    from nff.config.conditions import configure_tessellation
    from nff.stages.pipeline import forward_pipeline
    from nff.sofa.mesh_builder import build_mesh_from_centroidal_state
    from nff.sofa.config_to_physical import physical_scale_from_config

    config_path = pathlib.Path(args.config)
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    phys   = physical_scale_from_config(raw)
    config = load_and_parse_config(str(config_path))

    topo = config.topology
    tess = build_tessellation(topo.get('pattern'), topo.get('width', 1), topo.get('height', 1))
    requested_area = topo.get('total_area')
    if requested_area:
        scale = np.sqrt(requested_area / tess.compute_total_area())
        tess.update_vertices(tess.vertices * scale)
    configure_tessellation(tess, SimpleNamespace(**topo))
    initial_state = CentroidalState.from_tessellation(tess, target_cfg=config.target)

    map_type = config.mapping.type
    map_params = config.mapping.params if isinstance(config.mapping.params, dict) else {}
    static_features = None
    if map_type.startswith('gnn_'):
        from nff.models.graph_builder import build_static_features
        gnn_cfg = map_params
        static_features = build_static_features(initial_state, map_type)
        static_features = {**static_features, 'num_layers': int(gnn_cfg.get('num_layers', 3))}
        import jax
        from nff.models.mpnn import init_mpnn
        key = jax.random.PRNGKey(int(gnn_cfg.get('seed', 0)))
        map_params = init_mpnn(key, static_features['node_feat_dim'],
                               int(gnn_cfg.get('hidden_dim', 32)),
                               int(gnn_cfg.get('num_layers', 3)))

    result = forward_pipeline(
        initial_state,
        target_cfg=config.target, validity_cfg=config.validity,
        physics_cfg=config.physics, map_type=map_type,
        map_params=map_params, static_features=static_features,
        load_specs=topo.get('loads', []) or [],
    )

    valid_state = result['valid_state']
    scaled = valid_state._replace(
        face_centroids        = valid_state.face_centroids        * phys.jax_scale,
        centroid_node_vectors = valid_state.centroid_node_vectors * phys.jax_scale,
    )
    nodes, hexes, bc = build_mesh_from_centroidal_state(
        scaled,
        fold_length=phys.fold_length,
        sheet_thickness=phys.sheet_thickness,
        arm_width_physical=phys.arm_width,
    )
    n_faces = len(np.array(scaled.face_centroids))
    masks = {f'f{i}': bc[f'f{i}'] for i in range(n_faces)}
    return nodes, hexes, masks, n_faces


# ── quad-face extraction ──────────────────────────────────────────────────────

def _top_quads(nodes, hexes):
    """Return (Q,4,2) XY coordinates of the z=max face of each hex."""
    # Hex node layout: [0..3]=bottom, [4..7]=top
    top_idx   = hexes[:, [4, 5, 6, 7]]   # (H, 4) node indices
    xy_mm     = nodes[:, :2] * 1e3        # metres → mm
    return xy_mm[top_idx]                 # (H, 4, 2)


def _color_quads(hexes, masks, n_faces):
    """Assign a color to each hex based on which face it belongs to."""
    colors_list = _face_colors(n_faces)
    quad_colors = np.full(len(hexes), '#dddddd', dtype=object)
    for fi in range(n_faces):
        m = masks.get(f'f{fi}')
        if m is None:
            continue
        mask = m.astype(bool)
        top_nodes = hexes[:, 4:]          # (H, 4)
        hits = mask[top_nodes].any(axis=1)
        quad_colors[hits] = colors_list[fi]
    return quad_colors


# ── full-mesh overview ────────────────────────────────────────────────────────

def _draw_overview(ax, nodes, hexes, masks, n_faces, title='Hex mesh (top face)'):
    quads      = _top_quads(nodes, hexes)
    colors     = _color_quads(hexes, masks, n_faces)
    xy_mm      = nodes[:, :2] * 1e3

    coll = PolyCollection(quads, facecolors=list(colors),
                          edgecolors='k', linewidths=0.3, zorder=2)
    ax.add_collection(coll)
    ax.set_aspect('equal')
    span = max(xy_mm[:, 0].max() - xy_mm[:, 0].min(),
               xy_mm[:, 1].max() - xy_mm[:, 1].min()) * 0.08
    ax.set_xlim(xy_mm[:, 0].min() - span, xy_mm[:, 0].max() + span)
    ax.set_ylim(xy_mm[:, 1].min() - span, xy_mm[:, 1].max() + span)
    ax.set_xlabel('mm'); ax.set_ylabel('mm')
    ax.set_title(title, fontsize=10)

    handles = [mpatches.Patch(color=_face_colors(n_faces)[i], label=f'F{i}')
               for i in range(min(n_faces, 12))]
    ax.legend(handles=handles, fontsize=7, loc='upper right')


# ── hinge close-ups ───────────────────────────────────────────────────────────

def _hinge_bbox(nodes, hexes, masks, fi, fk, pad_frac=0.6):
    """Return (xmin, xmax, ymin, ymax) bounding box centred on the hinge."""
    xy_mm = nodes[:, :2] * 1e3
    mi = masks.get(f'f{fi}')
    mk = masks.get(f'f{fk}')
    pts = []
    for m in (mi, mk):
        if m is not None:
            pts.append(xy_mm[m.astype(bool)])
    if not pts:
        return None
    all_pts = np.vstack(pts)
    # Centre on centroid of union; zoom to face size
    cx = all_pts[:, 0].mean()
    cy = all_pts[:, 1].mean()
    span = max(all_pts[:, 0].max() - all_pts[:, 0].min(),
               all_pts[:, 1].max() - all_pts[:, 1].min()) * pad_frac
    return cx - span, cx + span, cy - span, cy + span


def _draw_hinge_closeup(ax, nodes, hexes, masks, n_faces, fi, fk, h_idx):
    quads  = _top_quads(nodes, hexes)
    colors = _color_quads(hexes, masks, n_faces)
    bb     = _hinge_bbox(nodes, hexes, masks, fi, fk)
    if bb is None:
        ax.set_visible(False)
        return

    xmin, xmax, ymin, ymax = bb
    coll = PolyCollection(quads, facecolors=list(colors),
                          edgecolors='k', linewidths=0.5, zorder=2)
    ax.add_collection(coll)
    ax.set_aspect('equal')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_title(f'Hinge {h_idx}: F{fi}↔F{fk}', fontsize=9)
    ax.set_xlabel('mm', fontsize=8)


# ── node connectivity check ───────────────────────────────────────────────────

def _check_hinge_profiles(nodes, hexes, hinge_pairs, strip_hex_ids_per_pair):
    """
    For each hinge (fi, fk), analyse the pre-identified strip hex elements
    and report quad width statistics to diagnose twisting.
    """
    all_quads = _top_quads(nodes, hexes)

    print("\n── Hinge strip analysis ─────────────────────────────────────────")
    for h, (fi, fk) in enumerate(hinge_pairs):
        ids = strip_hex_ids_per_pair.get(h, [])
        if not ids:
            print(f"  Hinge {h} (F{fi}↔F{fk}): no strip hexes found")
            continue

        strip_quads = all_quads[ids]   # (Q, 4, 2)

        widths = []
        for q in strip_quads:
            w1 = np.linalg.norm(q[1] - q[0])
            w2 = np.linalg.norm(q[2] - q[3])
            widths.append((w1 + w2) / 2)

        widths = np.array(widths)
        cv = widths.std() / widths.mean() if widths.mean() > 0 else 0
        print(f"  Hinge {h:2d} (F{fi:2d}↔F{fk:2d}): {len(strip_quads):3d} strip quads  "
              f"width mm: min={widths.min()*1e3:.3f}  max={widths.max()*1e3:.3f}  "
              f"cv={cv:.3f}  "
              f"{'TWISTED?' if cv > 0.15 else 'ok'}")
    print("──────────────────────────────────────────────────────────────────")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Visualize CS hex mesh before FEM.')
    parser.add_argument('--mesh',   default=None, help='Path to cs_mesh.npz')
    parser.add_argument('--config', default=None, help='Path to sofa YAML (builds fresh mesh)')
    parser.add_argument('--out',    default=None, help='Output PNG path (default: next to mesh/config)')
    args = parser.parse_args()

    if not args.mesh and not args.config:
        # Default to the most recent run
        runs = sorted(pathlib.Path(REPO / 'data' / 'outputs' / 'runs').glob('*/cs_mesh.npz'))
        if not runs:
            parser.error('No cs_mesh.npz found. Pass --mesh or --config.')
        args.mesh = str(runs[-1])
        print(f'Using most recent mesh: {args.mesh}')

    print('Loading mesh ...')
    nodes, hexes, masks, n_faces = _load_or_build_mesh(args)
    print(f'  {len(nodes)} nodes, {len(hexes)} hexes, {n_faces} faces')

    # Build per-face node sets.  Hinge-strip nodes belong to NO face mask — the
    # hinge strip is a separate set of hexes whose top-face nodes are not claimed
    # by any face.  We detect hinge-strip hexes by finding hexes whose top-face
    # nodes are split across two face masks (each node is in at most one face).
    face_node_ids = {}
    for i in range(n_faces):
        m = masks.get(f'f{i}')
        if m is not None:
            face_node_ids[i] = set(np.where(m.astype(bool))[0])

    # A hex is a FACE hex only if ALL 8 of its nodes belong to exactly one face.
    # A hex is a HINGE hex if its nodes span multiple faces or include unclaimed nodes.
    hinge_pairs = []
    hex_face_label = np.full(len(hexes), -1, dtype=np.int32)  # -1 = hinge strip
    for hi, h in enumerate(hexes):
        faces_hit = set()
        all_claimed = True
        for ni in h:
            found = False
            for fi, node_set in face_node_ids.items():
                if ni in node_set:
                    faces_hit.add(fi)
                    found = True
            if not found:
                all_claimed = False
        if len(faces_hit) == 1 and all_claimed:
            hex_face_label[hi] = next(iter(faces_hit))

    # Hinge hexes are those labeled -1. Cluster them into connected strips,
    # then identify the two adjacent face indices per cluster.
    hinge_hex_ids = np.where(hex_face_label == -1)[0]
    strip_hex_ids_per_pair: dict = {}  # hinge_pair_index → list of hex indices
    if len(hinge_hex_ids) > 0:
        # Build node→hex adjacency for hinge hexes
        from collections import defaultdict
        node_to_hinge_hex = defaultdict(list)
        for idx in hinge_hex_ids:
            for ni in hexes[idx]:
                node_to_hinge_hex[ni].append(idx)

        # BFS to cluster hinge hexes into connected strips
        visited = set()
        strip_clusters = []
        for start in hinge_hex_ids:
            if start in visited:
                continue
            cluster = []
            queue = [start]
            while queue:
                cur = queue.pop()
                if cur in visited:
                    continue
                visited.add(cur)
                cluster.append(cur)
                for ni in hexes[cur]:
                    for nb in node_to_hinge_hex[ni]:
                        if nb not in visited:
                            queue.append(nb)
            strip_clusters.append(cluster)

        # For each cluster, find the two adjacent face indices and record hex ids
        strip_hex_ids_per_cluster = {}  # cluster_index → list of hex indices
        for ci, cluster in enumerate(strip_clusters):
            strip_hex_ids_per_cluster[ci] = cluster
            cluster_nodes = set()
            for idx in cluster:
                cluster_nodes.update(int(n) for n in hexes[idx])
            adj_faces = set()
            for ni in cluster_nodes:
                for fi, node_set in face_node_ids.items():
                    if ni in node_set:
                        adj_faces.add(fi)
            if len(adj_faces) == 2:
                pair = tuple(sorted(adj_faces))
                if pair not in hinge_pairs:
                    hinge_pairs.append(pair)
                    strip_hex_ids_per_pair[len(hinge_pairs) - 1] = cluster
            elif len(adj_faces) > 2:
                adj_list = sorted(adj_faces)
                for ii in range(len(adj_list)):
                    for jj in range(ii+1, len(adj_list)):
                        pair = (adj_list[ii], adj_list[jj])
                        if pair not in hinge_pairs:
                            hinge_pairs.append(pair)
                            strip_hex_ids_per_pair[len(hinge_pairs) - 1] = cluster
    else:
        # Fallback: shared boundary nodes (for parametric meshes)
        for fi in range(n_faces):
            for fk in range(fi+1, n_faces):
                mi = masks.get(f'f{fi}')
                mk = masks.get(f'f{fk}')
                if mi is None or mk is None:
                    continue
                shared = (mi.astype(bool)) & (mk.astype(bool))
                if shared.sum() >= 2:
                    hinge_pairs.append((fi, fk))

    print(f'  Detected {len(hinge_pairs)} hinge pairs: {hinge_pairs}')
    _check_hinge_profiles(nodes, hexes, hinge_pairs, strip_hex_ids_per_pair)

    # ── figure layout ─────────────────────────────────────────────────────────
    n_hinges    = len(hinge_pairs)
    n_closeup_cols = min(n_hinges, 4)
    n_closeup_rows = (n_hinges + n_closeup_cols - 1) // n_closeup_cols if n_hinges else 0

    total_rows = 1 + n_closeup_rows
    fig = plt.figure(figsize=(max(14, 3.5 * n_closeup_cols), 5 * total_rows))

    # Overview spanning full width
    ax_overview = fig.add_subplot(total_rows, 1, 1)
    _draw_overview(ax_overview, nodes, hexes, masks, n_faces,
                   title=f'Hex mesh top view (before FEM) — {len(nodes)} nodes, {len(hexes)} hexes')

    # Close-up per hinge
    for h, (fi, fk) in enumerate(hinge_pairs):
        row = 1 + h // n_closeup_cols
        col = h % n_closeup_cols
        ax = fig.add_subplot(total_rows, n_closeup_cols, n_closeup_cols + 1 + h)
        _draw_hinge_closeup(ax, nodes, hexes, masks, n_faces, fi, fk, h)

    fig.tight_layout()

    # Output path
    if args.out:
        out_path = pathlib.Path(args.out)
    elif args.mesh:
        out_path = pathlib.Path(args.mesh).parent / 'mesh_visualization.png'
    else:
        out_path = REPO / 'sofa' / 'output' / 'mesh_visualization.png'
        out_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'\nSaved → {out_path}')


if __name__ == '__main__':
    main()
