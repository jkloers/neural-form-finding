"""Physics-driven inverse design of a closed RDPQK sheet — with per-stage audit.

Trains the closed_les design variables (aspect ratios + boundary points) so the
physics-deployed boundary matches the config target, and writes everything to a
timestamped run folder: data/outputs/runs/run_<ts>_<config_name>/.

Per-stage validity audit (Stage 1 is deactivated, so Stage-0 output == Stage-2
input):
    Stage 0  — flat sheet straight from the LES builder.
    Stage 2  — deployed sheet after the physics solver.
Each is checked for face overlap / coverage with Shapely, to localize whether
invalid (overlapping) configurations come from the builder or the physics.

Run:
    JAX_PLATFORMS=cpu conda run -n kgnn_mac \
        python nff/scripts/visualize_closed_physics.py --config-name circlefit_moment
"""

import os
import shutil
import datetime
import argparse

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from shapely.geometry import Polygon
from shapely.ops import unary_union

from nff.config.experiment import load_and_parse_config
from nff.stages.pipeline import forward_pipeline
from nff.training.trainer import train_pipeline
from nff.utils.visualization import plot_tessellation, animate_tessellation
from nff.scripts.train import _build_initial_state, _init_map_params
from nff.scripts.deploy_closed_tessellation import deformed_vertices, write_deformed_into


def fit_circle(cloud):
    n = cloud.shape[0]
    A = np.concatenate([2.0 * cloud, np.ones((n, 1))], axis=1)
    rhs = np.sum(cloud ** 2, axis=1)
    cx, cy, c = np.linalg.solve(A.T @ A + 1e-8 * np.eye(3), A.T @ rhs)
    center = np.array([cx, cy])
    radius = float(np.sqrt(max(c + cx * cx + cy * cy, 1e-12)))
    return center, radius


def overlap_report(tess):
    """Shapely audit, robust to invalid geometry.

    Distinguishes two failure modes:
      - twisted_faces: individual panels that are self-intersecting (bow-tie quads);
      - overlap_total : area of inter-face overlap (panels passing through each other).
    buffer(0) repairs self-intersections so the union can still be computed.
    """
    raw = [Polygon(tess.vertices[f.vertex_indices]) for f in tess.faces]
    twisted = sum(0 if p.is_valid else 1 for p in raw)
    polys = [p if p.is_valid else p.buffer(0) for p in raw]
    sum_area = sum(p.area for p in polys)
    try:
        union = unary_union(polys)
        union_area = union.area
        bbox = union.bounds
        coverage = union_area / ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) + 1e-12)
    except Exception:
        union_area, coverage = float("nan"), float("nan")
    max_pair = 0.0
    for a in range(len(polys)):
        for b in range(a + 1, len(polys)):
            try:
                max_pair = max(max_pair, polys[a].intersection(polys[b]).area)
            except Exception:
                pass
    return {
        "twisted_faces": twisted,
        "overlap_total": sum_area - union_area,
        "overlap_max_pair": max_pair,
        "coverage": coverage,
        "sum_area": sum_area,
        "union_area": union_area,
    }


def snapshots(config, params, initial_state, tessellation):
    """Return (stage0_tess, stage2_tess, boundary_cloud, fields) for given params."""
    result = forward_pipeline(
        initial_state, config.target, config.validity, config.physics,
        map_type=config.mapping.type, map_params=params,
        static_features=snapshots.static_features,
        load_specs=config.topology.get('loads', []),
    )
    vs = result['valid_state']                       # == Stage-0 output (Stage 1 off)
    disp = result['solution'].fields[-1]
    stage0 = write_deformed_into(tessellation, deformed_vertices(vs, jnp.zeros_like(disp)))
    stage2 = write_deformed_into(tessellation, deformed_vertices(vs, disp))

    b_ids = np.asarray(vs.boundary_face_node_ids)
    fc = np.asarray(vs.face_centroids) + np.asarray(disp[:, :2])
    th = np.asarray(disp[:, 2])
    cnv = np.asarray(vs.centroid_node_vectors)
    bf, bl = b_ids[:, 0], b_ids[:, 1]
    vec = cnv[bf, bl]
    ct, st = np.cos(th[bf]), np.sin(th[bf])
    rot = np.stack([ct * vec[:, 0] - st * vec[:, 1], st * vec[:, 0] + ct * vec[:, 1]], axis=-1)
    cloud = fc[bf] + rot
    return stage0, stage2, cloud, result['solution'].fields


def plot_stage(ax, tess, title, rep):
    plot_tessellation(tess, ax=ax, show_target=False, show_hinges=False,
                      show_face_indices=False, show_hinge_indices=False, color_faces="#F58025")
    ax.set_aspect("equal")
    ax.axis("off")
    ok = rep["overlap_total"] < 1e-6 and rep["twisted_faces"] == 0
    ax.set_title(f"{title}\n{'VALID' if ok else 'INVALID'}  "
                 f"twisted={rep['twisted_faces']}  overlap={rep['overlap_total']:.2e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-dir", default="closed")
    ap.add_argument("--config-name", default="circlefit_moment")
    args = ap.parse_args()

    config_path = f"data/configs/{args.config_dir}/{args.config_name}.yaml"
    config = load_and_parse_config(config_path)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("data", "outputs", "runs", f"run_{ts}_{args.config_name}")
    os.makedirs(run_dir, exist_ok=True)
    shutil.copy(config_path, os.path.join(run_dir, "config.yaml"))
    print(f"Run directory: {run_dir}")

    initial_state, tessellation = _build_initial_state(config)
    params0, static_features = _init_map_params(config, initial_state)
    snapshots.static_features = static_features
    circle_fit_mode = config.training.geometric_loss_type == "circle_fit"

    s0_i, s2_i, cloud_i, _ = snapshots(config, params0, initial_state, tessellation)

    params_opt, history = train_pipeline(
        params0, initial_state, config.target, config.validity, config.physics,
        config.training, map_type=config.mapping.type, use_jit=True,
        load_specs=config.topology.get('loads', []), static_features=static_features,
    )
    s0_t, s2_t, cloud_t, fields_t = snapshots(config, params_opt, initial_state, tessellation)

    # ── Per-stage validity audit ──
    rep = {
        "stage0_init": overlap_report(s0_i), "stage2_init": overlap_report(s2_i),
        "stage0_trained": overlap_report(s0_t), "stage2_trained": overlap_report(s2_t),
    }
    print("\n── Validity audit (0 = valid tiling) ──")
    for k, v in rep.items():
        print(f"  {k:16s}: twisted_faces={v['twisted_faces']:3d}  "
              f"overlap_total={v['overlap_total']:.3e}  max_pair={v['overlap_max_pair']:.3e}  "
              f"coverage={v['coverage']:.3f}")

    # ── Stages figure (Stage 0 vs Stage 2, initial vs trained) ──
    fig, ax = plt.subplots(2, 2, figsize=(13, 13), facecolor="white")
    plot_stage(ax[0, 0], s0_i, "Initial — Stage 0 (builder / Stage-2 input)", rep["stage0_init"])
    plot_stage(ax[0, 1], s2_i, "Initial — Stage 2 (physics deploy)", rep["stage2_init"])
    plot_stage(ax[1, 0], s0_t, "Trained — Stage 0 (builder / Stage-2 input)", rep["stage0_trained"])
    plot_stage(ax[1, 1], s2_t, "Trained — Stage 2 (physics deploy)", rep["stage2_trained"])
    fig.savefig(os.path.join(run_dir, "stages.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Composite (flat | init deploy | trained deploy + fitted circle | loss) ──
    fig, axc = plt.subplots(1, 4, figsize=(26, 6.8), facecolor="white")
    plot_tessellation(s0_t, ax=axc[0], show_target=False, show_hinges=False,
                      show_face_indices=False, show_hinge_indices=False, color_faces="#F58025")
    axc[0].set_title("Trained flat design (Stage 0)")
    for a, dep, cloud, title in ((axc[1], s2_i, cloud_i, "Initial — physics deploy"),
                                 (axc[2], s2_t, cloud_t, "Trained — physics deploy")):
        plot_tessellation(dep, ax=a, show_target=False, show_hinges=False,
                          show_face_indices=False, show_hinge_indices=False, color_faces="#F58025")
        a.scatter(cloud[:, 0], cloud[:, 1], s=16, color="#457B9D", zorder=6)
        cc, rr = fit_circle(cloud) if circle_fit_mode else (np.asarray(config.target.center, float),
                                                            float(config.target.radius))
        a.add_patch(Circle(cc, rr, fill=False, edgecolor="#D62828", lw=2.0, ls="--", zorder=7))
        a.set_title(title)
    for a in axc[:3]:
        a.set_aspect("equal"); a.axis("off")
    axc[3].plot([h.get('chamfer_total', np.nan) for h in history], color="#F58025", lw=2)
    axc[3].set_xlabel("epoch"); axc[3].set_ylabel("geometric loss")
    axc[3].set_title("Training"); axc[3].grid(alpha=0.3)
    fig.savefig(os.path.join(run_dir, "composite.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Deployment animation (trained) ──
    frames = [np.array(s0_t.vertices, dtype=float)]
    vs_t = forward_pipeline(initial_state, config.target, config.validity, config.physics,
                            map_type=config.mapping.type, map_params=params_opt,
                            static_features=static_features,
                            load_specs=config.topology.get('loads', []))['valid_state']
    for k in range(fields_t.shape[0]):
        nodes = deformed_vertices(vs_t, fields_t[k])
        verts = np.array(tessellation.vertices, dtype=float)
        for f_id, face in enumerate(tessellation.faces):
            for local, gv in enumerate(face.vertex_indices):
                verts[gv] = nodes[f_id, local]
        frames.append(verts)
    animate_tessellation(tessellation, frames, filepath=os.path.join(run_dir, "deploy.gif"),
                         fps=10, show_hinges=False, show_face_indices=False,
                         show_hinge_indices=False, show_target=False, color_faces="#F58025")
    print(f"\nArtifacts written to {run_dir}")


if __name__ == "__main__":
    main()
