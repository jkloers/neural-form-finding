"""Animate how the closed-state design evolves during training.

Snapshots the design parameters (aspect ratios + boundary) every few epochs while
training the closed_les pipeline, then renders an animation of the flat Stage-0
sheet (the cut pattern's geometry) next to its physics deployment, with the
size-adaptive fitted target circle. Shows the inverse design "learning" the sheet.

Run:
    JAX_PLATFORMS=cpu conda run -n kgnn_mac \
        python nff/scripts/animate_training.py --config-name circlefit_moment_full --every 8
"""

import os
import shutil
import datetime
import argparse

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle

from nff.config.experiment import load_and_parse_config
from nff.stages.pipeline import forward_pipeline
from nff.stages.geometry import reconstruct_vertices
from nff.training.trainer import create_train_step, TrainState
from nff.utils.visualization import plot_tessellation
from nff.scripts.train import _build_initial_state, _init_map_params
from nff.scripts.deploy_closed_tessellation import deformed_vertices


def fit_circle(cloud):
    n = cloud.shape[0]
    A = np.concatenate([2.0 * cloud, np.ones((n, 1))], axis=1)
    rhs = np.sum(cloud ** 2, axis=1)
    cx, cy, c = np.linalg.solve(A.T @ A + 1e-8 * np.eye(3), A.T @ rhs)
    return np.array([cx, cy]), float(np.sqrt(max(c + cx * cx + cy * cy, 1e-12)))


def global_verts(tess, node_positions):
    v = np.array(tess.vertices, dtype=float)
    for f_id, face in enumerate(tess.faces):
        for local, gv in enumerate(face.vertex_indices):
            v[gv] = node_positions[f_id, local]
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-dir", default="closed")
    ap.add_argument("--config-name", default="circlefit_moment_full")
    ap.add_argument("--every", type=int, default=8, help="snapshot interval (epochs)")
    args = ap.parse_args()

    config = load_and_parse_config(f"data/configs/{args.config_dir}/{args.config_name}.yaml")
    initial_state, tessellation = _build_initial_state(config)
    params, static_features = _init_map_params(config, initial_state)
    load_specs = config.topology.get('loads', [])
    circle_fit = config.training.geometric_loss_type == "circle_fit"

    optimizer, train_step_fn = create_train_step(
        initial_state, config.target, config.validity, config.physics, config.training,
        map_type=config.mapping.type, use_jit=True,
        load_specs=load_specs, static_features=static_features,
    )
    state = TrainState(params=params, opt_state=optimizer.init(params),
                       rng=jax.random.PRNGKey(0))

    # Train, snapshotting params (jnp arrays are immutable, so the reference is a
    # valid snapshot) every `every` epochs.
    snaps = [(0, state.params)]
    for epoch in range(config.training.num_epochs):
        state, loss, aux = train_step_fn(state)
        if (epoch + 1) % args.every == 0 or epoch == config.training.num_epochs - 1:
            snaps.append((epoch + 1, state.params))
    print(f"captured {len(snaps)} snapshots")

    # Convert each snapshot to flat + deployed geometry.
    frames = []
    for ep, p in snaps:
        res = forward_pipeline(initial_state, config.target, config.validity, config.physics,
                               map_type=config.mapping.type, map_params=p,
                               static_features=static_features, load_specs=load_specs)
        ms, vs = res['mapped_state'], res['valid_state']
        disp = res['solution'].fields[-1]
        flat = global_verts(tessellation, np.asarray(reconstruct_vertices(ms.face_centroids, ms.centroid_node_vectors)))
        dep = global_verts(tessellation, deformed_vertices(vs, disp))
        b = np.asarray(vs.boundary_face_node_ids)
        fc = np.asarray(vs.face_centroids) + np.asarray(disp[:, :2])
        th = np.asarray(disp[:, 2]); cnv = np.asarray(vs.centroid_node_vectors)
        bf, bl = b[:, 0], b[:, 1]; vec = cnv[bf, bl]
        ct, stt = np.cos(th[bf]), np.sin(th[bf])
        cloud = fc[bf] + np.stack([ct * vec[:, 0] - stt * vec[:, 1], stt * vec[:, 0] + ct * vec[:, 1]], axis=-1)
        cc, rr = fit_circle(cloud) if circle_fit else (np.asarray(config.target.center, float), float(config.target.radius))
        frames.append((ep, flat, dep, cc, rr))

    # Fixed camera bounds.
    all_flat = np.concatenate([f[1] for f in frames]); all_dep = np.concatenate([f[2] for f in frames])
    def bounds(a, pad=0.5):
        return (a[:, 0].min() - pad, a[:, 0].max() + pad, a[:, 1].min() - pad, a[:, 1].max() + pad)
    fb, db = bounds(all_flat), bounds(all_dep)

    tflat, tdep = tessellation.copy(), tessellation.copy()
    fig, (axf, axd) = plt.subplots(1, 2, figsize=(15, 7.6), facecolor="white")

    def draw(k):
        ep, flat, dep, cc, rr = frames[k]
        axf.clear(); axd.clear()
        tflat.update_vertices(flat)
        plot_tessellation(tflat, ax=axf, show_target=False, show_hinges=False,
                          show_face_indices=False, show_hinge_indices=False, color_faces="#F58025")
        axf.set_xlim(fb[0], fb[1]); axf.set_ylim(fb[2], fb[3]); axf.set_aspect("equal"); axf.axis("off")
        axf.set_title(f"Flat design (Stage 0) — epoch {ep}")
        tdep.update_vertices(dep)
        plot_tessellation(tdep, ax=axd, show_target=False, show_hinges=False,
                          show_face_indices=False, show_hinge_indices=False, color_faces="#F58025")
        axd.add_patch(Circle(cc, rr, fill=False, edgecolor="#D62828", lw=2.0, ls="--"))
        axd.set_xlim(db[0], db[1]); axd.set_ylim(db[2], db[3]); axd.set_aspect("equal"); axd.axis("off")
        axd.set_title(f"Physics deploy — epoch {ep}")

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("data", "outputs", "runs", f"run_{ts}_{args.config_name}_training")
    os.makedirs(run_dir, exist_ok=True)
    shutil.copy(f"data/configs/{args.config_dir}/{args.config_name}.yaml", os.path.join(run_dir, "config.yaml"))
    ani = animation.FuncAnimation(fig, draw, frames=len(frames), blit=False)
    out = os.path.join(run_dir, "training_evolution.gif")
    ani.save(out, writer="pillow", fps=6)
    plt.close(fig)
    print(f"saved {out}")

    # Static strip of a few snapshots for quick inspection.
    picks = [0, len(frames) // 3, 2 * len(frames) // 3, len(frames) - 1]
    figs, axs = plt.subplots(2, len(picks), figsize=(5 * len(picks), 9.5), facecolor="white")
    for col, k in enumerate(picks):
        ep, flat, dep, cc, rr = frames[k]
        tflat.update_vertices(flat); tdep.update_vertices(dep)
        plot_tessellation(tflat, ax=axs[0, col], show_target=False, show_hinges=False,
                          show_face_indices=False, show_hinge_indices=False, color_faces="#F58025")
        axs[0, col].set_xlim(fb[0], fb[1]); axs[0, col].set_ylim(fb[2], fb[3])
        axs[0, col].set_aspect("equal"); axs[0, col].axis("off"); axs[0, col].set_title(f"flat — epoch {ep}")
        plot_tessellation(tdep, ax=axs[1, col], show_target=False, show_hinges=False,
                          show_face_indices=False, show_hinge_indices=False, color_faces="#F58025")
        axs[1, col].add_patch(Circle(cc, rr, fill=False, edgecolor="#D62828", lw=2.0, ls="--"))
        axs[1, col].set_xlim(db[0], db[1]); axs[1, col].set_ylim(db[2], db[3])
        axs[1, col].set_aspect("equal"); axs[1, col].axis("off"); axs[1, col].set_title(f"deploy — epoch {ep}")
    strip = os.path.join(run_dir, "training_evolution_strip.png")
    figs.savefig(strip, dpi=130, bbox_inches="tight"); plt.close(figs)
    print(f"saved {strip}")


if __name__ == "__main__":
    main()
