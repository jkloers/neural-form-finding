"""Driver for the closed-state (closed_les) inverse-design pipeline.

Trains ONCE (snapshotting design params during training) and writes EVERY visual
into a single run folder:

    data/outputs/runs/run_<ts>_<config>/
        config.yaml
        training_loss.png
        stage_0_initial_mapping.png
        stage_2_static_equilibrium.png        (target = fitted circle / rectangle)
        animation.gif                          (deployment)
        energy_balance.png
        cut_pattern.png                        (kirigami cut sheet)
        stages.png                             (panel area change: flat vs deployed)
        loading_diagram.png                    (BC / load schematic)
        training_evolution.gif                 (design morphing over epochs)

Run:
    JAX_PLATFORMS=cpu conda run -n kgnn_mac \
        python nff/scripts/run_closed.py --config-name circle_chamfer_10x10_best --every 6
"""

import os
import math
import shutil
import datetime
import argparse

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
from matplotlib.patches import Circle, Rectangle
from shapely.geometry import Polygon
from shapely.ops import unary_union

from nff.config.experiment import load_and_parse_config, TargetConfig
from nff.config.targets import get_target_points
from nff.stages.pipeline import forward_pipeline
from nff.stages.geometry import reconstruct_vertices, compute_face_areas, deformed_vertices
from nff.stages.physics.kinematics import face_to_node_kinematics_fn
from nff.training.trainer import create_train_step, TrainState
from nff.utils.visualization import (
    write_deformed_into, plot_loading_diagram, plot_area_change, animate_closed_evolution,
)
from nff.utils.pipeline_viz import visualize_pipeline_results, plot_loss_history
from nff.topology.closed_builder_jax import solve_cut_vertices_jax, boundary_flat_from_logits
from nff.scripts.closed_paper_export import write_cut_patterns
from nff.scripts.closed_setup import (build_closed_initial_state, init_closed_les_params,
                                      build_surrogate_energy)


def _fit_circle(cloud):
    """Algebraic (Kåsa) best-fit circle of a point cloud -> (center, radius)."""
    n = cloud.shape[0]
    A = np.concatenate([2.0 * cloud, np.ones((n, 1))], axis=1)
    cx, cy, c = np.linalg.solve(A.T @ A + 1e-8 * np.eye(3), A.T @ np.sum(cloud ** 2, axis=1))
    return np.array([cx, cy]), float(np.sqrt(max(c + cx * cx + cy * cy, 1e-12)))


def _boundary_cloud(vs, disp):
    """Deployed boundary-vertex positions from a valid state + displacement field."""
    b = np.asarray(vs.boundary_face_node_ids)
    fc = np.asarray(vs.face_centroids) + np.asarray(disp[:, :2])
    th = np.asarray(disp[:, 2])
    cnv = np.asarray(vs.centroid_node_vectors)
    bf, bl = b[:, 0], b[:, 1]
    vec = cnv[bf, bl]
    ct, st = np.cos(th[bf]), np.sin(th[bf])
    return fc[bf] + np.stack([ct * vec[:, 0] - st * vec[:, 1], st * vec[:, 0] + ct * vec[:, 1]], axis=-1)


def _overlap(tess):
    """(twisted_face_count, total_inter-face overlap area) via shapely."""
    polys = [Polygon(tess.vertices[f.vertex_indices]) for f in tess.faces]
    twisted = sum(0 if p.is_valid else 1 for p in polys)
    polys = [p if p.is_valid else p.buffer(0) for p in polys]
    return twisted, sum(p.area for p in polys) - unary_union(polys).area


def _global_verts(tess, node_positions):
    """Flatten per-face node positions into the global vertex array of ``tess``."""
    v = np.array(tess.vertices, dtype=float)
    for f_id, face in enumerate(tess.faces):
        for local, gv in enumerate(face.vertex_indices):
            v[gv] = node_positions[f_id, local]
    return v


def _deployed_hinge_xy(vs, disp):
    """(n_hinges, 2) deployed hinge midpoints, in bond_connectivity order (= per-hinge D order).

    Each bond connects two NODES; the deployed node positions come from the rigid-tile kinematics
    (face centroid + rotated centroid->node vector + displacement), reshaped to the flat node index
    space that ``bond_connectivity`` indexes.
    """
    fc = np.asarray(vs.face_centroids) + np.asarray(disp[:, :2])
    th = np.asarray(disp[:, 2]); ct, st = np.cos(th), np.sin(th)
    cnv = np.asarray(vs.centroid_node_vectors)                      # (nf, nn, 2)
    rx = ct[:, None] * cnv[:, :, 0] - st[:, None] * cnv[:, :, 1]
    ry = st[:, None] * cnv[:, :, 0] + ct[:, None] * cnv[:, :, 1]
    node_xy = (fc[:, None, :] + np.stack([rx, ry], -1)).reshape(-1, 2)   # (nf*nn, 2)
    bc = np.asarray(vs.bond_connectivity)
    return 0.5 * (node_xy[bc[:, 0]] + node_xy[bc[:, 1]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-dir", default="closed")
    ap.add_argument("--config-name", default="circle_chamfer_10x10_best")
    ap.add_argument("--every", type=int, default=6)
    args = ap.parse_args()

    cfg_path = f"data/configs/{args.config_dir}/{args.config_name}.yaml"
    config = load_and_parse_config(cfg_path)
    circle_fit = config.training.geometric_loss_type == "circle_fit"
    load_specs = config.topology.get('loads', [])

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("data", "outputs", "runs", f"run_{ts}_{args.config_name}")
    os.makedirs(run_dir, exist_ok=True)
    shutil.copy(cfg_path, os.path.join(run_dir, "config.yaml"))
    print(f"Run directory: {run_dir}")

    initial_state, tessellation = build_closed_initial_state(config)
    params, static_features = init_closed_les_params(config)
    bond_energy, stability_fn, geometry_fn, damage_fn, w_lig_logit0 = build_surrogate_energy(
        config, static_features, initial_state, params)
    if w_lig_logit0 is not None:                        # learnable per-hinge ligament width (option A)
        params = {**params, 'w_lig_logit': w_lig_logit0}
    # design-dependent HingeGeometry for the non-training deploys (target sizing + final visuals)
    def _geom_at(mp):
        return geometry_fn(mp) if geometry_fn is not None else None

    # ── Target. circle_fit: size-free (the loss fits its own circle). Otherwise a
    # FIXED target (circle or rectangle) sized to the untrained physics deploy,
    # matched with the bidirectional (reciprocal) chamfer. ──
    rect_mode = config.target.type == 'rectangle'
    target_cloud_override = None
    rect_geom = None
    if circle_fit:
        target_eff = config.target
    else:
        res0 = forward_pipeline(initial_state, config.target, config.validity, config.physics,
                                map_type=config.mapping.type, map_params=params,
                                static_features=static_features, load_specs=load_specs,
                                bond_energy_fn=bond_energy, hinge_geometry=_geom_at(params))
        cl0 = _boundary_cloud(res0['valid_state'], res0['solution'].fields[-1])
        sc = float(config.topology.get('target_radius_scale', 1.0))
        if rect_mode:
            if 'target_half_w' in config.topology:
                # FIXED rectangle straight from config — does NOT adapt to the deploy.
                cen0 = np.array([float(config.topology.get('target_cx', config.target.center[0])),
                                 float(config.topology.get('target_cy', config.target.center[1]))])
                hw = float(config.topology['target_half_w'])
                hh = float(config.topology['target_half_h'])
            else:
                # Auto-fit: bounding box of the (untrained) deployed boundary, scaled.
                mn, mx = cl0.min(axis=0), cl0.max(axis=0)
                C = (mn + mx) / 2.0
                scx = float(config.topology.get('target_scale_x', sc))
                scy = float(config.topology.get('target_scale_y', sc))
                # Anchor the (shrunk) target at the CLAMPED tile so the pinned clamp stays INSIDE it:
                # scale the bbox about the clamp centroid, not the bbox centre. With scale<1 this
                # shrinks toward the clamp (stiffening incentive on the free/pulled side) while keeping
                # the clamp enclosed (the clamp is a fixed point of the scaling). No clamp -> centre.
                clamp_faces = config.topology.get('bc_clamped') or []
                clamp_faces = clamp_faces if isinstance(clamp_faces, list) else []
                if clamp_faces:
                    P = np.asarray(initial_state.face_centroids)[np.asarray(clamp_faces, int)].mean(0)
                    cen0 = P + np.array([scx, scy]) * (C - P)
                else:
                    cen0 = C
                hw, hh = (mx[0] - mn[0]) / 2.0 * scx, (mx[1] - mn[1]) / 2.0 * scy
            rect_geom = (cen0, hw, hh)
            target_cloud_override = get_target_points(
                {'type': 'rectangle', 'center': cen0, 'radius': max(hw, hh),
                 'half_w': hw, 'half_h': hh}, n_points=400)
            target_eff = TargetConfig(type='circle',
                                      center=(float(cen0[0]), float(cen0[1])), radius=float(max(hw, hh)))
            print(f"  fixed target RECTANGLE: center=({cen0[0]:.2f}, {cen0[1]:.2f})  "
                  f"half_w={hw:.2f} half_h={hh:.2f}")
        else:
            cen0, rad0 = _fit_circle(cl0)
            rad0 *= sc
            target_eff = TargetConfig(type='circle',
                                      center=(float(cen0[0]), float(cen0[1])), radius=float(rad0))
            print(f"  fixed target circle (deploy best-fit): center=({cen0[0]:.2f}, {cen0[1]:.2f})  radius={rad0:.3f}")

    # ── Train once, snapshotting params; track best (min chamfer). ──
    optimizer, step = create_train_step(
        initial_state, target_eff, config.validity, config.physics, config.training,
        map_type=config.mapping.type, use_jit=True,
        load_specs=load_specs, static_features=static_features,
        target_cloud=target_cloud_override, bond_energy_fn=bond_energy, stability_fn=stability_fn,
        hinge_geometry_fn=geometry_fn)
    state = TrainState(params=params, opt_state=optimizer.init(params), rng=jax.random.PRNGKey(0))

    history, snaps = [], [(0, state.params)]
    best = (math.inf, state.params)
    for epoch in range(config.training.num_epochs):
        state, loss, aux = step(state)
        history.append(aux)
        ch = float(aux.get('chamfer_total', math.inf))
        if math.isfinite(ch) and ch < best[0]:
            best = (ch, state.params)
        if (epoch + 1) % args.every == 0 or epoch == config.training.num_epochs - 1:
            snaps.append((epoch + 1, state.params))
        if epoch % 25 == 0 or epoch == config.training.num_epochs - 1:
            msg = f"  epoch {epoch:3d}  loss={float(loss):.4e}  chamfer={ch:.4e}"
            if aux.get('hinge_max_D') is not None:    # surrogate damage metrics, when tracked
                msg += (f"  maxD={float(aux['hinge_max_D']):.2f} meanD={float(aux['hinge_mean_D']):.2f}"
                        f" p90D={float(aux['hinge_p90_D']):.2f} over={int(aux['hinge_n_over'])}")
            print(msg)
    best_params = best[1]
    print(f"  best chamfer={best[0]:.4e}")
    import pickle
    with open(os.path.join(run_dir, "best_params.pkl"), "wb") as _pf:
        pickle.dump({k: np.asarray(v) for k, v in best_params.items()}, _pf)  # trained design, for analysis
    geom_best = _geom_at(best_params)                         # final design's HingeGeometry
    hinge_w_lig_best = geom_best.w_lig if geom_best is not None else None
    if hinge_w_lig_best is not None and isinstance(best_params, dict) and 'w_lig_logit' in best_params:
        print(f"  learned w_lig [mm]: min={float(hinge_w_lig_best.min()):.2f} "
              f"mean={float(hinge_w_lig_best.mean()):.2f} max={float(hinge_w_lig_best.max()):.2f}")

    result = forward_pipeline(initial_state, config.target, config.validity, config.physics,
                              map_type=config.mapping.type, map_params=best_params,
                              static_features=static_features, load_specs=load_specs,
                              bond_energy_fn=bond_energy, hinge_geometry=geom_best)

    # ── Target params for the pretty viz (fitted circle is size-adaptive). ──
    vs, disp = result['valid_state'], result['solution'].fields[-1]
    if circle_fit:
        cc, rr = _fit_circle(_boundary_cloud(vs, disp))
        target_params = {'type': 'circle', 'center': [float(cc[0]), float(cc[1])], 'radius': rr}
    elif rect_mode:
        cen0, hw, hh = rect_geom
        target_params = {'type': 'rectangle', 'center': [float(cen0[0]), float(cen0[1])],
                         'radius': float(max(hw, hh)), 'half_w': float(hw), 'half_h': float(hh)}
    else:
        target_params = {'type': 'circle', 'center': target_eff.center, 'radius': target_eff.radius}

    # ── Pretty pipeline visuals + loss history. ──
    plot_loss_history(history, config, run_dir=run_dir)
    visualize_pipeline_results(result, tessellation, config, target_params,
                               args.config_name, run_dir=run_dir, load_specs=load_specs)

    # ── Cut patterns: precise per-hinge PNG + true 1:1 A4 print PDF (kerf slots, ligaments,
    #    fillets + HOLD/PULL marks for the two-hands pinch test). See closed_paper_export. ──
    struct, sliders = static_features['struct'], static_features['sliders']
    cut_coords = np.asarray(solve_cut_vertices_jax(
        struct, boundary_flat_from_logits(sliders, best_params['bnd_logits']),
        jax.nn.sigmoid(best_params['z'])))
    write_cut_patterns(run_dir, initial_state=initial_state, cut_coords=cut_coords, struct=struct,
                       config=config, hinge_model=config.hinge_model, load_specs=load_specs,
                       config_name=args.config_name, hinge_w_lig=hinge_w_lig_best)

    # ── Validity audit + per-panel area change (flat vs deployed). ──
    s0 = write_deformed_into(tessellation, deformed_vertices(vs, jnp.zeros_like(disp)))
    s2 = write_deformed_into(tessellation, deformed_vertices(vs, disp))
    (t0, o0), (t2, o2) = _overlap(s0), _overlap(s2)
    print(f"  validity: stage0 twisted={t0} overlap={o0:.2e} | stage2 twisted={t2} overlap={o2:.2e}")

    # The closed_les map rebuilds panel shapes from the learned per-cut r, so the
    # trained design has non-uniform panels vs the uniform initial sheet. Area is
    # taken from centroid_node_vectors (the rigid panel shape) — NOT the shared-
    # vertex tessellation copy, whose closed-sheet corners alias.
    a_init = np.asarray(compute_face_areas(initial_state.centroid_node_vectors))
    a_trained = np.asarray(compute_face_areas(result['mapped_state'].centroid_node_vectors))
    rel = a_trained / np.maximum(a_init, 1e-9) - 1.0
    print(f"  trained-vs-initial face area: min={rel.min():+.2%} max={rel.max():+.2%}")
    plot_area_change(s0, s2, rel, os.path.join(run_dir, "stages.png"))

    # One clean loading schematic.
    plot_loading_diagram(s2, config.topology.get('bc_clamped', []), load_specs,
                         os.path.join(run_dir, "loading_diagram.png"))

    # ── Training-evolution animation. ──
    frames = []
    for ep, p in snaps:
        geom_p = _geom_at(p)
        res = forward_pipeline(initial_state, config.target, config.validity, config.physics,
                               map_type=config.mapping.type, map_params=p,
                               static_features=static_features, load_specs=load_specs,
                               bond_energy_fn=bond_energy, hinge_geometry=geom_p)
        m, v = res['mapped_state'], res['valid_state']
        d = res['solution'].fields[-1]
        flat = _global_verts(tessellation, np.asarray(reconstruct_vertices(m.face_centroids, m.centroid_node_vectors)))
        dep = _global_verts(tessellation, np.asarray(deformed_vertices(v, d)))
        # per-hinge ductile damage D + deployed hinge positions, for the colored-dot overlay
        hinge_xy = D = None
        if damage_fn is not None and geom_p is not None:
            _cnv = v.centroid_node_vectors; _nf, _nn, _ = _cnv.shape
            _nd = face_to_node_kinematics_fn(d, _cnv).reshape(_nf * _nn, 3)
            D = np.asarray(damage_fn(_nd, geom_p, res['reference_bond_vectors']))
            hinge_xy = _deployed_hinge_xy(v, d)
        frames.append((ep, flat, dep, _boundary_cloud(v, d), hinge_xy, D))

    def add_target(ax, cloud):
        if circle_fit:
            c, r = _fit_circle(cloud)
            ax.add_patch(Circle(c, r, fill=False, edgecolor="#D62828", lw=2.0, ls="--"))
        elif rect_mode:
            cen0, hw, hh = rect_geom
            ax.add_patch(Rectangle((cen0[0] - hw, cen0[1] - hh), 2 * hw, 2 * hh,
                                   fill=False, edgecolor="#D62828", lw=2.0, ls="--"))
        else:
            ax.add_patch(Circle(target_eff.center, target_eff.radius,
                                fill=False, edgecolor="#D62828", lw=2.0, ls="--"))

    fail_line = float(getattr(config.hinge_model, 'fail_line', 1.0))
    animate_closed_evolution(tessellation, frames, add_target,
                             os.path.join(run_dir, "training_evolution.gif"), fail_line=fail_line)

    print(f"\nAll visuals written to {run_dir}")


if __name__ == "__main__":
    main()
