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
import json
import math
import pickle
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
from nff.stages.physics.displacement import parse_displacement_specs, DOF_NAMES
from nff.training.trainer import create_train_step, TrainState
from nff.utils.visualization import (
    write_deformed_into, plot_loading_diagram, plot_area_change, animate_closed_evolution,
)
from nff.utils.pipeline_viz import visualize_pipeline_results, plot_loss_history
from nff.topology.closed_builder_jax import solve_cut_vertices_jax, boundary_flat_from_logits
from nff.fabrication.cut_export import write_cut_patterns
from nff.closed.setup import build_closed_initial_state, init_closed_les_params
from nff.hinge import build_hinge_model
from nff.closed.deploy import _fit_circle, _boundary_cloud, _global_verts, _deployed_hinge_xy
from nff.hinge.path_ensemble import resolve_length_scale   # mm per pipeline unit, for reporting


def _overlap(tess):
    """(twisted_face_count, total_inter-face overlap area) via shapely."""
    polys = [Polygon(tess.vertices[f.vertex_indices]) for f in tess.faces]
    twisted = sum(0 if p.is_valid else 1 for p in polys)
    polys = [p if p.is_valid else p.buffer(0) for p in polys]
    return twisted, sum(p.area for p in polys) - unary_union(polys).area



def _cut_axis_or_none(config, state, static_features, map_params):
    """The per-hinge cut axis for this design, or None if it cannot be built."""
    try:
        from nff.hinge.frame import hinge_cut_axis
        return hinge_cut_axis(config, state, static_features, map_params)
    except Exception:
        return None


def _emit_load_path(result, config, deployed_tess, run_dir, bond_energy, hinge_geometry,
                    cut_axis=None):
    """Extract the load path of the deployed state and draw it.

    Best-effort: a load path is a diagnostic, and a run that has already produced its shape outputs
    should not fail because this could not be extracted. `updated_lagrangian` runs are skipped by
    `extract_force_field` itself, which raises rather than returning a wrong answer.
    """
    import traceback

    from nff.analysis.load_path_viz import plot_load_path
    from nff.analysis.load_paths import extract_force_field, save_field

    try:
        field = extract_force_field(result, config.physics, bond_energy_fn=bond_energy,
                                    hinge_geometry=hinge_geometry, cut_axis=cut_axis)
        if not field.checks.get('converged', False):
            print("  load path SKIPPED: Stage 2 did not converge to tolerance, so the per-hinge "
                  "forces are not trustworthy. Raise physics.solver_maxiter (use 20000).")
            return
        # Point the sidecar at the run's OWN config copy, so an offline replot can
        # recover the boundary conditions and still draw how the sheet was held.
        save_field(field, os.path.join(run_dir, "load_path"),
                   meta={"config": os.path.join(run_dir, "config.yaml")})
        _, stats = plot_load_path(field, deployed_tess,
                                  os.path.join(run_dir, "load_path.png"), config=config)
        print(f"  Saved load path to {os.path.join(run_dir, 'load_path.png')}  "
              f"({stats['tension_routes']} tension routes, {stats['shaded_tiles']} shaded tiles)")
    except NotImplementedError as exc:
        print(f"  load path SKIPPED: {exc}")
    except Exception:                                    # diagnostics must not sink a finished run
        print("  load path FAILED (run outputs are unaffected):")
        traceback.print_exc(limit=3)


# Below this many hinges the endpoint view has too few points to fill a hex grid and degenerates
# into one hinge per cell; the whole-path view still reads. Measured: 275 hinges fill it, 12 do not.
_ENDPOINT_MIN_HINGES = 60
_SMALL_SHEET_GRIDSIZE = 40   # path view on a small sheet; the 84 default is for ~275 hinges


def _emit_compression_shift(config, static_features, initial_state, params_untrained,
                            params_trained, res_untrained, res_trained, run_dir):
    """Where training moved the hinge paths, in the ligament-strain plane.

    Compares this run's OWN untrained design against its trained one, so every run produces the
    figure with no second run required. Each hexagon holds the signed change in how many hinges
    reach that ``(eta_a, eta_s)`` state; orange lost them, green gained them.

    The VIEW is chosen from the hinge count, because the two have opposite failure modes.

    ENDPOINT mode puts one point per hinge on the plane. Path count is conserved but cells-crossed
    is not, so under the path view a design that deploys less draws shorter paths, touches fewer
    cells and tilts the field orange without rerouting anything; endpoints are conserved exactly and
    the panel becomes a redistribution map. That is the better view -- but only when there are
    enough hinges to fill cells. On a 12-hinge sheet it degenerates into one hinge per cell and
    reads as scattered dots, while the 275-hinge sheet it was designed on fills the grid.

    So: endpoints above ``_ENDPOINT_MIN_HINGES``, the whole-path view below it, where 40-odd load
    steps per hinge still make a legible field. ⚠ The path view cannot claim exact conservation, and
    the endpoint view cannot show TRANSIENT compression -- which is what the path-min penalty
    actually acts on. Neither is strictly better; ``plot_compression_shift.py`` renders either.

    Costs no extra solve: both deployments are already in hand, and ``extract_hinge_paths`` replays a
    solved ``SolutionData`` rather than re-running the pipeline.

    Diagnostics must never sink a finished run, so every failure here is reported and swallowed.
    """
    import traceback

    from nff.analysis.compression_shift_viz import build_figure
    from nff.hinge.paths import build_hinge_geometry, extract_hinge_paths

    if res_untrained is None:
        print("  compression shift SKIPPED: no untrained deploy was computed (circle_fit target).")
        return
    try:
        ls = resolve_length_scale(config)
        pair = []
        for design, res in ((params_untrained, res_untrained), (params_trained, res_trained)):
            geom = build_hinge_geometry(config, static_features, initial_state, design)
            pair.append(extract_hinge_paths(res['solution'], res['valid_state'], geom, ls,
                                            reference_bond_vectors=res.get('reference_bond_vectors')))
        from nff.analysis.compression_shift_viz import GRIDSIZE, GRIDSIZE_ENDPOINTS
        use_ep = int(pair[0].n_hinges) >= _ENDPOINT_MIN_HINGES
        out, stats = build_figure([tuple(pair)], os.path.join(run_dir, "compression_shift.png"),
                                  use_endpoints=use_ep,
                                  gridsize=GRIDSIZE_ENDPOINTS if use_ep else _SMALL_SHEET_GRIDSIZE)
        cb, ca = stats['compressive_pct_before'][0], stats['compressive_pct_after'][0]
        tb, ta = stats['max_theta_deg_before'][0], stats['max_theta_deg_after'][0]
        print(f"  Saved compression shift to {out}  [{'endpoint' if use_ep else 'path'} view]  "
              f"(compressive path points {cb:.1f}% -> {ca:.1f}%, theta max {tb:.1f} -> {ta:.1f} deg)")
    except Exception:                                    # diagnostics must not sink a finished run
        print("  compression shift FAILED (run outputs are unaffected):")
        traceback.print_exc(limit=3)


def _write_loss_history(run_dir, history):
    """Dump the per-epoch metrics to JSON alongside the plot.

    The history was previously plotted and discarded, which made weight-tuning runs impossible to
    compare after the fact. Only scalar metrics are kept -- ``grad_norms`` is a whole PyTree.
    """
    rows = []
    for aux in history:
        row = {}
        for key, value in aux.items():
            if key == 'grad_norms':
                continue
            try:
                row[key] = float(value)
            except (TypeError, ValueError):
                continue          # non-scalar diagnostic; the plot does not read it either
        rows.append(row)
    tmp = os.path.join(run_dir, "loss_history.json.tmp")
    with open(tmp, "w") as fh:
        json.dump(rows, fh, indent=1)
    os.replace(tmp, os.path.join(run_dir, "loss_history.json"))


def _write_design_traj(run_dir, traj):
    """Persist the per-epoch design trajectory so a film can be rendered without re-training.

    The in-memory ``snaps`` list drives the legacy evolution GIF and is discarded with the process.
    A closed run is hours long, so the trajectory that describes what the optimizer actually did is
    written here instead, atomically, on the same cadence as the best-params checkpoint: a killed
    run then still yields every epoch it reached.

    Args:
        run_dir: the run folder.
        traj: list of ``(epoch, {leaf_name: np.ndarray})``. The design PyTree carries two or three
            leaves depending on ``hinge_model.learn_w_lig``, so the leaves are read off the dicts
            rather than assumed.
    """
    if not traj:
        return
    payload = {"epochs": np.asarray([ep for ep, _ in traj], dtype=np.int32)}
    for leaf in traj[0][1]:
        payload[leaf] = np.stack([np.asarray(p[leaf]) for _, p in traj])
    path = os.path.join(run_dir, "design_traj.npz")
    np.savez_compressed(path + ".tmp.npz", **payload)
    os.replace(path + ".tmp.npz", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-dir", default="closed")
    ap.add_argument("--config-name", default="circle_chamfer_10x10_best")
    ap.add_argument("--every", type=int, default=6)
    ap.add_argument("--traj-every", type=int, default=1,
                    help="cadence (epochs) of the DISK design trajectory written to "
                         "design_traj.npz, which the training film replays. Independent of "
                         "--every, which drives the in-memory evolution GIF: that one re-runs the "
                         "physics once per snapshot after training, so it cannot be set to 1.")
    ap.add_argument("--no-evolution-gif", action="store_true",
                    help="skip the post-training evolution GIF, whose cost is one forward_pipeline "
                         "per --every snapshot. The training film supersedes it.")
    ap.add_argument("--init-noise", type=float, default=None,
                    help="Gaussian sigma (LOGIT units) on the starting design; overrides the "
                         "config. Perturbs z and bnd_logits, i.e. the Stage-0 DESIGN -- the "
                         "topology, stiffnesses and bond permutation are always built at the "
                         "uniform r_init, so this is a random design on a FIXED topology. Valid by "
                         "construction (r = sigmoid(z) stays in (0,1)). ⚠ Above ~0.7 the cut "
                         "vertices approach their neighbours and the hinge frame degenerates to "
                         "arctan2(0,0) -- finite garbage, not an error.")
    ap.add_argument("--init-seed", type=int, default=None,
                    help="seed for --init-noise (NumPy default_rng)")
    args = ap.parse_args()

    cfg_path = f"data/configs/{args.config_dir}/{args.config_name}.yaml"
    config = load_and_parse_config(cfg_path)
    # Before any state is built: both keys are read inside init_closed_les_params.
    if args.init_noise is not None:
        config.topology['init_noise'] = float(args.init_noise)
    if args.init_seed is not None:
        config.topology['init_seed'] = int(args.init_seed)
    _noise = float(config.topology.get('init_noise', 0.0))
    print(f"  starting design: init_noise={_noise:g} init_seed={int(config.topology.get('init_seed', 0))}"
          + ("  (uniform r_init)" if _noise == 0.0 else "  (seeded random start)"))
    circle_fit = config.training.geometric_loss_type == "circle_fit"
    load_specs = config.topology.get('loads', [])

    # Displacement control: the actuation is an imposed motion, so the schematics draw the
    # prescribed DOFs where they would otherwise draw force arrows (same {face, dof, value} shape,
    # rotations already in radians). Empty under force control -> every diagram is unchanged.
    prescribed = parse_displacement_specs(config.physics.prescribed_displacements)
    disp_specs = [{'face': p.face, 'dof': p.dof, 'value': p.value} for p in prescribed]
    bc_specs = load_specs or disp_specs
    if prescribed:
        print(f"  displacement control: {len(prescribed)} prescribed DOFs — " +
              ", ".join(f"face {p.face} {DOF_NAMES[p.dof]}={p.value:+.4f}" for p in prescribed))

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("data", "outputs", "runs", f"run_{ts}_{args.config_name}")
    os.makedirs(run_dir, exist_ok=True)
    shutil.copy(cfg_path, os.path.join(run_dir, "config.yaml"))
    print(f"Run directory: {run_dir}")

    initial_state, tessellation = build_closed_initial_state(config)
    params, static_features = init_closed_les_params(config)
    bond_energy, hinge_probe_fn, geometry_fn, damage_fn, w_lig_logit0 = build_hinge_model(
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
    ume_mode = config.target.type == 'ume'
    target_cloud_override = None
    rect_geom = None
    ume_geom = None
    res0 = None                 # untrained deploy; stays None under circle_fit, which skips it
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
            # Optional orientation: a swinging deployment is a ROTATED shape, and an axis-aligned
            # box cannot hug one. 0.0 (the default) reproduces every pre-existing config exactly.
            rot = float(config.topology.get('target_rot_deg', 0.0) or 0.0)
            rect_geom = (cen0, hw, hh, rot)
            target_cloud_override = get_target_points(
                {'type': 'rectangle', 'center': cen0, 'radius': max(hw, hh),
                 'half_w': hw, 'half_h': hh, 'rot_deg': rot}, n_points=400)
            target_eff = TargetConfig(type='circle',
                                      center=(float(cen0[0]), float(cen0[1])), radius=float(max(hw, hh)))
            print(f"  fixed target RECTANGLE: center=({cen0[0]:.2f}, {cen0[1]:.2f})  "
                  f"half_w={hw:.2f} half_h={hh:.2f}"
                  + (f"  rot={rot:+.2f}deg" if rot else ""))
        elif ume_mode:
            # FIXED blossom straight from config, like the fixed rectangle above and unlike the
            # circle below: it does NOT adapt to the deploy. A target that resizes itself to each
            # design cannot be the one target two designs are compared on.
            cen0 = np.array([float(config.topology.get('target_cx', config.target.center[0])),
                             float(config.topology.get('target_cy', config.target.center[1]))])
            radius = float(config.topology.get('target_r', config.target.radius)) * sc
            petals = int(config.topology.get('target_petals', 5))
            eps = float(config.topology.get('target_eps', 0.0))
            ume_geom = (cen0, radius, petals, eps)
            target_cloud_override = get_target_points(
                {'type': 'ume', 'center': cen0, 'radius': radius, 'petals': petals, 'eps': eps},
                n_points=400)
            target_eff = TargetConfig(type='ume', center=(float(cen0[0]), float(cen0[1])),
                                      radius=float(radius))
            eps_star = 1.0 / (petals ** 2 + 1.0)
            print(f"  fixed target UME: center=({cen0[0]:.2f}, {cen0[1]:.2f})  mean radius={radius:.4f}"
                  f"  petals={petals}  eps={eps:.4f}"
                  f"  ({'convex' if eps < eps_star else 're-entrant'}, threshold {eps_star:.4f})")
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
        target_cloud=target_cloud_override, bond_energy_fn=bond_energy,
        hinge_probe_fn=hinge_probe_fn, hinge_geometry_fn=geometry_fn)
    state = TrainState(params=params, opt_state=optimizer.init(params), rng=jax.random.PRNGKey(0))

    _p0 = {k: np.asarray(v) for k, v in state.params.items()}
    history, snaps, traj = [], [(0, _p0)], [(0, _p0)]
    # Selection is on the TOTAL loss, not chamfer alone: with damage/compression switched on, a
    # chamfer-only rule would happily return the shape-best design that tore or buckled its hinges.
    # The consequence to keep in mind is that the ranking moves whenever a weight is retuned, so
    # best-loss values are comparable only within one weight setting (chamfer is printed alongside).
    best = (math.inf, state.params, math.inf)
    for epoch in range(config.training.num_epochs):
        state, loss, aux = step(state)
        # ONE device->host transfer per epoch, not one per metric read. `grad_norms` is dropped
        # here rather than at write time: it is a whole PyTree per epoch, and keeping it for the
        # length of the run was the run's largest retained allocation for a value nothing reads.
        tot, aux = jax.device_get((loss, {k: v for k, v in aux.items() if k != 'grad_norms'}))
        history.append(aux)
        ch = float(aux.get('chamfer_total', math.inf))
        tot = float(tot)
        if math.isfinite(tot) and tot < best[0]:
            best = (tot, state.params, ch)
        if (epoch + 1) % args.every == 0 or epoch == config.training.num_epochs - 1:
            # numpy, so the snapshot list does not pin one device copy of the design per snapshot
            snaps.append((epoch + 1, {k: np.asarray(v) for k, v in state.params.items()}))
        if args.traj_every > 0 and ((epoch + 1) % args.traj_every == 0
                                    or epoch == config.training.num_epochs - 1):
            traj.append((epoch + 1, {k: np.asarray(v) for k, v in state.params.items()}))
        if epoch % 25 == 0 or epoch == config.training.num_epochs - 1:
            # Checkpoint the running best. A closed run is hours long, and until this existed a kill
            # at any point -- an OS memory kill, a sleep, a stray Ctrl-C -- discarded the whole run,
            # because the only write was after the loop. Written atomically so a kill DURING the
            # dump cannot leave a truncated pickle where a valid one used to be.
            _ck = os.path.join(run_dir, "best_params.pkl")
            with open(_ck + ".tmp", "wb") as _pf:
                pickle.dump({k: np.asarray(v) for k, v in best[1].items()}, _pf)
            os.replace(_ck + ".tmp", _ck)
            # Same reasoning as the checkpoint above, for the two artefacts the film reads: both
            # were written only after the loop, so a kill discarded every metric and every design.
            _write_design_traj(run_dir, traj)
            _write_loss_history(run_dir, history)
            msg = f"  epoch {epoch:3d}  loss={tot:.4e}  chamfer={ch:.4e}"
            if aux.get('hinge_max_D') is not None:    # surrogate damage metrics, when tracked
                msg += (f"  maxD={float(aux['hinge_max_D']):.3g} meanD={float(aux['hinge_mean_D']):.3g}"
                        f" gap={float(aux.get('damage_path_gap', 0.0)):.2g}")
            if aux.get('min_eta_a') is not None:      # compression metrics, when tracked
                msg += (f"  minEtaA={float(aux['min_eta_a']):.3f}"
                        f" nComp={int(aux['n_hinges_compressive'])}")
            if aux.get('mean_abs_eta_s') is not None:  # shear-path metrics, when tracked
                msg += (f"  meanEtaS={float(aux['mean_abs_eta_s']):.4f}"
                        f" maxEtaS={float(aux['max_abs_eta_s']):.3f}")
            print(msg)
    best_params = best[1]
    print(f"  best loss={best[0]:.4e}  (chamfer there={best[2]:.4e})")
    _write_loss_history(run_dir, history)
    _write_design_traj(run_dir, traj)
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

    # ── Reaction readout: under displacement control the force is an OUTPUT. ──
    if result['reactions'] is not None:
        rr = result['reactions']
        R = np.asarray(rr.values)                       # (n_steps, n_prescribed)
        for (f, d), u, r in zip(np.asarray(rr.face_DOF_pairs), np.asarray(rr.prescribed_values), R[-1]):
            print(f"  reaction: face {int(f)} {DOF_NAMES[int(d)]}={float(u):+.4f} -> {float(r):+.2f}")
        print(f"  reaction total = {float(R[-1].sum()):+.2f}   (per step: "
              + ", ".join(f"{float(s):.1f}" for s in R.sum(axis=1)) + ")")

    # ── Target params for the pretty viz (fitted circle is size-adaptive). ──
    vs, disp = result['valid_state'], result['solution'].fields[-1]
    if circle_fit:
        cc, rr = _fit_circle(_boundary_cloud(vs, disp))
        target_params = {'type': 'circle', 'center': [float(cc[0]), float(cc[1])], 'radius': rr}
    elif rect_mode:
        cen0, hw, hh, rot = rect_geom
        target_params = {'type': 'rectangle', 'center': [float(cen0[0]), float(cen0[1])],
                         'radius': float(max(hw, hh)), 'half_w': float(hw), 'half_h': float(hh),
                         'rot_deg': float(rot)}
    elif ume_mode:
        cen0, radius, petals, eps = ume_geom
        target_params = {'type': 'ume', 'center': [float(cen0[0]), float(cen0[1])],
                         'radius': float(radius), 'petals': int(petals), 'eps': float(eps)}
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
                       config=config, hinge_model=config.hinge_model, load_specs=bc_specs,
                       config_name=args.config_name, hinge_w_lig=hinge_w_lig_best)

    # ── How precisely did the deployed boundary land on the target? ──
    # `chamfer_total` cannot answer this. It is `precision + coverage * coverage_loss`, and the
    # coverage half asks every one of the 400 sampled target points to have a boundary vertex near
    # it -- which a handful of boundary vertices can never do. Its floor is ~(perimeter/n_verts)^2/12
    # regardless of the design, so chamfer_total is mostly a constant and a design can lower it
    # while moving the boundary AWAY from the target. Report the honest thing instead: the exact
    # distance from each boundary vertex to the target perimeter, in millimetres.
    _cloud = _boundary_cloud(vs, disp)
    if rect_mode:
        _cen, _hw, _hh, _rot = rect_geom
        # ⚠ This SDF is analytic and axis-aligned, so it must be evaluated in the RECTANGLE's own
        # frame. Skipping the rotation here would report the distance to a box the run never
        # trained against -- silently, since the number looks perfectly reasonable either way.
        _rel = _cloud - np.asarray(_cen)
        if _rot:
            _a = np.radians(-_rot)
            _rel = _rel @ np.array([[np.cos(_a), -np.sin(_a)], [np.sin(_a), np.cos(_a)]]).T
        _d = np.abs(_rel) - np.array([_hw, _hh])
        _err = np.abs(np.linalg.norm(np.maximum(_d, 0.0), axis=-1)
                      + np.minimum(np.max(_d, axis=-1), 0.0))     # outside + inside distance
        _ls = resolve_length_scale(config)
        print(f"  boundary vs target ({len(_err)} vertices): mean={_err.mean()*_ls:.2f}mm  "
              f"max={_err.max()*_ls:.2f}mm  rms={np.sqrt((_err**2).mean())*_ls:.2f}mm"
              f"   [{_err.mean()/(2*_hh):.3%} of target height]")
    # Deployment angle = relative rotation across each hinge; the sheet's single kinematic coordinate.
    _bc = np.asarray(vs.bond_connectivity)
    _nn2 = np.asarray(vs.centroid_node_vectors).shape[1]
    _th = np.asarray(disp[:, 2])
    _dth = np.degrees(np.abs(_th[_bc[:, 1] // _nn2] - _th[_bc[:, 0] // _nn2]))
    print(f"  deployment angle: mean={_dth.mean():.2f}deg  [{_dth.min():.2f}, {_dth.max():.2f}]")

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
    plot_loading_diagram(s2, config.topology.get('bc_clamped', []), bc_specs,
                         os.path.join(run_dir, "loading_diagram.png"),
                         title="Prescribed displacement" if disp_specs and not load_specs else "Loading",
                         clamped_dofs=config.topology.get('clamped_dofs'),
                         y_roller_faces=config.topology.get('y_roller_faces'),
                         x_roller_faces=config.topology.get('x_roller_faces'))

    # One clean load-path figure, from the SAME deployed state and the same BC drawing as the
    # loading diagram above, so the two cannot disagree about how the sheet is held.
    _emit_load_path(result, config, s2, run_dir, bond_energy, geom_best,
                    cut_axis=_cut_axis_or_none(config, initial_state, static_features, best_params))

    # Where training moved the hinge paths, in the (eta_a, eta_s) ligament-strain plane.
    _emit_compression_shift(config, static_features, initial_state, params, best_params,
                            res0, result, run_dir)

    # ── Training-evolution animation. ──
    if args.no_evolution_gif:
        print(f"\nAll visuals written to {run_dir}")
        return
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
            cen0, hw, hh, rot = rect_geom
            ax.add_patch(Rectangle((cen0[0] - hw, cen0[1] - hh), 2 * hw, 2 * hh,
                                   angle=rot, rotation_point='center',
                                   fill=False, edgecolor="#D62828", lw=2.0, ls="--"))
        elif ume_mode:
            cen0, radius, petals, eps = ume_geom
            pts = get_target_points({'type': 'ume', 'center': cen0, 'radius': radius,
                                     'petals': petals, 'eps': eps}, n_points=400)
            ax.plot(pts[:, 0], pts[:, 1], color="#D62828", lw=2.0, ls="--")
        else:
            ax.add_patch(Circle(target_eff.center, target_eff.radius,
                                fill=False, edgecolor="#D62828", lw=2.0, ls="--"))

    fail_line = float(getattr(config.hinge_model, 'fail_line', 1.0))
    animate_closed_evolution(tessellation, frames, add_target,
                             os.path.join(run_dir, "training_evolution.gif"), fail_line=fail_line)

    print(f"\nAll visuals written to {run_dir}")


if __name__ == "__main__":
    main()
