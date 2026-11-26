"""Sweep a load and watch the 1-DOF assumption, the load path, and the voids together.

Two experiments share this driver, because the three measurements are the same in both:

    --axis open        one +x pull on a right-edge tile, swept from nothing to the end of
                       deployment. Asks how the sheet gets to its deployed state.
    --axis cantilever  the +x pull HELD at its calibrated value, plus a downward load on the
                       middle of the top border, and that swept. Asks what destroys the state.

The sweep is one independent incremental solve per load level, not the intermediate steps of a
single ramp. Two reasons: the ramp scales every load together, so it cannot hold one load fixed
while raising another, and log spacing is needed because the response is strongly nonlinear (a
15x10 square-tile sheet reaches 34 of its 90 degrees under the first 8% of the load). Each solve
keeps its whole history anyway, so the per-level endpoints can be cross-checked against a ramp.

Run:
    JAX_PLATFORMS=cpu PYTHONPATH=$(pwd) python legacy/experiments/diagnostics/sweep_localisation.py \
        --arm squares --axis open --n-loads 24
"""

import argparse
import copy
import json
import os
import pickle
import time

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np

from nff.config.experiment import load_and_parse_config
from nff.closed.setup import (build_closed_initial_state, build_surrogate_energy, hinge_cut_axis,
                              init_closed_les_params)
from nff.closed import force_stats as FS
from nff.closed import one_dof as OD
from nff.closed import void_geometry as VG
from nff.analysis.load_paths import (deployed_nodes, extract_force_field, force_budget,
                                   interface_decomposition)
from nff_paper.closed_support.shield_kinematics import _checkerboard_signs, build_deploy_plan
from nff.stages.pipeline import forward_pipeline
from nff.topology.closed_builder_jax import (boundary_flat_from_logits, build_boundary_edges,
                                             build_deploy_structure, solve_cut_vertices_jax)

BASE_CONFIG = "data/configs/closed/figmat_1_deploy.yaml"
TRAINED_RUN = "data/outputs/runs/run_20260806_161121_figmat_1_deploy"

# Figure 6 Problem I moved to the right edge: col 14, row 5 (face = col * N + row).
LOAD_FACE = 145
# Middle of the top border: col 7, row 9. Distinct from the pin at face 9.
CANT_FACE = 79
# Measured: 247.74 N on face 145 reproduces Problem I's deployment (theta_rel max 71.31 deg).
F_OPEN_CAL = 247.74

# arm -> (r_init or None to keep, k_stretch factor, k_shear factor, isotropic, params_path)
ARMS = {
    "squares":  dict(r_init=0.5,  ks=1.0,   kh=1.0,   iso=False, params=None),
    "rdpqk":    dict(r_init=0.45, ks=1.0,   kh=1.0,   iso=False, params=None),
    "trained":  dict(r_init=0.45, ks=1.0,   kh=1.0,   iso=False,
                     params=os.path.join(TRAINED_RUN, "best_params.pkl")),
    # Designed against the same target through the closed-form 1-DOF deployment alone: no
    # equilibrium, no load, no solver. The literature's geometric inverse design, so this arm asks
    # how much non-opening action a pattern that never saw a force can absorb. Same r_init as
    # `trained`, because the topology is always built at the uniform value and only the design
    # differs.
    "kinematic": dict(r_init=0.45, ks=1.0,  kh=1.0,   iso=False,
                      params="data/outputs/kinematic_designs/figmat_geom/best_params.pkl"),
    # Trained ON this sweep's own load pair, at ratio 1.0, so the comparison is on-design for once.
    # `cant` carries compression + shear_path at parity re-measured for this load; `cant_shape`
    # drops both, which separates designing against the actuation from penalising hinge states.
    "cant":      dict(r_init=0.45, ks=1.0,  kh=1.0,   iso=False,
                      params="data/outputs/runs/run_20260812_151537_cant_r1p0/best_params.pkl"),
    "cant_shape": dict(r_init=0.45, ks=1.0, kh=1.0,   iso=False,
                       params="data/outputs/runs/run_20260812_151538_cant_r1p0_shape/best_params.pkl"),
    "stiff10":  dict(r_init=0.5,  ks=10.0,  kh=10.0,  iso=False, params=None),
    "stiff100": dict(r_init=0.5,  ks=100.0, kh=100.0, iso=False, params=None),
    "iso":      dict(r_init=0.5,  ks=1.0,   kh=1.0,   iso=True,  params=None),
}


def build_arm_config(arm: str, base: str):
    """Apply one arm's overrides to the Figure 6 Problem I config, in memory only."""
    spec = ARMS[arm]
    c = copy.deepcopy(load_and_parse_config(base))
    c.topology['r_init'] = float(spec['r_init'])
    c.topology['init_noise'] = 0.0
    c.topology['init_seed'] = 0
    # The spring constants live in the topology dict -- `configure_tessellation` reads them from
    # there -- not in a `material` field on the parsed config.
    c.topology['k_stretch'] = float(c.topology['k_stretch']) * spec['ks']
    c.topology['k_shear'] = float(c.topology['k_shear']) * spec['kh']
    if spec['iso']:
        # Every closed-sheet hinge falls back to a degenerate reference bond vector, which swaps
        # k_stretch and k_shear on roughly half of them. An isotropic point spring makes the frame
        # unable to matter, so this arm is the control for that wart.
        c.topology['k_shear'] = float(c.topology['k_stretch'])
    c.physics.solver_maxiter = 20000
    return c, spec['params']


def loads_for(axis: str, level: float):
    """Load list at one sweep level. ``level`` is newtons (open) or a ratio (cantilever)."""
    if axis == "open":
        return [{'face': LOAD_FACE, 'dof': 0, 'value': float(level)}]
    return [{'face': LOAD_FACE, 'dof': 0, 'value': F_OPEN_CAL},
            {'face': CANT_FACE, 'dof': 1, 'value': -float(level) * F_OPEN_CAL}]


def solve_at(config, params_path, loads, n_steps: int):
    """One incremental Stage-2 solve. Returns everything the measurements need."""
    c = copy.deepcopy(config)
    c.topology['loads'] = loads
    c.physics.num_load_steps = int(n_steps)
    state, _ = build_closed_initial_state(c)
    params, sf = init_closed_les_params(c)
    bond_energy, _, geometry_fn, _, w_lig_logit0 = build_surrogate_energy(c, sf, state, params)
    if w_lig_logit0 is not None:
        params = {**params, 'w_lig_logit': w_lig_logit0}
    if params_path:
        with open(params_path, 'rb') as fh:
            params = {k: np.asarray(v) for k, v in pickle.load(fh).items()}
    hinge_geometry = (geometry_fn(params)
                      if (geometry_fn is not None and bond_energy is not None) else None)
    results = forward_pipeline(state, c.target, c.validity, c.physics, map_type=c.mapping.type,
                               map_params=params, static_features=sf, load_specs=loads,
                               bond_energy_fn=bond_energy, hinge_geometry=hinge_geometry)
    try:
        cut_axis = hinge_cut_axis(c, state, sf, params)
    except Exception:
        cut_axis = None
    return results, c, params, sf, bond_energy, hinge_geometry, cut_axis


def measure(results, config, cut_axis, bond_energy, hinge_geometry, void_index, plan, flat_tiles,
            sigma, length_scale: float):
    """Q1 + Q2 + Q3 at the endpoint of one solve, plus the Q1 history."""
    vs = results['valid_state']
    fields = np.asarray(results['solution'].fields)                 # (n_steps, n_faces, 3)
    disp = fields[-1]
    theta = disp[:, 2]

    field = extract_force_field(results, config.physics, bond_energy_fn=bond_energy,
                                hinge_geometry=hinge_geometry, cut_axis=cut_axis, step=-1)
    face_pairs = np.asarray(field.face_pairs)

    # ── Q1: 1-DOF kinematics ────────────────────────────────────────────────────────────────
    fit = OD.fit_one_dof(theta, sigma)
    loc = OD.localisation_ratio(theta, face_pairs, fit.psi)
    dist = OD.graph_distance(face_pairs, theta.size, [LOAD_FACE])
    dec_theta = OD.decay_length(sigma * theta - fit.psi, dist)
    dec_resid = OD.decay_length(fit.residual, dist)
    rel = np.abs(OD.relative_rotations(theta, face_pairs))
    dstats = OD.distribution_stats(rel)
    cfg_fit = OD.fit_configuration(plan, flat_tiles, np.asarray(
        deployed_nodes(vs.face_centroids, vs.centroid_node_vectors, disp)))

    # ── Q2: force network ───────────────────────────────────────────────────────────────────
    dec = interface_decomposition(field)
    budget = force_budget(field)
    conc = FS.channel_concentration(dec)
    press = FS.tile_pressure(field)
    cb = FS.checkerboard_amplitude(field, values=press)
    clust = FS.force_clustering(field, dec)
    f_mag = np.asarray(dec['f_mag'])
    hinge_dist = np.minimum(dist[face_pairs[:, 0]], dist[face_pairs[:, 1]])
    dec_force = OD.decay_length(f_mag, hinge_dist)

    # ── Q3: voids ───────────────────────────────────────────────────────────────────────────
    nodes = np.asarray(deployed_nodes(vs.face_centroids, vs.centroid_node_vectors, disp))
    vr = VG.void_report(void_index, nodes, length_scale=length_scale)

    return dict(
        theta=theta, theta_history=fields[:, :, 2], residual=fit.residual, distance=dist,
        rel_rotation=rel, face_pairs=face_pairs, pressure=press, f_mag=f_mag,
        para_defect=vr['para_defect'], rhombus_defect=vr['rhombus_defect'], pin_gap=vr['pin_gap'],
        void_area=vr['area_octagon'],
        summary=dict(
            psi=fit.psi, swing=fit.swing, rms=fit.rms, rms_over_psi=fit.rms_over_psi, r2=fit.r2,
            localisation=loc, theta_rel_max=float(rel.max()), theta_rel_min=float(rel.min()),
            decay_theta=dec_theta['lam'], decay_resid=dec_resid['lam'], decay_force=dec_force['lam'],
            inv_decay_theta=dec_theta['inv_lam'], inv_decay_resid=dec_resid['inv_lam'],
            inv_decay_force=dec_force['inv_lam'], decay_theta_r2=dec_theta['r2'],
            cfg_psi=cfg_fit['psi'], cfg_rmse_mm=cfg_fit['rmse'] * length_scale,
            cfg_max_mm=cfg_fit['max_err'] * length_scale,
            theta_n_eff=dstats['n_eff'], theta_gini=dstats['gini'],
            theta_top_decile=dstats['top_decile_share'],
            budget={k: v for k, v in budget.items() if isinstance(v, (int, float))},
            conc=conc, checkerboard=cb, clustering=clust,
            para_mean=vr['para_mean'], para_max=vr['para_max'],
            rhombus_mean=vr['rhombus_mean'], gap_mean=vr['gap_mean'], gap_max=vr['gap_max'],
            void_area_total=vr['area_total'], n_negative_void=vr['n_negative_area'],
            checks=field.checks,
        ))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", default="squares", choices=sorted(ARMS))
    ap.add_argument("--axis", default="open", choices=("open", "cantilever"))
    ap.add_argument("--config", default=BASE_CONFIG)
    ap.add_argument("--n-loads", type=int, default=24)
    ap.add_argument("--f-min", type=float, default=None,
                    help="open: newtons (default 0.5). cantilever: ratio (default 0.05)")
    ap.add_argument("--f-max", type=float, default=None,
                    help="open: newtons (default 2000). cantilever: ratio (default 3.0)")
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--linear", action="store_true",
                    help="Sample the window uniformly instead of geometrically. Use it for a narrow "
                         "range around a feature; geometric spacing is for sweeps spanning decades.")
    ap.add_argument("--solver-tol", type=float, default=1e-8,
                    help="Stage-2 gradient tolerance. The shipped 1e-5 is ABSOLUTE, so it fails "
                         "the 1e-6 equilibrium check at low load even though the state is already "
                         "right to 6 figures; 1e-8 converges over the whole sweep for ~1.8x cost.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-contact", action="store_true",
                    help="Disable the void-angle contact barrier. Near the COMPACT state the voids "
                         "are still nearly shut, so the barrier is a live stiff constraint there "
                         "even though it is inactive at working deployment -- this isolates it.")
    args = ap.parse_args()

    f_min = args.f_min if args.f_min is not None else (0.5 if args.axis == "open" else 0.05)
    f_max = args.f_max if args.f_max is not None else (2000.0 if args.axis == "open" else 3.0)
    # Geometric spacing suits a sweep that spans decades; a narrow window around a feature wants
    # UNIFORM spacing, or the plotted curve is dense on one side and carries a lone far point joined
    # by a long straight segment, which reads as a sampling artifact rather than as the physics.
    space = np.linspace if args.linear else np.geomspace
    levels = space(f_min, f_max, int(args.n_loads))

    out_dir = args.out or f"data/outputs/localisation/{args.axis}_{args.arm}"
    os.makedirs(out_dir, exist_ok=True)

    config, params_path = build_arm_config(args.arm, args.config)
    config.physics.solver_tol = float(args.solver_tol)
    config.physics.solver_maxiter = 60000
    if args.no_contact:
        config.physics.use_contact = False
    M, N = int(config.topology['M']), int(config.topology['N'])
    spacing = float(config.topology.get('spacing', 1.0))
    spacing_y = float(config.topology.get('spacing_y', spacing))
    length_scale = float(getattr(config.hinge_model, 'length_scale', 1.0))

    # Static structures: built once, reused at every level.
    struct = build_deploy_structure(M, N)
    bnd = build_boundary_edges(struct, spacing, spacing_y)
    void_index = VG.build_void_index(struct)
    plan = build_deploy_plan(struct, M, N, spacing=spacing, spacing_y=spacing_y)
    sigma = _checkerboard_signs(M, N).astype(float)

    # The 1-DOF reference sheet must be the arm's OWN flat design, otherwise the configuration fit
    # compares against a different tessellation.
    import jax.numpy as jnp
    bflat = boundary_flat_from_logits(bnd, jnp.asarray(bnd['init_logits']))
    if params_path:
        with open(params_path, 'rb') as fh:
            trained = {k: np.asarray(v) for k, v in pickle.load(fh).items()}
        r_field = jax.nn.sigmoid(jnp.asarray(trained['z']))
        if 'bnd_logits' in trained:
            bflat = boundary_flat_from_logits(bnd, jnp.asarray(trained['bnd_logits']))
    else:
        r_field = jnp.full((struct['rows'], struct['cols']), float(config.topology['r_init']))
    flat_tiles = np.asarray(solve_cut_vertices_jax(struct, bflat, r_field)[
        np.asarray(struct['corner_pid'])])

    print(f"[sweep] arm={args.arm} axis={args.axis} n_voids={void_index.n_voids} "
          f"levels {levels[0]:.4g}..{levels[-1]:.4g} steps={args.steps}", flush=True)

    rows, per_level = [], {}
    for i, lv in enumerate(levels):
        t0 = time.time()
        loads = loads_for(args.axis, float(lv))
        try:
            res, c, params, sf, be, hg, ca = solve_at(config, params_path, loads, args.steps)
            m = measure(res, c, ca, be, hg, void_index, plan, flat_tiles, sigma, length_scale)
        except Exception as exc:
            print(f"  [{i:2d}] level={lv:9.4g}  FAILED {type(exc).__name__}: {exc}", flush=True)
            jax.clear_caches()
            continue
        s = m['summary']
        s['level'] = float(lv)
        s['wall_s'] = time.time() - t0
        rows.append(s)
        for k, v in m.items():
            if k != 'summary':
                per_level[f"{k}__{i}"] = np.asarray(v)
        print(f"  [{i:2d}] level={lv:9.4g}  psi={np.degrees(s['psi']):7.3f}d "
              f"swing={np.degrees(s['swing']):6.3f}d R2={s['r2']:.6f} loc={s['localisation']:6.3f} "
              f"1/lam={s['inv_decay_theta']:+7.4f} para={s['para_mean']:.3e} "
              f"cb={s['checkerboard'].get('corr_detrended', float('nan')):+.3f} "
              f"gini={s['budget'].get('gini', float('nan')):.3f} "
              f"conv={s['checks'].get('converged')} ({s['wall_s']:.1f}s)", flush=True)
        jax.clear_caches()

    np.savez_compressed(os.path.join(out_dir, "sweep.npz"),
                        levels=np.asarray([r['level'] for r in rows]), sigma=sigma, **per_level)
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump({"arm": args.arm, "axis": args.axis, "config": args.config,
                   "load_face": LOAD_FACE, "cant_face": CANT_FACE, "f_open_cal": F_OPEN_CAL,
                   "n_steps": args.steps, "length_scale": length_scale, "M": M, "N": N,
                   "solver_tol": float(args.solver_tol),
                   "k_stretch": float(config.topology['k_stretch']),
                   "k_shear": float(config.topology['k_shear']),
                   "k_rot": float(config.topology['k_rot']),
                   "n_voids": int(void_index.n_voids), "rows": rows},
                  fh, indent=2, default=float)
    print(f"[sweep] wrote {out_dir}/sweep.npz + summary.json  ({len(rows)}/{len(levels)} levels)")


if __name__ == "__main__":
    main()
