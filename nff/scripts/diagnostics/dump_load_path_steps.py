"""Deployed tile geometry at EVERY load step, for comparison against a photographed deployment.

``run_closed`` keeps only the final state, but a photograph of a partly deployed sheet has to be
matched against the step that looks like it. The whole path already exists inside
``SolutionData.fields`` -- this dumps its geometry so a photo can be placed on it.

⚠ GEOMETRY ONLY, and deliberately so. ``extract_force_field`` rebuilds the FULL external load at
every step, but Stage 2 applies only ``(k+1)/num_steps`` of it, so every force, energy and residual
it reports at an intermediate step is wrong by that factor. Tile positions are unaffected because
they come from the solved displacement field. ``--with-forces`` opts into the final step only.

    JAX_PLATFORMS=cpu conda run -n kgnn_mac python \
        nff/scripts/diagnostics/dump_load_path_steps.py \
            --run-dir data/outputs/runs/run_20260813_151917_pet45_3x3_mid_phys
"""

import argparse
import copy
import json
import os

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np                                                   # noqa: E402

from nff.config.experiment import load_and_parse_config              # noqa: E402
from nff.closed.load_paths import deployed_nodes, solve_deployed_state  # noqa: E402

# Past this multiple of the design load the sheet stops opening its hinges and starts shearing the
# ligaments, several times outside the surrogate's trained shear range. Measured, not assumed.
OVERLOAD_CEILING = 1.25


def tile_rotations(flat, deployed):
    """Best-fit rotation carrying each flat tile onto its deployed tile, in degrees.

    Args:
        flat: (n_tiles, n_corners, 2) undeformed tile corners.
        deployed: (..., n_tiles, n_corners, 2) deployed tile corners.

    Returns:
        (..., n_tiles) signed rotation in degrees.
    """
    P = flat - flat.mean(-2, keepdims=True)
    Q = deployed - deployed.mean(-2, keepdims=True)
    H = np.einsum("...ki,...kj->...ij", P, Q)
    return np.degrees(np.arctan2(H[..., 0, 1] - H[..., 1, 0], H[..., 0, 0] + H[..., 1, 1]))


def hinge_opening(rot_deg, face_pairs):
    """Relative rotation across every hinge, wrapped into [0, 180)."""
    d = np.abs(rot_deg[..., face_pairs[:, 0]] - rot_deg[..., face_pairs[:, 1]]) % 360.0
    return np.minimum(d, 360.0 - d)


def min_area_rect(points):
    """(width, height, angle_deg) of the minimum-area rectangle enclosing a point cloud.

    Rotating calipers over the convex hull edges. Width is the shorter side.
    """
    from scipy.spatial import ConvexHull
    hull = points[ConvexHull(points).vertices]
    edges = np.roll(hull, -1, axis=0) - hull
    angles = np.arctan2(edges[:, 1], edges[:, 0])
    best = None
    for a in angles:
        c, s = np.cos(-a), np.sin(-a)
        rot = hull @ np.array([[c, -s], [s, c]]).T
        w = np.ptp(rot[:, 0])
        h = np.ptp(rot[:, 1])
        if best is None or w * h < best[0] * best[1]:
            best = (w, h, np.degrees(a))
    w, h, a = best
    return (min(w, h), max(w, h), a)


def hull_area(points):
    from scipy.spatial import ConvexHull
    return float(ConvexHull(points).volume)


def target_rect_mm(config, length_scale):
    """(4, 2) corners of the config's target rectangle, in pipeline mm, CCW from lower-left."""
    t = config.topology
    keys = ('target_cx', 'target_cy', 'target_half_w', 'target_half_h')
    if not all(k in t for k in keys):
        return None
    cx, cy, hw, hh = (float(t[k]) * length_scale for k in keys)
    rot = np.radians(float(t.get('target_rot_deg', 0.0)))
    corners = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]])
    c, s = np.cos(rot), np.sin(rot)
    return corners @ np.array([[c, -s], [s, c]]).T + np.array([cx, cy])


def domain_guard(results, state, hinge_geometry, config, length_scale):
    """Where the deployment sits relative to the surrogate's trust box.

    A deployed geometry is only a prediction where the surrogate was trained, and this sheet leaves
    its box in COMPRESSION before it reaches the design load -- so the geometry alone is not
    interpretable and the box has to be reported beside it. The box comes from the checkpoint, not
    from the module default, because a wide checkpoint carries its own.

    Returns:
        dict of the diagnostics, or ``{'available': False, ...}`` on the ROM branch.
    """
    if hinge_geometry is None:
        return {"available": False, "reason": "no surrogate hinge geometry (ROM springs)"}
    try:
        from nff.closed.hinge_paths import extract_hinge_paths, path_diagnostics
        from nff.models.hinge_surrogate import DOMAIN, load_hinge_surrogate
        hm = config.hinge_model
        _, stats, _ = load_hinge_surrogate(hm.checkpoint)
        dom = stats.get("domain", DOMAIN)
        paths = extract_hinge_paths(results['solution'], state, hinge_geometry, length_scale)
        d = path_diagnostics(paths, dom)
        return {"available": True,
                "endpoint_in_domain_frac": d["endpoint_in_domain_frac"],
                "in_domain_frac": d["in_domain_frac"],
                "min_eta_a": d["compression"]["min_eta_a"],
                "eta_a_range": d["eta_a_range"], "eta_s_range": d["eta_s_range"],
                "theta_range_deg": d["theta_range_deg"],
                "box_occupancy": d["box_occupancy"], "domain": d["domain"]}
    except Exception as exc:                          # a diagnostic must never kill the dump
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}


def format_guard(g):
    if not g.get("available"):
        return f"  [domain] unavailable — {g.get('reason', '')}"
    dom = g["domain"]
    ea, es = g["eta_a_range"], g["eta_s_range"]
    out = [f"  [domain] endpoint in box {g['endpoint_in_domain_frac']:.3f}   "
           f"whole path {g['in_domain_frac']:.3f}",
           f"           eta_a {ea[0]:+.4f}..{ea[1]:+.4f}  (box {dom['eta_a_min']:+.4f}"
           f"..{dom['eta_a_max']:+.4f})",
           f"           eta_s {es[0]:+.4f}..{es[1]:+.4f}  (box +-{dom['eta_s_max']:.4f})",
           f"           theta {g['theta_range_deg'][1]:.2f} deg  "
           f"(box {np.degrees(dom['theta_max']):.1f} deg)"]
    if ea[0] < dom["eta_a_min"]:
        out.append(f"           ⚠ COMPRESSION is {ea[0] / dom['eta_a_min']:.2f}x past the trained "
                   f"floor: the model is extrapolating where a real sheet would buckle")
    if abs(es[0]) > dom["eta_s_max"] or abs(es[1]) > dom["eta_s_max"]:
        out.append(f"           ⚠ SHEAR is outside the box; treat this solve as non-physical")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, help="run folder with config.yaml + best_params.pkl")
    p.add_argument("--out", default=None, help="output stem (default <run-dir>/load_path_steps)")
    p.add_argument("--maxiter", type=int, default=20000, help="override physics.solver_maxiter")
    p.add_argument("--load-scale", type=float, default=1.0,
                   help="multiply every load value; >1 deploys PAST the design load")
    p.add_argument("--steps", type=int, default=None, help="override physics.num_load_steps")
    p.add_argument("--with-forces", action="store_true",
                   help="also extract the FINAL-step force field (intermediate steps are unsound)")
    args = p.parse_args()

    cfg_path = os.path.join(args.run_dir, "config.yaml")
    params_path = os.path.join(args.run_dir, "best_params.pkl")
    config = copy.deepcopy(load_and_parse_config(cfg_path))
    if args.maxiter and args.maxiter != config.physics.solver_maxiter:
        print(f"  solver_maxiter {config.physics.solver_maxiter} -> {args.maxiter}")
        object.__setattr__(config.physics, 'solver_maxiter', int(args.maxiter))
    if args.steps:
        # PhysicsConfig is a frozen equinox Module, so plain assignment raises.
        object.__setattr__(config.physics, 'num_load_steps', int(args.steps))
    if args.load_scale != 1.0:
        # The load is baked into the tessellation when the state is built, so it has to be rewritten
        # HERE -- handing different load_specs to forward_pipeline afterwards silently does nothing.
        config.topology['loads'] = [{**dict(s), 'value': float(s['value']) * args.load_scale}
                                    for s in config.topology.get('loads', []) or []]
        newtons = ", ".join(f"{s['value']:.2f} N" for s in config.topology['loads'])
        print(f"  load x{args.load_scale:.3f} -> {newtons}")
        if args.load_scale > OVERLOAD_CEILING:
            print(f"  ⚠ {args.load_scale:.2f}x is past the {OVERLOAD_CEILING}x ceiling: between 1.25x "
                  f"and 1.5x this sheet stops opening and starts SHEARING its ligaments, several "
                  f"times outside the surrogate's shear box. What comes back is the domain barrier "
                  f"and an extrapolation, not a prediction. Read the guard block below before "
                  f"quoting anything.")

    print(f"  solving {os.path.basename(args.run_dir)} ...")
    results, _, bond_energy, hinge_geometry, cut_axis = solve_deployed_state(config, params_path)

    vs = results['valid_state']
    length_scale = float(config.hinge_model.length_scale)
    fc = np.asarray(vs.face_centroids)
    cnv = np.asarray(vs.centroid_node_vectors)
    # Hinges are indexed the same way ``extract_force_field`` indexes them: bonds between NODES of
    # the flattened (n_faces * n_nodes) array, so the pair order matches ``load_path.npz``.
    bond_pairs = np.asarray(vs.bond_connectivity)
    n_nodes = cnv.shape[1]
    face_pairs = np.stack([bond_pairs[:, 0] // n_nodes, bond_pairs[:, 1] // n_nodes], axis=-1)

    disp = np.asarray(results['solution'].fields)                    # (K, n_faces, 3)
    K = disp.shape[0]
    zero = np.zeros_like(disp[0])
    flat_nodes = deployed_nodes(fc, cnv, zero) * length_scale
    flat_xy = fc * length_scale

    nodes = np.stack([deployed_nodes(fc, cnv, disp[k]) for k in range(K)]) * length_scale
    xy = (fc[None] + disp[:, :, :2]) * length_scale
    # Hinge position: midpoint of the bonded node pair, which is what the green tape marks.
    flatn = nodes.reshape(K, -1, 2)
    hinge_xy = 0.5 * (flatn[:, bond_pairs[:, 0]] + flatn[:, bond_pairs[:, 1]])

    rot = tile_rotations(flat_nodes, nodes)                          # (K, n_faces)
    opening = hinge_opening(rot, face_pairs)                         # (K, n_hinges)
    guard = domain_guard(results, vs, hinge_geometry, config, length_scale)

    clamped = [int(c) for c in (config.topology.get('bc_clamped') or [])]
    loaded, load_dirs = [], []
    for spec in config.topology.get('loads', []) or []:
        f = spec.get('face')
        faces = list(f) if isinstance(f, (list, tuple)) else [f]
        loaded += faces
        # The applied direction, as a unit vector in the design frame, so a figure can draw the
        # load arrow from the boundary condition rather than from the response to it.
        dof = int(spec.get('dof', 1))
        sign = 1.0 if float(spec.get('value', 1.0)) >= 0 else -1.0
        vec = [sign, 0.0] if dof == 0 else ([0.0, sign] if dof == 1 else [0.0, 0.0])
        load_dirs += [vec] * len(faces)
    loaded = [int(f) for f in loaded]

    rows = []
    for k in range(K):
        pts = nodes[k].reshape(-1, 2)
        w, h, a = min_area_rect(pts)
        rows.append({
            "step": k,
            "t": (k + 1) / K * args.load_scale,
            "mean_opening_deg": float(opening[k].mean()),
            "max_opening_deg": float(opening[k].max()),
            "min_opening_deg": float(opening[k].min()),
            "rect_w_mm": float(w), "rect_h_mm": float(h), "rect_angle_deg": float(a),
            "hull_area_mm2": hull_area(pts),
            "u_pull_mm": float(np.linalg.norm(xy[k, loaded[0]] - flat_xy[loaded[0]])) if loaded
                         else float("nan"),
            "grip_sep_mm": float(np.linalg.norm(xy[k, loaded[0]] - xy[k, clamped[0]]))
                           if (loaded and clamped) else float("nan"),
        })

    stem = args.out or os.path.join(args.run_dir, "load_path_steps")
    os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)
    arrays = dict(disp=disp, face_xy=xy, face_nodes=nodes, hinge_xy=hinge_xy,
                  t=np.array([r["t"] for r in rows]), rot_deg=rot, opening_deg=opening,
                  load_scale=np.float64(args.load_scale),
                  flat_face_xy=flat_xy, flat_face_nodes=flat_nodes, face_pairs=face_pairs,
                  length_scale=np.float64(length_scale),
                  clamped_faces=np.array(clamped, dtype=np.int32),
                  loaded_faces=np.array(loaded, dtype=np.int32),
                  load_dirs=np.array(load_dirs, dtype=float).reshape(-1, 2))
    tgt = target_rect_mm(config, length_scale)
    if tgt is not None:
        arrays["target_rect_mm"] = tgt
    np.savez_compressed(stem + ".npz", **arrays)

    meta = {"run_dir": os.path.abspath(args.run_dir), "n_steps": K,
            "load_scale": float(args.load_scale),
            "load_newtons": [float(t['value']) for t in (config.topology.get('loads') or [])],
            "domain_guard": guard,
            "n_faces": int(fc.shape[0]), "n_hinges": int(face_pairs.shape[0]),
            "length_scale_mm_per_unit": length_scale,
            "clamped_faces": clamped, "loaded_faces": loaded,
            "target_rot_deg": float(config.topology.get('target_rot_deg', 0.0)),
            "steps": rows}
    with open(stem + ".json", "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"  wrote {stem}.npz / .json   {K} steps, {fc.shape[0]} tiles, "
          f"{face_pairs.shape[0]} hinges")
    print(format_guard(guard))
    print(f"  {'step':>5} {'t':>6} {'mean open':>10} {'max open':>9} {'rect w x h':>18} "
          f"{'|u_pull|':>9} {'grip sep':>9}")
    for r in rows[:: max(1, K // 8)] + [rows[-1]]:
        print(f"  {r['step']:5d} {r['t']:6.3f} {r['mean_opening_deg']:9.2f}d "
              f"{r['max_opening_deg']:8.2f}d {r['rect_w_mm']:8.1f} x {r['rect_h_mm']:7.1f} "
              f"{r['u_pull_mm']:9.1f} {r['grip_sep_mm']:9.1f}")

    if args.with_forces:
        from nff.closed.load_paths import extract_force_field, save_field
        print("\n  --with-forces: FINAL step only; intermediate steps carry a wrong load scale.")
        field = extract_force_field(results, config.physics, bond_energy_fn=bond_energy,
                                    hinge_geometry=hinge_geometry, cut_axis=cut_axis)
        save_field(field, stem + "_forces", meta={"run_dir": args.run_dir, "step": -1})
        print(f"  wrote {stem}_forces.npz")


if __name__ == "__main__":
    main()
