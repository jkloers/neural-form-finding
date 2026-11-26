"""Opening angle of a deployed sheet, measured the same way on both deployment arms.

The angle is the RELATIVE rotation across a hinge: the difference between the deployed
rotations of the two tiles it joins. It has to be relative. The two arms sit in different
frames, because the clamp pins one sublattice under equilibrium while the closed-form map is
symmetric about zero, so no absolute tile rotation compares across them. Twice the mean
absolute tile rotation reproduces the kinematic arm exactly and is still wrong on the other, so
a definition validated on one arm proves nothing about the pair.

Writes ``opening_angle.npz`` (``seed``, ``theta_deg``, ``theta_sd_deg``) into each run folder.

    JAX_PLATFORMS=cpu conda run -n kgnn_mac python \
        nff_paper/analysis/opening_angle.py --arm both
"""

import argparse
import glob
import os
import sys

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np                                                   # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))


def tile_rotations(flat, deployed):
    """Best-fit rotation carrying each flat tile onto its deployed tile, in degrees.

    Args:
        flat: (n_tiles, n_corners, 2) undeformed tile corners.
        deployed: (n_tiles, n_corners, 2) deployed tile corners.

    Returns:
        (n_tiles,) signed rotation in degrees.
    """
    P = flat - flat.mean(1, keepdims=True)
    Q = deployed - deployed.mean(1, keepdims=True)
    H = np.einsum("tki,tkj->tij", P, Q)
    return np.degrees(np.arctan2(H[:, 0, 1] - H[:, 1, 0], H[:, 0, 0] + H[:, 1, 1]))


def hinge_opening(rot_deg, face_pairs):
    """Relative rotation across every hinge, wrapped into [0, 180)."""
    d = np.abs(rot_deg[face_pairs[:, 0]] - rot_deg[face_pairs[:, 1]]) % 360.0
    return np.minimum(d, 360.0 - d)


def hinge_face_pairs(config_path):
    """Tile adjacency, shared by both arms.

    Both charts take ``coords[corner_pid]``, so face ``i`` is the same tile on either arm and
    one adjacency serves both.
    """
    from nff.closed.setup import build_closed_initial_state
    from nff.config.experiment import load_and_parse_config

    state, _ = build_closed_initial_state(load_and_parse_config(config_path))
    return np.asarray(state.hinge_face_pairs)


def measure_kinematic(run, pairs):
    """Closed-form arm: the forward map returns flat and deployed tiles in one call."""
    from nff_paper.closed_support.design_ensemble import circle_problem, problem_statics

    cfg = circle_problem()
    statics = problem_statics(cfg)
    forward_fn = statics["forward_fn"]
    n_bnd = statics["bnd"]["n_logits"]

    seeds, theta, sd = [], [], []
    for path in sorted(glob.glob(os.path.join(run, "designs", "seed*.npz"))):
        with np.load(path) as z:
            v = np.asarray(z["traj_params"])[-1]
            out = forward_fn({"bnd_logits": v[:n_bnd], "psi_raw": v[n_bnd],
                              "z": v[n_bnd + 1:].reshape(cfg.M + 1, cfg.N + 1)})
            rot = tile_rotations(np.asarray(out["flat_tiles"]), np.asarray(out["deployed"]))
            op = hinge_opening(rot, pairs)
            seeds.append(int(z["seed"])); theta.append(op.mean()); sd.append(op.std())
    return np.asarray(seeds), np.asarray(theta), np.asarray(sd)


def measure_physics(run, config_path, pairs):
    """Equilibrium arm: one Stage-2 solve per design, so this is the slow half."""
    import jax.numpy as jnp

    from nff.closed.chart_ablation import fixed_target
    from nff.closed.setup import build_closed_initial_state, init_closed_les_params
    from nff.config.experiment import load_and_parse_config
    from nff.stages.geometry import deformed_vertices
    from nff.stages.pipeline import forward_pipeline

    cfg = load_and_parse_config(config_path)
    _, static_features = init_closed_les_params(cfg)
    state, _ = build_closed_initial_state(cfg)
    target, _ = fixed_target(cfg)

    seeds, theta, sd = [], [], []
    for path in sorted(glob.glob(os.path.join(run, "designs", "seed*.npz"))):
        with np.load(path) as z:
            params = {"z": jnp.asarray(z["params__z"]),
                      "bnd_logits": jnp.asarray(z["params__bnd_logits"])}
            seed = int(z["seed"])
        try:
            res = forward_pipeline(state, target, cfg.validity, cfg.physics,
                                   map_type="closed_les", map_params=params,
                                   static_features=static_features,
                                   load_specs=cfg.topology.get("loads", []))
        except Exception as exc:                       # a design that fails to deploy is
            print(f"  seed {seed:3d} FAILED {type(exc).__name__}", flush=True)
            continue                                   # counted by its absence, not hidden
        vs = res["valid_state"]
        flat = np.asarray(vs.face_centroids)[:, None, :] + np.asarray(vs.centroid_node_vectors)
        dep = np.asarray(deformed_vertices(vs, res["solution"].fields[-1]))
        op = hinge_opening(tile_rotations(flat, dep), pairs)
        seeds.append(seed); theta.append(op.mean()); sd.append(op.std())
        if len(seeds) % 25 == 0:
            print(f"  {len(seeds)} done, running mean {np.mean(theta):.2f} deg", flush=True)
    return np.asarray(seeds), np.asarray(theta), np.asarray(sd)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=("kinematic", "physics", "both"), default="both")
    ap.add_argument("--kinematic", default="data/outputs/design_cloud/circle_7x7_n200_s1")
    ap.add_argument("--physics", default="data/outputs/design_cloud/physics_n200_e150_s1")
    ap.add_argument("--config", default="data/configs/closed/sheet_7x7_circle.yaml")
    args = ap.parse_args()

    pairs = hinge_face_pairs(args.config)
    print(f"  hinge face pairs: {pairs.shape}")

    jobs = []
    if args.arm in ("kinematic", "both"):
        jobs.append(("kinematic", args.kinematic,
                     lambda: measure_kinematic(args.kinematic, pairs)))
    if args.arm in ("physics", "both"):
        jobs.append(("physics", args.physics,
                     lambda: measure_physics(args.physics, args.config, pairs)))

    for name, run, fn in jobs:
        seeds, theta, sd = fn()
        print(f"  {name:9s} n={len(theta)}  mean {theta.mean():.2f} deg  sd {theta.std():.2f}"
              f"  range [{theta.min():.2f}, {theta.max():.2f}]"
              f"  within-design sd median {np.median(sd):.3f} deg")
        np.savez(os.path.join(run, "opening_angle.npz"),
                 seed=seeds, theta_deg=theta, theta_sd_deg=sd)
        print(f"  wrote {os.path.join(run, 'opening_angle.npz')}")


if __name__ == "__main__":
    main()
