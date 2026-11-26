"""Can a single moment on one tile open the whole sheet evenly? -- the gate on the circle problem.

The proposed drive is one fully clamped tile and a moment on the tile beside it. That is attractive
because it is minimal, but a single-tile grip is known to deploy a sheet very unevenly: the recorded
experience on this project is a hinge-angle spread of order 20x, with some hinges never opening at
all. A sheet whose near tiles are at full deployment while its far tiles are still shut cannot match
a circle under any design, so this is checked before the circle target is built rather than after.

The sweep reports, per moment level, the relative rotation across every hinge -- its maximum, mean
and minimum -- and the ratio min/max, which is the number that decides. It also reports the deployed
outline's radial spread about its centroid, since matching a circle means driving that spread to
zero and a drive that cannot even produce a roughly round outline will not get there.

    /opt/miniconda3/envs/kgnn_mac/bin/python -u \\
        nff/scripts/diagnostics/probe_moment_drive.py --clamp 12 --drive 13
"""

import argparse
from copy import deepcopy

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np

from nff.closed.deploy import _boundary_cloud
from nff.closed.setup import build_closed_initial_state, init_closed_les_params
from nff.config.experiment import load_and_parse_config
from nff.stages.pipeline import forward_pipeline

MIN_RATIO = 0.5      # working go/no-go: the quietest hinge must reach half the loudest


def build_loads(mode, M, M_N, clamp, drive):
    """The load set for each candidate drive.

    ``single``      one moment on one tile -- minimal, and the thing being tested.
    ``checkerboard`` a moment on EVERY tile, alternating sign with (i+j) parity. This is the
                    natural actuation of a rotating-squares mechanism: adjacent tiles counter-rotate,
                    so a checkerboard moment field drives exactly the deployment mode rather than
                    fighting it.
    ``radial``      outward forces on the boundary tiles, pointing away from the sheet centre --
                    the drive whose symmetry matches a circular target.
    """
    if mode == 'single':
        return [{'face': int(drive), 'dof': 2, 'value': float(M)}]
    if mode == 'checkerboard':
        out = []
        for i in range(M_N):
            for j in range(M_N):
                f = i * M_N + j
                if f == clamp:
                    continue
                out.append({'face': f, 'dof': 2, 'value': float(M) * (1 if (i + j) % 2 == 0 else -1)})
        return out
    if mode == 'radial':
        out, c = [], (M_N - 1) / 2.0
        for i in range(M_N):
            for j in range(M_N):
                f = i * M_N + j
                if f == clamp or not (i in (0, M_N - 1) or j in (0, M_N - 1)):
                    continue
                dx, dy = i - c, j - c
                n = max((dx * dx + dy * dy) ** 0.5, 1e-9)
                out.append({'face': f, 'dof': 0, 'value': float(M) * dx / n})
                out.append({'face': f, 'dof': 1, 'value': float(M) * dy / n})
        return out
    raise ValueError(mode)


def deploy(cfg, moment, clamp, drive, mode='single', M_N=5):
    """Deploy under a chosen drive and return per-hinge relative rotation and the boundary."""
    import equinox as eqx
    c = deepcopy(cfg)
    topo = dict(c.topology)
    # ⚠ THE CLAMP KEY IS ``bc_clamped``, NOT ``clamped_faces``. The config parser maps
    # boundary_conditions.clamped_faces -> topology['bc_clamped'] (experiment.py:612) and
    # closed/setup.py reads that. Writing ``clamped_faces`` post-parse is silently ignored, which
    # leaves the ORIGINAL clamp in place -- here the whole bottom edge -- so the sheet being driven
    # is not the one intended.
    topo['bc_clamped'] = [int(clamp)]
    topo['clamped_dofs'] = [0, 1, 2]                 # a single tile must be fully restrained:
    topo.pop('y_pin_faces', None)                    # a net torque would otherwise spin the sheet
    topo['loads'] = build_loads(mode, moment, M_N, clamp, drive)
    c.topology = topo
    # ⚠ solver_maxiter 1500 lets Stage-2 stop on the ITERATION CAP and return a silently wrong
    # state; a large moment needs both a real iteration budget and finer load stepping.
    c = eqx.tree_at(lambda x: x.physics.solver_maxiter, c, 20000)
    c = eqx.tree_at(lambda x: x.physics.num_load_steps, c, 24)

    params, sf = init_closed_les_params(c)
    state, _ = build_closed_initial_state(c)
    res = forward_pipeline(state, c.target, c.validity, c.physics, map_type='closed_les',
                           map_params=params, static_features=sf, load_specs=topo['loads'])
    vs = res['valid_state']
    disp = np.asarray(res['solution'].fields[-1])
    bc = np.asarray(vs.bond_connectivity)
    n_nodes = np.asarray(vs.centroid_node_vectors).shape[1]
    th = np.asarray(disp[:, 2])
    theta = np.degrees(np.abs(th[bc[:, 1] // n_nodes] - th[bc[:, 0] // n_nodes]))
    bd = np.asarray(_boundary_cloud(vs, disp))
    jax.clear_caches()
    return theta, bd


def roundness(bd):
    """Radial spread of the deployed outline about its own centroid: 0 for a perfect circle."""
    c = bd.mean(axis=0)
    r = np.linalg.norm(bd - c, axis=1)
    return float(r.std() / max(r.mean(), 1e-12)), float(r.mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="closed/sheet_5x5_square")
    ap.add_argument("--clamp", type=int, default=12)
    ap.add_argument("--drive", type=int, default=13)
    ap.add_argument("--moments", type=float, nargs="+",
                    default=[1e2, 1e3, 1e4, 1e5, 1e6, 1e7])
    ap.add_argument("--mode", default="single",
                    choices=["single", "checkerboard", "radial"])
    args = ap.parse_args()

    cfg = load_and_parse_config(f"data/configs/{args.config}.yaml")
    scale = float(cfg.hinge_model.length_scale)
    print(f"  drive = {args.mode.upper()};  clamp face {args.clamp} (all 3 DOF)")
    print(f"  ideal full deployment for this square sheet: theta = 90 deg (lam = sqrt(2))\n")
    print(f"  {'moment':>10}{'th_max':>9}{'th_mean':>9}{'th_min':>9}{'min/max':>9}"
          f"{'radial spread':>15}{'mean r [mm]':>13}  verdict")

    ok_any = False
    for M in args.moments:
        try:
            theta, bd = deploy(cfg, M, args.clamp, args.drive, args.mode)
        except Exception as e:
            print(f"  {M:10.0e}   FAILED: {type(e).__name__}: {str(e)[:52]}")
            continue
        ratio = theta.min() / max(theta.max(), 1e-12)
        spread, rmean = roundness(bd)
        good = ratio >= MIN_RATIO
        ok_any |= good
        print(f"  {M:10.0e}{theta.max():9.2f}{theta.mean():9.2f}{theta.min():9.2f}"
              f"{ratio:9.2f}{spread:15.3f}{rmean*scale:13.0f}  "
              f"{'uniform' if good else 'UNEVEN'}", flush=True)

    print()
    if ok_any:
        print(f"  GATE PASSED: at least one moment opens the sheet with min/max >= {MIN_RATIO}.")
        print("  The single-tile moment drive can carry the circle problem.")
    else:
        print(f"  GATE FAILED: no moment reaches min/max >= {MIN_RATIO}.")
        print("  A single moment does not open this sheet evenly, so no design matches a circle.")
        print("  Fix before proceeding: drive a RING of tiles, or use displacement control")
        print("  (PR #18) which prescribes the angle instead of bisecting a load for it.")


if __name__ == "__main__":
    main()
