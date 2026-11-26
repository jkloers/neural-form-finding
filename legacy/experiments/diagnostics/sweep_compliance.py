"""Is the departure from the ideal mechanism just hinge compliance? Sweep the slip stiffness.

A stiff mode responds in proportion to ``F/k``, so if the gap between the real sheet and the ideal
1-DOF mechanism is nothing but the hinge's own compliance, every measure of that gap should fall as
``1/k``. This sweeps the slip stiffnesses (``k_stretch`` and ``k_shear`` together) at a FIXED load and
measures two independent departures:

    void distortion  -- how far the voids are from parallelograms, a geometric measure
    angle residual   -- how much of the tile-rotation field the ideal checkerboard fails to explain

Both are dimensionless and neither is used to fit the other, so agreeing on a slope of -1 is a real
check rather than a restatement.

⚠ Only the slip stiffnesses are scaled. ``k_rot`` is left alone: it is the mechanism's own stiffness,
and scaling it would change which deployment the load produces rather than how far that deployment
departs from the ideal.

Run:
    JAX_PLATFORMS=cpu PYTHONPATH=$(pwd) python legacy/experiments/diagnostics/sweep_compliance.py \
        --n-stiff 16 --k-min 0.3 --k-max 300
"""

import argparse
import copy
import json
import os
import time

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np

from nff.closed import void_geometry as VG
from nff.analysis.load_paths import deployed_nodes
from nff_paper.closed_support.one_dof import fit_one_dof
from nff_paper.closed_support.shield_kinematics import _checkerboard_signs
from legacy.experiments.diagnostics.sweep_localisation import (BASE_CONFIG, build_arm_config, loads_for,
                                                        solve_at)
from nff.topology.closed_builder_jax import build_deploy_structure


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=BASE_CONFIG)
    ap.add_argument("--axis", default="cantilever", choices=("open", "cantilever"))
    ap.add_argument("--level", type=float, default=1.0,
                    help="Load level held fixed while the stiffness varies. The default 1.0 is a "
                         "cantilever ratio safely below the 1x arm's switch at 1.64.")
    ap.add_argument("--n-stiff", type=int, default=16)
    ap.add_argument("--k-min", type=float, default=0.3)
    ap.add_argument("--k-max", type=float, default=300.0)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--solver-tol", type=float, default=1e-8)
    ap.add_argument("--out", default=os.path.join("data/outputs/localisation", "compliance"))
    args = ap.parse_args()

    mults = np.geomspace(args.k_min, args.k_max, int(args.n_stiff))
    os.makedirs(args.out, exist_ok=True)

    base, params_path = build_arm_config("squares", args.config)
    base.physics.solver_tol = float(args.solver_tol)
    base.physics.solver_maxiter = 60000
    k_stretch0 = float(base.topology['k_stretch'])
    k_shear0 = float(base.topology['k_shear'])
    M, N = int(base.topology['M']), int(base.topology['N'])
    sigma = _checkerboard_signs(M, N).astype(float)
    void_index = VG.build_void_index(build_deploy_structure(M, N))
    length_scale = float(getattr(base.hinge_model, 'length_scale', 1.0))
    loads = loads_for(args.axis, float(args.level))

    print(f"[compliance] axis={args.axis} level={args.level}  slip stiffness "
          f"x{mults[0]:.3g}..x{mults[-1]:.3g} in {len(mults)} steps", flush=True)
    print(f"\n{'x k':>9} {'psi deg':>9} {'void distortion':>16} {'angle resid':>12} "
          f"{'conv':>5} {'t(s)':>6}", flush=True)

    rows = []
    for m in mults:
        t0 = time.time()
        cfg = copy.deepcopy(base)
        cfg.topology['k_stretch'] = k_stretch0 * float(m)
        cfg.topology['k_shear'] = k_shear0 * float(m)
        try:
            res, c, _, _, _, _, _ = solve_at(cfg, params_path, loads, args.steps)
            vs = res['valid_state']
            disp = np.asarray(res['solution'].fields[-1])
            nodes = np.asarray(deployed_nodes(vs.face_centroids, vs.centroid_node_vectors, disp))
            rep = VG.void_report(void_index, nodes, length_scale=length_scale)
            fit = fit_one_dof(disp[:, 2], sigma)
        except Exception as exc:
            print(f"{m:9.3g}  FAILED {type(exc).__name__}: {exc}", flush=True)
            jax.clear_caches()
            continue
        row = dict(mult=float(m), psi=fit.psi, para_mean=rep['para_mean'],
                   angle_resid=fit.rms_over_psi, r2=fit.r2, wall_s=time.time() - t0)
        rows.append(row)
        print(f"{m:9.3g} {np.degrees(fit.psi):9.3f} {rep['para_mean']:16.4e} "
              f"{fit.rms_over_psi:12.4e} {'yes':>5} {row['wall_s']:6.1f}", flush=True)
        jax.clear_caches()

    with open(os.path.join(args.out, "compliance.json"), "w") as fh:
        json.dump({"axis": args.axis, "level": args.level, "rows": rows}, fh, indent=2, default=float)

    if len(rows) > 2:
        k = np.array([r['mult'] for r in rows])
        for key in ("para_mean", "angle_resid"):
            v = np.array([r[key] for r in rows])
            good = v > 0
            slope = np.polyfit(np.log10(k[good]), np.log10(v[good]), 1)[0]
            print(f"[compliance] {key}: log-log slope {slope:+.4f} over "
                  f"{np.log10(k.max() / k.min()):.1f} decades")
    print(f"[compliance] wrote {args.out}/compliance.json  ({len(rows)}/{len(mults)})")


if __name__ == "__main__":
    main()
