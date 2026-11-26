"""Where the work of deployment goes, across a load sweep: rotation against hinge slip.

This is the governing quantity of the localisation study. The sheet has one very soft coordinate
(hinge rotation) and 446 stiff ones (hinge slip), so the fraction of stored work that leaves the
soft coordinate is what the 1-DOF angle residual and the void parallelogram defect are both reading.

Exact, not reconstructed: the three springs are independent quadratics, so evaluating the solver's
own ligament energy three times with the other two stiffnesses zeroed returns each term, and the
three sum to the total. ``force_stats.energy_split`` does it that way and the sum is checked here.

⚠ Report only the TOTAL non-rotational share. The axial/shear split within it is NOT trustworthy on
a closed sheet: ``build_reference_bond_vectors`` returns one degenerate row ``(1e-4, 0)`` for every
hinge, so ``k_stretch`` multiplies the global +x component and ``k_shear`` the +y one, while 49% of
hinges are not aligned with that frame. Rotation is unambiguous and axial+shear is its complement,
so the total is safe; the breakdown is scrambled on about half the sheet.

Run:
    JAX_PLATFORMS=cpu PYTHONPATH=$(pwd) python nff/scripts/diagnostics/energy_partition.py \
        --arm squares --n-loads 9
"""

import argparse
import json
import os
import time

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np

from nff.closed.force_stats import energy_split
from nff.closed.one_dof import fit_one_dof, localisation_ratio, relative_rotations
from nff.closed.shield_kinematics import _checkerboard_signs
from nff.scripts.diagnostics.sweep_localisation import (BASE_CONFIG, build_arm_config, loads_for,
                                                        solve_at)
from nff.stages.geometry import build_reference_bond_vectors

SUM_TOL = 1e-6


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", default="squares")
    ap.add_argument("--axis", default="open", choices=("open", "cantilever"))
    ap.add_argument("--config", default=BASE_CONFIG)
    ap.add_argument("--n-loads", type=int, default=9)
    ap.add_argument("--f-min", type=float, default=0.5)
    ap.add_argument("--f-max", type=float, default=2000.0)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--solver-tol", type=float, default=1e-8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    levels = np.geomspace(args.f_min, args.f_max, int(args.n_loads))
    out_dir = args.out or f"data/outputs/localisation/energy_{args.axis}_{args.arm}"
    os.makedirs(out_dir, exist_ok=True)

    config, params_path = build_arm_config(args.arm, args.config)
    config.physics.solver_tol = float(args.solver_tol)
    config.physics.solver_maxiter = 60000
    M, N = int(config.topology['M']), int(config.topology['N'])
    sigma = _checkerboard_signs(M, N).astype(float)

    print(f"[energy] arm={args.arm} axis={args.axis} levels {levels[0]:.4g}..{levels[-1]:.4g}",
          flush=True)
    print(f"\n{'level':>8} {'psi deg':>9} {'theta deg':>10} {'rotation %':>11} {'NON-ROT %':>10} "
          f"{'1-DOF R2':>10} {'loc':>7} {'sum chk':>9} {'t(s)':>6}", flush=True)

    rows = []
    for lv in levels:
        t0 = time.time()
        loads = loads_for(args.axis, float(lv))
        try:
            res, c, params, sf, be, hg, ca = solve_at(config, params_path, loads, args.steps)
            vs = res['valid_state']
            disp = np.asarray(res['solution'].fields[-1])
            rbv = res.get('reference_bond_vectors')
            rbv = np.asarray(rbv if rbv is not None else build_reference_bond_vectors(vs))
            es = energy_split(vs, disp, rbv,
                              k_stretch=float(c.topology['k_stretch']),
                              k_shear=float(c.topology['k_shear']),
                              k_rot=float(c.topology['k_rot']),
                              linearized=bool(c.physics.linearized_strains))
            shares = es['shares']
            chk = abs(sum(shares.values()) - 1.0)
            if chk > SUM_TOL:
                raise ValueError(f"energy shares sum to {sum(shares.values())!r}, off by {chk:.2e}")
            fit = fit_one_dof(disp[:, 2], sigma)
            fp = np.asarray(vs.bond_connectivity)
            n_nodes = np.asarray(vs.centroid_node_vectors).shape[1]
            pairs = np.column_stack([fp[:, 0] // n_nodes, fp[:, 1] // n_nodes])
            loc = localisation_ratio(disp[:, 2], pairs, fit.psi)
            theta = float(np.abs(relative_rotations(disp[:, 2], pairs)).mean())
        except Exception as exc:
            print(f"{lv:8.4g}  FAILED {type(exc).__name__}: {exc}", flush=True)
            jax.clear_caches()
            continue

        row = dict(level=float(lv), psi=fit.psi, theta_mean=theta, r2=fit.r2, localisation=loc,
                   rotation=shares['rotation'], axial=shares['axial'], shear=shares['shear'],
                   non_rotational=1.0 - shares['rotation'], total=es['total'],
                   sum_check=chk, converged=bool(res is not None), wall_s=time.time() - t0)
        rows.append(row)
        print(f"{lv:8.4g} {np.degrees(fit.psi):9.3f} {np.degrees(theta):10.3f} "
              f"{100*shares['rotation']:11.5f} {100*row['non_rotational']:10.5f} {fit.r2:10.6f} "
              f"{loc:7.3f} {chk:9.1e} {row['wall_s']:6.1f}", flush=True)
        jax.clear_caches()

    with open(os.path.join(out_dir, "energy_partition.json"), "w") as fh:
        json.dump({"arm": args.arm, "axis": args.axis, "config": args.config,
                   "note": "non_rotational = axial + shear; the axial/shear SPLIT is not "
                           "trustworthy on a closed sheet (degenerate reference frame), the TOTAL "
                           "is.", "rows": rows}, fh, indent=2, default=float)
    if rows:
        nr = np.array([r['non_rotational'] for r in rows])
        lv = np.array([r['level'] for r in rows])
        th = np.degrees([r['theta_mean'] for r in rows])
        print(f"\n[energy] non-rotational share: {100*nr[0]:.2f}% at F={lv[0]:.4g} -> "
              f"MIN {100*nr.min():.2f}% at F={lv[nr.argmin()]:.4g} "
              f"(theta={th[nr.argmin()]:.1f} deg) -> {100*nr[-1]:.2f}% at F={lv[-1]:.4g}")
    print(f"[energy] wrote {out_dir}/energy_partition.json  ({len(rows)}/{len(levels)})")


if __name__ == "__main__":
    main()
