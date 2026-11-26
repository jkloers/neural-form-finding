"""Is the cantilever collapse a limit point? Track the Stage-2 Hessian across the load ratio.

A force-controlled solver cannot follow a descending equilibrium branch, so a snap in a sweep is
consistent with a limit point but does not prove one -- the sweep only shows that the solver landed
somewhere else. The proof is in the tangent stiffness: at a limit point the Hessian of the total
potential energy loses positive definiteness, so ``n_neg`` (its count of negative eigenvalues) goes
from zero to non-zero, and the eigenvector belonging to the smallest eigenvalue IS the mode the
sheet collapses into.

Nothing here is new machinery. ``_build_energy_over_free_dofs`` and ``_spectrum`` come from
``diagnose_conditioning``, the load loop from ``sweep_localisation.solve_at``. The one addition is
scattering the critical eigenvector back to ``(n_faces, 3)`` so it can be drawn.

⚠ The Hessian is read UNREGULARISED. ``physics.backward_reg`` is a Tikhonov ridge on the adjoint
solve only, and adding it here would mask exactly the indefiniteness being looked for.

⚠ Each level is an independent solve whose loads are baked into the state, so the energy is rebuilt
per level and evaluated at that level's own equilibrium. The gradient is therefore expected to
vanish, and it is asserted rather than assumed -- a large ``grad_norm`` means the state is not
stationary and its spectrum says nothing.

Run:
    JAX_PLATFORMS=cpu PYTHONPATH=$(pwd) python legacy/experiments/diagnostics/snap_spectrum.py \
        --axis cantilever --f-min 1.40 --f-max 2.00 --n-loads 12
"""

import argparse
import json
import os
import time

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np

from legacy.experiments.diagnostics.diagnose_conditioning import (_build_energy_over_free_dofs,
                                                           _spectrum)
from legacy.experiments.diagnostics.sweep_localisation import (BASE_CONFIG, build_arm_config, loads_for,
                                                        solve_at)
from nff.closed.one_dof import fit_one_dof
from nff.closed.shield_kinematics import _checkerboard_signs


def critical_mode(energy_free_fn, free_DOF_ids, final_free, n_faces: int):
    """Spectrum at one equilibrium, plus the critical eigenvector as an (n_faces, 3) field.

    Returns the ``_spectrum`` dict (without the full eigenvalue list), the gradient norm as the
    stationarity check, and ``mode`` -- the unit eigenvector of the smallest eigenvalue scattered
    back onto the face DOFs, zero on constrained ones.
    """
    H = np.asarray(jax.hessian(energy_free_fn)(final_free))
    H = 0.5 * (H + H.T)                                  # symmetrise the last ulp
    evals, evecs = np.linalg.eigh(H)
    spec = _spectrum(H)
    spec.pop("w", None)
    grad = np.asarray(jax.grad(energy_free_fn)(final_free))

    mode = np.zeros(n_faces * 3, dtype=float)
    mode[np.asarray(free_DOF_ids)] = evecs[:, 0]
    return spec, float(np.linalg.norm(grad)), evals[0], mode.reshape(n_faces, 3)


def emit_states(args):
    """Solve at each requested level and persist the DEPLOYED PANEL CORNERS.

    The sweeps kept per-tile scalars but never node positions, so anything that draws the sheet has
    to re-solve. Saving the corners once means a figure never re-solves again.
    """
    from nff.closed.load_paths import deployed_nodes

    config, params_path = build_arm_config(args.arm, args.config)
    config.physics.solver_tol = float(args.solver_tol)
    config.physics.solver_maxiter = 60000
    M, N = int(config.topology['M']), int(config.topology['N'])
    sigma = _checkerboard_signs(M, N).astype(float)

    out_dir = args.out or "data/outputs/localisation"
    os.makedirs(out_dir, exist_ok=True)
    store, meta = {}, []
    for k, lv in enumerate(args.emit_states):
        res, c, params, sf, be, hg, ca = solve_at(config, params_path,
                                                  loads_for(args.axis, float(lv)), args.steps)
        vs = res['valid_state']
        disp = np.asarray(res['solution'].fields[-1])
        nodes = np.asarray(deployed_nodes(vs.face_centroids, vs.centroid_node_vectors, disp))
        fit = fit_one_dof(disp[:, 2], sigma)
        store[f"nodes__{k}"] = nodes
        store[f"flat__{k}"] = np.asarray(deployed_nodes(vs.face_centroids,
                                                        vs.centroid_node_vectors,
                                                        np.zeros_like(disp)))
        meta.append(dict(level=float(lv), psi=float(fit.psi), r2=float(fit.r2)))
        print(f"[states] level={lv:.4g}  psi={np.degrees(fit.psi):8.3f} deg  "
              f"nodes {nodes.shape}", flush=True)
        jax.clear_caches()

    store["clamped"] = np.asarray(config.topology.get('bc_clamped', []) or [], dtype=np.int64)
    store["levels"] = np.asarray([m['level'] for m in meta], dtype=float)
    store["psi"] = np.asarray([m['psi'] for m in meta], dtype=float)
    path = os.path.join(out_dir, "two_states.npz")
    np.savez_compressed(path, **store)
    print(f"[states] wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", default="squares")
    ap.add_argument("--axis", default="cantilever", choices=("open", "cantilever"))
    ap.add_argument("--config", default=BASE_CONFIG)
    ap.add_argument("--n-loads", type=int, default=12)
    ap.add_argument("--f-min", type=float, default=1.40)
    ap.add_argument("--f-max", type=float, default=2.00)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--solver-tol", type=float, default=1e-8)
    ap.add_argument("--out", default=None)
    ap.add_argument("--emit-states", nargs="*", type=float, default=None,
                    help="Instead of a spectrum sweep, solve at these load levels and persist the "
                         "DEPLOYED PANEL CORNERS. The sweeps never saved node positions, so any "
                         "figure that draws the sheet needs this once.")
    args = ap.parse_args()

    if args.emit_states is not None:
        emit_states(args)
        return

    levels = np.geomspace(args.f_min, args.f_max, int(args.n_loads))
    out_dir = args.out or f"data/outputs/localisation/spectrum_{args.axis}_{args.arm}"
    os.makedirs(out_dir, exist_ok=True)

    config, params_path = build_arm_config(args.arm, args.config)
    config.physics.solver_tol = float(args.solver_tol)
    config.physics.solver_maxiter = 60000
    M, N = int(config.topology['M']), int(config.topology['N'])
    sigma = _checkerboard_signs(M, N).astype(float)

    print(f"[spectrum] arm={args.arm} axis={args.axis} levels "
          f"{levels[0]:.4g}..{levels[-1]:.4g}  (unregularised H, n_free from the solve)", flush=True)
    print(f"\n{'level':>8} {'psi deg':>9} {'lam_min':>12} {'n_neg':>6} {'n_zero':>7} "
          f"{'cond':>11} {'grad':>10} {'t(s)':>6}", flush=True)

    rows, modes = [], {}
    for i, lv in enumerate(levels):
        t0 = time.time()
        loads = loads_for(args.axis, float(lv))
        try:
            res, c, params, sf, be, hg, ca = solve_at(config, params_path, loads, args.steps)
            vs = res['valid_state']
            fields = np.asarray(res['solution'].fields)
            n_faces = fields.shape[1]
            energy_free_fn, free_ids, _ = _build_energy_over_free_dofs(
                vs, c.physics, loads, bool(c.physics.use_contact),
                bond_energy_fn=be, hinge_geometry=hg)
            final_free = fields[-1].reshape(-1)[np.asarray(free_ids)]
            spec, gnorm, lam0, mode = critical_mode(energy_free_fn, free_ids, final_free, n_faces)
            fit = fit_one_dof(fields[-1][:, 2], sigma)
        except Exception as exc:
            print(f"{lv:8.4g}  FAILED {type(exc).__name__}: {exc}", flush=True)
            jax.clear_caches()
            continue

        row = dict(level=float(lv), psi=fit.psi, r2=fit.r2, lam_min_signed=float(lam0),
                   grad_norm=gnorm, wall_s=time.time() - t0, **spec)
        rows.append(row)
        modes[f"mode__{i}"] = mode
        print(f"{lv:8.4g} {np.degrees(fit.psi):9.3f} {lam0:12.4e} {spec['n_neg']:6d} "
              f"{spec['n_zero']:7d} {spec['cond']:11.3e} {gnorm:10.2e} {row['wall_s']:6.1f}",
              flush=True)
        jax.clear_caches()

    np.savez_compressed(os.path.join(out_dir, "spectrum.npz"),
                        levels=np.asarray([r['level'] for r in rows]), **modes)
    with open(os.path.join(out_dir, "spectrum.json"), "w") as fh:
        json.dump({"arm": args.arm, "axis": args.axis, "rows": rows}, fh, indent=2, default=float)

    if rows:
        neg = [r for r in rows if r['n_neg'] > 0]
        if neg and len(neg) < len(rows):
            first = neg[0]['level']
            prev = max(r['level'] for r in rows if r['level'] < first)
            print(f"\n[spectrum] n_neg crosses 0 -> {neg[0]['n_neg']} between {prev:.4g} and "
                  f"{first:.4g}: the branch LOSES positive definiteness there.")
        elif not neg:
            print("\n[spectrum] n_neg = 0 at every level. The Hessian stays positive definite, so "
                  "these are two coexisting MINIMA and the solver jumped between them; the word "
                  "'limit point' is not earned by this evidence.")
        else:
            print("\n[spectrum] n_neg > 0 at EVERY level -- the sweep starts past the critical "
                  "point; widen --f-min.")
    print(f"[spectrum] wrote {out_dir}/spectrum.npz + spectrum.json  ({len(rows)}/{len(levels)})")


if __name__ == "__main__":
    main()
