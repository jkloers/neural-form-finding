"""What the interaction range of the sheet depends on: deployment, load, and stiffness contrast.

A local load on the deployed sheet raises a disturbance that falls off over a few tiles. The
question this answers is whether that fall-off is a constant of the sheet or a function of what is
done to it. Three arms:

    --arm deployment  the background hinge rotation is swept and the decay measured at each.
    --arm magnitude   full nonlinear solves at growing probe, to find where the load stops being a
                      perturbation and the linear answer stops applying.
    --arm stiffness   the two stiff springs are scaled together, which is the direct test of the
                      claim that the length is set by the contrast between the soft mode and the
                      rest.

★ The magnitude arm is deliberately small, because most of that question is settled without running
anything. In the linear layer the response is exactly ``K^-1 f``, so any field homogeneous in the
response has a shape that cannot depend on the probe magnitude. ``pocket.energy_decomposition``
splits the measured energy change into its even and odd parts, which ARE homogeneous, so the decay
of each is a property of the sheet alone. What is left to measure is only where linearity fails.

⚠⚠ The probe is a FRACTION OF THE BACKGROUND DRIVE, never a fixed number of newtons. The same
absolute probe is a small perturbation on a fully deployed sheet and a re-deployment on a barely
opened one, which would confound the deployment arm with the magnitude arm entirely.

⚠ ``own`` is second order in the probe but is differenced from energies that are not, so too small a
probe returns round-off. Every level reports ``own_rel`` from ``decomposition_precision``; a level
above ``--precision-gate`` is refused rather than recorded.

Run:
    JAX_PLATFORMS=cpu PYTHONPATH=$(pwd) python nff/scripts/diagnostics/sweep_pocket.py \
        --config pocket_15 --arm deployment --n-levels 9
"""

import argparse
import copy
import json
import os
import time

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np

from nff.config.experiment import load_and_parse_config
from nff.closed.one_dof import graph_distance, relative_rotations
from nff.closed.pocket import decay_ratios, load_specs, scatter_load
from nff.closed.shield_kinematics import _checkerboard_signs
from nff.scripts.diagnostics.diagnose_conditioning import _build_energy_over_free_dofs
from nff.scripts.diagnostics.probe_pocket import (Sheet, build_probe, checkerboard_background,
                                                  resolve_config)
from nff.scripts.diagnostics.sweep_localisation import solve_at

# Measured on pocket_11: theta is linear in the per-tile background moment, -1 -> 13.15 deg,
# -3 -> 39.44, -6 -> 78.89. The sign is load-bearing: the opposite one drives the sheet into the
# contact barrier, where every magnitude returns theta = 0 and looks like a solver failure.
MOMENT_PER_DEGREE = -3.42 / 45.0


def background_for_angle(sigma: np.ndarray, target_deg: float) -> list:
    """Background drive expected to open the sheet to ``target_deg``, from the measured slope."""
    return checkerboard_background(sigma, MOMENT_PER_DEGREE * float(target_deg))


def probe_from_background(magnitude_per_tile: float, fraction: float) -> float:
    """Probe magnitude as a fraction of the background drive, so the two axes stay separate."""
    return abs(float(magnitude_per_tile)) * float(fraction)


# ══════════════════════════════════════════════════════════════════════════════

class Tangent:
    """The tangent stiffness at one background equilibrium, and the projections built from it."""

    def __init__(self, sheet: Sheet):
        energy_free_fn, free_ids, _ = _build_energy_over_free_dofs(
            sheet.valid_state, sheet.parsed.physics, sheet.background_loads,
            bool(sheet.parsed.physics.use_contact),
            bond_energy_fn=sheet.bond_energy, hinge_geometry=sheet.hinge_geometry)
        self.free_ids = np.asarray(free_ids)
        self.n_faces = sheet.n_faces
        final_free = sheet.displacement.reshape(-1)[self.free_ids]
        self.grad_norm = float(np.linalg.norm(np.asarray(jax.grad(energy_free_fn)(final_free))))
        K = np.asarray(jax.hessian(energy_free_fn)(final_free))
        self.K = 0.5 * (K + K.T)
        self.evals, self.evecs = np.linalg.eigh(self.K)

    def scatter(self, free_vec: np.ndarray) -> np.ndarray:
        full = np.zeros(self.n_faces * 3)
        full[self.free_ids] = free_vec
        return full.reshape(self.n_faces, 3)

    def restrict(self, field: np.ndarray) -> np.ndarray:
        return np.asarray(field).reshape(-1)[self.free_ids]

    def stiff_part(self, delta_field: np.ndarray, n_soft: int) -> np.ndarray:
        """Drop the softest ``n_soft`` modes from a displacement field.

        ⚠ This must be applied to a NONLINEAR response too, and not only when the response is built
        modally. Filtering one side and not the other compares a stiff-subspace field against a full
        one: measured, that returned a first-neighbour drop of 0.7 against 8.7 and read as a
        collapse of the localisation with load, which it was not.
        """
        d = self.restrict(delta_field)
        coef = self.evecs.T @ d
        coef[:n_soft] = 0.0
        return self.scatter(self.evecs @ coef)

    def response(self, f_free: np.ndarray, n_soft: int) -> np.ndarray:
        coef = self.evecs.T @ f_free
        return self.scatter(self.evecs[:, n_soft:] @ (coef[n_soft:] / self.evals[n_soft:]))


def linear_decay(sheet: Sheet, probe_magnitude: float, angle: float, n_soft: int,
                 tangent: "Tangent" = None) -> dict:
    """One background state, one probe, and the decay of each part of the energy change.

    Builds the response over the stiff modes only, and returns the decay of the probe's own stress
    field and of its redistribution of the background stress.
    """
    tan = tangent if tangent is not None else Tangent(sheet)
    grad_norm, evals = tan.grad_norm, tan.evals

    probe = build_probe(sheet, float(angle), float(probe_magnitude))
    f_free = tan.restrict(scatter_load(probe, sheet.faces, sheet.n_faces))
    delta = tan.response(f_free, n_soft)

    parts = sheet.probe_fields(delta)
    dist = graph_distance(sheet.pairs, sheet.n_faces, [sheet.centre])
    out = {"grad_norm": grad_norm, "lam0": float(evals[0]), "lam1": float(evals[1]),
           "n_free": int(tan.free_ids.size), "precision": parts["precision"],
           "probe": float(probe_magnitude), "angle": float(angle)}
    for name in ("own", "redist", "total"):
        out[name] = decay_ratios(parts[name], dist).as_dict()
    return out, parts, dist


def sheet_state(config, sigma, background, n_steps: int, args) -> Sheet:
    return Sheet(config, n_steps, background_loads=background, pair_axis=args.pair_axis)


def theta_of(sheet: Sheet) -> dict:
    rel = np.abs(relative_rotations(sheet.displacement[:, 2], sheet.pairs))
    return {"mean": float(np.degrees(rel.mean())), "min": float(np.degrees(rel.min())),
            "max": float(np.degrees(rel.max())),
            "uniformity": float(rel.min() / rel.max())}


def report(tag, theta, row):
    own, redist = row["own"], row["redist"]
    print(f"  {tag}  theta {theta['mean']:6.2f} deg  "
          f"own r1 {own['ratios'][0]:6.2f} r2 {own['ratios'][1]:5.2f} "
          f"lam {own['decay_length']:5.2f} (R2 {own['r2']:.3f})   "
          f"redist r1 {redist['ratios'][0]:6.2f}   prec {row['precision']['own_rel']:.1e}",
          flush=True)


def gate(tag, theta, row, args) -> bool:
    """Refuse a level whose background is not an equilibrium or whose probe is round-off.

    A Hessian taken away from equilibrium is a tangent stiffness of nothing, and ``own`` differenced
    from a probe that is too small is noise wearing the shape of an answer. Both are silent
    failures, so both are checked at every level rather than at the ones that look suspicious.
    """
    if row["grad_norm"] > args.grad_gate:
        print(f"  [skip] {tag}: background not stationary (|grad| {row['grad_norm']:.2e})",
              flush=True)
        row["rejected"] = "not stationary"
    elif row["precision"]["own_rel"] > args.precision_gate:
        print(f"  [skip] {tag}: probe too small, own is round-off "
              f"({row['precision']['own_rel']:.1e})", flush=True)
        row["rejected"] = "round-off"
    else:
        report(tag, theta, row)
        return True
    return False


# ══════════════════════════════════════════════════════════════════════════════

def arm_deployment(config, sigma, args) -> list:
    angles = np.linspace(args.angle_min, args.angle_max, int(args.n_levels))
    rows = []
    for a in angles:
        t0 = time.time()
        background = background_for_angle(sigma, float(a))
        sheet = sheet_state(config, sigma, background, args.steps, args)
        theta = theta_of(sheet)
        probe = probe_from_background(MOMENT_PER_DEGREE * float(a), args.probe_frac)
        row, parts, _ = linear_decay(sheet, probe, args.angle, args.n_soft)
        row.update({"target_deg": float(a), "theta": theta, "wall_s": time.time() - t0})
        gate(f"target {a:5.1f} deg", theta, row, args)
        rows.append(row)
        jax.clear_caches()
    return rows


def arm_stiffness(config, sigma, args) -> list:
    rows = []
    # Over the range panel (c) of the manuscript figure already uses, so the two panels sit on one
    # axis and the reader compares them directly.
    for factor in np.geomspace(args.k_min, args.k_max, int(args.n_levels)):
        factor = float(factor)
        t0 = time.time()
        c = copy.deepcopy(config)
        c.topology['k_stretch'] = float(config.topology['k_stretch']) * factor
        c.topology['k_shear'] = float(config.topology['k_shear']) * factor
        background = background_for_angle(sigma, args.angle_ref)
        sheet = sheet_state(c, sigma, background, args.steps, args)
        theta = theta_of(sheet)
        probe = probe_from_background(MOMENT_PER_DEGREE * args.angle_ref, args.probe_frac)
        row, parts, _ = linear_decay(sheet, probe, args.angle, args.n_soft)
        row.update({"stiffness_factor": factor, "theta": theta, "wall_s": time.time() - t0})
        gate(f"k x{factor:<5.0f}      ", theta, row, args)
        rows.append(row)
        jax.clear_caches()
    return rows


def arm_magnitude(config, sigma, args) -> list:
    """Nonlinear solves at growing probe, against the linear prediction at the same state.

    Two things are read per level. The decay of the SAME quantity the linear arm reports, which
    requires the soft modes to be projected out of the nonlinear response as well; and the distance
    between the nonlinear response and the linear prediction, which is what "the load stopped being
    a perturbation" actually means and needs no projection at all.
    """
    rows = []
    background = background_for_angle(sigma, args.angle_ref)
    sheet = sheet_state(config, sigma, background, args.steps, args)
    theta = theta_of(sheet)
    base = probe_from_background(MOMENT_PER_DEGREE * args.angle_ref, args.probe_frac)

    tan = Tangent(sheet)
    ref, _, dist = linear_decay(sheet, base, args.angle, args.n_soft, tangent=tan)
    print(f"  [linear reference] own r1 {ref['own']['ratios'][0]:.2f}  "
          f"lam {ref['own']['decay_length']:.2f}   grad {tan.grad_norm:.1e}", flush=True)

    for i, frac in enumerate(np.geomspace(args.mag_min, args.mag_max, int(args.n_levels))):
        t0 = time.time()
        mag = base * float(frac)
        probe = build_probe(sheet, float(args.angle), mag)
        loads = sheet.background_loads + load_specs(probe, sheet.faces)
        res, *_ = solve_at(sheet.config, None, loads, int(args.steps))
        delta = np.asarray(res['solution'].fields[-1]) - sheet.displacement

        # The linear prediction for the SAME probe, unfiltered, and the nonlinear answer beside it.
        f_free = tan.restrict(scatter_load(probe, sheet.faces, sheet.n_faces))
        lin_full = tan.scatter(np.linalg.solve(tan.K, f_free))
        departure = float(np.linalg.norm(delta - lin_full) / np.linalg.norm(lin_full))

        parts = sheet.probe_fields(tan.stiff_part(delta, args.n_soft))
        row = {"magnitude": mag, "probe_over_reference": float(frac), "theta": theta,
               "precision": parts["precision"], "departure": departure,
               "wall_s": time.time() - t0}
        for name in ("own", "redist", "total"):
            row[name] = decay_ratios(parts[name], dist).as_dict()
        rows.append(row)
        print(f"  probe x{frac:8.3g}  own r1 {row['own']['ratios'][0]:6.2f}  "
              f"lam {row['own']['decay_length']:5.2f}  departure {departure:8.2e}   "
              f"{row['wall_s']:5.1f}s", flush=True)
        jax.clear_caches()
    return {"reference": ref, "levels": rows}


# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="pocket_15")
    ap.add_argument("--arm", default="deployment",
                    choices=("deployment", "magnitude", "stiffness"))
    ap.add_argument("--n-levels", type=int, default=9)
    ap.add_argument("--angle-min", type=float, default=8.0, help="target hinge rotation, degrees")
    ap.add_argument("--angle-max", type=float, default=84.0)
    ap.add_argument("--angle-ref", type=float, default=45.0,
                    help="deployment held fixed by the magnitude and stiffness arms")
    ap.add_argument("--angle", type=float, default=0.0, help="probe direction, radians")
    ap.add_argument("--probe-frac", type=float, default=3.0,
                    help="probe magnitude as a multiple of the per-tile background moment")
    ap.add_argument("--k-min", type=float, default=0.3)
    ap.add_argument("--k-max", type=float, default=300.0)
    ap.add_argument("--mag-min", type=float, default=0.03)
    ap.add_argument("--mag-max", type=float, default=30.0)
    ap.add_argument("--pair-axis", default="x", choices=("x", "y"))
    ap.add_argument("--n-soft", type=int, default=2)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--maxiter", type=int, default=200000)
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--grad-gate", type=float, default=1e-6)
    ap.add_argument("--precision-gate", type=float, default=1e-6)
    ap.add_argument("--out", default="data/outputs/pocket/sweep")
    args = ap.parse_args()

    path = resolve_config(args.config)
    config = copy.deepcopy(load_and_parse_config(path))
    config.physics.solver_maxiter = int(args.maxiter)
    config.physics.solver_tol = float(args.tol)
    M, N = int(config.topology['M']), int(config.topology['N'])
    sigma = _checkerboard_signs(M, N).astype(float)

    os.makedirs(args.out, exist_ok=True)
    print(f"[sweep] {args.arm} on {M}x{N}, probe = {args.probe_frac}x the per-tile background "
          f"moment", flush=True)

    t0 = time.time()
    if args.arm == "deployment":
        result = {"rows": arm_deployment(config, sigma, args)}
    elif args.arm == "stiffness":
        result = {"rows": arm_stiffness(config, sigma, args)}
    else:
        result = arm_magnitude(config, sigma, args)

    out = {"arm": args.arm, "config": path, "M": M, "N": N, "args": vars(args), **result}
    dest = os.path.join(args.out, f"{args.arm}.json")
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    print(f"[sweep] wrote {dest}  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
