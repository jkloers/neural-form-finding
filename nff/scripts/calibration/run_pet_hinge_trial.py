"""Run one PET-calibrated CalculiX hinge fold and save the response (runs in the ccx env).

Drives the single-hinge RVE with the calibrated ``PETIsotropic`` law along a pure-rotation
deployment ray, to compare against a physical hinge-fold experiment. Saves an .npz of the
moment-angle path + failure info; plot separately (matplotlib lives in kgnn_mac).

    conda run -n ccx --no-capture-output python -m nff.scripts.calibration.run_pet_hinge_trial \
        --w-lig 5.0 --theta 90 --steps 18 --out data/experiments/processed/pet_hinge_w5.npz
"""
from __future__ import annotations

import argparse

import numpy as np

from nff.rve.hinge_function import HingeConstants, DeploymentRay, HingeGeometry, evaluate_hinge
from nff.rve.materials import get_material


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--w-lig", type=float, default=5.0, help="ligament width [mm]")
    ap.add_argument("--thickness", type=float, default=0.5, help="PET gauge [mm]")
    ap.add_argument("--alpha", type=float, default=90.0)
    ap.add_argument("--fillet", type=float, default=0.16)
    ap.add_argument("--kerf", type=float, default=0.2)
    ap.add_argument("--eps-f", type=float, default=None,
                    help="fracture strain override; default = the material's own (PET: 1.784, measured)")
    ap.add_argument("--r-win", type=float, default=None,
                    help="Saint-Venant window radius [mm]; default = max(12, 2.4*w_lig)")
    ap.add_argument("--theta", type=float, default=90.0, help="rotation ramp [deg] (keep small for a shear/axial trial)")
    ap.add_argument("--eta-s", type=float, default=0.0, help="shear ratio: s1 = eta_s * w_lig (Instron shear mode)")
    ap.add_argument("--eta-a", type=float, default=0.0, help="axial ratio: a1 = eta_a * w_lig (Instron opening mode)")
    ap.add_argument("--steps", type=int, default=18)
    ap.add_argument("--ncpus", type=int, default=1, help="threads for the ccx equation solver")
    ap.add_argument("--n-through", type=int, default=2, help="elements through the thickness (mesh convergence)")
    ap.add_argument("--imp-amp", type=float, default=None, help="out-of-plane buckle seed [mm]; None -> 0.3*t")
    ap.add_argument("--min-inc", type=float, default=1e-3, help="min load increment (smaller babies through the buckling snap)")
    ap.add_argument("--stabilize", type=float, default=None, help="*STATIC,STABILIZE damping to walk past buckling (uz-only; contaminates F/M)")
    ap.add_argument("--workdir", default=None, help="ccx scratch dir (set distinct dirs for parallel runs sharing geometry)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    r_win = args.r_win if args.r_win is not None else max(12.0, 2.4 * args.w_lig)
    const = HingeConstants(
        thickness=args.thickness, w_c=args.kerf, r_win=r_win,
        fillet_ratio=args.fillet, material=get_material("pet"), eps_f=args.eps_f, n_through=args.n_through,
        imp_amp=args.imp_amp, min_inc=args.min_inc, stabilize=args.stabilize,
    )
    geo = HingeGeometry(w_lig=args.w_lig, alpha_deg=args.alpha, fillet_ratio=args.fillet)
    ray = DeploymentRay(theta1_deg=args.theta, eta_a=args.eta_a, eta_s=args.eta_s, n_steps=args.steps)

    mode = "shear" if args.eta_s else ("axial" if args.eta_a else "rotation")
    print(f"PET hinge {mode}: w_lig={args.w_lig}mm t={args.thickness}mm alpha={args.alpha} "
          f"-> theta={args.theta}deg eta_s={args.eta_s} eta_a={args.eta_a} ({args.steps} steps), eps_f={const.eps_f}")
    res = evaluate_hinge(geo, ray, const, ncpus=args.ncpus, workdir=args.workdir)

    np.savez(
        args.out,
        theta_deg=res.theta_deg, a=res.a, s=res.s,
        M_theta=res.M_theta, F_a=res.F_a, F_s=res.F_s, W=res.W,
        damage=res.damage, peeq_lig=res.peeq_lig, eta_mean_lig=res.eta_mean_lig, uz_max=res.uz_max,
        damage_at_tear=res.damage_at_tear,
        regime=res.regime, failure_theta_deg=res.failure_theta_deg,
        w_lig=args.w_lig, thickness=args.thickness, alpha=args.alpha, eps_f=const.eps_f,
        eta_s=args.eta_s, eta_a=args.eta_a, mode=mode, n_elems=res.n_elems,
    )
    ft = res.failure_theta_deg
    print(f"ok={res.ok}  n_elems={res.n_elems}  n_samples={res.n_samples}")
    print(f"peak M_theta = {np.nanmax(res.M_theta):.1f} N.mm at theta={res.theta_deg[int(np.nanargmax(res.M_theta))]:.1f} deg")
    print(f"max damage = {np.nanmax(res.damage):.4f}  peak ligament PEEQ = {np.nanmax(res.peeq_lig):.3f}"
          f"  <eta> = {np.nanmean(res.eta_mean_lig):.3f}  max uz = {np.nanmax(res.uz_max):.3f} mm")
    print(f"failure angle = {'—' if np.isnan(ft) else f'{ft:.1f} deg'}  (survives fold)" if np.isnan(ft)
          else f"failure angle = {ft:.1f} deg")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
