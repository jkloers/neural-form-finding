"""Quantify the history a state-function surrogate cannot carry when a hinge opens then closes.

The oracle labels a hinge by an accumulated quantity. With ``PEEQ(x, t) = int_0^t sqrt(2/3)
||eps_p_dot(x, tau)|| d tau``, the damage measure of :mod:`nff.rve.damage` is

    Delta(t) = ( 1 / (V_lig * eps_f) ) * int_Omega PEEQ(x, t) dV

which is non-decreasing along ANY route and is constant on elastic segments. The surrogate is a
state function ``D(u)`` of ``u = (a, s, theta)`` alone. When a route visits the same state twice,
at ``t1 < t2``, the two labels differ by

    G = Delta(t2) - Delta(t1) >= 0

and no single-valued ``D`` can represent both. Because the campaign
(:func:`nff.rve.dataset.sample_jobs`) draws proportional rays only, the fitted ``D`` reproduces the
FIRST visit, so the whole of ``G`` is missing at the second: the surrogate under-predicts damage on
a returning route, never over-predicts.

Two modes, answering the two halves of that statement.

``corpus``
    Exact, no FEA. How far do the harvested deployment paths actually walk back? Reports the
    retrace depth

        R = max_{i <= j} [ max_{i <= k <= j} d(u_k, u_i) - d(u_j, u_i) ]

    the largest amount of travel a path undoes, in the ligament-tip metric
    ``d = ||(da, ds, w_lig*dtheta)|| / w_lig`` used by
    :func:`nff.scripts.surrogate.replay_hinge_path.resample_polyline`, so rotation and
    translation are commensurate and R is a pure number in ligament widths. ``R = 0`` exactly when
    the path never moves back toward a state it has passed.

``replay``
    One CalculiX solve that drives a proportional ray out to ``u_max`` and back down the same ray.
    Every fraction ``lambda`` of the ray is then visited twice on the SAME mesh in the SAME solve,
    so ``G(lambda)`` is read off a self-paired comparison with no cross-solve noise.

    conda run -n ccx --no-capture-output python -m nff.scripts.surrogate.audit_return_memory \
        replay --theta-max 40 --out data/surrogates/return_memory_w18_th40.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np


# -- corpus mode ---------------------------------------------------------------------------------

def retrace_depth(U: np.ndarray) -> np.ndarray:
    """Largest travel each path undoes, in ligament widths.

    Args:
        U: (n_paths, n_points, 3) states ``(eta_a, eta_s, theta[rad])``. Dividing the millimetre
           translations by ``w_lig`` has already made the three components commensurate.

    Returns:
        (n_paths,) retrace depth; 0 for a path that never moves back toward an earlier state.
    """
    D = np.linalg.norm(U[:, :, None, :] - U[:, None, :, :], axis=-1)
    k = np.arange(U.shape[1])
    before = k[None, :, None] > k[None, None, :]                 # column index below the reference
    running_max = np.maximum.accumulate(np.where(before, -np.inf, D), axis=2)
    return np.where(before, 0.0, running_max - D).reshape(len(U), -1).max(axis=1)


def audit_corpus(prior_dir: str) -> dict:
    """Retrace statistics over every harvested hinge path of one path prior."""
    npz = np.load(os.path.join(prior_dir, "paths.npz"))
    eta = npz["eta"]                                             # (E, T+1, H, 3)
    if "hinge_mask" not in npz.files:
        raise SystemExit(f"{prior_dir} carries no hinge_mask; its non-hinge slots hold diverged "
                         "ROM rows and would dominate every percentile")
    n_ex, n_pts, n_hinge, _ = eta.shape
    U = np.transpose(eta, (0, 2, 1, 3)).reshape(n_ex * n_hinge, n_pts, 3)[npz["hinge_mask"].ravel()]
    R = retrace_depth(U)
    theta_deg = np.degrees(U[:, :, 2])
    drawdown = (np.maximum.accumulate(theta_deg, axis=1) - theta_deg).max(axis=1)
    return dict(
        prior=os.path.basename(prior_dir.rstrip("/")), n_paths=int(len(U)), n_points=int(n_pts),
        retrace_ligwidths={f"p{q}": float(np.percentile(R, q)) for q in (50, 75, 90, 95, 99, 99.9, 100)},
        frac_retrace_above={str(t): float((R > t).mean()) for t in (0.012, 0.05, 0.118, 0.25, 0.5)},
        n_retrace_above_0118=int((R > 0.118).sum()),
        theta_drawdown_deg={f"p{q}": float(np.percentile(drawdown, q)) for q in (50, 90, 95, 99, 99.9, 100)},
        frac_theta_drawdown_above={f"{t}deg": float((drawdown > t).mean()) for t in (0.5, 2.0, 5.0, 10.0)},
    )


# -- replay mode ---------------------------------------------------------------------------------

def build_out_and_back_states(u_max: np.ndarray, n_out: int, lam_end: float) -> np.ndarray:
    """States of a proportional ray out to ``u_max`` and back down to ``lam_end * u_max``."""
    out = np.linspace(1.0 / n_out, 1.0, n_out)
    back = np.linspace(1.0, lam_end, n_out + 1)[1:]
    return np.outer(np.r_[out, back], u_max)


def memory_gap(lam: np.ndarray, damage: np.ndarray, peak: int) -> list[dict]:
    """Pair each returning increment with the outbound state of equal ray fraction.

    The outbound leg is resampled onto the returning fractions, so the pairing is exact in the ray
    coordinate rather than in the increment index (CalculiX subdivides steps unevenly). The
    undeformed state is prepended as a known outbound sample: the solve starts there and Delta is
    exactly 0, which ``np.interp`` would otherwise replace by the lowest solved fraction.
    """
    out_lam = np.r_[0.0, lam[:peak + 1]]
    out_dmg = np.r_[0.0, damage[:peak + 1]]
    order = np.argsort(out_lam)
    rows = []
    for i in range(peak + 1, len(lam)):
        d_out = float(np.interp(lam[i], out_lam[order], out_dmg[order]))
        rows.append(dict(lam=float(lam[i]), excursion=float(lam[peak] - lam[i]),
                         damage_out=d_out, damage_return=float(damage[i]),
                         gap=float(damage[i] - d_out),
                         ratio=float(damage[i] / d_out) if d_out > 0 else float("inf")))
    return rows


def run_replay(args) -> dict:
    """Solve one out-and-back ray and report ``G`` against the excursion amplitude."""
    from nff.rve.ccx_solver import deploy
    from nff.rve.hinge_function import (DeploymentPath, HingeConstants, HingeGeometry,
                                        assemble_response, solver_kwargs, to_rve_params)
    from nff.rve.materials import get_material

    u_max = np.array([args.eta_a * args.w_lig, args.eta_s * args.w_lig, np.radians(args.theta_max)])
    states = build_out_and_back_states(u_max, args.n_out, args.lam_end)
    const = HingeConstants(thickness=args.thickness, w_c=args.kerf, r_win=args.r_win,
                           fillet_ratio=args.fillet, material=get_material("pet"),
                           n_through=args.n_through, imp_amp=args.imp_amp, min_inc=args.min_inc,
                           stabilize=None, stop_at_fracture=False)
    geo = HingeGeometry(w_lig=args.w_lig, alpha_deg=args.alpha, fillet_ratio=args.fillet)
    path = DeploymentPath(polyline=states, tag=f"oab_w{args.w_lig:g}_th{args.theta_max:g}")

    print(f"out and back  w_lig {args.w_lig} mm  alpha {args.alpha} deg  t {args.thickness} mm  "
          f"r_win {args.r_win} mm  PET")
    print(f"  peak state a {u_max[0]:+.3f} mm  s {u_max[1]:+.3f} mm  theta {args.theta_max:.1f} deg")
    print(f"  {len(states)} steps: {args.n_out} out, {args.n_out} back to lambda {args.lam_end}")
    sys.stdout.flush()

    t0 = time.time()
    parsed = deploy(to_rve_params(geo, const), timeout=args.timeout, eps_f=None, ncpus=args.ncpus,
                    workdir=args.workdir or f"/tmp/hinge_oab/{path.tag}_r{args.r_win:g}",
                    **solver_kwargs(geo, path, const))
    res = assemble_response(geo, path, const, parsed)
    wall = time.time() - t0

    lam = res.theta_deg / args.theta_max
    peak = int(np.argmax(res.theta_deg))
    print(f"\nsolve {wall:.0f} s  ok={parsed['ok']}  {res.n_samples} increments of {len(states)} "
          f"steps  peak at increment {peak}\n")
    print("   #    lam   theta     damage   peeq_lig        W      uz  leg")
    for i in range(res.n_samples):
        print(f"{i:4d} {lam[i]:6.3f} {res.theta_deg[i]:7.2f} {res.damage[i]:10.6f} "
              f"{res.peeq_lig[i]:10.4f} {res.W[i]:8.2f} {res.uz_max[i]:7.2f}  "
              f"{'out' if i <= peak else 'back'}")

    rows = memory_gap(lam, res.damage, peak)
    if rows:
        print("\n  lambda  excursion   damage_out  damage_return        gap    ratio")
        for r in rows:
            print(f"  {r['lam']:6.3f}  {r['excursion']:9.3f}  {r['damage_out']:11.6f}  "
                  f"{r['damage_return']:13.6f}  {r['gap']:9.6f}  {r['ratio']:7.3f}")
    return dict(w_lig=args.w_lig, alpha_deg=args.alpha, thickness=args.thickness,
                r_win=args.r_win, n_through=args.n_through, theta_max=args.theta_max,
                eta_a=args.eta_a, eta_s=args.eta_s, lam_end=args.lam_end, eps_f=const.eps_f,
                n_elems=int(parsed["n_elems"]), ok=bool(parsed["ok"]), wall_s=wall,
                peak_index=peak, lam=lam.tolist(), theta_deg=res.theta_deg.tolist(),
                damage=res.damage.tolist(), peeq_lig=res.peeq_lig.tolist(), W=res.W.tolist(),
                uz_max=res.uz_max.tolist(), pairs=rows)


# -- CLI -----------------------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    c = sub.add_parser("corpus", help="retrace statistics of the harvested paths (no FEA)")
    c.add_argument("priors", nargs="+", help="path-prior directories")
    c.add_argument("--out", default=None)

    r = sub.add_parser("replay", help="one out-and-back CalculiX solve (needs the ccx env)")
    r.add_argument("--w-lig", type=float, default=18.0)
    r.add_argument("--thickness", type=float, default=0.5)
    r.add_argument("--alpha", type=float, default=90.0)
    r.add_argument("--theta-max", type=float, default=40.0, help="peak fold [deg]")
    r.add_argument("--eta-a", type=float, default=0.0)
    r.add_argument("--eta-s", type=float, default=0.0)
    r.add_argument("--n-out", type=int, default=15, help="steps on each leg")
    r.add_argument("--lam-end", type=float, default=0.0, help="ray fraction to return to")
    r.add_argument("--r-win", type=float, default=100.0)
    r.add_argument("--n-through", type=int, default=2)
    r.add_argument("--fillet", type=float, default=0.16)
    r.add_argument("--kerf", type=float, default=0.2)
    r.add_argument("--imp-amp", type=float, default=None)
    r.add_argument("--min-inc", type=float, default=1e-4)
    r.add_argument("--ncpus", type=int, default=1)
    r.add_argument("--timeout", type=float, default=7200)
    r.add_argument("--workdir", default=None)
    r.add_argument("--out", required=True)

    args = ap.parse_args()
    result = ([audit_corpus(p) for p in args.priors] if args.mode == "corpus" else run_replay(args))
    if args.mode == "corpus":
        print(json.dumps(result, indent=1))
    if args.out:
        json.dump(result, open(args.out, "w"), indent=1)
        print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
