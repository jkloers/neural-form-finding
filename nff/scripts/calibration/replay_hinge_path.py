"""Replay one harvested hinge displacement path through the CalculiX oracle (runs in the ccx env).

Picks a measured polyline out of a path-prior harvest and drives the single-hinge RVE along it
state by state, instead of along a proportional ray to its endpoint. The oracle is elastoplastic,
so ``W`` is a path-dependent work: the route is part of the sample, not just the destination.

Saves everything the campaign would record, plus the deformed mesh and per-element PEEQ of the
final frame so the deployment can be rendered and the damaged zone highlighted.

    conda run -n ccx --no-capture-output python -m nff.scripts.calibration.replay_hinge_path \
        --prior data/fea/path_priors/sheet_4x8ft_1tile_v2_n3600 \
        --w-lig 18 --seed 0 --out data/experiments/processed/replay_sample.npz
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from nff.rve.ccx_solver import deploy
from nff.rve.damage import (LIG_CENTER_FRAC, LIG_RADIUS_FRAC, _element_mean, _nodal_peeq,
                            element_volumes, ligament_elements)
from nff.rve.hinge_function import (DeploymentPath, HingeConstants, HingeGeometry,
                                    assemble_response, descriptor, solver_kwargs, to_rve_params)
from nff.rve.materials import get_material

REGIME_NAME = {0: "elastic", 1: "plastic", 2: "FAILED"}


def resample_polyline(poly: np.ndarray, n: int, w_lig: float) -> np.ndarray:
    """Re-space a polyline at uniform arc length, preserving its SHAPE exactly.

    The harvest stores one point per ROM load step, and the ROM's first load step already spends
    57% of the rotation (its k_rot is the buckled secant, so rotation is nearly free to it). Handing
    those points to CalculiX one-per-*STEP would put more than half the deformation in a single
    step. Plasticity is rate-independent -- only the ORDER of states matters -- so re-spacing along
    the path is free of physical consequence and gives the solver an even walk.

    Arc length uses the ligament-tip metric ``sqrt(da^2 + ds^2 + (w_lig*dtheta)^2)``, which puts the
    rotation on the same footing as the translations instead of comparing mm with radians.
    """
    q = np.asarray(poly, float)
    step = np.diff(q, axis=0) * np.array([1.0, 1.0, w_lig])
    seg = np.r_[0.0, np.cumsum(np.linalg.norm(step, axis=1))]
    if seg[-1] <= 0:
        return q
    tgt = np.linspace(0.0, seg[-1], n + 1)
    return np.stack([np.interp(tgt, seg, q[:, j]) for j in range(3)], axis=1)


def pick_path(prior_dir: str, seed: int):
    """One random measured polyline -> (a[mm], s[mm], theta[rad]) per point, plus its provenance.

    The harvest stores ``eta = (a, s)/w_lig_harvest``; the transferable quantity is the PHYSICAL
    displacement, so it is de-normalised here by the harvest's own ``w_lig_mm`` and re-normalised
    (if at all) by the geometry actually being simulated. Sampling eta directly would silently
    rescale every path whenever the ligament-width standard moves.
    """
    npz = np.load(os.path.join(prior_dir, "paths.npz"))
    A = {k: npz[k] for k in npz.files}                 # materialise once (npz re-decompresses)
    manifest = json.load(open(os.path.join(prior_dir, "manifest.json")))
    w_lig_harvest = float(manifest["w_lig_mm"])

    eta, alpha = A["eta"], A["alpha"]                  # (E, T+1, H, 3), (E, H)
    mask = A["hinge_mask"] if "hinge_mask" in A else np.ones(alpha.shape, bool)
    ex, hi = np.nonzero(mask)
    rng = np.random.default_rng(seed)
    k = int(rng.integers(len(ex)))
    e, h = int(ex[k]), int(hi[k])

    poly = eta[e, :, h, :].astype(float).copy()        # (T+1, 3), row 0 = the origin
    poly[:, 0] *= w_lig_harvest                        # eta_a -> a [mm]
    poly[:, 1] *= w_lig_harvest                        # eta_s -> s [mm]
    return poly, float(np.degrees(alpha[e, h])), dict(
        prior=os.path.basename(prior_dir.rstrip("/")), example=e, hinge=h,
        w_lig_harvest_mm=w_lig_harvest, seed=seed,
        clamped_face=int(A["clamped_face"][e]) if "clamped_face" in A else -1,
        loaded_face=int(A["loaded_face"][e]) if "loaded_face" in A else -1,
        load_value_N=float(A["load_value"][e]) if "load_value" in A else float("nan"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prior", default="data/fea/path_priors/sheet_4x8ft_1tile_v2_n3600")
    ap.add_argument("--seed", type=int, default=0, help="which measured path to draw")
    ap.add_argument("--w-lig", type=float, default=18.0, help="ligament width to simulate [mm]")
    ap.add_argument("--thickness", type=float, default=0.5, help="PET gauge [mm]")
    ap.add_argument("--alpha", type=float, default=None, help="override the path's own alpha [deg]")
    ap.add_argument("--fillet", type=float, default=0.16)
    ap.add_argument("--kerf", type=float, default=0.2)
    ap.add_argument("--r-win", type=float, default=100.0, help="Saint-Venant window radius [mm]")
    ap.add_argument("--n-through", type=int, default=2)
    ap.add_argument("--imp-amp", type=float, default=None, help="buckle seed [mm]; None -> 0.3*t")
    ap.add_argument("--min-inc", type=float, default=1e-3)
    ap.add_argument("--stabilize", type=float, default=None)
    ap.add_argument("--ncpus", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--resample", type=int, default=30,
                    help="re-space the polyline at uniform arc length into N steps; 0 = verbatim")
    ap.add_argument("--ray", action="store_true",
                    help="drive a straight proportional ray to the SAME endpoint (the A/B control)")
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    poly, alpha_deg, prov = pick_path(args.prior, args.seed)
    alpha_deg = args.alpha if args.alpha is not None else alpha_deg
    if args.resample:
        poly = resample_polyline(poly, args.resample, args.w_lig)
    states = poly[1:]                                  # the origin is implicit at t = 0
    if args.ray:                                       # same endpoint, straight route
        n = len(states)
        states = np.outer(np.arange(1, n + 1) / n, states[-1])

    const = HingeConstants(
        thickness=args.thickness, w_c=args.kerf, r_win=args.r_win, fillet_ratio=args.fillet,
        material=get_material("pet"), n_through=args.n_through, imp_amp=args.imp_amp,
        min_inc=args.min_inc, stabilize=args.stabilize, stop_at_fracture=False)
    geo = HingeGeometry(w_lig=args.w_lig, alpha_deg=alpha_deg, fillet_ratio=args.fillet)
    path = DeploymentPath(polyline=states, tag=f"e{prov['example']}h{prov['hinge']}")

    a1, s1, th1 = path.targets(geo)
    print(f"path {prov['prior']} example {prov['example']} hinge {prov['hinge']}  "
          f"({'STRAIGHT RAY' if args.ray else 'measured polyline'}, {len(states)} states"
          f"{', arc-length resampled' if args.resample and not args.ray else ''})")
    dth = np.degrees(np.diff(np.r_[0.0, states[:, 2]]))
    print(f"  theta per step [deg]: max {dth.max():.2f}  min {dth.min():.2f}  "
          f"first {dth[0]:.2f} ({100*dth[0]/max(dth.sum(), 1e-9):.0f}% of the total)")
    print(f"  endpoint a={a1:+.3f} mm  s={s1:+.3f} mm  theta={th1:.2f} deg   "
          f"peak theta {path.theta1_deg:.2f} deg")
    print(f"  geometry w_lig={geo.w_lig} mm  alpha={geo.alpha_deg:.1f} deg  t={const.thickness} mm  "
          f"r_win={const.r_win} mm  eta_a={a1/geo.w_lig:+.3f}  eta_s={s1/geo.w_lig:+.3f}")
    mat = const.material
    print(f"  material {mat.name}  E={mat.params['E']:.0f} MPa  eps_f={const.eps_f}")

    kw = solver_kwargs(geo, path, const)
    t0 = time.time()
    parsed = deploy(to_rve_params(geo, const), timeout=args.timeout,
                    eps_f=None, ncpus=args.ncpus,
                    workdir=args.workdir or f"/tmp/hinge_replay/{path.tag}", **kw)
    wall = time.time() - t0
    res = assemble_response(geo, path, const, parsed)

    xyz, conn = parsed["xyz"], parsed["conn"]
    lig = ligament_elements(xyz, conn, geo.w_lig)
    vol = element_volumes(xyz, conn)
    print(f"\nmesh   {parsed['n_nodes']} nodes  {parsed['n_elems']} C3D15  "
          f"ligament disc {int(lig.sum())} elems ({100*lig.mean():.1f}%)  "
          f"volume {vol.sum():.1f} mm^3")
    print(f"       through-thickness {const.thickness/args.n_through:.3f} mm/elem  "
          f"lc_min {kw['lc_min']:.3f} mm  lc_max {kw['lc_max']:.3f} mm  "
          f"in-plane/through aspect {kw['lc_min']/(const.thickness/args.n_through):.1f}")
    print(f"solve  {wall:.1f} s wall  ok={parsed['ok']}  {res.n_samples} increments "
          f"({len(states)} steps requested)")

    print("\n  #    theta      a        s   |        W      F_a      F_s    M_th |"
          "   damage  peeq_lig   <eta>    uz_max  regime")
    idx = np.unique(np.linspace(0, res.n_samples - 1, min(res.n_samples, 22)).astype(int))
    for i in idx:
        print(f"{i:4d} {res.theta_deg[i]:8.2f} {res.a[i]:+8.3f} {res.s[i]:+8.3f} | "
              f"{res.W[i]:8.2f} {res.F_a[i]:+8.2f} {res.F_s[i]:+8.2f} {res.M_theta[i]:+8.1f} | "
              f"{res.damage[i]:8.5f} {res.peeq_lig[i]:9.4f} {res.eta_mean_lig[i]:+7.3f} "
              f"{res.uz_max[i]:9.3f}  {REGIME_NAME[int(res.regime[i])]}")

    print(f"\npeak   W {np.nanmax(res.W):.1f} N.mm | |M| {np.nanmax(np.abs(res.M_theta)):.1f} N.mm | "
          f"damage {np.nanmax(res.damage):.5f} | peeq_lig {np.nanmax(res.peeq_lig):.4f} "
          f"({100*np.nanmax(res.peeq_lig)/const.eps_f:.1f}% of eps_f) | uz {np.nanmax(res.uz_max):.2f} mm "
          f"({np.nanmax(res.uz_max)/const.thickness:.0f} t)")
    print(f"       plastic-weighted <eta> {np.nanmean(res.eta_mean_lig):+.3f} | "
          f"theta reached {np.nanmax(res.theta_deg):.2f} of {path.theta1_deg:.2f} deg requested"
          f"{'  <-- TRUNCATED' if np.nanmax(res.theta_deg) < 0.98*path.theta1_deg else ''}")

    # final frame: deformed coordinates + per-element PEEQ, for the 3D render
    frames = parsed["frames"]
    last = frames[-1] if frames else {}
    disp = last.get("DISP", np.zeros_like(xyz))
    nodal = _nodal_peeq(last)
    elem_peeq = _element_mean(nodal, conn) if nodal is not None else np.full(len(conn), np.nan)

    np.savez(
        args.out,
        theta_deg=res.theta_deg, a=res.a, s=res.s, theta=res.theta,
        W=res.W, F_a=res.F_a, F_s=res.F_s, M_theta=res.M_theta,
        damage=res.damage, peeq_lig=res.peeq_lig, eta_mean_lig=res.eta_mean_lig,
        uz_max=res.uz_max, regime=res.regime,
        states=states, polyline_full=poly, is_ray=args.ray,
        xyz=xyz, conn=np.asarray(conn), disp=disp, elem_peeq=elem_peeq,
        lig_mask=lig, elem_vol=vol, n_elems=parsed["n_elems"], n_nodes=parsed["n_nodes"],
        wall_s=wall, ok=parsed["ok"], eps_f=const.eps_f,
        lig_center_frac=LIG_CENTER_FRAC, lig_radius_frac=LIG_RADIUS_FRAC,
        provenance=json.dumps(prov), desc=json.dumps(descriptor(geo, const)),
        **{k: v for k, v in dict(w_lig=geo.w_lig, alpha_deg=geo.alpha_deg,
                                 fillet_ratio=geo.fillet_ratio, thickness=const.thickness,
                                 r_win=const.r_win, n_through=const.n_through).items()})
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
