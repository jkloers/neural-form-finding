"""Pilot the PET solver settings before committing a 20-hour campaign (runs in the ccx env).

Answers the three questions that size and validate the campaign, on the standard hinge
(w_lig 18 mm, t 0.5 mm, alpha 90, r_win 100 mm):

1. HOW FAR can theta be driven? Every PET rotation run so far died at 43-58 deg against a 90 deg
   target at ligament PEEQ ~0.12 -- a buckling convergence death, not fracture -- while measured
   deployments ride to p99 70 deg. Axes: imperfection seed, min increment, viscous stabilisation,
   elements through thickness.
2. IS THE FAR-FIELD PLASTICITY REAL? Pinning uz = 0 on both handle arcs while the ligament bows
   ~20 mm out of plane forces a plastic kink at the truncation radius: 68% of the plastic work
   lands outside the ligament disc, 25% in the outermost ring alone. Releasing the driven arc out
   of plane (``arcB_uz_free``) tests whether that dissipation is an artifact of the window.
3. WHAT DOES IT COST? Wall-clock per job at one thread, which sets jobs/hour and hence the
   campaign size.

    conda run -n ccx --no-capture-output python -m nff.scripts.calibration.pilot_hinge_settings \
        --parallel 9 --angle 90 --steps 30 --out data/experiments/processed/pilot.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np

from nff.rve.ccx_solver import parse_job, prepare_job, solve_job
from nff.rve.damage import element_volumes, ligament_elements
from nff.rve.hinge_function import (DeploymentRay, HingeConstants, HingeGeometry,
                                    assemble_response, solver_kwargs, to_rve_params)
from nff.rve.materials import get_material


# Which imposed motion drives yielding in the far field? Pure rotation at 90 deg put 100% of the
# plastic work in the ligament; a replayed path carrying ~1 mm of in-plane translation put 68% of it
# in a ring on the driven arc. Rotation is nearly a rigid-body mode of the whole window, translation
# is not -- so these four motions at matched theta separate the two, and the uz pin is crossed with
# them to test whether the out-of-plane clamp is what converts translation into a plastic kink.
_MOTIONS = [("rot", dict(eta_a=0.0, eta_s=0.0)),
            ("axial", dict(eta_a=0.2, eta_s=0.0)),
            ("shear", dict(eta_a=0.0, eta_s=0.2)),
            ("combined", dict(eta_a=0.2, eta_s=0.1))]


def build_cells(angle, steps):
    """(name, const-overrides, ray-overrides) -- one CalculiX job each."""
    cells = []
    for motion, ray_over in _MOTIONS:
        for uz_free in (False, True):
            cells.append((f"{motion}_{'free' if uz_free else 'pinned'}",
                          dict(imp_amp=0.15, arcB_uz_free=uz_free), ray_over))
    return cells


def run_cell(name, meta, timeout, ncpus, geo, const, ray):
    t0 = time.time()
    try:
        stdout = solve_job(meta, ncpus=ncpus, timeout=timeout, eps_f=None).stdout
    except Exception as exc:                                   # keep the grid alive
        stdout = f"[solve raised] {exc}"
    wall = time.time() - t0
    try:
        parsed = parse_job(meta, stdout)
        res = assemble_response(geo, ray, const, parsed)
    except Exception as exc:
        return dict(name=name, wall_s=wall, error=f"parse: {exc}", n_samples=0)

    frd = meta["job"] + ".frd"
    row = dict(name=name, wall_s=round(wall, 1), ok=bool(parsed["ok"]),
               n_samples=int(res.n_samples), n_elems=int(parsed["n_elems"]),
               frd_MB=round(os.path.getsize(frd) / 1e6, 1) if os.path.exists(frd) else 0.0)
    if res.n_samples == 0:
        return row
    xyz, conn = parsed["xyz"], parsed["conn"]
    lig = ligament_elements(xyz, conn, geo.w_lig)
    vol = element_volumes(xyz, conn)
    frames = parsed["frames"]
    from nff.rve.damage import _element_mean, _nodal_peeq                     # noqa: PLC0415
    nodal = _nodal_peeq(frames[-1]) if frames else None
    if nodal is not None:
        pe = _element_mean(nodal, conn)
        wk = vol * pe
        row["lig_work_frac"] = round(float(wk[lig].sum() / max(wk.sum(), 1e-30)), 4)
    row.update(
        theta_max_deg=round(float(np.nanmax(res.theta_deg)), 2),
        W_end=round(float(res.W[-1]), 2), W_max=round(float(np.nanmax(res.W)), 2),
        M_max=round(float(np.nanmax(np.abs(res.M_theta))), 2),
        uz_max=round(float(np.nanmax(res.uz_max)), 3),
        damage_max=round(float(np.nanmax(res.damage)), 6),
        peeq_lig_max=round(float(np.nanmax(res.peeq_lig)), 4),
    )
    # W and damage at a common rotation, so cells are comparable despite different ceilings
    for probe in (20.0, 40.0):
        if row["theta_max_deg"] >= probe:
            i = int(np.searchsorted(res.theta_deg, probe))
            i = min(i, res.n_samples - 1)
            row[f"W_at{int(probe)}"] = round(float(res.W[i]), 2)
            row[f"D_at{int(probe)}"] = round(float(res.damage[i]), 6)
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--angle", type=float, default=90.0, help="rotation target [deg]")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--w-lig", type=float, default=18.0)
    ap.add_argument("--thickness", type=float, default=0.5)
    ap.add_argument("--alpha", type=float, default=90.0)
    ap.add_argument("--r-win", type=float, default=100.0)
    ap.add_argument("--parallel", type=int, default=9)
    ap.add_argument("--ncpus", type=int, default=1, help="threads per ccx process")
    ap.add_argument("--timeout", type=float, default=2400)
    ap.add_argument("--out", default="data/experiments/processed/pilot.json")
    args = ap.parse_args()

    base = HingeConstants(thickness=args.thickness, w_c=0.2, r_win=args.r_win, fillet_ratio=0.16,
                          material=get_material("pet"), n_through=2, stop_at_fracture=False,
                          el_fields="PEEQ, S")      # E is written+parsed but never reaches a column
    geo = HingeGeometry(w_lig=args.w_lig, alpha_deg=args.alpha, fillet_ratio=0.16)
    cells = build_cells(args.angle, args.steps)
    print(f"pilot: {len(cells)} cells, {args.parallel} parallel x {args.ncpus} thread, "
          f"timeout {args.timeout:g}s")
    print(f"  hinge w_lig {geo.w_lig} mm  t {base.thickness} mm  alpha {geo.alpha_deg}  "
          f"r_win {base.r_win} mm  PET E={base.material.params['E']:.0f} MPa")
    print(f"  ray theta -> {args.angle} deg in {args.steps} steps\n")

    # phase 1: mesh + write decks SERIALLY (gmsh is not thread-safe)
    prepared = []
    for name, c_over, r_over in cells:
        const = replace(base, **c_over)
        ray = DeploymentRay(theta1_deg=args.angle, n_steps=args.steps, tag=name, **r_over)
        kw = solver_kwargs(geo, ray, const)
        meta = prepare_job(to_rve_params(geo, const),
                           workdir=f"/tmp/hinge_pilot/{name}", **kw)
        prepared.append((name, meta, geo, const, ray))
        print(f"  meshed {name:22s} {len(meta['conn'])} elems")

    # phase 2: solve in parallel
    print(f"\nsolving {len(prepared)} cells ...")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        rows = list(pool.map(
            lambda p: run_cell(p[0], p[1], args.timeout, args.ncpus, p[2], p[3], p[4]),
            prepared))
    total = time.time() - t0

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(dict(cells=rows, angle=args.angle, steps=args.steps, w_lig=args.w_lig,
                   thickness=args.thickness, r_win=args.r_win, parallel=args.parallel,
                   ncpus=args.ncpus, total_wall_s=round(total, 1)),
              open(args.out, "w"), indent=2)

    hdr = (f"{'cell':24s} {'ok':>3s} {'theta':>7s} {'incs':>5s} {'wall':>7s} {'W_end':>8s} "
           f"{'W@20':>8s} {'M_max':>7s} {'uz':>7s} {'D_max':>9s} {'ligwork':>8s}")
    print("\n" + hdr + "\n" + "-" * len(hdr))
    for r in sorted(rows, key=lambda r: -r.get("theta_max_deg", -1)):
        print(f"{r['name']:24s} {str(r.get('ok', ''))[:3]:>3s} "
              f"{r.get('theta_max_deg', float('nan')):7.2f} {r.get('n_samples', 0):5d} "
              f"{r.get('wall_s', 0):7.0f} {r.get('W_end', float('nan')):8.1f} "
              f"{r.get('W_at20', float('nan')):8.1f} {r.get('M_max', float('nan')):7.1f} "
              f"{r.get('uz_max', float('nan')):7.2f} {r.get('damage_max', float('nan')):9.5f} "
              f"{r.get('lig_work_frac', float('nan')):8.3f}")
    good = [r for r in rows if r.get("n_samples", 0) > 0]
    print(f"\ntotal wall {total/60:.1f} min for {len(cells)} cells at {args.parallel} parallel "
          f"-> {3600*len(cells)/max(total, 1):.0f} jobs/hour")
    if good:
        print(f"best theta reached: {max(r['theta_max_deg'] for r in good):.1f} deg")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
