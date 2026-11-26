"""Is the hinge oracle REPRODUCIBLE, or does it pick a buckling branch at random?

The surrogate plateaus at ~21% held-out energy error while every model-side lever is inert, and on
clean converged data 19.7% of near-identical input pairs disagree on W by >30% -- a third of them
with a different ``uz_max``, i.e. a different post-buckling branch. That scatter is the floor: no
architecture beats the conditional mean of a multi-valued map.

Which of two worlds we are in decides everything downstream, and they differ by orders of magnitude
in cost:

  reproducible  -> the scatter is REAL variation between nearby-but-distinct geometries, so only
                   denser sampling helps (spacing scales as N^-0.29 here => ~47x more data for 5%).
  branch-flippy -> the solve picks a mode from numerical noise, so more samples just re-flip the
                   coin; the fix is to make the bifurcation deterministic, which collapses the floor
                   at the density we already have.

CalculiX is deterministic -- an identical deck gives an identical answer -- so this perturbs things
that are physically IMMATERIAL and asks whether W moves:

  * imperfection amplitude +/-3%   (an arbitrary seed must not matter)
  * mesh characteristic length -5% (re-meshes the same geometry; tests whether the DISCRETIZATION
    picks the branch)
  * an amplitude sweep 0.1 / 0.3 / 1.0 x thickness -- ⚠ if W depends strongly on this, the whole
    campaign's energies are partly an artifact of the arbitrary ``0.3*t`` default.

    CCX_BIN=/opt/miniconda3/envs/ccx/bin/ccx conda run -n kgnn_mac \
        python nff/scripts/diagnostics/diagnose_oracle_reproducibility.py --n-geoms 4 --parallel 4
"""
import argparse
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from nff.rve.ccx_solver import prepare_job, solve_job, parse_job
from nff.rve.hinge_function import (HingeConstants, HingeGeometry, DeploymentRay,
                                    assemble_response, solver_kwargs, to_rve_params)

DATA = "data/fea/hinge_dataset_pet_merged"
WORK = "/tmp/hinge_repro"


def campaign_constants(manifest):
    """The PET campaign's own constants, so replicas differ ONLY in the perturbed knob."""
    c = manifest["const"]
    return dict(thickness=c["thickness"], w_c=c["w_c"], r_win=c["r_win"],
                fillet_ratio=c["fillet_ratio"], material="PET", n_through=c["n_through"],
                lc_fillet_frac=c["lc_fillet_frac"], lc_min_floor=c["lc_min_floor"],
                min_inc=c["min_inc"], field_freq=c["field_freq"], el_fields=c["el_fields"],
                stop_at_fracture=False, arcB_uz_free=c["arcB_uz_free"])


def pick_geometries(manifest, n_geoms):
    """Converged deep-fold jobs (where the error lives) + one control that should NOT buckle.

    The control is the harness check: a narrow ligament at small rotation stays in-plane, so its
    replica spread must be ~0. A nonzero spread there means the driver is wrong, not the physics.
    """
    # w_lig <= 25 only: the manifest still lists the wider jobs whose ROWS were dropped at merge
    # (Saint-Venant), and those are outside the domain the surrogate is trained on.
    ok = [r for r in manifest["jobs"]
          if r.get("stop_reason") == "completed" and np.isfinite(r.get("eta_a", np.nan))
          and r.get("w_lig", 99) <= 25.0]
    deep = sorted([r for r in ok if r.get("theta1_deg", 0) > 45], key=lambda r: r["w_lig"])
    picks = [deep[int(f * (len(deep) - 1))] for f in np.linspace(0.2, 0.95, n_geoms)]
    flat = [r for r in ok if r.get("theta1_deg", 99) < 12 and r["w_lig"] < 9]
    if flat:
        picks.append({**flat[0], "_control": True})
    return picks


def replicas(thickness, lc_frac):
    """(label, imp_amp, lc_fillet_frac). ``None`` imp_amp = the campaign default 0.3*thickness."""
    base = 0.3 * thickness
    return [("baseline", None, lc_frac),
            ("imp x0.97", 0.97 * base, lc_frac),          # immaterial: must not move W
            ("imp x1.03", 1.03 * base, lc_frac),          # immaterial: must not move W
            ("mesh x0.95", None, 0.95 * lc_frac),         # immaterial: re-mesh, same geometry
            ("imp 0.1t", 0.1 * thickness, lc_frac),       # amplitude sweep
            ("imp 1.0t", 1.0 * thickness, lc_frac)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-geoms", dest="n_geoms", type=int, default=4)
    ap.add_argument("--n-steps", dest="n_steps", type=int, default=6)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--ncpus", type=int, default=2, help="ccx threads PER solve")
    ap.add_argument("--timeout", type=float, default=1200)
    ap.add_argument("--out", default="data/fea/oracle_reproducibility.json")
    args = ap.parse_args()

    manifest = json.load(open(DATA + ".json"))
    base = campaign_constants(manifest)
    geoms = pick_geometries(manifest, args.n_geoms)
    shutil.rmtree(WORK, ignore_errors=True)

    jobs = []
    for r in geoms:
        for label, imp, lcf in replicas(base["thickness"], base["lc_fillet_frac"]):
            jobs.append((r, label, imp, lcf))
    print(f"{len(geoms)} geometries x {len(jobs)//len(geoms)} replicas = {len(jobs)} solves "
          f"({args.parallel} in parallel, {args.ncpus} ccx threads each)\n", flush=True)

    # PHASED, exactly like nff.rve.dataset.run_jobs: prepare SERIALLY (gmsh is a process-level
    # singleton -- meshing in threads emits "Gmsh has already been initialized" and every solve
    # fails), then solve in parallel (ccx is a subprocess), then parse serially. run_jobs itself
    # cannot be reused here because it takes ONE const for all jobs and each replica needs its own.
    prepared = []
    for r, label, imp, lcf in jobs:
        geo = HingeGeometry(w_lig=r["w_lig"], alpha_deg=r["alpha_deg"],
                            fillet_ratio=base["fillet_ratio"])
        ray = DeploymentRay(theta1_deg=r["theta1_deg"], eta_a=r["eta_a"], eta_s=r["eta_s"],
                            n_steps=args.n_steps)
        const = HingeConstants(**{**base, "lc_fillet_frac": lcf}, imp_amp=imp)
        # distinct workdir per replica -- the default depends on geometry+ray only, never on the
        # solver settings, so replicas would otherwise share one directory.
        wd = f"{WORK}/{r['job_id']}_{label.replace(' ', '').replace('.', '')}"
        try:
            m = prepare_job(to_rve_params(geo, const), workdir=wd,
                            **solver_kwargs(geo, ray, const))
            prepared.append((r, label, geo, ray, const, m, None))
        except Exception as e:
            prepared.append((r, label, geo, ray, const, None, f"prepare: {type(e).__name__}: {e}"))
    print(f"prepared {sum(1 for p in prepared if p[5])}/{len(prepared)} decks (serial gmsh)",
          flush=True)

    def _solve(p):
        _, _, _, _, _, m, err = p
        if err:
            return ""
        try:
            return solve_job(m, ncpus=args.ncpus, timeout=args.timeout, eps_f=None).stdout
        except Exception:
            m["stop_reason"] = "timeout"
            return ""
    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        stdouts = list(ex.map(_solve, prepared))

    results = []
    for (r, label, geo, ray, const, m, err), so in zip(prepared, stdouts):
        if err:
            results.append(dict(job_id=r["job_id"], label=label, ok=False, stop=err,
                                W=[], uz=[], theta=[])); continue
        try:
            res = assemble_response(geo, ray, const, parse_job(m, so))
            results.append(dict(job_id=r["job_id"], label=label, ok=True, stop=res.stop_reason,
                                W=np.asarray(res.W, float).tolist(),
                                uz=np.asarray(res.uz_max, float).tolist(),
                                theta=np.asarray(res.theta_deg, float).tolist()))
        except Exception as e:
            results.append(dict(job_id=r["job_id"], label=label, ok=False,
                                stop=f"parse: {type(e).__name__}", W=[], uz=[], theta=[]))

    # ---- per geometry: spread of W across replicas at the common increment count ----
    print(f"{'job':>6}{'w_lig':>8}{'theta1':>8}{'n_rep':>7}{'W spread':>11}{'uz spread':>11}  note")
    summary = []
    for r in geoms:
        rs = [x for x in results if x["job_id"] == r["job_id"] and x["ok"] and x["W"]]
        if len(rs) < 2:
            print(f'{r["job_id"]:6d}{r["w_lig"]:8.2f}{r["theta1_deg"]:8.1f}{len(rs):7d}'
                  f'{"n/a":>11}{"n/a":>11}  too few replicas'); continue
        k = min(len(x["W"]) for x in rs)
        Wm = np.array([x["W"][:k] for x in rs])
        Um = np.array([x["uz"][:k] for x in rs])
        # spread at the LAST common increment, relative to the mean there
        wsp = float(np.ptp(Wm[:, -1]) / max(abs(np.mean(Wm[:, -1])), 1e-12))
        usp = float(np.ptp(Um[:, -1]) / max(abs(np.mean(Um[:, -1])), 1e-12))
        # immaterial-only subset: the +/-3% and mesh replicas, which MUST agree
        imm = [i for i, x in enumerate(rs) if x["label"] in ("baseline", "imp x0.97", "imp x1.03",
                                                             "mesh x0.95")]
        wsp_imm = float(np.ptp(Wm[imm, -1]) / max(abs(np.mean(Wm[imm, -1])), 1e-12)) if len(imm) > 1 else float("nan")
        note = "CONTROL (should not buckle)" if r.get("_control") else ""
        print(f'{r["job_id"]:6d}{r["w_lig"]:8.2f}{r["theta1_deg"]:8.1f}{len(rs):7d}'
              f'{wsp*100:10.2f}%{usp*100:10.2f}%  immaterial-only {wsp_imm*100:.2f}%  {note}',
              flush=True)
        summary.append(dict(job_id=r["job_id"], w_lig=r["w_lig"], theta1_deg=r["theta1_deg"],
                            control=bool(r.get("_control")), n_replicas=len(rs),
                            W_spread=wsp, uz_spread=usp, W_spread_immaterial=wsp_imm,
                            per_replica={x["label"]: (x["W"][k-1], x["stop"]) for x in rs}))

    real = [s for s in summary if not s["control"]]
    if real:
        worst = max(s["W_spread_immaterial"] for s in real if np.isfinite(s["W_spread_immaterial"]))
        print(f"\nworst IMMATERIAL-perturbation spread across deep-fold geometries: {worst*100:.2f}%")
        print("  <2%  => oracle reproducible; the scatter is real variation -> only densification helps")
        print("  >10% => branch selection unstable; densification is futile -> fix the bifurcation")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(dict(summary=summary, raw=results, n_steps=args.n_steps), f, indent=2)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
