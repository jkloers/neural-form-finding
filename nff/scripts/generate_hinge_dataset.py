"""Generate (or rehearse) the hinge constitutive dataset.

    # rehearsal: pick the fillet coefficient c and confirm alpha-robustness + a healthy
    # prepare->solve->parse path, through the SAME code the campaign uses
    conda run -n ccx python nff/scripts/generate_hinge_dataset.py --rehearsal

    # full campaign
    conda run -n ccx python nff/scripts/generate_hinge_dataset.py \
        --n 300 --out sofa/output/hinge_dataset --parallel 6

Everything routes through ``nff.rve.hinge_function`` (the hinge as a function) and
``nff.rve.dataset`` (sample -> evaluate -> write); this script only wires the CLI.
"""

import argparse
import json

import numpy as np

from nff.rve.hinge_function import (HingeConstants, HingeGeometry, DeploymentRay, REGIME_NAME,
                                    evaluate_hinge)
from nff.rve.dataset import sample_jobs, generate_dataset, run_jobs


# ── rehearsal: determine rho = c * w_lig and check alpha robustness ────────────────

def _rehearsal_trials():
    """(label, fillet_ratio c, geometry, pure-rotation ray) covering a c-sweep + alpha + size."""
    ray = lambda: DeploymentRay(theta1_deg=60.0, eta_a=0.0, eta_s=0.0, n_steps=15)
    trials = []
    for c in (0.08, 0.12, 0.16, 0.20, 0.25, 0.30):                # fillet-coefficient sweep
        trials.append((f"c{c:.2f}_a90_w5", c, HingeGeometry(5.0, 90.0), ray()))
    for a in (45.0, 135.0):                                       # tilted-cut robustness at c=0.16
        trials.append((f"c0.16_a{a:.0f}_w5", 0.16, HingeGeometry(5.0, a), ray()))
    for w in (1.0, 10.0):                                         # slenderness extremes
        trials.append((f"c0.16_a90_w{w:.0f}", 0.16, HingeGeometry(w, 90.0), ray()))
    return trials


def run_rehearsal(parallel, timeout):
    trials = _rehearsal_trials()
    by_c = {}                                                     # group by c -> one HingeConstants
    for label, c, geo, ray in trials:
        by_c.setdefault(round(c, 3), []).append((label, geo, ray))

    print(f"Rehearsal: {len(trials)} trials (r_win=30, t=1.0, w_c=0.2, S235, pure rotation to 60deg)\n")
    hdr = "{:<16} {:>4} {:>5} {:>4} {:>5} | {:>6} {:>5} {:>8} {:>6} {:>8} {:>4}".format(
        "trial", "c", "alph", "wlg", "rho", "elems", "nsmp", "th_fail", "uz/t", "Wfin", "ok")
    print(hdr + "\n" + "-" * len(hdr))
    rows = []
    for c, group in sorted(by_c.items()):
        const = HingeConstants(fillet_ratio=c)
        responses = run_jobs([(g, r) for _, g, r in group], const,
                             n_parallel=parallel, timeout=timeout)
        for (label, geo, ray), resp in zip(group, responses):
            if resp is None or resp.n_samples == 0:
                print(f"{label:<16} {c:>4.2f} {geo.alpha_deg:>5.0f} {geo.w_lig:>4.0f} "
                      f"{geo.rho(const):>5.2f} |  FAILED (no data)")
                rows.append(dict(label=label, c=c, ok=False)); continue
            thf = resp.failure_theta_deg
            uzt = float(np.nanmax(resp.uz_max)) / const.thickness
            wfin = float(resp.W[np.isfinite(resp.W)][-1]) if np.isfinite(resp.W).any() else float("nan")
            print("{:<16} {:>4.2f} {:>5.0f} {:>4.0f} {:>5.2f} | {:>6} {:>5} {:>8} {:>6.2f} {:>8.0f} {:>4}"
                  .format(label, c, geo.alpha_deg, geo.w_lig, geo.rho(const), resp.n_elems,
                          resp.n_samples, f"{thf:.1f}" if np.isfinite(thf) else "none", uzt, wfin,
                          str(resp.ok)))
            rows.append(dict(label=label, c=c, alpha=geo.alpha_deg, w_lig=geo.w_lig,
                             rho=geo.rho(const), n_elems=resp.n_elems, n_samples=resp.n_samples,
                             failure_theta_deg=None if not np.isfinite(thf) else thf,
                             uz_over_t=uzt, W_final=wfin, ok=resp.ok,
                             regime=[REGIME_NAME[int(x)] for x in resp.regime],
                             theta_deg=resp.theta_deg.tolist(), W=resp.W.tolist(),
                             M_theta=resp.M_theta.tolist(), peeq_p99=resp.peeq_p99.tolist()))
    out = "data/outputs/hinge_rehearsal.json"
    with open(out, "w") as f:
        json.dump(rows, f, indent=2, default=float)
    print(f"\nPick c = smallest value whose fold spreads (th_fail high / no early tear).\nsaved -> {out}")


# ── full campaign ─────────────────────────────────────────────────────────────────

def run_campaign(args):
    const = HingeConstants(fillet_ratio=args.fillet_ratio, n_through=args.n_through,
                           thickness=args.thickness)
    jobs = sample_jobs(args.n, seed=args.seed, n_steps=args.steps,
                       theta1_deg=(args.angle, args.angle),
                       w_lig=(args.w_lig_min, args.w_lig_max),
                       eta_a=(0.0, args.eta_a_max), eta_s=(-args.eta_s_max, args.eta_s_max))
    print(f"Campaign: {args.n} jobs (t={const.thickness}mm, w_lig=[{args.w_lig_min},{args.w_lig_max}]mm, "
          f"c={const.fillet_ratio}, n_through={const.n_through}, to {args.angle:.0f}deg, "
          f"eta_a<={args.eta_a_max} |eta_s|<={args.eta_s_max}, fracture_margin={args.fracture_margin}) "
          f"-> {args.out}.npz")
    summary = generate_dataset(jobs, args.out, const, n_parallel=args.parallel,
                               timeout=args.timeout, batch_size=args.batch_size,
                               fracture_margin=args.fracture_margin)
    print(f"  jobs usable   : {summary['n_usable']}/{summary['n_jobs']}  "
          f"({summary['n_errored']} errored, {summary['n_finished_to_cap']} survived to cap)")
    print(f"  samples       : {summary['n_samples']}  "
          f"(elastic {summary['n_elastic']} / plastic {summary['n_plastic']} / failed {summary['n_failed']})")
    print(f"  wrote {args.out}.npz + {args.out}.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rehearsal", action="store_true", help="fillet/alpha check, no dataset written")
    ap.add_argument("--n", type=int, default=1000, help="number of (geometry, ray) jobs")
    ap.add_argument("--out", default="sofa/output/hinge_dataset", help="output path (no extension)")
    ap.add_argument("--parallel", type=int, default=9)
    ap.add_argument("--timeout", type=float, default=300, help="per-sim cap [s]; stop-at-fracture ends most sooner")
    ap.add_argument("--batch-size", dest="batch_size", type=int, default=50, help="checkpoint after every N jobs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=20, help="load steps per ray")
    ap.add_argument("--angle", type=float, default=60.0, help="full-deployment rotation [deg]")
    ap.add_argument("--fillet-ratio", dest="fillet_ratio", type=float, default=0.16)
    ap.add_argument("--n-through", dest="n_through", type=int, default=2)
    # geometry + displacement envelope (exposed so a deeper campaign is one command)
    ap.add_argument("--w-lig-min", dest="w_lig_min", type=float, default=1.0, help="ligament width lo [mm]")
    ap.add_argument("--w-lig-max", dest="w_lig_max", type=float, default=20.0, help="ligament width hi [mm]")
    ap.add_argument("--thickness", type=float, default=1.0, help="sheet gauge [mm]; 1.0-2.0 = laser-cut standard")
    ap.add_argument("--eta-a-max", dest="eta_a_max", type=float, default=1.0, help="max axial neck-strain ratio a/w_lig")
    ap.add_argument("--eta-s-max", dest="eta_s_max", type=float, default=0.7, help="max |shear| neck-strain ratio s/w_lig")
    ap.add_argument("--fracture-margin", dest="fracture_margin", type=float, default=1.1,
                    help="stop-at-fracture threshold x eps_f; raise (e.g. 2.5) to run PAST first fracture (D regime)")
    args = ap.parse_args()

    if args.rehearsal:
        run_rehearsal(args.parallel, args.timeout)
    else:
        run_campaign(args)


if __name__ == "__main__":
    main()
