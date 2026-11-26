"""Optimize the FORCE and the TARGET SHAPE for the single-tile diagonal grip (clamp 0 -> pull 8).

The `pet_1t` family was calibrated to keep >=95% of ridden hinge-states inside the surrogate's trust
box WITH NO COMPRESSION PENALTY IN THE LOSS. Those grips are compression-limited, not
rotation-limited -- nothing reaches past ~20 deg before the axial `eta_a` hits the compressive
floor -- so that calibration is what caps the force, and therefore the deployment.

This sweep asks whether the penalty lifts the cap. With `compression` at 10x gradient parity and
`damage` at 1x, the force is pushed PAST the old limit and the penalty is asked to pull the states
back in. Precedent on the distributed grip: joint in-box 78.5% -> 100.0% at equal deployment.

    Stage A  force ladder at fixed pull-in            -> how much extra theta the penalty buys
    Stage B  pull-in ladder at Stage A's winner       -> how much headroom the target should carry
    Stage C  shape-only control + seed robustness

Each stage is a separate invocation on purpose: read the table before choosing the next stage's
centre. Every arm is resumable -- an arm whose JSON already exists is skipped.

    PYTHONPATH=$(pwd) conda run -n kgnn_mac python legacy/experiments/diagnostics/sweep_grip_c.py \
        --stage A --jobs 4
    ... then, with the winner ...
    ... --stage B --force-scale 1.8 --jobs 2
    ... --stage C --force-scale 1.8 --pull-in-x 0.90 --pull-in-y 0.97 --jobs 4
"""

import argparse
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.getcwd())
from nff.scripts.design.calibrate_target import _set_in_block   # noqa: E402  (text substitution)

BASE = "data/configs/closed/gripC_base.yaml"
CKPT = "data/surrogates/hinge_surrogate_pet_v3"
DATA = "data/fea/hinge_dataset_pet_merged"
CONFIG_DIR = "data/configs/closed"
OUT_DIR = "data/outputs/sweep_grip_c"
PATHS_DIR = "data/outputs/hinge_paths"

# The family's inherited design headroom: the target is this fraction of the measured deployed bbox.
PULL_IN_X, PULL_IN_Y = 0.953, 0.990
COMPRESSION_X_PARITY = 10.0     # arm-H recipe: compression dominates, damage sits at parity
DAMAGE_X_PARITY = 1.0
INIT_NOISE, INIT_SEED = 0.5, 0

PY = [sys.executable]


def sh(cmd, log_path=None):
    """Run a subprocess, tee to a log, return (ok, stdout)."""
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = proc.stdout + proc.stderr
    if log_path:
        with open(log_path, "a") as fh:
            fh.write(f"\n$ {' '.join(cmd)}\n{out}\n")
    return proc.returncode == 0, out


def grep1(pattern, text, cast=float, default=None, group=1):
    m = re.search(pattern, text)
    return cast(m.group(group)) if m else default


def arms_for(stage, force_scale, pull_in_x, pull_in_y):
    """(name, load_scale, pull_in_x, pull_in_y, comp_mult, dmg_mult, seed) per arm."""
    if stage == "A":
        return [(f"gripC_f{s:g}".replace(".", "p"), s, PULL_IN_X, PULL_IN_Y,
                 COMPRESSION_X_PARITY, DAMAGE_X_PARITY, INIT_SEED)
                for s in (1.0, 1.4, 1.8, 2.2)]
    if stage == "B":
        # TARGET SHAPE ladder: how much pull-in should the target ask the design to supply?
        # 0.953/0.990 is the family's inherited headroom and is covered by stage W's w1 arm, so the
        # two new points bracket it: no headroom at all, and roughly double.
        # Weights are 1x parity compression / 0 damage -- stage W's result, NOT the 10x recipe that
        # collapsed stage A.
        return [(f"gripC_pi{px:g}".replace(".", "p"), force_scale, px, py, 1.0, 0.0, INIT_SEED)
                for px, py in ((1.00, 1.00), (0.90, 0.97))]
    if stage == "C":
        arms = [("gripC_shapeonly", force_scale, pull_in_x, pull_in_y, 0.0, 0.0, INIT_SEED)]
        arms += [(f"gripC_seed{s}", force_scale, pull_in_x, pull_in_y,
                  COMPRESSION_X_PARITY, DAMAGE_X_PARITY, s) for s in (1, 2, 3, 4, 5)]
        return arms
    if stage == "W":
        # ── WEIGHT ladder, added after Stage A. ────────────────────────────────────────────
        # Stage A ran compression at 10x parity (the arm-H recipe from the DISTRIBUTED grip) and
        # every arm collapsed: theta_max 17.8 deg untrained -> 4.4-7.8 deg trained, boundary error
        # 78-143 mm against the family's 8.38, face area to -87/+345%. Per-axis in-box was
        # eta_a 100% / eta_s 100% -- the penalty did not fail, it OVER-succeeded: |eta_a| <= 0.077
        # against a floor of 0.140, bought by not deploying at all.
        #
        # That is the difference between the two grips. On the distributed grip compression is
        # incidental, so leaning on it costs little. Grip C is COMPRESSION-LIMITED -- compression is
        # the binding constraint on deployment -- so a compression penalty pushes directly against
        # the thing being optimized, and 10x parity wins that fight outright.
        #
        # So: fix the force at the family's calibrated value and ladder the WEIGHT instead, with a
        # shape-only control at the bottom. Damage is held at 0 except in the last arm, to separate
        # it from compression -- its 1x-parity weight is 2402, measured against a near-zero gradient
        # at the untrained design, and it is a prime suspect in its own right.
        return [("gripC_w0",     force_scale, pull_in_x, pull_in_y, 0.0,  0.0, INIT_SEED),
                ("gripC_w0p3",   force_scale, pull_in_x, pull_in_y, 0.3,  0.0, INIT_SEED),
                ("gripC_w1",     force_scale, pull_in_x, pull_in_y, 1.0,  0.0, INIT_SEED),
                ("gripC_w3",     force_scale, pull_in_x, pull_in_y, 3.0,  0.0, INIT_SEED),
                ("gripC_w1_d1",  force_scale, pull_in_x, pull_in_y, 1.0,  1.0, INIT_SEED)]
    if stage == "F":
        # Stage A redone at a SANE weight. Stage A's force ladder was uninterpretable because
        # compression at 10x parity collapsed the deployment before force could matter. The weight
        # ladder (stage W) showed 1x parity is the smallest weight that brings min eta_a to the
        # -0.140 floor, so the force question is asked again there, with damage at 0 (stage W
        # measured it as very nearly inert: 14.5+2402 moved theta 12.29 -> 12.02).
        return [(f"gripC_g{s:g}".replace(".", "p"), s, PULL_IN_X, PULL_IN_Y, 1.0, 0.0, INIT_SEED)
                for s in (1.4, 1.8)]
    raise SystemExit(f"unknown stage {stage!r}")


def run_arm(arm, every):
    name, scale, px, py, comp_mult, dmg_mult, seed = arm
    res_path = os.path.join(OUT_DIR, f"arm_{name}.json")
    if os.path.exists(res_path):
        print(f"  [{name}] already done -- skipping")
        return json.load(open(res_path))

    log = os.path.join(OUT_DIR, f"{name}.log")
    open(log, "w").close()
    cfg_path = os.path.join(CONFIG_DIR, f"{name}.yaml")
    rec = dict(name=name, load_scale=scale, pull_in_x=px, pull_in_y=py,
               compression_x_parity=comp_mult, damage_x_parity=dmg_mult, init_seed=seed,
               init_noise=INIT_NOISE)

    # 1 ── force + target. --from-deployed because a concentrated grip does not open uniformly,
    #      so the analytic lam(theta) rectangle would simply be the wrong shape.
    parity_json = os.path.join(OUT_DIR, f"{name}_parity.json")
    # Step-level resume: calibration and parity are deterministic given (base, scale, pull-in,
    # seed), and each costs minutes. If a previous invocation already produced them, reuse them --
    # an interrupted sweep should not repay for work whose output is sitting on disk.
    reuse = os.path.exists(cfg_path) and os.path.exists(parity_json)
    if reuse:
        print(f"  [{name}] reusing existing config + parity")
        cfg_text = open(cfg_path).read()
        rec["force_N"] = grep1(r"value:\s*([\d.]+)\}", cfg_text)
        rec["target"] = {k: grep1(rf"target_{k}:\s*([-\d.]+)", cfg_text)
                         for k in ("half_w", "half_h", "cx", "cy")}
        rec["theta_untrained_max"] = grep1(r"hinge angle\s+max ([\d.]+)", cfg_text)
    else:
        print(f"  [{name}] calibrating (load x{scale:g}, pull-in {px:g}/{py:g}) ...")
        ok, out = sh(PY + ["nff/scripts/design/calibrate_target.py", "--config", BASE,
                           "--load-scale", str(scale), "--from-deployed",
                           "--pull-in-x", str(px), "--pull-in-y", str(py),
                           "--init-noise", str(INIT_NOISE), "--init-seed", str(seed),
                           "--write-config", cfg_path], log)
        if not ok:
            rec["error"] = "calibrate_target failed"
            json.dump(rec, open(res_path, "w"), indent=2)
            return rec
        rec["force_N"] = grep1(r"LOAD\s+x[\d.]+\s+->\s+([\d.]+) N", out)
        rec["target"] = dict(zip(("half_w", "half_h", "cx", "cy"), (
            grep1(r"half_w ([\d.]+)", out), grep1(r"half_h ([\d.]+)", out),
            grep1(r"center \(([-\d.]+),", out), grep1(r"center \([-\d.]+, ([-\d.]+)\)", out))))
        rec["theta_untrained_max"] = grep1(r"hinge angle: max ([\d.]+)", out)

        # 2 ── gradient parity ON THIS ARM's config. Never inherited: the load just changed, and a
        #      weight carried across a re-calibration is how `compression: 1.0` ended up 8.7x small.
        print(f"  [{name}] measuring gradient parity ...")
        ok, out = sh(PY + ["nff/scripts/diagnostics/gradient_parity.py", "--config", cfg_path,
                           "--terms", "compression,damage", "--out", parity_json], log)
        if not ok:
            rec["error"] = "gradient_parity failed"
            json.dump(rec, open(res_path, "w"), indent=2)
            return rec

    parity = json.load(open(parity_json))
    w_comp = comp_mult * parity["terms"]["compression"]["parity_weight"]
    w_dmg = dmg_mult * parity["terms"]["damage"]["parity_weight"]
    rec["parity"] = {k: v["parity_weight"] for k, v in parity["terms"].items()}
    rec["weights"] = dict(compression=w_comp, damage=w_dmg,
                          chamfer=parity["ref_weight"])

    lines = open(cfg_path).read().split("\n")
    _set_in_block(lines, "loss_weights", "compression", f"{w_comp:.6g}")
    _set_in_block(lines, "loss_weights", "damage", f"{w_dmg:.6g}")
    open(cfg_path, "w").write("\n".join(lines))

    # 3 ── train (the long step) -- emits every visual into its own run dir
    print(f"  [{name}] training (compression={w_comp:.3g} damage={w_dmg:.3g}) ...")
    ok, out = sh(PY + ["nff/scripts/run_closed.py", "--config-name", name, "--every", str(every)],
                 log)
    run_dir = grep1(r"Run directory: (\S+)", out, str)
    rec["run_dir"] = run_dir
    if not ok or not run_dir:
        rec["error"] = "run_closed failed"
        json.dump(rec, open(res_path, "w"), indent=2)
        return rec
    rec["best_loss"] = grep1(r"best loss=([\d.eE+-]+)", out)
    rec["chamfer_at_best"] = grep1(r"chamfer there=([\d.eE+-]+)", out)
    rec["boundary_mean_mm"] = grep1(r"boundary vs target \(\d+ vertices\): mean=([\d.]+)mm", out)
    rec["boundary_max_mm"] = grep1(r"boundary vs target .*max=([\d.]+)mm", out)
    rec["theta_mean"] = grep1(r"deployment angle: mean=([\d.]+)deg", out)
    rec["theta_min"] = grep1(r"deployment angle: mean=[\d.]+deg\s+\[([\d.]+),", out)
    rec["theta_max"] = grep1(r"deployment angle: mean=[\d.]+deg\s+\[[\d.]+, ([\d.]+)\]", out)
    rec["overlap_stage2"] = grep1(r"stage2 twisted=\d+ overlap=([\d.eE+-]+)", out)
    rec["area_min_pct"] = grep1(r"face area: min=([-+\d.]+)%", out)
    rec["area_max_pct"] = grep1(r"face area: min=[-+\d.]+% max=([-+\d.]+)%", out)
    rec["w_lig_min"] = grep1(r"learned w_lig \[mm\]: min=([\d.]+)", out)
    rec["w_lig_max"] = grep1(r"learned w_lig \[mm\]: min=[\d.]+ mean=[\d.]+ max=([\d.]+)", out)
    # Stage-2 convergence: without it the load path is skipped and every number above is void.
    rec["stage2_converged"] = "did not converge" not in out.lower()

    # 4 ── ride the trained design and score it against the checkpoint's own trust box
    paths_out = os.path.join(PATHS_DIR, os.path.basename(run_dir))
    print(f"  [{name}] extracting hinge paths ...")
    ok, out = sh(PY + ["nff/scripts/surrogate/extract_hinge_paths.py",
                       "--config", os.path.join(run_dir, "config.yaml"),
                       "--params", os.path.join(run_dir, "best_params.pkl"),
                       "--steps", "40", "--out", paths_out], log)
    if ok:
        hp = os.path.join(paths_out, "hinge_paths.json")
        if os.path.exists(hp):
            # These live under diagnostics.compression, NOT at the top level.
            full = json.load(open(hp))
            d = full.get("diagnostics", {}).get("compression", {})
            rec["theta_range_deg"] = full.get("diagnostics", {}).get("theta_range_deg")
            per_hinge = d.get("min_eta_a_per_hinge") or []
            rec["min_eta_a"] = float(min(per_hinge)) if per_hinge else d.get("min_eta_a")
            # DEPTH, not the compressive-sample fraction: the two move in OPPOSITE directions
            # under this penalty, and the fraction alone reads as though it backfired.
            rec["mean_compression_depth"] = (
                sum(max(0.0, -v) for v in per_hinge) / len(per_hinge)) if per_hinge else None

        ood_json = os.path.join(paths_out, "ood_report.json")
        print(f"  [{name}] scoring trust-box coverage ...")
        ok2, out2 = sh(PY + ["-m", "nff.scripts.surrogate.score_deployment_ood",
                             "--paths", os.path.join(paths_out, "hinge_paths"),
                             "--surrogate", CKPT, "--data", DATA, "--out", ood_json], log)
        if ok2 and os.path.exists(ood_json):
            pri = json.load(open(ood_json))["checkpoints"]["primary"]
            rec["in_box_joint"] = pri["in_box_joint"]
            rec["in_box_axes"] = {k: v["in_box"] for k, v in pri["axes"].items()}
            if "nn" in pri:
                rec["nn_p90"] = pri["nn"]["query_p90"]
                rec["nn_p50"] = pri["nn"]["query_p50"]
                rec["nn_yardstick_p90"] = pri["nn"]["yardstick_p90"]

    json.dump(rec, open(res_path, "w"), indent=2)
    print(f"  [{name}] done -> {res_path}")
    return rec


def table(rows):
    hdr = (f"{'arm':<18} {'N':>7} {'theta_max':>9} {'in-box':>7} {'min_eta_a':>10} "
           f"{'depth':>7} {'bnd_mm':>7} {'NN p90':>7} {'area%':>14} {'w_lig':>12}")
    print("\n" + hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in rows:
        if not r:
            continue
        f = lambda k, fmt=".3f", d="  --  ": (format(r[k], fmt) if r.get(k) is not None else d)
        area = (f"{f('area_min_pct','+.1f')}/{f('area_max_pct','+.1f')}"
                if r.get("area_min_pct") is not None else "  --  ")
        wl = (f"{f('w_lig_min','.1f')}-{f('w_lig_max','.1f')}"
              if r.get("w_lig_min") is not None else "  --  ")
        ib = (f"{r['in_box_joint']*100:.1f}%" if r.get("in_box_joint") is not None else "  --  ")
        print(f"{r['name']:<18} {f('force_N','.2f'):>7} {f('theta_max','.2f'):>9} {ib:>7} "
              f"{f('min_eta_a','+.4f'):>10} {f('mean_compression_depth','.4f'):>7} "
              f"{f('boundary_mean_mm','.2f'):>7} {f('nn_p90','.3f'):>7} {area:>14} {wl:>12}")
    print("\n  trust-box floor eta_a >= -0.140.  Report min eta_a / depth / in-box, NEVER the "
          "compressive-sample\n  fraction alone -- depth and fraction move in OPPOSITE directions "
          "under this penalty.\n  Face area is reported because an in-box win does NOT establish "
          "manufacturability.\n")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=("A", "B", "C", "W", "F"), default="A")
    p.add_argument("--force-scale", type=float, default=1.0,
                   help="Stage A's winning load scale (stages B and C)")
    p.add_argument("--pull-in-x", type=float, default=PULL_IN_X, help="Stage B's winner (stage C)")
    p.add_argument("--pull-in-y", type=float, default=PULL_IN_Y)
    p.add_argument("--jobs", type=int, default=3, help="arms to run concurrently")
    p.add_argument("--every", type=int, default=6, help="run_closed snapshot cadence")
    p.add_argument("--no-caffeinate", action="store_true")
    args = p.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(PATHS_DIR, exist_ok=True)

    caf = None
    if not args.no_caffeinate:
        # -w <our pid>: the assertion dies with this process. `caffeinate` is a SEPARATE process,
        # so a bare `caffeinate -dimsu &` would outlive an interrupted sweep and hold the machine
        # awake indefinitely. If one is ever orphaned: pkill -f "caffeinate -dimsu".
        caf = subprocess.Popen(["caffeinate", "-dimsu", "-w", str(os.getpid())])
        print(f"  caffeinate -dimsu holding no-sleep for pid {os.getpid()} (pid {caf.pid})")

    arms = arms_for(args.stage, args.force_scale, args.pull_in_x, args.pull_in_y)
    print(f"\n  STAGE {args.stage}: {len(arms)} arm(s), {args.jobs} at a time, base {BASE}\n")
    try:
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            rows = list(ex.map(lambda a: run_arm(a, args.every), arms))
    finally:
        if caf is not None:
            caf.terminate()

    rows = [r for r in rows if r]
    table(rows)
    collated = os.path.join(OUT_DIR, f"stage_{args.stage}.json")
    json.dump(rows, open(collated, "w"), indent=2)
    print(f"  wrote {collated}\n")

    bad = [r["name"] for r in rows if r.get("error") or r.get("stage2_converged") is False]
    if bad:
        print(f"  ⚠ arms with no usable result: {', '.join(bad)}\n")


if __name__ == "__main__":
    main()
