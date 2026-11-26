"""Emit every applicable figure for a completed grip-C sweep stage.

``run_closed`` already writes the per-run visuals (stages, loading diagram, animation, training
evolution, compression shift, cut pattern + DXF + A4 PDF, load path). This adds the ones that need
the EXTRACTED hinge paths or a comparison ACROSS arms, so a stage ends with everything we can draw.

⚠ ``energy_plot`` stays off for every surrogate run and is not emitted here: the energy
decomposition is hard-wired to the linear-spring ligament with no ``bond_energy_fn`` hook, so the
balance fails by construction. That figure would be wrong, not extra.

    PYTHONPATH=$(pwd) python legacy/experiments/figures/emit_grip_c_figures.py \
        --sweep data/outputs/sweep_grip_c/stage_A.json
"""

import argparse
import json
import os
import subprocess
import sys

OUT_DIR = "data/outputs/sweep_grip_c"
PATHS_DIR = "data/outputs/hinge_paths"


def run(cmd, label):
    proc = subprocess.run(cmd, capture_output=True, text=True)
    ok = proc.returncode == 0
    print(f"    {'ok  ' if ok else 'FAIL'}  {label}")
    if not ok:
        tail = (proc.stdout + proc.stderr).strip().split("\n")[-3:]
        for ln in tail:
            print(f"           {ln[:140]}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", nargs="+", required=True)
    ap.add_argument("--fig-dir", default=os.path.join(OUT_DIR, "figures"))
    args = ap.parse_args()

    rows = []
    for p in args.sweep:
        with open(p) as fh:
            d = json.load(fh)
        rows.extend(d if isinstance(d, list) else [d])
    rows = [r for r in rows if r and not r.get("error") and r.get("run_dir")]
    os.makedirs(args.fig_dir, exist_ok=True)
    py = [sys.executable]

    print(f"\n  {len(rows)} arm(s)\n")
    for r in rows:
        name, run_dir = r["name"], r["run_dir"]
        paths = os.path.join(PATHS_DIR, os.path.basename(run_dir))
        print(f"  [{name}]")
        if os.path.exists(paths + "/hinge_paths.npz"):
            # --paths takes the DIRECTORY; the script appends the `hinge_paths.npz` stem itself.
            run(py + ["nff/scripts/surrogate/plot_hinge_path_distribution.py",
                      "--paths", paths,
                      "--label", name,
                      "--out", os.path.join(args.fig_dir, f"{name}_path_distribution.png")],
                "hinge path distribution")
        run(py + ["legacy/experiments/figures/plot_compression_map.py",
                  "--config", os.path.join(run_dir, "config.yaml"),
                  "--params", os.path.join(run_dir, "best_params.pkl"),
                  "--steps", "40", "--label", name,
                  "--out", os.path.join(args.fig_dir, f"{name}_compression_map.png")],
            "compression map")

    # ── cross-arm ──
    print("\n  [cross-arm]")
    run(py + ["legacy/experiments/figures/plot_grip_c_pareto.py", "--sweep", *args.sweep,
              "--out", os.path.join(args.fig_dir, "pareto.png")], "force / target-shape Pareto")

    # Lowest-force arm against highest-force arm: what the extra force did to the ridden states.
    with_force = sorted([r for r in rows if r.get("force_N")], key=lambda r: r["force_N"])
    if len(with_force) >= 2:
        lo, hi = with_force[0], with_force[-1]
        pair = (f"{os.path.join(PATHS_DIR, os.path.basename(lo['run_dir']))},"
                f"{os.path.join(PATHS_DIR, os.path.basename(hi['run_dir']))}")
        run(py + ["legacy/experiments/figures/plot_compression_shift.py", "--pair", pair,
                  "--out", os.path.join(args.fig_dir, "compression_shift_force.png")],
            f"compression shift {lo['name']} -> {hi['name']}")

    print(f"\n  figures in {args.fig_dir}\n")


if __name__ == "__main__":
    main()
