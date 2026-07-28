"""Analyze a tensile sample: Bluehill raw curves + manual summary -> properties.

Matches the Bluehill raw curve blocks (in order) to the ``included`` rows of the
operator's summary sheet (the source of truth for area, gauge length, direction),
then extracts E / yield / UTS / draw-plateau / strain-at-break per specimen and prints
a table. Optionally writes per-specimen processed stress-strain CSVs and a plot.

Usage::

    python -m nff.scripts.calibration.analyze_run \
        --csv   data/experiments/raw/20260721_212033_1.csv \
        --summary "data/experiments/raw/Critical tile lenght for gravity - Strip Tensile.csv" \
        [--compliance 0.0] [--plot out.png] [--processed-dir data/experiments/processed]

Notes:
    * Crosshead strain is compliance-corrupted -> reported E is a LOWER BOUND unless
      ``--compliance`` (mm/N, from a known-E reference strip) is supplied, or a video
      extensometer is used (see video_extensometer + the plan doc).
    * Area comes from the summary sheet, not the (possibly stale) Bluehill geometry.
"""
from __future__ import annotations

import argparse

from nff.calibration import bluehill_io, stress_strain, summary_io


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True, help="Bluehill results+raw CSV")
    ap.add_argument("--summary", help="manual summary sheet (source of truth)")
    ap.add_argument("--compliance", type=float, default=0.0,
                    help="machine compliance C_machine [mm/N] to subtract (default 0)")
    ap.add_argument("--draw-window", nargs=2, type=float, metavar=("LO", "HI"),
                    help="crosshead-disp window [mm] to median for the draw plateau")
    ap.add_argument("--plot", help="write a stress-strain overlay PNG here")
    ap.add_argument("--processed-dir", help="write per-specimen processed CSVs here")
    args = ap.parse_args()

    runs = bluehill_io.load_bluehill_csv(args.csv)
    summ = [r for r in summary_io.load_summary(args.summary) if r.included] if args.summary else []

    print(f"{len(runs)} raw curves; {len(summ)} included summary rows\n")
    header = f"{'name':6} {'dir':4} {'A':6} {'L':6} {'E':9} {'yield':7} {'UTS':7} {'draw':7} {'eps_br':7}"
    print(header)
    print("-" * len(header))

    curves = []
    for k, run in enumerate(runs):
        meta = summ[k] if k < len(summ) else None
        area = (meta.a_mean_mm2 if meta and meta.a_mean_mm2 else run.a_mean_mm2) or 1.0
        L0 = (meta.L_mm if meta and meta.L_mm else 103.0)
        name = meta.name if meta else str(run.index)
        direction = (meta.direction if meta else None) or "?"

        strain = stress_strain.crosshead_strain(
            run.disp_mm, L0, force_N=run.force_N, C_machine_mm_per_N=args.compliance
        )
        res = stress_strain.analyze(
            run.force_N, area, strain,
            draw_window_mm=tuple(args.draw_window) if args.draw_window else None,
            disp_mm=run.disp_mm,
        )
        curves.append((name, res))

        e = f"{res.E_MPa/1000:.2f}GPa" if res.E_MPa else "   -   "
        dr = f"{res.draw_plateau_MPa:.1f}" if res.draw_plateau_MPa else "  -  "
        print(f"{name:6} {direction:4} {area:6.2f} {L0:6.1f} {e:9} "
              f"{res.yield_MPa:6.1f} {res.uts_MPa:6.1f} {dr:>6} {res.strain_at_break*100:6.1f}%")

        if args.processed_dir:
            import os
            os.makedirs(args.processed_dir, exist_ok=True)
            out = os.path.join(args.processed_dir, f"{name}_stress_strain.csv")
            with open(out, "w") as f:
                f.write("strain,stress_MPa\n")
                for s, sig in zip(res.strain, res.stress_MPa):
                    f.write(f"{s:.6f},{sig:.4f}\n")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 5))
        for name, res in curves:
            ax.plot(res.strain * 100, res.stress_MPa, lw=1, label=name)
        ax.set_xlabel("engineering strain [%]")
        ax.set_ylabel("engineering stress [MPa]")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        print(f"\nplot -> {args.plot}")


if __name__ == "__main__":
    main()
