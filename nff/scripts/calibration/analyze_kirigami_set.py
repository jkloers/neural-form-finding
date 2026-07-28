"""Analyze a cold-draw tensile set (plain per-specimen CSVs + drawn-dimension summary).

Matches each summary row to its ``<name>.csv`` raw file, toe-corrects, and extracts
initial-tangent E, yield, draw plateau, draw ratio lambda (from measured drawn dims), and
the true-stress draw anchor. Aggregates MD/CD, builds the monotonic true-stress ``*PLASTIC``
table, writes an overlay figure + per-specimen processed curves, and prints a report.

Usage::

    python -m nff.scripts.calibration.analyze_kirigami_set \
        --dir data/experiments/raw/kirigami_20260723 \
        --summary "data/experiments/raw/kirigami_20260723/Kirigami experiments - Tensile test (safe).csv" \
        --E 3000 --plot data/experiments/processed/kirigami_20260723_overlay.png \
        --processed-dir data/experiments/processed
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from nff.calibration import bluehill_io, stress_strain, summary_io


def _toe_correct(disp_mm, force_N, thresh_N=5.0):
    i0 = int(np.argmax(force_N > thresh_N))
    return disp_mm - disp_mm[i0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", required=True, help="folder of per-specimen <name>.csv files")
    ap.add_argument("--summary", required=True, help="metadata sheet with drawn dims")
    ap.add_argument("--E", type=float, default=3000.0, help="elastic modulus [MPa] for eps_p (default 3000 = literature PET)")
    ap.add_argument("--draw-lo", type=float, default=12.0, help="draw-plateau window start [mm disp]")
    ap.add_argument("--plot")
    ap.add_argument("--processed-dir")
    args = ap.parse_args()

    rows = summary_io.load_summary(args.summary)
    recs, curves = [], []
    for m in rows:
        if not m.included:
            continue
        fn = os.path.join(args.dir, m.name.replace(".", "_") + ".csv")
        if not os.path.exists(fn):
            print(f"  [skip] no file for {m.name}")
            continue
        run = bluehill_io.load_raw_csv(fn)
        A0, L = m.a_mean_mm2, m.L_mm
        disp = _toe_correct(run.disp_mm, run.force_N)
        strain = disp / L
        stress = run.force_N / A0

        E = stress_strain.initial_tangent_modulus(stress, strain)
        yld = float(np.max(stress))
        drew = m.lam is not None
        dm = (disp >= args.draw_lo) & (disp <= disp[-1] - 1)
        plateau = float(np.median(stress[dm])) if dm.sum() > 5 else None
        lam = m.lam
        sig_true = lam * plateau if (plateau and lam) else None
        eps_true = float(np.log(lam)) if lam else None
        recs.append(dict(name=m.name, dir=m.direction, drew=drew, E=E, yld=yld,
                         plateau=plateau, lam=lam, sig_true=sig_true, eps_true=eps_true))
        curves.append((m.name, m.direction, strain, stress))

        if args.processed_dir:
            os.makedirs(args.processed_dir, exist_ok=True)
            with open(os.path.join(args.processed_dir, f"{m.name}_stress_strain.csv"), "w") as f:
                f.write("strain,stress_MPa\n")
                for s, sig in zip(strain, stress):
                    f.write(f"{s:.6f},{sig:.4f}\n")

    # ---- report -----------------------------------------------------------
    print(f"\n{'spec':6}{'dir':4}{'E_tan':8}{'yield':7}{'plateau':8}{'lam':6}{'sig_true':9}{'eps_true':9}")
    print("-" * 57)
    for r in recs:
        tag = "" if r["drew"] else "  *BROKE*"
        e = f"{r['E']/1000:.2f}GPa" if r["E"] else "  -  "
        pl = f"{r['plateau']:.1f}" if r["plateau"] else "  -"
        st = f"{r['sig_true']:.0f}" if r["sig_true"] else " -"
        et = f"{r['eps_true']:.2f}" if r["eps_true"] else " -"
        lm = f"{r['lam']:.2f}" if r["lam"] else "BROKE"
        print(f"{r['name']:6}{r['dir']:4}{e:8}{r['yld']:7.1f}{pl:>8}{lm:>6}{st:>9}{et:>9}{tag}")

    drawn = [r for r in recs if r["drew"]]

    def agg(sel, key):
        vals = [r[key] for r in sel if r[key] is not None]
        return (np.mean(vals), np.std(vals)) if vals else (float("nan"), float("nan"))

    print("\n=== aggregates (drawn specimens) ===")
    for label, sel in [("CD", [r for r in drawn if r["dir"] == "CD"]),
                       ("MD", [r for r in drawn if r["dir"] == "MD"]),
                       ("ALL", drawn)]:
        if not sel:
            continue
        ym, ys = agg(sel, "yld"); lm, ls = agg(sel, "lam")
        pm, _ = agg(sel, "plateau"); sm, _ = agg(sel, "sig_true"); em, _ = agg(sel, "eps_true")
        print(f"  {label} (n={len(sel)}): yield={ym:.1f}±{ys:.1f}MPa  plateau={pm:.1f}MPa  "
              f"lambda={lm:.2f}±{ls:.2f}  sig_true={sm:.0f}MPa @ eps={em:.2f}")

    # ---- *PLASTIC table ---------------------------------------------------
    ym, _ = agg(drawn, "yld"); sm, _ = agg(drawn, "sig_true"); em, _ = agg(drawn, "eps_true")
    table = stress_strain.build_plastic_table(round(ym, 1), round(sm, 0), round(em, 2), args.E)
    print(f"\n=== monotonic *PLASTIC table (true stress, true plastic strain; E={args.E:.0f} MPa) ===")
    print("*PLASTIC")
    for sig, ep in table:
        print(f"{sig:.1f}, {ep:.3f}")
    print(f"\neps_f (ductile) >= {em:.2f}  (stopped before break -> lower bound)")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        col = {"CD": "#F58025", "MD": "#264653"}
        for name, direction, strain, stress in curves:
            ax.plot(strain * 100, stress, lw=0.8, color=col.get(direction, "gray"),
                    alpha=0.85, label=direction)
        h, l = ax.get_legend_handles_labels()
        ax.legend(dict(zip(l, h)).values(), dict(zip(l, h)).keys(), title="direction")
        ax.set_xlabel("engineering strain [%]")
        ax.set_ylabel("engineering stress [MPa]")
        fig.tight_layout()
        os.makedirs(os.path.dirname(args.plot), exist_ok=True)
        fig.savefig(args.plot, dpi=150)
        print(f"\nplot -> {args.plot}")


if __name__ == "__main__":
    main()
