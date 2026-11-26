"""Accuracy-vs-cost Pareto for the surrogate capacity sweep.

The pairing is the whole point of the figure: held-out error alone would say "take the biggest net",
and parameter count alone would say "take the smallest". What decides it is that Stage-2 pays the
Hessian-vector product on the inner CG loop of the implicit-diff backward solve, so the cost axis is
measured microseconds, not parameters.

Left panel is the Pareto front (error vs hvp). Right panel is the seed spread, when the sweep was
run with more than one seed -- without it a 0.5 pt gap between two arms cannot be read as real.

    conda run -n kgnn_mac python -m legacy.experiments.figures.plot_surrogate_capacity \
        --sweep data/surrogates/capacity/sweep_energy.json --out data/surrogates/capacity/pareto.png
"""
import argparse
import json
from collections import defaultdict

import numpy as np

from nff.utils.figstyle import GREY, INK, ORANGE, TEAL, apply_charter, use_agg


def _arch_label(row):
    return f"{'x'.join(str(h) for h in row['hidden'])}·{row['m_out']}"


def build_figure(sweep, out_path, *, metric="test"):
    use_agg()
    import matplotlib.pyplot as plt
    apply_charter()

    arms = [a for a in sweep["arms"] if a.get(f"{metric}_metrics")]
    by_arch = defaultdict(list)
    for a in arms:
        by_arch[(tuple(a["hidden"]), a["m_out"])].append(a)

    xs, ys, yerr, labels, n_par = [], [], [], [], []
    for key, group in sorted(by_arch.items(), key=lambda kv: group_cost(kv[1])):
        errs = [g[f"{metric}_metrics"]["energy_rel"] * 100 for g in group]
        xs.append(group_cost(group))
        ys.append(float(np.mean(errs)))
        yerr.append(float(np.std(errs)) if len(errs) > 1 else 0.0)
        labels.append(_arch_label(group[0]))
        n_par.append(group[0]["cost"]["n_params_energy"])

    multi_seed = any(e > 0 for e in yerr)
    fig, axes = plt.subplots(1, 2 if multi_seed else 1, figsize=(11 if multi_seed else 6.2, 4.4))
    ax = axes[0] if multi_seed else axes

    ax.errorbar(xs, ys, yerr=yerr if multi_seed else None, fmt="o", ms=7, color=TEAL,
                ecolor=GREY, elinewidth=1.1, capsize=3, zorder=3)
    ax.axhline(5.0, color=ORANGE, lw=1.2, ls="--", zorder=1)
    ax.annotate("target", xy=(max(xs), 5.0), xytext=(-4, 4), textcoords="offset points",
                ha="right", va="bottom", color=ORANGE, fontsize=8)
    for x, y, lab in zip(xs, ys, labels):
        ax.annotate(lab, (x, y), xytext=(0, -12), textcoords="offset points",
                    ha="center", fontsize=7.5, color=INK)
    ax.set_xscale("log")
    ax.set_xlabel("Hessian-vector product  [µs]")
    ax.set_ylabel(f"held-out energy relRMSE  [%]  ({metric})")
    ax.set_ylim(bottom=0)

    if multi_seed:
        ax2 = axes[1]
        for i, (key, group) in enumerate(sorted(by_arch.items(), key=lambda kv: group_cost(kv[1]))):
            errs = [g[f"{metric}_metrics"]["energy_rel"] * 100 for g in group]
            ax2.scatter([i] * len(errs), errs, s=26, color=TEAL, alpha=0.75, zorder=3)
            ax2.plot([i - 0.22, i + 0.22], [np.mean(errs)] * 2, color=INK, lw=1.6, zorder=4)
        ax2.axhline(5.0, color=ORANGE, lw=1.2, ls="--", zorder=1)
        ax2.set_xticks(range(len(labels)))
        ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=7.5)
        ax2.set_ylabel(f"energy relRMSE  [%]  ({metric})")
        ax2.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"wrote {out_path}")
    return fig


def group_cost(group):
    return float(np.mean([g["cost"]["hvp_us"] for g in group]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--metric", default="test", choices=["test", "val"])
    args = ap.parse_args()
    with open(args.sweep) as f:
        sweep = json.load(f)
    build_figure(sweep, args.out, metric=args.metric)


if __name__ == "__main__":
    main()
