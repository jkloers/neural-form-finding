"""Do the tension routes of Fig. 6 run through one unit-cell family more than chance?

The families of Fig. 6 form spatial regions and the tension routes are contiguous chains of panels,
and by eye the two look related. That is a claim about SPATIAL COINCIDENCE, and it is not the claim
`docs/unit_cell_families_analysis.md` already bounded: that study reduced force onto each cell
individually and asked whether a family label predicts it. A region can coincide with a route while
no single cell's geometry predicts what that cell carries, so the two results are independent and
this one has to be measured on its own terms.

Nothing is re-solved. The five runs behind the figure each store their force field in
``load_path.npz``, and the routes come from ``force_stats.tension_routes``, which is the same object
``load_path_viz.draw_tension_routes`` draws -- so what this counts and what the plate shows cannot
disagree. The strong-link threshold is that module's own default, the MEAN transmitted force.

★ **The null is the whole difficulty.** Shuffling family labels over the lattice would destroy the
region structure the families are known to have (contiguity 1.11-1.40x chance), and a contiguous
route laid against a scattered label field would look significant for free. Instead the family grid
is CYCLICALLY SHIFTED on its own 7x5 torus: every region keeps its exact size and shape, and only
its alignment with the routes is destroyed. The 34 non-trivial shifts are the reference set, so a
per-sheet p cannot fall below 1/35 = 0.029.

Two statistics are reported because they answer slightly different questions and the coarser one is
the less sensitive: ``cells`` counts a unit cell once if any route touches it, ``tiles`` counts every
visit and so weights a route that runs along a region.

⚠ Five designs is the entire corpus of the figure. A relation that reverses sign between actuations
cannot be resolved at that n, and the output says so rather than reporting the pooled mean alone.

    JAX_PLATFORMS=cpu conda run -n kgnn_mac python \\
        nff/scripts/diagnostics/route_family_overlap.py
"""

import argparse
import csv
import json
import math
import os
from collections import defaultdict

import numpy as np

from nff.closed.cell_families import cell_lattice
from nff.closed.force_stats import tension_routes
from nff.closed.load_paths import interface_decomposition, load_field
from nff.closed.void_geometry import build_void_index
from nff.config.experiment import load_and_parse_config
from nff.topology.closed_builder_jax import build_deploy_structure

ROWS = [
    ("data/outputs/runs/run_20260806_161121_figmat_1_deploy",     "I"),
    ("data/outputs/runs/run_20260806_115352_figmat_2_cantilever", "II"),
    ("data/outputs/runs/run_20260806_161121_figmat_3_moment",     "III"),
    ("data/outputs/runs/run_20260806_161121_figmat_4_asym",       "IV"),
    ("data/outputs/runs/run_20260806_115352_figmat_5_vertical",   "V"),
]


def route_counts(run_dir: str, cell_of_face: np.ndarray, n_cells: int, min_run: int) -> dict:
    """How many times a tension route enters each unit cell, for one design."""
    field = load_field(os.path.join(run_dir, "load_path"))
    dec = interface_decomposition(field)
    keep = dec['f_norm'] >= float(dec['f_norm'].mean())
    network = tension_routes(field, dec, keep, min_run=min_run)

    counts = np.zeros(n_cells)
    for route in network['routes']:
        for face in route['node_path']:
            cell = cell_of_face[int(face)]
            if cell >= 0:
                counts[cell] += 1
    return {'counts': counts, 'n_routes': len(network['routes'])}


def _ratios(weight: np.ndarray, families: np.ndarray, n_families: int) -> np.ndarray:
    """Each family's share of the route weight over its share of the sheet. One means neutral."""
    total = weight.sum()
    if total <= 0:
        return np.full(n_families, np.nan)
    return np.array([(weight[families == c].sum() / total) / (families == c).mean()
                     if np.any(families == c) else np.nan for c in range(n_families)])


def _fisher(p_values: np.ndarray) -> tuple:
    """Fisher's combination, with the closed-form chi-square survival for even degrees of freedom."""
    statistic = float(-2.0 * np.log(np.clip(p_values, 1e-9, 1.0)).sum())
    k, x = len(p_values), statistic / 2.0
    survival = sum(x ** j / math.factorial(j) for j in range(k)) * math.exp(-x)
    return statistic, min(survival, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/outputs/cell_families")
    ap.add_argument("--min-run", type=int, default=3,
                    help="shortest run of panels that counts as a route, as the figure draws it")
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.corpus, "runs.json")))
    n_families = int(meta['n_families'])
    rows, cols = meta['lattice']

    families_by_label = defaultdict(dict)
    with open(os.path.join(args.corpus, "cells.csv")) as handle:
        for row in csv.DictReader(handle):
            families_by_label[row['label']][int(float(row['cell']))] = int(float(row['family']))

    config = load_and_parse_config(os.path.join(ROWS[0][0], "config.yaml"))
    M, N = int(config.topology['M']), int(config.topology['N'])
    lattice = cell_lattice(build_void_index(build_deploy_structure(M, N)))
    cell_of_face = np.full(M * N, -1, dtype=int)
    for c, faces in enumerate(lattice.owner_face):
        cell_of_face[faces] = c

    shifts = [(dr, dc) for dr in range(rows) for dc in range(cols)]
    for statistic in ("tiles", "cells"):
        print(f"\n── weighting: {statistic} "
              + ("(every visit counts)" if statistic == "tiles" else "(a cell counts once)"))
        print(f"  {'sheet':<6}{'routes':>7}   "
              + "".join(f"{'family ' + str(c):>20}" for c in range(n_families)))

        ratios = np.zeros((len(ROWS), n_families))
        p_values = np.zeros((len(ROWS), n_families))
        for i, (run_dir, label) in enumerate(ROWS):
            found = route_counts(run_dir, cell_of_face, lattice.n_cells, args.min_run)
            weight = found['counts'] if statistic == "tiles" else (found['counts'] > 0).astype(float)
            grid = np.array([families_by_label[label][c]
                             for c in range(lattice.n_cells)]).reshape(rows, cols)

            per_shift = np.stack([_ratios(weight, np.roll(grid, s, axis=(0, 1)).ravel(), n_families)
                                  for s in shifts])
            reference = np.median(per_shift[1:], axis=0)
            line = f"  {label:<6}{found['n_routes']:7d}   "
            for c in range(n_families):
                column = per_shift[:, c]
                # two-sided: how extreme the true labelling is among all shifts of the same map
                p = float((np.abs(column - reference[c]) >= abs(column[0] - reference[c])).mean())
                ratios[i, c], p_values[i, c] = column[0], p
                line += f"{column[0]:13.2f}  p={p:.3f}"
            print(line)

        print(f"  {'':<6}{'':>7}   " + "".join(
            f"{'mean ' + f'{ratios[:, c].mean():.2f}':>20}" for c in range(n_families)))
        for c in range(n_families):
            chi2, p = _fisher(p_values[:, c])
            verdict = "DEVIATES" if p < 0.05 else "consistent with chance"
            direction = ("one sign" if (ratios[:, c] > 1).all() or (ratios[:, c] < 1).all()
                         else "SIGN REVERSES between actuations")
            print(f"    family {c}: Fisher X2 = {chi2:5.2f} over {len(ROWS)} sheets, "
                  f"p = {p:.4f}  [{verdict}; {direction}]")

    print("\n  A ratio of one means the routes visit that family exactly as often as its share of\n"
          "  the sheet. Each p is two-sided against that sheet's own 34 cyclic shifts, so no\n"
          "  per-sheet p can fall below 0.029, and Fisher combines the five.")
    print("  ⚠ Five designs is the whole corpus of the figure. A relation whose sign depends on the\n"
          "    actuation cannot be established at that n, only reported.")


if __name__ == "__main__":
    main()
