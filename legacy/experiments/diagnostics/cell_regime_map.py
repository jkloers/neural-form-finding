"""The effective-medium regime of every unit cell, and what it does and does not explain.

Companion to ``cell_family_corpus.py`` / ``cell_family_map.py``. Those asked which force channel a
family carries and found essentially nothing. This asks the question Zheng, Niloy, Tobasco, Celli &
Plucinsky (Proc. R. Soc. A 479:20220665, 2023) say is the discriminating one: each cell has a shape
tensor, and the sign of its effective Poisson ratio decides whether an actuation disturbance decays
away from where it is applied (elliptic) or persists into the bulk (hyperbolic).

Everything here is physics-free -- the shape tensor comes from the exact rigid kinematics, so a
sheet needs only its ``config.yaml`` and ``best_params.pkl`` and no solve. The force quantities it
tests against are read from the corpus ``cells.csv``, which does need one.

Four tests, two of them pre-registered in ``docs/unit_cell_families_analysis.md`` before running:

    1  regime vs the sheet's measured opening-angle spread, and vs how far it deploys
    2  do the three families differ in the theory's own quantities
    3  PRE-REGISTERED: is departure from the mechanism concentrated at family boundaries
    4  does a neighbour mismatch in the shape tensor predict that departure

    JAX_PLATFORMS=cpu conda run -n kgnn_mac python \\
        legacy/experiments/diagnostics/cell_regime_map.py
"""

import argparse
import csv
import glob
import json
import os
import pickle
from collections import defaultdict

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp                                                  # noqa: E402
import numpy as np                                                       # noqa: E402

from nff.closed.cell_families import CellLattice, cell_lattice           # noqa: E402
from nff.closed.effective_medium import (boundary_cells, build_cell_frame,  # noqa: E402
                                         neighbour_mismatch, poisson_ratio,
                                         shape_tensor_drift, shape_tensor_parameters)
from nff.closed.force_stats import polynomial_detrend                    # noqa: E402
from nff.closed.setup import init_closed_les_params                      # noqa: E402
from nff.closed.shield_kinematics import build_deploy_plan               # noqa: E402
from nff.closed.void_geometry import build_void_index                    # noqa: E402
from nff.config.experiment import load_and_parse_config                  # noqa: E402
from legacy.experiments.diagnostics.cell_family_map import _sign_test, _spearman  # noqa: E402
from nff.topology.closed_builder_jax import (boundary_flat_from_logits,  # noqa: E402
                                             solve_cut_vertices_jax)


def measure_regime(run_dir: str, psi: float) -> dict:
    """Shape tensor, Poisson ratio and shear for every cell of one design, without any solve."""
    config = load_and_parse_config(os.path.join(run_dir, "config.yaml"))
    _, static = init_closed_les_params(config)
    saved = pickle.load(open(os.path.join(run_dir, "best_params.pkl"), "rb"))
    struct = static['struct']

    boundary = boundary_flat_from_logits(static['sliders'], jnp.asarray(saved['bnd_logits']))
    flat = solve_cut_vertices_jax(struct, boundary, jax.nn.sigmoid(jnp.asarray(saved['z'])))[
        np.asarray(struct['corner_pid'])]

    M, N = int(config.topology['M']), int(config.topology['N'])
    plan = build_deploy_plan(struct, M, N,
                             spacing=float(config.topology.get('spacing', 1.0)),
                             spacing_y=float(config.topology.get('spacing_y', 1.0)))
    lattice = cell_lattice(build_void_index(struct))
    frame = build_cell_frame(lattice)

    params = shape_tensor_parameters(plan, flat, lattice, frame, psi)
    drift = shape_tensor_drift(plan, flat, lattice, frame, [psi * 0.5, psi * 0.75, psi])
    return {'nu': np.asarray(poisson_ratio(plan, flat, lattice, frame, psi)),
            'alpha': params['alpha'], 'beta': params['beta'], 'defect': params['defect'],
            'shear': params['shear_deg'], 'lambda_1': params['lambda_1'],
            'lambda_2': params['lambda_2'], 'alpha_drift': drift['alpha_spread'], 'psi': psi}


def _by_family(values, families, xy, degree, n_families=3):
    """Median of a per-cell field within each family, after a spatial detrend."""
    z = polynomial_detrend(np.asarray(values, dtype=float), xy, degree=degree)
    return [float(np.median(z[families == c])) if np.any(families == c) else float('nan')
            for c in range(n_families)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/outputs/cell_families")
    ap.add_argument("--degrees", type=int, nargs="+", default=[0, 1, 2, 3])
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.corpus, "runs.json")))
    rows = []
    with open(os.path.join(args.corpus, "cells.csv")) as handle:
        for row in csv.DictReader(handle):
            for k, v in row.items():
                if k not in ("run", "label", "block"):
                    row[k] = float(v)
            rows.append(row)
    by_label = defaultdict(list)
    for row in rows:
        by_label[row['label']].append(row)
    for label in by_label:
        by_label[label].sort(key=lambda r: r['cell'])

    lattice_shape = tuple(meta['lattice'])
    n_cells = int(meta['n_cells'])
    lattice = CellLattice(keep=np.arange(n_cells), shape=lattice_shape,
                          owner_face=np.arange(4 * n_cells).reshape(n_cells, 4))

    regime = {}
    print(f"  {'sheet':<6}{'psi':>7}  {'nu_21 median':>13}{'%hyperbolic':>12}"
          f"{'defect med':>12}{'shear p90':>11}")
    for label, run in meta['runs'].items():
        r = measure_regime(run['run'], np.radians(run['psi_deg']))
        regime[label] = r
        print(f"  {label:<6}{run['psi_deg']:+7.1f}  {np.median(r['nu']):+13.3f}"
              f"{100 * (r['nu'] > 0).mean():11.1f}%{np.median(r['defect']):+12.3f}"
              f"{np.percentile(np.abs(r['shear']), 90):11.1f}", flush=True)

    pooled_nu = np.concatenate([r['nu'] for r in regime.values()])
    pooled_shear = np.abs(np.concatenate([r['shear'] for r in regime.values()]))
    print(f"\n  pooled {pooled_nu.size} cells: nu_21 median {np.median(pooled_nu):+.3f}, "
          f"hyperbolic {100 * (pooled_nu > 0).mean():.1f}%; "
          f"lattice shear median {np.median(pooled_shear):.1f} deg, "
          f"p90 {np.percentile(pooled_shear, 90):.1f} deg")
    print("  ⚠ Zheng's Eq. 4 shape tensor is DIAGONAL. That shear is the part it has no slot for.")

    labels = list(by_label)
    hyper = [float((regime[l]['nu'] > 0).mean()) for l in labels]
    spread = [meta['runs'][l]['opening_spread_deg'] for l in labels]
    opening = [abs(regime[l]['psi']) for l in labels]
    print(f"\nTEST 1 -- regime against what the sheet actually did, over {len(labels)} sheets")
    print(f"  Spearman(hyperbolic fraction, opening-angle spread) = {_spearman(hyper, spread):+.3f}")
    print(f"  Spearman(hyperbolic fraction, |psi|)                = {_spearman(hyper, opening):+.3f}"
          "    <- deploying further drives a sheet hyperbolic")

    print("\nTEST 2 -- do the families differ in the theory's own quantities?")
    for key in ('defect', 'nu', 'shear'):
        for degree in args.degrees:
            per_sheet = []
            for label in labels:
                fam = np.array([int(r['family']) for r in by_label[label]])
                xy = np.array([[r['row'], r['col']] for r in by_label[label]], dtype=float)
                value = np.clip(regime[label][key], -5, 5) if key == 'nu' else regime[label][key]
                per_sheet.append(_by_family(value, fam, xy, degree))
            m = np.array(per_sheet)
            flags = []
            for a in range(3):
                for b in range(a + 1, 3):
                    p, k, n = _sign_test(m[:, a] - m[:, b])
                    if p < 0.05:
                        flags.append(f"{a}v{b} p={p:.4f} ({k}/{n})")
            print(f"  {key:<8} deg{degree}  family medians {np.nanmedian(m, axis=0).round(3)}"
                  + ("   " + "  ".join(flags) if flags else "   -"))

    print("\nTEST 3 -- PRE-REGISTERED: is departure from the mechanism at family boundaries?")
    for response in ('resid_absmax', 'energy_sum'):
        diffs = []
        for label in labels:
            fam = np.array([int(r['family']) for r in by_label[label]])
            value = np.abs(np.array([r[response] for r in by_label[label]]))
            xy = np.array([[r['row'], r['col']] for r in by_label[label]], dtype=float)
            z = polynomial_detrend(value, xy, degree=1)
            edge = boundary_cells(lattice, fam)
            if edge.all() or not edge.any():
                continue
            diffs.append(float(z[edge].mean() - z[~edge].mean()))
        p, k, n = _sign_test(diffs)
        print(f"  {response:<14} boundary - interior: median {np.median(diffs):+.4g}, "
              f"{k}/{n} sheets positive, p={p:.4f}")

    print("\nTEST 4 -- does a neighbour mismatch in the shape tensor predict that departure?")
    for key in ('lambda_1', 'lambda_2', 'nu'):
        for degree in (0, 1, 2):
            rhos = []
            for label in labels:
                mismatch = neighbour_mismatch(lattice, regime[label][key])
                value = np.abs(np.array([r['resid_absmax'] for r in by_label[label]]))
                xy = np.array([[r['row'], r['col']] for r in by_label[label]], dtype=float)
                rhos.append(_spearman(mismatch, polynomial_detrend(value, xy, degree=degree)))
            p, k, n = _sign_test(rhos)
            star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"  mismatch({key:<8}) deg{degree}: rho={np.median(rhos):+.3f}{star:<4} "
                  f"{k}/{n} positive, p={p:.4f}")

    out = os.path.join(args.corpus, "regime.json")
    json.dump({l: {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()}
               for l, r in regime.items()}, open(out, "w"))
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
