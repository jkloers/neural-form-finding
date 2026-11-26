"""Which unit-cell family an actuation recruits, and which force channel each family carries.

Two questions, two admissible corpora, and the difference matters more than either answer.

**Arm A -- recruitment.** Does the composition of a sheet depend on how it is actuated? This is a
BETWEEN-sheet comparison, so it is licensed only on the figmat block, where the target is held
byte-identical and the start is uniform. The null is that the five sheets are exchangeable AT EACH
SITE: for every one of the 35 cell sites the five sheets' family labels are shuffled among the five
sheets. That null preserves the spatial pattern exactly -- a label never moves to a different site
-- so the test is not inflated by the spatial coherence the families are already known to have. A
label-shuffling null over all cells at once would be, and would report significance for free.

**Arm B -- mechanism.** Within one sheet, does a family carry more of one channel than another?
This is a WITHIN-sheet comparison. Where the sheet's target sat cannot bias it, so all fifteen
sheets are admissible, and each is one independent replicate. Two guards are applied before any
family mean is read:

    detrend   a low-order polynomial in the cell's lattice position is removed first. A whole-sheet
              aggregate has already reported the opposite of the visible structure three times on
              this figure; the guard is not optional.
    within    every response is standardised INSIDE its own sheet, so a sheet that simply carries
              more force cannot masquerade as a family effect.

The family means are then compared across the fifteen sheets by a paired sign test, which assumes
nothing about the distribution of a force and treats each sheet as the replicate it is.

    JAX_PLATFORMS=cpu conda run -n kgnn_mac python \\
        legacy/experiments/diagnostics/cell_family_map.py
"""

import argparse
import csv
import json
import os
from collections import defaultdict
from math import comb

import numpy as np

from nff.closed.force_stats import polynomial_detrend

PREDICTORS = ["gain", "span_mm", "aspect", "long_mm", "short_mm"]

RESPONSES = [
    ("f_n_mean", "normal force (+ tension)"),
    ("f_t_absmax", "tangential force"),
    ("f_mag_mean", "transmitted force"),
    ("shear_frac", "shear-dominated share"),
    ("pressure_mean", "tile pressure"),
    ("couple_mean", "tile couple"),
    ("energy_sum", "stored hinge energy"),
    ("theta_absmax", "panel rotation"),
    ("resid_absmax", "departure from 1-DOF"),
]


def _read(path):
    rows = []
    with open(path) as handle:
        for row in csv.DictReader(handle):
            for k, v in row.items():
                if k not in ("run", "label", "block"):
                    row[k] = float(v)
            rows.append(row)
    return rows


def _by_label(rows):
    out = defaultdict(list)
    for r in rows:
        out[r['label']].append(r)
    for label in out:
        out[label].sort(key=lambda r: r['cell'])
    return out


def _chi2(table):
    """Pearson's statistic of a contingency table, used only as a permutation statistic."""
    table = np.asarray(table, dtype=float)
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / max(table.sum(), 1e-12)
    return float((((table - expected) ** 2) / np.maximum(expected, 1e-12)).sum())


def _sign_test(differences):
    """Two-sided exact sign test on paired differences; ties are dropped."""
    d = np.asarray(differences, dtype=float)
    d = d[np.isfinite(d) & (d != 0.0)]
    n, k = len(d), int((d > 0).sum())
    if n == 0:
        return float('nan'), 0, 0
    tail = min(k, n - k)
    p = 2.0 * sum(comb(n, i) for i in range(tail + 1)) / 2.0 ** n
    return min(p, 1.0), k, n


def arm_a(by_label, blocks, n_families, n_perm, rng):
    """Composition against actuation, on the controlled-target block only."""
    labels = [l for l in by_label if blocks[l] == 'figmat']
    grids = np.stack([[int(r['family']) for r in by_label[l]] for l in labels])   # (n_sheets, n_cells)
    table = np.stack([np.bincount(g, minlength=n_families) for g in grids])
    observed = _chi2(table)

    # Exchangeable AT EACH SITE: shuffle the column of labels standing over one cell site.
    count = 0
    for _ in range(n_perm):
        shuffled = np.take_along_axis(
            grids, rng.permuted(np.tile(np.arange(len(labels))[:, None], grids.shape[1]), axis=0), 0)
        count += _chi2(np.stack([np.bincount(g, minlength=n_families)
                                 for g in shuffled])) >= observed
    return {'labels': labels, 'table': table.tolist(), 'chi2': observed,
            'p': (count + 1) / (n_perm + 1), 'n_perm': n_perm}


def arm_b(by_label, n_families, degree):
    """Family against force channel, within each sheet, detrended and standardised."""
    out = {}
    for key, name in RESPONSES:
        per_sheet = {}
        for label, rows in by_label.items():
            values = np.array([r[key] for r in rows], dtype=float)
            fam = np.array([int(r['family']) for r in rows])
            xy = np.array([[r['row'], r['col']] for r in rows], dtype=float)
            good = np.isfinite(values)
            if good.sum() < len(values) * 0.5:
                continue
            values = np.where(good, values, np.nanmean(values[good]))
            resid = polynomial_detrend(values, xy, degree=degree)
            spread = resid.std()
            z = resid / spread if spread > 1e-12 else resid
            per_sheet[label] = [float(z[fam == c].mean()) if np.any(fam == c) else float('nan')
                                for c in range(n_families)]
        means = np.array(list(per_sheet.values()))
        pairs = {}
        for a in range(n_families):
            for b in range(a + 1, n_families):
                p, k, n = _sign_test(means[:, a] - means[:, b])
                pairs[f"{a}-{b}"] = {'p': p, 'n_positive': k, 'n_sheets': n,
                                     'median_gap': float(np.nanmedian(means[:, a] - means[:, b]))}
        out[key] = {'name': name, 'per_sheet': per_sheet,
                    'family_mean': np.nanmean(means, axis=0).tolist(),
                    'pairs': pairs}
    return out


def _spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)).astype(float), np.argsort(np.argsort(b)).astype(float)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else float('nan')


def arm_d(by_label, degrees):
    """The CONTINUOUS arm: does a cell's own geometry predict what it carries?

    The three families are a coarsening of a two-parameter descriptor, and a coarsening costs
    power, so the same question is asked of the descriptor itself. Within each sheet the response
    is detrended and ranked against each geometric predictor; the sign of that correlation is then
    tested across the fifteen sheets.

    The detrend order is SWEPT rather than chosen. The families form spatial regions, so a
    high-order detrend in lattice position can remove the very variation being tested; reporting
    one order alone would leave that indistinguishable from a physical null.

    ``area_mm2`` against ``gain`` is not a result but the wiring check: the closed form makes it an
    identity up to the sheet's opening angle, so it must come back near one at degree zero.
    """
    out = {}
    for degree in degrees:
        table = {}
        for pred in PREDICTORS:
            for resp, _ in RESPONSES + [("area_mm2", "void area")]:
                rhos = []
                for rows in by_label.values():
                    x = np.array([r[pred] for r in rows], dtype=float)
                    y = np.array([r[resp] for r in rows], dtype=float)
                    xy = np.array([[r['row'], r['col']] for r in rows], dtype=float)
                    if not np.isfinite(y).all() or not np.isfinite(x).all():
                        continue
                    rhos.append(_spearman(x, polynomial_detrend(y, xy, degree=degree)))
                p, k, n = _sign_test(rhos)
                table[f"{resp}|{pred}"] = {'median_rho': float(np.median(rhos)), 'p': p,
                                           'n_positive': k, 'n_sheets': n}
        out[str(degree)] = table
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/outputs/cell_families")
    ap.add_argument("--n-perm", type=int, default=20000)
    ap.add_argument("--degree", type=int, default=2, help="spatial detrend order for Arm B")
    ap.add_argument("--degrees", type=int, nargs="+", default=[0, 1, 2, 3],
                    help="detrend orders swept by the continuous arm")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = _read(os.path.join(args.corpus, "cells.csv"))
    meta = json.load(open(os.path.join(args.corpus, "runs.json")))
    by_label = _by_label(rows)
    blocks = {l: rs[0]['block'] for l, rs in by_label.items()}
    n_families = int(meta['n_families'])
    centroids = np.asarray(meta['family_centroids_mm'])
    rng = np.random.default_rng(args.seed)

    print("families (mm), and the closed-form gain r(1-r)L^2 of each centroid:")
    for c in range(n_families):
        long_side, short = centroids[c]
        span = long_side + short
        print(f"  family {c}: {long_side:.0f} x {short:.0f}   L={span:.0f} mm   r={short/span:.3f}"
              f"   gain={short * long_side / 1e3:.1f} x10^3 mm^2")

    a = arm_a(by_label, blocks, n_families, args.n_perm, rng)
    print(f"\nARM A -- composition vs actuation, on {len(a['labels'])} controlled-target sheets")
    print("  sheet   " + "  ".join(f"fam{c}" for c in range(n_families)))
    for label, counts in zip(a['labels'], a['table']):
        print(f"  {label:<7} " + "  ".join(f"{n:4d}" for n in counts))
    print(f"  chi2 = {a['chi2']:.2f},  p = {a['p']:.4f} "
          f"(within-site permutation, {a['n_perm']} draws)")

    b = arm_b(by_label, n_families, args.degree)
    print(f"\nARM B -- family vs channel, within {len(by_label)} sheets, "
          f"degree-{args.degree} detrend, standardised per sheet")
    print(f"  {'response':<26}" + "".join(f"{'fam'+str(c):>8}" for c in range(n_families))
          + "   significant pairs (sign test, p<0.05)")
    for key, res in b.items():
        sig = [f"{pair} p={v['p']:.3f} d={v['median_gap']:+.2f}"
               for pair, v in res['pairs'].items() if v['p'] < 0.05]
        print(f"  {res['name']:<26}" + "".join(f"{m:+8.2f}" for m in res['family_mean'])
              + "   " + ("  ".join(sig) if sig else "-"))

    d = arm_d(by_label, args.degrees)
    print(f"\nARM D -- cell geometry vs channel, {len(by_label)} sheets, median within-sheet "
          f"Spearman\n  (stars: paired sign test on the sign across sheets; "
          f"* p<0.05  ** p<0.01  *** p<0.001)")
    for degree in args.degrees:
        table = d[str(degree)]
        print(f"\n  detrend degree {degree}")
        print(f"    {'response':<26}" + "".join(f"{p:>14}" for p in PREDICTORS))
        for resp, name in RESPONSES + [("area_mm2", "void area (wiring check)")]:
            line = f"    {name:<26}"
            for pred in PREDICTORS:
                cell = table[f"{resp}|{pred}"]
                star = ("***" if cell['p'] < 0.001 else "**" if cell['p'] < 0.01
                        else "*" if cell['p'] < 0.05 else "")
                line += f"{cell['median_rho']:+8.2f}{star:<6}"
            print(line)

    out = {'arm_a': a, 'arm_b': b, 'arm_d': d, 'n_families': n_families,
           'family_centroids_mm': centroids.tolist(),
           'blocks': blocks, 'degree': args.degree}
    path = os.path.join(args.corpus, "family_map.json")
    json.dump(out, open(path, "w"), indent=1, default=float)
    print(f"\n  wrote {path}")


if __name__ == "__main__":
    main()
