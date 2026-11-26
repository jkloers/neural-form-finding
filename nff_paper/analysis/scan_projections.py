"""Which plane shows both the families AND the landscape?

Two planes are already known and each fails one half of the job. The leading pair of principal
components of the converged designs carries the objective faithfully, so it can hold iso-loss
contours, but it does not separate the starting designs from family B. A discriminant plane
separates all three groups and carries no landscape at all: the objective on it is nothing like
the objective at the designs.

This scans pairs of principal components for one that does both, scoring every candidate on

    separation   the WEAKEST of the three group pairings, so a plane wins only if it also
                 separates the pairing that currently fails
    fidelity     how far a design's own objective is from the objective at its projected
                 position, the quantity the figure gates its contours on

    JAX_PLATFORMS=cpu conda run -n kgnn_mac python \
        nff_paper/analysis/scan_projections.py \
        --run data/outputs/design_cloud/circle_7x7_n200_s1
"""

import argparse
import json
import os
import sys

import jax

jax.config.update("jax_enable_x64", True)

import jax.flatten_util                                              # noqa: E402
import jax.numpy as jnp                                              # noqa: E402
import numpy as np                                                   # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from nff_paper.closed_support.analytic_targets import circle_outline_unit          # noqa: E402
from nff_paper.closed_support.design_ensemble import (build_symmetry, circle_problem,  # noqa: E402
                                        distance_matrix, load_ensemble, problem_statics,
                                        split_families)
from nff_paper.closed_support.shield_design import _interior_cut_bases, build_loss  # noqa: E402


def pca_basis(vectors):
    """Mean and the full ordered set of principal directions, with their variances."""
    X = np.asarray(vectors, dtype=float)
    mean = X.mean(axis=0)
    _, S, Vt = np.linalg.svd(X - mean, full_matrices=False)
    return mean, Vt, (S ** 2) / max(len(X) - 1, 1)


def separation(coords, groups):
    """The weakest of the pairwise group separations in a plane.

    Each pairing is scored along its OWN centroid axis: the distance between the two means
    divided by the spread of each group along that direction. Taking the minimum is what
    makes the score demanding, since a plane that splits the families but leaves the starts
    sitting on top of one of them has not solved the problem being posed.
    """
    labels = np.unique(groups)
    worst = np.inf
    per_pair = {}
    for a in range(len(labels)):
        for b in range(a + 1, len(labels)):
            ga, gb = coords[groups == labels[a]], coords[groups == labels[b]]
            delta = gb.mean(axis=0) - ga.mean(axis=0)
            dist = float(np.linalg.norm(delta))
            if dist < 1e-12:
                per_pair[f"{labels[a]}|{labels[b]}"] = 0.0
                worst = 0.0
                continue
            axis = delta / dist
            sa = float(np.std(ga @ axis))
            sb = float(np.std(gb @ axis))
            score = dist / max(sa + sb, 1e-12)
            per_pair[f"{labels[a]}|{labels[b]}"] = score
            worst = min(worst, score)
    return float(worst), per_pair


def fidelity(loss_at, mean, basis_pair, vectors, true_loss):
    """Median relative gap between a design's objective and the objective at its projection."""
    proj = mean[None, :] + ((vectors - mean) @ basis_pair.T) @ basis_pair
    on_plane = loss_at(proj)
    return float(np.median(np.abs(on_plane - true_loss) / true_loss))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--n-components", type=int, default=15)
    ap.add_argument("--max-gap", type=float, default=0.10,
                    help="a plane is only usable for contours below this fidelity gap")
    args = ap.parse_args()

    records = load_ensemble(args.run)
    with open(os.path.join(args.run, "summary.json")) as fh:
        meta = json.load(fh)["config"]
    cfg = circle_problem(M=meta["M"], N=meta["N"], sheet_mm=meta["sheet_mm"],
                         radius_mm=meta["radius_mm"])
    statics = problem_statics(cfg)
    group = build_symmetry(cfg, statics)
    loss_fn = build_loss(statics["forward_fn"], circle_outline_unit(480), statics["plan"],
                         cfg, _interior_cut_bases(statics["struct"]))

    labels, _ = split_families(distance_matrix([r["coords"] for r in records], group))
    finals = np.stack([r["traj_params"][-1] for r in records])
    starts = np.stack([r["traj_params"][0] for r in records])

    sample = {k[len("params__"):]: jnp.asarray(v)
              for k, v in records[0].items() if k.startswith("params__")}
    _, unravel = jax.flatten_util.ravel_pytree(sample)
    ev = jax.jit(jax.vmap(lambda vv: loss_fn(unravel(vv), 1.0)[0]))

    def loss_at(vecs):
        out = []
        for i in range(0, len(vecs), 512):
            out.append(np.asarray(ev(jnp.asarray(vecs[i:i + 512]))))
        return np.concatenate(out)

    true_loss = loss_at(finals)
    # starts = 0, family A = 1, family B = 2
    all_vec = np.vstack([starts, finals])
    all_grp = np.concatenate([np.zeros(len(starts), dtype=int), labels + 1])

    rows = []
    for basis_name, source in (("finals", finals), ("all", all_vec)):
        mean, V, var = pca_basis(source)
        k = min(args.n_components, V.shape[0])
        frac = var / var.sum()
        print(f"\n  basis '{basis_name}': first {k} components hold "
              f"{100 * frac[:k].sum():.1f}% of the spread")
        for i in range(k):
            for j in range(i + 1, k):
                pair = V[[i, j]]
                coords = (all_vec - mean) @ pair.T
                sep, per_pair = separation(coords, all_grp)
                gap = fidelity(loss_at, mean, pair, finals, true_loss)
                rows.append({"basis": basis_name, "i": int(i), "j": int(j),
                             "separation": sep, "gap": gap,
                             "var_frac": float(frac[i] + frac[j]),
                             "pairs": per_pair})

    usable = [r for r in rows if r["gap"] <= args.max_gap]
    usable.sort(key=lambda r: -r["separation"])
    rows.sort(key=lambda r: -r["separation"])

    def show(title, items, n=10):
        print(f"\n  ── {title} ─────────────────────────────────────")
        print(f"    {'basis':>7} {'pair':>9} {'separation':>11} {'gap':>8} {'var':>7}   "
              f"{'starts|A':>9} {'starts|B':>9} {'A|B':>7}")
        for r in items[:n]:
            p = r["pairs"]
            tag = f"PC{r['i']}-PC{r['j']}"
            print(f"    {r['basis']:>7} {tag:>9} "
                  f"{r['separation']:>11.2f} {r['gap']:>8.3f} {100*r['var_frac']:>6.1f}%   "
                  f"{p['0|1']:>9.2f} {p['0|2']:>9.2f} {p['1|2']:>7.2f}")

    show(f"best separation among planes with gap <= {args.max_gap:g}", usable)
    show("best separation overall, fidelity ignored", rows)

    # What the two known planes score, for comparison.
    mean_f, Vf, varf = pca_basis(finals)
    base_pair = Vf[[0, 1]]
    base_sep, base_pairs = separation((all_vec - mean_f) @ base_pair.T, all_grp)
    base_gap = fidelity(loss_at, mean_f, base_pair, finals, true_loss)
    print(f"\n  the plane in use now (finals PC0-PC1): separation {base_sep:.2f}, "
          f"gap {base_gap:.3f}")
    if usable:
        b = usable[0]
        print(f"  best usable candidate:            {b['basis']} PC{b['i']}-PC{b['j']}, "
              f"separation {b['separation']:.2f}, gap {b['gap']:.3f}")
        print(f"  -> {'BETTER' if b['separation'] > base_sep else 'no better'} than the "
              f"plane in use")
    else:
        print(f"  no plane reaches the {args.max_gap:g} fidelity bar")

    out = os.path.join(args.run, "projection_scan.json")
    with open(out, "w") as fh:
        json.dump({"max_gap": args.max_gap, "baseline": {"separation": base_sep,
                                                         "gap": base_gap,
                                                         "pairs": base_pairs},
                   "rows": rows}, fh, indent=2)
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
