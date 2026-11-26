"""Merge sharded :mod:`audit_path_dependence` reports into one distribution.

The audit runs one seed range per process so a fan-out can use every core. Each shard writes its own
JSON; this concatenates their ``pairs`` and recomputes the summary over the pooled set.

Shards are keyed by seed, and a seed identifies a path, so a seed appearing in two shards would be
the same measurement counted twice. Duplicates are dropped rather than averaged, and reported.

    conda run -n kgnn_mac python -m nff.scripts.surrogate.merge_path_dependence \
        data/surrogates/path_dependence_shards/*.json \
        --out data/surrogates/path_dependence_pet_v3.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from nff.scripts.surrogate.audit_path_dependence import summarise

BEND_EDGES = (0.0, 0.002, 0.01, 0.05, 0.15, 0.40, np.inf)   # ligament widths

# nonproportionality() divides by sqrt(mean(x^2)) + 1e-3*w_lig. On a hinge that barely translates
# the floor term exceeds the signal, so the departure it returns is noise rather than a bent route.
# Such a route IS a ray -- a pure-rotation one -- and it should be read as the experiment's own null
# control, not mixed into the relation. 8.9% of the w18 prior's moving routes sit below this.
MIN_BEND = 2e-3      # ligament widths; below this the two legs are the same route


def enrich_with_translation(pairs, prior, w_lig, resample):
    """Add ``translation`` and ``bend`` to each pair by rebuilding its path from its seed.

    ``pick_path`` is deterministic in the seed, so nothing has to be stored at solve time and a
    campaign already running can still be analysed this way.

    ``bend`` is the quantity the relation should be read against. The two legs of a pair are two
    state lists of the same length, the polyline and ``(k/n) * u_end``, so the largest separation
    between them IS how far apart the routes ran, in ligament widths:

        bend = max_k || (a, s)_k - (k/n) (a, s)_end || / w_lig

    That is dimensional, unlike the campaign's ``nonproportionality``, which divides the residual by
    the signal and so reports a large departure for a hinge whose translation is 24 micrometres on an
    18 mm ligament. Such a route is straight by any physical reading, and only an absolute measure
    says so.
    """
    from nff.scripts.surrogate.replay_hinge_path import pick_path, resample_polyline
    for p in pairs:
        try:
            poly, _, _ = pick_path(prior, int(p["seed"]))
            if resample:
                poly = resample_polyline(poly, resample, w_lig)
            st = poly[1:]
            n = len(st)
            ray = np.outer(np.arange(1, n + 1) / n, st[-1])
            p["translation"] = float(max(np.abs(st[:, 0]).max(), np.abs(st[:, 1]).max()) / w_lig)
            p["bend"] = float(np.linalg.norm(st[:, :2] - ray[:, :2], axis=-1).max() / w_lig)
        except Exception:
            p["translation"] = p["bend"] = float("nan")
    return pairs


def departure_relation(pairs, edges=BEND_EDGES):
    """Energy ratio against how far the route departs from the ray it is compared with.

    A pooled median answers "how big is the gap", which is worth little on its own: the pairs are
    drawn from a prior whose routes span nearly straight to strongly bent. The relation answers the
    question the hypothesis actually poses, and carries its own control. The gap must approach one
    as the bend approaches zero, because there the two routes coincide. A set that shows a gap at
    zero bend is measuring solver noise or a mismatched endpoint, not path dependence.

    Read against ``bend`` and not the campaign's ``nonproportionality``: see
    :func:`enrich_with_translation` for why a relative residual mislabels an almost stationary hinge.
    """
    d = np.array([p.get("bend", np.nan) for p in pairs], float)
    r = np.array([p.get("W_ratio", np.nan) for p in pairs], float)
    ok = np.isfinite(d) & np.isfinite(r)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = ok & (d >= lo) & (d < hi)
        if not m.any():
            continue
        out.append(dict(lo=float(lo), hi=(None if not np.isfinite(hi) else float(hi)),
                        n=int(m.sum()), median_W_ratio=float(np.median(r[m])),
                        min_W_ratio=float(r[m].min()), max_W_ratio=float(r[m].max())))
    rec = dict(bands=out, n_used=int(ok.sum()))
    if ok.sum() >= 3:
        rec["spearman"] = float(_spearman(d[ok], r[ok]))

    # The pure-rotation routes, held out of the relation and reported as the null control.
    null = np.isfinite(r) & np.isfinite(d) & (d < MIN_BEND)
    if null.any():
        rec["null_control"] = dict(
            n=int(null.sum()), max_bend=MIN_BEND,
            median_W_ratio=float(np.median(r[null])),
            min_W_ratio=float(r[null].min()), max_W_ratio=float(r[null].max()))
    return rec


def _spearman(x, y):
    """Rank correlation, so a monotone relation is not judged by how straight it is."""
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return (rx * ry).sum() / denom if denom else np.nan


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("shards", nargs="+", help="shard JSON paths")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pairs, skipped, seen, dupes, priors, geoms = [], [], set(), [], set(), []
    for path in args.shards:
        if not os.path.exists(path):
            print(f"  missing, skipping: {path}")
            continue
        with open(path) as f:
            rec = json.load(f)
        priors.add(rec.get("prior"))
        geoms.append(rec.get("geometry"))
        for p in rec.get("pairs", []):
            if p["seed"] in seen:
                dupes.append(p["seed"])
                continue
            seen.add(p["seed"])
            pairs.append(p)
        skipped.extend(rec.get("skipped", []))

    if len(priors) > 1:
        raise SystemExit(f"shards disagree on the prior: {priors}. Pooling them would mix "
                         "two path populations into one distribution.")
    if any(g != geoms[0] for g in geoms):
        raise SystemExit("shards disagree on the hinge geometry; the ratios are not comparable.")

    pairs.sort(key=lambda p: p["seed"])
    g = geoms[0] or {}
    prior = next(iter(priors)) if priors else None
    if prior:
        enrich_with_translation(pairs, prior, g.get("w_lig", 18.0), g.get("resample"))
    by_departure = departure_relation(pairs)
    out = dict(prior=prior, n_shards=len(args.shards),
               n_compared=len(pairs), n_skipped=len(skipped), skipped=skipped,
               n_duplicate_seeds=len(dupes), geometry=geoms[0] if geoms else None,
               summary=summarise(pairs), by_departure=by_departure, pairs=pairs)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print(f"merged {len(args.shards)} shards -> {len(pairs)} pairs, {len(skipped)} skipped"
          + (f", {len(dupes)} duplicate seeds dropped" if dupes else ""))
    for k, v in out["summary"].items():
        if v:
            print(f"  {k:14s} n {v['n']:3d}  median {v['median']:.3f}  "
                  f"IQR [{v['p25']:.3f}, {v['p75']:.3f}]  p90 {v['p90']:.3f}  "
                  f"range [{v['min']:.3f}, {v['max']:.3f}]")
    if by_departure["bands"]:
        print("\n  energy ratio against BEND away from the compared ray, in ligament widths "
              "(must approach 1 as bend -> 0):")
        for b in by_departure["bands"]:
            hi = "inf" if b["hi"] is None else f"{b['hi']:.2f}"
            print(f"    {b['lo']:.2f}-{hi:>4s}  n {b['n']:3d}  median {b['median_W_ratio']:.3f}  "
                  f"range [{b['min_W_ratio']:.3f}, {b['max_W_ratio']:.3f}]")
        if "spearman" in by_departure:
            print(f"    rank correlation {by_departure['spearman']:+.3f}")
    nc = by_departure.get("null_control")
    if nc:
        print(f"\n  null control, routes whose bend is under {nc['max_bend']:.3f} ligament "
              f"widths and which are therefore already rays (n {nc['n']}): "
              f"median ratio {nc['median_W_ratio']:.4f}, "
              f"range [{nc['min_W_ratio']:.4f}, {nc['max_W_ratio']:.4f}]")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
