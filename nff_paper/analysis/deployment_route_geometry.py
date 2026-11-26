"""How far the routes a deployed sheet rides depart from proportional ones.

The condensed energy is fitted as a state function, which is exact when every route is a ray from
the origin (Sec. sec:path-dependence). Whether that exactness is worth anything depends on the
routes the sheet actually rides, and those are harvested, not assumed.

Each hinge of each harvested deployment gives one polyline in ``(a, s, theta)``. This scores every
one of them with :func:`nff.utils.routes.nonproportionality`, the residual fraction of the best ray
through the origin: ``0`` is a perfect ray and larger is further from one. It reports the
distribution, not a threshold count, because the threshold that splits campaign jobs into families
is a sampling convention and carries no meaning for a deployment.

⚠ The priors are harvested under whichever hinge model was in the Stage-2 loop. Under the reduced
model of Sec. sec:rom the linear ``k_rot`` is a secant fitted to the buckled response, so it
under-prices the first few degrees, spends rotation early, and yields rotation-first routes. Read
the numbers below as an UPPER bound on the departure a surrogate-in-the-loop deployment would ride.

    conda run -n kgnn_mac python -m nff_paper.analysis.deployment_route_geometry
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from nff.utils.routes import DEF_TOL, nonproportionality

PRIORS = ("data/fea/path_priors/envelope_v3_w18",
          "data/fea/path_priors/sheet_4x8ft_1tile_contact_n2700")
BANDS = ((0.0, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 0.20),
         (0.20, 0.40), (0.40, 0.80), (0.80, np.inf))
MIN_THETA_DEG = 1.0        # below this a hinge has not moved and its route direction is noise


def route_departures(prior, *, min_theta_deg=MIN_THETA_DEG, max_examples=None, seed=0):
    """-> (departures, n_seen, n_still) over every masked hinge of every harvested deployment."""
    with np.load(os.path.join(prior, "paths.npz"), allow_pickle=True) as z:
        eta = z["eta"]                                  # (n_examples, n_steps, n_hinges, 3)
        w_lig = z["w_lig"]
        mask = z["hinge_mask"] if "hinge_mask" in z.files else None
    n_ex, _, n_h, _ = eta.shape
    idx = np.arange(n_ex)
    if max_examples and max_examples < n_ex:
        idx = np.sort(np.random.default_rng(seed).choice(n_ex, max_examples, replace=False))

    out, n_still = [], 0
    for i in idx:
        for h in range(n_h):
            if mask is not None and not mask[i, h]:
                continue
            path = eta[i, :, h, :]
            if not np.isfinite(path).all():
                continue
            w = float(w_lig[i, h]) if w_lig.ndim == 2 else float(w_lig[i])
            theta = path[:, 2]
            # A hinge that never opens has no route to speak of: its (a, s) are rounding noise and
            # the ratio that defines non-proportionality would divide one small number by another.
            if np.degrees(np.abs(theta).max()) < min_theta_deg:
                n_still += 1
                continue
            out.append(max(nonproportionality(path[:, 0] * w, theta, w),
                           nonproportionality(path[:, 1] * w, theta, w)))
    return np.asarray(out, float), len(idx) * n_h, n_still


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--priors", nargs="*", default=list(PRIORS))
    ap.add_argument("--min-theta-deg", dest="min_theta_deg", type=float, default=MIN_THETA_DEG)
    ap.add_argument("--max-examples", dest="max_examples", type=int, default=None)
    ap.add_argument("--out", default="data/outputs/deployment_route_geometry.json")
    args = ap.parse_args()

    report = {}
    for prior in args.priors:
        if not os.path.exists(os.path.join(prior, "paths.npz")):
            print(f"  missing, skipping: {prior}")
            continue
        d, n_seen, n_still = route_departures(prior, min_theta_deg=args.min_theta_deg,
                                              max_examples=args.max_examples)
        if not len(d):
            continue
        q = np.percentile(d, [10, 25, 50, 75, 90])
        rec = dict(n_routes=int(len(d)), n_hinges_seen=int(n_seen), n_below_theta_floor=int(n_still),
                   min_theta_deg=args.min_theta_deg,
                   p10=q[0], p25=q[1], median=q[2], p75=q[3], p90=q[4],
                   share_below_campaign_tol=float(np.mean(d <= DEF_TOL)),
                   bands={f"{lo}-{hi}": float(np.mean((d >= lo) & (d < hi))) for lo, hi in BANDS})
        report[prior] = rec
        print(f"\n{prior}")
        print(f"  {rec['n_routes']} moving hinge routes of {n_seen} "
              f"({n_still} below the {args.min_theta_deg} deg floor, not scored)")
        print(f"  departure from a proportional route: median {rec['median']:.3f}   "
              f"p25 {rec['p25']:.3f}  p75 {rec['p75']:.3f}  p90 {rec['p90']:.3f}")
        print(f"  share at or below the campaign's {DEF_TOL} convention: "
              f"{rec['share_below_campaign_tol']*100:.1f} per cent")
        for k, v in rec["bands"].items():
            print(f"     {k:>12s}  {v*100:5.1f}%")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
