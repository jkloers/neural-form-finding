"""Score a deployment's hinge-states against a checkpoint's TRUST REGION, and size the error bar.

The PET surrogate is trained on ``origin_ray`` jobs only, because those are the ones a state
function can legitimately fit (see ``nff/utils/routes.py``). The deployed sheet does not ride origin
rays -- it rides solver-chosen curves, the ``free_path`` family, on which the same checkpoint scores
several times worse. That gap is not a bug to hide; it is the honest error bar an origin_ray-only
surrogate ships with, and it can only be quoted if the OOD exposure is actually measured.

This reports what fraction of the ridden hinge-states leave the checkpoint's own
``stats["domain"]`` box, per axis and jointly, and HOW FAR they leave it in units of the box
half-width -- a state 0.01 outside is extrapolation in name only, one 3x outside is not.

Reads the same npz that ``extract_hinge_paths`` / ``sample_hinge_paths`` write (``eta`` is
(n_steps, n_hinges, 3) = (eta_a, eta_s, theta)), accepting either a single path prefix or an
ensemble directory of per-design files.

    JAX_PLATFORMS=cpu conda run -n kgnn_mac python -m nff.scripts.diagnostics.score_deployment_ood \
        --paths data/outputs/hinge_paths/sheet_4x8ft_rom_CALIBRATED_n64_s0.5/designs \
        --surrogate data/surrogates/hinge_surrogate_pet_pb2_dmg
"""
import argparse
import glob
import json
import os

import jax
jax.config.update("jax_enable_x64", True)   # before load_hinge_surrogate, whose float64 upcast is
                                            # silently truncated to float32 otherwise -- the exact
                                            # hazard its own docstring warns about
import numpy as np

from nff.models.hinge_surrogate import DOMAIN, _feature_cols, load_hinge_surrogate


def _whiten(a, s, theta, w_lig, alpha, stats):
    """Raw hinge state -> the STANDARDIZED vector the network actually sees.

    Distances must be measured here, not in raw (eta, theta) units: the net's notion of "far" is set
    by the training spread of each feature, and the raw axes differ by an order of magnitude.
    """
    kind = stats.get("feat_kind", "raw")
    cols = _feature_cols(a, s, theta, w_lig, alpha, [], kind, np)
    X = np.stack(cols, axis=-1)
    return (X - np.asarray(stats["feat_mean"])) / np.asarray(stats["feat_std"])


def nn_distance(query, reference, chunk=2048):
    """Distance from each query row to its NEAREST reference row. Chunked to bound memory."""
    out = np.empty(len(query))
    for i in range(0, len(query), chunk):
        d = np.linalg.norm(query[i:i + chunk, None, :] - reference[None, :, :], axis=-1)
        out[i:i + chunk] = d.min(axis=1)
    return out


def training_reference(data_path, stats, *, route_family, w_lig_max, val_frac, test_frac, seed):
    """-> (train rows, held-out rows) in whitened space, matching how the checkpoint was trained.

    The held-out rows are the YARDSTICK. A surrogate exists to interpolate, so "far from training
    data" only means something relative to how far a row the net was VALIDATED on typically sits.
    Comparing a deployment's NN distance against that scale separates ordinary void-filling (what
    the surrogate is for) from genuine extrapolation (what it cannot be trusted for).
    """
    from nff.scripts.train_hinge_surrogate import converged_mask, load_dataset
    from nff.utils.routes import route_family_mask
    from nff.utils.splits import split_by_job

    data, _, _ = load_dataset(data_path, w_lig_max=w_lig_max)
    a, s, th = data["u"][:, 0], data["u"][:, 1], data["u"][:, 2]
    w_lig, alpha = data["g"][:, 0], data["g"][:, 1]
    keep = converged_mask(data) & route_family_mask(a, s, th, w_lig, data["job_id"], route_family)
    tr, va, _ = split_by_job(data, val_frac, seed, test_frac=test_frac)
    X = _whiten(a, s, th, w_lig, alpha, stats)
    return X[tr & keep], X[va & keep]


def load_eta(paths_arg):
    """-> pooled eta, per-file eta, files, and pooled (w_lig, alpha) broadcast to one row per state.

    ``w_lig``/``alpha`` are per-HINGE in the npz while ``eta`` is (n_steps, n_hinges, 3), so they are
    tiled over steps; the geometry columns are needed to rebuild the network's own feature vector.
    """
    if os.path.isdir(paths_arg):
        files = sorted(glob.glob(os.path.join(paths_arg, "*.npz")))
    else:
        files = [paths_arg if paths_arg.endswith(".npz") else paths_arg + ".npz"]
    if not files:
        raise SystemExit(f"no .npz under {paths_arg}")
    per_file, geom = [], []
    for f in files:
        with np.load(f) as z:
            if "eta" not in z:
                raise SystemExit(f"{f} has no 'eta' array -- not a hinge-paths file")
            eta = np.asarray(z["eta"])
            n_steps = eta.shape[0]
            per_file.append(eta.reshape(-1, 3))
            geom.append(np.stack([np.tile(np.asarray(z["w_lig"]), n_steps),
                                  np.tile(np.asarray(z["alpha"]), n_steps)], axis=-1))
    return np.concatenate(per_file), per_file, files, np.concatenate(geom)


def box_exceedance(eta, dom):
    """Signed distance outside each face of the trust box, in units of that axis's half-width.

    Normalizing by the half-width is what makes the three axes comparable: eta_a and theta have
    box widths that differ by an order of magnitude, so raw exceedances cannot be pooled.
    """
    a, s, th = eta[:, 0], eta[:, 1], eta[:, 2]
    a_lo, a_hi = dom.get("eta_a_min", 0.0), dom["eta_a_max"]
    th_lo, th_hi = dom.get("theta_min", -dom["theta_max"]), dom["theta_max"]
    half = dict(eta_a=max((a_hi - a_lo) / 2.0, 1e-12),
                eta_s=max(dom["eta_s_max"], 1e-12),
                theta=max((th_hi - th_lo) / 2.0, 1e-12))
    over = dict(
        eta_a=(np.maximum(a - a_hi, 0.0) + np.maximum(a_lo - a, 0.0)) / half["eta_a"],
        eta_s=np.maximum(np.abs(s) - dom["eta_s_max"], 0.0) / half["eta_s"],
        theta=(np.maximum(th - th_hi, 0.0) + np.maximum(th_lo - th, 0.0)) / half["theta"],
    )
    inside = {k: (v <= 0.0) | np.isclose(v, 0.0) for k, v in over.items()}
    return over, inside, dict(eta_a=(a_lo, a_hi), eta_s=(-dom["eta_s_max"], dom["eta_s_max"]),
                              theta=(th_lo, th_hi))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paths", required=True,
                    help="hinge-paths npz prefix, or a directory of per-design npz files")
    ap.add_argument("--surrogate", required=True, help="checkpoint prefix (no .pkl)")
    ap.add_argument("--compare", default=None,
                    help="second checkpoint to score the SAME paths against (e.g. the steel net "
                         "the configs used before) -- coverage only means something relative")
    ap.add_argument("--out", default=None, help="write the report as json here")
    ap.add_argument("--data", default=None,
                    help="training dataset. Supplying it enables the NN-DISTANCE report, which is "
                         "the metric that actually distinguishes interpolation from extrapolation "
                         "-- the box below is only a p1/p99 marginal hull, and being inside it does "
                         "not mean being near data")
    ap.add_argument("--route-family", dest="route_family", default="origin_ray",
                    help="must match what the checkpoint was trained on, else the reference set is "
                         "not the data the net actually saw")
    ap.add_argument("--w-lig-max", dest="w_lig_max", type=float, default=25.0)
    ap.add_argument("--val-frac", dest="val_frac", type=float, default=0.15)
    ap.add_argument("--test-frac", dest="test_frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    eta, per_file, files, geom = load_eta(args.paths)
    print(f"{eta.shape[0]} hinge-states from {len(files)} file(s)\n")

    q = lambda x, p: float(np.percentile(x, p))
    print("ridden envelope")
    for i, name in enumerate(["eta_a", "eta_s", "theta"]):
        col = eta[:, i]
        print(f"  {name:6s} p1 {q(col, 1):+.3f}  p50 {q(col, 50):+.3f}  p99 {q(col, 99):+.3f}   "
              f"min {col.min():+.3f}  max {col.max():+.3f}")

    report = dict(paths=args.paths, n_points=int(eta.shape[0]), n_files=len(files), checkpoints={})
    for label, ckpt in [("primary", args.surrogate)] + ([("compare", args.compare)] if args.compare else []):
        _, stats, _ = load_hinge_surrogate(ckpt + ".pkl")
        dom = {k: float(v) for k, v in dict(stats.get("domain", DOMAIN)).items()}
        over, inside, edges = box_exceedance(eta, dom)
        joint = inside["eta_a"] & inside["eta_s"] & inside["theta"]
        print(f"\n[{label}] {os.path.basename(ckpt)}")
        print("  box: " + "  ".join(f"{k} [{lo:+.3f}, {hi:+.3f}]" for k, (lo, hi) in edges.items()))
        entry = dict(checkpoint=ckpt, domain=dom, in_box_joint=float(joint.mean()), axes={})
        for k in ["eta_a", "eta_s", "theta"]:
            out_mask = ~inside[k]
            exc = over[k][out_mask]
            print(f"  {k:6s} in-box {inside[k].mean() * 100:5.1f}%   "
                  f"outside: p50 {(q(exc, 50) if exc.size else 0.0):.2f} "
                  f"p90 {(q(exc, 90) if exc.size else 0.0):.2f} "
                  f"max {(exc.max() if exc.size else 0.0):.2f}  (half-widths)")
            entry["axes"][k] = dict(in_box=float(inside[k].mean()),
                                    out_p50=float(q(exc, 50)) if exc.size else 0.0,
                                    out_p90=float(q(exc, 90)) if exc.size else 0.0,
                                    out_max=float(exc.max()) if exc.size else 0.0)
        print(f"  JOINT in-box {joint.mean() * 100:.1f}%")

        if args.data:
            X_tr, X_ho = training_reference(args.data, stats, route_family=args.route_family,
                                            w_lig_max=args.w_lig_max, val_frac=args.val_frac,
                                            test_frac=args.test_frac, seed=args.seed)
            d_ho = nn_distance(X_ho, X_tr)
            yard = float(np.percentile(d_ho, 90))
            Q = _whiten(eta[:, 0] * geom[:, 0], eta[:, 1] * geom[:, 0], eta[:, 2],
                        geom[:, 0], geom[:, 1], stats)
            d_q = nn_distance(Q, X_tr)
            print(f"  NN distance to training data [training feature-std], {len(X_tr)} ref rows:")
            print(f"    held-out rows (YARDSTICK)  p50 {q(d_ho, 50):6.3f}  p90 {yard:6.3f}")
            print(f"    deployment                 p50 {q(d_q, 50):6.3f}  p90 {q(d_q, 90):6.3f}"
                  f"  max {d_q.max():6.3f}")
            print(f"    -> median is {q(d_q, 50) / max(q(d_ho, 50), 1e-9):.1f}x the held-out median;"
                  f"  {np.mean(d_q > yard) * 100:.1f}% of states beyond the held-out p90")
            entry["nn"] = dict(yardstick_p50=float(q(d_ho, 50)), yardstick_p90=yard,
                               query_p50=float(q(d_q, 50)), query_p90=float(q(d_q, 90)),
                               query_max=float(d_q.max()),
                               frac_beyond_yardstick=float(np.mean(d_q > yard)),
                               n_reference_rows=int(len(X_tr)))
        if len(per_file) > 1:
            fr = np.array([box_exceedance(e, dom)[1] for e in per_file], dtype=object)
            per_design = np.array([
                float((i_["eta_a"] & i_["eta_s"] & i_["theta"]).mean()) for i_ in fr])
            print(f"  per-design in-box: min {per_design.min() * 100:.1f}%  "
                  f"med {np.median(per_design) * 100:.1f}%  max {per_design.max() * 100:.1f}%")
            entry["per_design_in_box"] = dict(min=float(per_design.min()),
                                              median=float(np.median(per_design)),
                                              max=float(per_design.max()))
        report["checkpoints"][label] = entry

    if args.compare:
        p = report["checkpoints"]["primary"]["in_box_joint"]
        c = report["checkpoints"]["compare"]["in_box_joint"]
        print(f"\ncoverage {p * 100:.1f}% vs {c * 100:.1f}% -> "
              f"{'better' if p > c else 'worse'} by {abs(p - c) * 100:.1f} points")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
