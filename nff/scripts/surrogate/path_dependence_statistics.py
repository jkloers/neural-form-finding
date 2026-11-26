"""The statistics §5.2 prints for Hypothesis H4: bias, rank correlation, and the scaling exponent.

WHY THIS EXISTS. The section used to answer "is the condensed energy a function of state?" with a
narrative of bands and worst cases, which states a bound without stating what governs it. The
paired campaign supports a sharper reading, and this script is the provenance for every number the
section now prints.

Three questions, three standard instruments:

  1. Is there a systematic offset? A signed-rank test on ``log(W_path / W_ray)``. There is not
     (p = 0.47), so the ray is neither systematically stiffer nor softer than the route.
  2. Is the discrepancy governed by the separation? Spearman rank correlation, which does not
     assume the shape of the relation. It is (rho = 0.84, p = 4e-6).
  3. At what ORDER? A power law fitted over the pairs above the solver noise floor. The exponent is
     2.3 with a bootstrap interval of [1.9, 2.8] excluding one, so the discrepancy is SECOND order
     in the separation. That is the load-bearing result: at fixed endpoints the first variation of
     the work with respect to the route vanishes, and the straight ray is a stationary point of the
     work. Hypothesis H4 survives because the functional is insensitive to the route, not because
     deployed paths happen to be straight.

The noise floor matters and is not a free parameter. Below ``MIN_BEND`` the two legs are the same
route to within solver resolution, so their ratio carries no signal about scaling; the exponent is
reported against the cutoff so the reader can see it is stable (1.96 to 2.50 over cutoffs from 0 to
6e-3) rather than an artefact of one choice.

    conda run -n kgnn_mac python -m nff.scripts.surrogate.path_dependence_statistics \
        --pairs data/surrogates/path_dependence_pet_v3.json \
        --prior data/fea/path_priors/envelope_v3_w18
"""
from __future__ import annotations

import argparse
import json

import numpy as np

MIN_BEND = 2e-3          # ligament widths; below this the two legs are the same route
SUB_DEGREE = 1.0         # deg; a hinge that never turns this far is not a path


def separation_of_prior(prior_dir: str) -> np.ndarray:
    """Separation, endpoint translation and endpoint rotation of every harvested hinge path.

    The separation is in ligament widths. The endpoint translation and rotation are returned
    alongside because they are what reconciles the two departure measures: a hinge that ends at a
    few hundredths of a ligament width against tens of degrees of fold rides the rotation axis, so
    it reads as near-a-ray in absolute terms and as far-from-proportional in relative ones.

    ``eta`` is already ``(a, s) / w_lig``, so the norm is in ligament widths with no further
    scaling. Sub-degree hinges are dropped: their separation is a ratio of two quantities that are
    both at the solver floor.
    """
    z = np.load(f"{prior_dir}/paths.npz")
    eta = z["eta"]                                            # (n_ex, n_step, n_hinge, 3)
    paths = eta.transpose(0, 2, 1, 3).reshape(-1, eta.shape[1], 3)
    moving = np.degrees(np.abs(paths[:, -1, 2])) >= SUB_DEGREE
    states = paths[moving][:, 1:, :]                          # drop the undeformed origin
    n = states.shape[1]
    ray = (np.arange(1, n + 1) / n)[None, :, None] * states[:, -1:, :]
    separation = np.linalg.norm(states[:, :, :2] - ray[:, :, :2], axis=-1).max(axis=1)
    endpoint = states[:, -1, :]
    return separation, np.linalg.norm(endpoint[:, :2], axis=1), np.abs(endpoint[:, 2])


def scaling_exponent(bend: np.ndarray, err: np.ndarray, cutoff: float, n_boot: int = 5000):
    """Power-law exponent of ``err`` against ``bend``, with a bootstrap interval."""
    keep = bend > cutoff
    x, y = np.log(bend[keep]), np.log(err[keep])
    slope, intercept = np.polyfit(x, y, 1)
    r2 = np.corrcoef(x, y)[0, 1] ** 2
    rng = np.random.default_rng(0)
    draws = []
    for _ in range(n_boot):
        i = rng.integers(0, keep.sum(), keep.sum())
        if len(set(x[i])) >= 3:
            draws.append(np.polyfit(x[i], y[i], 1)[0])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return slope, np.exp(intercept), r2, lo, hi, int(keep.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/surrogates/path_dependence_pet_v3.json")
    ap.add_argument("--prior", default="data/fea/path_priors/envelope_v3_w18")
    args = ap.parse_args()

    pairs = json.load(open(args.pairs))["pairs"]
    bend = np.array([p["bend"] for p in pairs], float)
    log_ratio = np.log(np.array([p["W_ratio"] for p in pairs], float))
    err = np.abs(log_ratio)

    from scipy.stats import spearmanr, wilcoxon
    print(f"pairs: {len(pairs)}   separations {bend.min():.5f} to {bend.max():.5f}")
    print(f"bias      : {(log_ratio > 0).sum()} above one, {(log_ratio < 0).sum()} below, "
          f"signed-rank p = {wilcoxon(log_ratio).pvalue:.2f}")
    rho, p_rho = spearmanr(bend, err)
    print(f"relation  : Spearman rho = {rho:.2f}  (p = {p_rho:.1e})")

    print("exponent against the noise-floor cutoff:")
    for cutoff in (0.0, 1e-3, MIN_BEND, 4e-3, 6e-3):
        slope, _, r2, lo, hi, n = scaling_exponent(bend, err, cutoff)
        print(f"    cutoff {cutoff:.0e}  n={n:2d}   exponent {slope:.2f}  [{lo:.2f}, {hi:.2f}]"
              f"   R2 {r2:.2f}")

    keep = bend > MIN_BEND
    central = np.exp(np.mean(np.log(err[keep]) - 2 * np.log(bend[keep])))
    envelope = (err[keep] / bend[keep] ** 2).max()
    print(f"quadratic : central {central:.2f} * separation^2, "
          f"envelope {envelope:.2f} (no pair exceeds it)")

    sep, tr, th = separation_of_prior(args.prior)
    print("\nwhy the two departure measures disagree: the deployed hinge is nearly pure rotation,")
    print(f"    endpoint translation {np.median(tr):.3f} ligament widths against "
          f"{np.degrees(np.median(th)):.1f} deg of fold (medians)")
    print(f"\ndeployed population: {len(sep)} paths, "
          f"{100 * (sep <= bend.max()).mean():.1f}% inside the tested range")
    for q in (50, 75, 90, 95, 99):
        s = np.percentile(sep, q)
        print(f"    p{q:<3d} separation {s:.4f}   energy misread by {100 * (np.exp(central * s ** 2) - 1):.3f}%")


if __name__ == "__main__":
    main()
