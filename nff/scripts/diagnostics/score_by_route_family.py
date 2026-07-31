"""Score a surrogate checkpoint PER ROUTE FAMILY on the held-out split.

The one number that matters for the path hypothesis is not the pooled error: a checkpoint trained on
the mixed PET campaign scores 5.97% on ``origin_ray`` rows and 43.5% on ``free_path`` rows, and the
pooled ~21% is just their mixture. Reporting pooled is how that went unexplained for weeks.

Handles checkpoints trained with ``--oracle-history`` (an extra geometry column), so the
feasibility ceiling can be measured on the same footing as a shippable model. ⚠ an oracle-history
score is a CEILING -- ``z`` is not available at inference.

    JAX_PLATFORMS=cpu conda run -n kgnn_mac python -m nff.scripts.diagnostics.score_by_route_family \
        --surrogate data/surrogates/hinge_surrogate_pet_orc_damage \
        --data data/fea/hinge_dataset_pet_merged --oracle-history damage
"""
import argparse
import pickle

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from nff.models.hinge_surrogate import (load_hinge_surrogate, apply_hinge_energy,
                                        apply_hinge_force, apply_hinge_failure)
from nff.scripts.train_hinge_surrogate import (load_dataset, check_force_sign, converged_mask,
                                               oracle_history_column)
from nff.utils.routes import FREE_PATH, ORIGIN_RAY, classify_jobs
from nff.utils.splits import split_by_job


def _rel(pred, true):
    return float(np.sqrt(np.mean((pred - true) ** 2)) / (np.sqrt(np.mean(true ** 2)) + 1e-12))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--surrogate", required=True)
    ap.add_argument("--data", default="data/fea/hinge_dataset_pet_merged")
    ap.add_argument("--oracle-history", dest="oracle_history", default=None,
                    help="comma-separated; must match what the checkpoint was trained with")
    ap.add_argument("--w-lig-max", dest="w_lig_max", type=float, default=25.0)
    ap.add_argument("--val-frac", dest="val_frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data, _, _ = load_dataset(args.data, w_lig_max=args.w_lig_max)
    sign, ratio, how = check_force_sign(data)
    data["F"] *= sign
    if args.oracle_history:
        kinds = [k.strip() for k in args.oracle_history.split(",") if k.strip()]
        data["g"] = np.concatenate(
            [data["g"]] + [oracle_history_column(data, k)[:, None] for k in kinds], axis=-1)

    meta = {}
    try:
        with open(args.surrogate + ".pkl", "rb") as f:
            meta = pickle.load(f).get("meta", {}) or {}
    except Exception:
        pass
    params, stats, _ = load_hinge_surrogate(args.surrogate + ".pkl")

    fam, _ = classify_jobs(data["u"][:, 0], data["u"][:, 1], data["u"][:, 2],
                           data["g"][:, 0], data["job_id"])
    fam_row = np.array([fam.get(int(j), "unknown") for j in data["job_id"]])

    # Reproduce the checkpoint's OWN split, or its training rows leak into "held-out". A model
    # trained with --route-family was split over that family alone, so splitting the full dataset
    # here would partition differently and score it partly on rows it had already seen (measured:
    # a leaked 1.41% against an honest 2.98%). Rows of the OTHER family were never in training, so
    # all of them are legitimately held out.
    trained_fam = meta.get("route_family", "all")
    if trained_fam != "all":
        own = fam_row == trained_fam
        sub = {k: (v[own] if getattr(v, "shape", (0,))[:1] == own.shape else v)
               for k, v in data.items()}
        _, va_own = split_by_job(sub, args.val_frac, args.seed)
        va = np.zeros(len(fam_row), bool)
        va[np.where(own)[0][va_own]] = True          # held-out rows of the trained family
        va |= ~own                                    # every row of the untrained family
    else:
        _, va = split_by_job(data, args.val_frac, args.seed)
    if meta.get("exclude_unconverged", "none") != "none":
        va = va & converged_mask(data)

    u, g = jnp.asarray(data["u"]), jnp.asarray(data["g"])
    W = np.asarray(apply_hinge_energy(params, u, g, stats))
    F = np.asarray(apply_hinge_force(params, u, g, stats))
    D = np.asarray(apply_hinge_failure(params, u, g, stats))

    print(f"{args.surrogate}  [force-sign {how} {ratio:+.3f}]")
    print(f"  trained: route_family={meta.get('route_family','?')} "
          f"oracle_history={meta.get('oracle_history', args.oracle_history)} "
          f"hidden={meta.get('hidden','?')}")
    if args.oracle_history:
        print("  ⚠ ORACLE-HISTORY CEILING -- z is unavailable at inference; not a shippable number.")
    if args.oracle_history == "damage":
        print("  ⚠ damage is BOTH input and target here, so `damage` below is circular -- read "
              "energy/force only.")
    print(f"\n  {'family':<12}{'n':>7}{'energy':>10}{'force':>10}{'damage':>10}")
    for f in (ORIGIN_RAY, FREE_PATH, "ALL"):
        m = va if f == "ALL" else (va & (fam_row == f))
        if m.sum() < 20:
            continue
        fc = float(np.mean([_rel(F[m, j], data["F"][m, j]) for j in range(3)]))
        print(f"  {f:<12}{int(m.sum()):7d}{_rel(W[m], data['W'][m])*100:9.2f}%"
              f"{fc*100:9.2f}%{_rel(D[m], data['margin'][m])*100:9.2f}%")


if __name__ == "__main__":
    main()
