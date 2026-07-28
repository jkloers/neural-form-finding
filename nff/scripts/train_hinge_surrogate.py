"""Train the condensed hinge-energy surrogate on the CalculiX dataset.

    JAX_PLATFORMS=cpu conda run -n kgnn_mac python nff/scripts/train_hinge_surrogate.py \
        --data data/fea/hinge_dataset --epochs 300 --lam 0.8 --out data/surrogates/hinge_surrogate

Pipeline:
  1. Load npz -> u=(a,s,theta), g=(w_lig, alpha[rad]), targets W, F=(F_a,F_s,M_theta),
     margin=damage (<PEEQ>_lig/eps_f, see nff.rve.damage).
  2. PRE-FLIGHT force-sign check: on pure-rotation (spine) rays, FD dW/dtheta must equal the
     stored M_theta. If the sign is flipped, flip F so F_target = +dW/du (envelope theorem).
  3. Split by JOB (= unseen geometry) -- never by sample; ray samples are correlated.
  4. Standardize inputs from TRAIN only; Adam + cosine LR; lambda-weighted Sobolev loss.
  5. Best-checkpoint on validation energy error; save params + stats + metrics.
"""

import argparse
import json
import os
import pickle

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import optax

from nff.models.hinge_surrogate import (init_hinge_surrogate, compute_norm_stats,
    apply_hinge_energy, apply_hinge_force, apply_hinge_failure, sobolev_loss)


def load_dataset(path):
    with np.load(path + ".npz") as npz:
        d = {k: np.asarray(npz[k]) for k in npz.files}
    eps_f = json.load(open(path + ".json"))["const"].get("eps_f", 0.25)
    # Damage-head target: THE damage measure, <PEEQ>_lig / eps_f (nff.rve.damage). Pre-2026-07-28
    # datasets carry a `damage_p99` column instead -- a whole-mesh p99 of a triaxiality-weighted
    # ratio, a different quantity on a different scale. Reject rather than silently retrain on it.
    if "damage" not in d:
        legacy = "`damage_p99`" if "damage_p99" in d else "no damage column"
        raise KeyError(
            f"{path}.npz carries {legacy}, not `damage`. The damage definition changed on "
            "2026-07-28 (whole-mesh p99 of PEEQ/eps_f(eta) -> ligament volume average "
            "<PEEQ>_lig/eps_f); the two are not interchangeable. Re-run the campaign via "
            "nff/scripts/generate_hinge_dataset.py.")
    # A non-finite target NaNs the WHOLE batch loss, which the nan_to_num guard on the gradient then
    # turns into silently zeroed energy AND force updates -- so drop those rows here instead.
    finite = np.isfinite(d["damage"])
    if not finite.all():
        print(f"damage: dropping {int((~finite).sum())}/{finite.size} non-finite rows")
        d = {k: v[finite] if v.shape[:1] == finite.shape else v for k, v in d.items()}
    # geometry g = (w_lig, alpha[rad][, fillet_ratio]); include the fillet DOF iff it was SWEPT
    g_cols = [d["w_lig"].astype(float), np.radians(d["alpha_deg"].astype(float))]
    if "fillet_ratio" in d and float(np.ptp(d["fillet_ratio"].astype(float))) > 1e-6:
        g_cols.append(d["fillet_ratio"].astype(float))
        print("fillet DOF present + swept -> 3-D geometry input (6 features)")
    data = dict(
        u=np.stack([d["a"], d["s"], d["theta"]], -1),
        g=np.stack(g_cols, -1),
        W=d["W"].astype(float),
        F=np.stack([d["F_a"], d["F_s"], d["M_theta"]], -1),
        margin=d["damage"].astype(float),
        job_id=d["job_id"],
    )
    return data, eps_f


def check_force_sign(data, n_probe=30):
    """On pure-rotation rays (a=s=0), FD dW/dtheta should equal M_theta. Return (+1 or -1, ratio)."""
    jid = data["job_id"]
    ratios = []
    for j in np.unique(jid):
        m = jid == j
        if m.sum() > 5 and np.abs(data["u"][m, 0]).max() < 1e-9 and np.abs(data["u"][m, 1]).max() < 1e-9:
            th, W, M = data["u"][m, 2], data["W"][m], data["F"][m, 2]
            o = np.argsort(th); th, W, M = th[o], W[o], M[o]
            dWdth = np.diff(W) / (np.diff(th) + 1e-12)
            Mmid = 0.5 * (M[:-1] + M[1:])
            good = np.abs(Mmid) > 1e-6
            if good.any():
                ratios.append(np.median(dWdth[good] / Mmid[good]))
        if len(ratios) >= n_probe:
            break
    r = float(np.median(ratios)) if ratios else 1.0
    return (1.0 if r >= 0 else -1.0), r


def split_by_job(data, val_frac, seed):
    jobs = np.unique(data["job_id"])
    rng = np.random.default_rng(seed); rng.shuffle(jobs)
    val_jobs = set(jobs[: int(val_frac * len(jobs))].tolist())
    val = np.isin(data["job_id"], list(val_jobs))
    return ~val, val


def _batch(data, idx):
    return {k: jnp.asarray(data[k][idx]) for k in ("u", "g", "W", "F", "margin")}


def evaluate(params, batch, stats):
    Wp = apply_hinge_energy(params, batch["u"], batch["g"], stats)
    Fp = apply_hinge_force(params, batch["u"], batch["g"], stats)
    mp = apply_hinge_failure(params, batch["u"], batch["g"], stats)
    rel = lambda p, t: float(jnp.sqrt(jnp.mean((p - t) ** 2)) / (jnp.sqrt(jnp.mean(t ** 2)) + 1e-12))
    return dict(energy_rel=rel(Wp, batch["W"]), force_rel=rel(Fp, batch["F"]),
                damage_rmse=float(jnp.sqrt(jnp.mean((mp - batch["margin"]) ** 2))))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/fea/hinge_dataset")
    ap.add_argument("--out", default="data/surrogates/hinge_surrogate")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lam", type=float, default=0.8, help="energy-vs-force priority (>0.5 = energy)")
    ap.add_argument("--w-damage", dest="w_damage", type=float, default=0.1,
                    help="weight on the damage head (variance-normalized, like energy/force)")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--val-frac", dest="val_frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hidden", default="64,64",
                    help="comma-sep hidden widths (default compact 64,64; original was 128,128,128)")
    args = ap.parse_args()

    data, eps_f = load_dataset(args.data)
    sign, ratio = check_force_sign(data)
    print(f"force-sign check: FD dW/dtheta / M_theta ~ {ratio:+.3f} -> "
          f"{'match (F = +dW/du)' if sign > 0 else 'FLIPPED (negating F)'}")
    data["F"] *= sign

    tr, va = split_by_job(data, args.val_frac, args.seed)
    n_tr, n_va = int(tr.sum()), int(va.sum())
    print(f"samples: {len(data['W'])}  |  train {n_tr} / val {n_va}  "
          f"(held-out geometries: {len(np.unique(data['job_id'][va]))} jobs)")

    tr_idx = np.where(tr)[0]
    _fillet = data["g"][tr, 2] if data["g"].shape[-1] >= 3 else None
    stats = compute_norm_stats(data["u"][tr, 0], data["u"][tr, 1], data["u"][tr, 2],
                               data["g"][tr, 0], data["g"][tr, 1], data["W"][tr], fillet_ratio=_fillet)
    # Data-driven trust region (the box the dataset actually covers) for the OOD barrier. Stored in
    # stats -> the closed pipeline's domain AUTO-matches this surrogate; a wider v2 dataset widens it
    # with no manual edit. Legacy checkpoints without it fall back to the hardcoded DOMAIN.
    _wl = np.maximum(data["g"][:, 0], 1e-9)
    stats["domain"] = dict(eta_a_max=float(np.percentile(data["u"][:, 0] / _wl, 99.0)),
                           eta_s_max=float(np.percentile(np.abs(data["u"][:, 1] / _wl), 99.0)),
                           theta_max=float(np.percentile(np.abs(data["u"][:, 2]), 99.0)))
    print(f"trust region (p99): eta_a<={stats['domain']['eta_a_max']:.2f} "
          f"|eta_s|<={stats['domain']['eta_s_max']:.2f} theta<={stats['domain']['theta_max']:.2f}rad")
    val_batch = _batch(data, va)

    key = jax.random.PRNGKey(args.seed)
    hidden = tuple(int(x) for x in args.hidden.split(","))
    params = init_hinge_surrogate(key, hidden=hidden, feat_dim=3 + data["g"].shape[-1])
    steps = max(1, n_tr // args.batch)
    sched = optax.cosine_decay_schedule(args.lr, args.epochs * steps)
    optimizer = optax.adam(sched)
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(params, opt_state, batch):
        (loss, aux), grads = jax.value_and_grad(
            lambda p: sobolev_loss(p, batch, stats, lam=args.lam, w_damage=args.w_damage),
            has_aux=True)(params)
        grads = jax.tree_util.tree_map(lambda x: jnp.nan_to_num(x), grads)  # NaN-safe
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    best = {"energy_rel": np.inf}
    best_params = params
    rng = np.random.default_rng(args.seed)
    for ep in range(args.epochs):
        perm = rng.permutation(tr_idx)
        for b in range(steps):
            params, opt_state, loss = train_step(
                params, opt_state, _batch(data, perm[b * args.batch:(b + 1) * args.batch]))
        if ep % 10 == 0 or ep == args.epochs - 1:
            m = evaluate(params, val_batch, stats)
            tag = ""
            if m["energy_rel"] < best["energy_rel"]:
                best, best_params, tag = m, jax.tree_util.tree_map(lambda x: x, params), "  <- best"
            print(f"ep {ep:4d}  loss {float(loss):.4g}  val energy_rel {m['energy_rel']:.4f}  "
                  f"force_rel {m['force_rel']:.4f}  damage_rmse {m['damage_rmse']:.4f}{tag}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".pkl", "wb") as f:
        pickle.dump({"params": best_params, "stats": stats, "eps_f": eps_f,
                     "lam": args.lam, "val_metrics": best}, f)
    with open(args.out + ".json", "w") as f:
        json.dump({"val_metrics": best, "n_train": n_tr, "n_val": n_va,
                   "force_sign": sign, "lam": args.lam}, f, indent=2)
    print(f"\nbest val: energy_rel {best['energy_rel']:.4f}  force_rel {best['force_rel']:.4f}  "
          f"damage_rmse {best['damage_rmse']:.4f}\nsaved -> {args.out}.pkl")


if __name__ == "__main__":
    main()
