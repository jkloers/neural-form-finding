"""Train the condensed hinge-energy surrogate on the CalculiX dataset.

    conda run -n kgnn_mac python nff/scripts/train_hinge_surrogate.py \
        --data data/fea/hinge_dataset_pet_v2 --epochs 300 --lam 0.7 \
        --out data/surrogates/hinge_surrogate_pet_v2

Pipeline:
  1. Load npz -> u=(a,s,theta), g=(w_lig, alpha[rad]), targets W, F=(F_a,F_s,M_theta),
     margin=damage (see nff.rve.damage). Rows above --w-lig-max are dropped (Saint-Venant).
  2. PRE-FLIGHT force-sign check: on pure-rotation (spine) rays, FD dW/dtheta must equal the
     stored M_theta. If the sign is flipped, flip F so F_target = +dW/du (envelope theorem).
  3. Split by JOB (= unseen geometry) -- never by sample; ray samples are correlated. Three ways:
     train / val (model selection) / test (reported once, never selected on).
  4. Standardize inputs AND target scales from TRAIN only; Adam + cosine LR; Sobolev loss.
  5. Best-checkpoint on the val loss; save params + stats + provenance + metrics.
"""

import argparse
import hashlib
import json
import os
import pickle
import subprocess
import time

import numpy as np
# The default backend here is Metal, which cannot carry float64 -- and W is differentiated twice
# downstream (IFT backward). Force CPU BEFORE importing jax; jax_md also fails if Metal inits first.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import optax

from nff.models.hinge_surrogate import (init_hinge_surrogate, compute_norm_stats,
    apply_hinge_energy, apply_hinge_force, apply_hinge_failure, sobolev_loss)
# Re-exported, not redefined: diagnostics that only need the split must not have to import JAX to
# get it (see nff/utils/splits.py). Existing `from ...train_hinge_surrogate import split_by_job`
# call sites -- nff/scripts/figures/plot_surrogate_parity.py -- keep working.
from nff.utils.splits import split_by_job  # noqa: F401


def load_dataset(path, w_lig_max=None):
    """-> (data, eps_f, const). ``const`` is the campaign's constant block, kept for provenance."""
    with np.load(path + ".npz") as npz:
        d = {k: np.asarray(npz[k]) for k in npz.files}
    const = json.load(open(path + ".json"))["const"]
    eps_f = const.get("eps_f", 0.25)
    # Damage-head target: THE damage measure (nff.rve.damage.plastic_damage) -- plastic strain over
    # the whole RVE window, per unit ligament volume, over eps_f. Pre-2026-07-28 datasets carry a
    # `damage_p99` column instead -- a whole-mesh p99 of a triaxiality-weighted ratio, a different
    # quantity on a different scale. Reject rather than silently retrain on it.
    if "damage" not in d:
        legacy = "`damage_p99`" if "damage_p99" in d else "no damage column"
        raise KeyError(
            f"{path}.npz carries {legacy}, not `damage`. The damage definition changed on "
            "2026-07-28 (whole-mesh p99 of PEEQ/eps_f(eta) -> the volume measure in "
            "nff.rve.damage.plastic_damage); the two are not interchangeable. Re-run the campaign "
            "via nff/scripts/generate_hinge_dataset.py.")
    # A non-finite target NaNs the WHOLE batch loss, which the nan_to_num guard on the gradient then
    # turns into silently zeroed energy AND force updates -- so drop those rows here instead.
    finite = np.isfinite(d["damage"])
    if not finite.all():
        print(f"damage: dropping {int((~finite).sum())}/{finite.size} non-finite rows")
        d = {k: v[finite] if v.shape[:1] == finite.shape else v for k, v in d.items()}
    # SAINT-VENANT: the RVE window is a fixed r_win, so a wide ligament stops being small compared
    # with it and the extracted W picks up panel compliance the condensation assumes away. Above
    # ~r_win/4 the hinge is no longer local; drop those rows rather than teach the net a hinge
    # energy that silently contains its neighbours.
    if w_lig_max is not None:
        keep = d["w_lig"] <= float(w_lig_max)
        if not keep.all():
            r_win = const.get("r_win", float("nan"))
            print(f"w_lig: dropping {int((~keep).sum())}/{keep.size} rows "
                  f"({len(np.unique(d['job_id'][~keep]))} jobs) above {w_lig_max} mm "
                  f"= {w_lig_max / r_win:.2f}*r_win")
            d = {k: v[keep] if v.shape[:1] == keep.shape else v for k, v in d.items()}
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
    return data, eps_f, const


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


def _batch(data, idx):
    return {k: jnp.asarray(data[k][idx]) for k in ("u", "g", "W", "F", "margin")}


def _git_rev():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, timeout=5).stdout.strip() or None
    except Exception:
        return None


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def evaluate(params, batch, stats):
    """Relative errors. ``force_rel`` is PER COMPONENT then averaged, so a good moment fit cannot
    hide bad F_a/F_s -- the pooled version is ~99% moment on the PET campaign."""
    Wp = apply_hinge_energy(params, batch["u"], batch["g"], stats)
    Fp = apply_hinge_force(params, batch["u"], batch["g"], stats)
    mp = apply_hinge_failure(params, batch["u"], batch["g"], stats)
    rel = lambda p, t: float(jnp.sqrt(jnp.mean((p - t) ** 2)) / (jnp.sqrt(jnp.mean(t ** 2)) + 1e-12))
    f_comp = [rel(Fp[:, j], batch["F"][:, j]) for j in range(3)]
    return dict(energy_rel=rel(Wp, batch["W"]),
                force_rel=float(np.mean(f_comp)),
                force_rel_a=f_comp[0], force_rel_s=f_comp[1], force_rel_m=f_comp[2],
                damage_rel=rel(mp, batch["margin"]),
                damage_rmse=float(jnp.sqrt(jnp.mean((mp - batch["margin"]) ** 2))))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/fea/hinge_dataset_pet_v2")
    ap.add_argument("--out", default="data/surrogates/hinge_surrogate_pet_v2")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lam", type=float, default=0.7, help="energy-vs-force priority (>0.5 = energy)")
    ap.add_argument("--w-damage", dest="w_damage", type=float, default=0.1,
                    help="weight on the damage head (scale-normalized, like energy/force)")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--val-frac", dest="val_frac", type=float, default=0.15)
    ap.add_argument("--test-frac", dest="test_frac", type=float, default=0.15,
                    help="held-out group scored ONCE at the end; 0 disables")
    ap.add_argument("--w-lig-max", dest="w_lig_max", type=float, default=25.0,
                    help="drop rows above this ligament width [mm] (~r_win/4: above it the RVE "
                         "no longer isolates hinge compliance from the panels)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hidden", default="64,64",
                    help="comma-sep hidden widths (default compact 64,64; original was 128,128,128)")
    ap.add_argument("--force", action="store_true", help="overwrite an existing --out checkpoint")
    args = ap.parse_args()

    if os.path.exists(args.out + ".pkl") and not args.force:
        raise SystemExit(f"{args.out}.pkl exists -- pass --force to overwrite, or pick a new name. "
                         "(Reusing a name is how a .pkl and its sidecar .json end up describing "
                         "different models.)")

    data, eps_f, const = load_dataset(args.data, w_lig_max=args.w_lig_max)
    sign, ratio = check_force_sign(data)
    print(f"force-sign check: FD dW/dtheta / M_theta ~ {ratio:+.3f} -> "
          f"{'match (F = +dW/du)' if sign > 0 else 'FLIPPED (negating F)'}")
    data["F"] *= sign

    split = split_by_job(data, args.val_frac, args.seed, test_frac=args.test_frac)
    tr, va, te = split if len(split) == 3 else (split[0], split[1], np.zeros_like(split[0]))
    n_tr, n_va, n_te = int(tr.sum()), int(va.sum()), int(te.sum())
    njobs = lambda m: len(np.unique(data["job_id"][m]))
    print(f"samples: {len(data['W'])}  |  train {n_tr} ({njobs(tr)} jobs) / "
          f"val {n_va} ({njobs(va)}) / test {n_te} ({njobs(te)})  -- split by unseen geometry")

    tr_idx = np.where(tr)[0]
    _fillet = data["g"][tr, 2] if data["g"].shape[-1] >= 3 else None
    # Target scales from TRAIN only -- passing F/D switches sobolev_loss to fixed, per-component
    # normalization (see there); without them it falls back to per-batch variance.
    stats = compute_norm_stats(data["u"][tr, 0], data["u"][tr, 1], data["u"][tr, 2],
                               data["g"][tr, 0], data["g"][tr, 1], data["W"][tr],
                               fillet_ratio=_fillet, F=data["F"][tr], D=data["margin"][tr])
    # Data-driven trust region (the box the TRAIN split actually covers) for the OOD barrier. Stored
    # in stats -> the closed pipeline's domain AUTO-matches this surrogate. TWO-SIDED: the PET
    # campaign samples the measured deployment envelope, which is ~half compression, so a
    # tension-only box would flag most of its own training data as out-of-domain.
    _wl = np.maximum(data["g"][tr, 0], 1e-9)
    _ea, _es, _th = data["u"][tr, 0] / _wl, data["u"][tr, 1] / _wl, data["u"][tr, 2]
    q = lambda v, p: float(np.percentile(v, p))
    stats["domain"] = dict(eta_a_min=q(_ea, 1.0), eta_a_max=q(_ea, 99.0),
                           eta_s_max=q(np.abs(_es), 99.0),
                           theta_min=min(0.0, q(_th, 1.0)), theta_max=q(_th, 99.0))
    _d = stats["domain"]
    print(f"trust region (p1/p99): eta_a in [{_d['eta_a_min']:+.2f}, {_d['eta_a_max']:+.2f}]  "
          f"|eta_s|<={_d['eta_s_max']:.2f}  theta in [{_d['theta_min']:+.2f}, {_d['theta_max']:.2f}] rad")
    print(f"target scales: W {float(stats['W_scale']):.4g} N.mm  "
          f"F {np.asarray(stats['sigma_F']).round(3).tolist()} (N, N, N.mm)  "
          f"D {float(stats['D_scale']):.4g}")
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

    # Selection on the SAME combination the loss optimizes. Selecting on energy alone lets a net
    # with a dead force head win -- and the force is what Stage-2 balances.
    score = lambda m: args.lam * m["energy_rel"] + (1.0 - args.lam) * m["force_rel"]
    best, best_score, best_params = None, np.inf, params
    rng = np.random.default_rng(args.seed)
    for ep in range(args.epochs):
        perm = rng.permutation(tr_idx)
        for b in range(steps):
            params, opt_state, loss = train_step(
                params, opt_state, _batch(data, perm[b * args.batch:(b + 1) * args.batch]))
        if ep % 10 == 0 or ep == args.epochs - 1:
            m = evaluate(params, val_batch, stats)
            tag = ""
            if score(m) < best_score:
                best, best_score, tag = m, score(m), "  <- best"
                best_params = jax.tree_util.tree_map(lambda x: x, params)
            print(f"ep {ep:4d}  loss {float(loss):.4g}  val energy_rel {m['energy_rel']:.4f}  "
                  f"force_rel {m['force_rel']:.4f} (a {m['force_rel_a']:.3f} s {m['force_rel_s']:.3f} "
                  f"M {m['force_rel_m']:.3f})  damage_rel {m['damage_rel']:.4f}{tag}", flush=True)

    # The test split is scored EXACTLY ONCE, here, on the already-chosen params.
    test_metrics = evaluate(best_params, _batch(data, te), stats) if n_te else None

    # W is differentiated twice downstream (IFT backward), so refuse to ship reduced precision.
    bad = [x.dtype for x in jax.tree_util.tree_leaves(best_params) if x.dtype != jnp.float64]
    assert not bad, (f"checkpoint is {bad[0]}, not float64 -- run with JAX_PLATFORMS=cpu and "
                     "jax_enable_x64 (the default Metal backend cannot carry float64)")

    meta = {**{k: v for k, v in const.items() if not isinstance(v, (list, dict))},
            "dataset": args.data, "damage_target": "damage (nff.rve.damage.plastic_damage)",
            "n_rows": len(data["W"]), "n_jobs": int(len(np.unique(data["job_id"]))),
            "w_lig_max": args.w_lig_max,
            "w_lig_range": [float(data["g"][:, 0].min()), float(data["g"][:, 0].max())],
            "hidden": hidden, "lam": args.lam, "w_damage": args.w_damage,
            "force_sign": sign, "git": _git_rev(), "trained_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                                               time.gmtime())}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".pkl", "wb") as f:
        pickle.dump({"params": best_params, "stats": stats, "eps_f": eps_f, "lam": args.lam,
                     "val_metrics": best, "test_metrics": test_metrics, "meta": meta}, f)
    # sha256 ties the sidecar to THIS pkl -- the old sidecars describe models they do not sit next to.
    with open(args.out + ".json", "w") as f:
        json.dump({"val_metrics": best, "test_metrics": test_metrics,
                   "n_train": n_tr, "n_val": n_va, "n_test": n_te,
                   "domain": stats["domain"], "meta": meta,
                   "pkl_sha256": _sha256(args.out + ".pkl")}, f, indent=2)

    print(f"\nbest VAL : energy_rel {best['energy_rel']:.4f}  force_rel {best['force_rel']:.4f}  "
          f"damage_rel {best['damage_rel']:.4f}")
    if test_metrics:
        print(f"held-out TEST: energy_rel {test_metrics['energy_rel']:.4f}  "
              f"force_rel {test_metrics['force_rel']:.4f}  "
              f"damage_rel {test_metrics['damage_rel']:.4f}   <- never selected on")
    print(f"saved -> {args.out}.pkl")


if __name__ == "__main__":
    main()
