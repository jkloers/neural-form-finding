"""Sweep the surrogate's ENERGY-net capacity, scored on held-out error AND Stage-2 cost.

Every architecture arm to date was run by hand, one ``--out`` name each, and never below 2x128 on
the ``origin_ray`` family -- so "is the net as small as it could be" has never been answered with
data. It matters more than a plain accuracy sweep suggests, because Stage-2 does not pay for the
network once per evaluation: the implicit-diff backward solve differentiates ``W`` TWICE, so the
Hessian-vector product is on the inner CG loop of every load step of every epoch. A 5x hvp is a 5x
tax on the whole design optimization, and it has to be weighed against the accuracy it buys.

So each arm reports both halves: held-out relRMSE from the trainer, and measured ``W`` / ``grad`` /
``hvp`` wall-clock at the deployment's hinge count.

The DAMAGE head is deliberately NOT swept here. ``params["energy"]`` and ``params["fail"]`` are
disjoint MLPs, so energy and force come out bit-identical across fail-head arms -- visible in the
existing sidecars, where ``pb_big`` / ``pb2_dmg`` / ``pb2_dmg2`` share ``energy_rel`` to 16 digits.
Sweeping the two jointly costs 3x the compute for zero extra information. Use ``--stage fail`` to
sweep the damage head afterwards, on the winning energy arm.

Arms are budget-matched by ``--min-updates`` rather than ``--epochs``: an epoch is
``n_train/batch`` steps, so equal epochs would hand the bigger grids a different amount of training.

    JAX_PLATFORMS=cpu conda run -n kgnn_mac python -m nff.scripts.surrogate.sweep_surrogate_capacity \
        --data data/fea/hinge_dataset_pet_merged --out-dir data/surrogates/capacity
"""
import argparse
import json
import os
import subprocess
import sys
import time

# Must precede the jax import: the Metal backend cannot carry float64, and the checkpoints are
# differentiated twice downstream.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from nff.models.hinge_surrogate import (FEAT_KIND_ETA, apply_hinge_energy, init_hinge_surrogate)

# (hidden, m_out) arms. The diagonal from the smallest plausible net up to the current best, plus
# off-diagonal probes that separate "wider trunk" from "wider readout" -- the two grow the parameter
# count very differently, and only the trunk grows the hvp much.
ENERGY_ARMS = [
    ((32, 32), 16),
    ((32, 32), 32),
    ((64, 64), 32),
    ((64, 64), 64),
    ((128, 128), 32),
    ((128, 128), 64),      # the smallest arm ever run on origin_ray: 3.33% val / 2.25% test
    ((128, 128), 128),
    ((256, 256), 64),
    ((256, 256), 128),
    ((256, 256, 256), 128),  # the current best: 2.72% val / 1.63% test
]

FAIL_ARMS = [(64, 64), (128, 128), (256, 256), (128, 128, 128)]

N_HINGES_DEPLOY = 12   # the 3x3 RDPQK standard sheet -- what the timing probe should reflect


def _tag(hidden, m_out):
    return f"h{'x'.join(str(h) for h in hidden)}_m{m_out}"


def _param_count(tree):
    return int(sum(np.prod(np.shape(x)) for x in jax.tree_util.tree_leaves(tree)))


def measure_stage2_cost(hidden, m_out, *, n_hinges=N_HINGES_DEPLOY, n_reps=200, seed=0):
    """Time the three surrogate calls Stage-2 actually makes, at the deployment's hinge count.

    ``hvp`` is the one that dominates: jaxopt's implicit diff solves ``H dx = b`` on the energy
    Hessian, so the network is differentiated twice per CG iteration. Timed on untrained weights --
    cost depends on shapes, not on values, and this keeps the probe independent of the training run.
    """
    key = jax.random.PRNGKey(seed)
    params = init_hinge_surrogate(key, hidden=hidden, m_out=m_out, fail_hidden=(64, 64))
    stats = dict(feat_mean=jnp.zeros(5), feat_std=jnp.ones(5), W_scale=1.0, feat_kind=FEAT_KIND_ETA)
    u = jax.random.normal(key, (n_hinges, 3)) * 0.1
    g = jnp.stack([jnp.full((n_hinges,), 12.0), jnp.full((n_hinges,), 1.5)], axis=-1)

    total_energy_fn = lambda p, uu: jnp.sum(apply_hinge_energy(p, uu, g, stats))
    energy_fn = jax.jit(total_energy_fn)
    force_fn = jax.jit(jax.grad(total_energy_fn, argnums=1))
    hvp_fn = jax.jit(lambda p, uu, v: jax.jvp(lambda x: jax.grad(total_energy_fn, argnums=1)(p, x),
                                              (uu,), (v,))[1])
    v = jnp.ones_like(u)

    calls = [("W_us", energy_fn, (params, u)),
             ("grad_us", force_fn, (params, u)),
             ("hvp_us", hvp_fn, (params, u, v))]
    out = {}
    for name, fn, argv in calls:
        fn(*argv).block_until_ready()                      # compile before timing
        t0 = time.perf_counter()
        for _ in range(n_reps):
            fn(*argv).block_until_ready()
        out[name] = (time.perf_counter() - t0) / n_reps * 1e6
    out["n_params_energy"] = _param_count(params["energy"])
    return out


def run_arm(cmd_args, out_path, *, dry_run=False):
    """Shell one training run. The trainer stays the single source of truth for training."""
    cmd = [sys.executable, "-m", "nff.scripts.train_hinge_surrogate", "--out", out_path, *cmd_args]
    print("  $ " + " ".join(cmd), flush=True)
    if dry_run:
        return None
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          env={**os.environ, "JAX_PLATFORMS": "cpu"})
    wall = time.perf_counter() - t0
    if proc.returncode != 0:
        print(proc.stdout[-3000:])
        print(proc.stderr[-3000:], file=sys.stderr)
        raise RuntimeError(f"training arm failed ({out_path}), exit {proc.returncode}")
    with open(out_path + ".json") as f:
        sidecar = json.load(f)
    return dict(train_wall_s=wall, val_metrics=sidecar["val_metrics"],
                test_metrics=sidecar.get("test_metrics"), meta=sidecar["meta"])


def composite_score(metrics, lam):
    """The trainer's own selection score: ``lam*energy + (1-lam)*force``.

    Energy alone is the wrong objective for a Stage-2 surrogate. The force IS ``dW/du``, i.e. the
    equilibrium residual the solver drives to zero, so a net with an excellent energy fit and a
    sloppy gradient finds the wrong equilibrium. Measured on the first PET capacity sweep, force
    spans 14.0-28.4% test while energy spans only 1.7-5.0% -- capacity buys far more gradient than
    it buys value, which an energy-only rule cannot see.
    """
    return lam * metrics["energy_rel"] + (1.0 - lam) * metrics["force_rel"]


def select_winner(results, *, lam=0.5, target=0.05, slack=0.01, key="composite"):
    """Cheapest arm inside the accuracy band; measured hvp breaks ties.

    Two rules are reported, never one, because they disagree and the disagreement is the finding:

    ``energy``    -- test energy relRMSE at or under ``target`` and within ``slack`` of the best.
    ``composite`` -- same band logic on ``lam*energy + (1-lam)*force`` (no ``target``; the 5% figure
                     is an energy target and does not transfer to a blended score).

    ⚠ The energy-only rule was the one fixed before the first sweep ran, and it was a specification
    error rather than a conservative choice: it omits the very quantity Stage-2 balances. It is kept
    so the two can be compared, not because it is right.
    """
    scored = [r for r in results if r.get("test_metrics")]
    if not scored:
        return None
    if key == "energy":
        vals = {id(r): r["test_metrics"]["energy_rel"] for r in scored}
        band = [r for r in scored if vals[id(r)] <= target and vals[id(r)] <= min(vals.values()) + slack]
    else:
        vals = {id(r): composite_score(r["test_metrics"], lam) for r in scored}
        band = [r for r in scored if vals[id(r)] <= min(vals.values()) + slack]
    return min(band, key=lambda r: r["cost"]["hvp_us"]) if band else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/fea/hinge_dataset_pet_merged")
    ap.add_argument("--out-dir", dest="out_dir", default="data/surrogates/capacity")
    ap.add_argument("--stage", default="energy", choices=["energy", "fail"],
                    help="'energy' sweeps the trunk; 'fail' sweeps the DAMAGE head only, on the "
                         "architecture given by --hidden/--m-out (run it second, on the winner)")
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--w-damage", dest="w_damage", type=float, default=0.1)
    ap.add_argument("--min-updates", dest="min_updates", type=int, default=26000,
                    help="budget-matched to every prior PET arm (26 009 updates)")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seeds", default="0", help="comma-separated; >1 seed measures run-to-run "
                                                 "spread, which decides whether a 0.5 pt gap is real")
    ap.add_argument("--hidden", default="256,256,256", help="--stage fail only: the fixed trunk")
    ap.add_argument("--m-out", dest="m_out", type=int, default=128, help="--stage fail only")
    ap.add_argument("--arms", default=None,
                    help="restrict to these arm tags, comma-separated (e.g. h64x64_m32,"
                         "h256x256x256_m128) -- for a multi-seed rerun of the finalists only")
    ap.add_argument("--dry-run", dest="dry_run", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    common = ["--data", args.data, "--route-family", "origin_ray",
              "--exclude-unconverged", "both", "--feat-kind", FEAT_KIND_ETA,
              "--lam", str(args.lam), "--w-damage", str(args.w_damage),
              "--min-updates", str(args.min_updates), "--batch", str(args.batch),
              "--lr", str(args.lr), "--force"]

    if args.stage == "energy":
        arms = [(h, m, (64, 64)) for h, m in ENERGY_ARMS]
    else:
        trunk = tuple(int(x) for x in args.hidden.split(","))
        arms = [(trunk, args.m_out, fh) for fh in FAIL_ARMS]
    if args.arms:
        keep = {t.strip() for t in args.arms.split(",") if t.strip()}
        arms = [a for a in arms if _tag(a[0], a[1]) in keep]
        missing = keep - {_tag(a[0], a[1]) for a in arms}
        if missing:
            raise SystemExit(f"unknown arm tag(s): {sorted(missing)}")

    results = []
    for hidden, m_out, fail_hidden in arms:
        cost = measure_stage2_cost(hidden, m_out)
        for seed in seeds:
            tag = _tag(hidden, m_out)
            if args.stage == "fail":
                tag += "_f" + "x".join(str(x) for x in fail_hidden)
            if len(seeds) > 1:
                tag += f"_s{seed}"
            out_path = os.path.join(args.out_dir, tag)
            print(f"[{tag}] energy params {cost['n_params_energy']}  hvp {cost['hvp_us']:.0f} us",
                  flush=True)
            arm_args = [*common, "--seed", str(seed),
                        "--hidden", ",".join(str(h) for h in hidden),
                        "--fail-hidden", ",".join(str(h) for h in fail_hidden),
                        "--m-out", str(m_out)]
            trained = run_arm(arm_args, out_path, dry_run=args.dry_run)
            row = dict(tag=tag, hidden=list(hidden), m_out=m_out, fail_hidden=list(fail_hidden),
                       seed=seed, checkpoint=out_path, cost=cost)
            if trained:
                row.update(trained)
                v, t = row["val_metrics"], row["test_metrics"] or {}
                print(f"    val E {v['energy_rel']:.4f} F {v['force_rel']:.4f} D {v['damage_rel']:.4f}"
                      f" | test E {t.get('energy_rel', float('nan')):.4f}"
                      f" | {row['train_wall_s']:.0f}s", flush=True)
            results.append(row)

    winners = {k: select_winner(results, lam=args.lam, key=k) for k in ("energy", "composite")}
    summary = dict(stage=args.stage, data=args.data, lam=args.lam, min_updates=args.min_updates,
                   batch=args.batch, lr=args.lr, seeds=seeds, n_hinges_probe=N_HINGES_DEPLOY,
                   selection_rule="cheapest hvp within 0.01 of the best arm; reported under BOTH "
                                  "the energy-only rule and the composite lam*energy+(1-lam)*force "
                                  "-- the composite is the one that reflects what Stage-2 balances",
                   winner_energy_rule=winners["energy"]["tag"] if winners["energy"] else None,
                   winner_composite_rule=winners["composite"]["tag"] if winners["composite"] else None,
                   arms=results)
    path = os.path.join(args.out_dir, f"sweep_{args.stage}.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {path}")
    for rule, w in winners.items():
        if w is None:
            print(f"[{rule}] no arm met the rule -- report that, do not relax it after the fact")
            continue
        t = w["test_metrics"]
        print(f"[{rule}] WINNER {w['tag']}: test energy {t['energy_rel']:.4f} force {t['force_rel']:.4f}"
              f" (composite {composite_score(t, args.lam):.4f}), hvp {w['cost']['hvp_us']:.0f} us, "
              f"energy params {w['cost']['n_params_energy']}")


if __name__ == "__main__":
    main()
