"""Is the internal variable ``z`` REALIZABLE — can it be rebuilt causally from ``(u, du)`` alone?

Sufficiency (feeding the true ``z`` as an oracle input) is only half the feasibility question. At
inference the Stage-2 solver has no oracle: it marches load increments and must carry ``z`` forward
itself. So ``z`` is usable only if it obeys a causal recurrence

    z_{k+1} = z_k + softplus(f(u_k, du_k, z_k; g))        z_0 = 0

driven by quantities the solver already has. ``softplus`` makes the increment non-negative, so ``z``
is monotone by construction -- which is what an accumulated-dissipation variable must be, and which
also fixes the defect that a state-function damage head "forgets" permanent set when a hinge swings
out and returns.

The test: fit that recurrence to the observed ``damage`` sequence within each job, then on HELD-OUT
JOBS integrate it forward from ``z_0 = 0`` -- never seeing the true ``z`` again -- and compare the
whole trajectory. Free-running integration, not one-step prediction: one-step error is optimistic
because it re-anchors on the truth at every step, while Stage-2 will accumulate its own drift.

    JAX_PLATFORMS=cpu conda run -n kgnn_mac \
        python -m nff.scripts.diagnostics.probe_history_recurrence
"""
import argparse
import os

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import optax

from nff.models.hinge_surrogate import _init_mlp, _mlp
from nff.scripts.train_hinge_surrogate import load_dataset, oracle_history_column
from nff.utils.routes import FREE_PATH, ORIGIN_RAY, classify_jobs
from nff.utils.splits import split_by_job


def build_sequences(data, z):
    """Per job, rows ordered along the path -> (u, du, z) sequences."""
    seqs = []
    jid = data["job_id"]
    for j in np.unique(jid):
        m = np.where(jid == j)[0]
        if len(m) < 3:
            continue
        o = m[np.argsort(np.linalg.norm(data["u"][m], axis=1))]
        u = data["u"][o]
        seqs.append(dict(job=int(j), u=u, g=data["g"][o], z=z[o],
                         du=np.diff(u, axis=0, prepend=np.zeros((1, 3)))))
    return seqs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/fea/hinge_dataset_pet_merged")
    ap.add_argument("--z", default="damage", help="history variable to reconstruct")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--hidden", default="64,64")
    ap.add_argument("--family", default=None, choices=[ORIGIN_RAY, FREE_PATH],
                    help="restrict TRAINING to one route family -- distinguishes 'z is not causal' "
                         "from 'the probe was under-powered on a minority family'")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data, _, _ = load_dataset(args.data, w_lig_max=25.0)
    z_true = oracle_history_column(data, args.z)
    z_scale = float(np.sqrt(np.mean(z_true ** 2)) + 1e-12)
    fam, _ = classify_jobs(data["u"][:, 0], data["u"][:, 1], data["u"][:, 2],
                           data["g"][:, 0], data["job_id"])

    tr, va = split_by_job(data, 0.15, args.seed)
    tr_jobs = set(np.unique(data["job_id"][tr]).tolist())
    seqs = build_sequences(data, z_true / z_scale)
    train_seqs = [s for s in seqs if s["job"] in tr_jobs
                  and (args.family is None or fam.get(s["job"]) == args.family)]
    val_seqs = [s for s in seqs if s["job"] not in tr_jobs
                and (args.family is None or fam.get(s["job"]) == args.family)]
    print(f"z='{args.z}'  scale {z_scale:.4g}   train {len(train_seqs)} jobs / val {len(val_seqs)}")

    # feature vector for f: (u, du, z, log w_lig, alpha) -- exactly what Stage-2 has at a step
    def feats(u, du, z, g):
        return jnp.concatenate([u, du, z[:, None], jnp.log(g[:, :1]), g[:, 1:2]], axis=-1)

    hidden = tuple(int(x) for x in args.hidden.split(","))
    params = _init_mlp(jax.random.PRNGKey(args.seed), [9, *hidden, 1])

    # TEACHER FORCING for the fit (one-step increments), FREE RUNNING for the verdict.
    U = jnp.asarray(np.concatenate([s["u"][:-1] for s in train_seqs]))
    DU = jnp.asarray(np.concatenate([s["du"][1:] for s in train_seqs]))
    Z = jnp.asarray(np.concatenate([s["z"][:-1] for s in train_seqs]))
    G = jnp.asarray(np.concatenate([s["g"][:-1] for s in train_seqs]))
    DZ = jnp.asarray(np.concatenate([np.diff(s["z"]) for s in train_seqs]))

    def loss(p):
        dz = jax.nn.softplus(_mlp(feats(U, DU, Z, G), p)[..., 0])
        return jnp.mean((dz - DZ) ** 2) / (jnp.mean(DZ ** 2) + 1e-12)

    opt = optax.adam(optax.cosine_decay_schedule(3e-3, args.steps))
    ost = opt.init(params)

    @jax.jit
    def step(p, ost):
        l, g_ = jax.value_and_grad(loss)(p)
        up, ost = opt.update(g_, ost, p)
        return optax.apply_updates(p, up), ost, l
    for i in range(args.steps):
        params, ost, l = step(params, ost)
    print(f"one-step (teacher-forced) train loss {float(l):.4g}")

    # FREE-RUNNING integration on held-out jobs: z_0 = 0, never re-anchored on truth
    def integrate(s):
        z, out = 0.0, []
        for k in range(len(s["u"])):
            out.append(z)
            if k + 1 < len(s["u"]):
                f = feats(jnp.asarray(s["u"][k:k + 1]), jnp.asarray(s["du"][k + 1:k + 2]),
                          jnp.asarray([z]), jnp.asarray(s["g"][k:k + 1]))
                z = z + float(jax.nn.softplus(_mlp(f, params)[0, 0]))
        return np.array(out)

    rows = {}
    for s in val_seqs:
        pred, true = integrate(s), s["z"]
        rows.setdefault(fam.get(s["job"], "unknown"), []).append((pred, true))
    print(f"\nFREE-RUNNING reconstruction of z on held-out jobs (relRMSE over the trajectory):")
    print(f"  {'family':<12}{'jobs':>6}{'relRMSE':>10}{'final-value err':>18}")
    for f in (ORIGIN_RAY, FREE_PATH):
        if f not in rows:
            continue
        p = np.concatenate([r[0] for r in rows[f]]); t = np.concatenate([r[1] for r in rows[f]])
        fin_p = np.array([r[0][-1] for r in rows[f]]); fin_t = np.array([r[1][-1] for r in rows[f]])
        rel = np.sqrt(np.mean((p - t) ** 2)) / (np.sqrt(np.mean(t ** 2)) + 1e-12)
        fin = np.sqrt(np.mean((fin_p - fin_t) ** 2)) / (np.sqrt(np.mean(fin_t ** 2)) + 1e-12)
        print(f"  {f:<12}{len(rows[f]):6d}{rel*100:9.2f}%{fin*100:17.2f}%")
    print("\n  <10% => z is realizable causally; the recurrence can be learned.")
    print("  >30% => z is NOT reconstructible from (u, du) alone; a different z is needed.")


if __name__ == "__main__":
    main()
