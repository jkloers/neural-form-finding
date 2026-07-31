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
  6. The per-epoch train loss (total + the three Sobolev terms) and every val evaluation are
     recorded into the sidecar `history` block, so the training curves are reconstructable from
     the checkpoint alone rather than from a lost stdout log.
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
    apply_hinge_energy, apply_hinge_force, apply_hinge_failure, sobolev_loss,
    FEAT_KIND_DEFAULT, FEAT_KIND_ETA, FEAT_KIND_RAW)
from nff.rve.hinge_function import STOP_UNCONVERGED, STOP_UNKNOWN, stop_code
# Re-exported, not redefined: diagnostics that only need the split must not have to import JAX to
# get it (see nff/utils/splits.py). Existing `from ...train_hinge_surrogate import split_by_job`
# call sites -- nff/scripts/figures/plot_surrogate_parity.py -- keep working.
from nff.utils.splits import split_by_job  # noqa: F401
from nff.utils.routes import FREE_PATH, ORIGIN_RAY, route_family_mask


def _stop_codes(path, d, job_id):
    """Per-row solver ``stop_code``, or None when the campaign did not record one.

    Datasets written before the column existed carry the reason only in the sidecar manifest's
    per-job list, so back-fill from there -- that makes every EXISTING campaign filterable with no
    re-run. Returning None (rather than a column of ``unknown``) is deliberate: ``unknown`` is itself
    a quarantined code, so guessing would silently drop the entire dataset.
    """
    if "stop_code" in d:
        return d["stop_code"].astype(int)
    try:
        with open(path + ".json") as f:
            jobs = json.load(f).get("jobs") or []
        by_job = {int(r["job_id"]): stop_code(r.get("stop_reason", "unknown"))
                  for r in jobs if "stop_reason" in r}
    except (OSError, ValueError, KeyError):
        return None
    return np.array([by_job.get(int(j), STOP_UNKNOWN) for j in job_id], int) if by_job else None


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
    codes = _stop_codes(path, d, d["job_id"])
    if codes is not None:
        data["stop_code"] = codes
    if "peeq_lig" in d:                       # carried for the oracle-history feasibility probe only
        data["peeq_lig"] = d["peeq_lig"].astype(float)
    return data, eps_f, const


def oracle_history_column(data, kind):
    """The TRUE per-row history variable -- a feasibility ceiling, never available at inference.

    ``W`` is a path functional, so mixing route families makes it multi-valued in ``(u, g)``. An
    internal variable ``z`` would restore single-valuedness. Before building the learned recurrence,
    this answers the cheaper question: does ANY scalar history variable carry enough information?

    ``damage``    accumulated plastic dissipation -- the classical internal variable, monotone.
    ``peeq_lig``  the hottest ligament element's plastic strain.
    ``path_len``  cumulative ``sum |du|`` along the job's own ordered path; the purest history
                  summary, and the only one derivable without any oracle field.
    """
    if kind == "damage":
        return data["margin"].astype(float)
    if kind == "peeq_lig":
        if "peeq_lig" not in data:
            raise KeyError("peeq_lig not in this dataset")
        return data["peeq_lig"]
    if kind != "path_len":
        raise ValueError(kind)
    u, jid = data["u"], data["job_id"]
    out = np.zeros(len(u))
    for j in np.unique(jid):
        m = np.where(jid == j)[0]
        o = m[np.argsort(np.linalg.norm(u[m], axis=1))]     # order along the path
        steps = np.linalg.norm(np.diff(u[o], axis=0), axis=1)
        out[o] = np.concatenate([[0.0], np.cumsum(steps)])
    return out


def converged_mask(data):
    """Rows whose SOLVE is trustworthy ground truth, or all-True when the campaign did not say.

    ``diverged``/``unknown`` rows come from a solve that failed at the buckling bifurcation; they are
    not samples of the same equilibrium branch as the rest (see STOP_UNCONVERGED).
    """
    if "stop_code" not in data:
        return np.ones(len(data["W"]), bool)
    return ~np.isin(data["stop_code"], STOP_UNCONVERGED)


def check_force_sign(data, n_probe=30):
    """Is the stored ``F`` the gradient ``+dW/du``? Return ``(sign, ratio, how)``.

    Two probes, in order of directness:

    ``spine``  pure-rotation rays (a = s = 0): FD ``dW/dtheta`` vs ``M_theta``.
    ``work``   the general identity along ANY ordered path -- ``F.du`` integrated trapezoidally must
               equal ``dW``. Works for a proportional fan and for a free-DOF curve alike.

    The spine probe alone is a TRAP on modern campaigns. It requires a job whose (a, s) are
    identically zero, and the PET campaign has **none** -- its spine is free-DOF (the solver picks
    a, s) and its fan draws eta from a continuous envelope. ``ratios`` came back empty, the old code
    fell back to ``r = 1.0``, and every PET run printed "match" WITHOUT HAVING CHECKED ANYTHING.
    A flipped sign would have had the Sobolev loss fighting itself, silently. Hence ``how``, which
    the caller must report, and the work probe, which actually runs on this data.
    """
    jid, u, W, F = data["job_id"], data["u"], data["W"], data["F"]
    ratios = []
    for j in np.unique(jid):
        m = jid == j
        if m.sum() > 5 and np.abs(u[m, 0]).max() < 1e-9 and np.abs(u[m, 1]).max() < 1e-9:
            th, Wj, M = u[m, 2], W[m], F[m, 2]
            o = np.argsort(th); th, Wj, M = th[o], Wj[o], M[o]
            dWdth = np.diff(Wj) / (np.diff(th) + 1e-12)
            Mmid = 0.5 * (M[:-1] + M[1:])
            good = np.abs(Mmid) > 1e-6
            if good.any():
                ratios.append(np.median(dWdth[good] / Mmid[good]))
        if len(ratios) >= n_probe:
            break
    if ratios:
        r = float(np.median(ratios))
        return (1.0 if r >= 0 else -1.0), r, "spine"

    # No pure-rotation job: fall back to the work identity along each job's own path.
    for j in np.unique(jid):
        m = np.where(jid == j)[0]
        if len(m) < 4:
            continue
        o = m[np.argsort(np.linalg.norm(u[m], axis=1))]
        du, dW = np.diff(u[o], axis=0), np.diff(W[o])
        work = np.sum(0.5 * (F[o][:-1] + F[o][1:]) * du, axis=1)
        good = np.abs(dW) > 1e-9
        if good.any():
            ratios.append(float(np.median(work[good] / dW[good])))
        if len(ratios) >= n_probe:
            break
    if not ratios:
        print("force-sign check: NO usable probe (no pure-rotation ray, no ordered path) "
              "-- force sign UNVERIFIED, assuming F = +dW/du")
        return 1.0, float("nan"), "unverified"
    r = float(np.median(ratios))
    return (1.0 if r >= 0 else -1.0), r, "work"


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
    ap.add_argument("--eval-every", dest="eval_every", type=int, default=10,
                    help="epochs between val evaluations (= the val-curve sampling rate)")
    ap.add_argument("--oracle-history", dest="oracle_history", default=None,
                    help="FEASIBILITY ONLY -- append the TRUE per-row history variable to the "
                         "geometry input. It is not available at inference, so any score is a "
                         "CEILING on an internal-variable model W(u,z;g), never a shippable number. "
                         "Tests whether a scalar history variable carries the information that makes "
                         "the mixed route families single-valued")
    ap.add_argument("--route-family", dest="route_family", default="all",
                    choices=["all", ORIGIN_RAY, FREE_PATH],
                    help="keep only jobs of one route family. `origin_ray` is the subset a STATE "
                         "function can legitimately fit: origin rays foliate the state space, so "
                         "history is recoverable from the state. Mixing in `free_path` (free-DOF) "
                         "jobs, whose curves cross those rays, makes W multi-valued in the net's "
                         "own input -- measured 2.50x cross-family sigma_pair on the PET campaign")
    ap.add_argument("--exclude-unconverged", dest="exclude_unconverged", default="both",
                    choices=["none", "eval", "both"],
                    help="quarantine rows from a DIVERGED solve: 'both' drops them everywhere, "
                         "'eval' keeps them in train but never scores against them, 'none' is the "
                         "pre-2026-07-30 behaviour. They were 4%% of the PET campaign and 50%% of "
                         "its squared error")
    ap.add_argument("--min-updates", dest="min_updates", type=int, default=0,
                    help="floor on TOTAL gradient updates; raises --epochs to reach it. An epoch is "
                         "n_train/batch steps, so the same --epochs buys 7.8x less training on the "
                         "PET campaign (15.9k rows) than on the S235 one (115k) -- set this instead")
    ap.add_argument("--feat-kind", dest="feat_kind", default=FEAT_KIND_DEFAULT,
                    choices=[FEAT_KIND_RAW, FEAT_KIND_ETA],
                    help="network input parameterization: 'eta' = dimensionless (a,s)/w_lig, "
                         "'raw' = the legacy dimensional mm")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hidden", default="64,64",
                    help="comma-sep hidden widths (default compact 64,64; original was 128,128,128). "
                         "At these sizes cost is kernel-launch bound, so WIDENING is much cheaper "
                         "than deepening: 2x128 is 3.2x the params for 2.2x the time, 3x128 3.9x for 5.7x")
    ap.add_argument("--m-out", dest="m_out", type=int, default=32,
                    help="width of the energy net's output vector h; W = W_scale*||h(u)-h(0)||^2")
    ap.add_argument("--force", action="store_true", help="overwrite an existing --out checkpoint")
    args = ap.parse_args()

    if os.path.exists(args.out + ".pkl") and not args.force:
        raise SystemExit(f"{args.out}.pkl exists -- pass --force to overwrite, or pick a new name. "
                         "(Reusing a name is how a .pkl and its sidecar .json end up describing "
                         "different models.)")

    data, eps_f, const = load_dataset(args.data, w_lig_max=args.w_lig_max)
    sign, ratio, how = check_force_sign(data)
    # `how` is reported, not hidden: 'unverified' means the probe never ran, which is what the old
    # code silently did on every PET campaign.
    print(f"force-sign check [{how}]: ratio ~ {ratio:+.3f} -> "
          f"{'match (F = +dW/du)' if sign > 0 else 'FLIPPED (negating F)'}"
          f"{'   <-- NOT ACTUALLY CHECKED' if how == 'unverified' else ''}")
    data["F"] *= sign

    if args.oracle_history:
        # comma-separated -> a k-dimensional z, so k=1 vs k=2 can be compared directly
        kinds = [k.strip() for k in args.oracle_history.split(",") if k.strip()]
        cols = [oracle_history_column(data, k)[:, None] for k in kinds]
        # Appended to g, so it flows through the SAME `_feature_cols` path a learned z would use.
        data["g"] = np.concatenate([data["g"]] + cols, axis=-1)
        print(f"ORACLE history {kinds} (k={len(kinds)}) appended to g -> {data['g'].shape[-1]} geom "
              f"features. This is a CEILING, not a shippable model: z is unavailable at inference.")

    # Route filtering happens BEFORE the split: a family is a property of the campaign design, not
    # of this run's train/val partition, and dropping a whole family afterwards would leave the
    # split sized for a dataset that is no longer there.
    if args.route_family != "all":
        keep = route_family_mask(data["u"][:, 0], data["u"][:, 1], data["u"][:, 2],
                                 data["g"][:, 0], data["job_id"], args.route_family)
        n_jobs_before = len(np.unique(data["job_id"]))
        data = {k: (v[keep] if getattr(v, "shape", (0,))[:1] == keep.shape else v)
                for k, v in data.items()}
        print(f"route family '{args.route_family}': kept {int(keep.sum())}/{keep.size} rows "
              f"({len(np.unique(data['job_id']))}/{n_jobs_before} jobs)")

    split = split_by_job(data, args.val_frac, args.seed, test_frac=args.test_frac)
    tr, va, te = split if len(split) == 3 else (split[0], split[1], np.zeros_like(split[0]))
    # The quarantine is applied AFTER the split, never before: job membership then stays identical
    # across the three policies, so their scores differ only by which rows were used -- which is what
    # makes them comparable at all.
    if args.exclude_unconverged != "none":
        conv = converged_mask(data)
        if "stop_code" not in data:
            print("WARNING: this dataset records no solver stop_reason (no `stop_code` column and no "
                  "usable sidecar manifest) -- cannot quarantine; proceeding as --exclude-unconverged none")
        else:
            drop_eval = int((~conv & (va | te)).sum())
            va, te = va & conv, te & conv
            msg = f"quarantine ({args.exclude_unconverged}): dropped {drop_eval} unconverged val/test rows"
            if args.exclude_unconverged == "both":
                msg += f", {int((~conv & tr).sum())} train rows"
                tr = tr & conv
            print(msg + f"  [{int((~conv).sum())} unconverged of {conv.size} total, "
                        f"{len(np.unique(data['job_id'][~conv]))} jobs]")
    n_tr, n_va, n_te = int(tr.sum()), int(va.sum()), int(te.sum())
    njobs = lambda m: len(np.unique(data["job_id"][m]))
    print(f"samples: {len(data['W'])}  |  train {n_tr} ({njobs(tr)} jobs) / "
          f"val {n_va} ({njobs(va)}) / test {n_te} ({njobs(te)})  -- split by unseen geometry")

    tr_idx = np.where(tr)[0]
    # every g column beyond (w_lig, alpha): fillet_ratio and/or an oracle history variable
    _fillet = ([data["g"][tr, 2 + i] for i in range(data["g"].shape[-1] - 2)]
               if data["g"].shape[-1] >= 3 else None)
    # Target scales from TRAIN only -- passing F/D switches sobolev_loss to fixed, per-component
    # normalization (see there); without them it falls back to per-batch variance.
    stats = compute_norm_stats(data["u"][tr, 0], data["u"][tr, 1], data["u"][tr, 2],
                               data["g"][tr, 0], data["g"][tr, 1], data["W"][tr],
                               fillet_ratio=_fillet, F=data["F"][tr], D=data["margin"][tr],
                               feat_kind=args.feat_kind)
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
    params = init_hinge_surrogate(key, hidden=hidden, m_out=args.m_out,
                                  feat_dim=3 + data["g"].shape[-1])
    steps = max(1, n_tr // args.batch)
    # An "epoch" is a dataset-sized amount of training, so --epochs alone does not fix the amount of
    # OPTIMIZATION: the same 300 epochs is 23.4k updates on the 115k-row S235 campaign and 3.0k on
    # the 15.9k-row PET one. Report the real number, and let --min-updates pin it.
    epochs = args.epochs
    if args.min_updates and epochs * steps < args.min_updates:
        epochs = -(-args.min_updates // steps)          # ceil
        print(f"--min-updates {args.min_updates}: raising epochs {args.epochs} -> {epochs}")
    print(f"schedule: {steps} steps/epoch x {epochs} epochs = {steps * epochs} updates  "
          f"(batch {args.batch}, features '{args.feat_kind}', hidden {hidden}, m_out {args.m_out})")
    sched = optax.cosine_decay_schedule(args.lr, epochs * steps)
    optimizer = optax.adam(sched)
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(params, opt_state, batch):
        (loss, aux), grads = jax.value_and_grad(
            lambda p: sobolev_loss(p, batch, stats, lam=args.lam, w_damage=args.w_damage),
            has_aux=True)(params)
        grads = jax.tree_util.tree_map(lambda x: jnp.nan_to_num(x), grads)  # NaN-safe
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss, aux

    # Selection on the SAME combination the loss optimizes. Selecting on energy alone lets a net
    # with a dead force head win -- and the force is what Stage-2 balances.
    score = lambda m: args.lam * m["energy_rel"] + (1.0 - args.lam) * m["force_rel"]
    best, best_score, best_params, best_epoch = None, np.inf, params, -1
    # Curves: the train row is the mean over the epoch's minibatches (a single last-batch value is
    # noise at batch 1024); the val rows are sparse by --eval-every, hence their own epoch column.
    hist_train, hist_val = [], []
    rng = np.random.default_rng(args.seed)
    for ep in range(epochs):
        perm = rng.permutation(tr_idx)
        ep_terms = np.zeros(4)
        for b in range(steps):
            params, opt_state, loss, aux = train_step(
                params, opt_state, _batch(data, perm[b * args.batch:(b + 1) * args.batch]))
            ep_terms += [float(loss), float(aux["energy"]), float(aux["force"]), float(aux["damage"])]
        ep_terms /= steps
        hist_train.append(dict(epoch=ep, loss=ep_terms[0], energy=ep_terms[1],
                               force=ep_terms[2], damage=ep_terms[3],
                               lr=float(sched((ep + 1) * steps))))
        if ep % args.eval_every == 0 or ep == epochs - 1:
            m = evaluate(params, val_batch, stats)
            tag = ""
            if score(m) < best_score:
                best, best_score, tag = m, score(m), "  <- best"
                best_params = jax.tree_util.tree_map(lambda x: x, params)
                best_epoch = ep
            hist_val.append(dict(epoch=ep, score=score(m), **m))
            print(f"ep {ep:4d}  loss {ep_terms[0]:.4g}  val energy_rel {m['energy_rel']:.4f}  "
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
            "hidden": hidden, "m_out": args.m_out, "lam": args.lam, "w_damage": args.w_damage,
            "feat_kind": args.feat_kind, "exclude_unconverged": args.exclude_unconverged,
            "route_family": args.route_family,
            "epochs": epochs, "updates": steps * epochs, "lr": args.lr, "batch": args.batch,
            "seed": args.seed, "best_epoch": best_epoch,
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
                   "history": {"train": hist_train, "val": hist_val},
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
