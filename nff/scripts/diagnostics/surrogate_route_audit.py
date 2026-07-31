"""Route-family audit: is the hinge surrogate's error a MULTI-VALUED target, not a hard function?

``W`` is fitted as a state function ``W(a, s, theta; g)``, but the oracle is elastoplastic, so the
stored energy is really a path-dependent WORK (``nff.rve.hinge_function.DeploymentPath`` says so
outright). Whether that matters depends entirely on how the campaign sampled its paths:

  * **origin rays** -- ``u(lambda) = lambda * u1``, the whole steel campaign and PET's fan. Rays from
    a common origin FOLIATE the state space: two of them meet only at 0. So the path that reached a
    state is recoverable FROM the state (direction ``u/||u||``, earlier states ``lambda*u``) -- the
    history is a function of the state and ``W`` is single-valued. A state function is exactly the
    right object to fit.
  * **free paths** -- PET's spine sets ``free_dofs=("a","s")`` (``nff/rve/path_prior.py:185``), so
    theta is imposed but ``a``/``s`` are chosen by the solver to minimise energy and read back off
    the rigid driver node (``nff/rve/ccx_solver.py:517-524``). That is a CURVE cutting across the
    foliation: it crosses origin rays belonging to other jobs, and at each crossing ``W`` takes two
    different values. The target is multi-valued in the net's own input, which no architecture and
    no amount of data can fix.

This script decides between those two worlds on data already on disk -- no training, no FEA, and
(unless ``--surrogate`` is given) no JAX:

  1. **classify** every job as ``origin_ray`` or ``free_path``, definitionally, by testing whether
     ``(a, s)`` is proportional to ``theta`` through the origin. Cross-checked against the campaign
     manifest's ``eta_a``/``eta_s``.
  2. **sigma_pair conditioned on family** -- the paired spread in ``W`` between rows of DIFFERENT
     jobs at near-identical geometry and kinematics, split by whether the two rows come from the
     same family or opposite ones, and reported against pair separation so it does not rest on one
     tolerance. ``sigma_pair(cross) >> sigma_pair(same)`` is multi-valuedness, measured directly.
  3. **oracle-feature probe** -- a k-NN baseline given extra columns it would not have at inference:
     ``uz_max`` (a STATE variable) and ``damage`` (accumulated plastic dissipation, a HISTORY
     variable). If ``damage`` collapses the cross-family error where ``uz_max`` does not, the
     missing information is history, and ``damage`` is the internal variable that carries it.
  4. **per-family signed residual** of a trained checkpoint (needs ``--surrogate`` and JAX). Mixing
     two route families makes the net regress to their mean, so it OVER-predicts one and
     UNDER-predicts the other. Opposite-signed bias separates "multi-valued target" from "harder
     function" on its own; a harder function gives symmetric scatter.

Run the steel campaign through it as the CONTROL -- it must come back ~100% ``origin_ray``. If it
does not, the foliation argument is wrong and the diagnosis needs re-deriving.

    conda run -n kgnn_mac python -m nff.scripts.diagnostics.surrogate_route_audit \
        --data data/fea/hinge_dataset_pet_v2 \
        --surrogate data/surrogates/hinge_surrogate_pet_v2

    conda run -n kgnn_mac python -m nff.scripts.diagnostics.surrogate_route_audit \
        --data data/fea/hinge_dataset          # the steel control

Reads the npz/json DIRECTLY rather than through ``train_hinge_surrogate.load_dataset``, which hard
-raises on any dataset predating the ``damage`` column -- and the steel control is exactly such a
dataset. Do not route this through the trainer's loader.
"""

import argparse
import json
import os

import numpy as np

from nff.utils.splits import split_by_job

ORIGIN_RAY = "origin_ray"
FREE_PATH = "free_path"
UNKNOWN = "unknown"

# Pair-matching tolerances. Geometry is matched tightly (the net sees log w_lig and alpha directly);
# kinematics are matched in the DIMENSIONLESS coordinates the oracle samples in, since a 1 mm
# opening means something different on a 5 mm ligament than on a 25 mm one.
DEF_GEOM_TOL = 0.05        # |Delta log w_lig|
DEF_ALPHA_TOL = 3.0        # [deg]
DEF_U_TOL = 0.10           # radius in (a/w_lig, s/w_lig, theta[rad])


# ── loading ─────────────────────────────────────────────────────────────────────

def load_columns(path):
    """``<path>.npz`` + ``<path>.json`` -> (cols, jobs_meta, const), with NO damage requirement.

    Deliberately not :func:`nff.scripts.train_hinge_surrogate.load_dataset`: that one raises on a
    missing ``damage`` column, which is precisely the steel control we need to audit.
    """
    with np.load(path + ".npz") as npz:
        cols = {k: np.asarray(npz[k]) for k in npz.files}
    with open(path + ".json") as f:
        summary = json.load(f)
    return cols, summary.get("jobs", []), summary.get("const", {})


def _finite_row_mask(cols, keys=("a", "s", "theta", "W")):
    """Rows whose core columns are all finite. A diverged solve once reached eta_a = -1.1e42, which
    IS finite -- so this is a NaN guard, not a divergence filter; divergence is handled by job."""
    m = np.ones(len(cols["W"]), bool)
    for k in keys:
        if k in cols:
            m &= np.isfinite(cols[k].astype(float))
    return m


# ── 1. route-family classification ──────────────────────────────────────────────

def _nonproportionality(x, theta, w_lig):
    """Residual fraction of the best origin ray ``x = k*theta``.

    Least squares THROUGH THE ORIGIN, because that is what an origin ray is: the undeformed state is
    implicit at lambda=0 and every sampled state is ``lambda*u1``. The denominator carries a
    ``w_lig`` floor so a pure-rotation spine (``x`` identically 0, residual identically 0) scores 0
    rather than 0/0.
    """
    tt = float(theta @ theta)
    k = float(x @ theta) / tt if tt > 0 else 0.0
    resid = float(np.sqrt(np.mean((x - k * theta) ** 2)))
    return resid / (float(np.sqrt(np.mean(x ** 2))) + 1e-3 * max(float(w_lig), 1e-9))


def classify_jobs(cols, *, tol=0.02, min_rows=3):
    """Per-job route family from the ROWS alone -> (family dict, per-job diagnostics).

    Definitional test, so it needs no manifest and works on any campaign: a job is an ``origin_ray``
    iff both ``a`` and ``s`` are proportional to ``theta`` through the origin. A free-DOF job's
    ``(a, s)`` are solver outputs tracing a curve, so they are not.
    """
    jid = cols["job_id"]
    a, s, th = cols["a"].astype(float), cols["s"].astype(float), cols["theta"].astype(float)
    w = cols["w_lig"].astype(float)
    fam, diag = {}, {}
    for j in np.unique(jid):
        m = jid == j
        if int(m.sum()) < min_rows:
            fam[int(j)] = UNKNOWN
            diag[int(j)] = dict(n_rows=int(m.sum()), nonprop=float("nan"))
            continue
        wl = float(np.median(w[m]))
        na = _nonproportionality(a[m], th[m], wl)
        ns = _nonproportionality(s[m], th[m], wl)
        nonprop = max(na, ns)
        fam[int(j)] = ORIGIN_RAY if nonprop <= tol else FREE_PATH
        diag[int(j)] = dict(n_rows=int(m.sum()), nonprop=float(nonprop),
                            nonprop_a=float(na), nonprop_s=float(ns),
                            w_lig=wl, max_abs_a=float(np.abs(a[m]).max()),
                            theta_max_deg=float(np.degrees(th[m].max())))
    return fam, diag


def manifest_families(jobs_meta, cols):
    """Cross-check label from the campaign manifest: free-DOF spine jobs record ``eta_a == eta_s ==
    0`` (they were built as ``DeploymentRay(th, 0, 0, ..., free_dofs=("a","s"))``) yet their rows
    carry solver-chosen nonzero ``a``. A PRESCRIBED zero-eta spine -- the steel one -- has rows that
    are identically zero. That difference is the whole classifier."""
    jid = cols["job_id"]
    a = cols["a"].astype(float)
    out = {}
    for m in jobs_meta:
        if "job_id" not in m or "eta_a" not in m:
            continue
        j = int(m["job_id"])
        ea, es = m.get("eta_a"), m.get("eta_s")
        if ea is None or es is None or not np.isfinite(float(ea)) or not np.isfinite(float(es)):
            continue                      # a DeploymentPath replay carries a polyline, not a ray
        rows = jid == j
        if not rows.any():
            continue
        zero_eta = abs(float(ea)) < 1e-12 and abs(float(es)) < 1e-12
        out[j] = FREE_PATH if (zero_eta and np.abs(a[rows]).max() > 1e-6) else ORIGIN_RAY
    return out


# ── 2. family-conditioned sigma_pair ────────────────────────────────────────────

def _pair_key(fi, fj):
    return "same_origin_ray" if fi == fj == ORIGIN_RAY else \
           "same_free_path" if fi == fj == FREE_PATH else "cross_family"


def sigma_pair_by_family(cols, fam, *, geom_tol=DEF_GEOM_TOL, alpha_tol=DEF_ALPHA_TOL,
                         u_tol=DEF_U_TOL, max_rows=40000, max_pairs=400000, n_bins=4, seed=0,
                         thickness=None):
    """Paired relative spread in ``W`` between DIFFERENT jobs at matched geometry + kinematics.

    Returns a dict keyed by family pairing, each holding the spread binned by pair SEPARATION -- the
    trend matters more than any single tolerance, since a spread that vanishes as separation -> 0 is
    ordinary interpolation error while one that survives is genuine multi-valuedness.

    Rows are matched in scaled coordinates (each axis divided by its own tolerance) so a single
    radius-1 neighbour query implements the whole tolerance box.

    **The headline number is GRADIENT-CORRECTED.** A raw ``|W_i - W_j|`` between two nearby-but-
    distinct states is dominated by the ordinary first-order change in ``W`` across the gap, which
    has nothing to do with path-dependence -- on a steep ray it swamps the effect entirely. But the
    campaign already stores ``F = dW/du`` (the Sobolev labels, exact by the envelope theorem), so
    the expected difference is predictable:

        Delta_expected = 0.5 * (F_i + F_j) . (u_j - u_i)          [trapezoidal, 2nd-order accurate]

    Subtracting it leaves ``O(|Delta u|^2)`` curvature plus any genuine multi-valuedness. That
    residual is the signal. ``u`` and ``F`` are used RAW (mm, mm, rad against N, N, N.mm) because
    that is the pairing in which ``F`` is work-conjugate to ``u`` and the product is an energy.
    """
    from scipy.spatial import cKDTree

    ok = _finite_row_mask(cols)
    idx = np.where(ok)[0]
    rng = np.random.default_rng(seed)
    if len(idx) > max_rows:
        idx = np.sort(rng.choice(idx, max_rows, replace=False))

    w = np.maximum(cols["w_lig"].astype(float)[idx], 1e-9)
    z = np.stack([
        np.log(w) / geom_tol,
        np.radians(cols["alpha_deg"].astype(float)[idx]) / np.radians(alpha_tol),
        (cols["a"].astype(float)[idx] / w) / u_tol,
        (cols["s"].astype(float)[idx] / w) / u_tol,
        cols["theta"].astype(float)[idx] / u_tol,
    ], axis=-1)

    pairs = cKDTree(z).query_pairs(r=1.0, output_type="ndarray")
    jid = cols["job_id"][idx]
    pairs = pairs[jid[pairs[:, 0]] != jid[pairs[:, 1]]]      # same job = same path, trivially equal
    if len(pairs) > max_pairs:
        pairs = pairs[rng.choice(len(pairs), max_pairs, replace=False)]
    if not len(pairs):
        return {}, 0

    W = cols["W"].astype(float)[idx]
    Wi, Wj = W[pairs[:, 0]], W[pairs[:, 1]]
    scale = 0.5 * (np.abs(Wi) + np.abs(Wj))
    good = scale > 0
    pairs, Wi, Wj, scale = pairs[good], Wi[good], Wj[good], scale[good]
    d_raw = np.abs(Wi - Wj) / scale

    # gradient-corrected: subtract the first-order change across the gap, using the stored dW/du
    have_F = all(k in cols for k in ("F_a", "F_s", "M_theta"))
    if have_F:
        u_raw = np.stack([cols["a"], cols["s"], cols["theta"]], -1).astype(float)[idx]
        F_raw = np.stack([cols["F_a"], cols["F_s"], cols["M_theta"]], -1).astype(float)[idx]
        du = u_raw[pairs[:, 1]] - u_raw[pairs[:, 0]]
        F_mid = 0.5 * (F_raw[pairs[:, 0]] + F_raw[pairs[:, 1]])
        expected = np.sum(np.where(np.isfinite(F_mid), F_mid, 0.0) * du, axis=-1)
        d_rel = np.abs((Wj - Wi) - expected) / scale
    else:
        d_rel = d_raw
    sep = np.linalg.norm(z[pairs[:, 0]] - z[pairs[:, 1]], axis=-1)     # in tolerance units

    famv = np.array([fam.get(int(j), UNKNOWN) for j in jid])
    keys = np.array([_pair_key(famv[i], famv[k]) for i, k in pairs])

    uz_over_t = None
    if thickness and "uz_max" in cols:
        uz_over_t = cols["uz_max"].astype(float)[idx] / float(thickness)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = {}
    for key in ("same_origin_ray", "same_free_path", "cross_family"):
        m = keys == key
        if not m.any():
            continue
        rec = dict(n_pairs=int(m.sum()),
                   # /sqrt(2): the pair difference is the spread of TWO draws; this is the
                   # per-point equivalent, directly comparable to a model's relRMSE.
                   sigma_pair=float(np.sqrt(np.mean(d_rel[m] ** 2)) / np.sqrt(2.0)),
                   sigma_pair_uncorrected=float(np.sqrt(np.mean(d_raw[m] ** 2)) / np.sqrt(2.0)),
                   gradient_corrected=bool(have_F),
                   median_rel_diff=float(np.median(d_rel[m])), by_separation=[])
        for lo, hi in zip(edges[:-1], edges[1:]):
            b = m & (sep >= lo) & (sep < hi)
            rec["by_separation"].append(dict(
                lo=float(lo), hi=float(hi), n=int(b.sum()),
                sigma_pair=float(np.sqrt(np.mean(d_rel[b] ** 2)) / np.sqrt(2.0)) if b.any() else None))
        if uz_over_t is not None:
            rec["by_uz_band"] = []
            for lo, hi in [(0, 1), (1, 5), (5, 15), (15, 30), (30, np.inf)]:
                u_pair = 0.5 * (uz_over_t[pairs[:, 0]] + uz_over_t[pairs[:, 1]])
                b = m & (u_pair >= lo) & (u_pair < hi)
                rec["by_uz_band"].append(dict(
                    lo=float(lo), hi=(None if not np.isfinite(hi) else float(hi)), n=int(b.sum()),
                    sigma_pair=float(np.sqrt(np.mean(d_rel[b] ** 2)) / np.sqrt(2.0)) if b.any() else None))
        out[key] = rec
    return out, int(len(pairs))


# ── 3. oracle-feature probe ─────────────────────────────────────────────────────

def _rel_rmse(pred, true):
    return float(np.sqrt(np.mean((pred - true) ** 2)) / (np.sqrt(np.mean(true ** 2)) + 1e-12))


def oracle_feature_probe(cols, fam, train_mask, test_mask, *, thickness, k=5, max_train=40000,
                         seed=0):
    """k-NN on ``W`` with and without columns the surrogate would NOT have at inference.

    The point is the CONTRAST, not the absolute number:

      * ``+uz_max`` is a STATE variable -- the buckling amplitude at this instant.
      * ``+damage`` is accumulated plastic dissipation, i.e. a HISTORY variable, monotone along the
        path (``nff.rve.damage.plastic_damage``).

    Under the multi-valuedness hypothesis, no state variable can recover the residual (that is why
    feeding the true ``uz_max`` moved an earlier k-NN baseline only 25.1% -> 22.4%), while a history
    variable can -- it is exactly the internal variable that makes ``W(u, z)`` single-valued again.
    A large gap between the two rows below is direct evidence for an internal-variable surrogate.
    """
    from scipy.spatial import cKDTree

    ok = _finite_row_mask(cols)
    W = cols["W"].astype(float)
    w = np.maximum(cols["w_lig"].astype(float), 1e-9)
    base = [cols["a"].astype(float) / w, cols["s"].astype(float) / w,
            cols["theta"].astype(float), np.log(w),
            np.radians(cols["alpha_deg"].astype(float))]

    variants = {"base": base}
    if "uz_max" in cols and thickness:
        variants["+uz_max (state)"] = base + [cols["uz_max"].astype(float) / float(thickness)]
    if "damage" in cols:
        d = cols["damage"].astype(float)
        if np.isfinite(d).any():
            variants["+damage (history)"] = base + [np.nan_to_num(d, nan=0.0)]
    if "uz_max" in cols and "damage" in cols and thickness:
        variants["+both"] = base + [cols["uz_max"].astype(float) / float(thickness),
                                    np.nan_to_num(cols["damage"].astype(float), nan=0.0)]

    rng = np.random.default_rng(seed)
    tr = np.where(train_mask & ok)[0]
    te = np.where(test_mask & ok)[0]
    if len(tr) > max_train:
        tr = np.sort(rng.choice(tr, max_train, replace=False))
    if len(tr) < k + 1 or not len(te):
        return {}

    famv = np.array([fam.get(int(j), UNKNOWN) for j in cols["job_id"]])
    out = {}
    for name, feats in variants.items():
        X = np.stack(feats, axis=-1)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd = np.where(sd < 1e-9, 1.0, sd)               # same convention as models._robust_std
        Xn = (X - mu) / sd
        _, nn = cKDTree(Xn[tr]).query(Xn[te], k=k)
        pred = W[tr][nn].mean(axis=-1) if k > 1 else W[tr][nn]
        rec = dict(rel_rmse=_rel_rmse(pred, W[te]), n_test=int(len(te)))
        for f in (ORIGIN_RAY, FREE_PATH):
            m = famv[te] == f
            if m.any():
                rec[f] = dict(rel_rmse=_rel_rmse(pred[m], W[te][m]), n=int(m.sum()),
                              signed_bias=float(np.mean((pred[m] - W[te][m]) / (np.abs(W[te][m]) + 1e-12))))
        out[name] = rec
    return out


# ── 4. per-family residual of a trained checkpoint ──────────────────────────────

def checkpoint_residuals(surrogate, cols, fam, test_mask, *, thickness):
    """Signed relative residual of a trained checkpoint, split by route family and uz/t band.

    THE discriminating number. A net trained on two route families regresses to their mean, so it
    over-predicts one and under-predicts the other: the biases come out OPPOSITE-SIGNED. A merely
    harder function gives symmetric scatter and near-zero bias in both.
    """
    import jax                                          # local: the rest of this script needs no JAX
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from nff.models.hinge_surrogate import apply_hinge_energy, load_hinge_surrogate

    params, stats, _ = load_hinge_surrogate(surrogate + ".pkl")
    m = test_mask & _finite_row_mask(cols)
    idx = np.where(m)[0]
    u = np.stack([cols["a"], cols["s"], cols["theta"]], -1).astype(float)[idx]
    g_cols = [cols["w_lig"].astype(float)[idx], np.radians(cols["alpha_deg"].astype(float)[idx])]
    if int(stats["feat_mean"].shape[-1]) >= 6:          # checkpoint carries the swept fillet DOF
        g_cols.append(cols["fillet_ratio"].astype(float)[idx])
    g = np.stack(g_cols, -1)

    pred = np.asarray(apply_hinge_energy(params, jnp.asarray(u), jnp.asarray(g), stats))
    true = cols["W"].astype(float)[idx]
    resid = (pred - true) / (np.abs(true) + 1e-12)
    famv = np.array([fam.get(int(j), UNKNOWN) for j in cols["job_id"][idx]])

    def block(mask):
        return dict(n=int(mask.sum()), rel_rmse=_rel_rmse(pred[mask], true[mask]),
                    signed_bias=float(np.mean(resid[mask])),
                    median_signed=float(np.median(resid[mask])))

    out = dict(overall=block(np.ones(len(idx), bool)), by_family={}, by_family_uz={})
    for f in (ORIGIN_RAY, FREE_PATH):
        sel = famv == f
        if sel.any():
            out["by_family"][f] = block(sel)
    if "uz_max" in cols and thickness:
        uz = cols["uz_max"].astype(float)[idx] / float(thickness)
        for f in (ORIGIN_RAY, FREE_PATH):
            bands = []
            for lo, hi in [(0, 1), (1, 5), (5, 15), (15, 30), (30, np.inf)]:
                sel = (famv == f) & (uz >= lo) & (uz < hi)
                bands.append(dict(lo=float(lo), hi=(None if not np.isfinite(hi) else float(hi)),
                                  **(block(sel) if sel.any() else dict(n=0))))
            out["by_family_uz"][f] = bands
    return out, dict(resid=resid, famv=famv)


# ── plots ───────────────────────────────────────────────────────────────────────

def _plot_sigma_pair(sig, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = {"same_origin_ray": "#2b7bba", "same_free_path": "#d1701c", "cross_family": "#b02020"}
    for key, rec in sig.items():
        xs = [0.5 * (b["lo"] + b["hi"]) for b in rec["by_separation"] if b["sigma_pair"] is not None]
        ys = [b["sigma_pair"] for b in rec["by_separation"] if b["sigma_pair"] is not None]
        if xs:
            axes[0].plot(xs, ys, "o-", color=colors.get(key), label=f"{key} (n={rec['n_pairs']})")
        bands = rec.get("by_uz_band", [])
        bx = [i for i, b in enumerate(bands) if b["sigma_pair"] is not None]
        by = [bands[i]["sigma_pair"] for i in bx]
        if bx:
            axes[1].plot(bx, by, "o-", color=colors.get(key), label=key)
    axes[0].set_xlabel("pair separation [tolerance units]")
    axes[0].set_ylabel(r"$\sigma_{pair}$ (relative)")
    axes[0].set_title("paired spread in W vs separation\n(surviving at 0 = multi-valued)")
    axes[1].set_xticks(range(5))
    axes[1].set_xticklabels(["<1", "1-5", "5-15", "15-30", ">30"])
    axes[1].set_xlabel(r"$u_{z,max}/t$ band")
    axes[1].set_title("...stratified by fold depth")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(out_dir, "sigma_pair_by_family.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def _plot_residuals(res_raw, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    resid, famv = res_raw["resid"], res_raw["famv"]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    bins = np.linspace(-1.5, 1.5, 80)
    for f, c in ((ORIGIN_RAY, "#2b7bba"), (FREE_PATH, "#d1701c")):
        m = famv == f
        if m.any():
            ax.hist(np.clip(resid[m], bins[0], bins[-1]), bins=bins, density=True, alpha=0.55,
                    color=c, label=f"{f}  bias={np.mean(resid[m]):+.3f}  n={int(m.sum())}")
    ax.axvline(0.0, color="k", lw=1)
    ax.set_xlabel(r"signed relative residual $(W_{pred}-W_{true})/|W_{true}|$")
    ax.set_ylabel("density")
    ax.set_title("held-out residual by route family\n(opposite-signed bias = mixed families)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(out_dir, "residual_by_family.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


# ── main ────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/fea/hinge_dataset_pet_v2", help="npz/json prefix")
    ap.add_argument("--surrogate", default=None,
                    help="checkpoint prefix for the per-family residual (needs JAX); optional")
    ap.add_argument("--out", default=None, help="output DIRECTORY (default data/outputs/route_audit/<name>)")
    ap.add_argument("--w-lig-max", dest="w_lig_max", type=float, default=25.0,
                    help="mirror the trainer's Saint-Venant row drop so numbers stay comparable")
    ap.add_argument("--val-frac", dest="val_frac", type=float, default=0.15)
    ap.add_argument("--test-frac", dest="test_frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prop-tol", dest="prop_tol", type=float, default=0.02,
                    help="non-proportionality above this = free_path")
    ap.add_argument("--geom-tol", dest="geom_tol", type=float, default=DEF_GEOM_TOL)
    ap.add_argument("--alpha-tol", dest="alpha_tol", type=float, default=DEF_ALPHA_TOL)
    ap.add_argument("--u-tol", dest="u_tol", type=float, default=DEF_U_TOL)
    ap.add_argument("--knn-k", dest="knn_k", type=int, default=5)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    name = os.path.basename(args.data.rstrip("/"))
    out_dir = args.out or os.path.join("data", "outputs", "route_audit", name)
    os.makedirs(out_dir, exist_ok=True)

    cols, jobs_meta, const = load_columns(args.data)
    thickness = const.get("thickness")
    n_all = len(cols["W"])
    if args.w_lig_max is not None:
        keep = cols["w_lig"].astype(float) <= args.w_lig_max
        if not keep.all():
            print(f"w_lig: dropping {int((~keep).sum())}/{keep.size} rows above {args.w_lig_max} mm")
            cols = {k: (v[keep] if v.shape[:1] == keep.shape else v) for k, v in cols.items()}
    print(f"{args.data}: {n_all} rows -> {len(cols['W'])} kept, "
          f"{len(np.unique(cols['job_id']))} jobs, thickness {thickness}")

    # 1. classify
    fam, diag = classify_jobs(cols, tol=args.prop_tol)
    census = {f: sum(1 for v in fam.values() if v == f) for f in (ORIGIN_RAY, FREE_PATH, UNKNOWN)}
    rows_by_fam = {f: int(sum(fam.get(int(j), UNKNOWN) == f for j in cols["job_id"]))
                   for f in (ORIGIN_RAY, FREE_PATH, UNKNOWN)}
    print("\nroute families (jobs / rows):")
    for f in (ORIGIN_RAY, FREE_PATH, UNKNOWN):
        print(f"  {f:12s} {census[f]:5d} jobs  {rows_by_fam[f]:7d} rows")

    man = manifest_families(jobs_meta, cols)
    confusion = {}
    for j, mf in man.items():
        if j in fam:
            confusion[f"{mf}|{fam[j]}"] = confusion.get(f"{mf}|{fam[j]}", 0) + 1
    if confusion:
        print("\nmanifest vs proportionality (manifest|rows: count) -- off-diagonal means "
              ".json and .npz disagree:")
        for k, v in sorted(confusion.items()):
            print(f"  {k:28s} {v}")

    # 2. sigma_pair
    sig, n_pairs = sigma_pair_by_family(cols, fam, geom_tol=args.geom_tol, alpha_tol=args.alpha_tol,
                                        u_tol=args.u_tol, seed=args.seed, thickness=thickness)
    corrected = any(r.get("gradient_corrected") for r in sig.values())
    print(f"\nsigma_pair (relative, per-point equivalent"
          f"{', gradient-corrected' if corrected else ', RAW -- no F columns'}):")
    print(f"  {'pairing':18s} {'corrected':>10s} {'raw':>10s} {'n':>8s}")
    for key, rec in sig.items():
        print(f"  {key:18s} {rec['sigma_pair']:10.4f} {rec['sigma_pair_uncorrected']:10.4f} "
              f"{rec['n_pairs']:8d}")
    if "cross_family" in sig and "same_origin_ray" in sig:
        ratio = sig["cross_family"]["sigma_pair"] / (sig["same_origin_ray"]["sigma_pair"] + 1e-12)
        verdict = "MULTI-VALUED -> path hypothesis carried" if ratio >= 2.0 else \
                  "no strong path signal -> path hypothesis refuted"
        print(f"  --> cross / same_origin_ray = {ratio:.2f}x   {verdict}")
    elif not sig:
        print("  (no matched pairs -- widen --u-tol / --geom-tol)")

    # split by job, reusing the trainer's own splitter so every number stays comparable
    tr, va, te = split_by_job({"job_id": cols["job_id"]}, args.val_frac, args.seed,
                              test_frac=args.test_frac)

    # 3. oracle-feature probe
    print(f"\noracle-feature probe (k-NN, k={args.knn_k}, job-disjoint held-out):")
    probe = oracle_feature_probe(cols, fam, tr, te, thickness=thickness, k=args.knn_k,
                                 seed=args.seed)
    for nm, rec in probe.items():
        extra = "  ".join(f"{f}={rec[f]['rel_rmse']:.3f}" for f in (ORIGIN_RAY, FREE_PATH)
                          if f in rec)
        print(f"  {nm:20s} relRMSE {rec['rel_rmse']:.4f}   {extra}")

    # 4. checkpoint residuals
    ck, res_raw = None, None
    if args.surrogate:
        ck, res_raw = checkpoint_residuals(args.surrogate, cols, fam, te, thickness=thickness)
        print("\ncheckpoint residuals by family (signed bias is the discriminating number):")
        for f, b in ck["by_family"].items():
            print(f"  {f:12s} relRMSE {b['rel_rmse']:.4f}  signed bias {b['signed_bias']:+.4f}  "
                  f"median {b['median_signed']:+.4f}  n={b['n']}")
        bl = [b["signed_bias"] for b in ck["by_family"].values()]
        if len(bl) == 2:
            print(f"  --> biases are {'OPPOSITE-signed (mixed families)' if bl[0] * bl[1] < 0 else 'same-signed'}")

    figs = []
    if not args.no_plots:
        if sig:
            figs.append(_plot_sigma_pair(sig, out_dir))
        if res_raw is not None:
            figs.append(_plot_residuals(res_raw, out_dir))

    report = dict(dataset=args.data, surrogate=args.surrogate, n_rows=int(len(cols["W"])),
                  n_jobs=int(len(np.unique(cols["job_id"]))), thickness=thickness,
                  tolerances=dict(prop=args.prop_tol, geom=args.geom_tol, alpha=args.alpha_tol,
                                  u=args.u_tol),
                  census_jobs=census, census_rows=rows_by_fam,
                  manifest_confusion=confusion, sigma_pair=sig, n_pairs=n_pairs,
                  oracle_probe=probe, checkpoint=ck,
                  per_job=diag)
    with open(os.path.join(out_dir, "route_audit.json"), "w") as f:
        json.dump(report, f, indent=2, default=float)

    for p in figs:
        print(f"  wrote {p}")
    print(f"\nDone. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
