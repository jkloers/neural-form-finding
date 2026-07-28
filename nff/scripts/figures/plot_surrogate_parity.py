"""Predicted-vs-true parity plot for the condensed hinge-energy surrogate.

Scientifically-honest generalization view: the surrogate is scored on the SAME held-out
split it was validated on -- geometries (jobs) never seen in training, split_by_job(seed=0,
val_frac=0.15) -- so no ray-correlated leakage inflates the fit. Three parity panels:

    energy  W(u; g)            [N.mm]
    force   dW/du = (F_a, F_s, M_theta)   [N, N, N.mm]  (envelope theorem)
    damage  D(u; g)            [-]        (ductile-failure head)

Each panel: held-out points colored (density via alpha), the faint training cloud behind
for context, the y=x identity line, and a corner box with R^2 and relative RMSE. Per the
project viz charter -- no values in titles, Princeton palette, minimal chrome.

    JAX_PLATFORMS=cpu conda run -n kgnn_mac python nff/scripts/figures/plot_surrogate_parity.py \
        --surrogate data/surrogates/hinge_surrogate_2x64_lam065 \
        --data data/fea/hinge_dataset --out data/surrogates/parity_2x64_lam065.png
"""

import argparse

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt

from nff.models.hinge_surrogate import (load_hinge_surrogate, apply_hinge_energy,
    apply_hinge_force, apply_hinge_failure)
from nff.scripts.train_hinge_surrogate import load_dataset, split_by_job, check_force_sign

# ── project charter (Princeton palette) ─────────────────────────────────────────
ORANGE, TEAL, RED, GREY, INK = "#F58025", "#2A9D8F", "#D62828", "#6C757D", "#1A1A1A"
TRAIN = "#C9CED4"   # faint background cloud


def apply_charter():
    plt.rcParams.update({
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.6,
        "axes.edgecolor": INK, "axes.linewidth": 0.9, "axes.labelcolor": INK,
        "text.color": INK, "xtick.color": GREY, "ytick.color": GREY,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "legend.frameon": True, "legend.framealpha": 0.9, "legend.edgecolor": "#D3D6DB",
    })


def _r2(true, pred):
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - np.mean(true)) ** 2) + 1e-30
    return 1.0 - ss_res / ss_tot


def _rel_rmse(true, pred):
    return np.sqrt(np.mean((pred - true) ** 2)) / (np.sqrt(np.mean(true ** 2)) + 1e-12)


def _finite(*arrs):
    """Common finite mask across all arrays (drops NaN/Inf targets, e.g. undefined damage labels)."""
    m = np.ones(len(arrs[0]), bool)
    for a in arrs:
        m &= np.isfinite(a)
    return m


def parity_panel(ax, true_tr, pred_tr, true_va, pred_va, color, xlabel, ylabel):
    """One parity panel: faint train cloud, colored held-out cloud, identity line, metric box."""
    mtr, mva = _finite(true_tr, pred_tr), _finite(true_va, pred_va)
    true_tr, pred_tr = true_tr[mtr], pred_tr[mtr]
    true_va, pred_va = true_va[mva], pred_va[mva]
    lo = min(true_tr.min(), true_va.min(), pred_va.min())
    hi = max(true_tr.max(), true_va.max(), pred_va.max())
    pad = 0.03 * (hi - lo + 1e-12)
    lims = (lo - pad, hi + pad)

    ax.plot(lims, lims, color=INK, lw=1.1, ls="--", zorder=1, alpha=0.8)
    ax.scatter(true_tr, pred_tr, s=4, c=TRAIN, alpha=0.30, linewidths=0, zorder=2,
               rasterized=True, label="train")
    ax.scatter(true_va, pred_va, s=6, c=color, alpha=0.35, linewidths=0, zorder=3,
               rasterized=True, label="held-out")

    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_aspect("equal", "box")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)

    r2, rr = _r2(true_va, pred_va), _rel_rmse(true_va, pred_va)
    ax.text(0.045, 0.955, f"$R^2 = {r2:.4f}$\nrel. RMSE $= {rr*100:.2f}\\%$\n$n = {len(true_va):,}$",
            transform=ax.transAxes, va="top", ha="left", fontsize=9.5, color=INK,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#D3D6DB", lw=0.9, alpha=0.92))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--surrogate", default="data/surrogates/hinge_surrogate_2x64_lam065")
    ap.add_argument("--data", default="data/fea/hinge_dataset")
    ap.add_argument("--out", default="data/surrogates/parity.png")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-frac", dest="val_frac", type=float, default=0.15)
    args = ap.parse_args()

    params, stats, _ = load_hinge_surrogate(args.surrogate + ".pkl")
    data, _ = load_dataset(args.data)
    # reproduce the training force-sign convention so the force parity is apples-to-apples
    sign, _ = check_force_sign(data)
    data["F"] *= sign
    tr, va = split_by_job(data, args.val_frac, args.seed)
    print(f"loaded {args.surrogate} | train {int(tr.sum())} / held-out {int(va.sum())} "
          f"({len(np.unique(data['job_id'][va]))} unseen jobs)")

    u, g = jnp.asarray(data["u"]), jnp.asarray(data["g"])
    W_pred = np.asarray(apply_hinge_energy(params, u, g, stats))
    F_pred = np.asarray(apply_hinge_force(params, u, g, stats))
    D_pred = np.asarray(apply_hinge_failure(params, u, g, stats))
    W_true, F_true, D_true = data["W"], data["F"], data["margin"]

    apply_charter()
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.2))

    # energy
    parity_panel(axes[0], W_true[tr], W_pred[tr], W_true[va], W_pred[va], ORANGE,
                 r"true energy  $W$  [N$\cdot$mm]", r"predicted  $\widehat{W}$  [N$\cdot$mm]")
    axes[0].set_title("Stored energy", pad=10, fontsize=13, color=INK)

    # force: F_a, F_s (N) and M_theta (N.mm) live on different scales, so each component is
    # normalized by its own true-RMS -> one dimensionless parity where all three fits are visible
    # on equal footing (identity still means pred=true). Per-component R^2 goes in the legend.
    Fc = [(0, TEAL, r"$F_a$"), (1, RED, r"$F_s$"), (2, GREY, r"$M_\theta$")]
    rms = np.sqrt(np.mean(F_true[va] ** 2, axis=0)) + 1e-12          # (3,) per-component scale
    Ft, Fp = F_true[va] / rms, F_pred[va] / rms
    lo = min(Ft.min(), Fp.min()); hi = max(Ft.max(), Fp.max())
    pad = 0.03 * (hi - lo); lims = (lo - pad, hi + pad)
    axes[1].plot(lims, lims, color=INK, lw=1.1, ls="--", zorder=1, alpha=0.8)
    for j, col, lab in Fc:
        r2j = _r2(Ft[:, j], Fp[:, j])
        axes[1].scatter(Ft[:, j], Fp[:, j], s=6, c=col, alpha=0.35, linewidths=0,
                        rasterized=True, label=f"{lab}  ($R^2\\!=\\!{r2j:.3f}$)", zorder=3)
    axes[1].set_xlim(lims); axes[1].set_ylim(lims); axes[1].set_aspect("equal", "box")
    axes[1].set_xlabel(r"true force / component RMS")
    axes[1].set_ylabel(r"predicted force / component RMS")
    rr = _rel_rmse(F_true[va].ravel(), F_pred[va].ravel())
    axes[1].text(0.045, 0.955, f"$R^2 = {_r2(Ft.ravel(), Fp.ravel()):.4f}$\nrel. RMSE $= {rr*100:.2f}\\%$",
                 transform=axes[1].transAxes, va="top", ha="left", fontsize=9.5, color=INK,
                 bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#D3D6DB", lw=0.9, alpha=0.92))
    axes[1].legend(loc="lower right", fontsize=9.0, handletextpad=0.3, borderpad=0.5, markerscale=1.6)
    axes[1].set_title(r"Internal force  $\partial W/\partial u$", pad=10, fontsize=13, color=INK)

    # damage
    parity_panel(axes[2], D_true[tr], D_pred[tr], D_true[va], D_pred[va], ORANGE,
                 r"true damage  $D$", r"predicted  $\widehat{D}$")
    axes[2].set_title("Ductile-failure head", pad=10, fontsize=13, color=INK)

    axes[0].legend(loc="lower right", fontsize=9.5, handletextpad=0.3, borderpad=0.5,
                   markerscale=2.2)

    fig.tight_layout(w_pad=2.4)
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"saved -> {args.out}")
    print(f"  energy  R2={_r2(W_true[va],W_pred[va]):.4f}  relRMSE={_rel_rmse(W_true[va],W_pred[va])*100:.2f}%")
    print(f"  force   R2={_r2(F_true[va].ravel(),F_pred[va].ravel()):.4f}  "
          f"relRMSE={_rel_rmse(F_true[va].ravel(),F_pred[va].ravel())*100:.2f}%")
    dm = _finite(D_true[va], D_pred[va])
    print(f"  damage  R2={_r2(D_true[va][dm],D_pred[va][dm]):.4f}  "
          f"relRMSE={_rel_rmse(D_true[va][dm],D_pred[va][dm])*100:.2f}%  (n={int(dm.sum()):,})")


if __name__ == "__main__":
    main()
