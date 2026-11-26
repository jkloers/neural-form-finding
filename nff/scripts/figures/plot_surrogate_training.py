"""Training curves for the condensed hinge-energy surrogate.

Reads the ``history`` block written by ``nff/scripts/train_hinge_surrogate.py`` into the
checkpoint sidecar, so the curves come from the checkpoint itself rather than from a stdout
log that may no longer exist. Three panels:

    train    the Sobolev loss and its three WEIGHTED contributions (lam, 1-lam, w_damage),
             which is what actually sums to the total -- the raw aux terms do not.
    val      relative error per head on the held-out geometries, with the selected epoch marked.
    force    the same val force error split per component -- a good moment fit hiding a dead
             dW/da or dW/ds is the exact failure the per-component normalization exists to catch.

    JAX_PLATFORMS=cpu conda run -n kgnn_mac python nff/scripts/figures/plot_surrogate_training.py \
        --surrogate data/surrogates/hinge_surrogate_pet_merged \
        --out data/surrogates/training_pet_merged.png
"""

import argparse
import json

import numpy as np

from nff.utils.figstyle import ORANGE, TEAL, RED, GREY, INK, use_agg, apply_charter

use_agg()
import matplotlib.pyplot as plt  # noqa: E402  (backend must be selected first)


def _col(rows, key):
    return np.array([r[key] for r in rows], float)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--surrogate", default="data/surrogates/hinge_surrogate_pet_merged",
                    help="checkpoint stem; the sidecar <stem>.json carries the history")
    ap.add_argument("--out", default="data/surrogates/training.png")
    args = ap.parse_args()

    side = json.load(open(args.surrogate + ".json"))
    hist = side.get("history")
    if not hist:
        raise SystemExit(
            f"{args.surrogate}.json carries no `history` block -- it predates history recording. "
            "Retrain with nff/scripts/train_hinge_surrogate.py to produce the curves.")
    tr, va = hist["train"], hist["val"]
    meta = side["meta"]
    lam, w_dam = float(meta["lam"]), float(meta["w_damage"])
    best_ep = meta.get("best_epoch", -1)

    ep = _col(tr, "epoch")
    total = _col(tr, "loss")
    # WEIGHTED contributions: the aux terms are the raw normalized residuals, and lam=0.7 vs
    # w_damage=0.1 means the unweighted curves do not add up to the total that Adam descends.
    c_energy, c_force, c_damage = lam * _col(tr, "energy"), (1 - lam) * _col(tr, "force"), w_dam * _col(tr, "damage")

    vep = _col(va, "epoch")

    apply_charter()
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))

    # ── train loss ──────────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(ep, total, color=INK, lw=1.8, zorder=4, label="total")
    ax.plot(ep, c_energy, color=ORANGE, lw=1.3, zorder=3, label=r"$\lambda\,\varepsilon_W$")
    ax.plot(ep, c_force, color=TEAL, lw=1.3, zorder=3, label=r"$(1-\lambda)\,\varepsilon_F$")
    ax.plot(ep, c_damage, color=RED, lw=1.3, zorder=3, label=r"$w_D\,\varepsilon_D$")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("training loss (scale-normalized)")
    ax.set_title("Sobolev objective", pad=10, fontsize=13, color=INK)
    ax.legend(loc="upper right", fontsize=9.5, handletextpad=0.5, borderpad=0.5)

    # ── val relative error per head ─────────────────────────────────────────────
    ax = axes[1]
    for key, col, lab in (("energy_rel", ORANGE, r"energy  $W$"),
                          ("force_rel", TEAL, r"force  $\partial W/\partial u$"),
                          ("damage_rel", RED, r"damage  $D$")):
        ax.plot(vep, _col(va, key), color=col, lw=1.5, marker="o", ms=2.6, label=lab)
    if best_ep >= 0:
        ax.axvline(best_ep, color=GREY, lw=1.0, ls="--", alpha=0.85, zorder=1)
        ax.annotate("selected", xy=(best_ep, 0.02), xycoords=("data", "axes fraction"),
                    xytext=(4, 0), textcoords="offset points", fontsize=9, color=GREY, va="bottom")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("held-out relative error")
    ax.set_title("Generalization to unseen geometries", pad=10, fontsize=13, color=INK)
    ax.legend(loc="upper right", fontsize=9.5, handletextpad=0.5, borderpad=0.5)

    # ── val force, per component ────────────────────────────────────────────────
    ax = axes[2]
    for key, col, lab in (("force_rel_a", TEAL, r"$F_a$"),
                          ("force_rel_s", RED, r"$F_s$"),
                          ("force_rel_m", GREY, r"$M_\theta$")):
        ax.plot(vep, _col(va, key), color=col, lw=1.5, marker="o", ms=2.6, label=lab)
    if best_ep >= 0:
        ax.axvline(best_ep, color=GREY, lw=1.0, ls="--", alpha=0.85, zorder=1)
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("held-out relative error")
    ax.set_title("Internal force, per component", pad=10, fontsize=13, color=INK)
    ax.legend(loc="upper right", fontsize=9.5, handletextpad=0.5, borderpad=0.5)

    fig.tight_layout(w_pad=2.4)
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"saved -> {args.out}")
    b = side["val_metrics"]
    print(f"  selected epoch {best_ep} of {meta.get('epochs', '?')}  |  val energy_rel "
          f"{b['energy_rel']:.4f}  force_rel {b['force_rel']:.4f}  damage_rel {b['damage_rel']:.4f}")
    t = side.get("test_metrics")
    if t:
        print(f"  held-out TEST  energy_rel {t['energy_rel']:.4f}  force_rel {t['force_rel']:.4f}  "
              f"damage_rel {t['damage_rel']:.4f}")


if __name__ == "__main__":
    main()
