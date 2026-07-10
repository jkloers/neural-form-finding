"""Distributions of the LEARNED closed-design quantities, from a run's best_params.pkl.

What the model actually learned, per hinge / per cut, at the trained state:
  * w_lig  — per-hinge ligament width [mm]      (learnable design DOF; init uniform w_lig_mm)
  * alpha  — per-hinge cut angle [deg]          (deterministic function of the void ratios r)
  * r      — per-cut void aspect ratio in (0,1) (the primary design DOF; init uniform r_init)

    JAX_PLATFORMS=cpu conda run -n kgnn_mac python -m nff.scripts.plot_learned_distributions \
        --run data/outputs/runs/run_<ts>_<cfg>
"""
import os
import argparse
import pickle

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt

from nff.config.experiment import load_and_parse_config
from nff.scripts.closed_setup import build_closed_initial_state, init_closed_les_params, build_surrogate_energy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run dir containing config.yaml + best_params.pkl")
    ap.add_argument("--out", default=None, help="output png (default <run>/learned_distributions.png)")
    args = ap.parse_args()

    cfg = load_and_parse_config(os.path.join(args.run, "config.yaml"))
    with open(os.path.join(args.run, "best_params.pkl"), "rb") as f:
        bp = {k: jnp.asarray(v) for k, v in pickle.load(f).items()}

    state, _ = build_closed_initial_state(cfg)
    params, sf = init_closed_les_params(cfg)
    _, _, geometry_fn, _, _ = build_surrogate_energy(cfg, sf, state, params)

    geo = geometry_fn(bp)                                  # trained HingeGeometry (per-hinge, bond order)
    w_lig = np.asarray(geo.w_lig)
    alpha = np.degrees(np.asarray(geo.alpha))
    r = np.asarray(jax.nn.sigmoid(bp["z"])).ravel()       # per-cut void ratio
    r_init = float(cfg.topology.get("r_init", 0.45))
    w_init = float(cfg.hinge_model.w_lig_mm)

    def _summary(name, x, unit):
        print(f"{name:8s} [{unit}]  n={len(x):3d}  min={x.min():.3f}  p10={np.percentile(x,10):.3f}  "
              f"median={np.median(x):.3f}  mean={x.mean():.3f}  p90={np.percentile(x,90):.3f}  max={x.max():.3f}")

    print(f"\nLearned distributions — {args.run}")
    _summary("w_lig", w_lig, "mm"); _summary("alpha", alpha, "deg"); _summary("r", r, "-")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), facecolor="white")
    for ax, (x, name, unit, init, col) in zip(axes, [
            (w_lig, "ligament width  w_lig", "mm", w_init, "#F58025"),
            (alpha, "cut angle  alpha", "deg", None, "#1565C0"),
            (r,     "void ratio  r = sigmoid(z)", "-", r_init, "#2A9D8F")]):
        ax.hist(x, bins=24, color=col, edgecolor="black", linewidth=0.4, alpha=0.85)
        ax.axvline(x.mean(), color="black", lw=1.5, ls="-", label=f"mean {x.mean():.2f}")
        if init is not None:
            ax.axvline(init, color="#D62828", lw=1.5, ls="--", label=f"init {init:.2f}")
        ax.set_xlabel(f"{name}  [{unit}]"); ax.set_ylabel("count"); ax.legend(fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Learned per-hinge / per-cut distributions (trained closed design)", fontsize=14, weight="bold")
    fig.tight_layout()
    out = args.out or os.path.join(args.run, "learned_distributions.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
