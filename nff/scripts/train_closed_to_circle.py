"""Train RDPQK aspect ratios AND boundary points so the deployed sheet -> circle.

Differentiable inverse design (Dang et al. Sec. V). Two sets of design variables:
  - per-cut aspect ratios ``r`` (latent z, bounded to (0.1, 0.9)),
  - boundary vertices sliding tangentially along the square outline (corners fixed).
The forward map ``(r, boundary) -> LES -> deploy at omega=pi/2`` is JAX-
differentiable, so both are driven by optax Adam.

Loss = relative radial variance of the deployed boundary vertices about their
centroid (scale/position invariant; zero iff the outline is a circle) + small
regularizers (keep r near 0.5, keep sliders inside their edge).

Run:
    JAX_PLATFORMS=cpu conda run -n kgnn_mac \
        python nff/scripts/train_closed_to_circle.py
"""

import os

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle

from nff.topology.closed_builder_jax import (
    build_deploy_structure, build_boundary_sliders, boundary_flat_from_sliders,
    forward_deploy,
)

OUTPUT_DIR = os.path.join("data", "outputs")

M, N = 6, 6
OMEGA = np.pi / 2
NUM_STEPS = 600
LR = 5e-3
R_INIT = 0.45
R_REG = 1e-3          # keep r near 0.5
BND_REG = 1e-2        # keep sliders inside their edge


def make_loss(struct, sliders):
    bnd_mask_j = jnp.array(struct["bnd_mask"])
    lo = jnp.array(sliders["lo"])
    hi = jnp.array(sliders["hi"])

    def r_of(z):
        return 0.5 + 0.4 * jnp.tanh(z)

    def loss_fn(params):
        z, bnd_free = params["z"], params["bnd_free"]
        r = r_of(z)
        boundary_flat = boundary_flat_from_sliders(sliders, bnd_free)
        deployed = forward_deploy(struct, boundary_flat, r, OMEGA)
        cloud = deployed[bnd_mask_j]
        center = cloud.mean(axis=0)
        radii = jnp.linalg.norm(cloud - center, axis=1)
        mean_r = radii.mean()
        circularity = jnp.mean((radii / mean_r - 1.0) ** 2)
        reg_r = R_REG * jnp.mean((r - 0.5) ** 2)
        reg_b = BND_REG * jnp.mean(jax.nn.relu(lo - bnd_free) ** 2
                                   + jax.nn.relu(bnd_free - hi) ** 2)
        return circularity + reg_r + reg_b, (deployed, cloud, center, mean_r, circularity)

    return loss_fn, r_of


def plot_state(ax, deployed, cloud, center, mean_r, title):
    for poly in np.asarray(deployed):
        ax.add_patch(Polygon(poly, closed=True, facecolor="#F58025",
                             edgecolor="black", lw=0.8, alpha=0.95))
    cloud = np.asarray(cloud)
    ax.scatter(cloud[:, 0], cloud[:, 1], s=14, color="#457B9D", zorder=5)
    ax.add_patch(Circle(np.asarray(center), float(mean_r), fill=False,
                        edgecolor="#D62828", lw=2.0, ls="--", zorder=6))
    allp = np.asarray(deployed).reshape(-1, 2)
    pad = 0.6
    ax.set_xlim(allp[:, 0].min() - pad, allp[:, 0].max() + pad)
    ax.set_ylim(allp[:, 1].min() - pad, allp[:, 1].max() + pad)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title)


def main():
    struct = build_deploy_structure(M, N)
    sliders = build_boundary_sliders(struct)
    rows, cols = struct["rows"], struct["cols"]

    loss_fn, r_of = make_loss(struct, sliders)
    grad_fn = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))

    z_init = float(jnp.arctanh((R_INIT - 0.5) / 0.4))
    params = {
        "z": jnp.full((rows, cols), z_init),
        "bnd_free": jnp.array(sliders["init"]),
    }

    opt = optax.adam(LR)
    opt_state = opt.init(params)

    (_, (dep0, cloud0, c0, mr0, circ0)) = loss_fn(params)
    print(f"init circularity={float(circ0):.5f}")

    history = []
    for step in range(NUM_STEPS):
        (loss, aux), grads = grad_fn(params)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        history.append(float(aux[4]))
        if step % 75 == 0 or step == NUM_STEPS - 1:
            print(f"step {step:4d}  loss={float(loss):.5f}  circularity={float(aux[4]):.5f}")

    (_, (depF, cloudF, cF, mrF, circF)) = loss_fn(params)
    print(f"final circularity={float(circF):.5f}  "
          f"(improvement {float(circ0) / max(float(circF), 1e-9):.1f}x)")
    print(f"r range: [{float(jnp.min(r_of(params['z']))):.3f}, "
          f"{float(jnp.max(r_of(params['z']))):.3f}]")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5), facecolor="white")
    plot_state(axes[0], dep0, cloud0, c0, mr0, f"Initial deploy (r={R_INIT})")
    plot_state(axes[1], depF, cloudF, cF, mrF, "Trained (r + boundary) -> circle")
    axes[2].plot(history, color="#F58025", lw=2)
    axes[2].set_xlabel("step")
    axes[2].set_ylabel("circularity loss")
    axes[2].set_title("Optimization")
    axes[2].grid(alpha=0.3)
    out_path = os.path.join(OUTPUT_DIR, "train_closed_to_circle.png")
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
