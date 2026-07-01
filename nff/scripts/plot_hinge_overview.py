"""Hinge overview figure (two panels):

  left  — the whole closed tessellation with every meshable hinge strip (green) on
          the cut pattern (boundary cuts included; each hinge keeps its ligament),
  right — the dimensioned engineering drawing of a single hinge.

Saves to ``data/outputs/hinge_overview.png``.

Run:
    JAX_PLATFORMS=cpu conda run -n kgnn_mac python nff/scripts/plot_hinge_overview.py
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nff.topology.closed_builder_jax import boundary_points_flat, solve_cut_vertices_jax
from nff.topology.hinge_descriptor import build_hinge_descriptor_structure
from nff.utils.visualization import plot_hinge_strips, plot_hinge_dimensions


def main(M: int = 6, N: int = 6, r: float = 0.4, sheet_m: float = 1.0,
         out: str = "data/outputs/hinge_overview.png") -> None:
    mm_per_unit = sheet_m * 1000.0 / N
    w_lig, kerf = 10.0 / mm_per_unit, 3.0 / mm_per_unit

    from nff.topology.hinge_descriptor import compute_hinge_descriptors, ManufacturingParams

    hstruct = build_hinge_descriptor_structure(M, N, ref_r=r)
    boundary_flat = jnp.asarray(boundary_points_flat(hstruct["deploy_struct"], 1.0))
    coords = np.asarray(solve_cut_vertices_jax(
        hstruct["deploy_struct"], boundary_flat, jnp.full((N + 1, N + 1), r)))

    # pick a central interior hinge to detail on the right, and use ITS angle
    desc = compute_hinge_descriptors(hstruct, coords, ManufacturingParams(ligament_width=w_lig))
    piv = coords[np.asarray(hstruct["pivot_pid"])]
    interior = np.where(np.asarray(desc["is_interior"]))[0]
    hid = int(interior[np.argmin(np.linalg.norm(piv[interior] - piv.mean(0), axis=1))])
    alpha_deg = float(np.degrees(np.asarray(desc["alpha"])[hid]))
    Pdet = piv[hid]

    fig = plt.figure(figsize=(19, 9.2), facecolor="white")
    axL = fig.add_subplot(1, 2, 1)
    axR = fig.add_subplot(1, 2, 2)

    plot_hinge_strips(coords, hstruct, w_lig, kerf, ax=axL, mm_per_unit=mm_per_unit,
                      scale_mm=200.0, title="meshable hinge strips on the closed tessellation")
    # mark the exact hinge detailed on the right
    axL.add_patch(plt.Circle(Pdet, 3.0*w_lig, fill=False, ec="#D62828", lw=2.0, zorder=20))
    axL.annotate("detailed at right", Pdet + np.array([0, 3.6*w_lig]), color="#D62828",
                 ha="center", va="bottom", fontsize=10, zorder=20)
    plot_hinge_dimensions(w_lig=10.0, w_c=3.0, alpha_deg=alpha_deg, fillet=2.0, thickness=1.0,
                          ax=axR, title="one hinge — geometry parameters")

    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


if __name__ == "__main__":
    main()
