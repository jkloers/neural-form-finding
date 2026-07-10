"""Two-panel hinge overview figure (regenerates data/outputs/hinge_overview.png).

Left:  the meshable hinge strips on the closed tessellation, with one hinge circled.
Right: a dimensioned single-hinge schematic (w_lig, w_c, alpha, rho, cut lengths).

The right panel is a parametric SCHEMATIC (not a specific hinge), drawn at a non-90 alpha so the
angle between the main and secondary cuts reads clearly.

    JAX_PLATFORMS=cpu conda run -n kgnn_mac python -m nff.scripts.figures.plot_hinge_overview
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import jax.numpy as jnp

from nff.topology.hinge_descriptor import (build_hinge_descriptor_structure,
                                           hinge_descriptors_from_design)
from nff.topology.closed_builder_jax import boundary_points_flat, solve_cut_vertices_jax
from nff.utils.visualization import plot_hinge_strips, plot_hinge_dimensions

_INK = "#1A1A1A"
# design ratios matching the real pipeline (so the DETAIL panel gives physical intuition):
_W_LIG_FRAC = 0.10          # ligament width = 1/10 tile
_KERF_FRAC = 0.06           # kerf / w_lig on the DETAIL (a thin slit — accurate)
_FILLET_FRAC = 0.16         # fillet rho / w_lig (the swept stress-relief DOF, mid value)
_RWIN_FACTOR = 2.4          # Saint-Venant / mesh window = 2.4 * w_lig
# the LEFT panel is a PRINCIPLE drawing (not to scale): the kerf + tip fillet are exaggerated so the
# cut geometry reads on the macro sheet (real kerf ~0.2mm would be invisible).
_KERF_MACRO = 0.30          # macro kerf / w_lig
_FILLET_MACRO = 0.30        # macro tip-fillet radius / w_lig
_MM_PER_TILE = 50.0         # physical tile pitch [mm] (matches the real beam) -> w_lig = 5 mm


def _pick_hinge(pivots, alphas, target=60.0, lo=48.0, hi=78.0):
    """A central hinge whose alpha is a clear non-90 value near `target` (for a legible detail)."""
    d = np.linalg.norm(pivots - pivots.mean(0), axis=1)
    central = d < np.percentile(d, 45)
    ok = central & (alphas > lo) & (alphas < hi)
    pool = np.where(ok)[0] if ok.any() else np.where(central)[0]
    return int(pool[np.argmin(np.abs(alphas[pool] - target))])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--M", type=int, default=8, help="tiles across (more = bigger sheet, smaller hinges)")
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--seed", type=int, default=3, help="random-tessellation seed")
    ap.add_argument("--supplement", action="store_true",
                    help="draw the detail at 180-alpha (the raw cut-opening) to compare the pi-alpha")
    ap.add_argument("--out", default="data/outputs/hinge_overview.png")
    args = ap.parse_args()

    # RANDOM tessellation: per-cell aspect ratios r in (0,1) -> varied, non-90 hinge angles
    hs = build_hinge_descriptor_structure(args.M, args.N, ref_r=0.45)
    ds = hs["deploy_struct"]
    bf = jnp.asarray(boundary_points_flat(ds, 1.0))
    rng = np.random.default_rng(args.seed)
    r_arr = jnp.asarray(rng.uniform(0.28, 0.64, size=(ds["rows"], ds["cols"])))
    coords = np.asarray(solve_cut_vertices_jax(ds, bf, r_arr))
    # alpha = the DESCRIPTOR tile-wedge angle the surrogate/RVE use (NOT the raw cut-direction angle,
    # which is its supplement 180-alpha) -- so the detail's RVE + mesh are built at the pipeline value.
    alphas = np.degrees(np.asarray(hinge_descriptors_from_design(hs, bf, r_arr)["alpha"]))
    pivots = coords[np.asarray(hs["pivot_pid"])]
    hid = _pick_hinge(pivots, alphas)
    alpha_sel = float(alphas[hid])

    w_lig = _W_LIG_FRAC                                  # coords units (tile pitch = 1)
    # Article layout: no panel titles, tight inter-panel gap, both panels centred in their half.
    # The figure is sized close to 2x the (~square) panel content so there is little padding to gap.
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 7.6), facecolor="white",
                                   gridspec_kw={"wspace": 0.0})

    # LEFT = principle drawing: exaggerated kerf + tip fillets so the cut geometry reads clearly
    plot_hinge_strips(coords, hs, w_lig, _KERF_MACRO * w_lig, r_win_factor=_RWIN_FACTOR, ax=axL,
                      mm_per_unit=_MM_PER_TILE, scale_mm=100.0, fillet=_FILLET_MACRO * w_lig)

    P, Rc = pivots[hid], 3.2 * w_lig                    # circle the detailed hinge — bigger, DOTTED
    axL.add_patch(Circle(P, Rc, edgecolor=_INK, facecolor="none", lw=1.1,
                         linestyle=(0, (1, 1.4)), zorder=30))
    axL.annotate("detailed at right", P + np.array([0.0, 1.35 * Rc]), color=_INK,
                 fontsize=11, ha="center", va="bottom", zorder=30)

    # RIGHT = the SAME hinge's alpha, real pipeline proportions + the real gmsh mesh.
    # w_lig = 5 units => 5 mm (1 unit = 1 mm; the panel's "10 mm" bar) matching the left's 5 mm.
    wl = 5.0
    alpha_draw = (180.0 - alpha_sel) if args.supplement else alpha_sel
    plot_hinge_dimensions(w_lig=wl, w_c=_KERF_FRAC * wl, alpha_deg=alpha_draw,
                          fillet=_FILLET_FRAC * wl, ax=axR)

    axL.set_anchor("E"); axR.set_anchor("W")            # pull both to the inner edge -> close together,
    fig.savefig(args.out, dpi=200, bbox_inches="tight")  # vertically centred
    plt.close(fig)
    print(f"saved {args.out}  (hinge {hid}, alpha {alpha_sel:.1f}°)")


if __name__ == "__main__":
    main()
