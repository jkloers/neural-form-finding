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
from matplotlib.patches import Circle, FancyBboxPatch
from matplotlib.lines import Line2D
import jax.numpy as jnp

from nff.topology.hinge_descriptor import (build_hinge_descriptor_structure,
                                           hinge_descriptors_from_design)
from nff.topology.closed_builder_jax import boundary_points_flat, solve_cut_vertices_jax
from nff.utils.visualization import plot_hinge_strips, plot_hinge_dimensions
from nff.utils.figstyle import (FS_ANNOT, GREY, INK, TEXT_W_BP, apply_charter, panel_letter,
                                paper_rc, save_paper_figure)

# ── The canvas is authored at its PRINTED size, so an authored fontsize IS its printed point
# size. TEXT_W_BP is \textwidth in sn-jnl.cls; the paper includes this plate at \linewidth.
# Do NOT enlarge the canvas without rescaling the type ladder, and do NOT add `bbox_inches`
# to savefig -- either one silently breaks that guarantee.
FIG_W_BP  = TEXT_W_BP                # \textwidth, shared with every other plate
FIG_H_BP  = FIG_W_BP / 1.97          # preserves the original 15:7.6 content aspect
FS_LABEL  = 7.0                      # dimension labels on the detail
_PLATE_RC = paper_rc()               # the shared ladder, family and Type-42 embedding

_INK = INK
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
    ap.add_argument("--out", default="data/outputs/hinge_overview.pdf")
    ap.add_argument("--no-boxes", action="store_true",
                    help="omit the grouping rule drawn round the plate. A slide has a speaker to group the panels, and a framed plate on a slide reads as a page")
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
    apply_charter(**_PLATE_RC)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(FIG_W_BP / 72.0, FIG_H_BP / 72.0),
                                   facecolor="white", gridspec_kw={"wspace": 0.02})
    for _ax in (axL, axR):
        _ax.set_facecolor("none"); _ax.set_zorder(2)   # above the shared ground below

    # LEFT = principle drawing: exaggerated kerf + tip fillets so the cut geometry reads clearly
    plot_hinge_strips(coords, hs, w_lig, _KERF_MACRO * w_lig, r_win_factor=_RWIN_FACTOR, ax=axL,
                      fillet=_FILLET_MACRO * w_lig, fs=FS_ANNOT)

    P, Rc = pivots[hid], 3.2 * w_lig                    # circle the detailed hinge — bigger, DOTTED
    _TIE = dict(edgecolor=_INK, facecolor="none", lw=0.9, linestyle=(0, (1.2, 1.6)))
    axL.add_patch(Circle(P, Rc, zorder=30, **_TIE))


    # RIGHT = the SAME hinge's alpha, real pipeline proportions + the real gmsh mesh.
    # w_lig = 5 units => 5 mm (1 unit = 1 mm; the panel's "10 mm" bar) matching the left's 5 mm.
    wl = 5.0
    alpha_draw = (180.0 - alpha_sel) if args.supplement else alpha_sel
    plot_hinge_dimensions(w_lig=wl, w_c=_KERF_FRAC * wl, alpha_deg=alpha_draw,
                          fillet=_FILLET_FRAC * wl, ax=axR, fs=FS_LABEL,
                          scale_bar=False)

    # ── one plate, not two drawings. A single faint ground behind both panels groups them;
    # the letters sit in figure space above each panel so nothing collides with the drawings.
    # A border round (b) was tried and rejected: it clipped L_sec and fouled the letter.
    if not args.no_boxes:
        fig.patches.append(FancyBboxPatch((0.008, 0.02), 0.984, 0.96,
                                          boxstyle="round,pad=0,rounding_size=0.012",
                                          transform=fig.transFigure, facecolor="#F7F7F5",
                                          edgecolor=GREY, lw=0.4, zorder=-10))
    fig.add_artist(Line2D([0.505, 0.505], [0.09, 0.91], color=GREY, lw=0.4,
                          transform=fig.transFigure, zorder=-5))
    for _x, _lab in ((0.030, "a"), (0.530, "b")):
        panel_letter(fig, _lab, x=_x, y=0.945, va="top", zorder=40)

    fig.subplots_adjust(left=0.015, right=0.985, bottom=0.03, top=0.90)   # bigger drawings
    pdf, lo, hi = save_paper_figure(fig, args.out, FIG_W_BP, FIG_H_BP, dpi=600)
    plt.close(fig)
    print(f"  {pdf} ({FIG_W_BP:.1f} x {FIG_H_BP:.1f} bp, type {lo:.1f}-{hi:.1f} pt)")
    print(f"saved {args.out}  (hinge {hid}, alpha {alpha_sel:.1f}°)")


if __name__ == "__main__":
    main()
