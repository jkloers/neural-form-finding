"""What a bent route is, and what bending one costs.

Two panels for Section 5.2. The section makes one claim and gives one reason, and each panel
carries one of them.

(a) WHAT A BEND IS. The worst pair solved: a route recorded from a deployed sheet, and the
    straight ray to the same finishing deformation. The two share their endpoint, so the only
    difference is the way there, and the bend is the widest gap between them measured at equal
    fractions of the journey. Drawn in the translation plane, in ligament widths, because that is
    the plane the metric lives in; the fold is not a coordinate here, it runs along both curves.

(b) WHAT IT COSTS. Energy error against bend for all twenty pairs, on log axes so a power law is
    a straight line, with the fitted exponent. Behind it, on a shared x axis, the distribution of
    bend over every route a deployed sheet produced. That pairing is the whole argument of the
    section: the deployed population sits two decades to the left of the pairs that cost anything.

⚠ Panel (a) draws ONE pair, not a typical one beside a worst one. Drawn to a common scale a median
route is a dot on its ray, which teaches nothing; drawn to its own scale it looks identical to the
worst one, which teaches something false. The population histogram in (b) is where the reader
learns that a typical route bends sixty times less, and it says so with an axis rather than with a
picture that has to be re-scaled to be seen.

    JAX_PLATFORMS=cpu conda run -n kgnn_mac python nff/scripts/figures/plot_path_dependence.py
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from nff.utils.figstyle import (FS_BASE, FS_TICK, GREY, INK, MUTE, TEAL,
                                apply_charter, figsize_in, page_bp, panel_letter,
                                save_paper_figure, use_agg)

# Panels (a) and (b) of the merged discussion plate; plot_taylor.py draws (c). THE THREE AXES
# ARE THE SAME RECTANGLE, so the numbers below are shared with that file and any change here
# must be made there too. Positions are set explicitly rather than by tight_layout: two separate
# PDFs sitting side by side on the page cannot be aligned by a layout engine that sees only one
# of them at a time, and unequal axes across a row of panels is the first thing the eye finds.
AX_W_BP = 66.0          # axes width, identical in all three panels
AX_H_BP = 94.0          # axes height, identical in all three panels
PAD_L_BP = 44.0         # room for the y label and its tick labels
PAD_B_BP = 44.0         # room for a two-line x label and its tick labels
PAD_T_BP = 18.0         # room for the panel letter
PAD_R_BP = 14.0
GAP_BP = 40.0           # between (a) and (b): (b) carries its own y label in here

CANVAS_W_BP = PAD_L_BP + AX_W_BP + GAP_BP + AX_W_BP + PAD_R_BP
CANVAS_H_BP = PAD_T_BP + AX_H_BP + PAD_B_BP


def load_pairs(path):
    """The twenty matched pairs, sorted by bend."""
    pairs = json.load(open(path))["pairs"]
    bend = np.array([p["bend"] for p in pairs], float)
    err = np.array([abs(p["W_ratio"] - 1.0) for p in pairs], float)
    order = np.argsort(bend)
    return [pairs[i] for i in order], bend[order], err[order]


def all_routes(prior_dir, seeds, w_lig, resample, floor):
    """Every RESOLVED recorded route, translation normalised by its own final translation.

    Normalising is what lets several be drawn together. In absolute ligament widths their
    translations span nearly three decades, so all but the largest lie flat against the axis;
    divided by its own endpoint every route runs 0 to 1 and the straight ray becomes one identity
    line for all of them.

    ⚠ THE FLOOR IS NOT COSMETIC. Nine of the twenty pairs end at a total translation near
    0.001 ligament widths, a few hundredths of a millimetre and the solver's own resolution.
    Dividing such a route by its endpoint amplifies solver noise to full scale: those curves are
    not monotone and pass 1.0 half way along. Every route above the floor is monotone. The floor
    is therefore a condition for the normalisation to mean anything, and the caption states it.
    """
    curves = []
    for seed in seeds:
        route, _, _, _ = route_and_ray(prior_dir, seed, w_lig, resample)
        t = np.linalg.norm(route, axis=1)
        if t[-1] > floor:
            curves.append(t / t[-1])
    return np.asarray(curves)


def route_and_ray(prior_dir, seed, w_lig, resample):
    """The recorded route and its straight ray, both in ligament widths, origin dropped.

    Rebuilt exactly as ``merge_path_dependence.enrich_with_translation`` rebuilds it: through
    ``pick_path`` on the pair's own seed and through the same arc-length resampling, so the curve
    the figure draws is the curve the recorded ``bend`` was measured on. Indexing ``eta`` directly
    gives a different, unresampled polyline and a bend twenty times smaller.
    """
    from nff.scripts.calibration.replay_hinge_path import pick_path, resample_polyline
    poly, _, _ = pick_path(prior_dir, int(seed))
    if resample:
        poly = resample_polyline(poly, int(resample), w_lig)
    st = poly[1:]
    n = len(st)
    ray_mm = np.outer(np.arange(1, n + 1) / n, st[-1])
    route = st[:, :2] / w_lig
    ray = ray_mm[:, :2] / w_lig
    gap = np.linalg.norm(route - ray, axis=1)
    fold_deg = np.degrees(st[:, 2])
    return route, ray, gap, fold_deg


def panel_bend(ax, curves):
    """Every recorded route against the straight ray, both normalised by their own endpoint.

    ⚠ NOT drawn in the translation plane. The pairs that cost any energy run essentially along the
    same line there: the off-line angle of every pair above a gap of 0.037 is under 2 degrees, so
    a plane plot puts the two legs on top of each other and the difference becomes invisible. The
    difference is a SCHEDULE, and against the fraction of the journey it is a vertical distance.
    """
    frac = np.arange(1, curves.shape[1] + 1) / curves.shape[1]
    mean, sd = curves.mean(axis=0), curves.std(axis=0)
    ax.plot([0, 1], [0, 1], "--", color=GREY, lw=1.1, zorder=4)
    for c in curves:
        ax.plot(frac, c, "-", color=TEAL, lw=0.6, alpha=0.28, zorder=2)
    ax.fill_between(frac, np.clip(mean - sd, 0, None), mean + sd, color=TEAL, alpha=0.20, lw=0,
                    zorder=3)
    ax.plot(frac, mean, "-", color=TEAL, lw=1.8, zorder=5)
    ax.set_xlabel("normalised arc length", fontsize=FS_BASE)
    ax.set_ylabel("normalised translation", fontsize=FS_BASE)
    ax.tick_params(labelsize=FS_TICK)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.05)


def panel_cost(ax, bend, err, sep_pop, floor):
    """Error against the widest gap for the pairs, over the deployed population's distribution.

    The trend line is the least-squares fit over the pairs ABOVE ``floor``, both slope and
    intercept. It is not a slope drawn through the worst pair: anchoring it there put the line a
    factor of two high, so every other pair sat under a curve advertised as their fit.

    Below ``floor`` the two legs of a pair are the same route to within the solver's resolution and
    their ratio carries no information about the scaling. Those pairs are drawn open and excluded
    from the fit, which is what makes the flattening on the left legible instead of alarming.
    """
    lo, hi = 2e-4, 6e-1
    hax = ax.twinx()
    hax.hist(sep_pop, bins=np.logspace(np.log10(lo), np.log10(hi), 46), color=MUTE, alpha=0.65,
             lw=0, zorder=1)
    hax.set_yticks([])
    hax.set_ylim(0, hax.get_ylim()[1] * 3.2)
    hax.grid(False)
    for side in ("top", "right", "left"):
        hax.spines[side].set_visible(False)

    keep = bend > floor
    slope, intercept = np.polyfit(np.log(bend[keep]), np.log(err[keep]), 1)
    ax.axvspan(lo, floor, color=MUTE, alpha=0.22, lw=0, zorder=0)
    xs = np.logspace(np.log10(floor), np.log10(bend[-1] * 1.3), 60)
    ax.plot(xs, np.exp(intercept) * xs ** slope, "--", color=INK, lw=0.9, zorder=3)
    ax.plot(bend[~keep], err[~keep], "o", ms=3.8, mfc="white", mec=INK, mew=0.5, zorder=4)
    ax.plot(bend[keep], err[keep], "o", ms=3.8, mfc=TEAL, mec=INK, mew=0.5, zorder=5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(3e-8, 4e-1)
    ax.set_zorder(hax.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.set_xlabel("path separation\n(ligament widths)", fontsize=FS_BASE)
    # The caption says "relative error in stored energy"; the axis need only carry the symbol.
    ax.set_ylabel(r"$|\Delta W|/W$", fontsize=FS_BASE)
    ax.tick_params(labelsize=FS_TICK)
    ax.set_yticks([1e-7, 1e-5, 1e-3, 1e-1])
    ax.set_xticks([1e-3, 1e-2, 1e-1])
    ax.set_xticks([], minor=True)
    return slope, int(keep.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/surrogates/path_dependence_pet_v3.json")
    ap.add_argument("--prior", default="data/fea/path_priors/envelope_v3_w18")
    ap.add_argument("--separation-floor", type=float, default=2e-3,
                    help="ligament widths; below this the two legs are one route to solver "
                         "resolution, so the pair is drawn open and left out of the fit")
    ap.add_argument("--translation-floor", type=float, default=0.01,
                    help="ligament widths; routes ending below this are at the solver floor and "
                         "cannot be normalised meaningfully")
    ap.add_argument("--out", default="paper/Images/path_dependence.png")
    ap.add_argument("--no-boxes", action="store_true",
                    help="omit the grouping rule drawn round the plate. A slide has a speaker to group the panels, and a framed plate on a slide reads as a page")
    ap.add_argument("--dpi", type=int, default=600)
    args = ap.parse_args()

    use_agg()
    import matplotlib.pyplot as plt
    apply_charter()

    pairs, bend, err = load_pairs(args.pairs)
    geom = json.load(open(args.pairs)).get("geometry", {})
    w_lig = float(geom.get("w_lig", 18.0))
    route, ray, gap, _ = route_and_ray(args.prior, pairs[-1]["seed"], w_lig, geom.get("resample"))
    check = gap.max()
    if not np.isclose(check, bend[-1], rtol=2e-3):
        raise ValueError(f"rebuilt bend {check:.5f} does not match the recorded {bend[-1]:.5f}")

    from nff.scripts.calibration.path_dependence_statistics import separation_of_prior
    out = separation_of_prior(args.prior)
    sep_pop = np.asarray(out[0] if isinstance(out, tuple) else out, float)

    w_bp, h_bp = CANVAS_W_BP, CANVAS_H_BP
    fig, axes = plt.subplots(1, 2, figsize=figsize_in(w_bp, h_bp))
    curves = all_routes(args.prior, [p["seed"] for p in pairs], w_lig, geom.get("resample"),
                        args.translation_floor)
    panel_bend(axes[0], curves)
    slope, n_fit = panel_cost(axes[1], bend, err, sep_pop, args.separation_floor)
    print(f"panel b: {len(bend)} pairs, {n_fit} above the {args.separation_floor:g} floor "
          f"inform the fit; exponent {slope:.2f}")
    for ax, letter in zip(axes, "ab"):
        panel_letter(ax, letter, x=-0.13, y=1.04, va="bottom")
    # Explicit positions, no tight_layout: see the note on the layout constants above.
    for ax, x0 in zip(axes, (PAD_L_BP, PAD_L_BP + AX_W_BP + GAP_BP)):
        ax.set_position([x0 / w_bp, PAD_B_BP / h_bp, AX_W_BP / w_bp, AX_H_BP / h_bp])
    # The grouping rule, drawn inside the graphic as Fig. 10 draws its two: this paper does not
    # frame floats in LaTeX. Same patch, weight and colour as plot_physical_validation.py.
    if not args.no_boxes:
        from matplotlib.patches import FancyBboxPatch
        fig.add_artist(FancyBboxPatch(
            (3.0 / w_bp, 3.0 / h_bp), 1.0 - 6.0 / w_bp, 1.0 - 6.0 / h_bp,
            boxstyle="round,pad=0,rounding_size=0.008", transform=fig.transFigure, zorder=0,
            facecolor="none", edgecolor=GREY, lw=0.5, alpha=0.55))
    pdf, lo_pt, hi_pt = save_paper_figure(fig, args.out, w_bp, h_bp, dpi=args.dpi)
    print(f"wrote {pdf}   type {lo_pt:.1f}-{hi_pt:.1f} pt")
    print(f"  population n={sep_pop.size}  median bend {np.median(sep_pop):.4f}")
    print(f"  worst pair bend {bend[-1]:.5f}  error {err[-1]*100:.2f}%")
    half = curves[:, curves.shape[1] // 2]
    print(f"  panel (a): {curves.shape[0]} of {len(pairs)} routes above the "
          f"{args.translation_floor} floor; median translation done at half way "
          f"{np.median(half):.3f}")


if __name__ == "__main__":
    main()
