"""Taylor remainder convergence: is the design gradient a gradient?

The verification standard of PDE-constrained optimisation. The first-order remainder
``R2(h) = |L(t + h d) - L(t) - h <grad L, d>|`` must fall as O(h^2) if the vector returned by
automatic differentiation is the gradient of the objective being optimised. An O(h) and an O(h^2)
reference are drawn, each set just above the curves it describes, so a curve is read against an
adjacent slope: parallel to the quadratic reference is a verified gradient, parallel to the linear
one is not.

The observed order is dimensionless, which is what makes the comparison admissible at all. A
28-parameter chart and a 90-parameter chart are judged on the same scale with no metric or
dimension argument available.

ONE PANEL. Two second panels were tried and both were dropped. The observed order of R2
DIFFERENCES neighbouring remainders, so it amplifies their noise: below the band it divided noise
by noise and scattered through 0.57, 0.17, -0.12, 0.48, and a reader set that against the
penalised arm's clean 1.00 and read the subject as the worse of the two. An adjoint-against-
central-difference scatter replaced it and was in turn dropped as not earning its half of the
plate. Both remain measured; ``gradcheck_stats`` still writes the cross-check into the sidecar as
a measured-not-drawn arm.

    JAX_PLATFORMS=cpu conda run -n kgnn_mac python nff/scripts/figures/plot_taylor.py
"""

import argparse
import json
import os

import numpy as np

from nff.utils.figstyle import (FS_ANNOT, FS_BASE, GREY, INK, ORANGE, TEAL,
                                apply_charter, figsize_in, page_bp, panel_letter, paper_rc,
                                save_paper_figure, use_agg)

use_agg()
import matplotlib.pyplot as plt                                        # noqa: E402

# Authored at the INTERIOR width of its box in the merged plate (\textwidth less
# 2*\fboxsep and 2*\fboxrule), so the graphic prints 1:1 and its type is the size it
# was set at. See feedback_author_figures_at_printed_size.
# Panel (c) of the merged discussion plate; plot_path_dependence.py draws (a) and (b). THE THREE
# AXES ARE THE SAME RECTANGLE, so these numbers are shared with that file and a change here must
# be made there too. Positions are explicit rather than laid out: two separate PDFs sitting side
# by side on the page cannot be aligned by a layout engine that sees only one of them at a time.
AX_W_BP = 66.0          # axes width, identical in all three panels
AX_H_BP = 94.0          # axes height, identical in all three panels
PAD_L_BP = 44.0
PAD_B_BP = 44.0
PAD_T_BP = 18.0
PAD_R_BP = 14.0

FIG_W_BP = PAD_L_BP + AX_W_BP + PAD_R_BP
FIG_H_BP = PAD_T_BP + AX_H_BP + PAD_B_BP

# Series carrying equal visual weight read as unrelated measurements. They are not: one is the
# objective this paper optimises, one is its control under the older hinge law, and one is the
# chart it is being argued against. The weights below encode that reading. The subject is the only
# saturated colour and the only heavy stroke, the control is neutral ink, and the alternative
# chart is muted.
HERO, CONTROL, ALT = ORANGE, INK, TEAL

# (legend label, npz key, colour, marker, linewidth). Ordered as the legend reads.
#
# The projection-only free-vertex arm is measured and kept in the npz, but is NOT drawn. It reaches
# order 2 like the two closed-chart arms, so as a fourth line it added a curve without adding a
# conclusion. Dropped for legibility; its number is reported in the prose instead.
# The labels name the two things that actually vary, chart and hinge law, in that order, and
# nothing else. "law", "chart with", "forward" and "tightened" were carried by every entry alike,
# so they separated no two curves and only widened the key: the longest label ran 42 characters
# and cut through the linear-hinge curve when the key was set inside the axes.
SERIES = [
    ('constructed chart, learned hinge law', 'shipping objective (surrogate)', HERO, '+', 1.5),
    ('constructed chart, linear hinge law', 'closed chart', CONTROL, '+', 1.1),
    ('free-vertex chart, non-overlap penalty',
     'free vertices + projection + overlap penalty', ALT, '+', 1.1),
]
SENSITIVITY = ('learned hinge law, forward solve', HERO, 'x', 1.1)
MS = 3.4                       # cross size; the mark should read as a tick, not as a dot
MEW = 0.9                      # a cross is drawn entirely by its edge

BOX_OUT = 3.0               # canvas edge -> grouping rule, as in Fig. 10
BOX_KW = dict(facecolor="none", edgecolor=GREY, lw=0.5, alpha=0.55)

H_MAX = 6e-2        # the coarsest steps are pre-asymptotic; the free-vertex arms blow up there


def house_frame(fig, w_bp, h_bp):
    """The hairline that groups a panel block, in the form Fig. 10 established.

    Not a LaTeX \\fbox. This paper draws its grouping rules inside the graphic, so the rule is
    part of the figure and scales and prints with it; the same patch, weight and colour are used
    by plot_physical_validation.py.
    """
    from matplotlib.patches import FancyBboxPatch
    fig.add_artist(FancyBboxPatch(
        (BOX_OUT / w_bp, BOX_OUT / h_bp), 1.0 - 2 * BOX_OUT / w_bp, 1.0 - 2 * BOX_OUT / h_bp,
        boxstyle="round,pad=0,rounding_size=0.008", transform=fig.transFigure, zorder=0,
        **BOX_KW))


def series_in(d, with_sensitivity):
    """The (label, key, colour, marker, filled) rows actually present in ``d``.

    The sensitivity rows are the shipping arm re-measured at a tighter forward solve. They are the
    evidence that the floor belongs to the solve and not to the adjoint, so they are drawn dashed
    in the shipping arm's own colour rather than as a separate method.
    """
    rows = [(lab, k, c, m, w, True) for lab, k, c, m, w in SERIES if f"{k}__h" in d]
    if not with_sensitivity:
        return rows
    lab, c, m, w = SENSITIVITY
    extra = sorted(k[:-3] for k in d.files
                   if k.endswith("__h") and k.startswith(SERIES[0][1] + " tol="))
    return rows + [(f"{lab} ${_pow10(k.split('tol=')[-1])}$", k, c, m, w, False) for k in extra]


def _pow10(text):
    """``1e-07`` as the exponent-form LaTeX a caption should carry."""
    return r"10^{%d}" % round(np.log10(float(text)))


def observed_order(h, r2, factor=2.0):
    """Observed order of ``r2``, taken across a ``factor`` change in ``h`` at every sample.

    Derived from the remainders here rather than read from the npz, because the stored ``o2`` is a
    ratio between ADJACENT samples and the sequence is sampled finely. Halving the step in log h
    halves the denominator of the order, so it doubles the noise the same measurement error
    produces: at sqrt(2) spacing the adjacent-pair order scatters between 0.8 and 3.0 on an arm
    that is cleanly quadratic. Spanning a fixed factor of two instead restores the signal-to-noise
    of the classical halving test while keeping one estimate per sample.

    Returns:
        ``(h_at, order)`` with ``h_at`` the finer endpoint of each pair.
    """
    h = np.asarray(h, float)
    if h.size < 3:
        return h[:0], h[:0]
    lag = max(1, int(round(np.log(factor) / np.log(h[0] / h[1]))))
    if lag >= h.size:
        return h[:0], h[:0]
    num = np.log(np.maximum(r2[:-lag], 1e-300) / np.maximum(r2[lag:], 1e-300))
    return h[lag:], num / np.log(h[:-lag] / h[lag:])


def _smooth(x, y, log_y, n=400):
    """A shape-preserving interpolant through the measured points, sampled for drawing.

    PCHIP rather than a natural cubic spline: it is monotone on monotone data and therefore cannot
    invent an overshoot between two samples. On a convergence plot an overshoot would read as a
    change of order that was never measured, which is the one artefact this figure must not carry.
    Fitted in the log coordinates the panel is drawn in, so the curve is smooth as seen.
    """
    from scipy.interpolate import PchipInterpolator
    lx = np.log10(x)
    ly = np.log10(np.maximum(y, 1e-300)) if log_y else y
    order = np.argsort(lx)
    if lx.size < 3:
        return x, y
    f = PchipInterpolator(lx[order], ly[order])
    gx = np.linspace(lx[order][0], lx[order][-1], n)
    gy = f(gx)
    return 10.0 ** gx, (10.0 ** gy if log_y else gy)


def _draw(ax, x, y, colour, marker, lw, solid, log=True, **kw):
    """One series: a smooth stroke through the measured points, with the points marked."""
    ax.set_xscale('log')
    if log:
        ax.set_yscale('log')
    sx, sy = _smooth(x, y, log_y=log)
    ax.plot(sx, sy, ls='-' if solid else (0, (4, 1.8)), color=colour, lw=lw,
            solid_capstyle='round', **kw)
    ax.plot(x, y, ls='none', marker=marker, ms=MS, markeredgecolor=colour,
            markerfacecolor='none', markeredgewidth=MEW,
            zorder=kw.get('zorder', 3) + 1)


def panel_remainder(ax, d, rows):
    """R2 against the step, read against the two reference slopes.

    Returns the guide segments, to be labelled once the layout is final.
    """
    for label, key, colour, marker, lw, filled in rows:
        h, r2 = d[f"{key}__h"], d[f"{key}__R2"]
        m = np.isfinite(r2) & (r2 > 0) & (h < H_MAX)
        _draw(ax, h[m], r2[m], colour, marker, lw, filled, label=label,
              zorder=4 if colour == HERO else 3)

    # Each guide is anchored to the curves it describes, not to the global maximum. Sharing one
    # anchor pushed the quadratic reference far above the quadratic curves, so the eye had to
    # carry it across two decades of empty space to use it. Sitting just above its own group, it
    # is read by comparing adjacent slopes, which is what the panel is asking of the reader.
    h_all = np.concatenate([d[f"{k}__h"] for _, k, *_ in rows])
    hr = np.array([h_all.min(), H_MAX])

    def top_of(keys, pad):
        vals = [d[f"{k}__R2"][np.isfinite(d[f"{k}__R2"]) & (d[f"{k}__h"] < H_MAX)] for k in keys]
        return max(v.max() for v in vals if v.size) * pad

    quad = [k for _, k, *_ in rows if (asymptotic_band(d, k)[0] or 0) >= 1.8]
    lin = [k for _, k, *_ in rows if (asymptotic_band(d, k)[0] or 0) < 1.8]
    guides = []
    for power, style, keys in ((2, (0, (1.1, 1.9)), quad or lin), (1, (0, (4.5, 2.0)), lin or quad)):
        y = top_of(keys, 4.0) * (hr / hr.max()) ** power
        ax.plot(hr, y, ls=style, color=GREY, lw=0.9, zorder=1)
        guides.append((hr, y, power))

    # No annotation on the flattening. It was labelled "forward-solve floor" on the plate, which
    # asserted the cause; the two tolerance curves show the reader that the floor moves with the
    # solve, and the prose states it. The leader also had to reach across the two quadratic
    # curves to find room, which put grey ink through the part of the plate carrying the result.

    ax.set_xlabel(r'perturbation step $h$', fontsize=FS_BASE)
    ax.set_ylabel(r'remainder $R_2$', fontsize=FS_BASE)
    return guides


def label_guides(ax, guides):
    """Set each reference-slope label along its own line, once the axes box is final.

    The visual angle of a line on log axes depends on the data limits and on the box the layout
    settles on, so it cannot be known while the panel is being drawn. Measuring it in display
    coordinates afterwards is what keeps the label parallel to the line it names.
    """
    # The transform is only correct once a renderer has been through the figure: before the first
    # draw it reports the default subplot box, not the one set_position asked for, and the angle
    # comes out steeper than the line by half again.
    ax.figure.canvas.draw()
    # Each guide clears its own group by a fixed factor at the coarse end and converges on the
    # data towards the fine end, so the coarse end is where both labels have air.
    for hr, y, power in guides:
        (x0, y0), (x1, y1) = ax.transData.transform([(hr[0], y[0]), (hr[1], y[1])])
        angle = float(np.degrees(np.arctan2(y1 - y0, x1 - x0)))
        t = 0.86
        xt = float(hr[0] * (hr[1] / hr[0]) ** t)
        yt = float(y[0] * (y[1] / y[0]) ** t)
        ax.text(xt, yt * 1.4, r'$O(h)$' if power == 1 else r'$O(h^{2})$', fontsize=FS_ANNOT,
                color=GREY, ha='center', va='bottom', rotation=angle, rotation_mode='anchor')


def gradcheck_stats(d, key):
    """Relative deviation from the diagonal, per direction. ``None`` if the arm carries no pairs.

    Returns:
        ``(fd, ref, rel)``, the central difference, the adjoint directional derivative and
        ``|fd - ref| / |ref|``, one entry per direction.
    """
    if f"{key}__fd" not in d.files:
        return None
    fd, ref = np.asarray(d[f"{key}__fd"]), np.asarray(d[f"{key}__ref"])
    return fd, ref, np.abs(fd - ref) / np.maximum(np.abs(ref), 1e-300)


def asymptotic_band(d, key, quadratic=1.8):
    """The convergent band, its order, and the step at which the remainder stops falling.

    Two regions bracket the band and neither belongs in a reported order. Above it the steps are
    pre-asymptotic, where the quadratic term does not yet dominate. Below it the remainder is a
    difference of converged objective values and cannot fall past the accuracy of the forward
    solve, so it flattens and the observed order collapses. A median over the whole sequence, or
    over its finer half, mixes all three regions and reports none of them.

    The band is the longest run of consecutive steps whose observed order reaches ``quadratic``,
    which is the region a reader is asked to read against slope 2.

    Returns:
        ``(order, h_hi, h_lo, r2_floor)``. ``order`` is the median over the band, ``h_hi`` and
        ``h_lo`` bound it, and ``r2_floor`` is the smallest remainder reached. All None if no
        step in range converges quadratically.
    """
    h, r2 = d[f"{key}__h"], d[f"{key}__R2"]
    ho, o2 = observed_order(h, r2)
    ok = np.isfinite(o2) & (ho < H_MAX) & (ho * 2.0 < H_MAX) & (o2 >= quadratic)
    best = run = None
    for j, good in enumerate(ok):
        # `run is None`, not `run or j`: a run starting at index 0 is falsy and was being reset
        # on the following step, truncating any band that begins at the first usable sample.
        run = (j if run is None else run) if good else None
        if good and (best is None or j - run > best[1] - best[0]):
            best = (run, j)
    if best is None:
        return None, None, None, float(np.nanmin(r2[h < H_MAX]))
    lo, hi = best
    return (float(np.median(o2[lo:hi + 1])), float(ho[lo] * 2.0), float(ho[hi]),
            float(np.nanmin(r2[h < H_MAX])))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", default="data/outputs/chart_ablation_28deg/taylor.npz")
    ap.add_argument("--out", default="paper/Images/taylor_convergence.png")
    ap.add_argument("--no-boxes", action="store_true",
                    help="omit the grouping rule drawn round the plate. A slide has a speaker to group the panels, and a framed plate on a slide reads as a page")
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--no-sensitivity", action="store_true",
                    help="omit the tightened-forward-solve series.")
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    rows = series_in(d, not args.no_sensitivity)

    # A grid on a figure whose whole content is slope competes with the very lines being read, so
    # it is dropped to a hairline and kept off the log decades of the left panel entirely.
    apply_charter(**paper_rc(**{"axes.grid": False, "axes.edgecolor": "#4A4A4A",
                                "axes.linewidth": 0.7, "xtick.color": "#4A4A4A",
                                "ytick.color": "#4A4A4A"}))
    fig, ax = plt.subplots(1, 1, figsize=figsize_in(FIG_W_BP, FIG_H_BP))
    guides = panel_remainder(ax, d, rows)
    # (c) of the merged plate: (a) and (b) are the path-dependence graphic above it.
    panel_letter(ax, 'c', x=-0.085, y=1.04)
    ax.tick_params(which='both', length=2.4, width=0.6)
    ax.tick_params(which='minor', length=1.4)

    # Explicit position, no tight_layout: see the note on the layout constants above.
    ax.set_position([PAD_L_BP / FIG_W_BP, PAD_B_BP / FIG_H_BP,
                     AX_W_BP / FIG_W_BP, AX_H_BP / FIG_H_BP])
    # After the position is final, when the axes box and therefore the on-page slope are known.
    label_guides(ax, guides)
    if not args.no_boxes:
        house_frame(fig, FIG_W_BP, FIG_H_BP)

    pdf, lo, hi = save_paper_figure(fig, args.out, FIG_W_BP, FIG_H_BP, dpi=args.dpi)
    print(f"  wrote {pdf} ({FIG_W_BP:.1f} x {FIG_H_BP:.1f} bp, type {lo:.1f}-{hi:.1f} pt)")
    plt.close(fig)

    meta = json.loads(str(d['meta'])) if 'meta' in d.files else {}
    stats = {key: asymptotic_band(d, key) for _, key, *_ in rows}
    _caption(os.path.splitext(args.out)[0] + ".txt", rows, stats, meta, args.npz, d)
    print(f"  wrote {os.path.splitext(args.out)[0]}.txt")
    for label, key, *_ in rows:
        n = meta.get(key, {}).get('n_param')
        print(f"    {label:<50} {_summary(stats[key])}"
              + (f"  ({n} params)" if n else ""))


def _plain(label):
    """A legend label as prose. The plate needs mathmode; a plain-text sidecar must not carry it."""
    return (label.replace('$+$', '+').replace('$10^{-7}$', '1e-7')
            .replace('$10^{-9}$', '1e-9').replace('$10^{-5}$', '1e-5'))


def _summary(stat):
    """One line per arm: the order, the band it holds over, and the smallest remainder reached."""
    order, h_hi, h_lo, r2_floor = stat
    if order is None:
        return f"never converges quadratically; smallest R2 {r2_floor:.1e}"
    return (f"order {order:.2f} over h = {h_hi:.2e} to {h_lo:.2e}, "
            f"smallest R2 {r2_floor:.1e}")


def _caption(path, rows, stats, meta, npz, d):
    with open(path, "w") as f:
        f.write(
            "Taylor remainder convergence of the design gradient.\n"
            "Generated by nff/scripts/figures/plot_taylor.py.\n\n"
            "The Taylor remainder against the perturbation step, averaged over eight random unit "
            "directions drawn in each chart's own parameterisation. Marks are the measured steps "
            "and the curve interpolates them. Each grey reference line is set just above the "
            "group it describes and is marked O(h) or O(h^2); a curve parallel to the quadratic "
            "reference carries a verified gradient. The subject curve stops falling where the "
            "remainder reaches the accuracy of the forward solve. Orange is the design objective "
            "the paper optimises, the constructed chart under the learned hinge law, and the "
            "dashed orange curve is that same objective with the forward solve tightened. Black "
            "is the constructed chart under the linear hinge law. Teal is the free-vertex chart "
            "carrying the non-overlap penalty it needs in order to stay embedded.\n\n"
            "Provenance, not part of the caption.\n"
            f"Input: {npz}, written by nff/scripts/diagnostics/taylor_test.py.\n"
            "Configs: closed/sheet_4x8ft_pet_28deg (surrogate arm, 2.50 N/face), "
            "closed/sheet_4x8ft_rom_28deg (the three chart arms, 13.20 N/face). Each is "
            "calibrated to its own hinge law, so neither load transfers to the other. "
            "backward_reg = 0.01 throughout; the adjoint solver changes at zero and the gradient "
            "is then wrong by orders of magnitude, so it is never measured there.\n"
            "Observed order over the band where the remainder converges quadratically, with the "
            "band and the smallest remainder reached. The band excludes the coarse steps, where "
            "the quadratic term does not yet dominate, and the fine steps below the floor:\n")
        for label, key, *_ in rows:
            n = meta.get(key, {}).get('n_param')
            f.write(f"  {_plain(label)}: {_summary(stats[key])}"
                    + (f" ({n} design parameters)" if n else "") + "\n")
        drawn = {k for _, k, *_ in rows}
        for key in (k for k in meta if k not in drawn):
            n = meta.get(key, {}).get('n_param')
            f.write(f"  [measured, not drawn] {key}: {_summary(asymptotic_band(d, key))}"
                    + (f" ({n} design parameters)" if n else "") + "\n")
        gh = next((m.get('gradcheck_h') for m in meta.values() if m.get('gradcheck_h')), None)
        f.write(f"[MEASURED, NOT DRAWN] Adjoint against a central difference: relative deviation "
                f"|fd - adjoint| / |adjoint| per direction, at h = {gh:g}. This was a second "
                f"panel for one revision and was dropped as not earning its half of the plate. "
                f"It is the one instrument here that takes no limit, so it is kept on record:\n"
                if gh else "[MEASURED, NOT DRAWN] Adjoint against a central difference:\n")
        for label, key, *_ in rows:
            st = gradcheck_stats(d, key)
            if st is None:
                continue
            rel = st[2]
            f.write(f"  {_plain(label)}: median {np.median(rel):.2e}, worst {rel.max():.2e}, "
                    f"over {rel.size} directions\n")
        f.write(
            "The step is fixed inside every band listed above, and it has to be. At the band EDGE "
            "the linear contamination below the band already matches the quadratic signal, so a "
            "deviation measured there is a property of the step rather than of the gradient. The "
            "old default h = 1e-3 sat at the surrogate arm's edge.\n"
            "On the penalised arm 9 of the 48 directions disagree in SIGN: the adjoint says the "
            "objective rises where it falls. That is the sharpest statement available of what an "
            "active penalty does to the gradient, and it is why the cross-check is kept here.\n"
            "The free-vertex chart with the validity projection and NO penalty is measured and "
            "kept in the npz, but is not drawn. It reaches order 2 like the two closed-chart "
            "arms, so as a fourth curve it added a line without adding a conclusion. Its order is "
            "listed above and reported in the prose.\n"
            "The remainder is averaged over eight directions at each of twenty-one steps in a "
            f"sqrt(2) sequence from h = 1e-1, and the coarsest steps above h = {H_MAX:g} are "
            "dropped: they are pre-asymptotic, and the free-vertex arms overflow there.\n"
            "The floor is the accuracy of the forward solve, not of the adjoint. It moves when "
            "solver_tol moves, from 1e-5 to 1e-7, and saturates there. It does not move at all "
            "when solver_maxiter is raised from 1500 to 6000, which returns bit-identical "
            "remainders, so the iteration cap is not what binds. Zeroing the two path-extremum "
            "weights, compression and ood, also leaves the remainders unchanged, so the "
            "non-smoothness of a maximum over load steps is not what sets it either.\n"
            "⚠ The subject arm leaves the quadratic law in TWO stages, and only the second is the "
            "tolerance. From h = 1.5e-3 to 4e-4 both tolerance arms lie on one another and fall "
            "as O(h), the tightened one continuing on that same line at order 0.95 to 0.99 for "
            "four further steps; below 4e-4 the loose arm alone flattens and scatters. So "
            "tightening the solve removes the flat stage and leaves the linear one untouched. "
            "That linear term is a residual inconsistency between the objective evaluated and the "
            "gradient returned, about 1.4 per cent of the directional derivative on the surrogate "
            "arm and 0.12 per cent on the ROM chart -- independently corroborated by panel (b), "
            "which reads 1.2e-2 and 6.2e-4 median by a completely different route. Its cause is "
            "NOT established; backward_reg = 0.01 is the obvious suspect and has not been tested.\n"
            "A dot-product test against a forward-mode JVP is impossible here. Stage 2 "
            "differentiates implicitly through a jaxopt custom_vjp, which JAX supports in reverse "
            "mode only. Every arm shares that one adjoint implementation, so an arm reaching order "
            "2 verifies it, and another arm's failure cannot be attributed to it.\n"
            "Regenerate: JAX_PLATFORMS=cpu conda run -n kgnn_mac --no-capture-output "
            "python nff/scripts/figures/plot_taylor.py\n")


if __name__ == "__main__":
    main()
