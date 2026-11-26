"""Discussion plate: two claims, one per COLUMN, and one angle throughout.

Left column -- **does the sheet follow the ideal deployment?**  x axis: relative hinge rotation.
    (a) two independent errors from the ideal deployment, with one unit cell of the sheet drawn at
        four openings along the curve. They agree at small opening and then part company: near full
        opening the sheet leaves the ideal deployment ENERGETICALLY while still LOOKING like it.
    (c) the radius of the disturbed zone around the grip and each support.

Right column -- **which state does it take?**  x axis: competing-load ratio.
    (b) hinge rotation against a downward load competing with a constant opening pull: the sheet
        holds a state and then jumps.
    (d) the smallest eigenvalue of the tangent stiffness over the same range; it never reaches zero,
        so every state is a genuine equilibrium and the jumps are changes of state, not a collapse.

⚠⚠ ONE ANGLE. ``theta`` is the RELATIVE HINGE rotation between adjacent panels, the symbol the
manuscript uses throughout. ``psi`` -- the per-panel rotation, half of it -- appears nowhere in this
figure: mixing the two is a bug the manuscript already shipped once and fixed by standardising on
``theta``.

Other house rules this plate is built to: nothing is written inside an axis; no blue, no red, no
green; the boundary conditions take their documented colours (load INK, clamp GREY, slider PURPLE);
``page_bp``'s ``aspect`` is WIDTH / HEIGHT so landscape needs aspect > 1.
"""

import argparse
import json
import os

import numpy as np

from nff_paper.closed_support.perturbation import perturbation_radius
from nff.utils import figstyle as FST

ROOT = "data/outputs/localisation"

ASPECT = 1.23
STATE_FILL = "#23282C"
WINDOW_FILL = "#FDE4CC"        # light orange: a pale tint of the energy curve's own hue
ONE_PERCENT = 1.0
N_STATES = 4
PATCH_COLS = (7, 8)            # an interior 2x2 patch = FOUR tiles, which is the rotating-squares
PATCH_ROWS = (4, 5)            # unit cell: four panels around ONE void. Boundary tiles are irregular.
JUMP_DEG = 4.0                 # a step in theta bigger than this starts a new branch (theta = 2 psi)
# The three concentrated boundary conditions, with their documented house colours.
BCS = ((145, "grip", FST.INK), (9, "pin", FST.GREY), (0, "roller", FST.PURPLE))


def _read(run, fname="summary.json"):
    with open(os.path.join(ROOT, run, fname)) as fh:
        return json.load(fh)


def _branch_segments(y, jump=JUMP_DEG):
    """Index runs that stayed on one branch, so a jump is a gap and never a stroke."""
    breaks = np.nonzero(np.abs(np.diff(y)) > jump)[0] + 1
    return np.split(np.arange(y.size), breaks)


def one_percent_window(theta, energy_pct, level=ONE_PERCENT):
    """The opening range over which ``energy_pct`` stays below ``level``, by interpolation.

    Derived rather than declared: the shaded band means exactly "the error is under one per cent
    here", and hardcoded edges drift away from the crossing they claim to mark as soon as the data
    changes.
    """
    i = int(np.argmin(energy_pct))
    lo_side, hi_side = slice(i, None, -1), slice(i, None)
    return (float(np.interp(level, energy_pct[lo_side], theta[lo_side])),
            float(np.interp(level, energy_pct[hi_side], theta[hi_side])))


def panel_departures(ax, theta, energy_pct, shape_pct, lo_edge=None, hi_edge=None,
                     headroom=40.0):
    """(a) Energy and shape errors from the ideal deployment, against hinge rotation."""
    if lo_edge is None or hi_edge is None:
        lo_edge, hi_edge = one_percent_window(theta, energy_pct)
    ax.axvspan(lo_edge, hi_edge, color=WINDOW_FILL, lw=0, zorder=0)
    ax.axhline(ONE_PERCENT, color=FST.GREY, lw=0.7, ls=(0, (1.4, 1.8)), zorder=1)
    # Each curve carries its definition rather than a phrase. ``W`` is the manuscript's hinge energy
    # and ``W_rot`` the part of it carried by relative rotation; the subscript is spelled out because
    # ``W_vartheta`` would read as the manuscript's ``W(vartheta)``. The axis supplies the per cent.
    ax.plot(theta, energy_pct, "-", color=FST.ORANGE, lw=1.5, solid_capstyle="round", zorder=4,
            label=r"$1-\mathcal{W}_{\mathrm{rot}}/\mathcal{W}$")
    ax.plot(theta, shape_pct, "-", color=FST.TEAL, lw=1.2, solid_capstyle="round", zorder=3,
            label=r"$1-R^2$")
    ax.set_yscale("log")
    ax.set_xlim(0, 90)
    # A clear decade below the teal plateau, so the legend has room of its own. At 0.5 the legend sat
    # on the curve it was labelling.
    ax.set_ylim(max(min(shape_pct.min(), energy_pct.min()) * 0.055, 1e-5), energy_pct.max() * headroom)
    ax.set_ylabel("relative error  (%)")
    ax.legend(frameon=False, fontsize=FST.FS_ANNOT, loc="lower left", ncol=1, handlelength=1.2,
              columnspacing=0.9, handletextpad=0.4, borderpad=0.15, labelspacing=0.25)


def panel_states(fig, ax, nodes_list, theta_list, theta, curve, width_frac=0.22, gap=0.002,
                 n_rows=10):
    """One unit cell of the sheet, bare and dark, at its own point on the curve in panel (a).

    Four panels around a single void rather than the whole sheet: at this size 150 panels are a smudge,
    and the unit cell is what carries the void geometry distinguishing one opening from another. No
    frame, no ground and **no panel edges**: an edge colour separates panels at the corners where they
    physically touch, which makes an opened sheet read as loose diamonds instead of a connected lace.
    """
    from matplotlib.collections import PolyCollection

    patch = [c * n_rows + r for c in PATCH_COLS for r in PATCH_ROWS]
    nodes_list = [n[patch] for n in nodes_list]
    allpts = np.concatenate([n.reshape(-1, 2) for n in nodes_list])
    cx = 0.5 * (allpts[:, 0].max() + allpts[:, 0].min())
    cy = 0.5 * (allpts[:, 1].max() + allpts[:, 1].min())
    half_w = 0.52 * (allpts[:, 0].max() - allpts[:, 0].min())
    half_h = 0.52 * (allpts[:, 1].max() - allpts[:, 1].min())

    pos = ax.get_position()
    w = pos.width * width_frac
    h = w * (half_h / half_w)
    inv = fig.transFigure.inverted()
    out = []

    for th, nodes in zip(theta_list, nodes_list):
        y_curve = float(np.interp(th, theta, curve))
        px, _ = inv.transform(ax.transData.transform((th, y_curve)))
        x0 = float(np.clip(px - w / 2, pos.x0 + 0.002, pos.x1 - w - 0.002))
        span = [ax.transData.inverted().transform(fig.transFigure.transform((xx, pos.y0)))[0]
                for xx in (x0, x0 + w)]
        inside = (theta >= min(span)) & (theta <= max(span))
        top_v = float(np.max(curve[inside])) if inside.any() else y_curve
        y0 = float(np.clip(inv.transform(ax.transData.transform((th, top_v)))[1] + gap,
                           pos.y0 + 0.002, pos.y0 + pos.height - h - 0.002))

        iax = fig.add_axes([x0, y0, w, h], zorder=6)
        iax.add_collection(PolyCollection([nd for nd in nodes], facecolors=STATE_FILL,
                                          edgecolors="none", zorder=2))
        iax.set_xlim(cx - half_w, cx + half_w)
        iax.set_ylim(cy - half_h, cy + half_h)
        iax.set_aspect("equal")
        iax.axis("off")
        iax.patch.set_visible(False)
        ax.plot([th], [y_curve], "o", ms=2.8, color=STATE_FILL, zorder=7)
        out.append(iax)
    return out


def panel_radius(ax, theta, radii):
    """(c) Radius of the disturbed zone around each boundary condition, in tiles.

    A gap in a curve is not missing data: it means that boundary condition has NO zone above the
    threshold at that opening, which for the roller is true across most of the window.
    """
    for src, lab, col in BCS:
        r = np.asarray(radii[src], dtype=float)
        ax.plot(theta, r, "-", color=col, lw=1.3, solid_capstyle="round", zorder=3, label=lab)
        ax.plot(theta, r, "o", color=col, ms=2.4, zorder=4,
                markeredgecolor="white", markeredgewidth=0.25)
    ax.set_xlim(0, 90)
    ax.set_xticks([0, 30, 60, 90])
    ax.set_ylim(0, None)
    ax.set_xlabel(r"$\vartheta$  (deg)")
    ax.set_ylabel("disturbed radius  (panels)")
    ax.legend(frameon=False, fontsize=FST.FS_ANNOT, loc="upper center", ncol=3, handlelength=1.2,
              columnspacing=0.9, handletextpad=0.4, borderpad=0.15)


def panel_states_vs_load(ax, ratio, theta_deg):
    """(b) Hinge rotation against the competing-load ratio: every solve marked.

    Each branch is nearly flat, so lines alone read as strokes someone drew rather than as
    measurements; every level is marked, and a line joins only levels that stayed on one branch.
    """
    for seg in _branch_segments(theta_deg):
        if seg.size > 1:
            ax.plot(ratio[seg], theta_deg[seg], "-", color=FST.INK, lw=1.2,
                    solid_capstyle="round", zorder=2)
    ax.plot(ratio, theta_deg, "o", color=FST.INK, ms=2.6, zorder=3,
            markeredgecolor="white", markeredgewidth=0.25)
    ax.set_ylim(0, theta_deg.max() * 1.14)
    ax.set_ylabel(r"$\vartheta$  (deg)")


def panel_stability(ax, ratio, lam_min):
    """(d) Smallest eigenvalue of the tangent stiffness: every state is strictly stable."""
    for seg in _branch_segments(lam_min, jump=1.0):
        if seg.size > 1:
            ax.plot(ratio[seg], lam_min[seg], "-", color=FST.INK, lw=1.2,
                    solid_capstyle="round", zorder=2)
    ax.plot(ratio, lam_min, "o", color=FST.INK, ms=2.6, zorder=3,
            markeredgecolor="white", markeredgewidth=0.25)
    ax.axhline(0.0, color=FST.GREY, lw=0.8, zorder=1)
    ax.set_ylim(0, lam_min.max() * 1.14)
    ax.set_xlabel("cantilever / opening force")
    ax.set_ylabel(r"$\lambda_{\min}$")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--energy", default="energy_open_squares")
    ap.add_argument("--cantilever", default="cantilever_window")
    ap.add_argument("--spectrum", default="spectrum_window")
    ap.add_argument("--sweep", default="open_squares")
    ap.add_argument("--states", default=os.path.join(ROOT, "insets", "two_states.npz"))
    ap.add_argument("--out", default=os.path.join(ROOT, "discussion_modes"))
    ap.add_argument("--lo-edge", type=float, default=None)
    ap.add_argument("--hi-edge", type=float, default=None)
    ap.add_argument("--scale", type=float, default=1.0)
    args = ap.parse_args()

    FST.use_agg()
    FST.apply_charter()
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    # ── left column: is it a mechanism? ────────────────────────────────────────────────────
    en = _read(args.energy, "energy_partition.json")["rows"]
    theta = np.degrees([r['theta_mean'] for r in en])
    energy_pct = 100 * np.array([r['non_rotational'] for r in en])
    shape_pct = 100 * np.maximum(1.0 - np.array([r['r2'] for r in en]), 1e-8)

    srows = _read(args.sweep)["rows"]
    npz = np.load(os.path.join(ROOT, args.sweep, "sweep.npz"))
    r_theta, radii = [], {src: [] for src, _, _ in BCS}
    for i, r in enumerate(srows):
        if f"residual__{i}" not in npz:
            continue
        out = perturbation_radius(npz[f"residual__{i}"], npz[f"face_pairs__{i}"],
                                  [src for src, _, _ in BCS])
        r_theta.append(float(np.interp(r['level'], [x['level'] for x in en], theta)))
        for src, _, _ in BCS:
            radii[src].append(out.radius_interp[src])
    r_theta = np.asarray(r_theta)

    # ── right column: which state? ─────────────────────────────────────────────────────────
    ct = _read(args.cantilever)["rows"]
    assert all(r['checks']['converged'] for r in ct), "an unconverged level is not reportable"
    ratio = np.array([r['level'] for r in ct])
    order = np.argsort(ratio)
    ratio = ratio[order]
    theta_load = 2.0 * np.degrees(np.abs([r['psi'] for r in ct]))[order]      # theta = 2 psi

    sp = _read(args.spectrum, "spectrum.json")["rows"]
    s_ratio = np.array([r['level'] for r in sp])
    lam = np.array([r['lam_min_signed'] for r in sp])
    s_order = np.argsort(s_ratio)
    s_ratio, lam = s_ratio[s_order], lam[s_order]
    n_neg = int(sum(r['n_neg'] for r in sp))
    assert n_neg == 0, (f"{n_neg} negative Hessian eigenvalues: the states are NOT all stable and "
                        "the caption's claim must change with the data")

    st = np.load(args.states)
    keep = np.array([0, 1, 2, int(st["levels"].size) - 1])[:N_STATES]
    nodes_list = [st[f"nodes__{k}"] for k in keep]
    state_theta = np.interp(np.asarray(st["levels"])[keep], [r['level'] for r in en], theta)

    w_bp, h_bp = FST.page_bp(1.0, aspect=ASPECT)
    assert w_bp > h_bp, "the plate must be landscape; page_bp's aspect is WIDTH/HEIGHT"

    with plt.rc_context(FST.paper_rc(scale=args.scale)):
        fig = plt.figure(figsize=FST.figsize_in(w_bp, h_bp))
        # left=0.125 reserves 46 bp for a two-line rotated label over log tick labels. At 0.115 the
        # stack is 44.3 bp wide in a 42.6 bp gutter, and nothing here crops, so the label was cut.
        gs = GridSpec(2, 2, figure=fig, left=0.125, right=0.985, bottom=0.105, top=0.955,
                      wspace=0.44, hspace=0.16)

        ax_a = fig.add_subplot(gs[0, 0])
        panel_departures(ax_a, theta, energy_pct, shape_pct, args.lo_edge, args.hi_edge)
        ax_a.tick_params(labelbottom=False)

        ax_b = fig.add_subplot(gs[0, 1])
        panel_states_vs_load(ax_b, ratio, theta_load)
        ax_b.tick_params(labelbottom=False)

        ax_c = fig.add_subplot(gs[1, 0], sharex=ax_a)
        panel_radius(ax_c, r_theta, radii)

        ax_d = fig.add_subplot(gs[1, 1], sharex=ax_b)
        panel_stability(ax_d, s_ratio, lam)

        # Each column shares one x variable: that IS the grouping, so it needs no heading.
        assert ax_a.get_xlim() == ax_c.get_xlim(), "the left column must share one x axis"
        assert ax_b.get_xlim() == ax_d.get_xlim(), "the right column must share one x axis"

        state_axes = panel_states(fig, ax_a, nodes_list, state_theta, theta, energy_pct)
        assert len({(a.get_xlim(), a.get_ylim()) for a in state_axes}) == 1, \
            "the states must share one scale or the progression is an artifact"

        for letter, axx in (("a", ax_a), ("b", ax_b), ("c", ax_c), ("d", ax_d)):
            bb = axx.get_position()
            FST.panel_letter(fig, letter, x=max(bb.x0 - 0.088, 0.004), y=bb.y1 + 0.014,
                             scale=args.scale)

        pdf, lo_pt, hi_pt = FST.save_paper_figure(fig, args.out, w_bp, h_bp, scale=args.scale,
                                                  strict_margins=True)

    print(f"[plot] {pdf}   {w_bp:.1f} x {h_bp:.1f} bp   aspect {w_bp/h_bp:.2f} landscape")
    print(f"[plot] type {lo_pt:.1f}-{hi_pt:.1f} pt printed")
    i = int(np.argmin(energy_pct))
    print(f"[plot] (a) energy min {energy_pct[i]:.2f}% at theta {theta[i]:.1f} deg; at max theta "
          f"energy {energy_pct[-1]:.1f}% vs shape {shape_pct[-1]:.2e}%")
    for src, lab, _ in BCS:
        v = np.asarray(radii[src], dtype=float)
        print(f"[plot] (c) {lab}: {np.nanmin(v):.2f}-{np.nanmax(v):.2f} tiles, "
              f"no zone at {int((~np.isfinite(v)).sum())}/{v.size} openings")
    print(f"[plot] (b) theta {theta_load.min():.1f}-{theta_load.max():.1f} deg, "
          f"{len(_branch_segments(theta_load))} branches")
    print(f"[plot] (d) lam_min {lam.min():.3f}-{lam.max():.3f} over {len(lam)} levels, "
          f"negative eigenvalues: {n_neg}")


if __name__ == "__main__":
    main()
