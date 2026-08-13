"""Accuracy of the condensed hinge-energy surrogate against the finite-element oracle.

Three parity panels: the stored energy, its gradient and the damage head. Everything else the
manuscript asserts about the surrogate is a scalar rather than a picture, so it is measured here and
written into the caption sidecar instead of being drawn: the error resolved across fold depth and
ligament width, the linear-spring control, the spread behind the quoted norm ratio, and the
deployment coverage.

    JAX_PLATFORMS=cpu conda run -n kgnn_mac --no-capture-output \
        python nff/scripts/figures/plot_surrogate_validity.py
"""
import argparse
import os

import numpy as np

from nff.scripts.figures._surrogate_eval import (DATASET, ROM_BAND, SURROGATE, deployment_states,
                                                 force_parity, load_scored, recorded_metrics,
                                                 rel_rmse, rom_energy, r2)
from nff.utils.figstyle import (FS_ANNOT, FS_BASE, GREY, INK, ORANGE, RED, TEAL, apply_charter,
                                figsize_in, page_bp, panel_letter, paper_rc, save_paper_figure,
                                use_agg)

# Authored at the width the manuscript includes it at, so an authored point size is a printed one.
# The three panels are SQUARE parity plots, so the canvas aspect is not free: at this width each
# panel comes out about 109 bp on a side, and any taller canvas is dead band above and below them.
FIG_W_BP, FIG_H_BP = page_bp(1.15, 2.85)       # \includegraphics[width=1.15\textwidth]
# Strokes and markers keep the printed size they were reviewed at, so they come down by the factor
# the old canvas was reduced by on its way to the page.
DRAW = 426.2 / 561.6
dw = lambda x: x * DRAW
dA = lambda x: x * DRAW ** 2                   # scatter `s` is an AREA
use_agg()
import matplotlib.pyplot as plt                                        # noqa: E402

BOX = dict(boxstyle="round,pad=0.28", fc="white", ec="#D3D6DB", lw=0.6, alpha=0.92)
DEPLOYMENTS = (("data/outputs/hinge_paths/pet45_baseline/hinge_paths", "shape objective only"),
               ("data/outputs/hinge_paths/pet45_cP10_dP/hinge_paths", "with compression penalty"))

THETA_EDGES = np.arange(0.0, 80.0, 10.0)        # degrees of relative rotation
WLIG_EDGES = np.arange(5.0, 27.0, 3.0)          # mm of ligament width
MIN_BIN = 30                                    # bins thinner than this are not reported


CORNERS = ((0.035, 0.965, "top", "left"), (0.965, 0.965, "top", "right"),
           (0.035, 0.035, "bottom", "left"), (0.965, 0.035, "bottom", "right"))


def _obstacles(ax, series, lims, n_line=600):
    """-> (N, 2) in axes coordinates: every drawn point plus a dense sample of the identity line.

    An annotation is placed against this, not against the eye. A parity cloud leaves two corners
    free in principle, but which two depends on where the outliers fell.
    """
    pts = [np.column_stack([np.asarray(x, float), np.asarray(y, float)]) for x, y in series]
    line = np.linspace(lims[0], lims[1], n_line)
    pts.append(np.column_stack([line, line]))
    return ax.transLimits.transform(np.vstack(pts))


def _overlaps_legend(ax, bb, pad):
    """The legend is placed first and is itself data-avoiding, so the box must clear it too."""
    leg = ax.get_legend()
    if leg is None:
        return False
    lb = leg.get_window_extent().transformed(ax.transAxes.inverted())
    return not (bb.x1 + pad < lb.x0 or bb.x0 - pad > lb.x1
                or bb.y1 + pad < lb.y0 or bb.y0 - pad > lb.y1)


def _corner(ax, text, obstacles, fs=FS_ANNOT, pad=0.012):
    """Place the annotation in the first corner whose rendered box covers nothing that was drawn."""
    fig = ax.figure
    for x, y, va, ha in CORNERS:
        t = ax.text(x, y, text, transform=ax.transAxes, va=va, ha=ha, fontsize=fs, color=INK,
                    bbox=BOX, zorder=6)
        fig.canvas.draw()
        bb = t.get_window_extent().transformed(ax.transAxes.inverted())
        hit = ((obstacles[:, 0] >= bb.x0 - pad) & (obstacles[:, 0] <= bb.x1 + pad)
               & (obstacles[:, 1] >= bb.y0 - pad) & (obstacles[:, 1] <= bb.y1 + pad))
        if not hit.any() and not _overlaps_legend(ax, bb, pad):
            return
        t.remove()
    raise SystemExit(f"no corner of {ax.get_title()!r} is free of data; the annotation would hide "
                     "part of the result. Widen the axis limits or shorten the box.")


def _parity(ax, true, pred, color):
    """-> the axis limits used, so the caller can build the obstacle set against them."""
    ax.scatter(true, pred, s=dA(6), c=color, alpha=0.35, linewidths=0, zorder=3)
    lo = min(float(true.min()), float(pred.min()))
    hi = max(float(true.max()), float(pred.max()))
    pad = 0.03 * (hi - lo)
    lims = (lo - pad, hi + pad)
    ax.plot(lims, lims, color=INK, lw=dw(1.1), ls="--", zorder=1, alpha=0.8)
    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax.set_aspect("equal", "box")
    return lims


def _binned(x, true, pred, edges):
    """Relative RMS error of ``pred`` per bin of ``x``. -> (centres, per cent, rows dropped)."""
    centres, vals, dropped = [], [], 0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi)
        if int(m.sum()) < MIN_BIN:
            dropped += int(m.sum())
            continue
        centres.append(0.5 * (lo + hi))
        vals.append(rel_rmse(pred[m], true[m]) * 100.0)
    return np.array(centres), np.array(vals), dropped


def _component_errors(sc, mask, surrogate):
    """Per-component relative RMS force errors, checked against the trainer's own record.

    The trainer reports the MEAN of the three, which is the number the manuscript quotes. Pooling
    the raw components instead would report the moment alone, since its RMS exceeds the two forces
    by an order of magnitude.
    """
    per = [rel_rmse(sc.F_pred[mask, j], sc.data["F"][mask, j]) for j in range(3)]
    rec = recorded_metrics(surrogate)
    want = [rec.get(k) for k in ("force_rel_a", "force_rel_s", "force_rel_m")]
    if all(w is not None for w in want):
        drift = max(abs(p - w) for p, w in zip(per, want))
        if drift > 1e-6:
            raise SystemExit(f"force scoring drifted from the checkpoint by {drift:.2e}: "
                             f"got {per}, recorded {want}.")
    return per, float(np.mean(per))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--surrogate", default=SURROGATE)
    ap.add_argument("--data", default=DATASET)
    ap.add_argument("--out", default="paper/Images/surrogate_validity.png")
    ap.add_argument("--dpi", type=int, default=600)
    args = ap.parse_args()

    sc = load_scored(args.surrogate, args.data)
    te = sc.test
    cap = {}

    # The canvas IS the printed size, so an authored point size is a printed point size. "Near"
    # was not enough: at 7.8 in against 1.15\textwidth the plate still reduced by 0.77 and its type
    # landed off the ladder every other figure keeps.
    apply_charter(**paper_rc())
    fig, axes = plt.subplots(1, 3, figsize=figsize_in(FIG_W_BP, FIG_H_BP))

    # (a) energy -------------------------------------------------------------------------------
    ax = axes[0]
    Wt, Wp = sc.data["W"], sc.W_pred
    lims = _parity(ax, Wt[te], Wp[te], ORANGE)
    e_r2, e_rr = r2(Wp[te], Wt[te]), rel_rmse(Wp[te], Wt[te])
    _corner(ax, f"$R^2 = {e_r2:.4f}$\nrel. RMSE $= {e_rr*100:.2f}$%",
            _obstacles(ax, [(Wt[te], Wp[te])], lims))
    ax.set_xlabel(r"oracle $\mathcal{W}$  [N mm]")
    ax.set_ylabel(r"surrogate $\mathcal{W}$  [N mm]")
    ax.set_title("Stored energy", pad=5, fontsize=FS_BASE, color=INK)
    cap["energy"] = (e_r2, e_rr)

    # (b) force --------------------------------------------------------------------------------
    # Each component is divided by its own RMS so that one parity can carry all three; the error
    # reported is the per-component relative error, which is what the trainer records.
    ax = axes[1]
    Ft_n, Fp_n, _, f_r2, _ = force_parity(sc, te)
    per, mean_rel = _component_errors(sc, te, args.surrogate)
    names = [r"$F_a$", r"$F_s$", r"$M_\vartheta$"]
    for j, (cj, nj) in enumerate(zip((TEAL, RED, GREY), names)):
        ax.scatter(Ft_n[:, j], Fp_n[:, j], s=dA(6), c=cj, alpha=0.35, linewidths=0, zorder=3,
                   label=f"{nj}  ({per[j]*100:.1f}%)")
    lim = np.percentile(np.abs(Ft_n), 99.8) * 1.15
    ax.plot((-lim, lim), (-lim, lim), color=INK, lw=dw(1.1), ls="--", zorder=1, alpha=0.8)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", "box")
    obs = _obstacles(ax, [(Ft_n[:, j], Fp_n[:, j]) for j in range(3)], (-lim, lim))
    # Kept short: at this canvas size the full "/ component RMS" label is taller than the axes.
    ax.set_xlabel("oracle force / RMS")
    ax.set_ylabel("surrogate force / RMS")
    ax.set_title(r"Internal force $\partial\mathcal{W}/\partial\mathbf{u}$", pad=5,
                 fontsize=FS_BASE, color=INK)
    ax.legend(loc="best", fontsize=FS_ANNOT, handletextpad=0.3, borderpad=0.35, markerscale=1.6)
    _corner(ax, f"$R^2 = {f_r2:.4f}$\nrel. RMSE $= {mean_rel*100:.1f}$%", obs)
    cap["force"] = (f_r2, mean_rel, per)

    # (c) damage -------------------------------------------------------------------------------
    ax = axes[2]
    Dt, Dp = sc.data["margin"], sc.D_pred
    lims = _parity(ax, Dt[te], Dp[te], TEAL)
    d_r2, d_rr = r2(Dp[te], Dt[te]), rel_rmse(Dp[te], Dt[te])
    _corner(ax, f"$R^2 = {d_r2:.4f}$\nrel. RMSE $= {d_rr*100:.2f}$%",
            _obstacles(ax, [(Dt[te], Dp[te])], lims))
    ax.set_xlabel(r"oracle $\Delta$")
    ax.set_ylabel(r"surrogate $\hat{D}$")
    ax.set_title("Ductile-damage head", pad=5, fontsize=FS_BASE, color=INK)
    cap["damage"] = (d_r2, d_rr)

    # Letters BEFORE tight_layout: they sit above the axes, and tight_layout only reserves room for
    # artists that already exist when it runs. Added afterwards they are simply cropped off the top.
    for ax, letter in zip(axes, "abc"):
        panel_letter(ax, letter, x=-0.20, y=1.04)
    fig.tight_layout(w_pad=1.4)

    pdf, lo, hi = save_paper_figure(fig, args.out, FIG_W_BP, FIG_H_BP, dpi=args.dpi)
    print(f"  wrote {pdf} ({FIG_W_BP:.1f} x {FIG_H_BP:.1f} bp, type {lo:.1f}-{hi:.1f} pt)")
    plt.close(fig)

    cap["prov"] = _provenance(sc, te)
    _caption(os.path.splitext(args.out)[0] + ".txt", sc, cap, args)
    print(f"  wrote {args.out}\n  wrote {pdf}")
    for k, v in cap.items():
        print(f"  {k}: {v}")


def _provenance(sc, te):
    """Every number the manuscript asserts about the surrogate that no panel shows.

    Dropping a panel must not drop the traceability of what the text still claims, so the caption
    sidecar is the record for all of it.
    """
    fold = _binned(np.degrees(sc.u[te, 2]), sc.data["W"][te], sc.W_pred[te], THETA_EDGES)
    width = _binned(sc.w_lig[te], sc.data["W"][te], sc.W_pred[te], WLIG_EDGES)

    band = te & sc.band()
    true_b = sc.data["W"][band]
    floor = np.sqrt(np.mean(true_b ** 2)) * 1e-3
    err_sur = np.abs(sc.W_pred[band] - true_b) / np.maximum(np.abs(true_b), floor)
    err_rom = np.abs(rom_energy(sc.u[band]) - true_b) / np.maximum(np.abs(true_b), floor)
    rom = (rel_rmse(sc.W_pred[band], true_b), rel_rmse(rom_energy(sc.u[band]), true_b),
           float(np.median(err_sur)), float(np.median(err_rom)), int(band.sum()))

    # relRMSE is a norm ratio, so a handful of high-energy rows can set it. The manuscript quotes it,
    # and must be able to quote the spread beside it: a median and a ninth decile say whether the
    # aggregate describes a typical row or a tail.
    true_t = sc.data["W"][te]
    floor_t = np.sqrt(np.mean(true_t ** 2)) * 1e-3
    err_t = np.abs(sc.W_pred[te] - true_t) / np.maximum(np.abs(true_t), floor_t)
    sq = (sc.W_pred[te] - true_t) ** 2
    top_decile = float(np.sort(sq)[::-1][:max(1, len(sq) // 10)].sum() / sq.sum())
    spread = (float(np.median(err_t)), float(np.percentile(err_t, 90)),
              rel_rmse(sc.W_pred[te], true_t), int(te.sum()), top_decile)

    dom = sc.stats["domain"]
    box_lo, box_hi = dom.get("eta_a_min", 0.0), dom["eta_a_max"]
    frac = {}
    for prefix, label in DEPLOYMENTS:
        _, eta = deployment_states(prefix, sc.stats)
        inside = ((eta[:, 0] >= box_lo) & (eta[:, 0] <= box_hi)
                  & (np.abs(eta[:, 1]) <= dom["eta_s_max"])
                  & (eta[:, 2] >= dom.get("theta_min", 0.0)) & (eta[:, 2] <= dom["theta_max"]))
        frac[label] = float(inside.mean())
    ridden = (box_lo, box_hi, float(np.degrees(dom["theta_max"])), frac)
    return fold, width, rom, spread, ridden


def _fmt(vals):
    return ", ".join(f"{v:.2f}" for v in vals)


def _caption(path, sc, cap, args):
    e_r2, e_rr = cap["energy"]
    f_r2, mean_rel, per = cap["force"]
    d_r2, d_rr = cap["damage"]
    fold, width, rom, spread, ridden = cap["prov"]
    c_th, v_th, drop_th = fold
    c_w, v_w, drop_w = width
    rr_s, rr_r, med_s, med_r, n_band = rom
    sp_med, sp_p90, sp_rr, sp_n, sp_top = spread
    box_lo, box_hi, th_box, frac = ridden
    with open(path, "w") as f:
        f.write(
            "Accuracy of the condensed hinge-energy surrogate against the finite-element oracle.\n"
            f"Checkpoint {os.path.basename(args.surrogate)}, campaign "
            f"{os.path.basename(args.data)}.\n\n"
            f"(a) Stored energy: R2 {e_r2:.4f}, relative RMS error {e_rr*100:.2f} per cent.\n"
            f"(b) Internal force: R2 {f_r2:.4f}, relative RMS error {mean_rel*100:.1f} per cent, "
            f"the mean over components of {per[0]*100:.1f} (F_a), {per[1]*100:.1f} (F_s) and "
            f"{per[2]*100:.1f} (M_theta), which is the metric the trainer records and is asserted "
            "against the checkpoint sidecar. Each component is divided by its own root-mean-square "
            "for the parity, so that three quantities differing by an order of magnitude in units "
            "share one axis.\n"
            f"(c) Ductile-damage head: R2 {d_r2:.4f}, relative RMS error {d_rr*100:.2f} per cent. "
            "No specimen in this campaign reached fracture, so the damage measure is reported "
            "against its own RMS and not against a calibrated tear line.\n\n"
            "Provenance, not part of the caption. The split is by hinge geometry and is "
            "reconstructed in the trainer's order, then asserted against the checkpoint sidecar: "
            f"{int(sc.train.sum())} fitted, {int(sc.val.sum())} for selection, "
            f"{int(sc.test.sum())} scored in every panel.\n"
            "The following are quoted in the text and have no panel.\n"
            f"Energy error in 10 degree bins of relative rotation, centres {_fmt(c_th)} degrees: "
            f"{_fmt(v_th)} per cent ({drop_th} states in bins below the {MIN_BIN}-state floor).\n"
            f"Energy error in 3 mm bins of ligament width, centres {_fmt(c_w)} mm: {_fmt(v_w)} per "
            f"cent ({drop_w} states below the floor).\n"
            "Linear-spring reference model on the same scored states restricted to ligament widths "
            f"{ROM_BAND[0]:.0f} to {ROM_BAND[1]:.0f} mm, the band bracketing the 18 mm at which its "
            f"stiffnesses were condensed (n = {n_band}): relative RMS error {rr_r*100:.0f} per cent "
            f"against {rr_s*100:.2f} for the surrogate, median {med_r*100:.0f} against "
            f"{med_s*100:.1f} per cent.\n"
            f"Spread of the held-out energy error over its {sp_n} rows, since the relative RMS "
            "error above is a norm ratio and not a typical row: median "
            f"{sp_med*100:.2f} per cent, ninth decile {sp_p90*100:.2f}, against the "
            f"{sp_rr*100:.2f} quoted. The worst tenth of the rows carries {sp_top*100:.0f} per "
            "cent of the squared error.\n"
            f"Trust region {box_lo:.3f} to {box_hi:.3f} in axial ratio and to {th_box:.1f} degrees "
            f"of rotation, holding {frac['shape objective only']*100:.0f} per cent of the states "
            "ridden by the deployment trained on shape alone and "
            f"{frac['with compression penalty']*100:.0f} per cent of those ridden with the "
            "compression penalty added.\n")


if __name__ == "__main__":
    main()
