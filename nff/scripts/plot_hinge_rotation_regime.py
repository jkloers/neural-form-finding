"""Rotation-only hinge response of the learned surrogate, as ONE region colored by fracture.

The surrogate does not give a single curve -- swept over the ligament width ``w_lig`` (the hinge
stiffness knob) it fills a whole REGION of the (theta, .) plane. We draw that region as a smooth band
colored by the surrogate's continuous ductile DAMAGE ``D`` (0 intact, >=1 fractured), with the
``D = 1`` fracture front drawn across it, the soft/stiff envelope curves as its edges, and one
representative curve highlighted inside.

Two quantities (``--quantity``):
  * ``energy``  -- stored energy ``W(theta)`` [N.mm]. Monotonic: the buckling/plastic softening shows
                   as a decreasing SLOPE, not a turnover (W = area under the moment, which stays >0).
  * ``moment``  -- restoring moment ``M = dW/dtheta`` [N.mm]. Rises, PEAKS as out-of-plane buckling +
                   steel plasticity set in, then softens down -- the classic buckling signature.

Pure rotation: ``a = s = 0``, cut angle ``alpha = 90 deg``. Run in the JAX env:

    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu conda run -n kgnn_mac \
        python nff/scripts/plot_hinge_rotation_regime.py --quantity moment
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Rectangle, ConnectionPatch
import matplotlib.patheffects as pe

from nff.scripts.render_hinge_3d import STRESS_CMAP     # shared green von Mises ramp (inset ↔ scale)

# ── project charter (Princeton palette) ───────────────────────────────────────────
ORANGE, RED, GREY, INK = "#F58025", "#D62828", "#6C757D", "#1A1A1A"
SLATE = "#5A5A5A"    # standard neutral grey for the uz/t buckle iso-lines
ROM_GREY = "#AEB4BD"  # light neutral for the (failing) paraboloid ROM -- deliberately understated


def build_damage_cmap(dcap):
    """Damage colormap = tints of the cut-pattern orange #F58025 (matches hinge_overview): light at
    D=0, deepening to full #F58025 at the top of the shown range. The D=1 fracture front is marked
    separately by the red iso-line, so the fill just needs a clean monotone orange deepening."""
    nodes = [(0.00, "#FEEFDD"), (0.35, "#FBD09E"), (0.62, "#F8AC5E"),
             (0.84, "#F69038"), (1.00, "#F58025")]
    return LinearSegmentedColormap.from_list("dmg_orange", nodes)


def apply_charter():
    plt.rcParams.update({
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.18, "grid.linewidth": 0.6,
        "axes.edgecolor": INK, "axes.linewidth": 0.9, "axes.labelcolor": INK,
        "text.color": INK, "xtick.color": GREY, "ytick.color": GREY,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "legend.frameon": True, "legend.framealpha": 0.92, "legend.edgecolor": "#D3D6DB",
    })


def surrogate_grid(ckpt, w_min, w_max, n_w=56, alpha_deg=90.0, n_theta=180, theta_max_rad=0.656):
    """Grid the surrogate over (w_lig, theta) at pure rotation.

    Returns ``theta_deg`` (n_theta,), ``w_ligs`` (n_w,), and ``W``, ``M=dW/dtheta``, ``D`` each
    (n_w, n_theta).
    """
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from nff.models.hinge_surrogate import (
        load_hinge_surrogate, apply_hinge_energy, apply_hinge_force, apply_hinge_failure,
        _geom_vector)

    params, stats, _ = load_hinge_surrogate(ckpt)
    theta = jnp.linspace(0.0, theta_max_rad, n_theta)
    z = jnp.zeros_like(theta)
    u = jnp.stack([z, z, theta], axis=-1)                  # (a, s, theta) pure rotation
    alpha = jnp.full_like(theta, np.radians(alpha_deg))
    w_ligs = np.linspace(w_min, w_max, n_w)

    W, M, D = [], [], []
    for wl in w_ligs:
        g = _geom_vector(jnp.full_like(theta, wl), alpha, 0.16, stats)
        W.append(np.asarray(apply_hinge_energy(params, u, g, stats)))
        M.append(np.asarray(apply_hinge_force(params, u, g, stats))[:, 2])   # dW/dtheta
        D.append(np.asarray(apply_hinge_failure(params, u, g, stats)))
    return np.degrees(np.asarray(theta)), w_ligs, np.asarray(W), np.asarray(M), np.asarray(D)


def build_uz_field(data_npz, theta_deg_grid, w_grid, kin_tol=0.18):
    """Out-of-plane buckle amplitude ``uz/t`` as a SMOOTH, MONOTONE field on the (w_lig, theta) grid.

    The surrogate has no ``uz`` head; we read it from the FEA database (near-pure-rotation samples).
    Thickness is a constant 1 mm here, so ``uz/t == uz_max``. To get clean contours even where data is
    sparse (past fracture), we take a robust BINNED MEDIAN on a coarse regular grid, then fill gaps
    ROW-WISE along theta and COLUMN-WISE along w by 1-D interpolation (flat extrapolation at the ends --
    no 2-D nearest-neighbour plateaus that produce contour loops), lift to the plot grid with a regular
    interpolator, and lightly Gaussian-smooth. Returns (n_w, n_theta).
    """
    from scipy.interpolate import RegularGridInterpolator
    from scipy.ndimage import gaussian_filter
    d = np.load(data_npz)
    a, s, th, wl, uz = d["a"], d["s"], d["theta_deg"], d["w_lig"], d["uz_max"]
    m = (a < kin_tol) & (np.abs(s) < kin_tol)
    th, wl, uz = th[m], wl[m], uz[m]

    te = np.linspace(0, theta_deg_grid.max(), 20)
    we = np.linspace(1, 10, 14)
    tcen, wcen = 0.5 * (te[:-1] + te[1:]), 0.5 * (we[:-1] + we[1:])
    ti = np.clip(np.digitize(th, te) - 1, 0, len(te) - 2)
    wi = np.clip(np.digitize(wl, we) - 1, 0, len(we) - 2)
    Z = np.full((len(wcen), len(tcen)), np.nan)
    for i in range(len(tcen)):
        for j in range(len(wcen)):
            sel = (ti == i) & (wi == j)
            if sel.sum() >= 3:
                Z[j, i] = np.median(uz[sel])

    def fill_1d(vec, x):
        v = ~np.isnan(vec)
        if v.sum() >= 2:
            return np.interp(x, x[v], vec[v])          # flat-extrapolates past the ends
        return np.full_like(vec, vec[v][0]) if v.sum() == 1 else vec
    for j in range(len(wcen)):                          # smooth along theta within each width
        Z[j] = fill_1d(Z[j], tcen)
    for i in range(len(tcen)):                          # then along width within each theta
        Z[:, i] = fill_1d(Z[:, i], wcen)
    Z = gaussian_filter(np.nan_to_num(Z, nan=float(np.nanmean(Z))), sigma=1.0)

    itp = RegularGridInterpolator((wcen, tcen), Z, bounds_error=False, fill_value=None)
    WW, TT = np.meshgrid(w_grid, theta_deg_grid, indexing="ij")
    Zf = itp(np.column_stack([WW.ravel(), TT.ravel()])).reshape(WW.shape)
    return gaussian_filter(Zf, sigma=1.2)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="data/surrogates/hinge_surrogate.pkl")
    ap.add_argument("--data", default="data/fea/hinge_dataset.npz")
    ap.add_argument("--quantity", choices=["energy", "moment"], default="energy")
    ap.add_argument("--out", default=None)
    ap.add_argument("--alpha-fill", type=float, default=0.78, help="region fill opacity (lower = lighter)")
    ap.add_argument("--d-show", type=float, default=1.3,
                    help="cap the damage landscape at this D (mask beyond it -- unsupported by data)")
    ap.add_argument("--paths", type=float, nargs="+", default=[9.0, 1.5],
                    help="w_lig [mm] of the load paths to trace (plain lines)")
    ap.add_argument("--rom-w", type=float, default=9.0, help="width the paraboloid ROM is calibrated to")
    ap.add_argument("--hinge-prefix", default="data/outputs/hinge_render_w9",
                    help="prefix for the hinge render PNGs (<prefix>_D0.5.png, _D1.0.png)")
    ap.add_argument("--compare", action="store_true",
                    help="also place the thin lower-path (w=1.5) hinge renders, for comparison")
    ap.add_argument("--hinge-prefix2", default="data/outputs/hinge_render_w1p5",
                    help="prefix for the lower-path (w=1.5) hinge renders in --compare mode")
    ap.add_argument("--stress-scale", action="store_true",
                    help="add the shared green von Mises scale for stress-coloured hinge insets")
    ap.add_argument("--vmin", type=float, default=0.0, help="von Mises scale min [MPa] (matches insets)")
    ap.add_argument("--vmax", type=float, default=866.0, help="von Mises scale max [MPa] (matches insets)")
    ap.add_argument("--fea-points", action="store_true",
                    help="overlay the raw CalculiX training points (same slice: pure rotation, alpha~90)")
    ap.add_argument("--fea-alpha-tol", type=float, default=20.0,
                    help="alpha half-window [deg] for the FEA overlay (wider = more but off-slice points)")
    args = ap.parse_args()
    out = args.out or f"data/outputs/hinge_rotation_{args.quantity}{'_compare' if args.compare else ''}.png"

    # bracket the fan EXACTLY by the two load paths -> the region envelope is the paths themselves and
    # the D/uz contours terminate cleanly at the story dots (no tail hooking past into a wider grid).
    theta, w_ligs, W, M, D = surrogate_grid(args.ckpt, min(args.paths), max(args.paths),
                                            n_w=180, n_theta=360)   # fine grid -> smooth D-cap edge
    uz = build_uz_field(args.data, theta, w_ligs)              # out-of-plane buckle uz/t (t=1 mm)
    Y = W if args.quantity == "energy" else M
    ylabel = (r"stored energy  $W$  [N$\cdot$mm]" if args.quantity == "energy"
              else r"restoring moment  $M=\mathrm{d}W/\mathrm{d}\theta$  [N$\cdot$mm]")
    title = "Surrogate-learned energy landscape of a steel hinge"

    def curve(w):                                              # nearest gridded trajectory for a width
        return Y[int(np.argmin(np.abs(w_ligs - w)))]

    def dcurve(w):                                             # ductile damage along that trajectory
        return D[int(np.argmin(np.abs(w_ligs - w)))]

    apply_charter()
    fig, ax = plt.subplots(figsize=(10.6, 6.3))
    d_show = args.d_show                                   # stop the landscape at D=d_show (data-supported)
    dcap = d_show
    cmap = build_damage_cmap(dcap)
    dnorm = Normalize(0.0, dcap)
    X = np.tile(theta, (len(w_ligs), 1))
    # rescale axes to the shown (D<=d_show) region: y to its tallest point, x just past where the whole
    # width-band has failed (so the fill ends naturally at the curved D=d_show contour, not a hard edge).
    ymax = 1.06 * float(np.nanmax(np.where(D <= d_show, Y, np.nan)))
    minD = D.min(axis=0)
    xhi = min(float(theta[-1]),
              float(np.interp(d_show, minD, theta)) + 1.2 if minD.max() >= d_show else float(theta[-1]))

    # 1. damage as the coloured landscape (Princeton orange). Levels stop AT d_show with NO `extend`, so
    #    contourf leaves the D>d_show corner unfilled bounded by the smooth marching-squares d_show
    #    contour -- a clean curved edge, unlike the ragged cell-drop a NaN mask produces.
    ax.contourf(X, Y, D, levels=np.linspace(0.0, dcap, 61), cmap=cmap, norm=dnorm,
                alpha=args.alpha_fill, zorder=2, antialiased=True)

    # 1b. raw CalculiX training points behind the surrogate surface -- a band of the 6-D dataset around
    #     this figure's slice (pure rotation a=s=0, alpha within +-tol of 90 deg, w_lig in the shown
    #     band). The surface is the alpha=90 reference; the points' spread about it is the alpha spread.
    fea_handle = None
    if args.fea_points:
        fd = np.load(args.data)
        yq = fd["W"] if args.quantity == "energy" else fd["M_theta"]
        m = ((np.abs(fd["a"]) < 1e-6) & (np.abs(fd["s"]) < 1e-6)
             & (np.abs(fd["alpha_deg"] - 90.0) <= args.fea_alpha_tol)
             & (fd["w_lig"] >= min(args.paths) - 1e-9) & (fd["w_lig"] <= max(args.paths) + 1e-9)
             & (yq <= ymax))
        fea_handle = ax.scatter(fd["theta_deg"][m], yq[m], s=3.5, c="#787878", alpha=0.08,
                                linewidths=0.0, zorder=4, rasterized=True)

    # 2. iso-lines: out-of-plane buckle altitude uz/t (smooth monotone landscape). Light, and stopped
    #    at the shown D=d_show boundary so they end with the coloured fill, not over blank paper.
    uz_levels = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
    uz_valid = np.where(D > d_show, np.nan, uz)
    cB = ax.contour(X, Y, uz_valid, levels=uz_levels, colors=SLATE, linewidths=0.7, alpha=0.5, zorder=5)

    def contour_pt(field, level, target_theta):
        """A point ON the (theta, W) contour of ``field=level`` nearest ``target_theta`` -- for
        placing labels in clear zones instead of wherever matplotlib drops them."""
        ti = int(np.argmin(np.abs(theta - target_theta)))
        col, wc = field[:, ti], Y[:, ti]
        for j in range(len(col) - 1):
            a, b = col[j], col[j + 1]
            if np.isfinite(a) and np.isfinite(b) and (a - level) * (b - level) <= 0 and a != b:
                f = (level - a) / (b - a)
                return (float(theta[ti]), float(wc[j] + f * (wc[j + 1] - wc[j])))
        return None

    # push uz/t labels to the clear right side (high theta), each at the highest theta where it lives
    uz_manual = [p for L in uz_levels
                 for p in (next((contour_pt(uz_valid, L, tt) for tt in np.linspace(xhi - 2.5, 12, 48)
                                 if contour_pt(uz_valid, L, tt)), None),) if p]
    uz_lbls = ax.clabel(cB, manual=uz_manual, fontsize=7, colors=SLATE, fmt=lambda v: f"$u_z/t$={v:g}")

    # 2b. damage iso-lines: the D = 0.5 and D = 1 (fracture) contours -- the hinge renders sit exactly
    #     where their load path crosses these, so the lines locate every snapshot. Labels placed in the
    #     open lower-left (D=0.5) and upper-mid (D=1) so they clear the uz/t labels and the insets.
    cD = ax.contour(X, Y, D, levels=[0.5, 1.0], colors=[ORANGE, RED], linewidths=[1.3, 1.8], zorder=6)
    d_manual = [(11.2, 0.24 * ymax),                          # clear mid-spot on the ~vertical D=0.5 line
                (23.4, 0.47 * ymax)]                          # upper red line, above the lower-right inset
    d_lbls = ax.clabel(cD, manual=d_manual, fontsize=8.5, colors=INK,
                       fmt={0.5: r"$D=0.5$", 1.0: r"$D=1$"})

    # white halo -> every contour label stays legible over the coloured landscape
    for t in list(uz_lbls) + list(d_lbls):
        t.set_path_effects([pe.withStroke(linewidth=1.8, foreground="white")])

    # 3. load paths: plain thin lines at fixed w_lig, truncated where they leave the shown D<=d_show band
    for w in sorted(args.paths, reverse=True):
        yv, dv = curve(w), dcurve(w)
        keep = dv <= d_show
        ax.plot(theta[keep], yv[keep], color=INK, lw=1.4, zorder=8, solid_capstyle="round")

    # 3b. story dots on each load path at D = 0.5 and D = 1 (where the hinge visuals are taken)
    for w in sorted(args.paths, reverse=True):
        yv, dv = curve(w), dcurve(w)
        for dval in (0.5, 1.0):
            tx = float(np.interp(dval, dv, theta))
            ax.plot([tx], [float(np.interp(tx, theta, yv))], "o", color=INK, mec="white",
                    mew=1.3, ms=7.5, zorder=11)

    # 4. the paraboloid ROM 1/2 k theta^2, calibrated at theta->0 on the w=rom_w curve. Deliberately
    #    LIGHT + with its formula: it is shown only because it FAILS away from the origin.
    th_rad = np.radians(theta)
    Yrom = curve(args.rom_w)
    fitm = theta <= 4.0
    if args.quantity == "energy":
        k = np.dot(0.5 * th_rad[fitm] ** 2, Yrom[fitm]) / np.dot(0.5 * th_rad[fitm] ** 2, 0.5 * th_rad[fitm] ** 2)
        rom = 0.5 * k * th_rad ** 2
    else:
        k = np.dot(th_rad[fitm], Yrom[fitm]) / np.dot(th_rad[fitm], th_rad[fitm])
        rom = k * th_rad
    ax.plot(theta, rom, ls=(0, (5, 2)), color=ROM_GREY, lw=1.6, zorder=3)
    ip = int(np.argmin(np.abs(rom - 0.62 * ymax)))            # a point partway up the ROM
    ax.annotate(r"$\frac{1}{2}\,k\,\theta^2$", (theta[ip], rom[ip]), fontsize=12, color=ROM_GREY,
                ha="right", va="center", xytext=(-8, 0), textcoords="offset points", zorder=3)

    # 5. hinge renders in cards, leader-lined to the load-path dots at D=0.5 and D=1. Each panel is
    #    sized to the render's cropped content so the leader + D label sit right on the hinge.
    fig_asp = fig.get_figwidth() / fig.get_figheight()
    bw, pad = 0.17, 0.008

    def place_hinge(path_w, prefix, dval, lbl, gap=0.05, dx_nudge=0.0):
        """Seat the render just above its story dot (auto axes-fraction from the dot, so it survives any
        x/y rescale); drop below the dot only if it would run off the top."""
        yv, dv = curve(path_w), dcurve(path_w)
        tx = float(np.interp(dval, dv, theta))
        dx, dy = tx, float(np.interp(tx, theta, yv))
        dax, day = dx / xhi, dy / ymax                    # dot in axes fraction (xlo = ylo = 0)
        img_path = f"{prefix}_D{lbl}.png"
        im, bh = None, 0.14
        if os.path.exists(img_path):
            im = plt.imread(img_path)                     # crop transparent padding -> tight to hinge
            ys, xs = np.where(im[:, :, 3] > 0.02)
            H, Wd = im.shape[:2]                           # headroom on top so the buckle peak clears
            im = im[max(ys.min() - 14, 0):min(ys.max() + 3, H),
                    max(xs.min() - 4, 0):min(xs.max() + 4, Wd)]
            bh = bw * fig_asp * im.shape[0] / im.shape[1]
        by = day + gap
        if by + bh > 0.97:                                # near the top -> drop the render below its dot
            by = day - gap - bh
        bx = min(max(dax - bw / 2 + dx_nudge, 0.006), 0.994 - bw)
        if im is not None:                                # no card: hinge reads on the warm landscape
            inset = ax.inset_axes([bx, by, bw, bh]); inset.imshow(im); inset.axis("off")
            inset.patch.set_alpha(0.0); inset.set_zorder(12)
        else:
            ax.add_patch(Rectangle((bx, by), bw, bh, transform=ax.transAxes, facecolor="white",
                         edgecolor=GREY, ls=(0, (4, 3)), lw=1.0, zorder=12))
        leader_y = (by + bh + pad) if (by + bh / 2) < day else (by - pad)
        ax.add_artist(ConnectionPatch((dx, dy), (bx + bw / 2, leader_y), coordsA=ax.transData,
                      coordsB=ax.transAxes, color=GREY, lw=0.8, zorder=10))

    place_hinge(args.rom_w, args.hinge_prefix, 0.5, "0.5")
    place_hinge(args.rom_w, args.hinge_prefix, 1.0, "1.0")
    if args.compare:                                       # thin lower path (w=1.5), close to its curve
        place_hinge(1.5, args.hinge_prefix2, 0.5, "0.5")
        place_hinge(1.5, args.hinge_prefix2, 1.0, "1.0")

    ax.set_xlabel(r"hinge rotation  $\theta$  [deg]")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, xhi)
    ax.set_ylim(0, ymax)

    # layout: reserve a right margin and give the scales their own well-separated axes. Both bars are
    # labelled consistently -- ticks and title on the RIGHT, all text in INK -- so they read as a set.
    fig.subplots_adjust(left=0.065, right=0.75 if args.stress_scale else 0.86,
                        top=0.965, bottom=0.10)
    cb_y, cb_h, cb_w = 0.12, 0.79, 0.018

    def style_cbar(cb, label):
        cb.set_label(label, fontsize=9.5, color=INK)
        cb.ax.tick_params(labelsize=8.5, colors=INK)          # default side is right for both -> uniform
        cb.outline.set_edgecolor("#D3D6DB")

    if args.stress_scale:
        cax_vm = fig.add_axes([0.805, cb_y, cb_w, cb_h])
        smv = ScalarMappable(norm=Normalize(args.vmin, args.vmax), cmap=STRESS_CMAP); smv.set_array([])
        style_cbar(fig.colorbar(smv, cax=cax_vm), r"von Mises stress  $\sigma_\mathrm{vM}$  [MPa]")
        dmg_x = 0.905
    else:
        dmg_x = 0.885

    cax_d = fig.add_axes([dmg_x, cb_y, cb_w, cb_h])
    sm = ScalarMappable(norm=dnorm, cmap=cmap); sm.set_array([])
    cb = fig.colorbar(sm, cax=cax_d)
    style_cbar(cb, r"ductile damage  $D$   ($D\geq1$: fracture)")
    cb.ax.axhline(1.0 / dcap, color=INK, lw=0.8, alpha=0.5)

    leg_h = [plt.Line2D([], [], color=INK, lw=1.4),
             plt.Line2D([], [], color=SLATE, lw=0.8),
             plt.Line2D([], [], color=ROM_GREY, lw=1.6, ls=(0, (5, 2)))]
    leg_l = ["load path (fixed $w_\\mathrm{lig}$)", r"buckle altitude  $u_z/t$",
             r"paraboloid ROM  $\frac{1}{2}k\theta^2$"]
    if fea_handle is not None:
        tol = args.fea_alpha_tol
        leg_h.append(plt.Line2D([], [], marker="o", markerfacecolor="#707070", markeredgecolor="none",
                                markersize=3.5, alpha=0.55, linestyle="none"))
        leg_l.append(rf"FEA data  ($\alpha \in [{90 - tol:g},\,{90 + tol:g}]^\circ$)")
    leg = ax.legend(leg_h, leg_l, loc="upper left", fontsize=8.5)
    leg.set_zorder(30)                                         # keep the ROM (and all lines) behind it
    leg.get_frame().set_edgecolor("#D3D6DB")

    # ligament-width labels, rotated PARALLEL to each load path at the label point (needs the final
    # data->display transform, so it happens after the manual layout is fixed)
    fig.canvas.draw()
    chip = dict(facecolor="white", alpha=0.82, edgecolor="none", boxstyle="round,pad=0.15")
    for w in sorted(args.paths, reverse=True):
        yv = curve(w)
        tl = (0.48 if w == min(args.paths) else 0.62) * theta[-1]   # nudge the thin-path label left
        i = int(np.clip(np.searchsorted(theta, tl), 1, len(theta) - 1))
        (x0, y0), (x1, y1) = (ax.transData.transform((theta[j], yv[j])) for j in (i - 1, i))
        ang = np.degrees(np.arctan2(y1 - y0, x1 - x0))
        ax.text(tl, float(np.interp(tl, theta, yv)), f"$w_\\mathrm{{lig}}={w:g}$ mm",
                rotation=ang, rotation_mode="anchor", ha="center", va="bottom",
                fontsize=8.5, color=INK, bbox=chip, zorder=9)

    fig.savefig(out, dpi=300)
    print(f"quantity={args.quantity}  paths={args.paths}  uz/t max={np.nanmax(uz):.2f}  "
          f"Dmax={D.max():.2f}  -> {out}")


if __name__ == "__main__":
    main()
