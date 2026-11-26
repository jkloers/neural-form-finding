"""Rotation-only hinge response of the learned surrogate: a fold that ESCAPES OUT OF PLANE.

The surrogate does not give a single curve -- swept over the ligament width ``w_lig`` (the hinge
stiffness knob) it fills a whole REGION of the (theta, .) plane. The story the region tells is
buckling, not tearing: a PET ligament accommodates rotation by bowing out of plane, and it does so
more the WIDER it is (measured, near-pure rotation past 40 deg: ``uz/t`` median 53.8 at
``w_lig`` 21-25 mm against 20.9 at 5-9 mm). So the ``uz/t`` iso-lines are the protagonist here.

Ductile damage ``D`` stays as the fill for exactly one reason -- to show it goes nowhere. Over this
whole slice the surrogate predicts ``D <= 0.03`` and the PET database tops out at ``D = 0.065``,
against fracture at ``D = 1``: PET hinges do not tear by folding, they buckle. The colorbar is
therefore a near-uniform wash by design, and its range IS the result.

Two quantities (``--quantity``):
  * ``energy``  -- stored energy ``W(theta)`` [N.mm]. Monotonic: the buckling softening shows as a
                   decreasing SLOPE, not a turnover (W = area under the moment, which stays >0).
  * ``moment``  -- restoring moment ``M = dW/dtheta`` [N.mm]. Rises, PEAKS as the ligament buckles
                   out of plane, then softens down -- the classic buckling signature.

Pure rotation: ``a = s = 0``, cut angle ``alpha = 90 deg``. The manuscript's Figure 7 is the
``--compare`` composition -- four hinge renders, the damage scale alone, and the headroom the top
render needs to clear the ``u_z/t`` labels. Run in the JAX env:

    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu conda run -n kgnn_mac \
        python nff_paper/figures/plot_hinge_rotation_regime.py \
            --compare --fea-points --headroom 1.28 --out paper/Images/energy_landscape.png
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

from nff_paper.figures.render_hinge_3d import STRESS_CMAP     # shared green von Mises ramp (inset ↔ scale)

# ── project charter (Princeton palette) ───────────────────────────────────────────
from nff.utils.figstyle import (FS_ANNOT, FS_BASE, FS_TICK, GREY, INK, RED, apply_charter,
                                figsize_in, page_bp, paper_rc, save_paper_figure)

# The canvas is authored at the width the manuscript includes it at, so an authored point size is a
# printed one. Drawn 763 bp wide and reduced to \linewidth, its labels printed at 3.4 to 5.8 pt.
FIG_W_BP, FIG_H_BP = page_bp(1.0, 1.6825)      # \includegraphics[width=1\linewidth]
# Strokes and markers do NOT follow the type. The drawing keeps the printed size it was reviewed at,
# so every weight tuned on the old canvas comes down by the factor that canvas was reduced by.
DRAW = 370.6 / 763.2
dw = lambda x: x * DRAW                        # linewidths, marker sizes
dA = lambda x: x * DRAW ** 2                   # scatter `s`, which is an AREA
SLATE = "#5A5A5A"    # standard neutral grey for the uz/t buckle iso-lines
ROM_GREY = "#AEB4BD"  # light neutral for the (failing) paraboloid ROM -- deliberately understated


def build_damage_cmap(dcap):
    """Damage colormap = tints of the cut-pattern orange #F58025 (matches hinge_overview): light at
    D=0, deepening to full #F58025 at the top of the shown range. The range is the material's own --
    for PET that is a few percent of fracture, so the wash stays pale and the colorbar carries the
    point."""
    nodes = [(0.00, "#FEEFDD"), (0.35, "#FBD09E"), (0.62, "#F8AC5E"),
             (0.84, "#F69038"), (1.00, "#F58025")]
    return LinearSegmentedColormap.from_list("dmg_orange", nodes)




def surrogate_grid(ckpt, w_min, w_max, n_w=56, alpha_deg=90.0, n_theta=180, theta_max_rad=0.785):
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


def build_uz_field(data_npz, theta_deg_grid, w_grid, thickness, kin_tol=0.5):
    """Out-of-plane buckle amplitude ``uz/t`` as a SMOOTH, MONOTONE field on the (w_lig, theta) grid.

    The surrogate has no ``uz`` head; we read it from the FEA database (near-pure-rotation samples,
    ``|a|, |s| < kin_tol`` mm -- the PET campaign samples rays from the origin and contains no
    EXACTLY pure-rotation row, so this has to be a tolerance, not an equality). To get clean contours
    even where data is sparse, we take a robust BINNED MEDIAN on a coarse regular grid, then fill gaps
    ROW-WISE along theta and COLUMN-WISE along w by 1-D interpolation (flat extrapolation at the ends --
    no 2-D nearest-neighbour plateaus that produce contour loops), lift to the plot grid with a regular
    interpolator, and lightly Gaussian-smooth. Returns (n_w, n_theta).
    """
    from scipy.interpolate import RegularGridInterpolator
    from scipy.ndimage import gaussian_filter
    d = np.load(data_npz)
    a, s, th, wl, uz = d["a"], d["s"], d["theta_deg"], d["w_lig"], d["uz_max"] / thickness
    m = (np.abs(a) < kin_tol) & (np.abs(s) < kin_tol)
    th, wl, uz = th[m], wl[m], uz[m]

    te = np.linspace(0, theta_deg_grid.max(), 20)
    we = np.linspace(w_grid.min(), w_grid.max(), 14)      # the shown width band, not a fixed range
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


def free_path_jobs(data_npz):
    """Job ids driven with FREE translational DOF, read from the campaign manifest beside the npz.

    The PET campaign mixes two route families and the merged dataset carries no column for it: a
    prescribed ``origin_ray`` job drives ``(a, s)`` along a fixed ray, while a ``free_dofs`` job lets
    the solver choose them. Because ``W`` is a path functional for an elastoplastic hinge, the two
    families disagree on the same endpoint, and a checkpoint trained on one mispredicts the other.

    A manifest job with ``eta_a == eta_s == 0`` is a free-route job, NOT a pure-rotation one: with
    free DOF the etas are ignored and recorded as zero. The tell is in the rows -- a prescribed zero
    would hold ``a`` at machine zero, whereas these drift to a median |a| of 1.5 mm.
    """
    path = os.path.splitext(data_npz)[0] + ".json"
    if not os.path.exists(path):
        return set()
    import json
    with open(path) as fh:
        man = json.load(fh)
    jobs = [x for v in man.values() if isinstance(v, list) and v and isinstance(v[0], dict) for x in v]
    return {x["job_id"] for x in jobs
            if "eta_a" in x and x["eta_a"] == 0.0 and x.get("eta_s") == 0.0}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="printed width / authored width. Type is authored at target/scale so it PRINTS at the ladder size. A plate shown on a slide at 0.75 of its authored width takes its 8pt labels to 6pt unless this is set")
    ap.add_argument("--ckpt", default="data/surrogates/hinge_surrogate_pet_v3.pkl")
    ap.add_argument("--data", default="data/fea/hinge_dataset_pet_merged.npz")
    ap.add_argument("--quantity", choices=["energy", "moment"], default="energy")
    ap.add_argument("--out", default=None)
    ap.add_argument("--alpha-fill", type=float, default=0.78, help="region fill opacity (lower = lighter)")
    ap.add_argument("--d-show", type=float, default=None,
                    help="cap the damage landscape at this D; default = the slice's own max (PET never "
                         "approaches D=1, so there is nothing to mask)")
    ap.add_argument("--paths", type=float, nargs="+", default=[25.0, 5.0],
                    help="w_lig [mm] of the load paths to trace (plain lines)")
    ap.add_argument("--theta-max", type=float, default=45.0,
                    help="rotation ceiling [deg]; 45 is the database-coverage ceiling for deployment")
    ap.add_argument("--thickness", type=float, default=0.5, help="sheet gauge [mm] -- sets uz/t")
    ap.add_argument("--kin-tol", type=float, default=0.25,
                    help="|a|,|s| tolerance [mm] for the FEA OVERLAY. The drawn surface is exactly "
                         "a=s=0, and W is sensitive to small translations: at 0.25 mm the overlaid "
                         "points sit 3.9%% (median) off the slice, at 0.5 mm 4.9%%, at 1.0 mm 6.2%%")
    ap.add_argument("--uz-kin-tol", type=float, default=0.5,
                    help="|a|,|s| tolerance [mm] for the uz/t field. Looser than --kin-tol on purpose: "
                         "this one is a BINNED MEDIAN, which wants samples per bin and is far less "
                         "sensitive to small translations than W is")
    ap.add_argument("--snap-theta", type=float, nargs="+", default=[22.5, 45.0],
                    help="rotations [deg] at which the hinge renders are taken (story dots)")
    ap.add_argument("--rom-w", type=float, default=25.0, help="width the paraboloid ROM is calibrated to")
    ap.add_argument("--wlabel-frac", type=float, nargs=2, default=[0.36, 0.68],
                    help="where along theta to seat the (thin, wide) w_lig labels, as a fraction of "
                         "theta_max -- pick gaps the hinge insets do not occupy")
    ap.add_argument("--headroom", type=float, default=1.06,
                    help="y-axis top as a multiple of the tallest curve; raise it to open space above "
                         "the upper load path for the hinge insets")
    ap.add_argument("--hinge-prefix", default="data/outputs/hinge_render_pet_w25",
                    help="prefix for the wide-path hinge render PNGs (<prefix>_th<deg>.png)")
    ap.add_argument("--compare", action="store_true",
                    help="also place the thin lower-path hinge renders, for comparison")
    ap.add_argument("--hinge-prefix2", default="data/outputs/hinge_render_pet_w5",
                    help="prefix for the thin lower-path hinge renders in --compare mode")
    ap.add_argument("--stress-scale", action="store_true",
                    help="add the shared teal von Mises scale for stress-coloured hinge insets")
    ap.add_argument("--vmin", type=float, default=0.0, help="von Mises scale min [MPa] (matches insets)")
    ap.add_argument("--vmax", type=float, default=60.0, help="von Mises scale max [MPa] (matches insets)")
    ap.add_argument("--fea-points", action="store_true",
                    help="overlay the raw CalculiX training points (same slice: pure rotation, alpha~90)")
    ap.add_argument("--fea-alpha-tol", type=float, default=20.0,
                    help="alpha half-window [deg] for the FEA overlay (wider = more but off-slice points)")
    ap.add_argument("--fea-inside-only", action=argparse.BooleanOptionalAction, default=True,
                    help="draw only the FEA points that fall INSIDE the drawn fan. This is an "
                         "outcome-based clip, not a physical criterion -- the points it removes sit "
                         "mostly at theta ~1 deg, where the fan is pinched to near-zero width so a "
                         "tiny absolute deviation reads as 'outside'. The count dropped is printed.")
    ap.add_argument("--dpi", type=int, default=300, help="output resolution")
    ap.add_argument("--route", choices=["origin_ray", "mark", "all"], default="origin_ray",
                    help="which route family to overlay. 'origin_ray' (default) shows only the family "
                         "the surface models; 'mark' also draws the free-route rows as a distinct "
                         "series; 'all' pools them, which is what made W look multi-valued")
    args = ap.parse_args()
    out = args.out or f"data/outputs/hinge_rotation_{args.quantity}{'_compare' if args.compare else ''}.png"

    # bracket the fan EXACTLY by the two load paths -> the region envelope is the paths themselves and
    # the D/uz contours terminate cleanly at the story dots (no tail hooking past into a wider grid).
    theta, w_ligs, W, M, D = surrogate_grid(args.ckpt, min(args.paths), max(args.paths),
                                            n_w=180, n_theta=360,
                                            theta_max_rad=np.radians(args.theta_max))
    uz = build_uz_field(args.data, theta, w_ligs, args.thickness, kin_tol=args.uz_kin_tol)
    Y = W if args.quantity == "energy" else M
    ylabel = (r"stored energy  $\mathcal{W}$  [N$\cdot$mm]" if args.quantity == "energy"
              else r"restoring moment  $M=\mathrm{d}W/\mathrm{d}\theta$  [N$\cdot$mm]")

    def curve(w):                                              # nearest gridded trajectory for a width
        return Y[int(np.argmin(np.abs(w_ligs - w)))]

    def dcurve(w):                                             # ductile damage along that trajectory
        return D[int(np.argmin(np.abs(w_ligs - w)))]

    apply_charter(**paper_rc(args.scale))
    fig, ax = plt.subplots(figsize=figsize_in(FIG_W_BP, FIG_H_BP))
    # PET tops out around D=0.03 on this slice, so by default nothing is capped: the landscape is
    # drawn over its own full damage range and the colorbar's SMALLNESS is the message.
    d_show = args.d_show if args.d_show is not None else float(D.max())
    dcap = d_show
    cmap = build_damage_cmap(dcap)
    dnorm = Normalize(0.0, dcap)
    X = np.tile(theta, (len(w_ligs), 1))
    # rescale axes to the shown (D<=d_show) region: y to its tallest point (times --headroom), x just
    # past where the whole width-band exceeds d_show. With PET's auto d_show nothing is excluded and
    # this reduces to the full grid; it still matters if a caller caps the landscape explicitly.
    ymax = args.headroom * float(np.nanmax(np.where(D <= d_show, Y, np.nan)))
    minD = D.min(axis=0)
    xhi = min(float(theta[-1]),
              float(np.interp(d_show, minD, theta)) + 1.2 if minD.max() >= d_show else float(theta[-1]))

    # 1. damage as the coloured landscape (Princeton orange). Levels stop AT d_show with NO `extend`, so
    #    contourf leaves the D>d_show corner unfilled bounded by the smooth marching-squares d_show
    #    contour -- a clean curved edge, unlike the ragged cell-drop a NaN mask produces.
    damage_field = ax.contourf(X, Y, D, levels=np.linspace(0.0, dcap, 61), cmap=cmap, norm=dnorm,
                               alpha=args.alpha_fill, zorder=2, antialiased=True)
    # 61 stacked translucent bands are the only heavy thing on the plate. Flattening JUST them to
    # one image at save resolution is what lets the figure ship as a vector PDF: the axes, the
    # iso-lines and every label stay real curves and real text. A ContourSet ignores
    # `set_rasterized`; the zorder threshold is the mechanism that actually works on one, and the
    # field is the only artist drawn below it.
    ax.set_rasterization_zorder(3)

    # 1b. raw CalculiX training points behind the surrogate surface -- a band of the 6-D dataset around
    #     this figure's slice (pure rotation a=s=0, alpha within +-tol of 90 deg, w_lig in the shown
    #     band). The surface is the alpha=90 reference; the points' spread about it is the alpha spread.
    fea_handle, free_handle, n_free = None, None, 0
    if args.fea_points:
        fd = np.load(args.data)
        yq = fd["W"] if args.quantity == "energy" else fd["M_theta"]
        m = ((np.abs(fd["a"]) < args.kin_tol) & (np.abs(fd["s"]) < args.kin_tol)
             & (np.abs(fd["alpha_deg"] - 90.0) <= args.fea_alpha_tol)
             & (fd["w_lig"] >= min(args.paths) - 1e-9) & (fd["w_lig"] <= max(args.paths) + 1e-9)
             & (yq <= ymax))
        # split by route family. The kinematic tolerance alone does NOT separate them: a free-route
        # job starts near a=s=0 and only drifts later, so its EARLY, low-theta rows slip through and
        # land above the fan -- stiffer than this surface, because they took a different (compressive)
        # route to the same rotation. Pooling the two families is what makes W look multi-valued.
        free = np.isin(fd["job_id"], list(free_path_jobs(args.data))) if "job_id" in fd.files \
            else np.zeros(len(yq), bool)
        keep = m & (~free if args.route in ("origin_ray", "mark") else np.ones_like(m))
        n_clip = 0
        if args.fea_inside_only:
            # clip to the band the fan actually spans. W is monotone in w_lig, so the two drawn load
            # paths ARE the envelope. Outcome-based, so it is counted and reported, never silent.
            hi_env = np.interp(fd["theta_deg"], theta, curve(max(args.paths)))
            lo_env = np.interp(fd["theta_deg"], theta, curve(min(args.paths)))
            inside = (yq <= hi_env) & (yq >= lo_env)
            n_clip = int((keep & ~inside).sum())
            keep = keep & inside
        fea_handle = ax.scatter(fd["theta_deg"][keep], yq[keep], s=dA(3.5), c="#787878", alpha=0.16,
                                linewidths=0.0, zorder=4, rasterized=True)
        print(f"FEA overlay: {int(keep.sum())} points drawn"
              + (f", {n_clip} clipped as outside the fan" if n_clip else ""))
        if args.route == "mark":
            mf = m & free
            n_free = int(mf.sum())
            free_handle = ax.scatter(fd["theta_deg"][mf], yq[mf], s=dA(13), facecolors="none",
                                     edgecolors=RED, linewidths=dw(0.7), alpha=0.75, zorder=7)

    # 2. iso-lines: out-of-plane buckle altitude uz/t (smooth monotone landscape). Light, and stopped
    #    at the shown D=d_show boundary so they end with the coloured fill, not over blank paper.
    uz_valid = np.where(D > d_show, np.nan, uz)
    lo_uz, hi_uz = float(np.nanmin(uz_valid)), float(np.nanmax(uz_valid))
    uz_levels = [L for L in (1, 2, 5, 10, 15, 20, 30, 40, 50, 60, 80) if lo_uz < L < hi_uz]
    # the protagonist of this figure -- drawn firmly, not as a faint annotation
    cB = ax.contour(X, Y, uz_valid, levels=uz_levels, colors=SLATE, linewidths=dw(1.0), alpha=0.85,
                    zorder=5)

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
                 for p in (next((contour_pt(uz_valid, L, tt) for tt in np.linspace(xhi - 5.0, 12, 48)
                                 if contour_pt(uz_valid, L, tt)), None),) if p]
    uz_lbls = ax.clabel(cB, manual=uz_manual, fontsize=FS_ANNOT, colors=SLATE,
                        fmt=lambda v: f"$u_z/t$={v:g}")

    # NB: there are no damage iso-lines here. The steel version of this figure drew D=0.5 and the
    # D=1 fracture front and seated its snapshots on them; PET reaches D=0.03 on this slice, so those
    # contours simply do not exist. The snapshots are anchored on ROTATION instead (--snap-theta).

    # white halo -> every contour label stays legible over the coloured landscape
    for t in list(uz_lbls):
        t.set_path_effects([pe.withStroke(linewidth=dw(1.8), foreground="white")])

    # 3. load paths: plain thin lines at fixed w_lig, truncated where they leave the shown D<=d_show band
    for w in sorted(args.paths, reverse=True):
        yv, dv = curve(w), dcurve(w)
        keep = dv <= d_show
        ax.plot(theta[keep], yv[keep], color=INK, lw=dw(1.4), zorder=8, solid_capstyle="round")

    # 3b. story dots at the rotations where the hinge visuals are taken. Both paths carry dots at the
    #     SAME theta, so the wide/thin renders are read at matched rotation -- which is the whole
    #     comparison: same fold angle, very different out-of-plane escape.
    for w in sorted(args.paths, reverse=True):
        yv, dv = curve(w), dcurve(w)
        for tx in args.snap_theta:
            if tx > theta[-1] or float(np.interp(tx, theta, dv)) > d_show:
                continue
            ax.plot([tx], [float(np.interp(tx, theta, yv))], "o", color=INK, mec="white",
                    mew=dw(1.3), ms=dw(7.5), zorder=11)

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
    ax.plot(theta, rom, ls=(0, (5, 2)), color=ROM_GREY, lw=dw(1.6), zorder=3)
    ip = int(np.argmin(np.abs(rom - 0.62 * ymax)))            # a point partway up the ROM
    ax.annotate(r"$\frac{1}{2}\,k\,\theta^2$", (theta[ip], rom[ip]), fontsize=FS_BASE,
                color=ROM_GREY,
                ha="right", va="center", xytext=(-8, 0), textcoords="offset points", zorder=3)

    # 5. hinge renders in cards, leader-lined to the load-path dots at D=0.5 and D=1. Each panel is
    #    sized to the render's cropped content so the leader + D label sit right on the hinge.
    fig_asp = fig.get_figwidth() / fig.get_figheight()
    bw, pad = 0.17, 0.008

    def place_hinge(path_w, prefix, tx, lbl, gap=0.05, dx_nudge=0.0, below=False):
        """Seat the render just off its story dot (auto axes-fraction from the dot, so it survives any
        x/y rescale). ``below`` seats it under the dot instead of over it -- used for the LOWER load
        path, so both paths' renders sit OUTSIDE the fan and their leaders point inward, rather than
        being dropped on top of the uz/t contours. Flips back if the chosen side runs off the axes."""
        yv = curve(path_w)
        dx, dy = tx, float(np.interp(tx, theta, yv))
        dax, day = dx / xhi, dy / ymax                    # dot in axes fraction (xlo = ylo = 0)
        img_path = f"{prefix}_{lbl}.png"
        im, bh = None, 0.14
        if os.path.exists(img_path):
            im = plt.imread(img_path)                     # crop transparent padding -> tight to hinge
            ys, xs = np.where(im[:, :, 3] > 0.02)
            H, Wd = im.shape[:2]                           # headroom on top so the buckle peak clears
            im = im[max(ys.min() - 14, 0):min(ys.max() + 3, H),
                    max(xs.min() - 4, 0):min(xs.max() + 4, Wd)]
            bh = bw * fig_asp * im.shape[0] / im.shape[1]
        if below:
            # sit under the dot, but never flip back over it: near theta=0 the lower path runs too low
            # to clear a render, so the panel is dropped to the axis floor and the leader just gets
            # longer. Both lower renders then share one baseline and read as a filmstrip.
            by = max(day - gap - bh, 0.012)
        else:
            by = day + gap
            if by + bh > 0.97:                            # near the top -> drop it below its dot instead
                by = day - gap - bh
        bx = min(max(dax - bw / 2 + dx_nudge, 0.006), 0.994 - bw)
        if im is not None:                                # no card: hinge reads on the warm landscape
            inset = ax.inset_axes([bx, by, bw, bh]); inset.imshow(im); inset.axis("off")
            inset.patch.set_alpha(0.0); inset.set_zorder(12)
        else:
            ax.add_patch(Rectangle((bx, by), bw, bh, transform=ax.transAxes, facecolor="white",
                         edgecolor=GREY, ls=(0, (4, 3)), lw=dw(1.0), zorder=12))
        leader_y = (by + bh + pad) if (by + bh / 2) < day else (by - pad)
        ax.add_artist(ConnectionPatch((dx, dy), (bx + bw / 2, leader_y), coordsA=ax.transData,
                      coordsB=ax.transAxes, color=GREY, lw=dw(0.8), zorder=10))

    w_hi, w_lo = max(args.paths), min(args.paths)
    for tx in args.snap_theta:
        place_hinge(w_hi, args.hinge_prefix, tx, f"th{tx:g}")
        if args.compare:                                   # thin lower path, for the matched-theta pair
            place_hinge(w_lo, args.hinge_prefix2, tx, f"th{tx:g}", below=True)

    ax.set_xlabel(r"hinge rotation  $\vartheta$  [deg]")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, xhi)
    ax.set_ylim(0, ymax)

    # Layout: reserve a right margin and give the scales their own well-separated axes. Both bars are
    # labelled consistently -- ticks and title on the RIGHT, all text in INK -- so they read as a set.
    # The margins are POINTS converted to fractions, not fractions: a fraction is a share of the
    # canvas, and the type it has to clear is not. As fractions tuned on the old 763 bp plate they
    # left 24 bp for a y-label plus its ticks and cut the x-label off the bottom of the page.
    fx = lambda bp: bp / FIG_W_BP
    fy = lambda bp: bp / FIG_H_BP
    # Right-hand furniture, measured from the right edge inwards: the damage bar's "0.0200" ticks
    # and rotated title, then the von Mises bar's, then the plot.
    dmg_x = 1.0 - fx(58.0 if args.stress_scale else 52.0)
    vm_x = dmg_x - fx(62.0)
    fig.subplots_adjust(left=fx(40.0), right=(vm_x if args.stress_scale else dmg_x) - fx(10.0),
                        top=1.0 - fy(7.0), bottom=fy(30.0))
    cb_y, cb_h, cb_w = fy(30.0), 1.0 - fy(44.0), fx(6.5)

    def style_cbar(cb, label):
        cb.set_label(label, fontsize=FS_BASE, color=INK)
        cb.ax.tick_params(labelsize=FS_TICK, colors=INK)      # default side is right for both -> uniform
        cb.outline.set_edgecolor("#D3D6DB")

    if args.stress_scale:
        cax_vm = fig.add_axes([vm_x, cb_y, cb_w, cb_h])
        smv = ScalarMappable(norm=Normalize(args.vmin, args.vmax), cmap=STRESS_CMAP); smv.set_array([])
        style_cbar(fig.colorbar(smv, cax=cax_vm), r"von Mises stress  $\sigma_\mathrm{vM}$  [MPa]")

    cax_d = fig.add_axes([dmg_x, cb_y, cb_w, cb_h])
    sm = ScalarMappable(norm=dnorm, cmap=cmap); sm.set_array([])
    cb = fig.colorbar(sm, cax=cax_d)
    # the bar's RANGE is the point -- fracture sits at D=1, far above anything this slice reaches.
    # Left to the caption rather than annotated on the scale.
    style_cbar(cb, r"ductile damage  $D$")

    leg_h = [plt.Line2D([], [], color=INK, lw=dw(1.4)),
             plt.Line2D([], [], color=SLATE, lw=dw(0.8)),
             plt.Line2D([], [], color=ROM_GREY, lw=dw(1.6), ls=(0, (5, 2)))]
    leg_l = ["load path (fixed $w_\\mathrm{lig}$)", r"out-of-plane buckle  $u_z/t$",
             r"quadratic model  $\frac{1}{2}k\vartheta^2$"]
    if fea_handle is not None:
        tol = args.fea_alpha_tol
        leg_h.append(plt.Line2D([], [], marker="o", markerfacecolor="#707070", markeredgecolor="none",
                                markersize=dw(3.5), alpha=0.55, linestyle="none"))
        fam = "" if args.route == "all" else ", prescribed path"
        leg_l.append(rf"finite-element data  ($\alpha \in [{90 - tol:g},\,{90 + tol:g}]^\circ${fam})")
    if free_handle is not None:
        leg_h.append(plt.Line2D([], [], marker="o", markerfacecolor="none", markeredgecolor=RED,
                                markersize=dw(4.5), linestyle="none"))
        leg_l.append(rf"free-path finite-element ({n_free}) -- not modelled here")
    # Tightened on every axis. At printed size the legend is real 6.6 pt type rather than the 4.1 pt
    # it used to reduce to, so its default padding costs plot area the fan needs.
    leg = ax.legend(leg_h, leg_l, loc="upper left", fontsize=FS_ANNOT, handlelength=1.5,
                    handletextpad=0.5, labelspacing=0.35, borderpad=0.4, borderaxespad=0.4)
    leg.set_zorder(30)                                         # keep the ROM (and all lines) behind it
    leg.get_frame().set_edgecolor("#D3D6DB")

    # ligament-width labels, rotated PARALLEL to each load path at the label point (needs the final
    # data->display transform, so it happens after the manual layout is fixed)
    fig.canvas.draw()
    chip = dict(facecolor="white", alpha=0.82, edgecolor="none", boxstyle="round,pad=0.15")
    for w in sorted(args.paths, reverse=True):
        yv = curve(w)
        # seat each width label in a theta gap between the insets (which straddle --snap-theta)
        tl = (args.wlabel_frac[0] if w == min(args.paths) else args.wlabel_frac[1]) * theta[-1]
        i = int(np.clip(np.searchsorted(theta, tl), 1, len(theta) - 1))
        (x0, y0), (x1, y1) = (ax.transData.transform((theta[j], yv[j])) for j in (i - 1, i))
        ang = np.degrees(np.arctan2(y1 - y0, x1 - x0))
        ax.text(tl, float(np.interp(tl, theta, yv)), f"$w_\\mathrm{{lig}}={w:g}$ mm",
                rotation=ang, rotation_mode="anchor", ha="center", va="bottom",
                fontsize=FS_ANNOT, color=INK, bbox=chip, zorder=9)

    pdf, lo, hi = save_paper_figure(fig, out, FIG_W_BP, FIG_H_BP, dpi=args.dpi)
    print(f"quantity={args.quantity}  paths={args.paths}  uz/t max={np.nanmax(uz):.2f}  "
          f"Dmax={D.max():.2f}  -> {pdf} ({FIG_W_BP:.1f} x {FIG_H_BP:.1f} bp, "
          f"type {lo:.1f}-{hi:.1f} pt)")


if __name__ == "__main__":
    main()
