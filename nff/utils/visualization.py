import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch, Circle, Rectangle
from matplotlib.ticker import FuncFormatter
import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from nff.config.targets import get_target_points


def plot_cut_pattern(coords, T, cols, ax=None, filepath=None, hinge_margin=0.06,
                     paper_color="#F58025", cut_color="#1A1A1A", lw=2.4, title=None):
    """Render the true kirigami cut pattern of the flat sheet.

    Each interior cut C_{i,j} is the long slit that runs from one hinge to its
    opposite hinge across the void — i.e. between the collinear neighbour vertices
    the LES links (horizontal: x_{i-1,j} -> x'_{i+1,j}; vertical: x'_{i,j-1} ->
    x_{i,j+1}), with the cut's own vertices x_{i,j}, x'_{i,j} lying in between. The
    slit is drawn as long as possible, retracted by ``hinge_margin`` at interior
    ends (leaving the small uncut hinge ligaments) and run all the way to the sheet
    border when an endpoint is a boundary vertex.

    Args:
        coords: (P, 2) cut endpoint positions, P = 2 * rows * cols.
                Endpoint s of cut (i, j) is index 2*(i*cols + j) + s.
        T: (rows, cols) topology matrix (sign = interior/boundary, |.| = h/v).
        cols: number of cut-grid columns (T.shape[1]).
        hinge_margin: uncut ligament length left at each interior cut end.
    """
    # Imported here (not at module level) so that importing this module for the
    # standard pipeline plots does not require shapely — only the closed-state
    # cut-pattern figure needs it.
    from shapely.geometry import MultiPoint
    coords = np.asarray(coords)
    rows = T.shape[0]

    def pid(i, j, s):
        return 2 * (i * cols + j) + s

    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(9, 9), facecolor="white")

    # Solid sheet = convex hull of the boundary-cut positions.
    bpos = [coords[pid(i, j, 0)] for i in range(rows) for j in range(cols) if T[i, j] < 0]
    hull = MultiPoint(bpos).convex_hull
    hx, hy = hull.exterior.xy
    ax.fill(hx, hy, facecolor=paper_color, edgecolor=cut_color, lw=2.8, zorder=1)

    # Every cut: long slit from hinge to opposite hinge across the void. Boundary
    # cuts (T < 0) run to the sheet edge — their out-of-grid collinear neighbour is
    # replaced by the cut's own pinned border vertex. (Omitting boundary cuts, as an
    # earlier version did, hid the slits that terminate at the hinges one step in
    # from the border and made those real hinges look spurious.)
    def _in_grid(a, b):
        return 0 <= a < rows and 0 <= b < cols
    for i in range(rows):
        for j in range(cols):
            if T[i, j] == 0:
                continue
            if abs(int(T[i, j])) == 1:                  # horizontal cut
                (ai, aj, as_), (bi, bj, bs) = (i - 1, j, 0), (i + 1, j, 1)
            else:                                       # vertical cut
                (ai, aj, as_), (bi, bj, bs) = (i, j - 1, 1), (i, j + 1, 0)
            a_in, b_in = _in_grid(ai, aj), _in_grid(bi, bj)
            A = coords[pid(ai, aj, as_)] if a_in else coords[pid(i, j, 0)]
            B = coords[pid(bi, bj, bs)] if b_in else coords[pid(i, j, 0)]
            d = B - A
            L = float(np.linalg.norm(d))
            if L < 1e-9:
                continue
            u = d / L
            # Retract only at an end that lands on an interior hinge (leave the
            # ligament); run to the border at a boundary/out-of-grid end.
            p0 = A + (hinge_margin if (a_in and T[ai, aj] > 0) else 0.0) * u
            p1 = B - (hinge_margin if (b_in and T[bi, bj] > 0) else 0.0) * u
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=cut_color, lw=lw,
                    solid_capstyle="round", zorder=2)

    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title)
    if own and filepath:
        plt.savefig(filepath, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved cut pattern to {filepath}")


def _plot_shapely(ax, geom, **kw):
    """Fill a Shapely (Multi)Polygon (with holes) on ``ax`` as matplotlib PathPatches."""
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch
    def _ring(r):
        c = list(r.coords)
        return c, [Path.MOVETO] + [Path.LINETO] * (len(c) - 2) + [Path.CLOSEPOLY]
    polys = geom.geoms if geom.geom_type == "MultiPolygon" else ([geom] if geom.geom_type == "Polygon" else [])
    for p in polys:
        v, cds = _ring(p.exterior)
        for hole in p.interiors:
            hv, hc = _ring(hole)
            v = v + hv; cds = cds + hc
        ax.add_patch(PathPatch(Path(v, cds), **kw))


def render_precise_cut_pattern(geom, filepath=None, title=None, paper="#F58025", ink="#1A1A1A",
                               accent="#2A9D8F"):
    """Render the PRECISE laser-cuttable cut pattern (kerf slots + fillets + ligaments) built by
    ``nff.fabrication.cut_pattern.build_cut_geometry`` (mm), with a dimensioned zoom on one hinge.

    The full sheet shows the actual cut-out part (slits are true kerf-width gaps, hinges are the
    real ``w_lig`` ligament bridges); the inset shows a single hinge's fillet / ligament / kerf.
    """
    fig, ax = plt.subplots(figsize=(12, 8.5), facecolor="white")
    _plot_shapely(ax, geom["sheet"], facecolor=paper, edgecolor="none", zorder=1)
    _plot_shapely(ax, geom["cuts"], facecolor=ink, edgecolor=ink, lw=0, zorder=2)  # removed material
    x0, y0, x1, y1 = geom["sheet"].bounds
    mx = 0.04 * (x1 - x0)
    ax.set_xlim(x0 - mx, x1 + mx); ax.set_ylim(y0 - mx, y1 + mx)
    ax.set_aspect("equal"); ax.axis("off")

    hinges = geom.get("hinges", [])
    if hinges:
        cen = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
        tip, hdir = min(hinges, key=lambda h: np.linalg.norm(np.asarray(h[0]) - cen))
        tip = np.asarray(tip, dtype=float)
        hdir = np.asarray(hdir, dtype=float); hdir /= (np.linalg.norm(hdir) + 1e-12)
        perp = np.array([-hdir[1], hdir[0]])               # left of the cut
        wl, rho, wc = geom["w_lig"], geom["rho"], geom["w_c"]
        d = 2.4 * wl                                        # zoom window ~2 ligaments
        axins = ax.inset_axes([0.67, 0.03, 0.32, 0.36])    # lower-right, clear of the title
        _plot_shapely(axins, geom["sheet"], facecolor=paper, edgecolor="none")
        _plot_shapely(axins, geom["cuts"], facecolor=ink, edgecolor=ink, lw=0)
        axins.set_xlim(tip[0] - d, tip[0] + d); axins.set_ylim(tip[1] - d, tip[1] + d)
        axins.set_aspect("equal")
        axins.set_xticks([]); axins.set_yticks([])
        for s in axins.spines.values():
            s.set_edgecolor(accent); s.set_linewidth(1.4)
        ax.indicate_inset_zoom(axins, edgecolor=accent, lw=1.2, alpha=0.8)

        # ── dimension arrows (mm data coords) ──
        dim = dict(arrowstyle="<|-|>", color=accent, lw=1.3, mutation_scale=8)
        off = 1.4 * wc + 0.22 * wl                          # ligament dim line, offset to the side
        a0, a1 = tip + off * perp, (tip - wl * hdir) + off * perp
        axins.annotate("", a1, a0, arrowprops=dim)
        axins.plot(*zip(tip, tip + off * perp), color=accent, lw=0.6)          # witness lines
        axins.plot(*zip(tip - wl * hdir, tip - wl * hdir + off * perp), color=accent, lw=0.6)
        axins.text(*(0.5 * (a0 + a1) + 0.22 * wl * perp), r"$w_{lig}$", color=ink,
                   fontsize=9, ha="center", va="bottom")

        axins.text(0.5, -0.09, f"hinge detail  ·  $w_{{lig}}$={wl:.1f} mm   kerf={wc:.2f} mm",
                   transform=axins.transAxes, ha="center", va="top", fontsize=8.5, color=ink)

    # scale bar (nice round length ~1/5 of the sheet width, in mm)
    span = x1 - x0
    raw = span / 5.0
    mag = 10.0 ** np.floor(np.log10(raw))
    nice = float(min([1, 2, 5, 10], key=lambda k: abs(k * mag - raw)) * mag)
    sx, sy = x0 + 0.02 * span, y0 - 0.03 * span
    ax.plot([sx, sx + nice], [sy, sy], color=ink, lw=3.5, solid_capstyle="butt",
            zorder=6, clip_on=False)
    ax.text(sx + nice / 2, sy - 0.012 * span, f"{nice:.0f} mm", color=ink, fontsize=11,
            ha="center", va="top")

    if title:
        ax.set_title(title, fontsize=13, color=ink)
    if filepath:
        plt.savefig(filepath, dpi=320, bbox_inches="tight"); plt.close(fig)
        print(f"  Saved precise cut pattern to {filepath}")
    return fig


# ── Closed-state hinge strips (condensation RVE geometry) ──────────────────────
# A hinge is the residual uncut steel left where the main cut is retracted. The
# meshable strip is the two tiles it joins, closed into a `w_lig`-wide ligament
# neck, clipped to a Saint-Venant window and stopped at the void (main-cut notch).

_HINGE_TEAL = "#2A9D8F"


def _closed_cut_lines(T, cols):
    """Every cut's slit endpoints (flat cut-vertex indices), INCLUDING boundary cuts.

    Boundary cuts run to the sheet edge: the out-of-grid collinear neighbour is
    replaced by the cut's own pinned border vertex. These are the cuts that
    terminate at the hinges sitting one step in from the border — omitting them (as
    the old interior-only version did) hid those cuts and made real hinges look
    spurious. Degenerate corner cuts are skipped.
    """
    rows = T.shape[0]
    def pid(i, j, s):
        return 2 * (i * cols + j) + s
    def in_grid(i, j):
        return 0 <= i < rows and 0 <= j < cols
    lines = {}
    for i in range(rows):
        for j in range(cols):
            if T[i, j] == 0:
                continue
            if abs(int(T[i, j])) == 1:
                (ai, aj, as_), (bi, bj, bs) = (i - 1, j, 0), (i + 1, j, 1)
            else:
                (ai, aj, as_), (bi, bj, bs) = (i, j - 1, 1), (i, j + 1, 0)
            pa = pid(ai, aj, as_) if in_grid(ai, aj) else pid(i, j, 0)
            pb = pid(bi, bj, bs) if in_grid(bi, bj) else pid(i, j, 0)
            if pa != pb:
                lines[(i, j)] = (pa, pb)
    return lines


def _draw_cuts_as_gaps(ax, coords, T, cols, retract, kerf, zorder=3, fillet=0.0):
    """Draw cuts as WHITE removed-material slits (kerf-wide), retracted at interior ends.

    ``fillet`` > 0 adds a white disc of that radius at each RETRACTED (interior) cut tip — the
    stress-relief fillet — drawn above the strips so it reads on the macro sheet.
    """
    coords = np.asarray(coords)
    hk = kerf / 2.0
    for (i, j), (pa, pb) in _closed_cut_lines(T, cols).items():
        A, B = coords[pa], coords[pb]
        u = (B - A) / (np.linalg.norm(B - A) + 1e-12)
        n = np.array([-u[1], u[0]])
        # Retract + fillet a cut end iff it leaves a ligament (a HINGE) -- the same rule as
        # cut_pattern.build_cut_geometry: interior neighbour, OR interior cut terminating at a
        # boundary cut. An off-grid neighbour (endpoint pinned to this cut's own border vertex) is
        # the sheet EDGE -> no ligament. Without the boundary case, boundary hinges lost their fillet.
        t_self = int(T[i, j])

        def _hinge_at(p):
            ni, nj = divmod(p // 2, cols)
            if (ni, nj) == (i, j):                 # own border vertex -> sheet edge, not a hinge
                return False
            t_nbr = int(T[ni, nj])
            return t_nbr > 0 or (t_nbr < 0 and t_self > 0)

        a_ret, b_ret = _hinge_at(pa), _hinge_at(pb)
        p0 = A + (retract if a_ret else 0.0) * u
        p1 = B - (retract if b_ret else 0.0) * u
        ax.add_patch(Polygon([p0 + hk*n, p1 + hk*n, p1 - hk*n, p0 - hk*n],
                             closed=True, facecolor="white", edgecolor="none", zorder=zorder))
        if fillet > 0:
            for p, ret in ((p0, a_ret), (p1, b_ret)):
                if ret:
                    ax.add_patch(Circle(p, fillet, facecolor="white", edgecolor="none", zorder=zorder + 3))


def hinge_strip_polygon(coords, hstruct, hid, w_lig, kerf, r_win):
    """Shapely polygon of one meshable hinge strip (condensation RVE domain).

    The two tiles the hinge joins, plus the LIGAMENT — the material left where the
    main cut is retracted by ``w_lig`` (the residual wedge between the two closing
    edges within ``w_lig`` of the pivot). The main-cut void beyond ``w_lig`` is then
    carved out with constant kerf. This is the exact per-hinge geometry: the ligament
    is not added arbitrarily, it is simply the region the (retracted) cut leaves.

    Args:
        coords: (P, 2) flat cut vertices (solve_cut_vertices_jax output).
        hstruct: build_hinge_descriptor_structure output.
        hid: hinge index.
        w_lig, kerf, r_win: ligament gap, kerf, Saint-Venant radius [same units as coords].

    Returns:
        A shapely (Multi)Polygon.
    """
    from shapely.geometry import Polygon as SPoly, Point, LineString
    from shapely.ops import unary_union

    coords = np.asarray(coords)
    corner_pid = hstruct["deploy_struct"]["corner_pid"]
    f1, f2 = hstruct["face_pairs"][hid]
    P = coords[hstruct["pivot_pid"][hid]]

    # ligament = residual wedge between the two closing edges within w_lig of the pivot
    e1 = coords[hstruct["edge_pid"][hid][0]]
    e2 = coords[hstruct["edge_pid"][hid][1]]
    u1 = (e1 - P) / (np.linalg.norm(e1 - P) + 1e-12)
    u2 = (e2 - P) / (np.linalg.norm(e2 - P) + 1e-12)
    wedge = SPoly([P, P + w_lig * u1, P + w_lig * u2])
    dom = unary_union([SPoly(coords[corner_pid[f1]]), SPoly(coords[corner_pid[f2]]), wedge])
    dom = dom.intersection(Point(P).buffer(r_win, resolution=64))

    # carve the main-cut void: retracted by w_lig at the pivot, constant kerf
    a, b = coords[hstruct["main_end_pid"][hid][0]], coords[hstruct["main_end_pid"][hid][1]]
    if np.linalg.norm(b - P) < np.linalg.norm(a - P):
        a, b = b, a
    um = (b - a) / (np.linalg.norm(b - a) + 1e-12)
    void = LineString([a + w_lig * um, b]).buffer(kerf / 2.0, cap_style=2)
    # carve the secondary-cut void too so the strip stops at it (does not cover it);
    # it does not pass between the two tiles, so this trims the edge, never severs.
    sa, sb = coords[hstruct["sec_end_pid"][hid][0]], coords[hstruct["sec_end_pid"][hid][1]]
    sec_void = LineString([sa, sb]).buffer(kerf / 2.0, cap_style=2)
    return dom.difference(void).difference(sec_void)


def _fill_shapely(ax, geom, **kw):
    for pg in ([geom] if geom.geom_type == "Polygon" else list(geom.geoms)):
        if pg.is_empty:
            continue
        xs, ys = pg.exterior.xy
        ax.fill(xs, ys, **kw)


def plot_hinge_strips(coords, hstruct, w_lig, kerf, r_win_factor=2.0,
                      ax=None, filepath=None, sheet_color="#F58025",
                      strip_color=_HINGE_TEAL, mm_per_unit=None, scale_mm=200.0, title=None,
                      fillet=0.0, fs=10.0):
    """Full sheet with every hinge strip (the meshable RVE domains) coloured.

    No sheet outline; cuts drawn as white removed-material gaps; each hinge's strip
    filled in ``strip_color``. If ``mm_per_unit`` is given, a scale bar is drawn.

    Args:
        coords: (P, 2) flat cut vertices.
        hstruct: build_hinge_descriptor_structure output.
        w_lig, kerf: ligament gap and kerf [same units as coords].
        r_win_factor: Saint-Venant window radius = r_win_factor * w_lig.
        mm_per_unit, scale_mm: draw a ``scale_mm`` scale bar when mm_per_unit is set.
    """
    from scipy.spatial import ConvexHull
    coords = np.asarray(coords)
    T, cols = hstruct["deploy_struct"]["T"], hstruct["deploy_struct"]["cols"]
    r_win = r_win_factor * w_lig

    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(9, 9), facecolor="white")

    def pid(i, j, s):
        return 2 * (i * cols + j) + s
    b = np.array([coords[pid(i, j, 0)] for i in range(T.shape[0]) for j in range(cols) if T[i, j] < 0])
    hull = b[ConvexHull(b).vertices]
    ax.fill(hull[:, 0], hull[:, 1], facecolor=sheet_color, edgecolor="none", zorder=1)
    # Cuts first (white voids), then the strips ON TOP: each strip carves its own cut
    # void (retracted by w_lig, constant kerf) and keeps the ligament wedge, so the
    # long slits still show elsewhere.
    _draw_cuts_as_gaps(ax, coords, T, cols, retract=w_lig, kerf=kerf, zorder=3, fillet=fillet)
    for hid in range(len(hstruct["pivot_pid"])):
        _fill_shapely(ax, hinge_strip_polygon(coords, hstruct, hid, w_lig, kerf, r_win),
                      facecolor=strip_color, edgecolor="none", alpha=1.0, zorder=5)

    if mm_per_unit:
        L = scale_mm / mm_per_unit
        x0 = hull[:, 0].min(); y0 = hull[:, 1].min() - 0.06 * np.ptp(hull[:, 1])
        ax.plot([x0, x0 + L], [y0, y0], color="#1A1A1A", lw=3, zorder=20)
        ax.annotate(f"{scale_mm:.0f} mm", (x0 + L/2, y0 - 0.015 * np.ptp(hull[:, 1])),
                    ha="center", va="top", fontsize=fs, zorder=20)     # number BELOW the bar (match detail panel)

    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title)
    if own and filepath:
        plt.savefig(filepath, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved hinge strips to {filepath}")


def plot_hinge_detail(coords, hstruct, hid, w_lig, kerf, descriptors=None,
                      r_win_factor=2.6, mm_per_unit=None, ax=None, filepath=None,
                      strip_color=_HINGE_TEAL, title=None):
    """Zoom on one hinge: meshable strip + descriptor parameters + scale bar.

    Args:
        coords: (P, 2) flat cut vertices.
        hstruct: build_hinge_descriptor_structure output.
        hid: hinge index (should be interior).
        w_lig, kerf: ligament width and kerf [same units as coords].
        descriptors: optional compute_hinge_descriptors output, for alpha / cut-length labels.
        mm_per_unit: if given, lengths are annotated in millimetres and a scale bar drawn.
    """
    coords = np.asarray(coords)
    r_win = r_win_factor * w_lig
    P = coords[hstruct["pivot_pid"][hid]]
    T, cols = hstruct["deploy_struct"]["T"], hstruct["deploy_struct"]["cols"]
    to_mm = (lambda v: v * mm_per_unit) if mm_per_unit else (lambda v: v)
    unit = "mm" if mm_per_unit else "u"

    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(8, 8), facecolor="white")

    _draw_cuts_as_gaps(ax, coords, T, cols, retract=w_lig, kerf=kerf)
    _fill_shapely(ax, hinge_strip_polygon(coords, hstruct, hid, w_lig, kerf, r_win),
                  facecolor=strip_color, edgecolor="#1A1A1A", lw=1.4, alpha=0.92, zorder=5)

    def cut_dir(pids):
        a, b = coords[pids[0]], coords[pids[1]]
        far = b if np.linalg.norm(b - P) > np.linalg.norm(a - P) else a
        return (far - P) / (np.linalg.norm(far - P) + 1e-12)
    um = cut_dir(np.asarray(hstruct["main_end_pid"][hid]))
    us = cut_dir(np.asarray(hstruct["sec_end_pid"][hid]))

    if descriptors is not None:
        import matplotlib.patches as _mp
        a_m = np.degrees(np.arctan2(um[1], um[0])); a_s = np.degrees(np.arctan2(us[1], us[0]))
        alpha_deg = np.degrees(float(np.asarray(descriptors["alpha"])[hid]))
        Lm = float(np.asarray(descriptors["L_main"])[hid])
        Ls = float(np.asarray(descriptors["descriptor"])[hid, 0]) * Lm
        ax.add_patch(_mp.Arc(P, 0.11 * r_win / 0.15, 0.11 * r_win / 0.15,
                             theta1=min(a_m, a_s), theta2=max(a_m, a_s), color="#1A1A1A", lw=1.5))
        ax.annotate(rf"$\alpha$={alpha_deg:.0f}°", P + 0.65 * r_win * (um + us) / (np.linalg.norm(um + us) + 1e-9),
                    fontsize=12)
        ax.annotate(f"main cut  L={to_mm(Lm):.0f} {unit}", P + 1.25 * r_win * um, fontsize=9,
                    color="#6C757D", ha="center",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
        ax.annotate(f"secondary cut  L={to_mm(Ls):.0f} {unit}", P + 1.25 * r_win * us, fontsize=9,
                    color="#6C757D", ha="center",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))

    nrm = np.array([-um[1], um[0]])
    ax.annotate("", P + w_lig / 2 * nrm, P - w_lig / 2 * nrm,
                arrowprops=dict(arrowstyle="<->", color="#D62828", lw=1.8))
    ax.annotate(rf"$w_{{lig}}$={to_mm(w_lig):.0f} {unit}", P + 0.4 * w_lig * nrm + 0.2 * w_lig * um,
                fontsize=10, color="#D62828", weight="bold")

    if mm_per_unit:
        L = 20.0 / mm_per_unit
        x0, y0 = P[0] - r_win, P[1] - r_win - 0.02
        ax.plot([x0, x0 + L], [y0, y0], color="#1A1A1A", lw=3, zorder=20)
        ax.annotate("20 mm", (x0 + L / 2, y0 - 0.01), ha="center", va="top", fontsize=9, zorder=20)

    z = r_win + 0.05
    ax.set_xlim(P[0] - z, P[0] + z); ax.set_ylim(P[1] - z, P[1] + z)
    ax.set_aspect("equal"); ax.axis("off")
    if title:
        ax.set_title(title)
    if own and filepath:
        plt.savefig(filepath, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved hinge detail to {filepath}")


def _rve_real_mesh(ax, region, *, w_lig, lc_min, lc_max, color="#728079", lw=0.4, zorder=3):
    """The REAL gmsh RVE surface mesh drawn over ``region`` — same domain + ligament-Ball refinement
    the CalculiX pipeline uses (``nff.rve.ccx_solver._build_mesh``), reduced to the 2D face triangles.

    Uses the actual mesher so the drawn density/grading match the physics mesh (fine at the
    ligament, coarse away). Falls back silently if gmsh is unavailable.
    """
    from matplotlib.tri import Triangulation
    try:
        import gmsh
    except Exception:
        return
    for poly in getattr(region, "geoms", [region]):
        if poly.is_empty or poly.area < 1e-9:
            continue
        coords = list(poly.exterior.coords)[:-1]
        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add("rve2d")
            pts = [gmsh.model.geo.addPoint(x, y, 0.0, lc_max) for (x, y) in coords]
            n = len(pts)
            loop = [gmsh.model.geo.addLine(pts[i], pts[(i + 1) % n]) for i in range(n)]
            gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop(loop)])
            gmsh.model.geo.synchronize()
            gmsh.model.mesh.field.add("Ball", 1)                  # refine only the ligament strip
            for k, v in (("XCenter", 0.0), ("YCenter", -0.5 * w_lig), ("ZCenter", 0.0),
                         ("Radius", 0.75 * w_lig), ("Thickness", 0.75 * w_lig),
                         ("VIn", lc_min), ("VOut", lc_max)):
                gmsh.model.mesh.field.setNumber(1, k, v)
            gmsh.model.mesh.field.setAsBackgroundMesh(1)
            for opt in ("MeshSizeExtendFromBoundary", "MeshSizeFromPoints", "MeshSizeFromCurvature"):
                gmsh.option.setNumber("Mesh." + opt, 0)
            gmsh.model.mesh.generate(2)
            tags, xyz, _ = gmsh.model.mesh.getNodes()
            xyz = np.asarray(xyz).reshape(-1, 3)
            remap = {int(t): i for i, t in enumerate(tags)}
            tris = np.empty((0, 3), int)
            for et, en in zip(*gmsh.model.mesh.getElements(dim=2)[:3:2]):
                if int(et) == 2:                                 # 3-node triangle
                    tris = np.asarray(en, int).reshape(-1, 3)
            if len(tris):
                conn = np.vectorize(remap.get)(tris)
                ax.triplot(Triangulation(xyz[:, 0], xyz[:, 1], conn),
                           color=color, lw=lw, zorder=zorder)
        finally:
            gmsh.finalize()


def plot_hinge_dimensions(w_lig=10.0, w_c=3.0, alpha_deg=62.0, fillet=2.0, thickness=1.0,
                          ax=None, filepath=None, steel_color="#D9DCE1",
                          lig_color=_HINGE_TEAL, title=None, fs=12.0, scale_bar=True):
    """Dimensioned engineering drawing of one hinge (parametric schematic).

    The hinge is the residual ligament left where the main cut stops ``w_lig`` short
    of the secondary cut. All lengths share one consistent unit; only symbols are
    drawn (no numeric values), so the same figure documents any design point.

    Args:
        w_lig: ligament gap (main-cut tip -> secondary cut).
        w_c: cut width (kerf).
        fs: base annotation size in points; every label scales from it.
        scale_bar: draw the millimetre scale bar.
        alpha_deg: angle between the two cuts [deg].
        fillet: cut-tip fillet radius (rho).
        thickness: sheet thickness (drawn in a side view).
    """
    from shapely.geometry import LineString as _LS, box as _sbox, Point as _Pt
    from matplotlib.patches import Arc as _Arc
    ink, dim = "#1A1A1A", "#444"
    wl, wc = w_lig, w_c
    a = np.radians(alpha_deg)
    us = np.array([1.0, 0.0])                          # secondary cut direction (horizontal)
    um = np.array([np.cos(-a), np.sin(-a)])            # main cut direction (down, tilted by alpha)
    tip = np.array([0.0, -wl])                         # main-cut tip, w_lig short of secondary (y=0)
    W, H = 2.6*wl, 2.6*wl

    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(11, 10), facecolor="white")
    ax.set_aspect("equal"); ax.axis("off")

    # adjacent faces fill the whole drawing window (grey context) — a zoom on the
    # local geometry the green RVE rests on: two faces below the secondary cut (either
    # side of the main cut) and the opposite face above it. Cuts are thin white voids.
    Xw, Yw = 3.7*wl, 3.3*wl
    # main cut void, kerf-wide, with a rho-radius fillet at its tip
    main_void = (_LS([tip, tip + (Yw + 2*wl)*um]).buffer(wc/2, cap_style=2)
                 .union(_Pt(tip[0], tip[1]).buffer(fillet, resolution=48)))
    below = _sbox(-Xw, -Yw, Xw, 0.0).difference(main_void)
    grey = "#D3D6DB"                                    # all context faces one colour
    _fill_shapely(ax, below, facecolor=grey, edgecolor="none", zorder=0)
    ax.add_patch(Polygon([(-Xw, wc), (Xw, wc), (Xw, Yw), (-Xw, Yw)], facecolor=grey, edgecolor="none", zorder=0))
    # These were 0.75*fs and printed at 5.25 pt. A fraction of the caller's size is not a size:
    # it drifts below the legibility floor as soon as the caller sizes for the page.
    ax.annotate("adjacent face", (-2.6*wl, -2.7*wl), fontsize=fs, color=dim, ha="center")
    ax.annotate("adjacent face", (2.6*wl, -2.7*wl), fontsize=fs, color=dim, ha="center")
    # voids are just voids — no outlines
    ax.add_patch(Polygon([(-Xw, 0), (Xw, 0), (Xw, wc), (-Xw, wc)], facecolor="white", edgecolor="none", zorder=1))
    _fill_shapely(ax, main_void.intersection(_sbox(-Xw, -Yw, Xw, wc)), facecolor="white", edgecolor="none", zorder=1)

    # green meshed region (rounded Saint-Venant disk RVE), resting on the faces
    Rw = 2.4 * wl
    green = _Pt(0.0, 0.0).buffer(Rw, resolution=80).intersection(_sbox(-Rw, -Rw, Rw, 0.0))
    green = green.difference(main_void)
    _fill_shapely(ax, green, facecolor=lig_color, edgecolor="none", alpha=1.0, zorder=2)
    # the REAL pipeline mesh (gmsh, ligament-refined) over the RVE — real grading = physical intuition
    _rve_real_mesh(ax, green, w_lig=wl, lc_min=max(0.5 * fillet, 0.09 * wl), lc_max=0.42 * wl,
                   color="#8CC0AF")     # mesh visible on the teal, a touch darker

    def _dim_linear(p1, p2, off, label, side=1, pad=1.6):
        p1, p2 = np.array(p1, float), np.array(p2, float)
        u = (p2-p1)/np.linalg.norm(p2-p1); n = np.array([-u[1], u[0]])*side
        a1, a2 = p1+n*off, p2+n*off
        ax.plot([p1[0], a1[0]+n[0]*1.2], [p1[1], a1[1]+n[1]*1.2], color=dim, lw=0.7, zorder=6)
        ax.plot([p2[0], a2[0]+n[0]*1.2], [p2[1], a2[1]+n[1]*1.2], color=dim, lw=0.7, zorder=6)
        ax.add_patch(FancyArrowPatch(a1, a2, arrowstyle="<|-|>", mutation_scale=10,
                                     color=dim, lw=1.0, zorder=6, shrinkA=0, shrinkB=0))
        ax.annotate(label, (a1+a2)/2 + n*pad, color=ink, fontsize=fs, ha="center", va="center", zorder=7)

    def _leader(p_from, p_to, label):
        ax.add_patch(FancyArrowPatch(p_to, p_from, arrowstyle="-|>", mutation_scale=11,
                                     color=dim, lw=1.0, zorder=6))
        ax.annotate(label, p_to, color=ink, fontsize=fs, ha="left", va="center", zorder=7)

    # w_lig: the ligament gap (main-cut tip -> secondary cut), on the right
    _dim_linear((wc/2+0.3*wl, -wl), (wc/2+0.3*wl, 0.0), off=0.9*wl, label=r"$w_{\mathrm{lig}}$", side=1)
    # w_c (kerf): the cut is too thin for an in-place dimension arrow -> leader callout pointing AT it
    _cut_pt = tip + 2.05*wl*um + (wc/2)*np.array([-um[1], um[0]])       # a point on the main-cut edge
    _lab_pt = _cut_pt + np.array([0.62*wl, -0.10*wl])                   # label just off the cut
    ax.add_patch(FancyArrowPatch(_lab_pt, _cut_pt, arrowstyle="-|>", mutation_scale=11,
                                 color=dim, lw=1.0, zorder=7, shrinkA=3, shrinkB=0))
    ax.annotate(r"$w_c$", _lab_pt + np.array([0.05*wl, 0.0]), color=ink, fontsize=fs,
                ha="left", va="center", zorder=7)
    # The window RADIUS, from the disc centre out to the rim. It was the full width, labelled
    # w_mesh: a second symbol for an object the manuscript dimensions by its radius r_win
    # everywhere else, and "mesh" reads it as a discretisation when it is a domain. The
    # extension line _dim_linear drops at the first point marks the centre, so the half-span
    # reads as a radius rather than an arbitrary chord.
    _dim_linear((0.0, 0.0), (Rw, 0.0), off=1.3*wl, label=r"$r_{\mathrm{win}}$", side=1)
    # alpha = angle between the main cut and the secondary cut, drawn AT THE TOP of the main cut
    # (its tip) against a short reference line PARALLEL to the secondary cut. Placing it at the tip
    # makes it independent of the ligament w_lig (the uncut strip is a separate parameter).
    L_ref = 1.15 * wl
    ax.plot([tip[0], tip[0] + L_ref*us[0]], [tip[1], tip[1] + L_ref*us[1]],
            color=dim, lw=0.7, ls=(0, (4, 3)), zorder=6)            # parallel-to-secondary reference
    ax.plot([tip[0], tip[0] + L_ref*um[0]], [tip[1], tip[1] + L_ref*um[1]],
            color=dim, lw=0.7, ls=(0, (4, 3)), zorder=6)            # main-cut axis (into the material)
    am = np.degrees(np.arctan2(um[1], um[0]))                        # main-axis angle (negative, tilts down)
    ax.add_patch(_Arc(tip, 0.75*wl, 0.75*wl, theta1=am, theta2=0.0, color=ink, lw=1.0, zorder=8))
    amid = np.radians(am / 2.0)
    ax.annotate(r"$\alpha$", tip + 0.52*wl*np.array([np.cos(amid), np.sin(amid)]),
                color=ink, fontsize=fs, ha="center", va="center", zorder=9)
    # fillet radius rho — radius arrow from the tip-fillet centre; label kept CLOSE to the centre
    u_r = np.array([np.cos(np.radians(212)), np.sin(np.radians(212))])
    ax.annotate("", tip + fillet*u_r, tip, zorder=9,
                arrowprops=dict(arrowstyle="-|>", color=ink, lw=1.0, shrinkA=0, shrinkB=0))
    ax.annotate(r"$R=\rho$", tip + (fillet + 0.62*wl)*u_r, color=ink, fontsize=fs,
                ha="right", va="center", zorder=9)
    # cut-length labels — plain text, next to their voids (they mark the cuts, not dimensions)
    ax.annotate(r"$L_{\mathrm{main}}$", (wc/2 + 0.35*wl, -2.5*wl), color=dim, fontsize=0.92*fs, ha="left", va="center")
    ax.annotate(r"$L_{\mathrm{sec}}$", (Rw + 0.15*wl, wc/2), color=dim, fontsize=0.92*fs, ha="left", va="center")

    # scale bar (all lengths are in millimetres)
    if scale_bar:
        sb = 10.0
        xb, yb = -Xw, -Yw - 1.0*wl
        ax.plot([xb, xb + sb], [yb, yb], color=ink, lw=3, zorder=8)
        # text gap below the bar ~1.5% of panel height, matching the strips-panel scale bar
        ax.annotate(f"{sb:.0f} mm", (xb + sb/2, yb - 0.14*wl), ha="center", va="top",
                    fontsize=0.84*fs, zorder=8)

    _ylo = -Yw - (2.0 if scale_bar else 1.0)*wl      # reclaim the bar's strip when it is off
    ax.set_xlim(-Xw - 0.3*wl, Xw + 0.9*wl); ax.set_ylim(_ylo, Yw + 0.5*wl)
    if title:
        ax.set_title(title)
    if own and filepath:
        plt.savefig(filepath, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved hinge dimensions to {filepath}")


def write_deformed_into(tessellation, node_positions):
    """Return a copy of ``tessellation`` with vertices set to deformed positions.

    Args:
        tessellation: reference Tessellation.
        node_positions: (n_faces, max_nodes, 2) per-face deformed node positions
            (e.g. from ``deformed_vertices``).

    Returns:
        A copy of ``tessellation`` with its global vertices set to the deformed
        positions.
    """
    deformed = tessellation.copy()
    new_vertices = np.array(deformed.vertices, dtype=float)
    for f_id, face in enumerate(deformed.faces):
        for local, gv in enumerate(face.vertex_indices):
            new_vertices[gv] = node_positions[f_id, local]
    deformed.update_vertices(new_vertices)
    return deformed


# ── Boundary-condition glyphs ─────────────────────────────────────────────────────────────────
# One symbol, one function. Both the tessellation loading diagram and the plain-rectangle schematic
# draw from here, so the two can never disagree about what a support looks like.

SUPPORT_GREY = "#6C757D"
# Applied actions are drawn in INK, not in the charter's RED. Red is the loudest thing on the page
# and it was being spent on the least informative mark: the reader already knows where the load is
# from the support layout, and five rows of red arrows drowned the data panels they sit beside.
LOAD_INK = "#1A1A1A"
LOAD_RED = LOAD_INK          # retained name, kept in step so no caller diverges from the charter


def _draw_fixed_support(ax, *, base, lo, hi, axis, side, span, zorder=2, slider=False,
                        weight_scale=1.0):
    """Hatched wall clamping a run of the boundary.

    Args:
        base: coordinate along ``axis`` of the wall's inner face (the face touching the structure).
        lo, hi: extent of the clamped run along the perpendicular axis.
        axis: translation the clamp fixes -- 0 (side wall) or 1 (floor/ceiling).
        side: +1/-1, outward direction along ``axis``.
        span: sheet span, the length every glyph dimension is taken from.
        slider: the wall restrains only the direction NORMAL to it, leaving the structure free to
            slide along it -- drawn as a line of rollers between wall and structure. A one-DOF
            restraint drawn as a solid clamp claims the boundary is held in two directions when it
            is held in one, which for a sheet that must expand along that wall is the difference
            between a mechanism that opens and one that jams.
    """
    pad = 0.022 * span
    t = 0.030 * span
    # A clamp on a single tile of a 150-tile sheet has a near-zero extent, which would draw a stub
    # barely wider than the hatch itself. Give every wall at least a readable minimum.
    # ⚠ KEEP THIS TIGHT. At 0.08*span the wall was wider than the tile it clamps, so on a 5x5 sheet
    # it read as a support under SEVERAL tiles and the problem statement became ambiguous: which
    # tile is held is the whole difference between one boundary-value problem and another.
    if hi - lo < 0.045 * span:
        mid = 0.5 * (lo + hi)
        lo, hi = mid - 0.0225 * span, mid + 0.0225 * span

    if slider:
        r = 0.013 * span
        n = max(2, min(14, int((hi - lo) / (3.4 * r))))
        for c in np.linspace(lo + r, hi - r, n):
            xy = (base + side * r, c) if axis == 0 else (c, base + side * r)
            ax.add_patch(Circle(xy, r, facecolor="white", edgecolor=SUPPORT_GREY,
                                lw=1.3 * weight_scale, zorder=zorder + 2))
        base = base + side * 2 * r                       # the wall backs off behind the rollers
    if axis == 0:
        xy = (base, lo - pad) if side > 0 else (base - t, lo - pad)
        w, h = t, (hi - lo) + 2 * pad
    else:
        xy = (lo - pad, base) if side > 0 else (lo - pad, base - t)
        w, h = (hi - lo) + 2 * pad, t
    ax.add_patch(Rectangle(xy, w, h, facecolor=SUPPORT_GREY, edgecolor=SUPPORT_GREY,
                           hatch="////", lw=0, alpha=0.55, zorder=zorder))


def _draw_point_support(ax, *, anchor, axis, side, span, rollers=True, zorder=3,
                        weight_scale=1.0):
    """Point support: a triangle whose apex sits ON the structure, on a hatched ground.

    With ``rollers=True`` the triangle rests on two circles -- the standard ROLLER, holding the
    structure along ``axis`` and letting it slide along the ground bar. With ``rollers=False`` the
    triangle sits directly on the ground -- the standard PIN, holding both translations while
    leaving the rotation free.

    That distinction is load-bearing here rather than decorative. A pin drawn as a hatched wall
    claims the tile's rotation is held too, and on a sheet whose deployment IS a tile rotation that
    is the difference between a mechanism with one free DOF and one with none.

    Every dimension is taken from the sheet ``span``, never from the support tile. Sizing off the
    tile is what made the previous three-circle glyph collapse into overlapping blobs on a
    150-tile sheet, where one tile is a few percent of the span.

    Args:
        anchor: (x, y) point on the structure the apex touches.
        axis: translation the ground bar is normal to -- 0 (bar vertical) or 1 (bar horizontal).
        side: +1/-1, outward direction along ``axis``.
    """
    u = np.zeros(2); u[axis] = side                      # outward, normal to the ground bar
    v = np.array([-u[1], u[0]])                          # along the ground bar
    a = np.asarray(anchor, dtype=float)

    h = 0.055 * span                                     # triangle height
    half_b = 0.042 * span                                # triangle half-base
    r = 0.016 * span                                     # roller radius

    ax.add_patch(Polygon([a, a + u * h - v * half_b, a + u * h + v * half_b], closed=True,
                         facecolor="white", edgecolor=SUPPORT_GREY, lw=1.8 * weight_scale,
                         zorder=zorder + 1))
    if rollers:
        for s in (-0.5, 0.5):
            ax.add_patch(Circle(a + u * (h + r) + v * (s * half_b), r, facecolor="white",
                                edgecolor=SUPPORT_GREY, lw=1.6 * weight_scale,
                                zorder=zorder + 2))

    ground = a + u * (h + (2 * r if rollers else 0.0))   # inner face of the ground bar
    half_g = 1.35 * half_b
    _draw_fixed_support(ax, base=float(ground[axis]),
                        lo=float(a[1 - axis] - half_g), hi=float(a[1 - axis] + half_g),
                        axis=axis, side=side, span=span * 0.7, zorder=zorder,
                        weight_scale=weight_scale)


def _clamp_style(trans_dofs, clamped_dofs, n_faces):
    """How a clamp should be drawn: ``'built_in'``, ``'pin'``, ``'slider_wall'`` or ``'roller'``.

    A support symbol states WHICH degrees of freedom are held. Reading that off the config rather
    than always drawing the same wall is the only way the diagram and the run can agree.
    """
    if len(trans_dofs) == 1:
        return 'slider_wall' if n_faces > 1 else 'roller'
    if 2 in clamped_dofs:
        return 'built_in'                                # x, y AND rotation: a true fixed end
    return 'built_in' if n_faces > 1 else 'pin'          # x, y with rotation free


def _draw_force_arrow(ax, tail, head, *, lw=1.7, scale=13, zorder=31):
    """One applied-force arrow, in INK. Short and thin by default: see :data:`LOAD_INK`."""
    ax.annotate('', xy=tuple(head), xytext=tuple(tail),
                arrowprops=dict(arrowstyle='-|>', color=LOAD_INK, lw=lw, mutation_scale=scale,
                                shrinkA=0, shrinkB=0),
                zorder=zorder)


def _draw_moment_arc(ax, centre, radius, sign, *, lw=1.7, zorder=30, dot=True, weight_scale=1.0):
    """Couple applied at ``centre``; ``sign`` > 0 is counter-clockwise.

    ONE open circular arc carrying a single head, plus a dot at the point of application. The
    previous glyph plotted a 260 deg ring of points and bolted an arrowhead on with a second call,
    then drew the whole thing three times behind a bracket to suggest a distributed moment; at
    figure scale that read as a knot of circles rather than as a couple.
    """
    a0, a1 = (55.0, 340.0) if sign > 0 else (340.0, 55.0)
    arc = np.deg2rad(np.linspace(a0, a1, 160))
    xs = centre[0] + radius * np.cos(arc)
    ys = centre[1] + radius * np.sin(arc)
    ax.plot(xs, ys, color=LOAD_INK, lw=lw, zorder=zorder, solid_capstyle='round')

    # The head must sit on the TANGENT at the arc's end, not on a chord back to an earlier sample.
    # Taking the direction from a few samples back makes it a secant -- at this radius that is about
    # 14 degrees of arc -- and the triangle then lands visibly skewed off the line it terminates.
    end = arc[-1]
    tangent = np.array([-np.sin(end), np.cos(end)]) * np.sign(a1 - a0)
    tip = np.array([xs[-1], ys[-1]])
    ax.annotate('', xy=tuple(tip), xytext=tuple(tip - tangent * 0.18 * radius),
                arrowprops=dict(arrowstyle='-|>', color=LOAD_INK, lw=lw,
                                mutation_scale=11 * weight_scale, shrinkA=0, shrinkB=0),
                zorder=zorder + 1)
    if dot:
        ax.plot([centre[0]], [centre[1]], marker='o', ms=3.0 * weight_scale, color=LOAD_INK,
                zorder=zorder + 1, linestyle='none')


def _place_arrow(point, normal, f_hat, length, *, gap):
    """Where one force arrow goes, given the loaded tile and the boundary it sits on.

    ``point`` is the application point projected onto the nearest sheet edge and ``normal`` is that
    edge's OUTWARD normal. The arrow is always drawn clear of the material:

    * pulling away from the edge (``f . n > 0``): TAIL on the edge, head outward.
    * pushing into it (``f . n < 0``): HEAD on the edge, tail further out.
    * parallel to it (``f . n ~ 0``): offset outward, arrow centred on the application point.

    ``normal`` of ``None`` marks an INTERIOR load, which has no edge to sit against: the head
    terminates at the point itself.

    This rule exists because the two panels of the matrix figure disagreed without it. The schematic
    anchored a downward load at the sheet's BOTTOM edge whatever tile carried it, so a cantilever
    load applied to the top-right tile was drawn underneath the sheet, contradicting the load-path
    panel next to it, which anchored on the tile.
    """
    p = np.asarray(point, dtype=float)
    d = np.asarray(f_hat, dtype=float)
    if normal is None:
        return p - d * length, p
    n = np.asarray(normal, dtype=float)
    dot = float(np.dot(d, n))
    if dot > 1e-9:                                       # pull
        tail = p + n * gap
        return tail, tail + d * length
    if dot < -1e-9:                                      # push
        head = p + n * gap
        return head - d * length, head
    off = p + n * (gap + 0.5 * length)                   # parallel: sit beside the edge
    return off - d * length / 2.0, off + d * length / 2.0


def _expand_load_specs(load_specs):
    """One spec per face, so a distributed load draws an arrow on every tile it acts on."""
    specs = []
    for s in (load_specs or []):
        f = s['face']
        for fi in (f if isinstance(f, (list, tuple, np.ndarray)) else [f]):
            specs.append({**s, 'face': int(fi)})
    return specs


def _draw_loads(ax, specs, *, centre_of, span, arrow_len, bbox, edge_tol=None, weight_scale=1.0):
    """Draw every applied force and couple for ``specs``, by ONE placement rule.

    Both the plain-rectangle schematic and the tessellation diagram call this, so the two panels of
    a figure cannot state different boundary conditions for the same run. Each loaded tile is
    projected onto its NEAREST sheet edge and the arrow is placed by :func:`_place_arrow`; a tile
    further than ``edge_tol`` from every edge is treated as an INTERIOR load and the arrow
    terminates on the tile itself.

    Args:
        centre_of: face index -> (2,) application point.
        arrow_len: arrow length in data units.
        bbox: (x0, y0, x1, y1) of the sheet.
        edge_tol: distance within which a tile counts as sitting ON an edge. Defaults to one tile
            pitch, inferred from the sheet span.
    """
    x0, y0, x1, y1 = bbox
    tol = edge_tol if edge_tol is not None else 0.09 * span
    gap = 0.018 * span

    NORMALS = np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]])

    def _edge_index(p, f_hat):
        """Which sheet edge a load at ``p`` belongs to, or ``None`` if the tile is interior.

        Distance alone is not enough at a CORNER, where two edges are equidistant to within the
        tessellation's own irregularity. Measured on this sheet: the top-right tile sits 0.5134 from
        the right edge and 0.4662 from the top, so a 0.047 wobble in tile shape threw an axial pull
        onto the top edge and drew it above the sheet. The edge whose outward normal lines up with
        the force is therefore worth half a tolerance of distance, which settles every corner
        deterministically and in the direction the physics implies.
        """
        d = np.array([p[0] - x0, x1 - p[0], p[1] - y0, y1 - p[1]])
        k = int(np.argmin(d - 0.5 * tol * np.abs(NORMALS @ f_hat)))
        return None if d[k] > tol else k

    def edge_of(k, p):
        """(point projected onto edge ``k``, its outward normal); ``k`` of None means interior."""
        if k is None:
            return np.asarray(p, dtype=float), None
        q = np.array(p, dtype=float)
        n = NORMALS[k].copy()
        if k < 2:
            q[0] = x0 if k == 0 else x1
        else:
            q[1] = y0 if k == 2 else y1
        return q, n

    forces = [s for s in specs
              if int(s.get('dof', -1)) in (0, 1) and float(s.get('value', 0.0)) != 0.0]

    # Arrows drawn side by side must not be longer than the gap between the tiles carrying them, or
    # a distributed traction fuses into one stroke and stops reading as a set of discrete actions.
    # Measured off the loaded tiles, so it adapts to any grip.
    length = arrow_len
    if len(forces) > 1:
        pts = np.array([centre_of(s['face']) for s in forces], dtype=float)
        d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
        d[np.diag_indices_from(d)] = np.inf
        length = min(arrow_len, max(0.85 * float(d.min()), 0.35 * arrow_len))

    moment_faces = {int(s['face']) for s in specs
                    if int(s.get('dof', -1)) == 2 and float(s.get('value', 0.0)) != 0.0}
    m_radius = 0.46 * arrow_len

    # Loads sharing a direction and a sign are ONE action, so they take ONE edge -- decided by
    # majority vote over the tiles. Letting each tile pick its own edge put nine arrows off the
    # right edge and the tenth off the top, which reads as two different tractions.
    def _key(s):
        return int(s['dof']), float(np.sign(s['value']))

    groups = {}
    for s in forces:
        groups.setdefault(_key(s), []).append(s)

    for (dof, sgn), grp in groups.items():
        f_hat = np.zeros(2)
        f_hat[dof] = sgn
        votes = [_edge_index(np.asarray(centre_of(s['face']), dtype=float), f_hat) for s in grp]
        edge = max(set(votes), key=votes.count)
        for s in grp:
            p = np.asarray(centre_of(s['face']), dtype=float)
            q, n = edge_of(edge, p)
            # The arrow reaches the point of application even when the same tile carries a couple:
            # the force acts THERE, and stopping it short of the arc read as though it acted on the
            # arc instead. The arc is small enough that the two glyphs coexist.
            tail, head = _place_arrow(q, n, f_hat, length, gap=gap)
            _draw_force_arrow(ax, tail, head, lw=1.7 * weight_scale, scale=13 * weight_scale)
            if n is None:                   # interior: mark where the load actually acts
                ax.plot([p[0]], [p[1]], marker='o', ms=2.6 * weight_scale, color=LOAD_INK,
                        zorder=32, linestyle='none')

    # Couples (dof 2). Drawn AT the loaded tile: a moment acts on the tile, not on the boundary, and
    # the tile carrying one is exactly where the reader should look.
    for s in specs:
        if int(s.get('dof', -1)) != 2 or float(s.get('value', 0.0)) == 0.0:
            continue
        _draw_moment_arc(ax, np.asarray(centre_of(s['face']), dtype=float),
                         m_radius, float(s['value']), lw=1.7 * weight_scale,
                         weight_scale=weight_scale, dot=int(s['face']) not in
                         {int(t['face']) for t in forces})


def faces_as_tessellation(face_nodes):
    """Duck-typed tessellation whose faces carry their OWN corners.

    Enough for :func:`draw_supports_and_loads`, which reads only ``vertices`` and
    ``faces[i].vertex_indices``. Every face owns its corners because
    :func:`write_deformed_into` welds shared vertices, and in a deployed kirigami one shared vertex
    belongs to up to four panels that have pulled apart by a void width, so last-write-wins puts
    three of them in the wrong place.

    Args:
        face_nodes: (n_faces, n_corners, 2) panel corners.
    """
    from types import SimpleNamespace

    nodes = np.asarray(face_nodes, dtype=float)
    n_faces, n_corners, _ = nodes.shape
    return SimpleNamespace(
        vertices=nodes.reshape(-1, 2),
        faces=[SimpleNamespace(vertex_indices=np.arange(k * n_corners, (k + 1) * n_corners))
               for k in range(n_faces)])


def _support_axis(offset, extent, trans_dofs):
    """Which axis a support's wall takes its normal from, or ``None`` for a rotation-only clamp."""
    if len(trans_dofs) == 1:
        return trans_dofs[0]
    if trans_dofs:
        return int(np.argmax(np.abs(offset / np.maximum(extent, 1e-9))))
    return None


def draw_supports_and_loads(ax, tessellation, clamped_faces, load_specs, *,
                            clamped_dofs=None, y_roller_faces=None, x_roller_faces=None,
                            weight_scale=1.0):
    """Draw the boundary conditions of a run onto an existing Axes.

    Extracted from :func:`plot_loading_diagram` so the load-path figure draws its supports and loads
    with the SAME code rather than a second implementation. The placement rules here are hard-won --
    the wall takes its normal from which translation the clamp fixes, arrows sit outside the tile on
    the side the force points, and rollers are drawn because a diagram showing only the clamp
    misstates a two-support problem -- and none of that should be re-derived per figure.

    Returns ``(x0, y0, x1, y1, span)`` of the drawn geometry, for the caller's axis limits.
    """
    clamped = {int(f) for f in (clamped_faces or [])}
    y_roll = {int(f) for f in (y_roller_faces or [])}
    x_roll = {int(f) for f in (x_roller_faces or [])}

    allv = np.asarray(tessellation.vertices, dtype=float)
    x0, y0 = allv.min(axis=0)
    x1, y1 = allv.max(axis=0)
    span = float(max(x1 - x0, y1 - y0))
    centre_all = np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0])

    def fv(fi):
        return np.asarray(tessellation.vertices[tessellation.faces[int(fi)].vertex_indices], dtype=float)

    _draw_loads(ax, _expand_load_specs(load_specs),
                centre_of=lambda fi: fv(fi).mean(axis=0), span=span, arrow_len=0.125 * span,
                bbox=(x0, y0, x1, y1), weight_scale=weight_scale)

    # Fixed-support wall behind the clamped faces. Orientation comes from WHICH translation the
    # clamp fixes: fixing y is a floor/ceiling, fixing x is a side wall. When it fixes both, the
    # wall is put on whichever axis the clamped tiles sit furthest out along, since that is the
    # edge a physical clamp would grip.
    if clamped:
        cd = [int(d) for d in (clamped_dofs if clamped_dofs is not None else (0, 1, 2))]
        trans = [d for d in cd if d in (0, 1)]
        pts = np.concatenate([fv(i) for i in clamped], axis=0)
        axis = _support_axis(pts.mean(axis=0) - centre_all, np.array([x1 - x0, y1 - y0]), trans)
        if axis is not None:
            side = 1.0 if (pts.mean(axis=0) - centre_all)[axis] >= 0 else -1.0
            a = pts[:, axis].max() if side > 0 else pts[:, axis].min()
            perp = 1 - axis
            style = _clamp_style(trans, cd, len(clamped))
            if style in ('pin', 'roller'):
                anchor = np.empty(2)
                anchor[axis], anchor[perp] = a, pts[:, perp].mean()
                _draw_point_support(ax, anchor=anchor, axis=axis, side=side, span=span,
                                    rollers=style == 'roller', weight_scale=weight_scale)
            else:
                _draw_fixed_support(ax, base=a + side * 0.01 * span,
                                    lo=pts[:, perp].min(), hi=pts[:, perp].max(), axis=axis,
                                    side=side, span=span, slider=style == 'slider_wall',
                                    weight_scale=weight_scale)

    # Roller supports -- the standard pin-on-rollers symbol, so they read as distinct from the
    # clamp's hatched fixed wall while sharing its ground.
    for faces, axis in ((y_roll, 1), (x_roll, 0)):
        for fi in sorted(faces):
            p = fv(fi)
            side = 1.0 if (p.mean(axis=0) - centre_all)[axis] >= 0 else -1.0
            a = p[:, axis].max() if side > 0 else p[:, axis].min()
            anchor = np.empty(2)
            anchor[axis] = a
            anchor[1 - axis] = p[:, 1 - axis].mean()
            _draw_point_support(ax, anchor=anchor, axis=axis, side=side, span=span,
                                weight_scale=weight_scale)

    return x0, y0, x1, y1, span


def draw_loading_schematic(ax, bbox, face_xy, *, clamped_faces=(), clamped_dofs=None,
                           y_roller_faces=(), x_roller_faces=(), load_specs=(),
                           face_color="#FDF1E6", edge_color="#1A1A1A", edge_width=1.0,
                           weight_scale=1.0):
    """The boundary-value problem alone: a plain rectangle, its supports and its loads.

    The tessellation is deliberately absent. In a figure whose other columns already show the cut
    pattern twice, redrawing 150 tiles here says nothing about the PROBLEM being posed and makes the
    supports hard to find. The sheet becomes one rectangle and the glyphs carry the whole content.

    Supports and loads are attached to the rectangle's EDGE at the position of the tiles they act
    on, not at those tiles' centroids, so every glyph sits outside the structure and the diagram
    stays readable at any tile count.

    Args:
        bbox: (x0, y0, x1, y1) of the flat sheet.
        face_xy: (n_faces, 2) face centroids, used only to place glyphs along the edge.
        clamped_faces, clamped_dofs, y_roller_faces, x_roller_faces, load_specs: as in
            :func:`draw_supports_and_loads`.

    Returns:
        ``(x0, y0, x1, y1, span)`` of the drawn geometry.
    """
    x0, y0, x1, y1 = (float(v) for v in bbox)
    span = float(max(x1 - x0, y1 - y0))
    centre_all = np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0])
    lim = np.array([[x0, y0], [x1, y1]])
    face_xy = np.asarray(face_xy, dtype=float)

    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=face_color,
                           edgecolor=edge_color, lw=edge_width * weight_scale, zorder=10))

    def snap(fi, axis, side):
        """Face ``fi`` projected onto the rectangle edge normal to ``axis``."""
        p = np.array(face_xy[int(fi)], dtype=float)
        p[axis] = lim[1 if side > 0 else 0, axis]
        return p

    _draw_loads(ax, _expand_load_specs(load_specs),
                centre_of=lambda fi: face_xy[int(fi)], span=span, arrow_len=0.125 * span,
                bbox=(x0, y0, x1, y1), weight_scale=weight_scale)

    clamped = [int(f) for f in (clamped_faces or [])]
    if clamped:
        cd = [int(d) for d in (clamped_dofs if clamped_dofs is not None else (0, 1, 2))]
        trans = [d for d in cd if d in (0, 1)]
        c = face_xy[clamped]
        axis = _support_axis(c.mean(axis=0) - centre_all, np.array([x1 - x0, y1 - y0]), trans)
        if axis is not None:
            side = 1.0 if (c.mean(axis=0) - centre_all)[axis] >= 0 else -1.0
            perp = 1 - axis
            base = (x1 if side > 0 else x0) if axis == 0 else (y1 if side > 0 else y0)
            style = _clamp_style(trans, cd, len(clamped))
            if style in ('pin', 'roller'):
                anchor = np.empty(2)
                anchor[axis], anchor[perp] = base, c[:, perp].mean()
                _draw_point_support(ax, anchor=anchor, axis=axis, side=side, span=span,
                                    rollers=style == 'roller', weight_scale=weight_scale)
            else:
                _draw_fixed_support(ax, base=base, lo=c[:, perp].min(), hi=c[:, perp].max(),
                                    axis=axis, side=side, span=span,
                                    slider=style == 'slider_wall', weight_scale=weight_scale)

    for faces, axis in ((y_roller_faces or (), 1), (x_roller_faces or (), 0)):
        for fi in sorted(int(f) for f in faces):
            side = 1.0 if (face_xy[fi] - centre_all)[axis] >= 0 else -1.0
            _draw_point_support(ax, anchor=snap(fi, axis, side), axis=axis, side=side, span=span,
                                weight_scale=weight_scale)

    return x0, y0, x1, y1, span


def plot_loading_diagram(tessellation, clamped_faces, load_specs, filepath,
                         title="Loading", clamped_dofs=None,
                         y_roller_faces=None, x_roller_faces=None):
    """One clean schematic of the boundary conditions and applied loads.

    Clamped faces are greyed and backed by a hatched fixed-support wall; loads are single bold
    arrows drawn OUTSIDE the loaded tile, pointing the way the force pushes or pulls.

    Everything here is derived from the actual boundary conditions — nothing about the layout is
    assumed. Two things used to be hard-coded and were wrong for any edge but the left one:

    * The wall was always a VERTICAL bar at the leftmost clamped vertex, spanning the FULL height
      of the structure. A bottom-edge clamp was therefore drawn as a left-hand wall running past
      tiles it does not hold. The wall now takes its normal from the clamped translation DOFs
      (fixing y -> a floor, fixing x -> a side wall), sits on whichever side of the structure the
      clamped faces are actually on, and spans only those faces.
    * An upward load was drawn BELOW its tile (an arrow pushing up into it). For a pull on the top
      edge that puts the arrow in the middle of the sheet, reading as a load on the tiles it
      overlaps. Arrows are now placed on the side the force points toward, tail on the tile's outer
      edge and head away from it, so a pull looks like a pull and never overlaps another tile.

    Args:
        tessellation: the tessellation to draw (typically the deployed state).
        clamped_faces: face indices held by the clamp.
        load_specs: list of {face, dof, value} load dicts.
        filepath: output PNG path.
        title: figure title.
        clamped_dofs: DOF indices the clamp fixes (0=x, 1=y, 2=theta). ``None`` -> all three.
            Only the translational ones orient the wall.
        y_roller_faces: faces pinned in y ONLY (free to slide in x).
        x_roller_faces: faces pinned in x ONLY (free to slide in y).
            ⚠ Rollers MUST be drawn. A single-tile grip needs a roller on a second tile to stop the
            assembly swinging about the clamp, and a diagram that shows only the clamp claims the
            sheet is held at one point when it is held at two -- it misstates the boundary
            conditions of the run it illustrates. They are drawn as the standard roller symbol
            (support line + circles) so they read as distinct from the clamp's hatched fixed wall.
    """
    clamped = {int(f) for f in (clamped_faces or [])}
    y_roll = {int(f) for f in (y_roller_faces or [])}
    x_roll = {int(f) for f in (x_roller_faces or [])}
    rollers = y_roll | x_roll

    def _face_color(i):
        if i in clamped:
            return "#9AA3AB"                       # fully held: same grey as the fixed wall
        if i in rollers:
            return "#C7CED4"                       # partially held: lighter, visibly not a clamp
        return "#F7C59F"
    colors = [_face_color(i) for i in range(len(tessellation.faces))]
    fig, ax = plt.subplots(figsize=(11, 8.5), facecolor="white")
    plot_tessellation(tessellation, ax=ax, show_target=False, show_hinges=False,
                      show_face_indices=False, show_hinge_indices=False,
                      show_external_forces=False, color_faces=colors)

    x0, y0, x1, y1, span = draw_supports_and_loads(
        ax, tessellation, clamped_faces, load_specs, clamped_dofs=clamped_dofs,
        y_roller_faces=y_roller_faces, x_roller_faces=x_roller_faces)

    ax.set_xlim(x0 - 0.14 * span, x1 + 0.28 * span)
    ax.set_ylim(y0 - 0.16 * span, y1 + 0.22 * span)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=15, weight="bold")
    fig.savefig(filepath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved loading diagram to {filepath}")


def plot_area_change(tess_flat, tess_deployed, rel, filepath,
                     titles=("Trained design (closed)", "Deployed")):
    """Two-panel figure colouring each panel by its area change vs the initial design.

    Args:
        tess_flat, tess_deployed: trained-design tessellations (flat / deployed).
        rel: (n_faces,) relative area change (trained / initial - 1).
        filepath: output PNG path.
        titles: (flat, deployed) panel titles.
    """
    cmap = plt.get_cmap("coolwarm")          # blue = shrunk, red = enlarged
    norm = mcolors.TwoSlopeNorm(vmin=min(float(rel.min()), -1e-3), vcenter=0.0,
                                vmax=max(float(rel.max()), 1e-3))
    area_colors = [cmap(norm(r)) for r in rel]

    fig, axes = plt.subplots(1, 2, figsize=(15, 7.6), facecolor="white")
    for ax, t, ttl in ((axes[0], tess_flat, titles[0]), (axes[1], tess_deployed, titles[1])):
        plot_tessellation(t, ax=ax, show_target=False, show_hinges=False,
                          show_face_indices=False, show_hinge_indices=False, color_faces=area_colors)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(ttl)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("panel area change vs initial design", fontsize=12)
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.0%}"))
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def animate_closed_evolution(tessellation, frames, add_target, filepath,
                             title="Opening process", fps=6, fail_line=1.0):
    """Animate the closed-state design morphing over training epochs (flat | deployed).

    Args:
        tessellation: reference Tessellation (copied internally for each panel).
        frames: list of (epoch, flat_verts, deployed_verts, boundary_cloud[, hinge_xy, D]), where
            *_verts are (n_global_vertices, 2) global vertex arrays and the optional
            (hinge_xy (n_hinges,2), D (n_hinges,)) overlay the per-hinge ductile damage as dots.
        add_target: callable (ax, boundary_cloud) -> None drawing the target on the deployed panel.
        filepath: output GIF path.
        fail_line: D value drawn as the colour pivot (model fracture-initiation, default 1.0).
    """
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.cm import ScalarMappable

    allf = np.concatenate([f[1] for f in frames])
    alld = np.concatenate([f[2] for f in frames])
    fb = (allf[:, 0].min() - .5, allf[:, 0].max() + .5, allf[:, 1].min() - .5, allf[:, 1].max() + .5)
    db = (alld[:, 0].min() - .5, alld[:, 0].max() + .5, alld[:, 1].min() - .5, alld[:, 1].max() + .5)

    # damage colour scale: green (safe) -> yellow at the fracture line -> red (past failure), shared
    has_dmg = len(frames[0]) >= 6 and frames[0][5] is not None
    norm = cmap = None
    if has_dmg:
        dmax = max(float(np.max(f[5])) for f in frames if f[5] is not None)
        norm = TwoSlopeNorm(vmin=0.0, vcenter=float(fail_line), vmax=max(2.0 * fail_line, dmax))
        cmap = plt.get_cmap("RdYlGn_r")

    tfl, tdp = tessellation.copy(), tessellation.copy()
    fig, (axf, axd) = plt.subplots(1, 2, figsize=(15, 7.6), facecolor="white")
    fig.suptitle(title, fontsize=16, weight="bold")
    if has_dmg:
        cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axd, fraction=0.046, pad=0.02)
        cb.set_label(f"hinge damage D  (={fail_line:g} model fracture-initiation)")

    def draw(k):
        ep, flat, dep, cloud = frames[k][:4]
        hinge_xy = frames[k][4] if len(frames[k]) >= 6 else None
        D = frames[k][5] if len(frames[k]) >= 6 else None
        axf.clear(); axd.clear()
        tfl.update_vertices(flat)
        plot_tessellation(tfl, ax=axf, show_target=False, show_hinges=False,
                          show_face_indices=False, show_hinge_indices=False, color_faces="#F58025")
        axf.set_xlim(fb[0], fb[1]); axf.set_ylim(fb[2], fb[3]); axf.set_aspect("equal"); axf.axis("off")
        axf.set_title(f"Closed design — epoch {ep}")
        tdp.update_vertices(dep)
        plot_tessellation(tdp, ax=axd, show_target=False, show_hinges=False,
                          show_face_indices=False, show_hinge_indices=False, color_faces="#F58025")
        add_target(axd, cloud)
        if hinge_xy is not None and D is not None:
            axd.scatter(hinge_xy[:, 0], hinge_xy[:, 1], c=D, cmap=cmap, norm=norm,
                        s=42, edgecolors="black", linewidths=0.5, zorder=5)
        axd.set_xlim(db[0], db[1]); axd.set_ylim(db[2], db[3]); axd.set_aspect("equal"); axd.axis("off")
        n_over = int(np.sum(D >= fail_line)) if D is not None else 0
        axd.set_title(f"Deploy — epoch {ep}" + (f"   (D≥{fail_line:g}: {n_over}/{len(D)})" if D is not None else ""))

    animation.FuncAnimation(fig, draw, frames=len(frames), blit=False).save(
        filepath, writer="pillow", fps=fps)
    plt.close(fig)
    print(f"  Saved training-evolution animation to {filepath}")


def plot_tessellation(tessellation, ax=None,
                      show_faces=True, 
                      show_hinges=True, 
                      show_vertices=False, 
                      show_face_indices=True,
                      show_hinge_indices=True,
                      show_external_forces=False,
                      show_kinematic_blocks=False,
                      show_target=True,
                      target_params=None,
                      show_border_edges=False,
                      title=None,
                      color_faces='#F58025',
                      mapping_fn=None,
                      map_params=None,
                      original_vertices=None,
                      edge_color='#1A1A1A',
                      edge_width=1.2,
                      face_alpha=0.85,
                      target_color="#009900",
                      target_width=2.5,
                      target_style='--'):
    """
    Plots the tessellation with configurable visibility for topological elements.

    The trailing style arguments default to this module's long-standing appearance, so callers that
    do not pass them get byte-identical output. A composite figure overrides them to drop the heavy
    black tile outline and the thick green target, which read as noise once several panels sit side
    by side.
    """
    if ax is None:
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(10, 10), facecolor='#FFFFFF')
        ax.set_facecolor('#FFFFFF')

    # 0. Transformation Grid (if mapping_fn is provided)
    if mapping_fn is not None and original_vertices is not None:
        import jax.numpy as jnp
        import jax
        
        min_xy = np.min(original_vertices, axis=0)
        max_xy = np.max(original_vertices, axis=0)
        
        num_lines = 25
        pts_per_line = 100
        x_starts = np.linspace(min_xy[0], max_xy[0], num_lines)
        y_starts = np.linspace(min_xy[1], max_xy[1], num_lines)
        
        y_range = jnp.linspace(min_xy[1], max_xy[1], pts_per_line)
        x_range = jnp.linspace(min_xy[0], max_xy[0], pts_per_line)
        
        f_point = lambda p: mapping_fn(p, map_params)
        f_vmap = jax.vmap(f_point)
        
        # Vertical lines
        for x in x_starts:
            pts = jnp.column_stack([jnp.full(pts_per_line, x), y_range])
            mapped_pts = f_vmap(pts)
            ax.plot(mapped_pts[:, 0], mapped_pts[:, 1], color='#457B9D', alpha=0.4, linewidth=1.0, zorder=6)
            
        # Horizontal lines
        for y in y_starts:
            pts = jnp.column_stack([x_range, jnp.full(pts_per_line, y)])
            mapped_pts = f_vmap(pts)
            ax.plot(mapped_pts[:, 0], mapped_pts[:, 1], color='#457B9D', alpha=0.4, linewidth=1.0, zorder=6)

    # 1. Faces
    if show_faces:
        is_color_list = isinstance(color_faces, list) and len(color_faces) == len(tessellation.faces)
        for i, face in enumerate(tessellation.faces):
            vertices = tessellation.vertices[face.vertex_indices]
            
            current_color = color_faces[i] if is_color_list else color_faces
            
            # Color kinematic blocks differently
            if show_kinematic_blocks and hasattr(face, 'dofs') and len(face.dofs) > 0:
                current_color = '#6C757D'  # gray for clamped faces
            
            polygon = Polygon(
                vertices, 
                closed=True, 
                facecolor=current_color,
                edgecolor=edge_color,
                linewidth=edge_width,
                alpha=face_alpha,
                zorder=10
            )
            ax.add_patch(polygon)

            if show_face_indices:
                centroid = vertices.mean(axis=0)
                ax.text(centroid[0], centroid[1], str(i), color='black', fontsize=10, 
                        ha='center', va='center', weight='bold', zorder=20)

    # 2. Vertices
    if show_vertices:
        X = tessellation.vertices
        ax.scatter(X[:, 0], X[:, 1], color='#E63946', s=20, zorder=25)
        for i, v in enumerate(X):
            ax.text(v[0], v[1], f"v{i}", color='#E63946', fontsize=8, ha='right', zorder=26)

    # 3. Hinges (vectors and midpoints)
    if show_hinges:
        num_faces = len(tessellation.faces)
        for i, hinge in enumerate(tessellation.hinges):
            if hinge.face1 < 0 or hinge.face2 < 0: continue
            
            v1 = tessellation.vertices[hinge.vertex1]
            v2 = tessellation.vertices[hinge.vertex2]
            v1_adj = tessellation.vertices[hinge.vertex_adjacent1]
            v2_adj = tessellation.vertices[hinge.vertex_adjacent2]

            midpoint = (v1 + v2) / 2
            
            # Deployment/contraction vectors
            hinge_vector1 = 0.3 * (v1_adj - v1)
            hinge_vector2 = 0.3 * (v2_adj - v2)

            ax.arrow(midpoint[0], midpoint[1], hinge_vector1[0], hinge_vector1[1], 
                     head_width=0.015, head_length=0.015, fc="#000000", ec="#000000", zorder=12)
            ax.arrow(midpoint[0], midpoint[1], hinge_vector2[0], hinge_vector2[1], 
                     head_width=0.015, head_length=0.015, fc="#000000", ec="#000000", zorder=12)
            
            ax.scatter(midpoint[0], midpoint[1], color='#F1FAEE', edgecolor="#000000", s=30, linewidth=1.5, zorder=14)
            
            if show_hinge_indices:
                ax.text(midpoint[0], midpoint[1] + 0.02, f"h{i}", color='#457B9D', fontsize=8, 
                        ha='center', va='bottom', weight='bold', zorder=22)

    # 4. External Forces & Moments
    if show_external_forces:
        # Size arrows relative to the domain and to the largest load in the scene,
        # so the biggest force is a clearly visible fraction of the geometry and the
        # rest are drawn in proportion (independent of absolute force magnitude).
        X_all = tessellation.vertices
        domain_scale = float(np.linalg.norm(X_all.max(axis=0) - X_all.min(axis=0))) if len(X_all) else 1.0
        max_force = 0.0
        max_moment = 0.0
        for face in tessellation.faces:
            if hasattr(face, 'loads') and face.loads:
                max_force = max(max_force, float(np.hypot(face.loads.get(0, 0.0), face.loads.get(1, 0.0))))
                max_moment = max(max_moment, abs(float(face.loads.get(2, 0.0))))
        ref_len = 0.16 * domain_scale          # length of the largest force arrow
        head = 0.045 * domain_scale
        shaft = 0.012 * domain_scale

        for face in tessellation.faces:
            if not (hasattr(face, 'loads') and face.loads):
                continue
            vertices = tessellation.vertices[face.vertex_indices]
            centroid = vertices.mean(axis=0)

            fx = face.loads.get(0, 0.0)
            fy = face.loads.get(1, 0.0)
            moment = face.loads.get(2, 0.0)

            # Draw force vector — length proportional to |F| / max|F|.
            if (fx != 0 or fy != 0) and max_force > 0:
                fmag = np.hypot(fx, fy)
                length = ref_len * fmag / max_force
                dx, dy = fx / fmag * length, fy / fmag * length
                ax.arrow(centroid[0], centroid[1], dx, dy,
                         head_width=head, head_length=head, width=shaft,
                         length_includes_head=True, fc="#D62828", ec="#8B0000",
                         linewidth=0.8, alpha=0.95, zorder=30)

            # Draw moment as a curved arrow, sized to the domain.
            if moment != 0:
                r = 0.07 * domain_scale
                if moment > 0:                  # counter-clockwise
                    start = (centroid[0] + r, centroid[1] - r / 2)
                    end = (centroid[0] - r / 2, centroid[1] + r)
                    rad = 0.6
                else:                           # clockwise
                    start = (centroid[0] - r, centroid[1] - r / 2)
                    end = (centroid[0] + r / 2, centroid[1] + r)
                    rad = -0.6
                arrow = FancyArrowPatch(start, end, connectionstyle=f"arc3,rad={rad}",
                                        color="#D62828",
                                        arrowstyle="Simple, tail_width=1.5, head_width=6, head_length=8",
                                        mutation_scale=max(10.0, 0.6 * domain_scale), zorder=30)
                ax.add_patch(arrow)

    # 4. Target Shape
    if show_target:
        target_pts = get_target_points(target_params, n_points=200)
        if len(target_pts) > 0:
            plot_pts = np.vstack([target_pts, target_pts[0]])
            ax.plot(plot_pts[:, 0], plot_pts[:, 1], color=target_color, linestyle=target_style,
                    linewidth=target_width, zorder=5)

    if title:
        ax.set_title(title, fontsize=16, weight='bold', color='black', pad=20)

    X = tessellation.vertices
    if len(X) > 0:
        x_min, y_min = X.min(axis=0)
        x_max, y_max = X.max(axis=0)

        if show_target and 'target_pts' in locals() and len(target_pts) > 0:
            x_min = min(x_min, target_pts[:, 0].min())
            x_max = max(x_max, target_pts[:, 0].max())
            y_min = min(y_min, target_pts[:, 1].min())
            y_max = max(y_max, target_pts[:, 1].max())

        center_x = (x_max + x_min) / 2
        center_y = (y_max + y_min) / 2

        delta_x = (x_max - x_min)
        delta_y = (y_max - y_min)
        max_range = max(delta_x, delta_y) * 1.1  # +10% margin
        
        ax.set_xlim(center_x - max_range/2, center_x + max_range/2)
        ax.set_ylim(center_y - max_range/2, center_y + max_range/2)

    ax.set_aspect('equal')
    ax.axis('off')
    return ax

def animate_tessellation(tessellation, state_history, filepath="closing_animation.gif", fps=15, target_params=None, **plot_kwargs):
    """
    Animates the tessellation process given a history of states and saves it to a file.
    """
    if not state_history:
        print("Warning: state_history is empty, cannot animate.")
        return
    
    if filepath is None:
        print("No filepath provided, skipping animation generation.")
        return

    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 10), facecolor='#FFFFFF')
    
    # Compute fixed bounds so the camera doesn't jitter
    all_X = np.concatenate(state_history)
    x_min, y_min = all_X.min(axis=0)
    x_max, y_max = all_X.max(axis=0)
    center_x = (x_max + x_min) / 2
    center_y = (y_max + y_min) / 2
    delta_x = (x_max - x_min)
    delta_y = (y_max - y_min)
    max_range = max(delta_x, delta_y) * 1.1

    def update(frame):
        ax.clear()
        tessellation.update_vertices(state_history[frame])
        
        # We reuse the existing plotting function
        plot_tessellation(tessellation, ax=ax, title="Opening process", target_params=target_params, **plot_kwargs)
        
        # Enforce fixed bounds over the automatic ones computed in plot_tessellation
        ax.set_xlim(center_x - max_range/2, center_x + max_range/2)
        ax.set_ylim(center_y - max_range/2, center_y + max_range/2)

    print(f"Generating animation with {len(state_history)} frames...")
    ani = animation.FuncAnimation(fig, update, frames=len(state_history), blit=False)
    
    if filepath is not None:
        writer = 'pillow' if filepath.endswith('.gif') else None
        ani.save(filepath, writer=writer, fps=fps)
        print(f"Animation successfully saved to {filepath}")
    else:
        # If no filepath is provided, do nothing or display it (if in a notebook)
        pass
    plt.close(fig)

def plot_tessellation_differences(tessellation, diff_values, ax=None, 
                                  title="Deformation Map",
                                  cmap_name='YlOrRd',
                                  **kwargs):
    """
    Plots the tessellation where faces are colored based on a difference metric.
    """
    if diff_values is None or len(diff_values) == 0:
        print("Warning: diff_values is empty.")
        return ax
        
    min_val = jnp.min(diff_values)
    max_val = jnp.max(diff_values)
    
    if max_val > min_val:
        norm = mcolors.Normalize(vmin=min_val, vmax=max_val)
    else:
        norm = mcolors.Normalize(vmin=min_val - 0.1, vmax=max_val + 0.1)
        
    cmap = cm.get_cmap(cmap_name)
    face_colors = [cmap(norm(val)) for val in diff_values]
    
    if ax is None:
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(10, 10), facecolor='#FFFFFF')
    else:
        fig = ax.figure
        
    ax = plot_tessellation(tessellation, ax=ax, title=title, color_faces=face_colors, **kwargs)
    
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.7)
    cbar.set_label('Deformation (%)', rotation=270, labelpad=15)
    
    return ax