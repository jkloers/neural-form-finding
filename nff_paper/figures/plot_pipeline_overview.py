"""Nature-style single-plate overview of the differentiable closed-kirigami pipeline.

Card-based training loop over TWO rows: each step is a boxed CARD (visual on top, formula + a small
explanatory diagram below). The backward pass runs along the bottom and rises into the THETA box (a).

          +--------- differentiable forward model ---------+
    [ a ] |  [ b ROM ]   >   [ c DEPLOY ]                  |
      ^   |     +-------------+  (wrap, inside the bar)    |
      |   |  [ d vs TARGET ]  |                            |
      |   +-------------------+       (L)     [ e : cut ]
      |                  riser <-- backward pass: dL/dtheta

The canvas is authored at its PRINTED size (see FIG_W_BP), so an authored fontsize IS its printed
point size. Do not enlarge the canvas without rescaling the type ladder, and do not add
`bbox_inches` to savefig -- either one silently breaks that guarantee.

Real geometry from a trained closed_les run; rendered clean/schematic, Princeton palette (no blue).

    JAX_PLATFORMS=cpu conda run -n kgnn_mac \
        python nff_paper/figures/plot_pipeline_overview.py --config-name overview
"""

import os
import glob
import pickle
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.text
import matplotlib.lines
from matplotlib.patches import (Polygon as MplPolygon, Circle, FancyArrowPatch, PathPatch,
                                FancyBboxPatch)
from matplotlib.path import Path as MplPath
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

# ── Princeton palette (house standard; NO blue, per user pref) ──
from nff.utils.figstyle import (FS_MICRO, FS_PANEL, GREEN, GREY, INK, ORANGE, PURPLE, RED, TEAL,
                                panel_letter, panel_letters_on,
                                TEXT_W_BP, apply_charter, paper_rc, slide_bp)
CREAM  = "#FDF3E7"
VIVID      = ORANGE      # outer physical cut patterns (a, e)
LIGHT      = "#FBDDB6"   # inner-panel fill (b, c, d)
INNER_EDGE = "#C77B3B"   # inner-panel outline (thin)
FACE_GREY  = "#DEDBD5"   # muted schematic faces (ROM two-faces detail)
FACE_EDGE  = "#9A968E"
BOX_FILL   = "#F7F5F2"   # card fill
FWD_FILL   = "#F1F3F5"   # forward-model container
BACK_FILL  = "#ECECEC"   # backward-pass lane fill
R_COLOR = ORANGE
GEOM_COLOR, PHYS_COLOR = GREEN, "#C1272D"   # physical / hinge-damage loss -> red
CUT_DIM    = "#7A7A7A"   # dim ROM cut centerlines on panel a (uniform aspect)
DMG_FILL   = "#F7DAD9"   # light red fill under the damage-loss training curve
TENSION_LB = "#C25A0A"   # the "tension" LABEL: ORANGE is 2.4:1 on cream, too weak for 6 pt text

ENERGY_CMAP = LinearSegmentedColormap.from_list("hingeW", [CREAM, "#F9B266", ORANGE, RED, "#7A1010"])


# This plate is a schematic, not a plot: no grid and thin axes. Family and Type-42 embedding come
# from the shared policy; the default size stays at the plate's own FS_TITLE, since every string on
# it is sized explicitly from the ladder below and this is only the fallback.
_PLATE_RC = {**paper_rc(), "axes.grid": False, "axes.linewidth": 0.5, "font.size": 7}


# ══════════════════════════════════════════════════════════════════════════════
# Page geometry — the canvas is authored at its PRINTED size, so authored pt = printed pt
# ══════════════════════════════════════════════════════════════════════════════

FIG_W_BP  = 1.18 * TEXT_W_BP         # tied to \includegraphics[width=1.18\textwidth] in
FIG_H_BP  = FIG_W_BP / 1.03          #   paper/sn-article.tex — change one, change the other
# At this height (425 bp) the plate + its 5-line caption is 480 bp, i.e. 87% of \textheight 550.6,
# so LaTeX gives it a FLOAT PAGE. That is deliberate: anything short enough to share a page
# (<= 359 bp of graphic) would cost 12% off every card, which is the legibility this rebuild bought.

fx = lambda bp: bp / FIG_W_BP        # bp -> figure fraction (x)
fy = lambda bp: bp / FIG_H_BP        # bp -> figure fraction (y)

# Type ladder. At scale 1.0 these ARE the printed point sizes. The ends are the shared ladder's
# own rungs; the three in between are card-formula sizes this plate is alone in needing, so they
# stay here rather than widening the policy every other figure follows.
FS_FORMULA = 7.5     # card formulas
FS_TITLE   = 7.0     # card titles, container label, lane caption, Adam badge
FS_SMALL   = 6.5     # secondary formulas, gloss lines
FS_LEADER  = FS_MICRO  # leaders inside the visuals, axis labels

# Drawing weights. The printed drawing grows 1.37x while the canvas scale grows 2.72x, so strokes
# and markers tuned for the old 20-inch canvas must come down by that ratio.
DRAW = 1.37 / 2.72
dw = lambda x, floor=0.3: max(floor, x * DRAW)   # linewidths, markersizes, mutation_scale
dA = lambda x: x * DRAW ** 2                     # scatter `s` (an AREA, so squared)


# ══════════════════════════════════════════════════════════════════════════════
# Data spine
# ══════════════════════════════════════════════════════════════════════════════

def _find_best_params(config_name):
    for r in reversed(sorted(glob.glob(f"data/outputs/runs/run_*_{config_name}"))):
        p = os.path.join(r, "best_params.pkl")
        if os.path.exists(p):
            return p
    return None


def _cut_geom(sf, params, build_cut_geometry, solve, bflat, fillet=True):
    """Precise cut geometry (fine kerf, sheet units). fillet=False -> no rounded cut tips."""
    boundary = bflat(sf['sliders'], params['bnd_logits'])
    coords = np.asarray(solve(sf['struct'], boundary, jax.nn.sigmoid(params['z'])))
    T, cols = sf['struct']['T'], sf['struct']['cols']
    u = float(np.ptp(coords, axis=0).max()) / 5.0
    geom = build_cut_geometry(coords, T, cols, w_c=0.028 * u, w_lig=0.12 * u,
                              rho=(0.03 * u if fillet else 0.0), length_scale=1.0)
    return geom, coords


def extract_spine(config_name):
    from nff.config.experiment import load_and_parse_config
    from nff.stages.pipeline import forward_pipeline
    from nff.stages.geometry import reconstruct_vertices, deformed_vertices
    from nff.stages.physics.kinematics import face_to_node_kinematics_fn
    from nff.stages.physics.energy import ligament_strains_linearized
    from nff.closed.setup import build_closed_initial_state, init_closed_les_params
    from nff.hinge import build_hinge_model
    from nff.closed.deploy import _deployed_hinge_xy, _boundary_cloud, _fit_circle
    from nff.topology.closed_builder_jax import solve_cut_vertices_jax, boundary_flat_from_logits
    from nff.fabrication.cut_pattern import build_cut_geometry

    cfg = load_and_parse_config(f"data/configs/closed/{config_name}.yaml")
    state, tess = build_closed_initial_state(cfg)
    params, sf = init_closed_les_params(cfg)
    bond_energy, *_ = build_hinge_model(cfg, sf, state, params)

    bp_path = _find_best_params(config_name)
    if bp_path is None:
        raise SystemExit(f"No best_params.pkl for '{config_name}'. Run run_closed.py first.")
    with open(bp_path, "rb") as f:
        bp = {k: jnp.asarray(v) for k, v in pickle.load(f).items()}

    res = forward_pipeline(state, cfg.target, cfg.validity, cfg.physics,
                           map_type=cfg.mapping.type, map_params=bp, static_features=sf,
                           load_specs=cfg.topology.get('loads', []),
                           bond_energy_fn=bond_energy, hinge_geometry=None)
    ms, vs = res['mapped_state'], res['valid_state']
    fields = np.asarray(res['solution'].fields)
    ref = res['reference_bond_vectors']
    cnv = vs.centroid_node_vectors
    nf, nn, _ = cnv.shape
    bc = np.asarray(vs.bond_connectivity)
    l0 = jnp.linalg.norm(ref, axis=-1)

    def per_hinge_W(disp):
        nd = face_to_node_kinematics_fn(jnp.asarray(disp), cnv).reshape(nf * nn, 3)
        a, s, r = ligament_strains_linearized(nd[bc[:, 0]], nd[bc[:, 1]], reference_vector=ref)
        return np.asarray(vs.k_stretch * (a * l0) ** 2 / 2 + vs.k_shear * (s * l0) ** 2 / 2
                          + vs.k_rot * r ** 2 / 2)

    def components(disp):
        nd = face_to_node_kinematics_fn(jnp.asarray(disp), cnv).reshape(nf * nn, 3)
        a, s, r = ligament_strains_linearized(nd[bc[:, 0]], nd[bc[:, 1]], reference_vector=ref)
        return (float(jnp.sum(vs.k_stretch * (a * l0) ** 2 / 2)),
                float(jnp.sum(vs.k_shear * (s * l0) ** 2 / 2)),
                float(jnp.sum(vs.k_rot * r ** 2 / 2)))

    panel_flat = np.asarray(reconstruct_vertices(ms.face_centroids, ms.centroid_node_vectors))
    n_steps = fields.shape[0]
    frame_ids = [0, n_steps // 3, 2 * n_steps // 3, n_steps - 1]
    panel_frames = np.stack([np.asarray(deformed_vertices(vs, jnp.asarray(fields[k])))
                             for k in frame_ids])

    disp = fields[-1]
    W = per_hinge_W(disp)
    hinge_xy_flat = np.asarray(_deployed_hinge_xy(vs, jnp.zeros((nf, 3))))
    hinge_xy_dep = np.asarray(_deployed_hinge_xy(vs, jnp.asarray(disp)))

    n_half = max(1, n_steps // 2)
    panel_half = np.asarray(deformed_vertices(vs, jnp.asarray(fields[n_half])))
    W_half = per_hinge_W(fields[n_half])
    hinge_xy_half = np.asarray(_deployed_hinge_xy(vs, jnp.asarray(fields[n_half])))
    fc_half = np.asarray(vs.face_centroids) + np.asarray(fields[n_half][:, :2])
    panel_flat_frame = np.asarray(deformed_vertices(vs, jnp.zeros((nf, 3))))

    # energy history over the whole load path (real components)
    comp = np.array([components(fields[k]) for k in range(n_steps)])
    comp = np.vstack([np.zeros((1, 3)), comp])
    energy_hist = dict(lam=np.linspace(0, 1, comp.shape[0]),
                       stretch=comp[:, 0], shear=comp[:, 1], rot=comp[:, 2], total=comp.sum(1))

    cloud = np.asarray(_boundary_cloud(vs, jnp.asarray(disp)))
    tcen0, _ = _fit_circle(cloud)
    # geometric mean-radius circle centred on the cloud centroid -> the boundary points straddle it
    # evenly (tighter-looking fit than the algebraic circle). TARGET_SCALE nudges the size.
    tcen = cloud.mean(0)
    trad = float(np.mean(np.linalg.norm(cloud - tcen, axis=1))) * 1.02

    geom_init, coords_init = _cut_geom(sf, params, build_cut_geometry, solve_cut_vertices_jax,
                                       boundary_flat_from_logits, fillet=False)   # no fillets on a
    geom_opt, _ = _cut_geom(sf, bp, build_cut_geometry, solve_cut_vertices_jax,
                            boundary_flat_from_logits, fillet=True)                # fillets on e
    slider_xy = np.concatenate([coords_init[np.asarray(e['pbase'])] for e in sf['sliders']['edges']])
    bmask = np.asarray(sf['struct']['les_idx']['boundary_mask'])
    r_vertex_xy = np.unique(np.round(coords_init[~bmask], 6), axis=0)
    # highlight ONE central interior horizontal cut on panel a: cut = A-B (neighbours), sliders = x,x'
    T, cols, rows = sf['struct']['T'], sf['struct']['cols'], sf['struct']['rows']
    pidf = lambda i, j, s: 2 * (i * cols + j) + s
    highlight_cut = None
    ci, cj = rows // 2, cols // 2
    for i, j in [(ci, cj), (ci, cj - 1), (ci - 1, cj), (ci + 1, cj), (ci, cj + 1), (ci - 1, cj - 1)]:
        if 0 < i < rows - 1 and 0 < j < cols - 1 and int(T[i, j]) == 1:
            # A, B are the facing endpoints of the two NEIGHBOURING cuts (Sec. sec:tessellation);
            # x, xp are THIS cut's own two endpoints, which r places on the segment [A, B].
            highlight_cut = dict(A=coords_init[pidf(i - 1, j, 0)], B=coords_init[pidf(i + 1, j, 1)],
                                 x=coords_init[pidf(i, j, 0)], xp=coords_init[pidf(i, j, 1)])
            break
    # every interior cut's two sliders x, x' + the cut direction (for the knobs on panel a)
    all_sliders = []
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            t = int(T[i, j])
            if t == 1:
                Aa, Bb = coords_init[pidf(i - 1, j, 0)], coords_init[pidf(i + 1, j, 1)]
            elif t == 2:
                Aa, Bb = coords_init[pidf(i, j - 1, 1)], coords_init[pidf(i, j + 1, 0)]
            else:
                continue
            d = Bb - Aa; nd = float(np.linalg.norm(d))
            if nd > 1e-9:
                all_sliders.append((coords_init[pidf(i, j, 0)], coords_init[pidf(i, j, 1)], d / nd))
    # boundary sliders (s) as knobs along each edge, extending inward
    sheet_c = coords_init.mean(0)
    bnd_sliders = []
    for e in sf['sliders']['edges']:
        u = np.array([1.0, 0.0]) if e['axis'] == 0 else np.array([0.0, 1.0])
        nrm = np.array([-u[1], u[0]])
        for pb in np.asarray(e['pbase']):
            pos = coords_init[int(pb)]
            side = 1.0 if np.dot(sheet_c - pos, nrm) >= 0 else -1.0
            bnd_sliders.append((pos, u, side))

    # void-diagram basis: deploy the SAME (initial) design shown in panel a, and take the void of the
    # EXACT highlighted cut = the deployed positions of its 4 collinear vertices A, x', x, B.
    void_draw = None
    if highlight_cut is not None:
        res0 = forward_pipeline(state, cfg.target, cfg.validity, cfg.physics, map_type=cfg.mapping.type,
                                map_params=params, static_features=sf,
                                load_specs=cfg.topology.get('loads', []),
                                bond_energy_fn=bond_energy, hinge_geometry=None)
        void_draw = _void_from_cut(res0['valid_state'], np.asarray(res0['solution'].fields), highlight_cut)

    topo = cfg.topology
    return dict(panel_flat=panel_flat, panel_frames=panel_frames, W=W, void_draw=void_draw,
                hinge_xy_flat=hinge_xy_flat, hinge_xy_dep=hinge_xy_dep,
                panel_half=panel_half, W_half=W_half, hinge_xy_half=hinge_xy_half, fc_half=fc_half,
                panel_flat_frame=panel_flat_frame, energy_hist=energy_hist,
                cloud=cloud, tcen=np.asarray(tcen), trad=float(trad),
                geom_init=geom_init, geom_opt=geom_opt, slider_xy=slider_xy, r_vertex_xy=r_vertex_xy,
                highlight_cut=highlight_cut, all_sliders=all_sliders, bnd_sliders=bnd_sliders,
                loaded=_flatten_faces(topo.get('loads', [])),
                clamped=list(topo.get('bc_clamped', []) or []))


def _flatten_faces(load_specs):
    out = []
    for spec in load_specs or []:
        fa = spec.get('face')
        out.extend(fa if isinstance(fa, list) else [fa])
    return [int(x) for x in out]


def _void_from_cut(vs0, fields0, hc):
    """Void of the EXACT highlighted cut = the deployed positions of its 4 collinear cut-vertices
    A, x', x, B (so it matches the cut spotlighted on panel a). Rotated so the long diagonal A-B (the
    line the sheet is cut along, seen when closed) is horizontal and centred on the void; returns the
    neighbours' facing endpoints A, B and this cut's own endpoints x, xp (the sliders) + tiles."""
    cnv = np.asarray(vs0.centroid_node_vectors); fc0 = np.asarray(vs0.face_centroids)
    nf, nn, _ = cnv.shape
    flat = (fc0[:, None] + cnv).reshape(-1, 2)
    disp = np.asarray(fields0[len(fields0) // 2])
    th = disp[:, 2]; ctd, std = np.cos(th), np.sin(th)
    rx = ctd[:, None] * cnv[:, :, 0] - std[:, None] * cnv[:, :, 1]
    ry = std[:, None] * cnv[:, :, 0] + ctd[:, None] * cnv[:, :, 1]
    dep = ((fc0 + disp[:, :2])[:, None] + np.stack([rx, ry], -1)).reshape(-1, 2)
    dep_of = lambda P: dep[int(np.argmin(np.linalg.norm(flat - np.asarray(P), axis=1)))]
    A, B, x, xp = dep_of(hc['A']), dep_of(hc['B']), dep_of(hc['x']), dep_of(hc['xp'])
    # design fractions of the two sliders along the cut (from the FLAT positions = r, 1-r)
    Af, Bf = np.asarray(hc['A']), np.asarray(hc['B'])
    uf = (Bf - Af) / (np.linalg.norm(Bf - Af) ** 2)
    t_x = float(np.dot(np.asarray(hc['x']) - Af, uf))
    t_xp = float(np.dot(np.asarray(hc['xp']) - Af, uf))
    voidc = (A + B + x + xp) / 4.0
    ang = np.arctan2((B - A)[1], (B - A)[0])
    c, s = np.cos(-ang), np.sin(-ang)
    R = np.array([[c, -s], [s, c]])
    tf = lambda P: (np.asarray(P) - voidc) @ R.T
    tiles = [np.array([tf(p) for p in dep[k * nn:(k + 1) * nn]]) for k in range(nf)]
    tiles = [t for t in tiles if np.linalg.norm(t.mean(0)) < 1.85]     # tight ring -> clear margins
    return dict(A=tf(A), B=tf(B), x=tf(x), xp=tf(xp), tiles=tiles, t_x=t_x, t_xp=t_xp)


# ══════════════════════════════════════════════════════════════════════════════
# Drawing primitives
# ══════════════════════════════════════════════════════════════════════════════

def _draw_panels(ax, nodes, facecolor=LIGHT, edgecolor=INK, alpha=1.0, lw=None, zorder=5):
    pc = PatchCollection([MplPolygon(nodes[i], closed=True) for i in range(nodes.shape[0])],
                         facecolor=facecolor, edgecolor=edgecolor, linewidths=dw(1.3) if lw is None
                         else lw, alpha=alpha, zorder=zorder, joinstyle="round")
    ax.add_collection(pc)
    return pc


def _fit_view(ax, pts, pad=0.12):
    mn, mx = pts.min(0), pts.max(0)
    c, r = (mn + mx) / 2, (mx - mn).max() / 2 * (1 + pad)
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_aspect("equal")
    ax.axis("off")


def _match_aspect(ax, xspan):
    """Set ylim so the DATA aspect equals the axes' own BOX aspect, and return that ylim.

    Without this an `equal` axes letterboxes inside its rect: the drawing shrinks but the
    point-sized labels do not, so columns spaced in data units stop fitting the text they hold.
    """
    bb = ax.get_position()
    yspan = xspan / ((bb.width * FIG_W_BP) / (bb.height * FIG_H_BP))
    ax.set_xlim(0.0, xspan); ax.set_ylim(0.0, yspan)
    ax.set_aspect("equal")
    return yspan


def _fit_box(ax, lo, hi):
    """Grow the tighter of [lo,hi]'s two spans to the axes' box aspect, so `equal` cannot letterbox."""
    bb = ax.get_position()
    ba = (bb.width * FIG_W_BP) / (bb.height * FIG_H_BP)
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    c = (lo + hi) / 2
    w, h = hi[0] - lo[0], hi[1] - lo[1]
    if w / h < ba:
        w = h * ba
    else:
        h = w / ba
    ax.set_xlim(c[0] - w / 2, c[0] + w / 2); ax.set_ylim(c[1] - h / 2, c[1] + h / 2)
    ax.set_aspect("equal")


def _shapely_to_patch(poly, **kw):
    polys = list(poly.geoms) if poly.geom_type.startswith("Multi") else [poly]
    verts, codes = [], []
    for p in polys:
        for ring in [p.exterior, *p.interiors]:
            xy = np.asarray(ring.coords)
            verts.extend(xy)
            codes.extend([MplPath.MOVETO] + [MplPath.LINETO] * (len(xy) - 2) + [MplPath.CLOSEPOLY])
    return PathPatch(MplPath(verts, codes), **kw)


def _rounded_path(pts_bp, r_bp):
    """Closed path through bp-space `pts_bp` with radius-`r_bp` corners, in figure fractions.

    Quadratic Beziers with the corner as the control point round convex AND reflex corners with the
    same code, which the forward-model container's single reflex vertex needs. Built in bp so the
    corners stay circular on the page rather than elliptical in figure-fraction space.
    """
    pts = [np.asarray(p, float) for p in pts_bp]
    n = len(pts)
    verts, codes = [], []
    for i, v in enumerate(pts):
        prv, nxt = pts[(i - 1) % n], pts[(i + 1) % n]
        up = (prv - v) / (np.linalg.norm(prv - v) + 1e-12)
        un = (nxt - v) / (np.linalg.norm(nxt - v) + 1e-12)
        rr = min(r_bp, np.linalg.norm(prv - v) / 2, np.linalg.norm(nxt - v) / 2)
        verts.extend([v + up * rr, v, v + un * rr])
        codes.extend([MplPath.MOVETO if i == 0 else MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3])
    verts.append(verts[0]); codes.append(MplPath.CLOSEPOLY)
    v = np.asarray(verts, float)
    return MplPath(np.c_[v[:, 0] / FIG_W_BP, v[:, 1] / FIG_H_BP], codes)


def _elbow_arrow(fig, pts_bp, color, lw=1.0, r_bp=4.0, ms=6.0, zorder=5):
    """Rounded axis-aligned polyline through bp-space waypoints, arrowhead on the final segment."""
    pts = [np.asarray(p, float) for p in pts_bp]
    verts, codes = [pts[0]], [MplPath.MOVETO]
    for i in range(1, len(pts) - 1):
        v, prv, nxt = pts[i], pts[i - 1], pts[i + 1]
        up = (prv - v) / (np.linalg.norm(prv - v) + 1e-12)
        un = (nxt - v) / (np.linalg.norm(nxt - v) + 1e-12)
        rr = min(r_bp, np.linalg.norm(prv - v) / 2, np.linalg.norm(nxt - v) / 2)
        verts.extend([v + up * rr, v, v + un * rr])
        codes.extend([MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3])
    verts.append(pts[-1]); codes.append(MplPath.LINETO)
    v = np.asarray(verts, float)
    frac = np.c_[v[:, 0] / FIG_W_BP, v[:, 1] / FIG_H_BP]
    fig.add_artist(PathPatch(MplPath(frac, codes), transform=fig.transFigure, fill=False,
                             edgecolor=color, lw=lw, zorder=zorder, capstyle="round"))
    tail = pts[-2] + (pts[-1] - pts[-2]) * 0.55
    _fig_arrow(fig, (tail[0] / FIG_W_BP, tail[1] / FIG_H_BP),
               (pts[-1][0] / FIG_W_BP, pts[-1][1] / FIG_H_BP), color, lw, ms=ms)


# Leaders are sized and placed for a 102 bp visual. The slide row gives each visual 61 bp, where
# a two-line boxed callout is wider than the card it sits in, so the row drops them: a presenter
# says what a leader says. Set by build_figure_slide, read here and by panel_target.
_SLIDE = False


def _leader(ax, pt, textpt, label, color, fontsize=FS_LEADER):
    if _SLIDE:
        return
    ax.annotate(label, xy=pt, xytext=textpt, fontsize=fontsize, color=color, ha="center", va="center",
                arrowprops=dict(arrowstyle="-", color=color, lw=dw(1.0)), zorder=12,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, lw=dw(0.8), alpha=0.95))


def _moment_arrow(ax, center, radius, color, ccw=True, lw=None):
    """Clean circular moment symbol: ~300deg arc + a proportional tangential arrowhead."""
    a0, a1 = (120.0, 400.0) if ccw else (60.0, -220.0)
    th = np.radians(np.linspace(a0, a1, 80))
    pts = center + radius * np.c_[np.cos(th), np.sin(th)]
    ax.plot(pts[:, 0], pts[:, 1], color=color, lw=dw(2.4) if lw is None else lw, zorder=11,
            solid_capstyle="round")
    e = th[-1]
    tip = center + radius * np.array([np.cos(e), np.sin(e)])
    tang = np.array([-np.sin(e), np.cos(e)]) * (1 if ccw else -1)     # travel direction
    radial = np.array([np.cos(e), np.sin(e)])
    hl, hw = radius * 0.55, radius * 0.42
    base = tip - tang * hl
    ax.add_patch(MplPolygon([tip, base + radial * hw / 2, base - radial * hw / 2],
                            closed=True, color=color, zorder=11))


def _formula(fig, x, y, text, color=INK, fs=FS_FORMULA):
    fig.text(x, y, text, ha="center", va="center", fontsize=fs, color=color, zorder=8,
             bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=color, lw=dw(1.0), alpha=0.97))


def _colored_line(fig, xc, y, parts, fs=FS_FORMULA, box=False):
    """Lay coloured (text, color) pieces left-to-right, centred on xc, using MEASURED widths.

    box=True draws a rounded white formula box behind them (so it matches _formula boxes).
    """
    r = fig.canvas.get_renderer()
    texts = [fig.text(0.5, y, t, fontsize=fs, color=c, ha="left", va="center", zorder=8)
             for t, c in parts]
    widths = [tx.get_window_extent(renderer=r).width / fig.bbox.width for tx in texts]
    total = sum(widths)
    x0 = xc - total / 2
    if box:
        h = 1.5 * fs / (fig.get_size_inches()[1] * 72.0)     # half-height, in points -> fraction
        fig.add_artist(FancyBboxPatch((x0 - fx(3), y - h), total + 2 * fx(3), 2 * h,
                                      boxstyle=f"round,pad={fx(1)},rounding_size={fx(3)}",
                                      transform=fig.transFigure, facecolor="white", edgecolor=INK,
                                      lw=dw(1.0), zorder=7))
    x = x0
    for tx, w in zip(texts, widths):
        tx.set_position((x, y))
        x += w


# ══════════════════════════════════════════════════════════════════════════════
# Card visuals
# ══════════════════════════════════════════════════════════════════════════════

def panel_cut(ax, geom, kind, S=None):
    ax.add_patch(_shapely_to_patch(geom['sheet'], facecolor=VIVID, edgecolor="none", zorder=2))
    if kind == 'design':
        # ROM view: EVERY cut is a thin dark centerline (one uniform aspect); one is emphasized below.
        for p0, p1 in geom['centerlines']:
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=CUT_DIM, lw=dw(1.0),
                    solid_capstyle="round", zorder=3)
    else:                                                       # panel e: the real cut pattern (slots)
        ax.add_patch(_shapely_to_patch(geom['cuts'], facecolor=INK, edgecolor="none", zorder=3))
    minx, miny, maxx, maxy = geom['sheet'].bounds
    W_, H_ = maxx - minx, maxy - miny
    # The slider knobs sit ON the sheet rim, so limits taken straight from the bounds clip them in
    # half. Pad both cards equally -- a and e are the plate's before/after pair.
    pad = 0.045 * max(W_, H_)
    ax.set_xlim(minx - pad, maxx + pad); ax.set_ylim(miny - pad, maxy + pad)
    ax.set_aspect("equal"); ax.axis("off")
    if kind == 'design':
        u5 = W_ / 5.0
        kl_r, kh_r = 0.36 * u5, 0.20 * u5        # r knobs   (16) -> 5.3 x 2.9 bp
        kl_t, kh_t = 0.42 * u5, 0.24 * u5        # tau knobs (16) -> 6.1 x 3.5 bp
        # ONE void-aspect-ratio slider per cut (a single r sets both cut-vertices), void side.
        for xx, xpp, uh in S['all_sliders']:
            nrm = np.array([-uh[1], uh[0]])
            _slider_knob(ax, xx, uh, nrm, -1.0, kl_r, kh_r, color=R_SLIDER, ec="white",
                         lw=dw(0.9), zorder=6)
        # boundary sliders tau: SAME knob design, different colour (green), along each edge
        for pos, uh, sd in S['bnd_sliders']:
            nrm = np.array([-uh[1], uh[0]])
            _slider_knob(ax, pos, uh, nrm, sd, kl_t, kh_t, color=TAU_SLIDER, ec="white",
                         lw=dw(0.9), zorder=6)
        # The hinge descriptor g_h is NOT annotated here: a hinge stores no geometry the cut
        # pattern shows, and it enters the model only through the energy, so card b carries it.
        # emphasize the ONE boundary slider the caption points to
        bpos, buh, bsd = min(S['bnd_sliders'], key=lambda t: t[0][1])
        bnrm = np.array([-buh[1], buh[0]])
        _slider_knob(ax, bpos, buh, bnrm, bsd, kl_t * 1.35, kh_t * 1.35, color=TAU_SLIDER,
                     ec=INK, lw=dw(1.0), zorder=8)
        _leader(ax, bpos, np.array([minx + 0.23 * W_, miny + 0.19 * H_]),
                "boundary\nposition $\\tau$", TAU_SLIDER, FS_LEADER)
        # HIGHLIGHT one cut: a THICKER dark centerline A-B + its r slider, framed as the zoom region
        #
        # Not on the slide row. The frame is a ZOOM MARKER, and the band it points at is one of the
        # two the row drops, so it would mark nothing. The labels are the harder reason: they are
        # set in absolute points against a visual that went from 102 bp to 61, so A lands on a cut
        # marker and B hangs off the frame. The parametrization slide carries this cut properly,
        # cropped from the page plate at about 105 per cent.
        hc = None if _SLIDE else S.get('highlight_cut')
        if hc is not None:
            A, B, x, xp = hc['A'], hc['B'], hc['x'], hc['xp']
            # the zoom marker, at the SAME aspect as the detail band below so the two frames rhyme
            m = 0.30 * u5
            lo = np.minimum.reduce([A, B, x, xp]) - m
            hi = np.maximum.reduce([A, B, x, xp]) + m
            c = (lo + hi) / 2.0
            bw = hi[0] - lo[0]; bh = bw / ZOOM_ASPECT
            ax.add_patch(FancyBboxPatch((c[0] - bw / 2, c[1] - bh / 2), bw, bh,
                                        boxstyle=f"round,pad=0,rounding_size={0.17 * u5}",
                                        facecolor="none", edgecolor=INK, lw=dw(1.1), zorder=10))
            ax.plot([A[0], B[0]], [A[1], B[1]], color=INK, lw=dw(3.0), solid_capstyle="round",
                    zorder=7)
            import matplotlib.patheffects as pe
            # A, B are the facing endpoints of the two NEIGHBOURING cuts and x, x' this cut's own.
            # Labelling all four is what lets the zoom below be read as the void they bound.
            for P, lab, off in ((A, r"$\boldsymbol{A}$", -1.0), (B, r"$\boldsymbol{B}$", 1.0)):
                ax.plot(*P, marker="o", mfc="white", mec=INK, ms=dw(5), zorder=8)
                ax.text(P[0] + off * 0.045 * W_, P[1] + 0.035 * H_, lab, ha="center", va="center",
                        fontsize=FS_MICRO, color=INK, zorder=11,
                        path_effects=[pe.withStroke(linewidth=dw(2.6), foreground="white")])
            sgn_x = 1.0 if x[1] >= xp[1] else -1.0
            for P, lab, sgn in ((x, r"$\boldsymbol{x}$", sgn_x),
                                (xp, r"$\boldsymbol{x}'$", -sgn_x)):
                P = np.asarray(P)
                ax.plot(*P, marker="o", mfc="white", mec=INK, ms=dw(4.4), zorder=8)
                ax.text(P[0], P[1] + sgn * 0.055 * H_, lab, ha="center", va="center",
                        fontsize=FS_MICRO, color=INK, zorder=11,
                        path_effects=[pe.withStroke(linewidth=dw(2.6), foreground="white")])
            u_hat = (B - A) / np.linalg.norm(B - A); nrm = np.array([-u_hat[1], u_hat[0]])
            _slider_knob(ax, x, u_hat, nrm, -1.0, kl_r, kh_r, ec=INK, lw=dw(0.9), zorder=9)


def panel_rom(ax, S):
    nodes = S['panel_flat']
    _draw_panels(ax, nodes, facecolor=LIGHT, edgecolor=INNER_EDGE, lw=dw(0.9), zorder=4)
    hp = S['hinge_xy_flat']
    # ⚠ The flat closed sheet puts hinge points in NEAR-COINCIDENT PAIRS: two cuts meeting at a
    # corner give two hinges a fraction of a marker apart. At a fixed marker size they merge into
    # one blob, which the page plate does too. They are two hinges and must not be deduplicated,
    # so the row draws them smaller instead, at 0.6 of the page diameter, with a thinner keyline
    # so a small marker is not mostly edge. Page output is untouched.
    _hs, _hlw = (11, 0.5) if _SLIDE else (56, 1.0)
    ax.scatter(hp[:, 0], hp[:, 1], s=dA(_hs), color=TEAL, edgecolor="white", linewidths=dw(_hlw),
               zorder=8)
    _fit_view(ax, nodes.reshape(-1, 2))
    c = nodes.mean(1).mean(0)
    hid = int(np.argmin(np.linalg.norm(hp - c, axis=1)))
    _leader(ax, hp[hid], hp[hid] + np.array([1.5, 1.3]) * np.ptp(nodes[0], 0).max() * 0.8,
            r"hinge $(a,s,\vartheta)$", TEAL)
    fid = int(np.argmin(np.linalg.norm(nodes.mean(1) - c, axis=1)))
    pc = nodes[fid].mean(0)
    _leader(ax, pc, pc + np.array([-1.6, -1.4]) * np.ptp(nodes[0], 0).max() * 0.7, "rigid panel", GREY)


def panel_deploy(ax, S, kinematic=False):
    """c: HALF deployment; flat (closed) state ghosted behind. Clamp + tile-fitted moment.

    ``kinematic`` draws the same opening with none of the mechanics on it: the hinges become
    plain marks instead of an energy field, and the applied moment goes, a kinematic deployment
    being the mechanism moving under no load at all. It is what the first sight of the pipeline
    shows, before either the hinge energy or equilibrium has been introduced.
    """
    half = S['panel_half']
    _draw_panels(ax, S['panel_flat_frame'], facecolor="none", edgecolor=GREY, alpha=0.28,
                 lw=dw(0.8), zorder=2)                                # closed (start) ghost
    _draw_panels(ax, half, facecolor=LIGHT, edgecolor=INNER_EDGE, lw=dw(0.9), zorder=4)
    span = np.ptp(half.reshape(-1, 2), 0).max()
    for fid in S['clamped']:
        _draw_panels(ax, half[[fid]], facecolor=GREY, alpha=0.7, edgecolor=INK, lw=dw(1.2), zorder=6)
        p = S['fc_half'][fid]
        _leader(ax, p, p + np.array([-0.34, -0.46]) * span, "clamped\ncentre", GREY, FS_LEADER)
    if not kinematic:
        for fid in S['loaded']:
            tile = half[fid]; tc = tile.mean(0)
            mom_r = 0.36 * float(np.min(np.linalg.norm(tile - tc, axis=1)))
            _moment_arrow(ax, tc, mom_r, RED, ccw=True, lw=dw(2.2))
            _leader(ax, tc + np.array([mom_r, mom_r]) * 0.9, tc + np.array([0.46, 0.50]) * span,
                    r"applied moment $M$", RED, FS_LEADER)
    if kinematic:
        ax.scatter(S['hinge_xy_half'][:, 0], S['hinge_xy_half'][:, 1], c=TEAL, s=dA(46),
                   edgecolor=INK, linewidths=dw(0.5), zorder=9)
        sc = norm = None
    else:
        W = S['W_half']
        norm = Normalize(vmin=float(np.quantile(W, 0.05)), vmax=float(np.quantile(W, 0.92)))
        sc = ax.scatter(S['hinge_xy_half'][:, 0], S['hinge_xy_half'][:, 1], c=W,
                        cmap=ENERGY_CMAP, norm=norm, s=dA(46), edgecolor=INK,
                        linewidths=dw(0.5), zorder=9)
    _fit_view(ax, S['panel_frames'][-1].reshape(-1, 2), pad=0.22)
    return sc, norm


def panel_target(ax, S):
    final = S['panel_frames'][-1]
    _draw_panels(ax, final, facecolor=LIGHT, edgecolor=INNER_EDGE, lw=dw(0.9), zorder=3)
    cen, rad = S['tcen'], S['trad']
    ax.add_patch(Circle(cen, rad, fill=False, edgecolor=INK, lw=dw(2.2), ls="--", zorder=6))
    # Two lines anchored to the top-right corner of the view, mirroring the green label opposite.
    # A single line offset outward from the circle is 35 bp wide and the view is only ~79 bp, so it
    # left the card and printed clipped.
    if not _SLIDE:
        tp = cen + rad * np.array([0.94, 0.34])      # low enough that the leader reads as a leader
        ax.annotate("target\nshape", xy=tp, xytext=cen + rad * np.array([1.24, 1.24]),
                    ha="right", va="top", multialignment="right",
                    fontsize=FS_LEADER, color=INK, zorder=12,
                    arrowprops=dict(arrowstyle="-", color=INK, lw=dw(0.9)))
    cl = S['cloud']
    ax.scatter(cl[:, 0], cl[:, 1], s=dA(44), color=GREEN, edgecolor="white", linewidths=dw(0.9),
               zorder=8)
    lp = cl[np.argmin(cl[:, 0] - 0.7 * cl[:, 1])]                 # an upper-left boundary point
    _leader(ax, lp, np.array([cen[0] - 1.02 * rad, cen[1] + 0.92 * rad]),
            "deployed\nboundary\nvertices", GREEN, FS_LEADER)
    corners = cen + rad * np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    _fit_view(ax, np.vstack([final.reshape(-1, 2), corners]), pad=0.30)   # extra corner room


# ══════════════════════════════════════════════════════════════════════════════
# Card details (below the visual)
# ══════════════════════════════════════════════════════════════════════════════

# The design-parameter blocks of eq:design-params, theta = (r, tau, {g_h}), each with its own
# colour so the formula and the visual read as one thing: the CUT LAYOUT is r = void aspect ratios
# (purple) and tau = boundary sliders (green); the HINGE DESIGNS are the descriptors g_h (teal,
# which is ALREADY the hinge colour everywhere in this plate -- one descriptor per hinge).
# NOT orange (= the paper), never blue.
R_SLIDER = PURPLE
R_SLIDER_FADE = "#8E79AD"     # mid purple (un-highlighted); #B7A6CC vanished against ORANGE
TAU_SLIDER = GREEN            # every boundary slider is full green -- there are only 16 of them
G_COLOR = TEAL                # the hinge descriptor g_h = (w_lig, alpha, rho_f, w_c)


def _slider_knob(ax, foot, u_hat, nrm, side, kl, kh, color=R_SLIDER, zorder=8, ec=None, lw=None):
    """A small asymmetric slider knob: half-a-square sitting ON the cut line, extending to ONE side."""
    c1, c2 = foot - (kl / 2) * u_hat, foot + (kl / 2) * u_hat
    rect = np.array([c1, c2, c2 + side * kh * nrm, c1 + side * kh * nrm])
    if ec is None:
        ec = INK if color in (R_SLIDER, TAU_SLIDER) else color
    ax.add_patch(MplPolygon(rect, closed=True, facecolor=color, edgecolor=ec,
                            lw=dw(0.5) if lw is None else lw, zorder=zorder, joinstyle="round"))


def detail_theta(ax, S):
    """a: a MAGNIFIED view of the cut framed on the sheet above, showing how the slider sets the void
    aspect ratio. A and B are the facing endpoints of the two neighbouring cuts and A-B is the line
    seen when closed; r places this cut's own endpoints x, x' on it, so the void that opens has
    sides |A-x| = (1-r)L and |x-B| = rL. Those two lengths are what the drawing dimensions: they are
    the paper's r (eq:aspect-ratio) and they hold at every opening angle, unlike the ratio of the
    void's two diagonals that this panel printed until 2026-08-23.

    Every length is a fraction of L = |A-B|, so the drawing is scale-free and the magnification is
    set only by _fit_box. w_lig is NOT here -- it is a leader on the sheet; dropping it is what pays
    for drawing everything else roughly twice as large.
    """
    ax.axis("off"); ax.set_aspect("equal")
    vd = S.get("void_draw")
    if vd is None:
        ax.text(0.5, 0.5, "(void geometry unavailable)", ha="center", fontsize=FS_MICRO, color=GREY)
        return
    import matplotlib.patheffects as pe
    halo = [pe.withStroke(linewidth=dw(2.6), foreground="white")]
    A, B, x, xp = vd["A"], vd["B"], vd["x"], vd["xp"]
    void = np.array([A, x, B, xp])
    u = (B - A); L = float(np.linalg.norm(u)); u_hat = u / L; nrm = np.array([-u_hat[1], u_hat[0]])
    ax.add_patch(MplPolygon(void, closed=True, facecolor="white", edgecolor=INK, lw=dw(1.6), zorder=4))
    ax.plot([A[0], B[0]], [A[1], B[1]], color=INK, ls=(0, (4, 2.5)), lw=dw(1.4), zorder=5)
    # All four corners carry the names Sec. sec:tessellation gives them: A and B are the facing
    # endpoints of the two NEIGHBOURING cuts, x and x' the two endpoints of THIS one, which r
    # places on [A, B]. They were printed C and D until 2026-08-23, which named nothing in the
    # paper and collided with the cut C_ij itself.
    # The upper summit's label is lifted clear: the two side lengths are printed just above the two
    # edges that meet there, and at 0.150 L it landed between them.
    top_is_x = x[1] >= xp[1]
    top, bot = (x, xp) if top_is_x else (xp, x)
    top_lab, bot_lab = ((r"$\boldsymbol{x}$", r"$\boldsymbol{x}'$") if top_is_x
                        else (r"$\boldsymbol{x}'$", r"$\boldsymbol{x}$"))
    for P, lab, dy in ((A, r"$\boldsymbol{A}$", 0.105), (B, r"$\boldsymbol{B}$", 0.105),
                       (top, top_lab, 0.205), (bot, bot_lab, -0.150)):
        ax.plot(*P, marker="o", mfc="white", mec=INK, ms=dw(7), zorder=6)
        ax.text(P[0], P[1] + dy * L, lab, ha="center", va="center", fontsize=FS_MICRO,
                color=INK, zorder=8, path_effects=halo)
    # ONE slider per cut: a single r sets BOTH cut-vertices, so keep both dotted projections to the
    # summits it opens but draw only ONE knob, on the SAME vertex the sheet above spotlights.
    corners = np.array([A, x, B, xp])
    feet = []
    for P, t in ((x, vd["t_x"]), (xp, vd["t_xp"])):
        foot = A + t * u
        side = 1.0 if np.dot(P - foot, nrm) >= 0 else -1.0
        summit = corners[int(np.argmin(np.linalg.norm(corners - foot, axis=1)))]
        ax.plot([foot[0], summit[0]], [foot[1], summit[1]], color=R_SLIDER, ls=":", lw=dw(0.9),
                zorder=5)
        feet.append((foot, side, t))
    foot, side, _ = feet[0]
    # the hero knob, ABOVE the void edge so overlapping it reads as "the control that sets this"
    _slider_knob(ax, foot, u_hat, nrm, side, kl=0.138 * L, kh=0.080 * L, ec=INK, lw=dw(1.0),
                 zorder=10)
    # l_par below, drawn as a BROKEN dimension with the label in the gap -- that buys back a whole
    # label row, which is most of what makes room for the magnification.
    yb = min(A[1], B[1], x[1], xp[1]) - 0.30 * L      # room for the lower summit's label
    for P in (A, B):
        ax.plot([P[0], P[0]], [P[1], yb + 0.02 * L], color=INK, lw=dw(0.6), zorder=5)
    xm = (A[0] + B[0]) / 2
    for tip, tail in ((A[0], xm - 0.075 * L), (B[0], xm + 0.075 * L)):
        ax.annotate("", xy=(tip, yb), xytext=(tail, yb),
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=dw(1.0), mutation_scale=dw(11)),
                    zorder=5)
    ax.text(xm, yb, r"$L$", ha="center", va="center", fontsize=FS_MICRO, color=INK,
            zorder=7, path_effects=halo)
    # The two sides r sets: |A-x| = (1-r)L and |x-B| = rL, invariant at EVERY opening angle. These
    # replace "r = l_perp/l_par" (printed here until 2026-08-23), which is the ratio of the void's
    # two DIAGONALS -- not r, and not invariant: it runs 0.09 to 1.47 over one opening sweep.
    # The void is a parallelogram, so the other two sides repeat these two and are left unlabelled.
    # Both labels go on the UPPER two edges (the lower half already carries the lower summit's label
    # and the L dimension), so which name lands on which edge follows the upper summit's identity.
    for P, Q, lab in ((A, top, r"$(1-r)L$" if top_is_x else r"$rL$"),
                      (top, B, r"$rL$" if top_is_x else r"$(1-r)L$")):
        mid = (P + Q) / 2.0
        e = Q - P
        n = np.array([-e[1], e[0]])
        n = n / (float(np.linalg.norm(n)) + 1e-12)
        if float(np.dot(n, mid)) < 0.0:          # the void is centred on the origin: point OUTWARD
            n = -n
        pos = mid + 0.17 * L * n
        ax.text(pos[0], pos[1], lab, ha="center", va="center", fontsize=FS_MICRO,
                color=R_SLIDER, zorder=9, path_effects=halo)
    ys_ = void[:, 1]
    xdim = max(x[0], xp[0], B[0])
    _fit_box(ax, (A[0] - 0.30 * L, yb - 0.10 * L), (xdim + 0.30 * L, ys_.max() + 0.46 * L))
    # a band of material behind the void, so the white quad reads as a HOLE rather than a shape
    x0, x1 = ax.get_xlim()
    ax.add_patch(MplPolygon([[x0, ys_.min() - 0.09 * L], [x1, ys_.min() - 0.09 * L],
                             [x1, ys_.max() + 0.09 * L], [x0, ys_.max() + 0.09 * L]],
                            closed=True, facecolor=LIGHT, edgecolor="none", zorder=0))


def detail_two_faces(ax):
    """b: how the hinge measures the relative motion of two INDEPENDENT rigid panels i, k. Three
    isolated modes side by side (rest state ghosted, left panel fixed), each dimensioning exactly ONE
    coordinate of eq:hinge-strain:

        opening   a = d . e_sec         the corotated separation resolved along the cut frame
        shear     s = d . e_sec_perp    the same separation resolved across it
        rotation  vartheta = phi_k - phi_i

    a and s are LENGTHS, not strains -- the dimensionless forms eta_a = a/w_lig, eta_s = s/w_lig
    belong to the surrogate's features, not here. Kinematics only (no energy)."""
    from matplotlib.patches import Arc as MplArc
    ax.axis("off")
    yspan = _match_aspect(ax, 8.4)      # was 9.6; the per-mode formula row is gone, so draw larger
    GHOST = "#B9B6B0"                              # rest-state ghost
    d, g = 0.40, 0.10                              # panel centre-to-tip; half the rest ligament
    yc = yspan * 0.48                              # one shared row of diagrams

    def diamond(cx, cy, ang=0.0):
        """A rotating-squares panel (square on its corner); the CORNER TIPS are the hinge points."""
        a = np.radians([0, 90, 180, 270]) + ang
        return np.c_[cx + d * np.cos(a), cy + d * np.sin(a)]

    def rot_about(P, ang, pts):
        c, s = np.cos(ang), np.sin(ang)
        return (pts - P) @ np.array([[c, s], [-s, c]]) + P

    def face(pts, ghost=False):
        ax.add_patch(MplPolygon(pts, closed=True, facecolor="none" if ghost else LIGHT,
                                edgecolor=GHOST if ghost else INNER_EDGE,
                                lw=dw(1.0) if ghost else dw(1.2),
                                ls=(0, (4, 3)) if ghost else "-", zorder=2 if ghost else 3))

    names = {"opening": "Opening", "shear": "Shear", "rotation": "Rotation"}
    # No per-mode formula line: each mode is DIMENSIONED in the drawing, and the definitions live in
    # eq:hinge-strain. Printing them again here only competed with the arrow labels a, s, vartheta.
    for mode, cx in zip(("opening", "shear", "rotation"), (1.40, 4.20, 7.00)):
        acx, bcx = cx - (g + d), cx + (g + d)      # panel i (fixed) / panel k (at rest) centres
        P1 = np.array([acx + d, yc])               # i's hinge CORNER (its right tip)
        P2r = np.array([bcx - d, yc])              # k's hinge CORNER at REST (its left tip)
        if mode == "opening":
            rf = diamond(bcx + 0.60, yc); P2 = P2r + np.array([0.60, 0.0])
        elif mode == "shear":
            rf = diamond(bcx, yc - 0.60); P2 = P2r + np.array([0.0, -0.60])
        else:
            th = np.radians(26.0); rf = rot_about(P2r, th, diamond(bcx, yc)); P2 = P2r.copy()
        face(diamond(bcx, yc), ghost=True)         # rest ghost of the right panel
        face(diamond(acx, yc)); face(rf)           # left (fixed) + right (deformed)
        ax.text(acx, yc, r"$i$", ha="center", va="center", fontsize=FS_MICRO, color=GREY, zorder=4)
        ax.text(*rf.mean(0), r"$k$", ha="center", va="center", fontsize=FS_MICRO, color=GREY, zorder=4)
        ax.plot([P1[0], P2r[0]], [P1[1], P2r[1]], color=GHOST, lw=dw(1.0), ls=(0, (3, 2)), zorder=3)
        ax.plot([P1[0], P2[0]], [P1[1], P2[1]], color=TEAL, lw=dw(2.6), solid_capstyle="round",
                zorder=5)
        ax.scatter([P1[0], P2[0]], [P1[1], P2[1]], s=dA(20), color=TEAL, edgecolor="white",
                   linewidths=dw(0.7), zorder=6)   # the two hinge CORNERS
        # A dimension needs two heads and two witness lines; over a ~6 bp span those collide into an
        # unreadable knot. A single motion arrow from the rest corner to the moved one says the same
        # thing and cannot tangle.
        if mode == "opening":
            yb = yc - d - 0.30
            ax.annotate("", xy=(P2[0], yb), xytext=(P2r[0], yb),
                        arrowprops=dict(arrowstyle="-|>", color=INK, lw=dw(1.3),
                                        mutation_scale=dw(8)))
            ax.text((P2r[0] + P2[0]) / 2, yb - 0.08, r"$a$", ha="center", va="top",
                    fontsize=FS_MICRO, color=INK)
        elif mode == "shear":
            xd = bcx + d + 0.34
            ax.annotate("", xy=(xd, P2[1]), xytext=(xd, yc),
                        arrowprops=dict(arrowstyle="-|>", color=INK, lw=dw(1.3),
                                        mutation_scale=dw(8)))
            # above the arrow, not beside it: beside it the label reached cx+1.64 against a
            # 1.60 half-column and clipped into the neighbouring mode
            ax.text(xd, yc + 0.14, r"$s$", ha="center", va="bottom",
                    fontsize=FS_MICRO, color=INK)
        else:                                      # relative rotation arc at the hinge
            L = 0.55
            ax.plot([P2r[0], P2r[0]], [P2r[1], P2r[1] + L], color=GHOST, lw=dw(1.0),
                    ls=(0, (3, 2)), zorder=4)
            ax.plot([P2r[0], P2r[0] - L * np.sin(th)], [P2r[1], P2r[1] + L * np.cos(th)],
                    color=TEAL, lw=dw(1.5), zorder=4)
            ax.add_patch(MplArc(P2r, 2 * (L + 0.05), 2 * (L + 0.05), angle=0, theta1=90,
                                theta2=90 + np.degrees(th), color=INK, lw=dw(1.1), zorder=4))
            ax.text(P2r[0] - 0.24, P2r[1] + L + 0.16, r"$\vartheta$", ha="center", va="center",
                    fontsize=FS_MICRO, color=INK)
        ax.text(cx, yspan - 0.75, names[mode], ha="center", va="center", fontsize=FS_MICRO,
                fontweight="bold", color=INK)


def detail_energy(ax, S):
    """c: real energy history of the loading -- hinge energy rises (rotation-dominated) with
    tension/shear/rotation contributions, as the external load is ramped up."""
    h = S['energy_hist']
    lam = h['lam']
    top = h['total'].max()
    # shear & tension are tiny next to rotation -> lift them to a visible band (schematic) so the
    # reader sees all three contributions; only their SHAPE matters here.
    # spread the two small components further apart than physics would: at 30 bp of plot height the
    # inline labels need ~0.3*top of separation to not collide. Only their SHAPE is meaningful here.
    sh = h['shear'] / max(h['shear'].max(), 1e-12) * 0.56 * top
    tn = h['stretch'] / max(h['stretch'].max(), 1e-12) * 0.20 * top
    # Rotation IS very nearly the total, so it is the filled band rather than a line; the two small
    # components ride on top of it. Filling to h['rot'] (not h['total']) is the honest version.
    ax.fill_between(lam, 0, h['rot'], facecolor=CREAM, edgecolor="none", zorder=1)
    ax.plot(lam, h['rot'], color=GREY, lw=dw(1.6), zorder=4)          # the band's own thin outline
    ax.plot(lam, sh, color=TEAL, lw=dw(2.0), zorder=6)
    ax.plot(lam, tn, color=ORANGE, lw=dw(2.2), zorder=6)
    ax.set_xlim(0, 1); ax.set_ylim(0, top * 1.16)
    ax.set_xlabel(r"load factor $\zeta$", fontsize=FS_LEADER, color=GREY, labelpad=1.5)
    # the numeric ticks say nothing here (schematic axis) -- label the three curves inline instead
    ax.set_xticks([]); ax.set_yticks([])
    # figstyle hides the top/right spines, which left this plot with no defined upper edge and made
    # it run into the formula above. A full thin frame is the fix.
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color(GREY); sp.set_linewidth(0.5)
    import matplotlib.patheffects as pe
    halo = [pe.withStroke(linewidth=dw(2.4), foreground="white")]   # the curves pass under these
    for ser, col, name, va, dy in ((h['rot'], GREY, "rotation", "bottom", 0.03),
                                   (sh, TEAL, "shear", "bottom", 0.03),
                                   (tn, TENSION_LB, "tension", "top", -0.04)):
        ax.text(0.97, ser[-1] + dy * top, name, ha="right", va=va, fontsize=FS_MICRO,
                color=col, zorder=7, path_effects=halo)


def detail_damage_curve(ax):
    """d: simplified training curve -- the hinge DAMAGE (physical) loss falling as the optimizer runs.
    Schematic monotone decay; the point is the downward TREND, in the panel-c plot's clean style."""
    ep = np.linspace(0.0, 1.0, 120)
    dmg = 0.10 + 0.90 * np.exp(-4.3 * ep)                  # normalized damage: high -> settles low
    ax.fill_between(ep, 0, dmg, color=DMG_FILL, zorder=1)
    ax.plot(ep, dmg, color=PHYS_COLOR, lw=dw(2.4), zorder=5)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.16)
    ax.set_xlabel("training epoch", fontsize=FS_LEADER, color=GREY, labelpad=1.5)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():                # same frame as the panel-c plot; they sit adjacent
        sp.set_visible(True); sp.set_color(GREY); sp.set_linewidth(0.5)
    ax.text(0.035, 1.10, r"hinge damage $\downarrow$", ha="left", va="top", fontsize=FS_MICRO,
            color=PHYS_COLOR)
    # superimpose a REAL hinge render, damaged (high von-Mises) regions in RED
    png = _damage_hinge_png()
    if png and os.path.exists(png):
        import matplotlib.image as mpimg
        img = mpimg.imread(png)
        a = img[..., 3] if img.shape[-1] == 4 else np.ones(img.shape[:2])
        ys, xk = np.where(a > 0.02)
        img = img[ys.min():ys.max() + 1, xk.min():xk.max() + 1]
        iax = ax.inset_axes([0.45, 0.34, 0.55, 0.62]); iax.imshow(img); iax.axis("off")


def _damage_hinge_png(out="data/outputs/hinge_top_damage.png"):
    """Top view of a real CalculiX hinge RVE, coloured by the RED damage ramp (ENERGY_CMAP von Mises)
    so the damaged ligament regions read RED. Idempotent."""
    if os.path.exists(out):
        return out
    npz = "data/outputs/hinge_frames_w1p5.npz"
    if not os.path.exists(npz):
        return None
    from nff_paper.figures import render_hinge_3d as R
    d = np.load(npz)
    k = int(np.argmin(np.abs(d['damage'] - 0.72)))          # well-damaged frame -> prominent red
    old_cmap = R.STRESS_CMAP
    R.STRESS_CMAP = ENERGY_CMAP                              # cream -> orange -> red -> dark red
    try:
        R.render(d['xyz'], d['conn'], d['disp'][k], float(d['w_lig']), float(d['thickness']), 30.0,
                 out, elev=90, azim=-90, vm=d['vm'][k], vmin=0.0,
                 vmax=float(np.percentile(d['vm'][k], 99)), scale=False)
    finally:
        R.STRESS_CMAP = old_cmap
    return out



# ══════════════════════════════════════════════════════════════════════════════
# Compose
# ══════════════════════════════════════════════════════════════════════════════

# ── Layout, in bp on the printed page (see FIG_W_BP). Row 1 = a b c, row 2 = d e. ──
# Everything vertical is DERIVED from the canvas height, so changing the aspect above rescales the
# plate without reopening any of the placement below.
CW1, CW2 = 110.4, 96.0                      # card widths, row 1 / row 2
LANE_Y, LANE_H = 11.0, 16.0                 # backward-pass lane (centre, height)
FY_BOT, R2_Y = 36.0, 40.0                   # container floor, row-2 card bottom
CH = (FIG_H_BP - 77.0) / 2                  # card height; 77 = every band that is not a card
_EXTRA = CH - 160.25                        # growth over the layout the bands were tuned at
FY_MID = R2_Y + CH + 5.75                   # container step: bar bottom, right of the leg
WRAP_Y = FY_MID + 5.0                       # the c -> d wrap runs inside the bar
R1_Y = FY_MID + 14.25                       # row-1 card bottom
FY_TOP = R1_Y + CH + 4.0
FX0, FX1, FXD1, X_LOSS = 143.4, 432.2, 251.4, 287.8
CARDS = {"a": (5.0, R1_Y, CW1), "b": (149.4, R1_Y, CW1), "c": (315.8, R1_Y, CW1),
         "d": (149.4, R2_Y, CW2), "e": (330.2, R2_Y, CW2)}
APAD = 4.0                                  # axes inset inside a card
DET_OFF, DET_H = 3.0, 42.0 + 0.45 * _EXTRA  # detail band, from the card bottom
VIS_H = 72.0 + 0.55 * _EXTRA                # visual band height
FORM_LO = DET_OFF + DET_H + 5.0             # two-line cards (c, d): gloss line
FORM_OFF = FORM_LO + 10.0                   # single formula line
FORM_HI = FORM_LO + 15.0                    # two-line cards: formula line
VIS_OFF = DET_OFF + DET_H + 30.0
TITLE_OFF = VIS_OFF + VIS_H + 7.0           # letter + title strip, inside the card
# Card a's detail is a ZOOM of one cut. Its frame and the marker drawn on the sheet share an aspect
# ratio, which is most of what ties them together -- so derive both from one place.
ZOOM_IX, ZOOM_IY, ZOOM_TY = 1.0, 1.5, 1.0
ZOOM_ASPECT = (CW1 - 2 * APAD - 2 * ZOOM_IX) / (DET_H - ZOOM_IY - ZOOM_TY)

_vis = lambda k: [fx(CARDS[k][0] + APAD), fy(CARDS[k][1] + VIS_OFF),
                  fx(CARDS[k][2] - 2 * APAD), fy(VIS_H)]
_det = lambda k, ix=0.0, iy=0.0, ty=0.0: [fx(CARDS[k][0] + APAD + ix),
                                          fy(CARDS[k][1] + DET_OFF + iy),
                                          fx(CARDS[k][2] - 2 * APAD - 2 * ix),
                                          fy(DET_H - iy - ty)]
_cx = lambda k: fx(CARDS[k][0] + CARDS[k][2] / 2)
_fy_in = lambda k, off: fy(CARDS[k][1] + off)


def build_figure(S, out_base, audit=False):
    apply_charter(**_PLATE_RC)
    fig = plt.figure(figsize=(FIG_W_BP / 72.0, FIG_H_BP / 72.0))

    titles = {"a": "cut pattern", "b": "hinge energy", "c": "kinematic deployment",
              "d": "deployed vs target", "e": "optimised pattern"}

    # Forward-model container: a stepped polygon. The top bar reaches down past row 1 so the c->d
    # wrap arrow runs inside it, which leaves the row-2 cell right of the leg outside -- that is
    # where card e sits. Column 1 of row 2 is left empty for the backward-pass riser into card a.
    fig.add_artist(PathPatch(_rounded_path([(FX0, FY_TOP), (FX1, FY_TOP), (FX1, FY_MID),
                                            (FXD1, FY_MID), (FXD1, FY_BOT), (FX0, FY_BOT)], 5.0),
                             transform=fig.transFigure, facecolor=FWD_FILL, edgecolor=GREY,
                             lw=dw(1.3), zorder=-3))
    fig.text(fx((FX0 + FX1) / 2), fy(FY_TOP + 6.0), "differentiable forward model",
             ha="center", va="center", fontsize=FS_TITLE, style="italic", color=GREY)

    # cards
    for k, (kx, ky, kw) in CARDS.items():
        edge = R_COLOR if k == "a" else (GREY if k in ("b", "c", "d") else INNER_EDGE)
        fig.add_artist(FancyBboxPatch((fx(kx), fy(ky)), fx(kw), fy(CH),
                                      boxstyle=f"round,pad={fx(1.5)},rounding_size={fx(4)}",
                                      transform=fig.transFigure, facecolor=BOX_FILL, edgecolor=edge,
                                      lw=dw(2.0) if k == "a" else dw(1.2), zorder=-2))
        panel_letter(fig, k, x=fx(kx + 5.0), y=_fy_in(k, TITLE_OFF), ha="left", va="center")
        fig.text(_cx(k), _fy_in(k, TITLE_OFF), titles[k], ha="center", va="center", fontsize=FS_TITLE)

    axv = {k: fig.add_axes(_vis(k)) for k in CARDS}
    panel_cut(axv["a"], S['geom_init'], "design", S)
    panel_rom(axv["b"], S)
    sc, norm = panel_deploy(axv["c"], S)
    panel_target(axv["d"], S)
    panel_cut(axv["e"], S['geom_opt'], "output")

    # ── a: the two design-parameter blocks of eq:design-params, one colour each ──
    _colored_line(fig, _cx("a"), _fy_in("a", FORM_OFF),
                  [(r"$\boldsymbol{\theta}=(\,$", INK), (r"$\boldsymbol{r}$", R_SLIDER),
                   (r"$,\ $", INK), (r"$\boldsymbol{\tau}$", TAU_SLIDER), (r"$,\ $", INK),
                   (r"$\{\boldsymbol{g}_h\}$", G_COLOR), (r"$\,)$", INK)],
                  fs=FS_PANEL, box=True)
    # Card a's detail is a magnified view of the cut framed on the sheet above. The tie is carried by
    # three strands, none of which crosses the theta box between them: matched frames (same aspect,
    # colour, weight, radius), the shared A/B letters, and the dashed A-B here being the solid line
    # there. A drawn connector would need a three-turn route through the 18.7 bp channels beside the
    # formula and would read as a stray rule.
    detail_theta(fig.add_axes(_det("a", ix=ZOOM_IX, iy=ZOOM_IY, ty=ZOOM_TY)), S)
    _zx0 = CARDS["a"][0] + APAD + ZOOM_IX
    _zy0 = CARDS["a"][1] + DET_OFF + ZOOM_IY
    _zx1, _zy1 = _zx0 + (CW1 - 2 * APAD - 2 * ZOOM_IX), _zy0 + (DET_H - ZOOM_IY - ZOOM_TY)
    fig.add_artist(PathPatch(_rounded_path([(_zx0, _zy0), (_zx1, _zy0), (_zx1, _zy1), (_zx0, _zy1)],
                                           2.5), transform=fig.transFigure, facecolor="none",
                             edgecolor=INK, lw=dw(1.1), zorder=9))
    fig.text(fx(_zx0 + 3.5), fy(_zy1 - 4.5), "zoom", ha="left", va="center", fontsize=FS_MICRO,
             color=GREY, style="italic", zorder=10)

    # ── b: the reduced hinge kinematics eq:hinge-strain, and the stored energy they enter. Two
    #      lines like card c -- the measures on top, the energy below -- with a short riser between
    #      them, so the direction of the map is DRAWN rather than left to the reader. The argument
    #      order is the manuscript's, W(eps; g): the deformation first, the hinge design second.
    _formula(fig, _cx("b"), _fy_in("b", FORM_HI + 2.0),
             r"$\boldsymbol{\varepsilon}_h(\boldsymbol{q})=(a_h,\,s_h,\,\vartheta_h)$",
             TEAL, FS_FORMULA)
    _fig_arrow(fig, (_cx("b"), _fy_in("b", FORM_LO + 8.2)),
               (_cx("b"), _fy_in("b", FORM_LO + 2.8)), GREY, lw=dw(1.1), ms=5.0, shrink=0.0)
    _colored_line(fig, _cx("b"), _fy_in("b", FORM_LO - 2.0),
                  [(r"$\mathcal{W}(\,$", INK), (r"$\boldsymbol{\varepsilon}_h$", TEAL),
                   (r"$\,;\,$", INK), (r"$\boldsymbol{g}_h$", G_COLOR), (r"$\,)$", INK)],
                  fs=FS_SMALL)
    detail_two_faces(fig.add_axes(_det("b")))

    # ── c: equilibrium, eq:equilibrium. Pi is NOT expanded here -- eq:equilibrium defines it, and
    #      the card is about the deployed state the minimisation returns.
    _formula(fig, _cx("c"), _fy_in("c", FORM_OFF),
             r"$\boldsymbol{q}^\star(\zeta;\boldsymbol{\theta})="
             r"\arg\min_{\boldsymbol{q}\in\mathcal{C}}\ \Pi$", INK, FS_FORMULA)
    detail_energy(fig.add_axes(_det("c", ix=6.0, iy=6.5, ty=3.5)), S)
    # inset far enough from the card edge that the "low"/"high" tick labels stay inside it
    cax = fig.add_axes([fx(CARDS["c"][0] + CW1 - APAD - 22.0), _fy_in("c", VIS_OFF + VIS_H * 0.24),
                        fx(4.5), fy(VIS_H * 0.50)])
    cb = fig.colorbar(sc, cax=cax, orientation="vertical")
    cb.set_ticks([norm.vmin, norm.vmax]); cb.set_ticklabels(["low", "high"])
    cb.ax.yaxis.set_ticks_position("right"); cb.ax.tick_params(length=0, labelsize=FS_MICRO)
    cb.ax.set_title(r"hinge $\mathcal{W}$", fontsize=FS_MICRO, color=INK, pad=2)

    # ── d: the design objective, eq:objective ──
    _colored_line(fig, _cx("d"), _fy_in("d", FORM_OFF),
                  [(r"$\mathcal{L}=$", INK), (r"$\mathcal{L}_{\mathrm{geom}}$", GEOM_COLOR),
                   (r"$\,+\,$", INK), (r"$\mathcal{L}_{\mathrm{phys}}$", PHYS_COLOR)],
                  fs=FS_SMALL, box=True)
    detail_damage_curve(fig.add_axes(_det("d", ix=6.0, iy=6.5)))

    _formula(fig, _cx("e"), _fy_in("e", FORM_OFF), r"$\boldsymbol{\theta}^\star$  optimised design",
             INNER_EDGE, FS_FORMULA)
    fig.text(_cx("e"), _fy_in("e", DET_OFF + DET_H / 2), "ready for\nreal-world\nfabrication",
             ha="center", va="center", fontsize=FS_SMALL, color="#8a3a00", style="italic")

    # forward chevrons (row 1) and the c -> d wrap, which stays inside the container bar
    chev_y = _fy_in("a", VIS_OFF + VIS_H / 2)
    for xch in (129.4, 287.8):
        fig.text(fx(xch), chev_y, "❯", ha="center", va="center", fontsize=FS_PANEL, color=GREY)
    _elbow_arrow(fig, [(371.0, R1_Y - 2.0), (371.0, WRAP_Y), (197.4, WRAP_Y),
                       (197.4, R2_Y + CH + 2.0)], GREY, lw=dw(2.0), ms=dw(15))

    # loss node between d and e, on the row-2 visual centreline
    ly_bp = R2_Y + VIS_OFF + VIS_H / 2
    fig.add_artist(FancyBboxPatch((fx(X_LOSS - 13), fy(ly_bp - 10)), fx(26), fy(20),
                                  boxstyle=f"round,pad={fx(1)},rounding_size={fx(3)}",
                                  transform=fig.transFigure, facecolor="white", edgecolor=RED,
                                  lw=dw(1.6), zorder=6))
    fig.text(fx(X_LOSS), fy(ly_bp), r"$\mathcal{L}$", ha="center", va="center", fontsize=FS_PANEL,
             color=RED, zorder=7)
    fig.text(fx(X_LOSS), fy(ly_bp + 16), "loss", ha="center", va="center", fontsize=FS_LEADER,
             color=RED)
    _fig_arrow(fig, (fx(CARDS["d"][0] + CW2 + 2), fy(ly_bp)), (fx(X_LOSS - 15), fy(ly_bp)),
               GREY, dw(2.0), ms=dw(15))
    _fig_arrow(fig, (fx(X_LOSS + 15), fy(ly_bp)), (fx(CARDS["e"][0] - 2), fy(ly_bp)),
               INK, dw(2.4), ms=dw(15))

    # ── backward pass: an INK lane along the bottom, rising into the theta box (a) ──
    lane_x0, lane_x1 = 27.0, X_LOSS
    fig.add_artist(FancyBboxPatch((fx(lane_x0), fy(LANE_Y - LANE_H / 2)), fx(lane_x1 - lane_x0),
                                  fy(LANE_H),
                                  boxstyle=f"round,pad={fx(1)},rounding_size={fx(8)}",
                                  transform=fig.transFigure, facecolor=BACK_FILL, edgecolor=INK,
                                  lw=dw(1.8), zorder=1))
    for xch in np.linspace(105.0, 272.0, 6):
        fig.text(fx(xch), fy(LANE_Y), "❮", ha="center", va="center", fontsize=FS_SMALL, color=INK,
                 zorder=2)
    # centred on the chevron run, not on the whole lane: centring on the lane (157.4) would pull the
    # caption off the chevrons and dump empty lane-top under the loss riser
    fig.text(fx(188.5), fy(LANE_Y + LANE_H / 2 + 5.5),
             r"backward pass: $\partial\mathcal{L}/\partial\boldsymbol{\theta}$",
             ha="center", va="center", fontsize=FS_TITLE, color=INK, style="italic")
    _fig_arrow(fig, (fx(X_LOSS), fy(ly_bp - 12)), (fx(X_LOSS), fy(LANE_Y + LANE_H / 2)),
               INK, dw(2.4), ms=dw(15))
    # The paper minimises with Adam (sn-article.tex:1004), not plain gradient descent; and eta is
    # already the surrogate network there, so it cannot double as a learning rate.
    bx0, bw = 5.0, 87.0
    fig.add_artist(FancyBboxPatch((fx(bx0), fy(LANE_Y - 9)), fx(bw), fy(18),
                                  boxstyle=f"round,pad={fx(1.5)},rounding_size={fx(4)}",
                                  transform=fig.transFigure, facecolor="white", edgecolor=INK,
                                  lw=dw(1.6), zorder=3))
    fig.text(fx(bx0 + bw / 2), fy(LANE_Y),
             r"$\boldsymbol{\theta}\leftarrow\mathrm{Adam}"
             r"(\boldsymbol{\theta},\,\partial\mathcal{L}/\partial\boldsymbol{\theta})$",
             ha="center", va="center", fontsize=FS_TITLE, color=INK, zorder=4)
    x_riser = CARDS["a"][0] + CW1 / 2
    _fig_arrow(fig, (fx(x_riser), fy(LANE_Y + 10)), (fx(x_riser), fy(R1_Y - 2)), INK, dw(2.6),
               ms=dw(15))
    fig.text(fx(x_riser + 8), fy((LANE_Y + R1_Y) / 2), r"updates $\boldsymbol{\theta}$", ha="left",
             va="center", fontsize=FS_LEADER, color=INK)

    if audit:
        _audit(fig)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_base}.{ext}", dpi=600)     # NO bbox_inches: it would break the print scale
    plt.close(fig)
    print(f"wrote {out_base}.png / .pdf  ({FIG_W_BP:.1f} x {FIG_H_BP:.1f} bp)")


def _fig_arrow(fig, p0, p1, color, lw=None, ms=6.0, shrink=None, zorder=5):
    """shrink: override the 2 pt trimmed off EACH end by default -- pass 0.0 for a short riser,
    which the default would otherwise reduce to a bare arrowhead."""
    kw = {} if shrink is None else dict(shrinkA=shrink, shrinkB=shrink)
    fig.add_artist(FancyArrowPatch(p0, p1, transform=fig.transFigure, arrowstyle="-|>",
                                   mutation_scale=ms, lw=dw(2.0) if lw is None else lw,
                                   color=color, zorder=zorder, **kw))


def _audit(fig):
    """Regression guard: at scale 1.0 an authored point size IS the printed one, so nothing may go
    below 6 pt, and no card formula may overflow its card."""
    fig.canvas.draw()
    sizes = sorted({(round(t.get_fontsize(), 2), t.get_text()[:44])
                    for t in fig.findobj(matplotlib.text.Text) if t.get_text().strip()})
    print("\n-- type audit (printed pt) --")
    for s, t in sizes[:6]:
        print(f"   {s:5.2f}  {t}")
    print(f"   ... {len(sizes)} distinct (size, text) pairs; min {sizes[0][0]:.2f} pt")
    lws = [ln.get_linewidth() for ln in fig.findobj(matplotlib.lines.Line2D)]
    print(f"-- linewidths: {min(lws):.2f} .. {max(lws):.2f} pt" if lws else "-- no lines")
    r = fig.canvas.get_renderer()
    to_bp = lambda px: px / fig.bbox.width * FIG_W_BP
    over = []
    for k, (kx, ky, kw) in CARDS.items():
        # A label overflows when either EDGE leaves the card, not only when the string is wider than
        # the card: "target shape" was 35 bp inside a 88 bp card and still printed clipped.
        cx0 = kx / FIG_W_BP * fig.bbox.width
        cx1 = (kx + kw) / FIG_W_BP * fig.bbox.width
        y0, y1 = (ky / FIG_H_BP * fig.bbox.height, (ky + CH) / FIG_H_BP * fig.bbox.height)
        for t in fig.findobj(matplotlib.text.Text):
            bb = t.get_window_extent(renderer=r)
            xmid, ymid = bb.x0 + bb.width / 2, bb.y0 + bb.height / 2
            if not (y0 < ymid < y1 and cx0 < xmid < cx1):
                continue                                  # not this card's text
            spill = max(cx0 - bb.x0, bb.x1 - cx1)
            if spill > 0.5:                               # sub-half-point is anti-aliasing, not a bug
                over.append((k, f"spills {to_bp(spill):.1f} bp", t.get_text()[:36]))
    print(f"-- overflowing card text: {over if over else 'none'}")
    assert sizes[0][0] >= 6.0, f"sub-6pt text would print illegibly: {sizes[:4]}"



# ══════════════════════════════════════════════════════════════════════════════
# Slide composition — the same five cards in ONE ROW
# ══════════════════════════════════════════════════════════════════════════════
#
# The page plate is 437 x 425 bp, near enough square that a 16:9 body (410 x 188 pt) is
# height-limited by it: it prints half the slide wide and takes its 7 pt type to 3.3. A row uses
# the width the page never had. Authored at 408 bp, so an authored point size is a projected one.
#
# What the row DROPS, and why: the formula band and the detail band under every card. Five cards
# across 408 bp leaves each 68 bp wide against the page's 110, so a formula set to fit would be
# 5 pt on a projector. The equations are not lost -- theta is the parametrisation slide's own
# display, W belongs to the hinge slides -- and this slide is the one place the pipeline is a
# picture rather than an argument. The visuals, the titles, the flow and the backward pass are
# what a row is for.

SL_W_BP, SL_H_BP = slide_bp(aspect=408.0 / 128.0)

SL_CW = 63.4                                 # card width; 5 of them and 4 gaps fill 408 bp
# The wide gap has to hold the loss node AND an arrow either side of it. At 34 it did not: the node
# is 16 bp wide, which left each arrow 5 bp of shaft under a 15 bp arrowhead, so both read as a
# floating head, and the forward-model container's border landed mid-arrow. 49 buys about 9 bp of
# shaft on the left and 12 on the right, paid for with 3 bp off each of the five cards.
SL_GAP, SL_GAP_L = 10.0, 49.0                # plain gap, and the d -> e gap that holds the loss
# The a -> b gap is wider than a plain one because THREE things share it: card a's edge, the
# forward-model container's left border, and the chevron. At 10 the border landed 1 bp from the
# chevron and the glyph was drawn straight over it. 18 leaves the chevron 7 bp of clear space
# either side.
SL_GAP_A = 18.0
SL_M = 2.0                                   # canvas margin
SL_LANE_Y, SL_LANE_H = 8.5, 13.0             # backward-pass lane (centre, height)
SL_CY, SL_CH = 30.0, 82.0                    # card bottom and height
SL_APAD = 3.5                                # axes inset inside a card
SL_TITLE_OFF = SL_CH - 7.0                   # letter + title strip, from the card bottom
SL_VIS_OFF, SL_VIS_H = 3.0, SL_CH - 14.0     # visual band

# Left edges, laid out left to right with the wide gap before e.
_sl_x = {}
_x = SL_M
for _k in "abcde":
    _sl_x[_k] = _x
    _x += SL_CW + (SL_GAP_L if _k == "d" else (SL_GAP_A if _k == "a" else SL_GAP))
SL_CARDS = {k: (_sl_x[k], SL_CY, SL_CW) for k in "abcde"}

SL_X_LOSS = _sl_x["d"] + SL_CW + SL_GAP_L / 2     # loss node, centred in the wide gap
# The container insets 5 bp from the cards it holds, the same on both sides, so neither border
# shares space with a chevron or an arrowhead.
SL_FX0 = _sl_x["b"] - 5.0                          # forward-model container, around b c d
SL_FX1 = _sl_x["d"] + SL_CW + 5.0
SL_FY_BOT, SL_FY_TOP = SL_CY - 5.0, SL_CY + SL_CH + 5.0

# The row a talk meets FIRST holds four cards, not five. At that point neither the hinge energy
# nor equilibrium has been introduced, and the deployment the five-card row shows is a physical
# one, so b and c collapse into a single kinematic-deployment card. The five-card row returns on
# the objective frame, by which point every stage of it has been explained.
SL_KIN_CARDS = "acde"
SL_KIN_TITLE = "kinematic deployment"


def _sl_layout(keys):
    """Left edges and card width for a row of ``keys``, filling the 408 bp measure.

    The gaps are fixed by what each has to hold: a to b carries the container's left border and a
    chevron, d to e carries the loss node and an arrow either side. So the CARDS absorb a dropped
    stage rather than the row leaving a hole, and four cards come out 81.8 bp wide against 63.4.
    """
    gaps = [SL_GAP_L if k == "d" else (SL_GAP_A if k == "a" else SL_GAP) for k in keys[:-1]]
    cw = (SL_W_BP - 2 * SL_M - sum(gaps)) / len(keys)
    x, xs = SL_M, {}
    for i, k in enumerate(keys):
        xs[k] = x
        x += cw + (gaps[i] if i < len(gaps) else 0.0)
    return xs, cw


def _fit_fontsize(fig, labels, avail_bp, fs0, floor=5.6):
    """Largest size at or below ``fs0`` at which every label fits ``avail_bp``."""
    r = fig.canvas.get_renderer()
    fs = fs0
    while fs > floor:
        t = fig.text(0.5, 0.5, max(labels, key=len), fontsize=fs)
        w = t.get_window_extent(renderer=r).width / fig.bbox.width * FIG_W_BP
        t.remove()
        if w <= avail_bp:
            break
        fs -= 0.1
    return round(fs, 1)


def build_figure_slide(S, out_base, kinematic=False):
    """One-row cut of the pipeline plate, authored at the size a 16:9 slide gives it.

    ``kinematic`` drops the hinge-energy card and shows the deployment with no mechanics on it,
    for the frame that meets the pipeline before any stage of it has been explained.
    """
    global FIG_W_BP, FIG_H_BP, _SLIDE
    FIG_W_BP, FIG_H_BP = SL_W_BP, SL_H_BP          # fx/fy and every helper read these
    _SLIDE = True

    # Local, not the module's: the four-card row is a different geometry and nothing outside this
    # call should see it. Every name here shadows its module-level default for this build only.
    keys = SL_KIN_CARDS if kinematic else "abcde"
    _sl_x, SL_CW = _sl_layout(keys)
    SL_CARDS = {k: (_sl_x[k], SL_CY, SL_CW) for k in keys}
    SL_X_LOSS = _sl_x["d"] + SL_CW + SL_GAP_L / 2
    SL_FX0 = _sl_x["b" if "b" in keys else "c"] - 5.0
    SL_FX1 = _sl_x["d"] + SL_CW + 5.0

    vis = lambda k: [fx(SL_CARDS[k][0] + SL_APAD), fy(SL_CARDS[k][1] + SL_VIS_OFF),
                     fx(SL_CW - 2 * SL_APAD), fy(SL_VIS_H)]
    cx = lambda k: fx(SL_CARDS[k][0] + SL_CW / 2)
    yin = lambda k, off: fy(SL_CARDS[k][1] + off)

    apply_charter(**_PLATE_RC)
    fig = plt.figure(figsize=(FIG_W_BP / 72.0, FIG_H_BP / 72.0))

    # A 68 bp card holds about 13 characters at a size that survives a projector, so three of the
    # page's five titles are shortened. The card carries the meaning the words drop: (c) shows the
    # deployment, (d) shows the sheet inside its dashed target, (e) shows the cut pattern again.
    titles = {"a": "cut pattern", "b": "hinge energy", "c": "deployment",
              "d": "vs target", "e": "optimised cut"}
    if kinematic:
        titles["c"] = SL_KIN_TITLE
    titles = {k: titles[k] for k in keys}
    # One size for all five, set by the longest, so the strip reads as one row rather than five
    # decisions. Measured, not guessed: a title that overruns its card runs UNDER the next card's
    # letter and nothing in the build reports it. That is what "deployed vs target" did, bottoming
    # out against the floor and still overflowing, which is why the words above are shorter.
    fs_title = _fit_fontsize(fig, list(titles.values()), SL_CW - 12.5, FS_TITLE, floor=5.6)

    # Forward-model container: a plain rounded box here, not the page's stepped polygon. The step
    # existed to let the c -> d arrow wrap from row 1 into row 2; in a row there is no wrap.
    fig.add_artist(PathPatch(_rounded_path([(SL_FX0, SL_FY_TOP), (SL_FX1, SL_FY_TOP),
                                            (SL_FX1, SL_FY_BOT), (SL_FX0, SL_FY_BOT)], 5.0),
                             transform=fig.transFigure, facecolor=FWD_FILL, edgecolor=GREY,
                             lw=dw(1.3), zorder=-3))
    fig.text(fx((SL_FX0 + SL_FX1) / 2), fy(SL_FY_TOP + 5.5), "differentiable forward model",
             ha="center", va="center", fontsize=FS_TITLE, style="italic", color=GREY)

    for k, (kx, ky, kw) in SL_CARDS.items():
        edge = R_COLOR if k == "a" else (GREY if k in ("b", "c", "d") else INNER_EDGE)
        fig.add_artist(FancyBboxPatch((fx(kx), fy(ky)), fx(kw), fy(SL_CH),
                                      boxstyle=f"round,pad={fx(1.5)},rounding_size={fx(4)}",
                                      transform=fig.transFigure, facecolor=BOX_FILL, edgecolor=edge,
                                      lw=dw(2.0) if k == "a" else dw(1.2), zorder=-2))
        # The letter owns the left of the strip and the title centres in the rest of it. Centring
        # the title on the whole card, as the page does, runs it under the letter at this width.
        # With the letters off there is no strip to dodge, so the title takes the whole card.
        panel_letter(fig, k, x=fx(kx + 3.5), y=yin(k, SL_TITLE_OFF), ha="left", va="center")
        _tx = (kx + 9.0 + (SL_CW - 12.5) / 2) if panel_letters_on() else (kx + SL_CW / 2)
        fig.text(fx(_tx), yin(k, SL_TITLE_OFF), titles[k],
                 ha="center", va="center", fontsize=fs_title)

    # Card c reserves the bottom of its visual band for the colour key, so the sheet is fitted into
    # what is left and the two cannot meet. A vertical bar inside the band, as the page has it, is
    # drawn straight over the deployed sheet at this width: the page affords it a 102 bp card and
    # 22 bp of inset, and 68 bp does not.
    SL_CBAR_H = 0.0 if kinematic else 11.0
    axv = {k: fig.add_axes(vis(k)) for k in SL_CARDS if k != "c"}
    axv["c"] = fig.add_axes([fx(SL_CARDS["c"][0] + SL_APAD),
                             yin("c", SL_VIS_OFF + SL_CBAR_H),
                             fx(SL_CW - 2 * SL_APAD), fy(SL_VIS_H - SL_CBAR_H)])
    panel_cut(axv["a"], S['geom_init'], "design", S)
    if not kinematic:
        panel_rom(axv["b"], S)
    sc, norm = panel_deploy(axv["c"], S, kinematic=kinematic)
    panel_target(axv["d"], S)
    panel_cut(axv["e"], S['geom_opt'], "output")

    # The hinge-energy key, horizontal in the strip reserved above, with its ends labelled beside
    # the bar rather than under it: 5.5 pt text needs 5.5 bp of height and the strip has 11 for
    # bar and label together. No "hinge W" title either, for the same reason the page can afford
    # one and this cannot.
    if not kinematic:
        _cb_w, _cb_y = 30.0, SL_VIS_OFF + 4.5
        _cb_x = SL_CARDS["c"][0] + SL_CW / 2 - _cb_w / 2
        cax = fig.add_axes([fx(_cb_x), yin("c", _cb_y), fx(_cb_w), fy(3.0)])
        cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
        cb.set_ticks([]); cb.outline.set_linewidth(dw(0.8))
        for _x, _lab, _ha in ((_cb_x - 1.5, "low", "right"),
                              (_cb_x + _cb_w + 1.5, "high", "left")):
            fig.text(fx(_x), yin("c", _cb_y + 1.5), _lab, ha=_ha, va="center",
                     fontsize=FS_MICRO, color=GREY)

    # forward chevrons, one in every gap inside the row except the wide one before e, which
    # carries the loss node and its two arrows instead
    chev_y = yin("a", SL_VIS_OFF + SL_VIS_H / 2)
    for k in keys[:-2]:
        _g = SL_GAP_A if k == "a" else SL_GAP
        fig.text(fx(_sl_x[k] + SL_CW + _g / 2), chev_y, "❯", ha="center", va="center",
                 fontsize=FS_PANEL, color=GREY)

    # loss node between d and e
    fig.add_artist(FancyBboxPatch((fx(SL_X_LOSS - 8), fy(chev_y * FIG_H_BP - 6.5)), fx(16), fy(13),
                                  boxstyle=f"round,pad={fx(1)},rounding_size={fx(3)}",
                                  transform=fig.transFigure, facecolor="white", edgecolor=RED,
                                  lw=dw(1.6), zorder=6))
    fig.text(fx(SL_X_LOSS), chev_y, r"$\mathcal{L}$", ha="center", va="center", fontsize=FS_PANEL,
             color=RED, zorder=7)
    fig.text(fx(SL_X_LOSS), fy(chev_y * FIG_H_BP + 12.0), "loss", ha="center", va="center",
             fontsize=FS_LEADER, color=RED)
    # Both arrows are measured from the loss node's own edges, not from a fixed offset, so a change
    # to the gap cannot silently shorten them back to a bare arrowhead.
    _loss_l, _loss_r = SL_X_LOSS - 8.0, SL_X_LOSS + 8.0
    _fig_arrow(fig, (fx(_sl_x["d"] + SL_CW + 7.0), chev_y), (fx(_loss_l - 1.5), chev_y),
               GREY, dw(2.0), ms=dw(13))
    _fig_arrow(fig, (fx(_loss_r + 1.5), chev_y), (fx(_sl_x["e"] - 2.0), chev_y),
               INK, dw(2.4), ms=dw(13))

    # ── backward pass: an INK lane along the bottom, rising into card a ──
    lane_x0, lane_x1 = _sl_x["a"] + SL_CW / 2 - 6.0, SL_X_LOSS
    fig.add_artist(FancyBboxPatch((fx(lane_x0), fy(SL_LANE_Y - SL_LANE_H / 2)),
                                  fx(lane_x1 - lane_x0), fy(SL_LANE_H),
                                  boxstyle=f"round,pad={fx(1)},rounding_size={fx(6)}",
                                  transform=fig.transFigure, facecolor=BACK_FILL, edgecolor=INK,
                                  lw=dw(1.8), zorder=1))
    for xch in np.linspace(lane_x0 + 22.0, lane_x1 - 12.0, 6):
        fig.text(fx(xch), fy(SL_LANE_Y), "❮", ha="center", va="center", fontsize=FS_SMALL,
                 color=INK, zorder=2)
    fig.text(fx((lane_x0 + lane_x1) / 2), fy(SL_LANE_Y + SL_LANE_H / 2 + 5.0),
             r"backward pass: $\partial\mathcal{L}/\partial\boldsymbol{\theta}$",
             ha="center", va="center", fontsize=FS_TITLE, color=INK, style="italic")
    _fig_arrow(fig, (fx(SL_X_LOSS), fy(chev_y * FIG_H_BP - 9.0)),
               (fx(SL_X_LOSS), fy(SL_LANE_Y + SL_LANE_H / 2)), INK, dw(2.4), ms=dw(15))
    x_riser = _sl_x["a"] + SL_CW / 2
    _fig_arrow(fig, (fx(x_riser), fy(SL_LANE_Y + SL_LANE_H / 2)), (fx(x_riser), fy(SL_CY - 2)),
               INK, dw(2.6), ms=dw(15))
    fig.text(fx(x_riser + 5), fy((SL_LANE_Y + SL_CY) / 2 + 1.0),
             r"$\boldsymbol{\theta}\leftarrow\mathrm{Adam}$", ha="left", va="center",
             fontsize=FS_LEADER, color=INK)

    for ext in ("png", "pdf"):
        fig.savefig(f"{out_base}.{ext}", dpi=600)
    plt.close(fig)
    print(f"wrote {out_base}.png / .pdf  ({FIG_W_BP:.1f} x {FIG_H_BP:.1f} bp)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", default="overview")
    ap.add_argument("--out", default="data/outputs/pipeline_overview")
    ap.add_argument("--audit", action="store_true", help="report printed type sizes and overflow")
    ap.add_argument("--slide", action="store_true",
                    help="one-row layout authored for a 16:9 slide instead of the page plate")
    ap.add_argument("--kinematic", action="store_true",
                    help="slide row with the hinge-energy card dropped and the deployment drawn "
                         "with no mechanics on it, for the first sight of the pipeline")
    args = ap.parse_args()
    S = extract_spine(args.config_name)
    if args.slide:
        build_figure_slide(S, args.out, kinematic=args.kinematic)
    else:
        build_figure(S, args.out, audit=args.audit)


if __name__ == "__main__":
    main()
