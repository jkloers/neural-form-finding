"""Nature-style single-plate overview of the differentiable closed-kirigami pipeline.

Card-based training loop: each step is a boxed CARD (visual on top, formula + a small explanatory
diagram below). The backward pass points straight at the THETA box (step a).

    [ a : theta ]  ->  | b ROM  ->  c DEPLOY  ->  d vs TARGET |  -> L  ->  [ e : optimized cut ]
         ^             +----------- differentiable forward model --------+
         +====  backward pass:  theta <- theta - eta dL/dtheta  <========== L

Real geometry from a trained closed_les run; rendered clean/schematic, Princeton palette (no blue).

    JAX_PLATFORMS=cpu conda run -n kgnn_mac \
        python nff/scripts/figures/plot_pipeline_overview.py --config-name overview
"""

import os
import glob
import pickle
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
ORANGE = "#F58025"   # paper / cut pattern / r
INK    = "#1A1A1A"   # cut / edges / text / backward pass
RED    = "#D62828"   # loads / target / total loss
GREY   = "#6C757D"   # clamp / secondary
TEAL   = "#2A9D8F"   # hinges / physical loss
PURPLE = "#6A4C93"   # boundary sliders s
GREEN  = "#1F8A4C"   # Princeton green: deployed boundary points / geometric loss
CREAM  = "#FDF3E7"
VIVID      = ORANGE      # outer physical cut patterns (a, e)
LIGHT      = "#FBDDB6"   # inner-panel fill (b, c, d)
INNER_EDGE = "#C77B3B"   # inner-panel outline (thin)
FACE_GREY  = "#DEDBD5"   # muted schematic faces (ROM two-faces detail)
FACE_EDGE  = "#9A968E"
BOX_FILL   = "#F7F5F2"   # card fill
FWD_FILL   = "#F1F3F5"   # forward-model container
BACK_FILL  = "#ECECEC"   # backward-pass lane fill
R_COLOR, S_COLOR = ORANGE, PURPLE
GEOM_COLOR, PHYS_COLOR = GREEN, "#C1272D"   # physical / hinge-damage loss -> red
CUT_DIM    = "#7A7A7A"   # dim ROM cut centerlines on panel a (uniform aspect)
DMG_FILL   = "#F7DAD9"   # light red fill under the damage-loss training curve

ENERGY_CMAP = LinearSegmentedColormap.from_list("hingeW", [CREAM, "#F9B266", ORANGE, RED, "#7A1010"])


def apply_charter():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": INK, "axes.linewidth": 0.8,
        "xtick.color": GREY, "ytick.color": GREY,
        "text.color": INK, "axes.labelcolor": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 12, "savefig.facecolor": "white",
    })


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
    from nff.closed.setup import (build_closed_initial_state, init_closed_les_params,
                                  build_surrogate_energy)
    from nff.closed.deploy import _deployed_hinge_xy, _boundary_cloud, _fit_circle
    from nff.topology.closed_builder_jax import solve_cut_vertices_jax, boundary_flat_from_logits
    from nff.topology.cut_pattern import build_cut_geometry

    cfg = load_and_parse_config(f"data/configs/closed/{config_name}.yaml")
    state, tess = build_closed_initial_state(cfg)
    params, sf = init_closed_les_params(cfg)
    bond_energy, *_ = build_surrogate_energy(cfg, sf, state, params)

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
    A, x', x, B (so it matches the cut spotlighted on panel a). Rotated so the cut A-B (long diagonal)
    is horizontal and centred on the void; returns the corners A, B (cut) and x, xp (sliders) + tiles."""
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

def _draw_panels(ax, nodes, facecolor=LIGHT, edgecolor=INK, alpha=1.0, lw=1.3, zorder=5):
    pc = PatchCollection([MplPolygon(nodes[i], closed=True) for i in range(nodes.shape[0])],
                         facecolor=facecolor, edgecolor=edgecolor, linewidths=lw, alpha=alpha,
                         zorder=zorder, joinstyle="round")
    ax.add_collection(pc)
    return pc


def _fit_view(ax, pts, pad=0.12):
    mn, mx = pts.min(0), pts.max(0)
    c, r = (mn + mx) / 2, (mx - mn).max() / 2 * (1 + pad)
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_aspect("equal")
    ax.axis("off")


def _shapely_to_patch(poly, **kw):
    polys = list(poly.geoms) if poly.geom_type.startswith("Multi") else [poly]
    verts, codes = [], []
    for p in polys:
        for ring in [p.exterior, *p.interiors]:
            xy = np.asarray(ring.coords)
            verts.extend(xy)
            codes.extend([MplPath.MOVETO] + [MplPath.LINETO] * (len(xy) - 2) + [MplPath.CLOSEPOLY])
    return PathPatch(MplPath(verts, codes), **kw)


def _leader(ax, pt, textpt, label, color, fontsize=9.5):
    ax.annotate(label, xy=pt, xytext=textpt, fontsize=fontsize, color=color, ha="center", va="center",
                arrowprops=dict(arrowstyle="-", color=color, lw=1.0), zorder=12,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, lw=0.8, alpha=0.95))


def _moment_arrow(ax, center, radius, color, ccw=True, lw=2.4):
    """Clean circular moment symbol: ~300deg arc + a proportional tangential arrowhead."""
    a0, a1 = (120.0, 400.0) if ccw else (60.0, -220.0)
    th = np.radians(np.linspace(a0, a1, 80))
    pts = center + radius * np.c_[np.cos(th), np.sin(th)]
    ax.plot(pts[:, 0], pts[:, 1], color=color, lw=lw, zorder=11, solid_capstyle="round")
    e = th[-1]
    tip = center + radius * np.array([np.cos(e), np.sin(e)])
    tang = np.array([-np.sin(e), np.cos(e)]) * (1 if ccw else -1)     # travel direction
    radial = np.array([np.cos(e), np.sin(e)])
    hl, hw = radius * 0.55, radius * 0.42
    base = tip - tang * hl
    ax.add_patch(MplPolygon([tip, base + radial * hw / 2, base - radial * hw / 2],
                            closed=True, color=color, zorder=11))


def _formula(fig, x, y, text, color=INK, fs=11):
    fig.text(x, y, text, ha="center", va="center", fontsize=fs, color=color, zorder=8,
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=color, lw=1.0, alpha=0.97))


def _colored_line(fig, xc, y, parts, fs=11, box=False):
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
        h = 1.7 * fs / (fig.get_size_inches()[1] * 72.0)
        fig.add_artist(FancyBboxPatch((x0 - 0.008, y - h), total + 0.016, 2 * h,
                                      boxstyle="round,pad=0.002,rounding_size=0.006",
                                      transform=fig.transFigure, facecolor="white", edgecolor=INK,
                                      lw=1.0, zorder=7))
    x = x0
    for tx, w in zip(texts, widths):
        tx.set_position((x, y))
        x += w


# ══════════════════════════════════════════════════════════════════════════════
# Card visuals
# ══════════════════════════════════════════════════════════════════════════════

def _seg_dist(p, a, b):
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-12), 0.0, 1.0)
    return float(np.linalg.norm(p - (a + t * ab)))


def _clear_hand_dir(P, segs, reach, bounds):
    """Pick the offset direction (from a fan around P) that lands the hand FARTHEST from any cut and
    stays inside the sheet — so the pointing hand sits on clear paper, never on a cut."""
    minx, miny, maxx, maxy = bounds
    mx, my = 0.04 * (maxx - minx), 0.04 * (maxy - miny)
    P = np.asarray(P, float)
    best, best_d = np.array([0.0, -1.0]), -1.0
    for ang in np.linspace(0, 2 * np.pi, 16, endpoint=False):
        dirv = np.array([np.cos(ang), np.sin(ang)])
        pos = P + dirv * reach
        if not (minx + mx < pos[0] < maxx - mx and miny + my < pos[1] < maxy - my):
            continue
        d = min((_seg_dist(pos, a, b) for a, b in segs), default=1e9)
        if d > best_d:
            best_d, best = d, dirv
    return best


def _drag_hand(ax, P, hand_dir, reach, fs=16, color=INK):
    """Playful 'grab & slide me' cue: a pointing-hand dingbat set off from a slider knob P along
    hand_dir, TILTED so its finger points back AT the knob at the offset angle (as if to drag it)."""
    hand_dir = np.asarray(hand_dir, float); hand_dir = hand_dir / (np.linalg.norm(hand_dir) + 1e-12)
    off = np.asarray(P, float) + hand_dir * reach
    ang = np.degrees(np.arctan2(-hand_dir[1], -hand_dir[0])) - 90.0     # ☝ finger (+y) -> toward P
    ax.text(off[0], off[1], "☝", fontsize=fs, rotation=ang, rotation_mode="anchor",
            ha="center", va="center", color=color, zorder=13)


def panel_cut(ax, geom, kind, S=None):
    ax.add_patch(_shapely_to_patch(geom['sheet'], facecolor=VIVID, edgecolor="none", zorder=2))
    if kind == 'design':
        # ROM view: EVERY cut is a thin dark centerline (one uniform aspect); one is emphasized below.
        for p0, p1 in geom['centerlines']:
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=CUT_DIM, lw=1.0,
                    solid_capstyle="round", zorder=3)
    else:                                                       # panel e: the real cut pattern (slots)
        ax.add_patch(_shapely_to_patch(geom['cuts'], facecolor=INK, edgecolor="none", zorder=3))
    minx, miny, maxx, maxy = geom['sheet'].bounds
    ax.set_xlim(minx, maxx); ax.set_ylim(miny, maxy)
    ax.set_aspect("equal"); ax.axis("off")
    W_, H_ = maxx - minx, maxy - miny
    if kind == 'design':
        kl, kh = 0.12 * (W_ / 5.0), 0.062 * (W_ / 5.0)          # small knobs
        cut_segs = [(np.asarray(a), np.asarray(b)) for a, b in geom['centerlines']]
        bounds = (minx, miny, maxx, maxy); reach_h = kl * 2.8   # for the drag-hand cues
        # ONE void-aspect-ratio slider per cut (a single r sets both cut-vertices), void side.
        for xx, xpp, uh in S['all_sliders']:
            nrm = np.array([-uh[1], uh[0]])
            _slider_knob(ax, xx, uh, nrm, -1.0, kl, kh, color=R_SLIDER_FADE, zorder=6)
        # boundary sliders s: SAME knob design, different colour (green), along each edge
        for pos, uh, sd in S['bnd_sliders']:
            nrm = np.array([-uh[1], uh[0]])
            _slider_knob(ax, pos, uh, nrm, sd, kl, kh, color=S_SLIDER_FADE, zorder=6)
        # emphasize the ONE boundary slider the caption points to (full green, slightly larger)
        bpos, buh, bsd = min(S['bnd_sliders'], key=lambda t: t[0][1])
        bnrm = np.array([-buh[1], buh[0]])
        _slider_knob(ax, bpos, buh, bnrm, bsd, kl * 1.3, kh * 1.3, color=S_SLIDER, zorder=8)
        _drag_hand(ax, bpos, _clear_hand_dir(bpos, cut_segs, reach_h, bounds), reach=reach_h)
        _leader(ax, bpos, np.array([minx + 0.34 * W_, miny + 0.11 * H_]),
                r"boundary sliders $s$", S_SLIDER, 8.5)
        # HIGHLIGHT one cut: a THICKER dark centerline A-B + its two sliders x, x' as full purple knobs
        hc = S.get('highlight_cut')
        if hc is not None:
            A, B, x, xp = hc['A'], hc['B'], hc['x'], hc['xp']
            ax.plot([A[0], B[0]], [A[1], B[1]], color=INK, lw=2.6, solid_capstyle="round", zorder=7)
            for P in (A, B):
                ax.plot(*P, marker="o", mfc="white", mec=INK, ms=4, zorder=8)
            u_hat = (B - A) / np.linalg.norm(B - A); nrm = np.array([-u_hat[1], u_hat[0]])
            _slider_knob(ax, x, u_hat, nrm, -1.0, kl, kh, zorder=9)
            _drag_hand(ax, x, _clear_hand_dir(x, cut_segs + [(A, B)], reach_h, bounds), reach=reach_h)
            mid = (A + B) / 2
            _leader(ax, x, mid + np.array([0.30 * W_, 0.34 * H_]),
                    "void aspect ratio\nslider", R_SLIDER, 8.5)
            _leader(ax, B, B + np.array([0.10 * W_, -0.20 * H_]), r"cut $A\!-\!B$", INK, 9)


def panel_rom(ax, S):
    nodes = S['panel_flat']
    _draw_panels(ax, nodes, facecolor=LIGHT, edgecolor=INNER_EDGE, lw=0.9, zorder=4)
    hp = S['hinge_xy_flat']
    ax.scatter(hp[:, 0], hp[:, 1], s=56, color=TEAL, edgecolor="white", linewidths=1.0, zorder=8)
    _fit_view(ax, nodes.reshape(-1, 2))
    c = nodes.mean(1).mean(0)
    hid = int(np.argmin(np.linalg.norm(hp - c, axis=1)))
    _leader(ax, hp[hid], hp[hid] + np.array([1.4, 1.2]) * np.ptp(nodes[0], 0).max() * 0.7,
            r"hinge $(a,s,\vartheta)$", TEAL)
    fid = int(np.argmin(np.linalg.norm(nodes.mean(1) - c, axis=1)))
    pc = nodes[fid].mean(0)
    _leader(ax, pc, pc + np.array([-1.5, -1.3]) * np.ptp(nodes[0], 0).max() * 0.6, "rigid panel", GREY)


def panel_deploy(ax, S):
    """c: HALF deployment; flat (closed) state ghosted behind. Clamp + tile-fitted moment."""
    half = S['panel_half']
    _draw_panels(ax, S['panel_flat_frame'], facecolor="none", edgecolor=GREY, alpha=0.28, lw=0.8,
                 zorder=2)                                            # closed (start) ghost
    _draw_panels(ax, half, facecolor=LIGHT, edgecolor=INNER_EDGE, lw=0.9, zorder=4)
    span = np.ptp(half.reshape(-1, 2), 0).max()
    for fid in S['clamped']:
        _draw_panels(ax, half[[fid]], facecolor=GREY, alpha=0.7, edgecolor=INK, lw=1.2, zorder=6)
        p = S['fc_half'][fid]
        _leader(ax, p, p + np.array([-0.36, -0.5]) * span, "clamped\ncentre", GREY, 9)
    for fid in S['loaded']:
        tile = half[fid]; tc = tile.mean(0)
        mom_r = 0.36 * float(np.min(np.linalg.norm(tile - tc, axis=1)))
        _moment_arrow(ax, tc, mom_r, RED, ccw=True, lw=2.2)
        _leader(ax, tc + np.array([mom_r, mom_r]) * 0.9, tc + np.array([0.42, 0.46]) * span,
                r"applied moment $M$", RED, 9)
    W = S['W_half']
    norm = Normalize(vmin=float(np.quantile(W, 0.05)), vmax=float(np.quantile(W, 0.92)))
    sc = ax.scatter(S['hinge_xy_half'][:, 0], S['hinge_xy_half'][:, 1], c=W, cmap=ENERGY_CMAP,
                    norm=norm, s=46, edgecolor=INK, linewidths=0.5, zorder=9)
    _fit_view(ax, S['panel_frames'][-1].reshape(-1, 2), pad=0.14)
    return sc, norm


def panel_target(ax, S):
    final = S['panel_frames'][-1]
    _draw_panels(ax, final, facecolor=LIGHT, edgecolor=INNER_EDGE, lw=0.9, zorder=3)
    cen, rad = S['tcen'], S['trad']
    ax.add_patch(Circle(cen, rad, fill=False, edgecolor=INK, lw=2.2, ls="--", zorder=6))   # black target
    tp = cen + rad * np.array([0.72, 0.72])
    ax.annotate("target shape", xy=tp, textcoords="offset points", xytext=(14, 7), ha="left",
                fontsize=9.5, color=INK, arrowprops=dict(arrowstyle="-", color=INK, lw=0.9))
    cl = S['cloud']
    ax.scatter(cl[:, 0], cl[:, 1], s=44, color=GREEN, edgecolor="white", linewidths=0.9, zorder=8)
    lp = cl[np.argmin(cl[:, 0] + 0.7 * cl[:, 1])]                 # a lower-left boundary point
    _leader(ax, lp, np.array([cen[0] - 1.02 * rad, cen[1] - 1.02 * rad]),
            "deployed\nboundary\npoints", GREEN, 8.5)
    corners = cen + rad * np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    _fit_view(ax, np.vstack([final.reshape(-1, 2), corners]), pad=0.22)   # extra corner room


# ══════════════════════════════════════════════════════════════════════════════
# Card details (below the visual)
# ══════════════════════════════════════════════════════════════════════════════

# Sliders share ONE knob DESIGN but two colours (user): r = void-aspect-ratio sliders (purple),
# s = boundary sliders (green). NOT orange (= the paper).
R_SLIDER = PURPLE
R_SLIDER_FADE = "#B7A6CC"   # light purple (un-highlighted)
S_SLIDER = GREEN
S_SLIDER_FADE = "#9AC6A6"   # light green


def _slider_knob(ax, foot, u_hat, nrm, side, kl, kh, color=R_SLIDER, zorder=8):
    """A small asymmetric slider knob: half-a-square sitting ON the cut line, extending to ONE side."""
    c1, c2 = foot - (kl / 2) * u_hat, foot + (kl / 2) * u_hat
    rect = np.array([c1, c2, c2 + side * kh * nrm, c1 + side * kh * nrm])
    ec = INK if color in (R_SLIDER, S_SLIDER) else color
    ax.add_patch(MplPolygon(rect, closed=True, facecolor=color, edgecolor=ec, lw=0.5,
                            zorder=zorder, joinstyle="round"))


def detail_theta(ax, S):
    """a: how the SLIDERS control the VOID ASPECT RATIO, as an engineering drawing over the EXACT
    deployed void of the cut spotlighted above. The cut = the fixed diagonal A-B (the line seen when
    closed). The two sliders x, x' sit ON the cut (half-squares, one on each side) and open it
    perpendicular into the width w. r = l/w."""
    ax.axis("off"); ax.set_aspect("equal")
    vd = S.get("void_draw")
    if vd is None:
        ax.text(0.5, 0.5, "(void geometry unavailable)", ha="center", fontsize=8, color=GREY)
        return
    import matplotlib.patheffects as pe
    halo = [pe.withStroke(linewidth=2.6, foreground="white")]   # keeps labels legible over tiles
    A, B, x, xp = vd["A"], vd["B"], vd["x"], vd["xp"]
    for t in vd["tiles"]:
        ax.add_patch(MplPolygon(t, closed=True, facecolor=LIGHT, edgecolor=INNER_EDGE, lw=1.05,
                                joinstyle="round", zorder=2))
    void = np.array([A, x, B, xp])
    ax.add_patch(MplPolygon(void, closed=True, facecolor="white", edgecolor=INK, lw=1.4, zorder=4))
    ax.plot([A[0], B[0]], [A[1], B[1]], color=INK, ls=(0, (5, 3)), lw=1.2, zorder=5)   # cut A-B
    for P, lab, off in ((A, "A", np.array([-0.24, 0.0])), (B, "B", np.array([0.22, 0.0]))):
        ax.plot(*P, marker="o", mfc="white", mec=INK, ms=5, zorder=6)
        ax.text(P[0] + off[0], P[1] + off[1], lab, ha="center", va="center", fontsize=8.5,
                color=INK, zorder=8, path_effects=halo)
    # ONE void-aspect-ratio slider per cut: a single r sets BOTH cut-vertices, so keep both dotted
    # projections to the summits it opens but draw only ONE knob. Keep the SAME vertex the upper
    # panel spotlights (x = feet[0]) so the two images match.
    u = (B - A); u_hat = u / np.linalg.norm(u); nrm = np.array([-u_hat[1], u_hat[0]])
    corners = np.array([A, x, B, xp])
    feet = []
    for P, t in ((x, vd["t_x"]), (xp, vd["t_xp"])):
        foot = A + t * u
        side = 1.0 if np.dot(P - foot, nrm) >= 0 else -1.0
        summit = corners[int(np.argmin(np.linalg.norm(corners - foot, axis=1)))]  # nearest summit
        ax.plot([foot[0], summit[0]], [foot[1], summit[1]], color=R_SLIDER, ls=":", lw=0.8, zorder=5)
        feet.append((foot, side, t))
    foot, side, _ = feet[0]                                 # x — matches the upper panel's spotlight
    _slider_knob(ax, foot, u_hat, nrm, side, kl=0.14, kh=0.075, zorder=8)
    # l dimension along the cut (below); w = the perpendicular opening (black, to match l).
    yb = min(A[1], B[1], x[1], xp[1]) - 0.5
    for P in (A, B):
        ax.plot([P[0], P[0]], [P[1], yb + 0.08], color=INK, lw=0.5, zorder=5)
    ax.annotate("", xy=(B[0], yb), xytext=(A[0], yb),
                arrowprops=dict(arrowstyle="<|-|>", color=INK, lw=0.8, mutation_scale=7), zorder=5)
    ax.text((A[0] + B[0]) / 2, yb - 0.06, r"$\ell$ (cut)", ha="center", va="top", fontsize=8.5,
            color=INK, zorder=7, path_effects=halo)
    xdim = max(x[0], xp[0], B[0]) + 0.55       # w dimension sits clear to the RIGHT of the void
    for P in (x, xp):
        ax.plot([P[0], xdim + 0.08], [P[1], P[1]], color=INK, lw=0.5, zorder=5)
    ax.annotate("", xy=(xdim, x[1]), xytext=(xdim, xp[1]),
                arrowprops=dict(arrowstyle="<|-|>", color=INK, lw=0.8, mutation_scale=7), zorder=5)
    ax.text(xdim + 0.18, (x[1] + xp[1]) / 2, r"$w$ (width)", ha="left", va="center", fontsize=10.5,
            color=INK, zorder=7, path_effects=halo)
    xs_, ys_ = void[:, 0], void[:, 1]
    cx = (xs_.min() + xs_.max()) / 2
    # the defining relation, as a compact caption BELOW the drawing (not over it)
    ax.text(cx, yb - 0.42, r"$r=\ell\,/\,w$", ha="center", va="top", fontsize=10, color=INK,
            zorder=7, path_effects=halo)
    # fit the view to the FULL surrounding faces (complete, not clipped) plus the dimension labels
    tp = np.vstack([np.asarray(t) for t in vd["tiles"]]) if vd["tiles"] else void
    lo = np.minimum(tp.min(0), void.min(0)); hi = np.maximum(tp.max(0), void.max(0))
    ax.set_xlim(min(lo[0], xs_.min()) - 0.35, max(hi[0], xdim + 1.2))
    ax.set_ylim(min(lo[1], yb - 0.55), max(hi[1] + 0.35, ys_.max() + 0.35))


def detail_two_faces(ax):
    """b: how the hinge measures the relative motion of two INDEPENDENT rigid faces. Three isolated
    modes (rest state ghosted, left face fixed), each dimensioning exactly ONE reduced coordinate:

        tension   a = l/l0 - 1            the bond stretches along its own axis
        shear     s = d_perp / l0         the faces slide transverse to the bond
        rotation  theta = theta2 - theta1 the faces tilt relative to each other

    Kinematics only (no energy here) -- these are exactly what ligament_strains_linearized measures:
    axial = deformation . ref, shear = ref x deformation, rot = theta2 - theta1."""
    from matplotlib.patches import Arc as MplArc
    ax.axis("off"); ax.set_aspect("equal")
    ax.set_xlim(0.0, 6.2); ax.set_ylim(0.0, 4.3)
    GHOST = "#B9B6B0"                              # rest-state ghost
    d, g = 0.36, 0.10                              # panel centre-to-tip; half the rest ligament
    acx0, bcx0 = 1.55 - g - d, 1.55 + g + d        # alpha (fixed) / beta (rest) panel centres

    def diamond(cx, cy, ang=0.0):
        """A rotating-squares panel (square on its corner); the CORNER TIPS are the hinge points."""
        a = np.radians([0, 90, 180, 270]) + ang
        return np.c_[cx + d * np.cos(a), cy + d * np.sin(a)]

    def rot_about(P, ang, pts):
        c, s = np.cos(ang), np.sin(ang)
        return (pts - P) @ np.array([[c, s], [-s, c]]) + P

    def face(pts, ghost=False):
        ax.add_patch(MplPolygon(pts, closed=True, facecolor="none" if ghost else LIGHT,
                                edgecolor=GHOST if ghost else INNER_EDGE, lw=1.0 if ghost else 1.2,
                                ls=(0, (4, 3)) if ghost else "-", zorder=2 if ghost else 3))

    ax.text(2.95, 4.18, "two independent panels, hinged at a shared corner",
            ha="center", va="center", fontsize=8.0, color=GREY, style="italic")

    names = {"tension": "Tension", "shear": "Shear", "rotation": "Rotation"}
    forms = {"tension": r"$a=\ell/\ell_0-1$", "shear": r"$s=\delta_\perp/\ell_0$",
             "rotation": r"$\vartheta=\theta_2-\theta_1$"}
    for mode, yc in zip(("tension", "shear", "rotation"), (3.55, 2.15, 0.55)):
        P1 = np.array([acx0 + d, yc])                      # alpha's hinge CORNER (its right tip)
        P2r = np.array([bcx0 - d, yc])                     # beta's hinge CORNER at REST (its left tip)
        if mode == "tension":
            rf = diamond(bcx0 + 0.40, yc); P2 = P2r + np.array([0.40, 0.0])
        elif mode == "shear":
            rf = diamond(bcx0, yc - 0.44); P2 = P2r + np.array([0.0, -0.44])
        else:
            th = np.radians(26.0); rf = rot_about(P2r, th, diamond(bcx0, yc)); P2 = P2r.copy()
        face(diamond(bcx0, yc), ghost=True)                # rest ghost of the right panel
        face(diamond(acx0, yc)); face(rf)                  # left (fixed) + right (deformed)
        ax.text(acx0, yc, r"$\alpha$", ha="center", va="center", fontsize=8, color=GREY, zorder=4)
        ax.text(*rf.mean(0), r"$\beta$", ha="center", va="center", fontsize=8, color=GREY, zorder=4)
        ax.plot([P1[0], P2r[0]], [P1[1], P2r[1]], color=GHOST, lw=1.0, ls=(0, (3, 2)), zorder=3)
        ax.plot([P1[0], P2[0]], [P1[1], P2[1]], color=TEAL, lw=2.6, solid_capstyle="round", zorder=5)
        ax.scatter([P1[0], P2[0]], [P1[1], P2[1]], s=20, color=TEAL, edgecolor="white",
                   linewidths=0.7, zorder=6)                # the two hinge CORNERS
        if mode == "tension":                              # dimension l (current); tick marks l0 (rest)
            yb = yc - d - 0.26
            for X in (P1[0], P2[0]):
                ax.plot([X, X], [yb - 0.05, yc - d], color=GHOST, lw=0.5, zorder=2)   # witness lines
            ax.annotate("", xy=(P2[0], yb), xytext=(P1[0], yb),
                        arrowprops=dict(arrowstyle="<->", color=INK, lw=0.9, mutation_scale=9))
            ax.text((P1[0] + P2[0]) / 2, yb - 0.05, r"$\ell$", ha="center", va="top", fontsize=8.5, color=INK)
            ax.plot([P2r[0], P2r[0]], [yb - 0.05, yc - d], color=FACE_EDGE, lw=0.8, zorder=2)  # l0 tick
            ax.text(P2r[0] - 0.04, yb + 0.03, r"$\ell_0$", ha="right", va="bottom", fontsize=7,
                    color=FACE_EDGE)
        elif mode == "shear":                              # transverse offset at beta's far tip
            xr = bcx0 + d; xd = xr + 0.20
            for yy, col in ((yc, GHOST), (P2[1], INK)):
                ax.plot([xr, xd + 0.06], [yy, yy], color=col, lw=0.5, zorder=3)       # witness lines
            ax.annotate("", xy=(xd, P2[1]), xytext=(xd, yc),
                        arrowprops=dict(arrowstyle="<->", color=INK, lw=0.9, mutation_scale=9))
            ax.text(xd + 0.12, (P2[1] + yc) / 2, r"$\delta_\perp$", ha="left", va="center",
                    fontsize=8.5, color=INK)
        else:                                              # relative rotation arc at the hinge
            L = 0.45
            ax.plot([P2r[0], P2r[0]], [P2r[1], P2r[1] + L], color=GHOST, lw=1.0, ls=(0, (3, 2)), zorder=4)
            ax.plot([P2r[0], P2r[0] - L * np.sin(th)], [P2r[1], P2r[1] + L * np.cos(th)],
                    color=TEAL, lw=1.5, zorder=4)
            ax.add_patch(MplArc(P2r, 2 * (L + 0.05), 2 * (L + 0.05), angle=0, theta1=90,
                                theta2=90 + np.degrees(th), color=INK, lw=1.1, zorder=4))
            ax.text(P2r[0] - 0.20, P2r[1] + L + 0.14, r"$\vartheta$", ha="center", va="center",
                    fontsize=9.5, color=INK)
        ax.text(3.35, yc + 0.17, names[mode], ha="left", va="center", fontsize=9.5,
                fontweight="bold", color=INK)
        ax.text(3.35, yc - 0.21, forms[mode], ha="left", va="center", fontsize=9.5, color=INK)


def detail_energy(ax, S):
    """c: real energy history of the loading -- hinge energy rises (rotation-dominated) with
    tension/shear/rotation contributions, as the external load is ramped up."""
    h = S['energy_hist']
    lam = h['lam']
    top = h['total'].max()
    # shear & tension are tiny next to rotation -> lift them to a visible band (schematic) so the
    # reader sees all three contributions; only their SHAPE matters here.
    sh = h['shear'] / max(h['shear'].max(), 1e-12) * 0.42 * top
    tn = h['stretch'] / max(h['stretch'].max(), 1e-12) * 0.26 * top
    ax.fill_between(lam, 0, h['total'], color=CREAM, zorder=1, label="hinge energy")
    ax.plot(lam, h['rot'], color=ORANGE, lw=2.2, zorder=6, label="rotation")
    ax.plot(lam, sh, color=TEAL, lw=1.6, zorder=5, label="shear")
    ax.plot(lam, tn, color=GREY, lw=1.6, zorder=5, label="tension")
    ax.set_xlim(0, 1); ax.set_ylim(0, top * 1.14)
    ax.set_xlabel(r"load parameter $\lambda$", fontsize=8.5, color=GREY, labelpad=2)
    ax.set_yticks([]); ax.tick_params(axis="x", labelsize=8, length=0)
    ax.spines["left"].set_visible(False)
    ax.legend(fontsize=7.2, loc="upper left", frameon=False, handlelength=1.1, labelspacing=0.22,
              borderaxespad=0.15, bbox_to_anchor=(0.0, 0.92))


def detail_damage_curve(ax):
    """d: simplified training curve -- the hinge DAMAGE (physical) loss falling as the optimizer runs.
    Schematic monotone decay; the point is the downward TREND, in the panel-c plot's clean style."""
    ep = np.linspace(0.0, 1.0, 120)
    dmg = 0.10 + 0.90 * np.exp(-4.3 * ep)                  # normalized damage: high -> settles low
    ax.fill_between(ep, 0, dmg, color=DMG_FILL, zorder=1)
    ax.plot(ep, dmg, color=PHYS_COLOR, lw=2.4, zorder=5)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.16)
    ax.set_xlabel("training epoch", fontsize=8.5, color=GREY, labelpad=2)
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.text(0.035, 1.10, r"hinge damage $\downarrow$", ha="left", va="top", fontsize=8.5,
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
    from nff.scripts.figures import render_hinge_3d as R
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


def _fea_hinge_png(out="data/outputs/hinge_top_fea.png"):
    """Render a REAL CalculiX hinge RVE from the top (von Mises stored energy/damage). Idempotent."""
    if os.path.exists(out):
        return out
    npz = "data/outputs/hinge_frames_w1p5.npz"
    if not os.path.exists(npz):
        return None
    from nff.scripts.figures.render_hinge_3d import render
    d = np.load(npz)
    k = int(np.argmin(np.abs(d['damage'] - 0.6)))               # partially damaged: clear stress
    render(d['xyz'], d['conn'], d['disp'][k], float(d['w_lig']), float(d['thickness']), 30.0,
           out, elev=90, azim=-90, vm=d['vm'][k], vmin=0.0,
           vmax=float(np.percentile(d['vm'][k], 99)), scale=False)
    return out


def detail_hinge_energy(ax, png):
    """d: a REAL FEA (CalculiX) top view of one hinge RVE -- two faces joined by the ligament,
    coloured by von Mises stress = where energy is stored and damage accrues."""
    import matplotlib.image as mpimg
    ax.axis("off")
    if png is None or not os.path.exists(png):
        ax.text(0.5, 0.5, "(FEA render unavailable)", ha="center", va="center", fontsize=8, color=GREY)
        return
    img = mpimg.imread(png)
    a = img[..., 3] if img.shape[-1] == 4 else np.ones(img.shape[:2])
    ys, xk = np.where(a > 0.02)
    img = img[ys.min():ys.max() + 1, xk.min():xk.max() + 1]
    H, W = img.shape[:2]
    ax.imshow(img, zorder=2)
    ax.set_xlim(-0.02 * W, 1.24 * W); ax.set_ylim(H * 1.02, -0.10 * H)
    ax.text(W * 0.18, H * 0.66, "face", ha="center", fontsize=8, color="white", zorder=5)
    ax.text(W * 0.82, H * 0.66, "face", ha="center", fontsize=8, color="white", zorder=5)
    ax.annotate("stored energy\n+ damage", xy=(W * 0.50, H * 0.42), xytext=(W * 1.02, H * 0.20),
                ha="left", va="center", fontsize=7.6, color=RED, zorder=6,
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.1, mutation_scale=8))
    ax.set_title("hinge — von Mises stress (FEA)", fontsize=8.6, color=GREY, pad=1)


# ══════════════════════════════════════════════════════════════════════════════
# Compose
# ══════════════════════════════════════════════════════════════════════════════

def build_figure(S, out_base):
    apply_charter()
    fig = plt.figure(figsize=(20, 10.6))

    pw = 0.142
    xs = {"a": 0.016, "b": 0.228, "c": 0.420, "d": 0.612, "e": 0.838}
    x_loss = 0.800
    VIS_Y, VIS_H = 0.605, 0.275
    FORM_Y = 0.560
    DET_Y, DET_H = 0.335, 0.185
    CARD_Y, CARD_H, cpad = 0.315, 0.595, 0.010
    titles = {"a": "cut pattern", "b": "reduced-order model", "c": "kinematic deployment",
              "d": "deployed vs target", "e": "optimized cut pattern"}

    # forward-model container behind the b,c,d cards
    fb0, fb1 = xs["b"] - cpad - 0.006, xs["d"] + pw + cpad + 0.006
    fig.add_artist(FancyBboxPatch((fb0, CARD_Y - 0.012), fb1 - fb0, CARD_H + 0.052,
                                  boxstyle="round,pad=0.004,rounding_size=0.012",
                                  transform=fig.transFigure, facecolor=FWD_FILL, edgecolor=GREY,
                                  lw=1.3, zorder=-3))
    fig.text((fb0 + fb1) / 2, CARD_Y + CARD_H + 0.052, "differentiable forward model",
             ha="center", fontsize=13, style="italic", color=GREY)

    # cards
    for k in xs:
        edge = R_COLOR if k == "a" else (GREY if k in ("b", "c", "d") else INNER_EDGE)
        fig.add_artist(FancyBboxPatch((xs[k] - cpad, CARD_Y), pw + 2 * cpad, CARD_H,
                                      boxstyle="round,pad=0.003,rounding_size=0.01",
                                      transform=fig.transFigure, facecolor=BOX_FILL, edgecolor=edge,
                                      lw=(2.0 if k == "a" else 1.2), zorder=-2))
        lt = {"a": "a", "b": "b", "c": "c", "d": "d", "e": "e"}[k]
        fig.text(xs[k] - cpad + 0.006, CARD_Y + CARD_H + 0.006, lt, fontsize=16, fontweight="bold")
        fig.text(xs[k] + pw / 2, CARD_Y + CARD_H + 0.012, titles[k], ha="center", fontsize=12)

    axv = {k: fig.add_axes([xs[k], VIS_Y, pw, VIS_H]) for k in xs}
    panel_cut(axv["a"], S['geom_init'], "design", S)
    panel_rom(axv["b"], S)
    sc, norm = panel_deploy(axv["c"], S)
    panel_target(axv["d"], S)
    panel_cut(axv["e"], S['geom_opt'], "output")

    # per-card formula + detail
    _colored_line(fig, xs["a"] + pw / 2, FORM_Y,
                  [(r"$\theta=(\,$", INK), (r"$r$", R_SLIDER), (r"$,\ $", INK),
                   (r"$s\,$", S_SLIDER), (r"$)$", INK)], fs=13, box=True)
    # all four mid-band detail images share ONE vertical band [DET_Y, DET_Y+DET_H] (aligned)
    detail_theta(fig.add_axes([xs["a"], DET_Y, pw, DET_H]), S)

    _formula(fig, xs["b"] + pw / 2, FORM_Y, r"$\varepsilon_h(q)=(a_h,\,s_h,\,\vartheta_h)$", TEAL, 11)
    detail_two_faces(fig.add_axes([xs["b"], DET_Y, pw, DET_H]))

    _formula(fig, xs["c"] + pw / 2, FORM_Y,
             r"$q^\star(\lambda;\theta)=\arg\min_{q\in\mathcal{C}}\ "
             r"\sum_{h\in\mathcal{H}} W\!\left(\varepsilon_h(q);g_h\right)-W_{\mathrm{ext}}$",
             INK, 8.6)
    detail_energy(fig.add_axes([xs["c"] + 0.012, DET_Y + 0.024, pw - 0.02, DET_H - 0.040]), S)
    cax = fig.add_axes([xs["c"] + pw - 0.010, VIS_Y + VIS_H * 0.27, 0.007, VIS_H * 0.46])
    cb = fig.colorbar(sc, cax=cax, orientation="vertical")
    cb.set_ticks([norm.vmin, norm.vmax]); cb.set_ticklabels(["low", "high"])
    cb.ax.yaxis.set_ticks_position("right"); cb.ax.tick_params(length=0, labelsize=8)
    cb.ax.set_title(r"hinge $W$", fontsize=8, color=INK, pad=3)

    _colored_line(fig, xs["d"] + pw / 2, FORM_Y,
                  [(r"$\mathcal{L}=$", INK), (r"$\mathcal{L}_{\mathrm{geom}}$", GEOM_COLOR),
                   (r"$\,+\,$", INK), (r"$\mathcal{L}_{\mathrm{phys}}$", PHYS_COLOR)], fs=12, box=True)
    _colored_line(fig, xs["d"] + pw / 2, FORM_Y - 0.042,
                  [("target fit", GEOM_COLOR), ("  +  ", GREY), ("hinge damage", PHYS_COLOR)], fs=8.3)
    detail_damage_curve(fig.add_axes([xs["d"] + 0.012, DET_Y + 0.024, pw - 0.02, DET_H - 0.040]))

    _formula(fig, xs["e"] + pw / 2, FORM_Y, r"$\theta^\ast$  optimized design", INNER_EDGE, 10.5)
    fig.text(xs["e"] + pw / 2, DET_Y + DET_H / 2, "ready for\nreal-world\nfabrication", ha="center",
             va="center", fontsize=11, color="#8a3a00", style="italic")

    # forward chevrons + reduction label
    def chev(x):
        fig.text(x, VIS_Y + VIS_H / 2, "❯", ha="center", va="center", fontsize=22, color=GREY)
    x_ab = (xs["a"] + pw + cpad + fb0) / 2   # centre of the gap between a's card and the box
    chev(x_ab)
    fig.text(x_ab, VIS_Y + VIS_H / 2 + 0.04, "reduce to a\nrigid-panel\nmodel", ha="center",
             va="bottom", fontsize=6.6, color=GREY)
    chev((xs["b"] + pw + xs["c"]) / 2)
    chev((xs["c"] + pw + xs["d"]) / 2)

    # loss node between d and e
    ly = VIS_Y + VIS_H / 2
    fig.add_artist(FancyBboxPatch((x_loss - 0.014, ly - 0.026), 0.028, 0.052,
                                  boxstyle="round,pad=0.002,rounding_size=0.008",
                                  transform=fig.transFigure, facecolor="white", edgecolor=RED,
                                  lw=1.6, zorder=6))
    fig.text(x_loss, ly, r"$\mathcal{L}$", ha="center", va="center", fontsize=15, color=RED, zorder=7)
    fig.text(x_loss, ly + 0.04, "loss", ha="center", fontsize=9.5, color=RED)
    _fig_arrow(fig, (xs["d"] + pw + cpad + 0.002, ly), (x_loss - 0.016, ly), GREY, 2.0)
    _fig_arrow(fig, (x_loss + 0.016, ly), (xs["e"] - cpad - 0.002, ly), INK, 2.4)

    # ── backward pass: bold INK lane from the loss back to the theta box (a) ──
    lane_y, lane_h = 0.135, 0.058
    lane_x0, lane_x1 = xs["a"] + pw * 0.22, x_loss
    fig.add_artist(FancyBboxPatch((lane_x0, lane_y - lane_h / 2), lane_x1 - lane_x0, lane_h,
                                  boxstyle="round,pad=0.002,rounding_size=0.02",
                                  transform=fig.transFigure, facecolor=BACK_FILL, edgecolor=INK,
                                  lw=1.8, zorder=1))
    for xchev in np.linspace(lane_x0 + 0.03, lane_x1 - 0.03, 9):
        fig.text(xchev, lane_y, "❮", ha="center", va="center", fontsize=15, color=INK, zorder=2)
    fig.text((lane_x0 + lane_x1) / 2, lane_y + lane_h / 2 + 0.015,
             r"backward pass — gradient descent  $\partial\mathcal{L}/\partial\theta$  through the differentiable physics",
             ha="center", fontsize=12.5, color=INK, style="italic")
    _fig_arrow(fig, (x_loss, ly - 0.028), (x_loss, lane_y + lane_h / 2), INK, 2.4)
    badge_w = pw * 0.22 + 0.09
    fig.add_artist(FancyBboxPatch((xs["a"], lane_y - 0.026), badge_w, 0.052,
                                  boxstyle="round,pad=0.004,rounding_size=0.01",
                                  transform=fig.transFigure, facecolor="white", edgecolor=INK,
                                  lw=1.6, zorder=3))
    fig.text(xs["a"] + badge_w / 2, lane_y,
             r"$\theta \leftarrow \theta - \eta\,\partial\mathcal{L}/\partial\theta$",
             ha="center", va="center", fontsize=11.5, color=INK, zorder=4)
    # arrow points DIRECTLY at the theta box (card a)
    _fig_arrow(fig, (xs["a"] + pw * 0.5, lane_y + 0.028), (xs["a"] + pw * 0.5, CARD_Y - 0.002), INK, 2.6)
    fig.text(xs["a"] + pw * 0.5 + 0.016, (lane_y + CARD_Y) / 2, r"updates $\theta$", ha="left",
             va="center", fontsize=9.5, color=INK)

    for ext in ("png", "pdf"):
        fig.savefig(f"{out_base}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_base}.png / .pdf")


def _fig_arrow(fig, p0, p1, color, lw=2.0):
    fig.add_artist(FancyArrowPatch(p0, p1, transform=fig.transFigure, arrowstyle="-|>",
                                   mutation_scale=15, lw=lw, color=color, zorder=5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", default="overview")
    ap.add_argument("--out", default="data/outputs/pipeline_overview")
    args = ap.parse_args()
    build_figure(extract_spine(args.config_name), args.out)


if __name__ == "__main__":
    main()
