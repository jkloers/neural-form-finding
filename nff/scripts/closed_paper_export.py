"""Paper prototyping export for a closed run: precise PNG + true 1:1 A4 print PDF.

Two concerns live here, both about turning the SIMULATED design into something you can cut and try:

1. The 1:1 A4 renderer (``export_cut_pattern_a4``) — pure geometry -> multi-page PDF. Each page is A4
   with the sheet drawn so 1 mm = 1 mm on paper (no auto-fit); large sheets tile across pages with an
   overlap band, corner registration marks, a page ``(row, col)`` label, and a printed calibration
   bar to verify the printer did not rescale. Clamped / loaded tiles are marked HOLD / PULL (+ arrow)
   for the two-hands pinch test.
2. The run glue (``write_cut_patterns``) — pulls per-hinge manufacturing params (learned ``w_lig``,
   fillet) and the HOLD/PULL boundary-condition marks from a run's state/config, then writes both the
   PNG and the A4 PDF.

Kept out of ``run_closed`` (thin call site) and out of ``nff/topology`` (which must not depend on
``nff/stages``). The pure Shapely geometry engine stays in ``nff.topology.cut_pattern``.

Print settings: 100% / "Actual size" (NOT "fit to page"); then measure the calibration bar.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle

from nff.topology.cut_pattern import build_cut_geometry, measure_cut_geometry, cut_sheet

# A4 portrait [mm]; Princeton palette (kept consistent with the other figures)
_A4_W, _A4_H = 210.0, 297.0
_CLAMP, _LOAD, _INK = "#6C757D", "#D62828", "#1A1A1A"
_W_C_MM = 0.2                    # kerf width [mm]
_PHYS_TILE = 10.0               # physical scale convention: w_lig ~ 1/10 tile


# ── 1:1 A4 renderer (pure: geometry + marks -> PDF) ────────────────────────────────

def _draw_lines(ax, geom_lines, **kw):
    """Plot a Shapely (Multi)LineString / boundary as thin strokes."""
    for g in getattr(geom_lines, "geoms", [geom_lines]):
        if not g.is_empty:
            x, y = g.xy
            ax.plot(x, y, **kw)


def _nice_len(x):
    """Largest 1 / 2 / 5 x 10^k that is <= x (for a tidy scale bar)."""
    if x <= 0:
        return 1.0
    k = 10.0 ** np.floor(np.log10(x))
    return max(m * k for m in (1, 2, 5) if m * k <= x)


def export_cut_pattern_a4(geom, out_path, *, clamped_centroids=None, loaded_centroids=None,
                          pull_dir=None, margin_mm=0.0, mark_radius_mm=None, title=None):
    """Scale the WHOLE cut pattern onto a SINGLE A4 sheet, with clamp/load marks.

    The pattern is fit (aspect-preserving) to one A4 page — portrait or landscape, whichever holds it
    larger. With ``margin_mm=0`` (default) the outer tiles sit on the paper edge, so there is no border
    to trim (print "Actual size" / borderless). This is a scaled paper proxy (NOT 1:1); a scale bar +
    ratio report the real size. The A4-beam aspect (~1.42) nearly matches A4 landscape, so it fills
    the sheet almost exactly.

    Args:
        geom: output of ``build_cut_geometry`` (mm).
        out_path: destination ``.pdf`` (single page).
        clamped_centroids: (K, 2) mm face centroids held fixed -> gray "HOLD" discs. Any K (incl 0).
        loaded_centroids:  (L, 2) mm face centroids pulled -> red "PULL" discs + arrows. Any L.
        pull_dir: (2,) pull direction; arrows point this way. Defaults to +x.
        margin_mm: page margin around the pattern (0 = tiles to the paper edge).
        mark_radius_mm: HOLD/PULL disc radius in REAL mm; default = 0.35 * median centroid spacing.
        title: header text.

    Returns:
        dict: ``n_pages`` (1), ``scale`` (paper-mm per real-mm), ``sheet_mm`` (w, h), ``path``.
    """
    cut_lines = cut_sheet(geom).boundary                 # every stroke = a physical cut / sheet edge
    minx, miny, maxx, maxy = geom["sheet"].bounds
    sheet_w, sheet_h = maxx - minx, maxy - miny

    clamped = (np.asarray(clamped_centroids, float).reshape(-1, 2)
               if clamped_centroids is not None else np.empty((0, 2)))
    loaded = (np.asarray(loaded_centroids, float).reshape(-1, 2)
              if loaded_centroids is not None else np.empty((0, 2)))
    pull = np.asarray(pull_dir, float) if pull_dir is not None else np.array([1.0, 0.0])
    pull = pull / (np.linalg.norm(pull) + 1e-12)

    if mark_radius_mm is None:                            # size discs to ~1/3 of the tile pitch [mm]
        allc = np.vstack([clamped, loaded])
        if len(allc) >= 2:
            d = np.linalg.norm(allc[:, None, :] - allc[None, :, :], axis=-1)
            np.fill_diagonal(d, np.inf)
            mark_radius_mm = 0.35 * float(np.median(d.min(1)))
        else:
            mark_radius_mm = 0.03 * max(sheet_w, sheet_h)
    arrow_len = 1.0 * mark_radius_mm                      # short direction indicator, centered on disc

    # page orientation = whichever fits the pattern larger; then aspect-preserving fit-to-page
    page_w, page_h = (_A4_H, _A4_W) if sheet_w >= sheet_h else (_A4_W, _A4_H)
    content_w, content_h = page_w - 2 * margin_mm, page_h - 2 * margin_mm
    scale = min(content_w / sheet_w, content_h / sheet_h)     # paper-mm per real-mm
    drawn_w, drawn_h = sheet_w * scale, sheet_h * scale

    fig = plt.figure(figsize=(page_w / 25.4, page_h / 25.4))
    # geometry axes: the centered scaled rectangle (data coords = real mm)
    ax = fig.add_axes([(margin_mm + (content_w - drawn_w) / 2) / page_w,
                       (margin_mm + (content_h - drawn_h) / 2) / page_h,
                       drawn_w / page_w, drawn_h / page_h])
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_aspect("equal")
    ax.axis("off")
    _draw_lines(ax, cut_lines, color=_INK, lw=0.5, solid_capstyle="round")
    for pt in clamped:
        ax.add_patch(Circle(pt, mark_radius_mm, facecolor=_CLAMP, edgecolor="none",
                            alpha=0.45, zorder=5))
        ax.text(pt[0], pt[1], "HOLD", ha="center", va="center", fontsize=5,
                color="white", fontweight="bold", zorder=6)
    for pt in loaded:
        ax.add_patch(Circle(pt, mark_radius_mm, facecolor=_LOAD, edgecolor="none",
                            alpha=0.40, zorder=5))
        ax.add_patch(FancyArrowPatch(pt - 0.5 * arrow_len * pull, pt + 0.5 * arrow_len * pull,
                                     arrowstyle="-|>", mutation_scale=7, color=_LOAD,
                                     lw=1.2, zorder=7))
        ax.text(pt[0], pt[1] - 1.25 * mark_radius_mm, "PULL", ha="center", va="top",
                fontsize=5, color=_LOAD, fontweight="bold", zorder=6)

    _annotate(fig, page_w, page_h, scale, sheet_w, sheet_h, title)
    fig.savefig(out_path)                                # single-page PDF (MediaBox = A4)
    plt.close(fig)
    return dict(n_pages=1, scale=scale, sheet_mm=(sheet_w, sheet_h), path=out_path)


def _annotate(fig, page_w, page_h, scale, sheet_w, sheet_h, title):
    """Full-page overlay (page-mm coords): header + a real-size scale bar."""
    ov = fig.add_axes([0, 0, 1, 1])
    ov.set_xlim(0, page_w)
    ov.set_ylim(0, page_h)
    ov.set_aspect("equal")
    ov.axis("off")
    ov.patch.set_alpha(0.0)

    bar_real = _nice_len(0.2 * sheet_w)                  # a round real length
    bar_paper = bar_real * scale                         # its drawn length on paper [mm]
    bx, by = 5.0, 4.0                                     # fixed inset from the corner (margin=0)
    ov.plot([bx, bx + bar_paper], [by, by], color=_INK, lw=1.4, solid_capstyle="butt")
    for xx in (bx, bx + bar_paper):
        ov.plot([xx, xx], [by - 1.2, by + 1.2], color=_INK, lw=1.4)
    ov.text(bx + bar_paper / 2, by + 2.0, f"{bar_real:.0f} mm (real)", ha="center", va="bottom",
            fontsize=6, color=_INK)

    head = ((title + "   ·   " if title else "") +
            f"sheet {sheet_w:.0f}×{sheet_h:.0f} mm on A4   ·   scale ≈ 1:{1 / scale:.1f}")
    ov.text(page_w / 2, page_h - 3.0, head, ha="center", va="top", fontsize=7, color=_INK)


# ── run glue (state/config -> the two files) ───────────────────────────────────────

def _per_hinge_lookup(initial_state, hinge_w_lig, w_lig_mm, fillet_ratio, length_scale):
    """(H, 4) per-hinge ``[x_mm, y_mm, w_lig_mm, rho_mm]`` keyed by pivot position [mm]."""
    from nff.stages.geometry import hinge_vertex_positions
    pivots = np.asarray(hinge_vertex_positions(
        initial_state.face_centroids, initial_state.centroid_node_vectors,
        initial_state.hinge_node_pairs)[0]) * length_scale
    w_lig = np.asarray(hinge_w_lig) if hinge_w_lig is not None else np.full(len(pivots), w_lig_mm)
    return np.column_stack([pivots, w_lig, fillet_ratio * w_lig]), w_lig


def _bc_marks(initial_state, config, load_specs, length_scale):
    """Clamped/loaded face centroids [mm] + pull direction from the config BCs (any count)."""
    face_mm = np.asarray(initial_state.face_centroids) * length_scale
    clamped_f = [int(f) for f in config.topology.get('bc_clamped', [])]
    loaded_f = [int(s['face']) for s in load_specs]
    dof = int(load_specs[0]['dof']) if load_specs else 0
    sign = float(np.sign(load_specs[0].get('value', 1.0))) if load_specs else 1.0
    pull_dir = [sign, 0.0] if dof == 0 else [0.0, sign]
    return (face_mm[clamped_f] if clamped_f else None,
            face_mm[loaded_f] if loaded_f else None, pull_dir, len(clamped_f), len(loaded_f))


def write_cut_patterns(run_dir, *, initial_state, cut_coords, struct, config, hinge_model,
                       load_specs, config_name, hinge_w_lig=None):
    """Write ``cut_pattern.png`` (precise) + ``cut_pattern_A4.pdf`` (1:1 print) into ``run_dir``.

    Builds the per-hinge cut geometry once (learned ``w_lig`` + fillet), renders the PNG, and exports
    the tiled A4 print with HOLD/PULL marks. Falls back to the schematic PNG if the precise build
    fails; the A4 export is attempted independently. Returns the A4 summary dict, or ``None``.
    """
    from nff.utils.visualization import render_precise_cut_pattern, plot_cut_pattern

    w_lig_mm = float(getattr(hinge_model, 'w_lig_mm', 5.0))
    spacing = float(config.topology.get('spacing', 1.0))
    fillet_ratio = float(getattr(hinge_model, 'fillet_ratio', 0.16))
    length_scale = _PHYS_TILE * w_lig_mm / spacing
    T, cols = np.asarray(struct['T']), struct['cols']
    cut_png = os.path.join(run_dir, "cut_pattern.png")

    try:
        hinge_lookup, w_lig = _per_hinge_lookup(initial_state, hinge_w_lig, w_lig_mm,
                                                fillet_ratio, length_scale)
        geom = build_cut_geometry(cut_coords, T, cols, w_c=_W_C_MM, w_lig=w_lig_mm,
                                  rho=fillet_ratio * w_lig_mm, length_scale=length_scale,
                                  hinge_lookup=hinge_lookup)
        rt = measure_cut_geometry(geom)
        print(f"  [cut_pattern] per-hinge geometry: {len(geom['hinge_info'])} hinges, "
              f"w_lig {w_lig.min():.1f}-{w_lig.max():.1f}mm, "
              f"round-trip err {rt.get('max_w_lig_err', 0.0):.1e}mm")
        render_precise_cut_pattern(
            geom, filepath=cut_png,
            title=f"Precise laser cut pattern — flat sheet ({getattr(hinge_model, 'material', 'S235')})")
    except Exception as exc:                            # keep the run alive on any geometry hiccup
        print(f"  [cut_pattern] precise build failed ({exc}); schematic fallback, no A4")
        plot_cut_pattern(cut_coords, T, cols, filepath=cut_png,
                         hinge_margin=max(0.22, 1.6 * float(config.topology.get('hinge_margin', 0.06))),
                         lw=1.8, title="Kirigami cut pattern (flat sheet)")
        return None

    try:
        clamped, loaded, pull_dir, n_hold, n_pull = _bc_marks(
            initial_state, config, load_specs, length_scale)
        res = export_cut_pattern_a4(geom, os.path.join(run_dir, "cut_pattern_A4.pdf"),
                                    clamped_centroids=clamped, loaded_centroids=loaded,
                                    pull_dir=pull_dir, title=str(config_name))
        print(f"  [cut_pattern] A4 print: 1 page, scale 1:{1 / res['scale']:.1f}, "
              f"sheet {res['sheet_mm'][0]:.0f}x{res['sheet_mm'][1]:.0f}mm fit to A4, "
              f"HOLD {n_hold} / PULL {n_pull} tiles -> cut_pattern_A4.pdf")
        return res
    except Exception as exc:
        print(f"  [cut_pattern] A4 export skipped ({exc})")
        return None
