"""The inverse-design matrix: one sheet, five mechanical problems, four readings each.

Each ROW is a boundary-value problem posed on the SAME closed sheet -- same tile count, same
material, same target rectangle -- and differing only in how the right boundary is actuated. Each
COLUMN is one reading of that row:

    0  the problem      the sheet as a plain rectangle, its supports and its loads
    1  trained design   the learned flat cut pattern, panels coloured by area change
    2  deployed         the same panels after Stage 2, against the target they were asked to match
    3  load path        where the force actually goes once the sheet has opened

The claim the figure makes is a comparison, so three things are held fixed across rows and must
stay that way: one colour norm for columns 1-2 pooled over every row, one axis extent per column,
and one target rectangle. A per-row norm or a per-row target would turn five comparable rows into
five unrelated figures.

Nothing but ``best_params.pkl`` and ``config.yaml`` is persisted by a run, so each row's design is
re-solved here (one forward pass, seconds) rather than read back.

The row order and labels ARE the figure's argument, and neither is stored in the runs, so the
manuscript's five rows are written out here rather than left to be reconstructed:

    JAX_PLATFORMS=cpu conda run -n kgnn_mac python \
        nff/scripts/figures/plot_inverse_design_matrix.py \
          --row "data/outputs/runs/run_20260806_161121_figmat_1_deploy:Concentrated axial load" \
          --row "data/outputs/runs/run_20260806_115352_figmat_2_cantilever:Axial and transverse load" \
          --row "data/outputs/runs/run_20260806_161121_figmat_3_moment:Axial load and couple" \
          --row "data/outputs/runs/run_20260806_161121_figmat_4_asym:Eccentric axial load" \
          --row "data/outputs/runs/run_20260806_115352_figmat_5_vertical:Self-equilibrated shear couple"
"""

import argparse
import os
import textwrap
from types import SimpleNamespace

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np                                                       # noqa: E402

from nff.utils.figstyle import use_agg                                   # noqa: E402
use_agg()

import matplotlib.pyplot as plt                                          # noqa: E402
import matplotlib.colors as mcolors                                      # noqa: E402
from matplotlib.lines import Line2D                                      # noqa: E402
from matplotlib.patches import Patch                                     # noqa: E402
from matplotlib.ticker import FuncFormatter                              # noqa: E402

from nff.utils.figstyle import (FS_ANNOT, FS_BASE, FS_PANEL, FS_TICK,    # noqa: E402
                                GREY, INK, ORANGE, TEXT_W_BP, apply_charter,
                                charter_diverging, figsize_in, panel_letter, paper_rc,
                                save_paper_figure, text_width_bp)
from nff.config.experiment import load_and_parse_config                  # noqa: E402
from nff.closed.setup import build_closed_initial_state                  # noqa: E402
from nff.closed.force_stats import tile_pressure                         # noqa: E402
from nff.closed.load_path_viz import COMPRESSION_FILL, draw_load_path    # noqa: E402
from nff.closed.load_paths import load_field, solve_deployed_state       # noqa: E402
from nff.stages.geometry import (compute_face_areas, deformed_vertices,  # noqa: E402
                                 reconstruct_vertices)
from nff.utils.visualization import draw_loading_schematic, plot_tessellation  # noqa: E402

TILE_EDGE = "#9CA3AB"          # the load-path column's tile edge; columns (b) and (c) match it
TILE_EDGE_W = 0.25
SHEAR_DOT = "#F2B705"          # golden yellow for the shear marks: the compression wash already
                               # spends TEAL, the module default, so shear read as its darkest
                               # tiles. Warm like the tension route but a full step lighter and
                               # yellower than PRINCETON ORANGE, and only ever drawn as a point.
ROMAN = ("I", "II", "III", "IV", "V", "VI", "VII", "VIII")

# ── Page geometry, in POINTS. The plate is authored at the width the manuscript includes it at, so
# an authored point size is a printed point size. Drawn 1257 bp wide and reduced to 1.2\textwidth,
# every label on it printed between 3.5 and 4.6 pt.
FIG_W_BP = 1.2 * TEXT_W_BP     # \includegraphics[width=1.2\textwidth]
SCHEM_FRAC = 0.62              # schematic column width, as a fraction of a data panel
GAP_BP = 6.0                   # the one spacing unit: gutter padding and inter-column breathing
RIGHT_BP = 22.0                # the load arrows of column (d) reach outside their own panel
HEAD_BP, FOOT_BP = 58.0, 52.0  # header band (letter + title + rule); colorbar + load-path key band
LET_Y_BP, TITLE_Y_BP, RULE_Y_BP = 8.0, 16.0, 54.0     # measured DOWN from the canvas top; the
                                                      # band holds a THREE-line title, which is
                                                      # what the narrow schematic column needs
# Strokes do not follow the type. The data panels are 96 bp wide against the 313 bp they were tuned
# at, so every load-path weight has to come down by that ratio or the routes swallow the sheet.
WEIGHT_SCALE = 0.45

# Two lines each, so the header band stays shallow and the titles stay in mechanics vocabulary
# rather than in the project's internal shorthand.
COLUMNS = ("Boundary conditions\nand loading",
           "Optimised cut pattern,\nundeployed",
           "Deployed configuration\nand net tile stress",
           "Internal force\ntransmission")

# Fields that MUST agree across rows, or the figure is comparing different sheets.
SHARED_KEYS = ("M", "N", "spacing", "spacing_y", "r_init", "hinge_margin",
               "target_cx", "target_cy", "target_half_w", "target_half_h")


def _shim(face_nodes):
    """Duck-typed ``Tessellation`` whose faces carry their OWN corners.

    ``write_deformed_into`` welds shared vertices, and in a deployed kirigami one shared vertex
    belongs to up to four tiles that have pulled apart by a void width -- last-write-wins then puts
    three of them in the wrong place. Giving every face its own corners reproduces the sheet
    exactly, and ``plot_tessellation`` needs nothing more than ``.vertices`` and
    ``.faces[i].vertex_indices``.
    """
    nodes = np.asarray(face_nodes, dtype=float)
    n_faces, n_nodes, _ = nodes.shape
    return SimpleNamespace(
        vertices=nodes.reshape(-1, 2),
        faces=[SimpleNamespace(vertex_indices=np.arange(k * n_nodes, (k + 1) * n_nodes))
               for k in range(n_faces)])


def _target_params(config):
    """The FIXED rectangle the run trained against, or the config's fallback target."""
    topo = config.topology
    if 'target_half_w' in topo:
        hw, hh = float(topo['target_half_w']), float(topo['target_half_h'])
        if float(topo.get('target_rot_deg', 0.0) or 0.0):
            raise SystemExit(f"config carries target_rot_deg={topo['target_rot_deg']}, but "
                             "_target_corners frames the panels from an axis-aligned corner pair "
                             "and would crop a rotated target. Handle the rotation before using "
                             "this figure with such a config.")
        return {'type': 'rectangle',
                'center': [float(topo['target_cx']), float(topo['target_cy'])],
                'radius': float(max(hw, hh)), 'half_w': hw, 'half_h': hh}
    return {'type': 'circle', 'center': config.target.center, 'radius': config.target.radius}


def load_row(run_dir, label):
    """Re-solve one run's design and gather everything the four panels need."""
    config = load_and_parse_config(os.path.join(run_dir, "config.yaml"))
    params_path = os.path.join(run_dir, "best_params.pkl")
    init_state, _ = build_closed_initial_state(config)

    results, _tess, _be, _hg, _ca = solve_deployed_state(config, params_path)
    valid, mapped = results['valid_state'], results['mapped_state']
    disp = results['solution'].fields[-1]

    flat = np.asarray(reconstruct_vertices(valid.face_centroids, valid.centroid_node_vectors))
    depl = np.asarray(deformed_vertices(valid, disp))

    # Area change is read off the PANEL SHAPES, not the welded tessellation copy whose closed-sheet
    # corners alias. Same derivation as run_closed.
    a_init = np.asarray(compute_face_areas(init_state.centroid_node_vectors))
    a_trained = np.asarray(compute_face_areas(mapped.centroid_node_vectors))
    rel = a_trained / np.maximum(a_init, 1e-9) - 1.0

    stem = os.path.join(run_dir, "load_path")
    field = load_field(stem) if os.path.exists(stem + ".npz") else None
    if field is not None:
        err = float(np.abs(np.asarray(field.face_nodes) - depl).max())
        scale = float(np.ptp(depl.reshape(-1, 2), axis=0).max())
        if err > 1e-3 * scale:
            raise SystemExit(
                f"{run_dir}: the re-solved deployment disagrees with the saved load-path geometry "
                f"by {err:.3e} ({err/scale:.2%} of the sheet). Columns 2 and 3 would show the sheet "
                f"in two different places -- re-run the load-path extraction for this run.")

    # Column (c)'s colour: the NET stress state of each tile, tension positive. Normalised by the
    # row's own mean |p|, because the five rows are driven at very different load magnitudes and a
    # shared absolute scale would show only which row was pulled hardest. This is a different
    # reading from the compression wash in (d), which is one-sided and per-interface: a tile
    # squeezed at one hinge and pulled at three is shaded there and reads as tension here.
    if field is None:
        pressure = None
    else:
        pressure = tile_pressure(field)
        pressure = pressure / (float(np.abs(pressure).mean()) or 1.0)

    flat_xy = np.asarray(init_state.face_centroids)
    v = np.asarray(_shim(reconstruct_vertices(
        init_state.face_centroids, init_state.centroid_node_vectors)).vertices)
    return dict(label=label, run_dir=run_dir, config=config, rel=rel, field=field,
                pressure=pressure,
                flat=_shim(flat), depl=_shim(depl), target=_target_params(config),
                sheet_bbox=(v[:, 0].min(), v[:, 1].min(), v[:, 0].max(), v[:, 1].max()),
                face_xy=flat_xy)


def check_comparable(rows):
    """Fail loudly if the rows are not posed on the same sheet against the same target."""
    ref = rows[0]['config'].topology
    for r in rows[1:]:
        for k in SHARED_KEYS:
            a, b = ref.get(k), r['config'].topology.get(k)
            if a is None and b is None:
                continue
            if a is None or b is None or abs(float(a) - float(b)) > 1e-9:
                raise SystemExit(
                    f"rows disagree on '{k}': {rows[0]['label']} has {a}, {r['label']} has {b}. "
                    f"The figure claims one sheet and one target across five problems; it cannot "
                    f"make that claim from runs that differ here.")

    # Supports SHOULD also match -- the rows are meant to differ only in the actuation -- but a
    # load case can legitimately need its roller moved, so this warns rather than stops. It is
    # printed because a silently different support is indistinguishable from a different design.
    def sup(t):
        return (tuple(t.get('bc_clamped') or ()), tuple(t.get('clamped_dofs') or ()),
                tuple(t.get('x_roller_faces') or ()), tuple(t.get('y_roller_faces') or ()))
    for r in rows[1:]:
        if sup(r['config'].topology) != sup(ref):
            print(f"  ! {r['label']} is held differently from {rows[0]['label']}: "
                  f"{sup(r['config'].topology)} vs {sup(ref)}")


def _extent(ax, box, pad=0.06):
    """Square-free extent with a margin, so every row of a column frames identically."""
    x0, y0, x1, y1 = box
    dx, dy = (x1 - x0), (y1 - y0)
    ax.set_xlim(x0 - pad * dx, x1 + pad * dx)
    ax.set_ylim(y0 - pad * dy, y1 + pad * dy)
    ax.set_aspect('equal')
    ax.axis('off')


def _union(boxes):
    b = np.asarray(boxes, dtype=float)
    return (b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max())


def _bbox_of(pts, pad=0.0):
    p = np.asarray(pts, dtype=float).reshape(-1, 2)
    x0, y0 = p.min(axis=0)
    x1, y1 = p.max(axis=0)
    return (x0 - pad, y0 - pad, x1 + pad, y1 + pad)


def _target_corners(t):
    if t['type'] == 'rectangle':
        cx, cy = t['center']
        return np.array([[cx - t['half_w'], cy - t['half_h']], [cx + t['half_w'], cy + t['half_h']]])
    cx, cy = t['center']
    r = t['radius']
    return np.array([[cx - r, cy - r], [cx + r, cy + r]])


def _fit(ax, box, aspect):
    """Frame ``box`` in an axes of the given width/height, padding SYMMETRICALLY to that aspect.

    Every data column shares one box size, so each column's own content is centred inside it and
    nothing is stretched. Revision 1 forced one shared extent across all four columns instead, which
    left each panel floating at a different size inside its block; that is what read as unequal
    spacing between the columns.
    """
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    if w / h < aspect:
        extra = (aspect * h - w) / 2.0
        x0, x1 = x0 - extra, x1 + extra
    else:
        extra = (w / aspect - h) / 2.0
        y0, y1 = y0 - extra, y1 + extra
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect('equal')
    ax.axis('off')


def _wrap_to(text, max_bp, fontsize=FS_BASE):
    """Greedy wrap so no line of ``text`` is wider than ``max_bp`` points when rendered.

    The column titles are authored as fixed two-line strings, which fitted only the canvas they were
    written on. Wrapping to the MEASURED column width keeps each title inside its own column when
    either the width or the type size changes.
    """
    lines, current = [], ""
    for word in " ".join(text.split()).split(" "):
        trial = f"{current} {word}".strip()
        if current and text_width_bp(trial, fontsize) > max_bp:
            lines.append(current)
            current = word
        else:
            current = trial
    return "\n".join(lines + [current] if current else lines)


def build(rows, out_stem, *, dpi=600, clip_pct=90.0, stress_clip_pct=75.0,
          formats=("pdf", "png")):
    apply_charter(**paper_rc(**{"axes.grid": False}))
    n = len(rows)

    # ── One colour norm, pooled over every row, clipped to a percentile. ──
    # A handful of panels reach +200% while the bulk sit inside +-60%; scaling to the extreme would
    # spend the whole range on those few and wash every other panel out. The colorbar carries
    # `extend` arrows so the clipping is stated rather than hidden.
    pooled = np.concatenate([r['rel'] for r in rows])
    lim = float(max(np.percentile(np.abs(pooled), clip_pct), 1e-3))
    cmap = charter_diverging()
    norm = mcolors.TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
    n_clipped = int((np.abs(pooled) > lim).sum())
    print(f"  colour scale +-{lim:.0%} ({clip_pct}th percentile); "
          f"{n_clipped}/{pooled.size} panels beyond it")
    # The ladder is printed so the clip is chosen from the distribution rather than by eye: too
    # generous a limit spends the whole ramp on a handful of outliers and washes every other panel.
    print("  |area change| percentiles: "
          + "  ".join(f"p{q}={np.percentile(np.abs(pooled), q):.0%}"
                      for q in (50, 75, 90, 95, 98, 100)))

    # ── A second norm for column (c): the net tile stress. ──
    # Its own scale, because it is a different quantity from the area change of (b) and sharing one
    # ramp between them would imply they are commensurate. Same clip convention, same map.
    # ⚠ Clipped TIGHTER than the area field, and deliberately. The stress distribution is heavily
    # skewed -- median |p| is about 0.6 against a maximum near 12 -- so the ninetieth percentile
    # spends the ramp on a handful of tiles at the grip and leaves the rest of the sheet pale. The
    # structure worth seeing is the sign of the moderate field, not the magnitude of the extremes.
    have_pressure = [r for r in rows if r.get('pressure') is not None]
    if have_pressure:
        pooled_p = np.concatenate([r['pressure'] for r in have_pressure])
        lim_p = float(max(np.percentile(np.abs(pooled_p), stress_clip_pct), 1e-6))
        norm_p = mcolors.TwoSlopeNorm(vmin=-lim_p, vcenter=0.0, vmax=lim_p)
        n_clipped_p = int((np.abs(pooled_p) > lim_p).sum())
        resolved = int(((pooled_p < 0) & (np.abs(pooled_p) < lim_p)).sum())
        print(f"  stress scale +-{lim_p:.2f} x mean|p| ({stress_clip_pct}th percentile); "
              f"{n_clipped_p}/{pooled_p.size} tiles beyond it; "
              f"{resolved}/{int((pooled_p < 0).sum())} compressive tiles resolved below the clip")
        print(f"  tiles in net compression per row: "
              + "  ".join(f"{r['label'][:4]}={int((r['pressure'] < 0).sum())}"
                          for r in have_pressure))
    else:
        norm_p, n_clipped_p = None, 0

    # ── Per-column extents, pooled over rows. Each data column fills the SAME box, so the rows
    # align and every panel is drawn as large as it can be. The cost, accepted deliberately, is that
    # the undeployed and deployed sheets are no longer at one scale; the caption carries the true
    # dimensions instead.
    def col_box(kind):
        if kind == 'schematic':
            # Just enough for the glyphs: supports reach ~0.11 span to the left, arrows ~0.13 to the
            # right, and a transverse load sits ~0.14 above the top edge.
            b = _union([r['sheet_bbox'] for r in rows])
            sp = max(b[2] - b[0], b[3] - b[1])
            return (b[0] - 0.15 * sp, b[1] - 0.15 * sp, b[2] + 0.17 * sp, b[3] + 0.15 * sp)
        if kind == 'flat':
            b = _union([_bbox_of(r['flat'].vertices) for r in rows])
        elif kind == 'depl':
            b = _union([_bbox_of(r['depl'].vertices) for r in rows]
                       + [_bbox_of(_target_corners(r['target'])) for r in rows])
        else:
            # Load path: room for the BC glyphs, which sit outside the sheet. The margins are kept
            # close to (b) and (c) in ASPECT, not just in size -- the common box takes the widest of
            # the three, so a load-path column padded only sideways would inset the other two.
            b = _union([_bbox_of(r['depl'].vertices) for r in rows])
            sp = max(b[2] - b[0], b[3] - b[1])
            return (b[0] - 0.14 * sp, b[1] - 0.10 * sp, b[2] + 0.12 * sp, b[3] + 0.10 * sp)
        sp = max(b[2] - b[0], b[3] - b[1])
        return (b[0] - 0.03 * sp, b[1] - 0.03 * sp, b[2] + 0.03 * sp, b[3] + 0.03 * sp)

    boxes = {k: col_box(k) for k in ('schematic', 'flat', 'depl', 'path')}
    # The data columns share one box aspect: the widest of the three, so none of them is cropped.
    aspect = max((boxes[k][2] - boxes[k][0]) / (boxes[k][3] - boxes[k][1])
                 for k in ('flat', 'depl', 'path'))

    # Row labels are wrapped and RIGHT-aligned into the gutter, and the gutter is sized from the
    # longest wrapped line. Left-aligning them ran "Self-equilibrated shear couple" straight across
    # the schematic it was labelling. The width is MEASURED from the glyphs rather than counted in
    # characters at a fixed inch-per-character: that constant was tuned on a canvas 2.8x this one,
    # so it survived neither the resize nor the change of type size.
    wrapped = ["\n".join(textwrap.wrap(r['label'], 17)) for r in rows]
    label_w = max(text_width_bp(ln, FS_BASE) for w in wrapped for ln in w.split("\n"))
    roman_w = max(text_width_bp(r, FS_PANEL, weight="bold") for r in ROMAN[:n])
    gutter = label_w + roman_w + 3 * GAP_BP

    # The canvas width is fixed by the page, so the panels take what the gutter leaves rather than
    # the reverse. Everything downstream is derived from this one width.
    panel_w = (FIG_W_BP - gutter - RIGHT_BP) / (3.0 + SCHEM_FRAC)
    panel_h = panel_w / aspect
    # The schematic is narrower than a data panel: it carries no detail to inspect, and the width it
    # saves goes straight into fitting the figure on a page. It is also sized to its OWN aspect and
    # centred in the row rather than padded out to the row height -- forcing it to the full height
    # of a data panel is what put a large blank band above and below every diagram.
    schem_w = SCHEM_FRAC * panel_w
    schem_aspect = ((boxes['schematic'][2] - boxes['schematic'][0])
                    / (boxes['schematic'][3] - boxes['schematic'][1]))
    schem_h = min(schem_w / schem_aspect, panel_h)

    head, foot = HEAD_BP, FOOT_BP
    fig_w = FIG_W_BP
    fig_h = head + n * panel_h + foot
    fig = plt.figure(figsize=figsize_in(fig_w, fig_h), facecolor="white")

    lefts = [gutter, gutter + schem_w, gutter + schem_w + panel_w,
             gutter + schem_w + 2 * panel_w]
    widths = [schem_w, panel_w, panel_w, panel_w]

    def cell(row, col):
        h = schem_h if col == 0 else panel_h
        top = head + row * panel_h + (panel_h - h) / 2.0      # centred in the row band
        return fig.add_axes([lefts[col] / fig_w, 1.0 - (top + h) / fig_h,
                             widths[col] / fig_w, h / fig_h])

    # ── Header band: panel letter over a two-line title, closed by a hairline rule. The three
    # heights are absolute points from the canvas top, not fractions of the band: as fractions the
    # letter, the second title line and the rule all landed on each other once the band was sized
    # in points instead of inches.
    for j, (letter, title) in enumerate(zip("abcd", COLUMNS)):
        xc = (lefts[j] + widths[j] / 2.0) / fig_w
        panel_letter(fig, letter, x=xc, y=1.0 - LET_Y_BP / fig_h, ha="center", va="center")
        fig.text(xc, 1.0 - TITLE_Y_BP / fig_h, _wrap_to(title, widths[j] - GAP_BP), ha="center",
                 va="top", fontsize=FS_BASE, color=INK, linespacing=1.35)
    fig.add_artist(Line2D([gutter / fig_w, (fig_w - RIGHT_BP) / fig_w],
                          [1.0 - RULE_Y_BP / fig_h] * 2,
                          color=INK, lw=0.5, alpha=0.35))

    for i, r in enumerate(rows):
        colours = [cmap(norm(v)) for v in r['rel']]
        # (c) carried the SAME array as (b) until the stress field was added, so the two columns
        # drew one quantity twice and (c) said nothing (b) had not already said.
        stress_colours = ([cmap(norm_p(v)) for v in r['pressure']]
                          if r.get('pressure') is not None and norm_p is not None else colours)
        topo = r['config'].topology
        yc = 1.0 - (head + (i + 0.5) * panel_h) / fig_h

        fig.text((gutter - GAP_BP) / fig_w, yc, wrapped[i], ha="right", va="center",
                 fontsize=FS_BASE, color=INK, linespacing=1.45)
        fig.text((gutter - GAP_BP - label_w - GAP_BP) / fig_w, yc, ROMAN[i],
                 ha="right", va="center", fontsize=FS_PANEL, color=INK, weight="bold")

        ax = cell(i, 0)
        draw_loading_schematic(
            ax, r['sheet_bbox'], r['face_xy'],
            clamped_faces=topo.get('bc_clamped') or [], clamped_dofs=topo.get('clamped_dofs'),
            y_roller_faces=topo.get('y_roller_faces') or [],
            x_roller_faces=topo.get('x_roller_faces') or [], load_specs=topo.get('loads') or [],
            weight_scale=WEIGHT_SCALE)
        _fit(ax, boxes['schematic'], schem_aspect)

        ax = cell(i, 1)
        plot_tessellation(r['flat'], ax=ax, show_target=False, show_hinges=False,
                          show_face_indices=False, show_hinge_indices=False,
                          color_faces=colours, edge_color=TILE_EDGE, edge_width=TILE_EDGE_W,
                          face_alpha=1.0)
        _fit(ax, boxes['flat'], aspect)

        ax = cell(i, 2)
        plot_tessellation(r['depl'], ax=ax, show_target=True, target_params=r['target'],
                          show_hinges=False, show_face_indices=False, show_hinge_indices=False,
                          color_faces=stress_colours, edge_color=TILE_EDGE,
                          edge_width=TILE_EDGE_W,
                          face_alpha=1.0, target_color=INK, target_width=0.5,
                          target_style=(0, (3, 2.2)))
        _fit(ax, boxes['depl'], aspect)

        ax = cell(i, 3)
        if r['field'] is None:
            ax.axis('off')
        else:
            draw_load_path(ax, r['field'], tessellation=r['depl'], config=r['config'],
                           canvas_aspect=aspect, weight_scale=WEIGHT_SCALE,
                           shear_colour=SHEAR_DOT)
            _fit(ax, boxes['path'], aspect)

    # ── One colorbar under (b) and one under (c); the load-path key under (d). ──
    # They are separate bars because they carry different quantities. A single bar spanning both
    # columns, which is what stood here while (c) repeated (b)'s colours, would now assert that a
    # panel area change and a tile stress share a scale.
    def _colorbar(column, scalar_norm, clipped, label, formatter):
        cb_w = 0.92 * panel_w
        cax = fig.add_axes([(lefts[column] + (panel_w - cb_w) / 2.0) / fig_w,
                            0.50 * foot / fig_h, cb_w / fig_w, 0.085 * foot / fig_h])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=scalar_norm); sm.set_array([])
        cbar = fig.colorbar(sm, cax=cax, orientation="horizontal",
                            extend="both" if clipped else "neither")
        # ⚠ ONE line. Two bars now sit where one used to, so each has about six tenths of the width
        # the old label was written for. A second line is drawn below the canvas and silently lost,
        # which is what a wrapped label did here. What the scale MEANS belongs in the caption.
        cbar.set_label(label, fontsize=FS_BASE, color=INK)
        cbar.ax.xaxis.set_major_formatter(FuncFormatter(formatter))
        cbar.ax.tick_params(labelsize=FS_TICK, length=2.0, width=0.5)
        cbar.outline.set_visible(False)

    _colorbar(1, norm, n_clipped, "panel area change", lambda v, _: f"{v:+.0%}")
    if norm_p is not None:
        _colorbar(2, norm_p, n_clipped_p, "net tile stress", lambda v, _: f"{v:+.1f}")

    lax = fig.add_axes([lefts[3] / fig_w, 0.10 * foot / fig_h,
                        panel_w / fig_w, 0.62 * foot / fig_h])
    lax.axis("off")
    # ⚠ The shear key is a POINT, not a line. Measured on these five deployments, sheared interfaces
    # do not line up into slip planes: with every shear-dominated hinge admitted, collinearity yields
    # one plane against ninety-nine isolated contacts. Drawing a line in the legend would promise a
    # structure the mechanics does not show.
    lax.legend(handles=[Line2D([], [], color=ORANGE, lw=2.0, label="tension route"),
                        Patch(facecolor=COMPRESSION_FILL, edgecolor="none", label="compression"),
                        Line2D([], [], color=SHEAR_DOT, marker='o', linestyle='none', markersize=3.6,
                               markeredgecolor='white', markeredgewidth=0.5,
                               label="shear interface")],
               loc="center", ncol=1, frameon=False, fontsize=FS_ANNOT, labelcolor=INK,
               handlelength=1.5, labelspacing=0.45, borderpad=0.0)

    pdf, lo, hi = save_paper_figure(fig, out_stem, fig_w, fig_h, dpi=dpi,
                                    png=("png" in formats))
    print(f"  wrote {pdf} ({fig_w:.1f} x {fig_h:.1f} bp, type {lo:.1f}-{hi:.1f} pt)")
    plt.close(fig)
    write_caption(rows, out_stem)


def write_caption(rows, out_stem):
    """Emit the caption beside the figure.

    Columns (b) and (c) are drawn at DIFFERENT scales so each fills its box, which is what makes the
    tessellation legible; the true dimensions therefore have to be stated in words or the reader
    cannot recover how much the sheet grows on deployment.
    """
    topo = rows[0]['config'].topology
    ls = float(getattr(getattr(rows[0]['config'], 'hinge_model', None), 'length_scale', 406.4))
    flat = _bbox_of(rows[0]['flat'].vertices)
    depl = _union([_bbox_of(r['depl'].vertices) for r in rows])
    fw, fh = (flat[2] - flat[0]) * ls, (flat[3] - flat[1]) * ls
    dw, dh = (depl[2] - depl[0]) * ls, (depl[3] - depl[1]) * ls
    txt = (
        f"Inverse design of one kirigami sheet under five mechanical problems. A single "
        f"{int(topo['M'])} x {int(topo['N'])} closed sheet, {fw:.0f} x {fh:.0f} mm flat, is held "
        f"identically in every row by a pin and a roller on its left edge and is deployed to the "
        f"same target rectangle; the rows differ only in how it is actuated. "
        f"(a) Boundary conditions and applied loading. (b) The optimised cut pattern in its "
        f"undeployed, flat state, panels coloured by area relative to the uniform sheet. "
        f"(c) The deployed configuration against the target, dashed, tiles coloured by their net "
        f"stress, tension positive, each row scaled by its own mean magnitude. (d) The internal "
        f"force transmission once deployed, as tension routes, compressed tiles and sheared "
        f"interfaces. Panels (b) to (d) are each drawn to fill their frame and are therefore NOT "
        f"at a common scale: the sheet measures {fw:.0f} x {fh:.0f} mm undeployed and reaches "
        f"{dw:.0f} x {dh:.0f} mm deployed, a linear expansion of about "
        f"{(depl[2]-depl[0])/(flat[2]-flat[0]):.2f} and close to a doubling of area."
    )
    with open(f"{out_stem}.txt", "w") as fh:
        fh.write(txt + "\n")
    print(f"  wrote {out_stem}.txt")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--row", action="append", required=True, metavar="RUN_DIR:LABEL",
                   help="one run folder and its row label; repeat once per problem, in order")
    p.add_argument("--out", default="paper/Images/inverse_design_matrix")
    p.add_argument("--dpi", type=int, default=600)
    p.add_argument("--clip-pct", type=float, default=90.0,
                   help="percentile of |area change| the colour scale saturates at")
    p.add_argument("--stress-clip-pct", type=float, default=75.0,
                   help="percentile of |tile stress| column (c) saturates at; tighter than the "
                        "area clip because the stress distribution is far more skewed")
    p.add_argument("--no-png", action="store_true", help="PDF only (the PNG is slow)")
    p.add_argument("--skip-check", action="store_true",
                   help="draw rows that are NOT posed on the same sheet (layout proofing only)")
    args = p.parse_args()

    rows = []
    for spec in args.row:
        run_dir, _, label = spec.partition(":")
        if not label:
            label = os.path.basename(run_dir.rstrip("/"))
        print(f"  solving {label}  ({run_dir}) ...")
        rows.append(load_row(run_dir, label))

    if not args.skip_check:
        check_comparable(rows)
    build(rows, args.out, dpi=args.dpi, clip_pct=args.clip_pct,
          stress_clip_pct=args.stress_clip_pct,
          formats=("pdf",) if args.no_png else ("pdf", "png"))


if __name__ == "__main__":
    main()
