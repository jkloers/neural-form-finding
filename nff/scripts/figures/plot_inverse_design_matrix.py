"""The inverse-design matrix: one sheet, five mechanical problems, four readings each.

Each ROW is a boundary-value problem posed on the SAME closed sheet -- same tile count, same
material, same target rectangle -- and differing only in how the right boundary is actuated. Each
COLUMN is one reading of that row:

    0  the problem      the sheet as a plain rectangle, its supports and its loads
    1  trained design   the learned flat cut pattern, panels coloured by area change
    2  deployed         the same panels after Stage 2, against the target they were asked to
                        match, tiled by unit cells and coloured where like cells cluster
    3  load path        where the force actually goes once the sheet has opened

Columns 1 and 2 carry DIFFERENT quantities and therefore different colour. Column 1 is a diverging
purple-to-green ramp on a continuous field; column 2 is three CATEGORICAL fills, red, slate and
sand, at matched lightness, because a family is a kind and not an amount. The two shared one ramp
until 2026-08-19, and a reader with only the two colorbars to tell them apart read them as one
field.

The unit of column 2 is the DISJOINT unit cell -- the four panels around one void, called a 2x2
motif by the helpers in nff/closed/force_stats.py -- and
three facts fix that choice. A panel on its own carries almost no spatial signal: per-panel aspect
ratio classifies into families at 0.89 to 1.32 times chance contiguity, because deployment
counter-rotates neighbouring panels. Overlapping 2x2 windows share two panels with each neighbour,
and that overlap alone holds contiguity at 0.85 to 1.09. Tiling the sheet with NON-overlapping
blocks lifts it to 1.12 to 1.44 in every row.

A motif's geometry is the two side lengths of its void, not their ratio. Both are fixed by the cut
pattern and unchanged by deployment, they are near-independent (pooled correlation 0.16), and the
void's opening angle is uniform over a sheet to under half a degree. Collapsing them to one ratio
discards the motif's SIZE, which is what separates two of the three families.

A motif is coloured only where at least three same-family motifs touch. Similarity and proximity
are separate claims and the figure asserts both at once; the rest stay neutral grey.

The claim the figure makes is a comparison, so four things are held fixed across rows and must
stay that way: one colour norm per quantity pooled over every row, one axis extent per column, one
scale shared by columns 1 and 2, and one target rectangle. A per-row norm or a per-row target
would turn five comparable rows into five unrelated figures.

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
                                GREEN, GREY, INK, ORANGE, PURPLE, TEXT_W_BP,
                                apply_charter, charter_diverging, figsize_in, panel_letter,
                                paper_rc, save_paper_figure, text_width_bp)
from nff.config.experiment import load_and_parse_config                  # noqa: E402
from nff.closed.setup import build_closed_initial_state                  # noqa: E402
from nff.closed.cell_families import (cell_descriptors, cell_lattice,  # noqa: E402
                                      family_map, fit_families)
from nff.closed.void_geometry import build_void_index                    # noqa: E402
from nff.topology.closed_builder_jax import build_deploy_structure       # noqa: E402
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
PANEL_FILL = "#EDEFF1"         # a motif with no like neighbour: neutral, so colour on this panel
                               # means similarity AND proximity, never one of the two
FAMILY_FILL = ("#E7A9A7", "#A9BBD1", "#EBD5A3")   # red, slate, sand. CATEGORICAL: the families are
                               # kinds, not amounts, so the three sit at matched lightness and none
                               # reads as "more". Slate is unspent on the plate; (b) is
                               # purple-green and (d) is orange, teal and gold
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
           "Deployed configuration\nand unit cells",
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

    flat_xy = np.asarray(init_state.face_centroids)
    v = np.asarray(_shim(reconstruct_vertices(
        init_state.face_centroids, init_state.centroid_node_vectors)).vertices)
    return dict(label=label, run_dir=run_dir, config=config, rel=rel, field=field,
                depl_nodes=depl,
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


def _match_span(box, reference):
    """``box`` about its OWN centre, at ``reference``'s span, so two equal axes share one scale.

    Two boxes of equal span drawn into axes of equal size put one data unit at one size on the
    page. ``_fit`` then pads both symmetrically to the common aspect, and its padding depends only
    on the span, so the shared scale survives the fit.

    The centre is the box's own and NOT the reference's: deployment is anchored at the pin and the
    roller on the left edge, so sharing the centre as well would sit the flat sheet off-centre in
    five consecutive frames, which reads as a layout fault rather than as an anchor.
    """
    cx, cy = 0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])
    w, h = reference[2] - reference[0], reference[3] - reference[1]
    return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


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


def build(rows, out_stem, *, dpi=600, clip_pct=90.0, n_families=3, families=True,
          cluster_min=3, formats=("pdf", "png"), width_bp=None, weight_scale=None,
          no_headers=False, no_row_labels=False, columns="abcd"):
    apply_charter(**paper_rc(**{"axes.grid": False}))
    n = len(rows)

    # ── One colour norm, pooled over every row, clipped to a percentile. ──
    # A handful of panels reach +200% while the bulk sit inside +-60%; scaling to the extreme would
    # spend the whole range on those few and wash every other panel out. The colorbar carries
    # `extend` arrows so the clipping is stated rather than hidden.
    pooled = np.concatenate([r['rel'] for r in rows])
    lim = float(max(np.percentile(np.abs(pooled), clip_pct), 1e-3))
    # (b) and (c) carry different quantities, so they carry different ramps. Purple-to-green is
    # a charter pair spent nowhere else on this plate, and it separates from (c)'s red on lightness
    # as well as on hue, so the two columns stay distinct for a dichromat.
    cmap_area = charter_diverging(low=PURPLE, high=GREEN, name="charter_area")
    norm = mcolors.TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
    n_clipped = int((np.abs(pooled) > lim).sum())
    print(f"  colour scale +-{lim:.0%} ({clip_pct}th percentile); "
          f"{n_clipped}/{pooled.size} panels beyond it")
    # The ladder is printed so the clip is chosen from the distribution rather than by eye: too
    # generous a limit spends the whole ramp on a handful of outliers and washes every other panel.
    print("  |area change| percentiles: "
          + "  ".join(f"p{q}={np.percentile(np.abs(pooled), q):.0%}"
                      for q in (50, 75, 90, 95, 98, 100)))

    # ── Column (c): the unit-cell families. ──
    # The unit is the DISJOINT unit cell -- the four panels around one void -- and its geometry is
    # that void's two side lengths. Both are fixed by the cut pattern and neither changes on
    # deployment, because the panels are rigid. See the module docstring for what was measured
    # against what: the panel and the OVERLAPPING window both fail, and the side RATIO alone
    # throws away the size that separates two of the three families.
    #
    # The measurement itself lives in nff/closed/cell_families.py, so this plate and any analysis
    # of the same sheet read one implementation and cannot disagree.
    M, N = int(rows[0]['config'].topology['M']), int(rows[0]['config'].topology['N'])
    ls = float(getattr(getattr(rows[0]['config'], 'hinge_model', None), 'length_scale', 406.4))
    centroids = None
    if families:
        void_index = build_void_index(build_deploy_structure(M, N))
        lattice = cell_lattice(void_index)
        n_cells, covered = lattice.n_cells, lattice.owner_face.size
        print(f"  motif lattice: {lattice.shape[0]} x {lattice.shape[1]} disjoint 2x2 blocks, "
              f"{n_cells} per row, {covered}/{M * N} panels covered "
              f"({M * N - covered} unpaired on the last row)")

        for r in rows:
            r['motif_sides'] = cell_descriptors(void_index, lattice, r['depl_nodes'], ls).sides
        # Pooled over the five rows, so a family means the same shape in every one of them and the
        # columns stay comparable -- the same discipline the area colour norm follows.
        pooled = np.vstack([r['motif_sides'] for r in rows])
        model, labels = fit_families(pooled, n_families)
        centroids = model.centroids
        print("  void side percentiles (mm): "
              + "  ".join(f"long p{q}={np.percentile(pooled[:, 0], q):.0f}"
                          f"/short p{q}={np.percentile(pooled[:, 1], q):.0f}"
                          for q in (10, 50, 90)))
        print(f"  {n_families} families (k-means on log side lengths, pooled): "
              + "  ".join(f"{centroids[c, 0]:.0f}x{centroids[c, 1]:.0f}mm n={(labels == c).sum()}"
                          for c in range(n_families))
              + f"  of {len(labels)}")

        # Whether like motifs sit TOGETHER is a claim, so it is measured rather than left to the
        # eye. Below one is alternation, not scatter; a plain neighbour count cannot tell them
        # apart, which is why the ratio is printed against its own chance level.
        for i, r in enumerate(rows):
            fam = family_map(labels[i * n_cells:(i + 1) * n_cells], lattice, min_size=cluster_min)
            r['contiguity'] = fam['contiguity']
            r['n_clustered'] = fam['n_clustered']
            r['groups'] = fam['groups']
            fill = np.array([PANEL_FILL] * (M * N), dtype=object)
            for k, face_ids in enumerate(lattice.owner_face):
                if fam['mask'].ravel()[k]:
                    fill[face_ids] = FAMILY_FILL[int(fam['grid'].ravel()[k])]
            r['motif_fill'] = list(fill)
        print(f"  clustered at min group {cluster_min}: "
              + "  ".join(f"{r['label'][:4]}={r['n_clustered']}/{n_cells}"
                          f" ({r['contiguity']['ratio']:.2f}x chance)" for r in rows))
        print("  largest group per row: "
              + "  ".join(f"{r['label'][:4]}={r['groups'][0][1] if r['groups'] else 0}"
                          for r in rows))
        for r in rows:
            if r['contiguity']['ratio'] < 1.05 or r['n_clustered'] < 0.5 * n_cells:
                print(f"  ! {r['label'][:4]}: {r['n_clustered']}/{n_cells} clustered at "
                      f"{r['contiguity']['ratio']:.2f}x chance. Its motifs do NOT form regions, "
                      f"and the panel must not be read as if they do.")

    # ── Per-column extents, pooled over rows. Each data column fills the SAME box, so the rows
    # align and every panel is drawn as large as it can be. Columns (b) and (c) are then put on ONE
    # span below, so the sheet's growth on deployment is its size on the page rather than a
    # sentence in the caption. (d) keeps its own margin, which its support glyphs need.
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
    # ★ The closed sheet is drawn at the DEPLOYED sheet's scale, so (b) is smaller than (c) by
    # exactly the expansion the deployment buys. Reviewers read the old plate, where each column
    # filled its own frame, as a sheet that deploys without growing.
    flat_span = boxes['flat'][2] - boxes['flat'][0]
    boxes['flat'] = _match_span(boxes['flat'], boxes['depl'])
    # The data columns share one box aspect: the widest of the three, so none of them is cropped.
    aspect = max((boxes[k][2] - boxes[k][0]) / (boxes[k][3] - boxes[k][1])
                 for k in ('flat', 'depl', 'path'))

    # Row labels are wrapped and RIGHT-aligned into the gutter, and the gutter is sized from the
    # longest wrapped line. Left-aligning them ran "Self-equilibrated shear couple" straight across
    # the schematic it was labelling. The width is MEASURED from the glyphs rather than counted in
    # characters at a fixed inch-per-character: that constant was tuned on a canvas 2.8x this one,
    # so it survived neither the resize nor the change of type size.
    wrapped = ["\n".join(textwrap.wrap(r['label'], 17)) for r in rows]
    if no_row_labels:
        # The deck puts one row on a frame and names it in the frame title, so the numeral and the
        # case name inside the plate say it twice. The gutter is sized FROM those labels and
        # panel_w takes what it leaves, so dropping them is not only tidier: on a five-case build
        # at 432 bp the gutter falls from about 150 bp to 12 and every panel grows by half.
        label_w = roman_w = 0.0
        gutter = 2 * GAP_BP
    else:
        label_w = max(text_width_bp(ln, FS_BASE) for w in wrapped for ln in w.split("\n"))
        roman_w = max(text_width_bp(r, FS_PANEL, weight="bold") for r in ROMAN[:n])
        gutter = label_w + roman_w + 3 * GAP_BP

    # The canvas width is fixed by the page, so the panels take what the gutter leaves rather than
    # the reverse. Everything downstream is derived from this one width. ``width_bp`` overrides it
    # for a canvas that is not the manuscript page -- a projected slide, whose text block is both
    # wider and far shallower than a column.
    canvas_w = FIG_W_BP if width_bp is None else float(width_bp)
    # A subset of the four columns takes the whole measure between them, so a two-column build has
    # panels more than twice as wide as the four-column one. That is what it is for: the deck shows
    # the schematic and the force transmission alone on a frame. Blowing up a crop of the
    # four-column build instead would print every load-path weight 2.3x too heavy, because ws below
    # ties them to the panel they are drawn in. Here panel_w grows and ws grows with it.
    SEL = sorted({"abcd".index(c) for c in columns})
    assert SEL, "at least one column must be drawn"
    UNIT = {0: SCHEM_FRAC, 1: 1.0, 2: 1.0, 3: 1.0}
    # RIGHT_BP is the margin column (d)'s load arrows reach into. Without (d) it is not needed.
    right = RIGHT_BP if 3 in SEL else GAP_BP
    panel_w = (canvas_w - gutter - right) / sum(UNIT[k] for k in SEL)
    panel_h = panel_w / aspect

    # Load-path stroke weights are tied to the panel they are drawn in, not to the type: widening
    # the canvas without widening them leaves hairlines, narrowing it swallows the sheet. The ratio
    # is against the panel the module constant was tuned at, so the manuscript canvas gets exactly
    # WEIGHT_SCALE back and nothing about the published plate moves.
    paper_panel_w = (FIG_W_BP - gutter - RIGHT_BP) / (3.0 + SCHEM_FRAC)
    ws = WEIGHT_SCALE * panel_w / paper_panel_w if weight_scale is None else float(weight_scale)
    # The schematic is narrower than a data panel: it carries no detail to inspect, and the width it
    # saves goes straight into fitting the figure on a page. It is also sized to its OWN aspect and
    # centred in the row rather than padded out to the row height -- forcing it to the full height
    # of a data panel is what put a large blank band above and below every diagram.
    schem_w = SCHEM_FRAC * panel_w
    schem_aspect = ((boxes['schematic'][2] - boxes['schematic'][0])
                    / (boxes['schematic'][3] - boxes['schematic'][1]))
    schem_h = min(schem_w / schem_aspect, panel_h)

    # The header band is a letter over a two-line column title, closed by a rule. On a slide the
    # speaker names the columns once and the deck shows one row per frame, so five copies of the
    # same four titles is five copies of clutter. fig_h is derived from head, so dropping it
    # SHORTENS the canvas rather than growing the panels: the panels are set by panel_w, which the
    # four-column layout fixes. 6bp is left so the top row does not start on the trim.
    head, foot = (6.0 if no_headers else HEAD_BP), FOOT_BP
    fig_w = canvas_w
    fig_h = head + n * panel_h + foot
    fig = plt.figure(figsize=figsize_in(fig_w, fig_h), facecolor="white")

    # Keyed by the ORIGINAL column index, so every reference below still names the column it means
    # whatever subset is drawn, and an unselected one is simply absent.
    widths = {0: schem_w, 1: panel_w, 2: panel_w, 3: panel_w}
    lefts, _x = {}, gutter
    for _k in SEL:
        lefts[_k] = _x
        _x += widths[_k]

    def cell(row, col):
        if col not in lefts:
            return None
        h = schem_h if col == 0 else panel_h
        top = head + row * panel_h + (panel_h - h) / 2.0      # centred in the row band
        return fig.add_axes([lefts[col] / fig_w, 1.0 - (top + h) / fig_h,
                             widths[col] / fig_w, h / fig_h])

    # ── Header band: panel letter over a two-line title, closed by a hairline rule. The three
    # heights are absolute points from the canvas top, not fractions of the band: as fractions the
    # letter, the second title line and the rule all landed on each other once the band was sized
    # in points instead of inches.
    if not no_headers:
        for j, (letter, title) in enumerate(zip("abcd", COLUMNS)):
            xc = (lefts[j] + widths[j] / 2.0) / fig_w
            panel_letter(fig, letter, x=xc, y=1.0 - LET_Y_BP / fig_h, ha="center", va="center")
            fig.text(xc, 1.0 - TITLE_Y_BP / fig_h, _wrap_to(title, widths[j] - GAP_BP),
                     ha="center", va="top", fontsize=FS_BASE, color=INK, linespacing=1.35)
        fig.add_artist(Line2D([gutter / fig_w, (fig_w - RIGHT_BP) / fig_w],
                              [1.0 - RULE_Y_BP / fig_h] * 2,
                              color=INK, lw=0.5, alpha=0.35))

    for i, r in enumerate(rows):
        colours = [cmap_area(norm(v)) for v in r['rel']]
        topo = r['config'].topology
        yc = 1.0 - (head + (i + 0.5) * panel_h) / fig_h

        if not no_row_labels:
            fig.text((gutter - GAP_BP) / fig_w, yc, wrapped[i], ha="right", va="center",
                     fontsize=FS_BASE, color=INK, linespacing=1.45)
            fig.text((gutter - GAP_BP - label_w - GAP_BP) / fig_w, yc, ROMAN[i],
                     ha="right", va="center", fontsize=FS_PANEL, color=INK, weight="bold")

        ax = cell(i, 0)
        if ax is not None:
            draw_loading_schematic(
                ax, r['sheet_bbox'], r['face_xy'],
                clamped_faces=topo.get('bc_clamped') or [], clamped_dofs=topo.get('clamped_dofs'),
                y_roller_faces=topo.get('y_roller_faces') or [],
                x_roller_faces=topo.get('x_roller_faces') or [], load_specs=topo.get('loads') or [],
                weight_scale=ws)
            _fit(ax, boxes['schematic'], schem_aspect)

        ax = cell(i, 1)
        if ax is not None:
            plot_tessellation(r['flat'], ax=ax, show_target=False, show_hinges=False,
                              show_face_indices=False, show_hinge_indices=False,
                              color_faces=colours, edge_color=TILE_EDGE, edge_width=TILE_EDGE_W,
                              face_alpha=1.0)
            _fit(ax, boxes['flat'], aspect)

        ax = cell(i, 2)
        if ax is not None:
            plot_tessellation(r['depl'], ax=ax, show_target=True, target_params=r['target'],
                              show_hinges=False, show_face_indices=False, show_hinge_indices=False,
                              color_faces=r.get('motif_fill', [PANEL_FILL] * len(r['rel'])),
                              edge_color=TILE_EDGE, edge_width=TILE_EDGE_W,
                              face_alpha=1.0, target_color=INK, target_width=0.5,
                              target_style=(0, (3, 2.2)))
            _fit(ax, boxes['depl'], aspect)

        ax = cell(i, 3)
        if ax is not None:
            if r['field'] is None:
                ax.axis('off')
            else:
                draw_load_path(ax, r['field'], tessellation=r['depl'], config=r['config'],
                               canvas_aspect=aspect, weight_scale=ws,
                               shear_colour=SHEAR_DOT)
                _fit(ax, boxes['path'], aspect)

    # ── One colorbar under (b) and one under (c); the load-path key under (d). ──
    # They are separate bars because they carry different quantities. A single bar spanning both
    # columns, which is what stood here while (c) repeated (b)'s colours, would now assert that a
    # panel area change and a tile stress share a scale.
    def _colorbar(column, colour_map, scalar_norm, clipped, label, formatter):
        cb_w = 0.92 * panel_w
        cax = fig.add_axes([(lefts[column] + (panel_w - cb_w) / 2.0) / fig_w,
                            0.50 * foot / fig_h, cb_w / fig_w, 0.085 * foot / fig_h])
        sm = plt.cm.ScalarMappable(cmap=colour_map, norm=scalar_norm); sm.set_array([])
        cbar = fig.colorbar(sm, cax=cax, orientation="horizontal",
                            extend="both" if clipped else "neither")
        # ⚠ ONE line. Two bars now sit where one used to, so each has about six tenths of the width
        # the old label was written for. A second line is drawn below the canvas and silently lost,
        # which is what a wrapped label did here. What the scale MEANS belongs in the caption.
        cbar.set_label(label, fontsize=FS_BASE, color=INK)
        cbar.ax.xaxis.set_major_formatter(FuncFormatter(formatter))
        cbar.ax.tick_params(labelsize=FS_TICK, length=2.0, width=0.5)
        cbar.outline.set_visible(False)

    if 1 in lefts:
        _colorbar(1, cmap_area, norm, n_clipped, "panel area change", lambda v, _: f"{v:+.0%}")
    # ⚠ Column (c) carries NO key. The three family fills are categorical and the plate says so by
    # using three colours at matched lightness; naming their sizes on the figure spent a 96 bp
    # column on four labels and printed a number the reader cannot use while looking at the sheet.
    # The dimensions and how they were measured moved into the body of sec:physical-inverse, which
    # is where a reader who wants them is already standing. Removed 2026-08-21 at the author's
    # instruction; the foot band under (c) is deliberately empty.

    lax = fig.add_axes([lefts[3 if 3 in lefts else SEL[-1]] / fig_w, 0.10 * foot / fig_h,
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

    # The caption quotes these ratios, so they are measured here rather than recomputed by hand.
    depl_span = boxes['depl'][2] - boxes['depl'][0]
    print(f"  frames: (b) and (c) share a {depl_span:.2f} x "
          f"{boxes['depl'][3] - boxes['depl'][1]:.2f} span; the flat sheet draws at "
          f"{flat_span / depl_span:.2f} of the deployed one")
    print(f"  (d) is drawn at {depl_span / (boxes['path'][2] - boxes['path'][0]):.2f}x that "
          f"scale, the margin its boundary glyphs need")

    pdf, lo, hi = save_paper_figure(fig, out_stem, fig_w, fig_h, dpi=dpi,
                                    png=("png" in formats))
    print(f"  wrote {pdf} ({fig_w:.1f} x {fig_h:.1f} bp, type {lo:.1f}-{hi:.1f} pt)")
    plt.close(fig)
    write_caption(rows, out_stem, centroids=centroids, cluster_min=cluster_min)


def write_caption(rows, out_stem, *, centroids=None, cluster_min=3):
    """Emit the caption beside the figure.

    Every numeral here is measured off the drawn geometry rather than written down, so the caption
    cannot drift from the plate. This file is the authority the manuscript caption is checked
    against, not the reverse.
    """
    topo = rows[0]['config'].topology
    ls = float(getattr(getattr(rows[0]['config'], 'hinge_model', None), 'length_scale', 406.4))
    flat = _union([_bbox_of(r['flat'].vertices) for r in rows])
    depl = _union([_bbox_of(r['depl'].vertices) for r in rows])
    fw, fh = (flat[2] - flat[0]) * ls, (flat[3] - flat[1]) * ls
    dw, dh = (depl[2] - depl[0]) * ls, (depl[3] - depl[1]) * ls
    linear, areal = (depl[2] - depl[0]) / (flat[2] - flat[0]), (dw * dh) / (fw * fh)
    family = ""
    if centroids is not None:
        words = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
        # The three mean sizes are NOT quoted here. They live in the body of sec:physical-inverse
        # now, and this file is what the manuscript caption is checked against, so a number left
        # here would put the caption and the section permanently out of step.
        n_motifs = len(rows[0]['motif_sides'])
        marked = [r['n_clustered'] for r in rows]
        family = (
            f"The sheet is tiled by {n_motifs} disjoint unit cells, each the four panels around "
            f"one void. A unit cell is classified by that void's two side lengths, which the cut "
            f"pattern fixes and deployment leaves unchanged. Pooled over all five designs these "
            f"form {words.get(len(centroids), len(centroids))} families; "
            f"the colours are categorical and carry no order. A unit cell takes its family colour "
            f"only where at least {words.get(cluster_min, cluster_min)} cells of the same family "
            f"touch, so colour asserts similar geometry and proximity together. Between "
            f"{min(marked)} and {max(marked)} cells are coloured in a row; the rest are grey, "
            f"as is the final panel column, which the tiling leaves unpaired. ")
    txt = (
        f"Inverse design of one kirigami sheet under five mechanical problems. A single "
        f"{int(topo['M'])} x {int(topo['N'])} closed sheet, {fw:.0f} x {fh:.0f} mm flat, is held "
        f"identically in every row by a pin and a roller on its left edge and is deployed to the "
        f"same target rectangle; the rows differ only in how it is actuated. "
        f"(a) Boundary conditions and applied loading. (b) The optimised cut pattern in its "
        f"undeployed, flat state, panels coloured by area relative to the uniform sheet, on a "
        f"purple-to-green scale of its own. "
        f"(c) The deployed configuration against the target, dashed. {family}"
        f"(d) The internal "
        f"force transmission once deployed, as tension routes, compressed panels and sheared "
        f"interfaces. Panels (b) and (c) are drawn at ONE scale, so the growth on deployment is "
        f"the difference in size on the page: the sheet measures {fw:.0f} x {fh:.0f} mm "
        f"undeployed and reaches {dw:.0f} x {dh:.0f} mm deployed, a linear expansion of "
        f"{linear:.2f} and an areal expansion of {areal:.2f}. Panel (d) is drawn slightly smaller, "
        f"the extra margin its support glyphs need."
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
    p.add_argument("--columns", default="abcd",
                   help="which of the four columns to draw, as a subset of abcd: (a) loading, "
                        "(b) undeployed cut pattern, (c) deployed against target, (d) force "
                        "transmission. The selected columns share the whole measure, so fewer "
                        "columns means larger panels")
    p.add_argument("--no-row-labels", action="store_true",
                   help="omit the numeral and case name in the left gutter, and give the gutter "
                        "to the panels. For a deck that shows one row per frame and names it in "
                        "the frame title")
    p.add_argument("--no-headers", action="store_true",
                   help="omit the column letters, titles and rule. A slide has a speaker to name "
                        "the columns, and the deck shows one row per frame")
    p.add_argument("--dpi", type=int, default=600)
    p.add_argument("--clip-pct", type=float, default=90.0,
                   help="percentile of |area change| the colour scale saturates at")
    p.add_argument("--n-families", type=int, default=3,
                   help="how many geometric families to split the 2x2 motifs into")
    p.add_argument("--cluster-min", type=int, default=3,
                   help="smallest run of touching same-family motifs that counts as a cluster; "
                        "motifs below it are drawn neutral")
    p.add_argument("--no-families", action="store_true",
                   help="draw column (c) with no motif colouring (layout proofing)")
    p.add_argument("--width-bp", type=float, default=None,
                   help="canvas width in points; default is the manuscript's 1.2\\textwidth. Set it "
                        "to the text block of another medium, e.g. a 16:9 slide, and the load-path "
                        "stroke weights follow the panel width automatically")
    p.add_argument("--weight-scale", type=float, default=None,
                   help="override the load-path stroke weight instead of deriving it from the width")
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
          n_families=args.n_families, families=not args.no_families,
          cluster_min=args.cluster_min,
          formats=("pdf",) if args.no_png else ("pdf", "png"),
          width_bp=args.width_bp, weight_scale=args.weight_scale,
          no_headers=args.no_headers, no_row_labels=args.no_row_labels,
          columns=args.columns)


if __name__ == "__main__":
    main()
