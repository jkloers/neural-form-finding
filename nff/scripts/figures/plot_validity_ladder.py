"""The validity ladder: why the cut pattern has to be built in rather than enforced.

Four mechanisms attempt the same task on the same sheet -- deploy to the target circle, subject to
the flat sheet being a perfect closed cut pattern. Three enforce validity AFTER the fact, by
projection and by penalties; the fourth cannot express an invalid sheet at all.

The colour carries one meaning each and nothing else:

    orange              material, wherever it lies -- including outside the stock
    orange + hatching   material lying on top of other material
    RED                 an open void: stock area no panel covers

so a reader who is told only "red is a hole in the sheet" can read every column. The last patched
rung's remaining defect is millimetric on a 2.8 m sheet and invisible at figure scale, so each
column carries a magnified inset at the magnification its own worst defect requires; the chart is
shown at the deepest magnification any rung needed, making "nothing at N x" like-for-like.

    JAX_PLATFORMS=cpu python nff/scripts/figures/plot_validity_ladder.py
"""

import argparse
import os
import pickle

import numpy as np

from nff.utils.figstyle import (FS_ANNOT, FS_BASE, FS_MICRO, GREY, INK, ORANGE, RED,
                                apply_charter, figsize_in, page_bp, panel_letter, paper_rc,
                                save_paper_figure, use_agg)

use_agg()
import matplotlib.pyplot as plt                                        # noqa: E402
from matplotlib.patches import FancyBboxPatch
from matplotlib.gridspec import GridSpec                               # noqa: E402
from matplotlib.patches import Circle, Polygon as MplPolygon           # noqa: E402
from shapely.geometry import LinearRing, Point                        # noqa: E402

DRAWN = ['naive', 'nonoverlap', 'tiled', 'chart']
OURS = 'chart'
TITLE = {'naive': 'free vertices',
         'nonoverlap': 'projection\n+ overlap penalty',
         'tiled': '+ tiling penalty',
         'chart': 'by construction'}   # the paper's own contrast: 'by construction
                                       # rather than by penalty' (sn-article.tex)
KERF = 0.41          # mm: the manufacturability threshold, about twice the laser kerf

# Matplotlib fixes hatch density at 6 lines per repeat per inch in device space, and this plate is
# authored at its printed size, so 7 repeats is 42 lines/inch = 0.6 mm on the page. At the previous
# '///' (1.4 mm) an overlap sliver caught one or two strokes and read as stray dashes, not a
# texture.
HATCH_OVERLAP = '///////'

MIN_GAP_BP = 2.0     # clear space demanded between a key and its right-aligned value

# The canvas is authored at the size it prints, so an authored point size IS a printed point size.
# The plate was previously drawn 835 bp wide and reduced to \linewidth, which put every label
# between 3.2 and 4.1 pt on the page and the panel edges at 0.16 pt, below what a press resolves.
FIG_W_BP, FIG_H_BP = page_bp(1.0, 1.35)      # \includegraphics[width=1\linewidth]
TITLE_TOP = 0.940    # figure y of the letter + column-title band, shared by all four columns
TITLE_DX  = 0.023    # letter-to-title offset; the title is LEFT-aligned, never centred on the
                     # column, so no title length can ever run back into its own letter
GROUP_TOP = 0.990    # figure y of the baseline / this-work box tops


def void_area_m2(r, scale):
    """Residual void AREA in m^2 -- the physical quantity, summed over the void polygons.

    The earlier figure reported voids as an 'equivalent uniform gap' in mm, obtained by spreading
    the missing area along the interior cut length. That is a derived length, not a measurement,
    and invited comparison with the hinge gap, which is a different thing entirely. The area is
    what is actually computed and what a reader can check against the sheet.
    """
    return sum(poly_area(c) for c in (r.get('void_polys') or [])) * scale ** 2 / 1e6


def chamfer_mm(r, target, scale, n_ring=720):
    """Symmetric Chamfer distance between the deployed outline and the target circle, in mm.

    Mean of the two directions -- outline-to-circle (precision) and circle-to-outline (coverage) --
    rather than the max of their RMS values used while tuning. Distances are taken to the outline
    POLYGON, not to its vertices, so the score is not floored by vertex spacing.
    """
    b = np.asarray(r['deployed_boundary'])
    c = np.array([target['cx'], target['cy']])
    ring = LinearRing(b[np.argsort(np.arctan2(b[:, 1] - c[1], b[:, 0] - c[0]))])
    t = np.linspace(0.0, 2.0 * np.pi, n_ring, endpoint=False)
    circle = c + target['r'] * np.stack([np.cos(t), np.sin(t)], axis=-1)
    precision = np.abs(np.linalg.norm(b - c, axis=1) - target['r']).mean()
    coverage = float(np.mean([ring.distance(Point(q)) for q in circle]))
    return 0.5 * (precision + coverage) * scale


def poly_area(c):
    x, y = c[:, 0], c[:, 1]
    return abs(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def worst_defect(r, scale):
    """Where to point the magnifier, and how far it must zoom to make the failure visible.

    Voids are ranked first because they are what the figure paints red; overlap is the fallback for
    a rung that has closed the sheet but still stacks panels.
    """
    for polys in (r.get('void_polys') or [], r.get('overlap_polys') or []):
        if polys:
            areas = [poly_area(c) for c in polys]
            c = polys[int(np.argmax(areas))]
            return c.mean(axis=0), np.sqrt(max(areas)) * scale
    return None, 0.0


def draw_flat(ax, r, stock, scale, lw=0.3, stock_lw=0.6):
    """The flat sheet: material orange everywhere, overlap hatched, voids red."""
    ax.plot(stock[:, 0] * scale, stock[:, 1] * scale, '--', color=GREY, lw=stock_lw,
            dashes=(3, 2), zorder=8)
    for f in range(r['nodes'].shape[0]):
        ax.add_patch(MplPolygon(r['nodes'][f] * scale, closed=True, facecolor=ORANGE,
                                edgecolor=INK, linewidth=lw, zorder=3))
    for c in r.get('overlap_polys') or []:
        ax.add_patch(MplPolygon(c * scale, closed=True, facecolor=ORANGE,
                                edgecolor=INK, linewidth=0.0, hatch=HATCH_OVERLAP, zorder=5))
    for c in r.get('void_polys') or []:
        ax.add_patch(MplPolygon(c * scale, closed=True, facecolor=RED,
                                edgecolor='none', zorder=6))


def draw_deployed(ax, r, target, scale, lw=0.3):
    """The deployed sheet against the target circle."""
    t = target
    ax.add_patch(Circle((t['cx'] * scale, t['cy'] * scale), t['r'] * scale, fill=False,
                        edgecolor=INK, lw=0.9, linestyle=(0, (3, 2)), zorder=9))
    nodes = r.get('deployed_nodes')
    if nodes is None:
        ax.text(0.5, 0.5, 'deployment\nfailed', transform=ax.transAxes, ha='center',
                va='center', fontsize=FS_ANNOT, color=RED)
        return
    for f in range(np.asarray(nodes).shape[0]):
        ax.add_patch(MplPolygon(np.asarray(nodes)[f] * scale, closed=True, facecolor=ORANGE,
                                edgecolor=INK, linewidth=lw, zorder=3))


def bare(ax):
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor('none')          # let a group box's tint show through
    for s in ax.spines.values():
        s.set_visible(False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pkl", default="data/outputs/validity_ladder/ladder_7x7_r2295.pkl")
    ap.add_argument("--no-boxes", action="store_true",
                    help="omit the two grouping rules over the columns, keeping the construction "
                         "column's ground. A slide has a speaker to group the columns")
    ap.add_argument("--out", default="paper/Images/validity_ladder.png")
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--one-scale", action="store_true",
                    help="draw the flat row and the deployed row on one window at one scale, so "
                         "the expansion on opening reads as a change of size")
    ap.add_argument("--arms", default=None,
                    help="comma-separated subset of " + ",".join(DRAWN) + ". Default draws all "
                         "four. One arm on its own is a single column")
    ap.add_argument("--width-bp", type=float, default=None,
                    help="canvas width. Default is the page column; a slide wants its own")
    ap.add_argument("--aspect", type=float, default=1.35,
                    help="canvas width / height")
    ap.add_argument("--numbers-array", action="store_true",
                    help="the numbers as a table: one row per quantity, labelled once at the "
                         "left, instead of a key and a value repeated in every column")
    ap.add_argument("--no-titles", action="store_true",
                    help="omit the column letters and titles, and give the band they sat in to "
                         "the drawings")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="printed width / authored width. Type is authored at target/scale so it PRINTS at the ladder size. A plate shown on a slide at 0.75 of its authored width takes its 8pt labels to 6pt unless this is set")
    args = ap.parse_args()

    # Hatch strokes are 1.0 pt by default, which on a 370 bp plate fills an overlap sliver solid
    # black and turns the texture into debris. At the 0.6 mm spacing of HATCH_OVERLAP a 0.40 pt
    # stroke is about a fifth ink, which reads as stripes rather than silting up in the narrowest
    # slivers.
    apply_charter(**paper_rc(args.scale, **{"hatch.linewidth": 0.40}))
    with open(args.pkl, 'rb') as fh:
        D = pickle.load(fh)
    stock, scale = D['stock'], D['scale']
    A_MM, CUT, TGT = D['stock_area_mm2'], D['cut_len_mm'], D['target']
    by_key = {r['key']: r for r in D['rungs']}
    # --arms narrows the comparison. One arm on its own is how the deck illustrates the
    # construction: the 'chart' rung is the closed chart with no equilibrium anywhere in it, so its
    # flat sheet and its deployed sheet are a purely kinematic cut-and-open pair.
    _want = args.arms.split(",") if args.arms else DRAWN
    drawn = [k for k in DRAWN if k in by_key and k in _want]
    assert drawn, f"no arm of {_want} is in the harvest"
    sheet_mm = (stock[:, 0].max() - stock[:, 0].min()) * scale
    n_panels = D['shape'][0]
    EP = next((R['epochs'] for R in by_key.values() if R.get('epochs')), None)

    # ── ONE frame for row 1 and one for row 2, so the dashed squares are identical across
    # columns and the eye can compare them directly.
    pts = [stock * scale] + [by_key[k]['exemplar']['nodes'].reshape(-1, 2) * scale for k in drawn]
    P = np.vstack(pts)
    m = 0.03 * max(np.ptp(P[:, 0]), np.ptp(P[:, 1]))
    FRAME = (P[:, 0].min() - m, P[:, 0].max() + m,
             P[:, 1].min() - m - 0.07 * (P[:, 1].max() - P[:, 1].min()), P[:, 1].max() + m)

    dep = [np.asarray(by_key[k]['exemplar']['deployed_nodes']).reshape(-1, 2) * scale
           for k in drawn if by_key[k]['exemplar'].get('deployed_nodes') is not None]
    c_mm = np.array([TGT['cx'], TGT['cy']]) * scale
    ring = np.vstack([c_mm + TGT['r'] * scale, c_mm - TGT['r'] * scale])
    Q = np.vstack(dep + [ring]) if dep else ring
    m2 = 0.03 * max(np.ptp(Q[:, 0]), np.ptp(Q[:, 1]))
    FRAME2 = (Q[:, 0].min() - m2, Q[:, 0].max() + m2,
              Q[:, 1].min() - m2 - 0.07 * (Q[:, 1].max() - Q[:, 1].min()), Q[:, 1].max() + m2)

    # --one-scale puts the flat row and the deployed row on ONE window, so the growth on opening is
    # a difference in size on the page rather than two drawings fitted to their own cells. This is
    # what fig:inverse-design-matrix does for its own columns (b) and (c).
    # Both conditions are needed. bare() sets an equal aspect, so a shared window alone is not
    # enough: the drawing is fitted to its box, and the two rows' boxes differ in height (2.5
    # against 2.0), which puts the scale of one on its width and the other on its height. The row
    # heights are equalised below for the same reason.
    if args.one_scale:
        R = np.vstack([P, Q])
        mr = 0.03 * max(np.ptp(R[:, 0]), np.ptp(R[:, 1]))
        FRAME = FRAME2 = (R[:, 0].min() - mr, R[:, 0].max() + mr,
                          R[:, 1].min() - mr - 0.07 * (R[:, 1].max() - R[:, 1].min()),
                          R[:, 1].max() + mr)

    # ONE window size for all columns -- see module docstring. Sized so the SMALLEST defect in
    # the figure is clearly legible; bigger defects overflow it, which is honest since they are
    # already obvious at sheet scale. The chart, having no defect, is centred on its own sheet.
    sizes = [sz for sz in (worst_defect(by_key[k]['exemplar'], scale)[1] for k in drawn) if sz > 0]
    win_mm = float(np.clip(3.0 * min(sizes) if sizes else 300.0, 120.0, 900.0))
    half = win_mm / (2 * scale)
    plan = {}
    for key in drawn:
        ctr, size = worst_defect(by_key[key]['exemplar'], scale)
        if ctr is None:
            ctr = by_key[key]['exemplar']['nodes'].reshape(-1, 2).mean(axis=0)
        plan[key] = (np.asarray(ctr), half)

    # A page column is 370.6 bp wide and this plate is authored at aspect 1.35, which on a 16:9
    # body of aspect 2.37 is height-bound: it reaches 57 per cent of the measure and leaves the
    # rest empty. Authoring it flatter and wider is what fills the slide. Four columns of square
    # sheets in two rows caps the aspect near 2.0, so 2.0 is the useful end of the range.
    global FIG_W_BP, FIG_H_BP
    if args.width_bp:
        FIG_W_BP = float(args.width_bp)
    FIG_H_BP = FIG_W_BP / float(args.aspect)
    fig = plt.figure(figsize=figsize_in(FIG_W_BP, FIG_H_BP))
    # top=0.855 reserves the band that holds the letters, the column titles and the tops of the
    # two group boxes. With all three gone there is nothing in it, so it goes to the drawings.
    _r2 = 2.5 if args.one_scale else 2.0      # equal boxes, so one window means one scale
    # --numbers-array labels its three rows ONCE, at the left, so the left margin has to hold
    # "Chamfer / radius" rather than only a rotated row name. The panels give up about eight per
    # cent of their width for it, which is what turns three floating values per column into a
    # table the eye can read down as well as across.
    _left = 0.150 if args.numbers_array else 0.075
    gs = GridSpec(3, len(drawn), height_ratios=[2.5, _r2, 0.95 if args.numbers_array else 1.05],
                  hspace=0.16, wspace=0.10,
                  left=_left, right=0.985, bottom=0.105, figure=fig,
                  top=(0.975 if (args.no_titles and args.no_boxes) else 0.855))

    numbers_text = []                     # checked for collisions once the layout is final
    # Row baselines for --numbers-array, as a fraction of the number band's own height.
    ARRAY_Y = (0.80, 0.50, 0.20)
    array_keys, array_first = [], {}
    for col, key in enumerate(drawn):
        R, r = by_key[key], by_key[key]['exemplar']
        ours = key == OURS
        ctr, half = plan[key]

        # ── row 1: the flat cut pattern, with the magnifier as an inset
        ax = fig.add_subplot(gs[0, col])
        draw_flat(ax, r, stock, scale)
        ax.set_xlim(FRAME[0], FRAME[1]); ax.set_ylim(FRAME[2], FRAME[3]); bare(ax)


        # Letter and title are FIGURE text on one shared top edge rather than an axes title, so a
        # two-line title cannot drop its own column's letter out of line with the other three.
        bb = gs[0, col].get_position(fig)
        panel_letter(fig, 'abcd'[col], x=bb.x0, y=TITLE_TOP, va='top')
        if not args.no_titles:
            fig.text(bb.x0 + TITLE_DX, TITLE_TOP, TITLE.get(key, R['label']), ha='left',
                     va='top', fontsize=FS_BASE, linespacing=1.4,
                     color=(ORANGE if ours else INK),
                     fontweight=('bold' if ours else 'normal'))

        # ── row 2: the deployed sheet against the target circle
        ad = fig.add_subplot(gs[1, col])
        draw_deployed(ad, r, TGT, scale)
        ad.set_xlim(FRAME2[0], FRAME2[1]); ad.set_ylim(FRAME2[2], FRAME2[3]); bare(ad)


        # ── row 3: three numbers, each anchored to something visible above
        at = fig.add_subplot(gs[2, col]); at.axis('off'); at.set_facecolor('none')
        live = [x for x in R['runs'] if not x.get('stalled')] or R['runs']
        va = float(np.median([void_area_m2(x, scale) for x in live]))
        ch = float(np.median([chamfer_mm(x, TGT, scale) for x in live
                              if x.get('deployed_boundary') is not None] or [np.nan]))
        # Every row is a percentage, so the four columns compare straight down the page and the
        # reader carries one unit rather than three. Voids used to be an area alone, which left
        # them without a denominator and unrelatable to the two rows beneath it. The absolutes are
        # in the text, where a sentence can hold both without crowding.
        ov = R['med_overlap']
        rows = [('residual voids', 'none' if va < 1e-6 else f'{100 * va * 1e6 / A_MM:.1f}%'),
                ('overlap', 'none' if ov < 1e-9 else f'{100 * ov:.2f}%'),
                ('Chamfer / radius',
                 '--' if not np.isfinite(ch) else f'{100 * ch / (TGT["r"] * scale):.2f}%')]
        ok = (R['med_tiling'] < 1e-9 and R['med_overlap'] < 1e-9 and R['med_gap'] < 1e-9)
        if args.numbers_array:
            # One ROW per quantity, each value centred in its own column, so the block reads down
            # a column and across a row. The keys are drawn once at the left, after this loop:
            # repeating them in all four columns is what the page figure does, and on a slide it is
            # four copies of the same three words.
            array_keys = [k for k, _ in rows]
            for i, (k, v) in enumerate(rows):
                good = v in ('none',)
                shape = k.startswith('Chamfer')
                t_val = at.text(0.5, ARRAY_Y[i], v, fontsize=FS_ANNOT,
                    ha='center', va='center', transform=at.transAxes,
                    color=(ORANGE if good else (RED if not shape else (INK if ok else GREY))),
                    fontweight=('bold' if good else 'normal'))
                if col == 0:
                    array_first[i] = t_val
            continue
        for i, (k, v) in enumerate(rows):
            good = v in ('none',)
            shape = k.startswith('Chamfer')
            # A shape error describes an object that cannot be built unless the sheet is cuttable,
            # so it is greyed wherever the sheet has voids or overlap. The number is still printed
            # -- a baseline beats the chart on shape, and hiding that would be dishonest -- but it
            # is scoped to a design with holes in it.
            # The grey key drops to the floor size and the value keeps the annotation size. At one
            # size "Chamfer / radius" and its number fill the column and meet in the middle; the
            # contrast also puts the weight on the measurement rather than on its name.
            t_key = at.text(0.0, 0.90 - i * 0.26, k, fontsize=FS_MICRO, color=GREY,
                            transform=at.transAxes)
            t_val = at.text(1.0, 0.90 - i * 0.26, v, fontsize=FS_ANNOT, ha='right',
                            transform=at.transAxes,
                            color=(ORANGE if good else (RED if not shape else (INK if ok else GREY))),
                            fontweight=('bold' if good else 'normal'))
            numbers_text.append((t_key, t_val))

    # ── the two groups, as BOXES rather than rules over the columns. A rule reads as a
    # decoration hanging above the plate; a box states that these columns are one thing and
    # those are another, which is the comparison the figure exists to make.
    # Positions come from the GridSpec itself; adding throwaway subplots to read their extents
    # leaves stray axes behind.
    p0 = gs[0, 0].get_position(fig)
    pl = gs[0, len(drawn) - 1].get_position(fig)
    pm = gs[2, 0].get_position(fig)
    pad_x, top, bot = 0.010, GROUP_TOP, pm.y0 + 0.022
    groups = [(pl.x0, pl.x1, ORANGE, '#FDF3E8')]
    if len(drawn) > 1:                      # no baseline box when only the chart was run
        pk = gs[0, len(drawn) - 2].get_position(fig)
        groups.insert(0, (p0.x0, pk.x1, GREY, 'none'))
    if args.no_boxes:
        # The two rules say "these three are baselines, this one is the construction". On a slide
        # the speaker says that, so the rules go. The construction column keeps its warm ground,
        # which carries the same grouping as a wash rather than as a frame, and the baseline group
        # keeps nothing: its box was facecolor 'none', so it was rule and nothing else.
        # With ONE arm drawn there is nothing to distinguish it from, so the ground goes too.
        groups = ([] if len(drawn) < 2 else
                  [(x0, x1, 'none', fc) for x0, x1, col, fc in groups if fc != 'none'])
    for x0, x1, col, fc in groups:
        fig.patches.append(FancyBboxPatch(
            (x0 - pad_x, bot), (x1 - x0) + 2 * pad_x, top - bot,
            boxstyle='round,pad=0,rounding_size=0.010', transform=fig.transFigure,
            facecolor=fc, edgecolor=col, lw=0.7, zorder=-5))

    # ── row labels. Without these a reader meets red holes in row 1 and white holes in row 2 and
    # has no way to know the first is a defect and the second is the mechanism doing its job.
    for row, txt in [(0, 'flat sheet'), (1, 'deployed')]:
        bb = gs[row, 0].get_position(fig)
        fig.text(bb.x0 - 0.028, (bb.y0 + bb.y1) / 2, txt, rotation=90, ha='center', va='center',
                 fontsize=FS_ANNOT, color=GREY, linespacing=1.5)

    # ── legend: one horizontal strip, clear of the caption
    # The overlap key carries the sheets' own hatch, so the reader matches one texture rather than
    # two. It survives the swatch's few points across only because HATCH_OVERLAP is dense enough to
    # put several strokes inside it.
    for i, (fc, hatch, lab) in enumerate([(ORANGE, None, 'material'),
                                          (ORANGE, HATCH_OVERLAP, 'overlap'),
                                          (RED, None, 'void')]):
        # anchored BELOW the numbers block rather than at a fixed figure y, which collided
        # with the verdict line whenever the row heights changed
        ly = gs[2, 0].get_position(fig).y0 - 0.038
        lx = 0.325 + i * 0.135
        fig.add_artist(plt.Rectangle((lx, ly), 0.018, 0.022, facecolor=fc,
                                     edgecolor=INK, lw=0.3, hatch=hatch,
                                     transform=fig.transFigure))
        fig.text(lx + 0.024, ly + 0.011, lab, fontsize=FS_ANNOT, color=INK, va='center')

    # Derived, never hardcoded: these change on every rerun and a stale caption is a wrong one.
    def _va(k):
        live = [x for x in by_key[k]['runs'] if not x.get('stalled')] or by_key[k]['runs']
        return float(np.median([void_area_m2(x, scale) for x in live]))

    def _ch(k):
        live = [x for x in by_key[k]['runs'] if not x.get('stalled')] or by_key[k]['runs']
        return float(np.median([chamfer_mm(x, TGT, scale) for x in live
                                if x.get('deployed_boundary') is not None] or [np.nan]))

    bl = [k for k in drawn if k != OURS]
    best_bl = min(bl, key=_ch) if bl else None
    # The baseline sentences describe arms that --arms can leave out, and min() over an empty
    # sequence is what a one-arm run used to die on. Built separately and only when there are
    # baselines to describe.
    baselines = '' if not bl else (
        'The three '
        'baselines enforce validity after the fact, by projection and by penalties, and each '
        f'leaves between {100 * min(_va(k) for k in bl) * 1e6 / A_MM:.1f} and '
        f'{100 * max(_va(k) for k in bl) * 1e6 / A_MM:.1f}% of '
        'the sheet missing, so their outlines cannot be cut as a single piece and their Chamfer '
        f'values are greyed. The best of them reaches '
        f'{100 * _ch(best_bl) / (TGT["r"] * scale):.2f}%. ')
    caption = (
        f'The validity ladder. A square sheet of {n_panels} panels is deployed to the dashed '
        f'circle, of radius {TGT["r"] * scale / sheet_mm:.2f} times the sheet side, every column '
        f'under an identical {EP}-epoch budget. Orange is '
        'material, hatching is material lying on material, red is a residual void: stock area no '
        'panel covers. Residual voids are reported as a fraction of the sheet area, and the shape '
        'error as the symmetric Chamfer distance between the deployed outline and the target '
        'circle, over the target radius. '
        + baselines +
        'The closed chart '
        f'cannot represent a broken sheet at all: zero residual voids, zero overlap, hinge gap at '
        f'machine precision, and a Chamfer distance of '
        f'{100 * _ch(OURS) / (TGT["r"] * scale):.2f}% of the target radius.')
    with open(os.path.splitext(args.out)[0] + '_caption.txt', 'w') as fh:
        fh.write(caption + '\n')

    if args.numbers_array:
        bb = gs[2, 0].get_position(fig)
        for i, k in enumerate(array_keys):
            t_key = fig.text(bb.x0 - 0.012, bb.y0 + ARRAY_Y[i] * (bb.y1 - bb.y0), k,
                             ha='right', va='center', fontsize=FS_MICRO, color=GREY)
            # What can collide here is a row label into the first column's value.
            numbers_text.append((t_key, array_first[i]))

    # A key and its value meet in the middle of a 78 bp column, so a longer number -- a three-digit
    # Chamfer, a different sheet -- can run one into the other and the figure would still be
    # written, with a number silently unreadable. Measure the drawn text and refuse instead.
    rend = fig.canvas.get_renderer()
    for t_key, t_val in numbers_text:
        gap_bp = ((t_val.get_window_extent(renderer=rend).x0
                   - t_key.get_window_extent(renderer=rend).x1) * 72.0 / fig.dpi)
        if gap_bp < MIN_GAP_BP:
            raise ValueError(f"numbers block collides: {t_key.get_text()!r} and "
                             f"{t_val.get_text()!r} are {gap_bp:.1f} bp apart, "
                             f"minimum {MIN_GAP_BP:.1f}")

    pdf, lo, hi = save_paper_figure(fig, args.out, FIG_W_BP, FIG_H_BP, dpi=args.dpi)
    plt.close(fig)
    print(f"  wrote {pdf} ({FIG_W_BP:.1f} x {FIG_H_BP:.1f} bp, type {lo:.1f}-{hi:.1f} pt)")
    for key in drawn:
        R = by_key[key]
        print(f"    {R['label']:<32} gap {R['med_gap']:9.2e} mm  overlap {100*R['med_overlap']:6.2f}%"
              f"  voids {R['med_tiling']*A_MM/CUT:8.2f} mm  fit {R.get('med_fit_mm', float('nan')):7.2f} mm"
              f"  moved {R.get('med_travel_mm', float('nan')):6.1f} mm  {R['n_live']}/{R['n']}")


if __name__ == "__main__":
    main()
