"""Render a labelled sweep around the inked-leaf katagami, one block of variations per sheet.

The element is a pale leaf carrying a nested dark core. Four things are varied, each in its own
block: how much of the element is ink, where the core sits inside it, whether a finer second
population lies under the figures, and what the sheet's own deformation does to the flow.

Panels are grouped by LAYOUT. Everything that changes only the ink is drawn over one packing, so a
tonal comparison is of the tone alone and not of a different scatter of leaves. Every recipe is
written to a manifest beside the images, because a variation nobody can rebuild is not a choice.

    PYTHONPATH=$(pwd) JAX_PLATFORMS=cpu python3 legacy/katagami/katagami_contact_sheet.py
"""

import argparse
import os

import numpy as np
import yaml

from legacy.katagami.katagami import load_flat_tiles_fn
from legacy.katagami.scale_field import (
    PALETTES,
    build_scale_field_fn,
    render_scale_preview_fn,
    transform_elements_fn,
)
from legacy.katagami.net_field import (
    build_net_sheet_fn,
    net_join_residual_fn,
    net_region_fn,
    render_net_preview_fn,
)
from nff_paper.closed_support import cell_motifs, web
from legacy.katagami.stencil import (
    cut_paths_fn,
    measure_stencil_fn,
    prune_dust_fn,
    resolve_paper_fn,
)
from legacy.katagami.topology_art.quad_warp import transform_path_fn
from legacy.katagami.topology_art.leaf import ElementSpec, element_layers_fn, leaf_path_fn
from legacy.katagami.topology_art.svg_export import fit_to_page_fn, write_svg_fn

PAGE_MM = (420.0, 297.0)
PALE, DARK = "#F2EFE6", "#141414"
BLOCK_ORDER = ["tone", "core", "spread", "ground", "flow", "cut", "calm", "net", "strict",
               "lace", "line", "cell", "motif", "vocab", "density"]

# The nets do not go through the element field at all: they are one continuous pattern warped by the
# grid of tile centroids, so they have their own small pipeline.
NET_SHEET = dict(synthetic=(20, 14), seed=3)
NET_PANELS = [
    dict(name="asanoha", net="asanoha", field="banded", amplitude=1.3, scale=2.0, width=2.6),
    dict(name="kikko", net="kikko", field="banded", amplitude=1.3, scale=2.0, width=2.6),
    dict(name="kagome", net="kagome", field="banded", amplitude=1.3, scale=2.0, width=2.6),
    dict(name="shippo", net="shippo", field="banded", amplitude=1.3, scale=2.0, width=2.6),
    dict(name="sayagata", net="sayagata", field="banded", amplitude=1.3, scale=2.0, width=2.6,
         note="a meander approximation, not the true linked fret"),

    dict(name="warp_gentle", net="shippo", field="smooth", amplitude=0.7, scale=2.0, width=2.6),
    dict(name="warp_moderate", net="shippo", field="smooth", amplitude=1.4, scale=2.0, width=2.6),
    dict(name="warp_strong", net="shippo", field="smooth", amplitude=2.4, scale=2.0, width=2.6),

    dict(name="asanoha_kakishibu", net="asanoha", field="banded", amplitude=1.3, scale=2.0,
         width=2.6, palette="kakishibu"),
    dict(name="shippo_ai", net="shippo", field="banded", amplitude=1.3, scale=2.0, width=2.6,
         palette="ai"),

    dict(name="kikko_under_figures", net="kikko", field="banded", amplitude=1.3, scale=1.4,
         width=1.5, note="the net as jimon ground, leaves over it",
         figures=dict(density=0.62, size_frac=2.4, smoothing=20, leaf_width=0.62, leaf_bow=0.22,
                      seed=3)),
]

# A window on the page, in mm, at the page's own aspect: the middle quarter of the sheet. A whole
# sheet shrunk to a contact-sheet cell shows the flow and hides the element, and a tight crop shows
# the element but not whether it makes a pattern. Half the page each way holds both.
DETAIL_MM = (105.0, 74.25, 315.0, 222.75)

SHEET = dict(synthetic=(20, 14), amplitude=1.8, seed=3)
FIELD = dict(smoothing=14, size_smoothing=4, size_gamma=1.6, size_frac=2.30,
             leaf_width=0.62, leaf_bow=0.22, seed=3)

# One entry per distinct packing. Anything not named here falls back to FIELD.
LAYOUTS = {
    "base":   dict(sheet=dict(field_kind="banded"), field=dict(density=0.30)),
    "airy":   dict(sheet=dict(field_kind="banded"), field=dict(density=0.52)),
    "airy_ground_third":  dict(sheet=dict(field_kind="banded"),
                               field=dict(density=0.52, ground_scale=0.34, ground_density=0.36)),
    "airy_ground_fifth":  dict(sheet=dict(field_kind="banded"),
                               field=dict(density=0.52, ground_scale=0.20, ground_density=0.30)),
    "airy_ground_coarse": dict(sheet=dict(field_kind="banded"),
                               field=dict(density=0.52, ground_scale=0.46, ground_density=0.42)),
    "flow_banded":  dict(sheet=dict(field_kind="banded"),  field=dict(density=0.30)),
    "flow_shear":   dict(sheet=dict(field_kind="shear"),   field=dict(density=0.30)),
    "flow_saddle":  dict(sheet=dict(field_kind="saddle"),  field=dict(density=0.30)),
    "flow_pinch":   dict(sheet=dict(field_kind="pinch"),   field=dict(density=0.30)),
    "flow_vortex2": dict(sheet=dict(field_kind="vortex2"), field=dict(density=0.30)),
    "flow_lens":    dict(sheet=dict(field_kind="lens"),    field=dict(density=0.30)),
    "flow_aniso":   dict(sheet=dict(field_kind="banded"),
                         field=dict(density=0.30, aniso_power=1.0)),
    # A constant slot only differs from a proportional rim where the elements differ in size, so
    # the axis is worth nothing on a calm sheet. These two keep the size field's tails.
    "spread":        dict(sheet=dict(field_kind="patchy", amplitude=2.6),
                          field=dict(density=0.30, size_gamma=2.8, size_smoothing=1)),
    "spread_ground": dict(sheet=dict(field_kind="patchy", amplitude=2.6),
                          field=dict(density=0.52, size_gamma=2.8, size_smoothing=1,
                                     ground_scale=0.30, ground_density=0.34)),
    # One element per tile. Which elements fuse when the sheet is cut is then decided by the
    # lattice rather than discovered in the dart-throwing.
    "lens_lattice":   dict(sheet=dict(field_kind="lens"),
                           field=dict(placement="lattice", regularity=1.0, bond=1.0)),
    "lens_loose":     dict(sheet=dict(field_kind="lens"),
                           field=dict(placement="lattice", regularity=0.92, bond=1.0,
                                      edge_looseness=0.5)),
    "lens_mid":       dict(sheet=dict(field_kind="lens"),
                           field=dict(placement="lattice", regularity=0.60, bond=1.0)),
    "lens_square":    dict(sheet=dict(field_kind="lens"),
                           field=dict(placement="lattice", regularity=1.0, bond=0.0)),
    "saddle_lattice": dict(sheet=dict(field_kind="saddle"),
                           field=dict(placement="lattice", regularity=0.92, bond=1.0,
                                      edge_looseness=0.5)),
    "vortex2_lattice": dict(sheet=dict(field_kind="vortex2"),
                            field=dict(placement="lattice", regularity=0.92, bond=1.0,
                                       edge_looseness=0.5)),
}

# The cut block has no outline and a border set in millimetres, because both are properties of the
# object rather than of the drawing. `duty` is 1 here so the quoted slot IS the border: at the
# default 0.5 a "3 mm" slot draws 1.5 mm.
CUT_INK = dict(rim=0.0, duty=1.0, core_shift=0.6)

# The calm layouts. No angular term in any of these fields, so none can carry a vortex, and the
# tiles are given a rectangular pitch: a sheet of perfect squares has an ISOTROPIC tile, whose
# principal axis is numerical noise rather than a direction. `aspect` is what buys a still flow.
LAYOUTS.update({
    # A tall tile pitch turns the sheet portrait, so each grid is chosen to bring the sheet back to
    # the page's own 1.41 and fill it.
    "seigaiha": dict(sheet=dict(field_kind="uniform", synthetic=(23, 10), aspect=1.6,
                                amplitude=0.0),
                     field=dict(placement="lattice", regularity=1.0, bond=1.0,
                                size_frac=2.55, leaf_width=0.86, leaf_bow=0.46, smoothing=30)),
    "nowaki":   dict(sheet=dict(field_kind="swell", synthetic=(21, 11), aspect=1.35,
                                amplitude=0.9),
                     field=dict(placement="lattice", regularity=0.94, bond=1.0,
                                edge_looseness=0.35, size_frac=3.6, leaf_width=0.44,
                                leaf_bow=0.14, smoothing=30, size_smoothing=8)),
    # Needles must stay clear of each other end to end. Left touching they weld into columns and
    # slice the ground into strips -- 22 of them -- which reads as a striped curtain, not as pine.
    "matsuba":  dict(sheet=dict(field_kind="uniform", synthetic=(21, 10), aspect=1.5,
                                amplitude=0.0),
                     field=dict(placement="lattice", regularity=0.86, bond=1.0,
                                size_frac=1.15, leaf_width=0.22, leaf_bow=0.05, smoothing=30,
                                angle_jitter=0.13)),
    "bokashi":  dict(sheet=dict(field_kind="gradient", synthetic=(21, 10), aspect=1.45,
                                amplitude=1.5),
                     field=dict(placement="lattice", regularity=1.0, bond=1.0,
                                size_frac=2.6, leaf_width=0.62, leaf_bow=0.26,
                                smoothing=40, size_smoothing=5, size_gamma=1.4)),
    "terrace":  dict(sheet=dict(field_kind="terrace", synthetic=(20, 10), aspect=1.4,
                                amplitude=1.2),
                     field=dict(placement="lattice", regularity=1.0, bond=1.0,
                                size_frac=2.6, leaf_width=0.70, leaf_bow=0.34,
                                smoothing=36, size_smoothing=4)),
})

# The strict block. Every panel here states its narrowest white BEFORE it is drawn and is then
# measured against that statement, which is what separates it from the `cut` block above: there the
# slot was set and the bridge came out at 0.43 of it, here the slot IS the narrowest white.
#
# `rim` is left out of these specs on purpose. The element's own border is solved from the slot by
# `rim_for_slot_fn`, because a nested core is inset evenly all round while the flanks converge at the
# tip, so the band that survives there is a fixed fraction of the nominal rim.
LAYOUTS.update({
    "strict_free":   dict(sheet=dict(field_kind="lens"),
                          field=dict(placement="lattice", regularity=1.0, bond=1.0)),
    "strict_square": dict(sheet=dict(field_kind="lens"),
                          field=dict(placement="lattice", regularity=1.0, bond=0.0)),
    "strict_loose":  dict(sheet=dict(field_kind="lens"),
                          field=dict(placement="lattice", regularity=0.94, bond=1.0,
                                     edge_looseness=0.35)),
    # A leaf comparable to its own cell, so a cell-confined hole still reads as a leaf. At the
    # default 2.30 the crop is the whole element and the sheet reads as paving.
    "strict_cell":   dict(sheet=dict(field_kind="lens"),
                          field=dict(placement="lattice", regularity=1.0, bond=1.0,
                                     size_frac=1.85)),
    # The calm pitch: an elongated tile settles every principal axis the same way, which is what
    # buys a still flow. Wide, deep leaves fill their cells, so the web comes out at one width.
    "strict_scales": dict(sheet=dict(field_kind="banded", synthetic=(22, 11), aspect=1.4,
                                     amplitude=1.3),
                          field=dict(placement="lattice", regularity=1.0, bond=1.0,
                                     size_frac=1.95, leaf_width=0.80, leaf_bow=0.40,
                                     smoothing=30)),
})

STRICT_PANELS = [
    dict(name="free_baseline", layout="strict_free", build="free", slot=2.5,
         note="painter's order, nothing guaranteed"),
    dict(name="gapped_2p5", layout="strict_free", build="gapped", slot=2.5,
         note="occluders grown by half the slot"),
    dict(name="gapped_4p0", layout="strict_free", build="gapped", slot=4.0,
         note="the same, at a 4 mm slot"),
    dict(name="gapped_square", layout="strict_square", build="gapped", slot=2.5,
         note="square bond, no row offset"),
    dict(name="gapped_loose", layout="strict_loose", build="gapped", slot=2.5,
         note="sites adrift, slot unchanged"),
    # The falsifying pair. On a regular lattice the free drawing lands on the slot anyway, so the
    # guarantee looks free; disorder is where the two part company.
    dict(name="free_jittered", layout="lens_mid", build="free", slot=2.5,
         note="sites at regularity 0.6, nothing guaranteed"),
    dict(name="gapped_jittered", layout="lens_mid", build="gapped", slot=2.5,
         note="the same disorder, slot guaranteed"),
    dict(name="cell_leaves", layout="strict_cell", build="cell", slot=3.0, mode="crop",
         core_shift=0.6, hole="leaf", note="one hole per cell, web is the skeleton"),
    dict(name="cell_whole", layout="strict_cell", build="cell", slot=3.0, mode="fit",
         core_shift=0.6, hole="leaf", note="leaves shrunk to fit, never cut"),
    dict(name="cell_scales", layout="strict_scales", build="cell", slot=2.5, mode="crop",
         core_shift=0.6, hole="leaf", note="calm pitch, cells filled"),
    dict(name="skeleton_under", layout="strict_free", build="skeleton", slot=1.6,
         note="free drawing over a spanning web"),
    # The device the double contour needs. Four bands enclose a pale island, and an enclosed island
    # is a separate piece of paper however wide its border.
    dict(name="thread_double", layout="strict_free", build="thread", slot=2.5, rings=4, duty=0.5,
         core_shift=0.35, note="double contour, islands threaded"),
    dict(name="thread_untied", layout="strict_free", build="free", slot=2.5, rings=4, duty=0.5,
         core_shift=0.35, note="the same element, no thread"),
]

# The lace block. An ARMATURE is any region unioned into the paper to hold what the drawing leaves
# loose, and here it is a design object rather than a rescue: the ladder runs from a nick per trapped
# island up to a whole komon net, and every rung reports what it costs in tone.
LAYOUTS.update({
    # Big leaves, because a comb of eleven bands insets its innermost ring by five of them. At the
    # 20x14 pitch the nesting saturates at six bands however many are asked for.
    "lace_coarse": dict(sheet=dict(field_kind="lens", synthetic=(10, 7)),
                        field=dict(placement="lattice", regularity=1.0, bond=1.0,
                                   leaf_width=0.72, leaf_bow=0.34)),
})

# The element that carries the double contour, and the packing all the ladder rungs share.
DOUBLE = dict(rings=4, duty=0.5, core_shift=0.35)

LACE_PANELS = [
    # A. the armature as a motif in its own right, under a dark shingled element
    dict(name="net_kikko", layout="strict_free", slot=2.5, armature="net:kikko",
         armature_mm=1.6, net_scale=2.5, note="tortoiseshell jimon, dark ground"),
    dict(name="net_asanoha", layout="strict_free", slot=2.5, armature="net:asanoha",
         armature_mm=1.6, net_scale=3.0, note="hemp leaf, three chords per hexagon"),
    dict(name="net_kagome", layout="strict_free", slot=2.5, armature="net:kagome",
         armature_mm=1.6, net_scale=2.5, note="basket weave"),
    dict(name="net_shippo", layout="strict_free", slot=2.5, armature="net:shippo",
         armature_mm=1.6, net_scale=2.5, note="seven treasures, circles"),
    dict(name="net_kikko_fine", layout="strict_free", slot=2.5, armature="net:kikko",
         armature_mm=1.0, net_scale=1.8, note="the same net, finer and denser"),
    dict(name="net_kikko_bold", layout="strict_free", slot=2.5, armature="net:kikko",
         armature_mm=2.5, net_scale=2.5, note="strut at the slot, one width throughout"),

    # B. the outlined element: how deep a comb can go, at one fixed armature
    *[dict(name=f"comb_{rings}", layout="lace_coarse", slot=2.0, rings=rings, duty=0.34,
           core_shift=0.0, armature="net:kikko", armature_mm=1.6, net_scale=2.0, nick_gaps=True,
           note=f"{rings} bands, duty 0.34")
      for rings in (3, 5, 7, 9, 11)],
    dict(name="comb_11_bare", layout="lace_coarse", slot=2.0, rings=11, duty=0.34, core_shift=0.0,
         armature="net:kikko", armature_mm=1.6, net_scale=2.0,
         note="11 bands, gaps not nicked"),

    # C. the ladder, one packing and one element, only the armature varying
    dict(name="ladder_none", layout="strict_free", slot=2.5, **DOUBLE, armature="none",
         note="nothing holding it"),
    dict(name="ladder_islands", layout="strict_free", slot=2.5, **DOUBLE, armature="islands",
         note="the census: every piece that falls out"),
    dict(name="ladder_nicks", layout="strict_free", slot=2.5, **DOUBLE, armature="nicks",
         note="one nick per trapped island"),
    dict(name="ladder_mst", layout="strict_free", slot=2.5, **DOUBLE, armature="mst",
         note="cheapest spanning tree, n-1 struts"),
    dict(name="ladder_chains", layout="strict_free", slot=2.5, **DOUBLE, armature="chains",
         note="one stroke per diagonal chain"),
    dict(name="ladder_grid", layout="strict_free", slot=2.5, **DOUBLE, armature="grid",
         note="the full lattice, ~2n struts"),

    # D. the repair, and colour
    dict(name="skeleton_unframed", layout="strict_free", slot=1.6, armature="skeleton",
         frame=None, note="no frame: the border strip is half a slot"),
    dict(name="skeleton_framed", layout="strict_free", slot=1.6, armature="skeleton",
         note="framed, which is the repair"),
    dict(name="net_kikko_kakishibu", layout="strict_free", slot=2.5, armature="net:kikko",
         armature_mm=1.6, net_scale=2.5, palette="kakishibu", note="persimmon tannin"),
    dict(name="net_kikko_ai", layout="strict_free", slot=2.5, armature="net:kikko",
         armature_mm=1.6, net_scale=2.5, palette="ai", note="indigo"),
]

# The line block. Here the element is not an area at all: it is a STROKE along its own contours, and
# nothing on the sheet is a solid mass. The two readings of such a drawing are different objects and
# they need opposite devices -- a stroke that is the PAPER traps islands and is held by adding paper
# (nicks); a stroke that is the SLOT divides the ground into faces and is joined by removing slot
# (bridges).
LAYOUTS.update({
    # Sparse enough that the loops do NOT overlap, which is the only way the paper polarity needs an
    # armature at all.
    "line_sparse": dict(sheet=dict(field_kind="lens"),
                        field=dict(placement="lattice", regularity=1.0, bond=1.0, size_frac=1.30)),
})

LINE_PANELS = [
    # A. the weight rule, same paths, both polarities
    dict(name="even_slot", layout="strict_free", width=2.5, kind="even", polarity="slot",
         note="even weight, dark line on cream"),
    dict(name="vary_slot", layout="strict_free", width=2.5, kind="varying", polarity="slot",
         note="brush weight, thins at the tip"),
    dict(name="even_net", layout="strict_free", width=2.5, kind="even", polarity="net",
         note="even weight, pale line on dark"),
    dict(name="vary_net", layout="strict_free", width=2.5, kind="varying", polarity="net",
         note="brush weight, pale on dark"),

    # B. the weight ladder
    dict(name="weight_1p2", layout="strict_free", width=1.2, kind="even", polarity="slot",
         note="1.2 mm line"),
    dict(name="weight_2p5", layout="strict_free", width=2.5, kind="even", polarity="slot",
         note="2.5 mm line"),
    dict(name="weight_4p5", layout="strict_free", width=4.5, kind="even", polarity="slot",
         note="4.5 mm line"),

    # C. concentric contours, drawn as separate lines rather than one thick band
    dict(name="contour_1", layout="strict_free", width=2.0, count=1, spacing=6.0, kind="even",
         polarity="slot", note="one contour"),
    dict(name="contour_2", layout="strict_free", width=2.0, count=2, spacing=6.0, kind="even",
         polarity="slot", note="two contours, 6 mm apart"),
    dict(name="contour_3", layout="strict_free", width=2.0, count=3, spacing=6.0, kind="even",
         polarity="slot", note="three contours"),

    # D. the drawing against the mesh: whether a stroke terminates behind the next leaf
    dict(name="drawing", layout="strict_free", width=2.5, kind="even", polarity="slot",
         occlude=True, note="strokes terminate behind the next leaf"),
    dict(name="mesh", layout="strict_free", width=2.5, kind="even", polarity="slot",
         occlude=False, note="every loop stays closed"),

    # E. the two duals
    dict(name="slot_dashed", layout="strict_free", width=2.5, kind="even", polarity="slot",
         bridge_pitch=22.0, note="slots interrupted at an even pitch"),
    dict(name="net_overlapped", layout="strict_free", width=2.5, kind="even", polarity="net",
         note="loops overlap, so the line is already one piece"),
    dict(name="net_sparse", layout="line_sparse", width=2.5, kind="even", polarity="net",
         note="loops apart: every one is its own island"),
    dict(name="net_sparse_held", layout="line_sparse", width=2.5, kind="even", polarity="net",
         armature="mst", note="the same, joined by a spanning armature"),

    # F. colour
    dict(name="even_slot_kakishibu", layout="strict_free", width=2.5, kind="even", polarity="slot",
         palette="kakishibu", note="persimmon tannin"),
    dict(name="even_net_ai", layout="strict_free", width=2.5, kind="even", polarity="net",
         palette="ai", note="indigo"),
]

# The cell block. One hole per Voronoi territory, each territory eroded by half the slot, so the
# white IS the thickened skeleton: one piece at exactly the slot, by construction, with valence-3
# junctions. Nothing here needs tuning for connectivity, so every panel is an aesthetic choice with
# the measurement riding along.
LAYOUTS.update({
    "cell_base":    dict(sheet=dict(field_kind="lens"),
                         field=dict(placement="lattice", regularity=1.0, bond=1.0,
                                    size_frac=1.85)),
    "cell_full":    dict(sheet=dict(field_kind="lens"),
                         field=dict(placement="lattice", regularity=1.0, bond=1.0,
                                    size_frac=2.60)),
    "cell_grazing": dict(sheet=dict(field_kind="lens"),
                         field=dict(placement="lattice", regularity=1.0, bond=1.0,
                                    size_frac=1.40)),
    # The calm pitch: an elongated tile settles every principal axis the same way.
    "cell_calm":    dict(sheet=dict(field_kind="banded", synthetic=(22, 11), aspect=1.4,
                                    amplitude=1.3),
                         field=dict(placement="lattice", regularity=1.0, bond=1.0,
                                    size_frac=1.95, leaf_width=0.80, leaf_bow=0.40,
                                    smoothing=30)),
    # Coarse slabs beside fine crazing: the size field's tails kept rather than averaged away.
    "cell_extreme": dict(sheet=dict(field_kind="patchy", amplitude=2.6),
                         field=dict(placement="lattice", regularity=1.0, bond=1.0,
                                    size_frac=2.20, size_gamma=2.8, size_smoothing=1)),
})

CELL_PANELS = [
    # A. the fill continuum: the axis joining the three panels that were liked
    dict(name="fill_full", layout="cell_full", slot=3.0, mode="crop",
         note="cells completely filled"),
    dict(name="fill_cropped", layout="cell_base", slot=3.0, mode="crop",
         note="leaf cut to its cell"),
    dict(name="fill_grazing", layout="cell_grazing", slot=3.0, mode="crop",
         note="leaf just reaching its cell"),
    dict(name="fill_fitted", layout="cell_base", slot=3.0, mode="fit",
         note="leaf shrunk to fit, never cut"),
    dict(name="fill_sparse", layout="cell_base", slot=3.0, mode="fit", fill=0.55,
         note="the same, at 55% of the room"),

    # B. cell character, one packing and one fill throughout
    dict(name="shape_facets", layout="cell_base", slot=3.0, mode="crop",
         note="hard Voronoi facets"),
    dict(name="shape_rounded_soft", layout="cell_base", slot=3.0, mode="crop", corner_radius=1.5,
         note="corners rounded 1.5 mm"),
    dict(name="shape_rounded_full", layout="cell_base", slot=3.0, mode="crop", corner_radius=4.0,
         note="corners rounded 4 mm, pebbles"),
    dict(name="shape_leaf", layout="cell_base", slot=3.0, mode="fit",
         note="whole leaves, pointing down-field"),
    dict(name="shape_disc", layout="cell_base", slot=3.0, mode="disc",
         note="each cell's inscribed circle"),

    # C. how much the cells differ in size, end to end
    dict(name="size_lloyd", layout="cell_base", slot=3.0, mode="crop", lloyd=10,
         note="relaxed to a honeycomb"),
    dict(name="size_calm", layout="cell_calm", slot=2.5, mode="crop",
         note="calm elongated pitch"),
    dict(name="size_present", layout="cell_base", slot=3.0, mode="crop",
         note="the tiling's own gradient"),
    dict(name="size_extreme", layout="cell_extreme", slot=3.0, mode="crop",
         note="coarse slabs beside fine crazing"),

    # D. the web itself
    dict(name="web_1p5", layout="cell_base", slot=1.5, mode="crop", note="1.5 mm web"),
    dict(name="web_2p5", layout="cell_base", slot=2.5, mode="crop", note="2.5 mm web"),
    dict(name="web_4p0", layout="cell_base", slot=4.0, mode="crop", note="4 mm web"),
    dict(name="web_6p0", layout="cell_base", slot=6.0, mode="crop", note="6 mm web"),
    dict(name="web_breathing", layout="cell_base", slot=3.0, mode="crop", breathe=1.0,
         note="web widens where the tiles are large"),

    # E. colour, and the reading turned over
    dict(name="colour_kakishibu", layout="cell_base", slot=3.0, mode="crop",
         corner_radius=1.5, palette="kakishibu", note="persimmon tannin"),
    dict(name="colour_ai", layout="cell_base", slot=3.0, mode="crop", corner_radius=1.5,
         palette="ai", note="indigo"),
    dict(name="polarity_inverse", layout="cell_base", slot=3.0, mode="crop", invert_ink=True,
         note="cells as ink: 280 loose pieces, not cuttable"),
]

# The motif block. Same construction as `cell`, but the hole is drawn from a vocabulary rather than
# from the leaf: the guarantee never looks at the shape, only at whether it stays inside its own
# territory, so the motif is free and the density is a separate dial.
LAYOUTS.update({
    "motif_48":   dict(sheet=dict(field_kind="lens", synthetic=(8, 6)),
                       field=dict(placement="lattice", regularity=1.0, bond=1.0, size_frac=1.85)),
    "motif_140":  dict(sheet=dict(field_kind="lens", synthetic=(14, 10)),
                       field=dict(placement="lattice", regularity=1.0, bond=1.0, size_frac=1.85)),
    "motif_540":  dict(sheet=dict(field_kind="lens", synthetic=(30, 18)),
                       field=dict(placement="lattice", regularity=1.0, bond=1.0, size_frac=1.85)),
    "motif_990":  dict(sheet=dict(field_kind="lens", synthetic=(45, 22)),
                       field=dict(placement="lattice", regularity=1.0, bond=1.0, size_frac=1.85)),
    # Dart-thrown against the size field, so the cells CROWD where the tiles are small instead of
    # merely shrinking. The lattice cannot do that: it always has exactly one site per tile.
    "motif_graded": dict(sheet=dict(field_kind="lens"),
                         field=dict(placement="poisson", density=0.42, size_frac=1.85)),
    # A second, finer population under the first: two cell scales in one diagram.
    "motif_mixed":  dict(sheet=dict(field_kind="lens"),
                         field=dict(placement="lattice", regularity=1.0, bond=1.0, size_frac=1.85,
                                    ground_scale=0.42, ground_density=0.52)),
})

VOCABULARY = ["disc", "triangle", "hexagon", "star", "crescent", "magatama", "sakura", "slit",
              "cross", "cluster3"]

MOTIF_PANELS = [
    # A. density: the same sheet cut at five counts
    *[dict(name=f"density_{count}", layout=layout, slot=3.0, mode="crop",
           note=f"{count} cells")
      for count, layout in ((48, "motif_48"), (140, "motif_140"), (280, "cell_base"),
                            (540, "motif_540"), (990, "motif_990"))],
    dict(name="density_graded", layout="motif_graded", slot=3.0, mode="crop",
         note="dart-thrown: cells crowd where tiles are small"),
    dict(name="density_mixed", layout="motif_mixed", slot=3.0, mode="crop",
         note="a finer population under the first"),

    # B. the vocabulary, one packing, one reach
    *[dict(name=f"motif_{name}", layout="cell_base", slot=3.0, motif=name, reach=1.9,
           note=name) for name in VOCABULARY],

    # C. what reach does to one motif
    dict(name="reach_1p0", layout="cell_base", slot=3.0, motif="sakura", reach=1.0,
         note="sakura inscribed, untouched"),
    dict(name="reach_1p9", layout="cell_base", slot=3.0, motif="sakura", reach=1.9,
         note="sakura outgrowing its cell"),
    dict(name="reach_2p6", layout="cell_base", slot=3.0, motif="sakura", reach=2.6,
         note="sakura cropped to paving"),

    # D. an elaborate figure only where there is room for it
    dict(name="detail_big_sakura", layout="cell_base", slot=3.0, reach=1.9,
         detail=dict(large="sakura", small="disc", quantile=0.55),
         note="blossom in the roomy cells, dot in the rest"),
    dict(name="detail_big_plain", layout="cell_base", slot=3.0, reach=1.9,
         detail=dict(large="hexagon", small="star5", quantile=0.55),
         note="the reverse: figure in the small cells"),

    # E. the control, and colour
    dict(name="motif_ring", layout="cell_base", slot=3.0, motif="ring", reach=1.0,
         note="a ring encloses its own white"),
    dict(name="colour_sakura_kakishibu", layout="cell_base", slot=3.0, motif="sakura", reach=1.9,
         palette="kakishibu", note="persimmon tannin"),
    dict(name="colour_magatama_ai", layout="cell_base", slot=3.0, motif="magatama", reach=2.2,
         palette="ai", note="indigo"),
]

# The vocabulary sheet: every motif in the registry on one packing, so the comparison is of the shape
# and nothing else. And the density ladder, spanning 12 cells to 6200 on the same page.
CELL_COUNTS = [(12, (4, 3)), (30, (6, 5)), (70, (10, 7)), (140, (14, 10)), (280, (20, 14)),
               (540, (30, 18)), (1000, (40, 25)), (1600, (50, 32)), (2560, (64, 40)),
               (4000, (80, 50)), (6200, (100, 62))]

for _count, _grid in CELL_COUNTS:
    LAYOUTS[f"grid_{_count}"] = dict(
        sheet=dict(field_kind="lens", synthetic=_grid),
        field=dict(placement="lattice", regularity=1.0, bond=1.0, size_frac=1.85))

VOCAB_PANELS = [
    dict(name=name, layout="cell_base", slot=3.0, motif=name, reach=1.9, note=name)
    for name in sorted(cell_motifs.UNIT_MOTIFS)
]

# At a FIXED slot a finer sheet is paler, because the slot is a length and the web grows with the
# number of edges. Holding the tone instead means shrinking the slot with the cell: the second ladder
# scales it as 1/sqrt(count), which is the only way a 6200-cell sheet stays dark.
DENSITY_PANELS = [
    *[dict(name=f"fixed_{count}", layout=f"grid_{count}", slot=3.0, mode="crop",
           note=f"{count} cells, 3 mm slot") for count, _ in CELL_COUNTS],
    *[dict(name=f"scaled_{count}", layout=f"grid_{count}",
           slot=round(3.0 * (280.0 / count) ** 0.5, 2), mode="crop",
           note=f"{count} cells, slot scaled to match")
      for count in (30, 140, 540, 1600, 6200)],
    *[dict(name=f"crescent_{count}", layout=f"grid_{count}", slot=slot, motif="crescent",
           reach=1.9, note=f"crescents at {count} cells")
      for count, slot in ((70, 4.5), (540, 2.2), (2560, 1.1))],
]

# rim, and shift/turn, are the two axes the element itself has. `rim_abs` is in page mm, so it is
# one slot width across the whole sheet however large the leaf carrying it.
PANELS = [
    dict(block="tone", name="rim_thin",     layout="base", note="rim 0.05 of leaf",
         spec=dict(rim=0.05)),
    dict(block="tone", name="rim_medium",   layout="base", note="rim 0.09 of leaf",
         spec=dict(rim=0.09)),
    dict(block="tone", name="rim_broad",    layout="base", note="rim 0.14 of leaf",
         spec=dict(rim=0.14)),
    dict(block="tone", name="slot_fine",    layout="base", note="constant slot 1.6 mm",
         spec=dict(rim=0.0, rim_abs=1.6)),
    dict(block="tone", name="slot_medium",  layout="base", note="constant slot 3.0 mm",
         spec=dict(rim=0.0, rim_abs=3.0)),
    dict(block="tone", name="slot_broad",   layout="base", note="constant slot 5.0 mm",
         spec=dict(rim=0.0, rim_abs=5.0)),

    dict(block="core", name="core_centred", layout="base", note="core centred",
         spec=dict(rim=0.11)),
    dict(block="core", name="core_shift_half", layout="base", note="core back 0.45 of rim",
         spec=dict(rim=0.11, core_shift=0.45)),
    dict(block="core", name="core_shift_full", layout="base", note="core back 0.85 of rim",
         spec=dict(rim=0.11, core_shift=0.85)),
    dict(block="core", name="core_turned",  layout="base", note="core turned 0.7 of rim",
         spec=dict(rim=0.11, core_turn=0.7)),
    dict(block="core", name="core_spine",   layout="base", note="core narrowed to a spine",
         spec=dict(rim=0.11, core_width_delta=-0.22)),
    dict(block="core", name="core_leaflike", layout="base", note="back 0.75, turned 0.4, spine",
         spec=dict(rim=0.11, core_shift=0.75, core_turn=0.4, core_width_delta=-0.10)),

    dict(block="ground", name="no_ground",  layout="airy", note="figures alone, bare ground",
         spec=dict(rim=0.11, core_shift=0.6)),
    dict(block="ground", name="ground_third_pale", layout="airy_ground_third",
         note="fine pale leaves at 1/3", spec=dict(rim=0.11, core_shift=0.6),
         ground=dict(rings=1, rim=0.0)),
    dict(block="ground", name="ground_third_inked", layout="airy_ground_third",
         note="fine inked leaves at 1/3", spec=dict(rim=0.11, core_shift=0.6),
         ground=dict(rim=0.0, rim_abs=1.2)),
    dict(block="ground", name="ground_fifth_pale", layout="airy_ground_fifth",
         note="fine pale leaves at 1/5", spec=dict(rim=0.11, core_shift=0.6),
         ground=dict(rings=1, rim=0.0)),
    dict(block="ground", name="ground_coarse_pale", layout="airy_ground_coarse",
         note="second scale at 1/2", spec=dict(rim=0.11, core_shift=0.6),
         ground=dict(rings=1, rim=0.0)),
    dict(block="ground", name="ground_inverted", layout="airy_ground_third",
         note="dark figures, pale ground", spec=dict(rim=0.11, core_shift=0.6, invert=True),
         ground=dict(rings=1, rim=0.0)),

    dict(block="flow", name="flow_banded",  layout="flow_banded",  note="banded, the reference",
         spec=dict(rim=0.11, core_shift=0.6), whole=True),
    dict(block="flow", name="flow_shear",   layout="flow_shear",   note="shear: fanning bands",
         spec=dict(rim=0.11, core_shift=0.6), whole=True),
    dict(block="flow", name="flow_saddle",  layout="flow_saddle",  note="saddle: two diagonals",
         spec=dict(rim=0.11, core_shift=0.6), whole=True),
    dict(block="flow", name="flow_pinch",   layout="flow_pinch",   note="pinch: one central knot",
         spec=dict(rim=0.11, core_shift=0.6), whole=True),
    dict(block="flow", name="flow_vortex2", layout="flow_vortex2", note="vortex2: paired eddies",
         spec=dict(rim=0.11, core_shift=0.6), whole=True),
    dict(block="flow", name="flow_lens",    layout="flow_lens",    note="lens: one body of scale",
         spec=dict(rim=0.11, core_shift=0.6), whole=True),
    dict(block="flow", name="flow_aniso",   layout="flow_aniso",
         note="banded, width follows the tile", spec=dict(rim=0.11, core_shift=0.6), whole=True),

    dict(block="spread", name="spread_frac", layout="spread",
         note="wide size range, rim 0.11 of leaf", spec=dict(rim=0.11, core_shift=0.6)),
    dict(block="spread", name="spread_slot", layout="spread",
         note="wide size range, constant slot 3 mm", spec=dict(rim=0.0, rim_abs=3.0,
                                                               core_shift=0.6)),
    dict(block="spread", name="spread_slot_pale", layout="spread",
         note="constant slot 5 mm", spec=dict(rim=0.0, rim_abs=5.0, core_shift=0.6)),
    dict(block="spread", name="spread_slot_ground", layout="spread_ground",
         note="constant slot 3 mm over a fine ground",
         spec=dict(rim=0.0, rim_abs=3.0, core_shift=0.6), ground=dict(rings=1, rim=0.0)),

    dict(block="cut", name="poisson_web", layout="flow_lens", note="scattered, outline drawn",
         spec=dict(rim=0.11, core_shift=0.6), whole=True, web=0.5),
    dict(block="cut", name="poisson_bare", layout="flow_lens", note="scattered, outline removed",
         spec=dict(rim=0.11, core_shift=0.6), whole=True),
    dict(block="cut", name="lattice_strict", layout="lens_lattice", note="one per tile, regular",
         spec=dict(rim_abs=4.5, **CUT_INK), whole=True),
    dict(block="cut", name="lattice_loose_edges", layout="lens_loose",
         note="regular body, edges adrift", spec=dict(rim_abs=4.5, **CUT_INK), whole=True),
    dict(block="cut", name="lattice_mid", layout="lens_mid", note="regularity 0.6",
         spec=dict(rim_abs=4.5, **CUT_INK), whole=True),
    dict(block="cut", name="lattice_square", layout="lens_square", note="square bond, no offset",
         spec=dict(rim_abs=4.5, **CUT_INK), whole=True),
    dict(block="cut", name="slot_3mm", layout="lens_loose", note="3 mm border, darkest",
         spec=dict(rim_abs=3.0, **CUT_INK), whole=True),
    dict(block="cut", name="slot_6mm", layout="lens_loose", note="6 mm border, palest",
         spec=dict(rim_abs=6.0, **CUT_INK), whole=True),
    dict(block="cut", name="saddle_lattice", layout="saddle_lattice", note="the saddle flow",
         spec=dict(rim_abs=4.5, **CUT_INK), whole=True),
    dict(block="cut", name="vortex2_lattice", layout="vortex2_lattice", note="the paired eddies",
         spec=dict(rim_abs=4.5, **CUT_INK), whole=True),

    dict(block="calm", name="seigaiha", layout="seigaiha", note="wave rows, one heading",
         spec=dict(rim_abs=4.0, **CUT_INK), whole=True),
    dict(block="calm", name="nowaki", layout="nowaki", note="grasses, a single long swell",
         spec=dict(rim_abs=2.4, **CUT_INK), whole=True),
    # Needles are sparse, so as paper they would be islands with nothing holding them. Read the
    # other way round they are holes in a whole sheet, which is what a komon is.
    dict(block="calm", name="matsuba", layout="matsuba", note="pine needles as holes in the ground",
         spec=dict(rings=1, rim=0.0, invert=True), whole=True, ground_role="paper"),
    dict(block="calm", name="bokashi", layout="bokashi", note="one heading, scale grading across",
         spec=dict(rim_abs=4.0, **CUT_INK), whole=True),
    dict(block="calm", name="terrace", layout="terrace", note="broad bands, still direction",
         spec=dict(rim_abs=4.0, **CUT_INK), whole=True),

    dict(block="calm", name="seigaiha_kakishibu", layout="seigaiha", note="persimmon tannin",
         spec=dict(rim_abs=4.0, **CUT_INK), whole=True, palette="kakishibu"),
    dict(block="calm", name="nowaki_kakishibu", layout="nowaki", note="persimmon tannin",
         spec=dict(rim_abs=2.4, **CUT_INK), whole=True, palette="kakishibu"),
    dict(block="calm", name="seigaiha_ai", layout="seigaiha", note="indigo",
         spec=dict(rim_abs=4.0, **CUT_INK), whole=True, palette="ai"),
    dict(block="calm", name="nowaki_ai", layout="nowaki", note="indigo",
         spec=dict(rim_abs=2.4, **CUT_INK), whole=True, palette="ai"),
]


def build_layout_fn(name, cache):
    """The packing for one layout, built once and reused by every panel that shares it."""
    if name in cache:
        return cache[name]
    layout = LAYOUTS[name]
    tiles, meta = load_flat_tiles_fn(**{**SHEET, **layout["sheet"]})
    grid = (meta["M"], meta["N"])
    elements, report = build_scale_field_fn(tiles, grid=grid, **{**FIELD, **layout["field"]})
    report["grid"] = grid
    matrix, offset = fit_to_page_fn(report["sheet_bounds"], PAGE_MM, margin_mm=0.0)
    placed = transform_elements_fn(elements, scale=float(matrix[0, 0]), offset=offset)
    # An armature drawn from the tiling itself -- a chain, or a net on the centroid mesh -- needs the
    # tiles AND the transform the elements were placed with. Two regions built under two different
    # transforms union into a net that has slid off its own leaves.
    report["tiles"] = tiles
    report["transform"] = (float(matrix[0, 0]), offset)
    print(f"[layout] {name:20s} {meta['n_tiles']} tiles  {report['n_elements']} elements"
          f"  ({report['n_ground']} ground)  axis p95 {report['neighbour_angle_p95_deg']:.1f} deg"
          f"  opposed {report['opposed_neighbours']}")
    cache[name] = (placed, report)
    return cache[name]


def render_net_panel_fn(panel, out_dir, *, dpi, min_piece_mm2, write_svg):
    """One net panel: warp it onto the sheet, draw it, and measure it as a cut object."""
    page = (0.0, 0.0, *PAGE_MM)
    tiles, meta = load_flat_tiles_fn(**NET_SHEET, field_kind=panel["field"],
                                     amplitude=panel["amplitude"])
    paths, report = build_net_sheet_fn(tiles, (meta["M"], meta["N"]), panel["net"],
                                       scale=panel["scale"])
    matrix, offset = fit_to_page_fn(report["mesh_bounds"], PAGE_MM, margin_mm=6.0)
    scale = float(matrix[0, 0])
    paths = [transform_path_fn(path, matrix, offset) for path in paths]

    elements, specs = None, None
    if panel.get("figures"):
        placed, field = build_scale_field_fn(tiles, grid=(meta["M"], meta["N"]),
                                             **panel["figures"])
        elements = transform_elements_fn(placed, scale=scale, offset=offset)
        specs = {"main": ElementSpec(rim=0.0, rim_abs=3.0, duty=1.0, core_shift=0.6)}

    colours = PALETTES[panel.get("palette", "sumi")]
    png = os.path.join(out_dir, f"net_{panel['name']}.png")
    render_net_preview_fn(paths, png, width_mm=panel["width"], crop=page, dpi=dpi,
                          figure_width_in=7.0, elements=elements, specs=specs,
                          fill=colours["fill"], ground=colours["ground"])

    region = net_region_fn(paths, panel["width"], page=page)
    if elements is not None:
        figures, _ = resolve_paper_fn(elements, specs, page=page)
        region = region.union(figures)
    region, dust = prune_dust_fn(region, min_area=min_piece_mm2)
    stencil = measure_stencil_fn(region, [], page=page)
    stencil["join_residual"] = net_join_residual_fn(paths)
    stencil.update(dust)
    stencil.update({k: report[k] for k in ("n_pieces", "cells", "cell_area_spread")})

    if write_svg:
        cut = os.path.join(out_dir, f"net_{panel['name']}_cut.svg")
        write_svg_fn(cut, {"paper": cut_paths_fn(region)}, width_mm=PAGE_MM[0],
                     height_mm=PAGE_MM[1], background=None,
                     styles={"paper": {"fill": colours["fill"], "fill-rule": "evenodd",
                                       "stroke": "none"}},
                     precision=6, title=f"net_{panel['name']}_cut")
        stencil["cut_svg"] = cut
    return png, stencil


def render_strict_panel_fn(panel, out_dir, cache, *, dpi, min_piece_mm2, write_svg):
    """One strict panel: state the narrowest white, build it, then measure it against the statement."""
    page = (0.0, 0.0, *PAGE_MM)
    placed, report = build_layout_fn(panel["layout"], cache)
    field = {**FIELD, **LAYOUTS[panel["layout"]]["field"]}
    slot = panel["slot"]

    # The border is solved, not chosen: the element's own band is the only candidate for the
    # narrowest white once the packing has been fixed by the construction.
    if panel.get("hole") == "leaf":
        # A cell-confined hole needs no border of its own: the slot between two territories IS the
        # whole white, so the element is cut as its own silhouette and the web is the skeleton.
        rim, spec = 0.0, ElementSpec(rings=1, rim=0.0, invert=True)
    else:
        shape = dict(width=field["leaf_width"], bow=field["leaf_bow"])
        probe = ElementSpec(rings=panel.get("rings", 2), duty=panel.get("duty", 1.0),
                            core_shift=panel.get("core_shift", 0.3))
        rim = web.rim_for_slot_fn(probe, slot=slot, **shape)
        spec = ElementSpec(rings=probe.rings, rim_abs=rim, duty=probe.duty,
                           core_shift=probe.core_shift)
    specs = {"main": spec, "ground": spec}

    thread = None
    if panel["build"] == "gapped":
        paper, _, stencil = web.build_gapped_web_fn(placed, specs, page=page, clearance=slot)
    elif panel["build"] == "cell":
        # A full slot of border, not the half a slot the erosion leaves on its own: otherwise the
        # narrowest white on the sheet is the frame rather than any channel in the pattern.
        paper, _, _, stencil = web.build_cell_web_fn(placed, specs, page=page, clearance=slot,
                                                     mode=panel.get("mode", "crop"), frame=slot)
    elif panel["build"] == "skeleton":
        paper, thread, stencil = web.build_skeleton_web_fn(placed, specs, page=page, clearance=slot)
    elif panel["build"] == "thread":
        paper, thread, stencil = web.build_thread_web_fn(
            placed, specs, page=page, grid=report["grid"], tile_of=[e.tile for e in placed],
            thread_mm=slot, mode="grid")
    elif panel["build"] == "free":
        # The comparison: the same drawing resolved exactly as the `cut` block resolves it, with
        # nothing added and nothing kept back.
        from shapely.geometry import box

        paper = box(*page).difference(web.resolve_dark_fn(placed, specs, page=page))
        stencil = {"construction": "free", "clearance_mm": slot}
    else:
        raise ValueError(f"unknown strict build {panel['build']!r}")

    paper, dust = prune_dust_fn(paper, min_area=min_piece_mm2)
    stencil.update(dust)
    stencil.update(measure_stencil_fn(paper, [], page=page))
    stencil["rim_abs_mm"] = float(rim)
    stencil["min_white_mm"] = web.min_width_fn(paper)
    # Just below the slot, never at it. A web of exactly one width is annihilated by an opening at
    # that width, so retention measured AT the slot reads near zero for the most uniform sheets in
    # the set -- the knife edge is the evidence of uniformity, not a defect.
    stencil["kept_near_slot"] = web.opening_retention_fn(paper, 0.95 * slot)

    png = os.path.join(out_dir, f"strict_{panel['name']}.png")
    web.render_region_fn(paper, png, page=page, dpi=dpi, figure_width_in=7.0,
                         **{k: v for k, v in PALETTES["sumi"].items() if k != "edge"})
    if write_svg:
        cut = os.path.join(out_dir, f"strict_{panel['name']}_cut.svg")
        write_svg_fn(cut, {"paper": cut_paths_fn(paper)}, width_mm=PAGE_MM[0],
                     height_mm=PAGE_MM[1], background=None,
                     styles={"paper": {"fill": PALETTES["sumi"]["fill"],
                                       "fill-rule": "evenodd", "stroke": "none"}},
                     precision=6, title=f"strict_{panel['name']}_cut")
        stencil["cut_svg"] = cut
    return png, stencil, {"spec": {"rings": spec.rings, "rim_abs": rim, "duty": spec.duty,
                                   "core_shift": spec.core_shift, "invert": spec.invert}}


CELL_HOLE = ElementSpec(rings=1, rim=0.0, invert=True)


def render_cell_panel_fn(panel, out_dir, cache, *, dpi, min_piece_mm2, write_svg,
                         block: str = "cell"):
    """One cell panel: a hole per Voronoi territory, and the web that leaves.

    The hole is cut as the element's own silhouette rather than as a nested band: inside a cell the
    slot between territories IS the whole white, so a border of the element's own would only be a
    second, competing web.
    """
    import dataclasses

    page = (0.0, 0.0, *PAGE_MM)
    placed, report = build_layout_fn(panel["layout"], cache)
    specs = {"main": CELL_HOLE, "ground": CELL_HOLE}
    slot = panel["slot"]
    extra = {}

    if panel.get("lloyd"):
        # Relax the sites to the centroids of their own cells, then carry the elements to them, so
        # the packing is a centroidal Voronoi diagram and the cells even out toward a honeycomb.
        sites = np.array([element.leaf.centre for element in placed], float)
        moved, extra = web.lloyd_sites_fn(sites, page=page, passes=panel["lloyd"])
        # Element is a NamedTuple and Leaf a frozen dataclass, so each needs its own replace.
        placed = [element._replace(leaf=dataclasses.replace(element.leaf, centre=centre))
                  for element, centre in zip(placed, moved)]

    clearance = slot
    if panel.get("breathe"):
        # A per-cell slot taken from the local element scale, so the web is broad where the tiles are
        # large. The guarantee becomes the smallest value in the array, which the report states.
        lengths = np.array([element.leaf.length for element in placed], float)
        ratio = (lengths / np.median(lengths)) ** float(panel["breathe"])
        clearance = slot * np.clip(ratio, 0.6, 1.8)

    motif_holes = None
    if panel.get("motif") or panel.get("detail"):
        # The motif vocabulary sits outside `web.py` on purpose: it needs nothing from the
        # construction, and the construction needs nothing from it beyond a shape per cell.
        sites = np.array([element.leaf.centre for element in placed], float)
        angles = np.array([element.leaf.angle for element in placed], float)
        territories = web.territories_fn(sites, page=page, clearance=clearance,
                                         frame=float(np.min(clearance)))
        motif_holes = cell_motifs.motif_holes_fn(
            territories, sites, angles, name=panel.get("motif", "disc"),
            reach=panel.get("reach", 1.0), detail=panel.get("detail"))

    paper, holes, skeleton, stencil = web.build_cell_web_fn(
        placed, specs, page=page, clearance=clearance, mode=panel.get("mode", "crop"),
        frame=float(np.min(clearance)), corner_radius=panel.get("corner_radius", 0.0),
        fill=panel.get("fill", 1.0), holes=motif_holes)
    stencil.update(extra)
    stencil["motif"] = panel.get("motif") or ("detail" if panel.get("detail") else panel.get("mode"))
    stencil["reach"] = float(panel.get("reach", 1.0))

    paper, dust = prune_dust_fn(paper, min_area=min_piece_mm2)
    stencil.update(dust)
    stencil.update(measure_stencil_fn(paper, [], page=page))
    # 2.5x the slot is ample: the web IS the slot by construction, and a looser bound costs a
    # buffer of the whole sheet per bisection step, which bites hard at a few thousand cells.
    bound = 2.5 * float(np.max(clearance))
    stencil["min_white_mm"] = web.min_width_fn(paper, upper=bound)
    stencil["skeleton_min_mm"] = web.min_width_fn(skeleton, upper=bound)

    colours = dict(PALETTES[panel.get("palette", "sumi")])
    if panel.get("invert_ink"):
        # Only the ink is turned over. Cutting it this way really would hand back one loose piece per
        # cell, so the panel says so rather than implying an object that cannot exist.
        colours["fill"], colours["ground"] = colours["ground"], colours["fill"]
        stencil["cells_if_inverted"] = int(sum(not hole.is_empty for hole in holes))

    png = os.path.join(out_dir, f"{block}_{panel['name']}.png")
    web.render_region_fn(paper, png, page=page, fill=colours["fill"], ground=colours["ground"],
                         dpi=dpi, figure_width_in=7.0)
    if write_svg and not panel.get("invert_ink"):
        cut = os.path.join(out_dir, f"{block}_{panel['name']}_cut.svg")
        write_svg_fn(cut, {"paper": cut_paths_fn(paper)}, width_mm=PAGE_MM[0],
                     height_mm=PAGE_MM[1], background=None,
                     styles={"paper": {"fill": colours["fill"], "fill-rule": "evenodd",
                                       "stroke": "none"}},
                     precision=6, title=f"{block}_{panel['name']}_cut")
        stencil["cut_svg"] = cut
    return png, stencil


def render_line_panel_fn(panel, out_dir, cache, *, dpi, min_piece_mm2, write_svg):
    """One line panel: strokes along the contours, and the device that reading needs to hold.

    The preview paints the PAPER in the fill colour on the hole colour for both polarities, which
    renders as a dark line on cream when the stroke is the slot and a pale line on dark when the
    stroke is the paper. One code path, and the picture follows from which region is the paper.
    """
    from shapely.geometry import box

    page = (0.0, 0.0, *PAGE_MM)
    placed, report = build_layout_fn(panel["layout"], cache)
    width = panel["width"]

    ink = web.stroke_field_fn(placed, page=page, width=width, count=panel.get("count", 1),
                              spacing=panel.get("spacing"), kind=panel["kind"],
                              occlude=panel.get("occlude", True))
    stencil = {"construction": f"line/{panel['polarity']}/{panel['kind']}",
               "count": panel.get("count", 1), "occlude": panel.get("occlude", True)}
    stencil.update(web.stroke_weight_report_fn(ink, width=width))

    if panel["polarity"] == "slot":
        # The stroke is the slot, so the ground is divided into faces. An even pitch of gaps is the
        # decorative dial; the face bridges are what make it one piece.
        if panel.get("bridge_pitch"):
            ink, pitched = web.bridge_strokes_fn(ink, placed, spacing=panel["bridge_pitch"],
                                                 width=width, page=page)
            stencil.update(pitched)
        ink, bridged = web.bridge_faces_fn(ink, page=page, width=width)
        stencil.update(bridged)
        paper = box(*page).difference(ink)
    else:
        # The stroke IS the paper, so what can go wrong is a loop that touches nothing.
        paper = ink
        if panel.get("armature") == "mst":
            sites = np.array([element.leaf.centre for element in placed], float)
            edges, extra = web.mst_thread_fn(sites)
            armature = web.thread_region_fn(sites, edges, width=width, page=page)
            stencil.update(extra)
            stencil["armature_cost"] = web.thread_cost_fn(paper, armature, page=page)
            paper = paper.union(armature)

    stencil["pieces_before_keep"] = len(web._components(paper))
    paper, dropped = web.keep_largest_fn(paper)
    paper, dust = prune_dust_fn(paper, min_area=min_piece_mm2)
    stencil.update(dropped)
    stencil.update(dust)
    stencil.update(measure_stencil_fn(paper, [], page=page))
    stencil["paper_min_mm"] = web.min_width_fn(paper, upper=6.0 * width)

    colours = PALETTES[panel.get("palette", "sumi")]
    png = os.path.join(out_dir, f"line_{panel['name']}.png")
    web.render_region_fn(paper, png, page=page, fill=colours["fill"], ground=colours["ground"],
                         dpi=dpi, figure_width_in=7.0)
    if write_svg:
        cut = os.path.join(out_dir, f"line_{panel['name']}_cut.svg")
        write_svg_fn(cut, {"paper": cut_paths_fn(paper)}, width_mm=PAGE_MM[0],
                     height_mm=PAGE_MM[1], background=None,
                     styles={"paper": {"fill": colours["fill"], "fill-rule": "evenodd",
                                       "stroke": "none"}},
                     precision=6, title=f"line_{panel['name']}_cut")
        stencil["cut_svg"] = cut
    return png, stencil


def build_armature_fn(panel, placed, report, specs, *, page, drawing, sheet_part, islands):
    """The region unioned into the paper to hold what the drawing leaves loose.

    One code path for every rung of the ladder, so the panels differ in the armature and in nothing
    else. Returns ``(region, extra)`` with an empty region for the unheld control.
    """
    from shapely.geometry import Polygon

    kind = panel["armature"]
    width = panel.get("armature_mm", panel["slot"])
    sites = np.array([element.leaf.centre for element in placed], float)
    factor, offset = report["transform"]

    if kind in ("none", "islands"):
        return Polygon(), {}
    if kind == "nicks":
        return web.island_tabs_fn(islands, placed, sheet_part, width=width, page=page)
    if kind == "mst":
        edges, extra = web.mst_thread_fn(sites)
        return web.thread_region_fn(sites, edges, width=width, page=page), extra
    if kind == "grid":
        edges, extra = web.lattice_edges_fn([e.tile for e in placed], report["grid"], mode="grid",
                                            sites=sites)
        return web.thread_region_fn(sites, edges, width=width, page=page), extra
    if kind == "chains":
        polylines, extra = web.chain_thread_fn(report["tiles"], scale=factor, offset=offset)
        return web.polyline_region_fn(polylines, width=width, page=page), extra
    if kind == "skeleton":
        # `frame` is spelled out per panel here because its default is the whole point of one pair:
        # unframed, the border strip is half a slot and the sheet's narrowest white is the page edge.
        frame = panel.get("frame", panel["slot"])
        region = web.skeleton_region_fn(sites, page=page, clearance=panel["slot"], frame=frame)
        return region, {"frame_mm": 0.0 if frame is None else float(frame),
                        "skeleton_share": float(region.area / (page[2] * page[3]))}
    if kind.startswith("net:"):
        region, extra = web.net_thread_fn(report["tiles"], report["grid"], kind[4:], width=width,
                                          scale=panel.get("net_scale", 2.5),
                                          transform=(factor, offset), page=page)
        if panel.get("nick_gaps") and islands:
            # A net is a global pattern, so it misses whatever elements fall inside one of its cells,
            # and those elements' islands are then cut away rather than held. Nicking only the ones
            # the net does not already reach keeps the motif intact and the sheet complete.
            missed = [(island, host) for island, host in islands if not island.intersects(region)]
            if missed:
                nicks, tab_extra = web.island_tabs_fn(missed, placed, sheet_part, width=width,
                                                      page=page)
                region = region.union(nicks)
                extra.update({f"gap_{key}": value for key, value in tab_extra.items()})
        return region, extra
    raise ValueError(f"unknown armature {kind!r}")


def render_lace_panel_fn(panel, out_dir, cache, *, dpi, min_piece_mm2, write_svg):
    """One lace panel: the drawing, the armature it is held by, and what that armature cost."""
    from shapely.geometry import Polygon, box

    page = (0.0, 0.0, *PAGE_MM)
    placed, report = build_layout_fn(panel["layout"], cache)
    field = {**FIELD, **LAYOUTS[panel["layout"]]["field"]}
    slot = panel["slot"]

    rings = panel.get("rings", 2)
    duty = panel.get("duty", 1.0 if rings == 2 else 0.5)
    shift = panel.get("core_shift", 0.3)
    probe = ElementSpec(rings=rings, duty=duty, core_shift=shift)
    rim = web.rim_for_slot_fn(probe, width=field["leaf_width"], bow=field["leaf_bow"], slot=slot)
    spec = ElementSpec(rings=rings, rim_abs=rim, duty=duty, core_shift=shift)
    specs = {"main": spec, "ground": spec}

    drawing = box(*page).difference(web.resolve_dark_fn(placed, specs, page=page))
    parts = web._components(drawing)
    sheet_part = max(parts, key=lambda part: part.area) if parts else Polygon()
    census, islands = web.island_census_fn(drawing, placed, specs)

    armature, extra = build_armature_fn(panel, placed, report, specs, page=page, drawing=drawing,
                                        sheet_part=sheet_part, islands=islands)
    paper = drawing if armature.is_empty else drawing.union(armature)

    # The control keeps every piece, or it would look repaired. Everything else keeps what the
    # armature actually holds and reports the rest as cut away.
    if panel["armature"] in ("none", "islands"):
        dropped = {"islands_dropped": 0, "islands_area": 0.0}
    else:
        paper, dropped = web.keep_largest_fn(paper)
    paper, dust = prune_dust_fn(paper, min_area=min_piece_mm2)

    guaranteed = slot if armature.is_empty else min(slot, panel.get("armature_mm", slot))
    stencil = {"construction": f"lace/{panel['armature']}", "slot_mm": float(slot),
               "armature_mm": float(panel.get("armature_mm", slot)),
               "guaranteed_mm": float(guaranteed), "rim_abs_mm": float(rim),
               "armature_components": len(web._components(armature))}
    stencil.update({f"census_{key}": value for key, value in census.items()})
    stencil.update(extra)
    stencil.update(dropped)
    stencil.update(dust)
    stencil.update(measure_stencil_fn(paper, [], page=page))
    stencil["min_white_mm"] = web.min_width_fn(paper)
    stencil["kept_near_slot"] = web.opening_retention_fn(paper, 0.95 * guaranteed)
    stencil["armature_cost"] = web.thread_cost_fn(drawing, armature, page=page)
    stencil.update(web.fans_opened_fn(placed, specs, armature))

    colours = PALETTES[panel.get("palette", "sumi")]
    png = os.path.join(out_dir, f"lace_{panel['name']}.png")
    if panel["armature"] == "islands":
        # The census drawn: the sheet in paper, every piece that would fall out of it in the hole
        # colour's complement, so the two are told apart at a glance.
        web.render_regions_fn([(sheet_part, colours["fill"]), *[(island, "#C8632F")
                                                                for island, _ in islands]],
                              png, page=page, ground=colours["ground"], dpi=dpi,
                              figure_width_in=7.0)
    else:
        web.render_region_fn(paper, png, page=page, fill=colours["fill"],
                             ground=colours["ground"], dpi=dpi, figure_width_in=7.0)
        if write_svg:
            cut = os.path.join(out_dir, f"lace_{panel['name']}_cut.svg")
            write_svg_fn(cut, {"paper": cut_paths_fn(paper)}, width_mm=PAGE_MM[0],
                         height_mm=PAGE_MM[1], background=None,
                         styles={"paper": {"fill": colours["fill"], "fill-rule": "evenodd",
                                           "stroke": "none"}},
                         precision=6, title=f"lace_{panel['name']}_cut")
            stencil["cut_svg"] = cut
    return png, stencil, {"spec": {"rings": rings, "rim_abs": rim, "duty": duty,
                                   "core_shift": shift}}


def panel_colours_fn(panel):
    """The palette, with the page painted as whatever the untouched sheet is.

    When the elements are holes rather than figures, the page IS the paper. Leaving it the hole
    colour paints hole on hole and the whole panel goes flat -- which is exactly what happened to
    the needles the first time this ran.
    """
    colours = dict(PALETTES[panel.get("palette", "sumi")])
    if panel.get("ground_role") == "paper":
        colours["ground"] = colours["fill"]
    return colours


def panel_specs_fn(panel):
    """The ink for each population in one panel."""
    return {"main": ElementSpec(**panel["spec"]),
            "ground": ElementSpec(**panel.get("ground", panel["spec"]))}


def write_panel_svg_fn(placed, specs, out_svg, *, title=None, colours=None, web=0.0):
    colours = colours or PALETTES["sumi"]
    pale = ({"fill": colours["fill"], "stroke": colours["edge"],
             "stroke-width": f"{web:g}", "stroke-linejoin": "round"}
            if web > 0.0 else {"fill": colours["fill"], "stroke": "none"})
    dark = {"fill": colours["edge"], "stroke": "none"}
    styling = {"pale": pale, "dark": dark}
    paths = []
    for element in placed:
        for depth, (leaf, tag) in enumerate(element_layers_fn(element.leaf,
                                                              specs[element.population])):
            style = styling[tag] if depth == 0 else {**styling[tag], "stroke": "none"}
            paths.append((leaf_path_fn(leaf), style))
    # The title comes from the panel, not from the path, so replaying a recipe to a scratch file
    # reproduces the recorded SVG byte for byte.
    return write_svg_fn(out_svg, {"scales": paths}, width_mm=PAGE_MM[0], height_mm=PAGE_MM[1],
                        background=colours["ground"], styles={"scales": pale}, precision=6,
                        title=title or os.path.splitext(os.path.basename(out_svg))[0])


def montage_fn(panels, out_png, *, title, columns=3, dpi=110):
    """Tile the panel images into one labelled sheet."""
    from nff.utils import figstyle

    figstyle.use_agg()
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    figstyle.apply_charter(**{"axes.grid": False})
    rows = int(np.ceil(len(panels) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(4.6 * columns, 3.9 * rows),
                                constrained_layout=True)
    for axis, panel in zip(np.atleast_1d(axes).ravel(), panels):
        axis.imshow(mpimg.imread(panel["png"]))
        axis.set_title(f"{panel['name']}\n{panel['note']}", fontsize=8, pad=6)
        axis.axis("off")
    for axis in np.atleast_1d(axes).ravel()[len(panels):]:
        axis.axis("off")
    figure.suptitle(title, fontsize=13)
    figure.savefig(out_png, dpi=dpi, facecolor="white")
    plt.close(figure)
    return out_png


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default="data/outputs/katagami/iterate")
    parser.add_argument("--blocks", nargs="*", default=None,
                        help="which blocks to render; default all")
    parser.add_argument("--dpi", type=int, default=150, help="panel resolution")
    parser.add_argument("--no-svg", action="store_true", help="skip the vector output")
    parser.add_argument("--min-piece-mm2", type=float, default=20.0,
                        help="cut fragments smaller than this are crumbs; dropped and reported")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    wanted = args.blocks or sorted(
        {p["block"] for p in PANELS} | {"strict", "lace", "line", "cell", "motif",
                                        "vocab", "density"},
        key=BLOCK_ORDER.index)
    cache = {}

    # Rendering a single block must not drop the others from the manifest, or a partial re-run
    # silently loses the recipes it did not touch.
    path = os.path.join(args.out_dir, "manifest.yaml")
    manifest = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            manifest = [p for p in (yaml.safe_load(handle) or {}).get("panels", [])
                        if p["block"] not in wanted]

    if "net" in wanted:
        shown = []
        for panel in NET_PANELS:
            png, stencil = render_net_panel_fn(
                panel, args.out_dir, dpi=args.dpi, min_piece_mm2=args.min_piece_mm2,
                write_svg=not args.no_svg)
            note = panel.get("note", f"{panel['net']}, {panel['field']} {panel['amplitude']:g}")
            shown.append({"name": panel["name"], "png": png, "note": (
                f"{note}\n{stencil['n_components']} piece"
                f"{'s' if stencil['n_components'] != 1 else ''}, bridge "
                f"{stencil['min_bridge_mm']:.1f} mm, ink {100 * stencil['coverage']:.0f}%"
                f"\n{stencil['n_pieces']} warped pieces, join {stencil['join_residual']:.0e}")})
            manifest.append({"block": "net", "name": panel["name"], "note": note,
                             "sheet": {**NET_SHEET, "field_kind": panel["field"],
                                       "amplitude": panel["amplitude"]},
                             "net": {k: panel[k] for k in ("net", "scale", "width")},
                             "figures": panel.get("figures"),
                             "palette": panel.get("palette", "sumi"),
                             "png": png, "stencil": stencil})
            print(f"[panel]  net/{panel['name']:20s} -> {png}")
        print(f"[block]  {montage_fn(shown, os.path.join(args.out_dir, 'block_net.png'), title='katagami nets -- warped by the tessellation')}")

    if "strict" in wanted:
        shown = []
        for panel in STRICT_PANELS:
            png, stencil, recipe = render_strict_panel_fn(
                panel, args.out_dir, cache, dpi=args.dpi, min_piece_mm2=args.min_piece_mm2,
                write_svg=not args.no_svg)
            shown.append({"name": panel["name"], "png": png, "note": (
                f"{panel['note']}\n{stencil['n_components']} piece"
                f"{'s' if stencil['n_components'] != 1 else ''}, ink "
                f"{100 * stencil['coverage']:.0f}%, slot {panel['slot']:g} mm"
                f"\nnarrowest white {stencil['min_white_mm']:.2f} mm, "
                f"{100 * stencil['kept_near_slot']:.0f}% of it at 0.95 slot")})
            manifest.append({"block": "strict", "name": panel["name"], "note": panel["note"],
                             "layout": panel["layout"],
                             "sheet": {**SHEET, **LAYOUTS[panel["layout"]]["sheet"]},
                             "field": {**FIELD, **LAYOUTS[panel["layout"]]["field"]},
                             "build": panel["build"], "slot_mm": panel["slot"],
                             "mode": panel.get("mode"), **recipe,
                             "png": png, "stencil": stencil})
            print(f"[panel]  strict/{panel['name']:20s} -> {png}")
        print("[block]  " + montage_fn(shown, os.path.join(args.out_dir, "block_strict.png"),
                                       title="katagami -- the white web made strict"))

    for block, panels, title in (("cell", CELL_PANELS, "katagami -- the cell web"),
                                 ("motif", MOTIF_PANELS, "katagami -- motifs and densities"),
                                 ("vocab", VOCAB_PANELS, "katagami -- the motif vocabulary"),
                                 ("density", DENSITY_PANELS,
                                  "katagami -- 12 cells to 6200")):
        if block not in wanted:
            continue
        shown = []
        for panel in panels:
            png, stencil = render_cell_panel_fn(
                panel, args.out_dir, cache, dpi=args.dpi, min_piece_mm2=args.min_piece_mm2,
                write_svg=not args.no_svg, block=block)
            shown.append({"name": panel["name"], "png": png, "note": (
                f"{panel['note']}\n{stencil['n_components']} piece"
                f"{'s' if stencil['n_components'] != 1 else ''}, ink "
                f"{100 * stencil['coverage']:.0f}%, slot {stencil['clearance_mm']:.1f}"
                f"{'-%.1f' % stencil['clearance_max_mm'] if stencil['clearance_max_mm'] > stencil['clearance_mm'] + 1e-6 else ''} mm"
                f"\nweb {stencil['skeleton_min_mm']:.2f} mm, gap med "
                f"{stencil['gap_median_mm']:.1f}, {100 * stencil['at_clearance']:.0f}% at slot")})
            manifest.append({"block": block, "name": panel["name"], "note": panel["note"],
                             "layout": panel["layout"],
                             "sheet": {**SHEET, **LAYOUTS[panel["layout"]]["sheet"]},
                             "field": {**FIELD, **LAYOUTS[panel["layout"]]["field"]},
                             "cell": {k: panel.get(k) for k in
                                      ("slot", "mode", "fill", "corner_radius", "lloyd",
                                       "breathe", "invert_ink", "motif", "reach", "detail")},
                             "palette": panel.get("palette", "sumi"),
                             "png": png, "stencil": stencil})
            print(f"[panel]  {block}/{panel['name']:20s} -> {png}")
        print("[block]  " + montage_fn(shown, os.path.join(args.out_dir, f"block_{block}.png"),
                                       title=title))

    if "line" in wanted:
        shown = []
        for panel in LINE_PANELS:
            png, stencil = render_line_panel_fn(
                panel, args.out_dir, cache, dpi=args.dpi, min_piece_mm2=args.min_piece_mm2,
                write_svg=not args.no_svg)
            shown.append({"name": panel["name"], "png": png, "note": (
                f"{panel['note']}\n{stencil['n_components']} piece"
                f"{'s' if stencil['n_components'] != 1 else ''} from "
                f"{stencil['pieces_before_keep']}, paper {100 * stencil['coverage']:.0f}%"
                f"\nline {stencil['ink_min_mm']:.2f} / {panel['width']:g} mm, "
                f"{100 * stencil['ink_at_weight']:.0f}% at weight, "
                f"bridges {stencil.get('n_face_bridges', 0)}")})
            manifest.append({"block": "line", "name": panel["name"], "note": panel["note"],
                             "layout": panel["layout"],
                             "sheet": {**SHEET, **LAYOUTS[panel["layout"]]["sheet"]},
                             "field": {**FIELD, **LAYOUTS[panel["layout"]]["field"]},
                             "stroke": {k: panel.get(k) for k in
                                        ("width", "kind", "count", "spacing", "polarity",
                                         "occlude", "bridge_pitch", "armature")},
                             "palette": panel.get("palette", "sumi"),
                             "png": png, "stencil": stencil})
            print(f"[panel]  line/{panel['name']:20s} -> {png}")
        print("[block]  " + montage_fn(shown, os.path.join(args.out_dir, "block_line.png"),
                                       title="katagami -- the sheet as a DRAWING"))

    if "lace" in wanted:
        shown = []
        for panel in LACE_PANELS:
            png, stencil, recipe = render_lace_panel_fn(
                panel, args.out_dir, cache, dpi=args.dpi, min_piece_mm2=args.min_piece_mm2,
                write_svg=not args.no_svg)
            shown.append({"name": panel["name"], "png": png, "note": (
                f"{panel['note']}\n{stencil['n_components']} piece"
                f"{'s' if stencil['n_components'] != 1 else ''}, ink "
                f"{100 * stencil['coverage']:.0f}%, cost "
                f"{100 * stencil['armature_cost']:.2f}%"
                f"\nwhite {stencil['min_white_mm']:.2f} / {stencil['guaranteed_mm']:g} mm, "
                f"islands {stencil['census_n_islands']}, "
                f"dropped {stencil['islands_dropped']}, fans "
                f"{stencil['rings_opened']}/{stencil['rings_closed']}")})
            manifest.append({"block": "lace", "name": panel["name"], "note": panel["note"],
                             "layout": panel["layout"],
                             "sheet": {**SHEET, **LAYOUTS[panel["layout"]]["sheet"]},
                             "field": {**FIELD, **LAYOUTS[panel["layout"]]["field"]},
                             "armature": panel["armature"], "slot_mm": panel["slot"],
                             "armature_mm": panel.get("armature_mm", panel["slot"]),
                             "net_scale": panel.get("net_scale"),
                             "palette": panel.get("palette", "sumi"), **recipe,
                             "png": png, "stencil": stencil})
            print(f"[panel]  lace/{panel['name']:20s} -> {png}")
        print("[block]  " + montage_fn(shown, os.path.join(args.out_dir, "block_lace.png"),
                                       title="katagami -- the armature as a motif"))

    for block in [b for b in wanted if b not in ("net", "strict", "lace", "line", "cell",
                                                "motif", "vocab", "density")]:
        panels = [p for p in PANELS if p["block"] == block]
        for panel in panels:
            placed, report = build_layout_fn(panel["layout"], cache)
            specs = panel_specs_fn(panel)
            png = os.path.join(args.out_dir, f"{block}_{panel['name']}.png")
            crop = (0.0, 0.0, *PAGE_MM) if panel.get("whole") else DETAIL_MM
            # No outline unless a panel asks for one. The stroke is the only thing drawing a
            # boundary between two elements that are one piece of paper, so it is off by default
            # and only the panel documenting the old look sets it.
            render_scale_preview_fn(placed, png, specs=specs, web_mm=panel.get("web", 0.0),
                                    crop=crop, dpi=args.dpi, figure_width_in=7.0,
                                    **panel_colours_fn(panel))
            panel["png"] = png

            stencil = None
            if block in ("cut", "calm"):
                page = (0.0, 0.0, *PAGE_MM)
                paper, pieces = resolve_paper_fn(placed, specs, page=page,
                                                 ground=panel.get("ground_role", "hole"))
                paper, dust = prune_dust_fn(paper, min_area=args.min_piece_mm2)
                stencil = measure_stencil_fn(paper, pieces, page=page, grid=report["grid"],
                                             tile_of=[e.tile for e in placed])
                stencil.update(dust)
                crumbs = (f", {dust['dust_dropped']} crumbs dropped"
                          if dust["dust_dropped"] else "")
                panel["note"] += (
                    f"\n{stencil['n_components']} piece"
                    f"{'s' if stencil['n_components'] != 1 else ''}{crumbs}"
                    f", bridge {stencil['min_bridge_mm']:.1f} mm, ink "
                    f"{100 * stencil['coverage']:.0f}%"
                    f"\ncontacts {stencil['degree_median']:.0f} typ / {stencil['degree_max']} max, "
                    f"{100 * stencil.get('lattice_incidence', 0.0):.0f}% on lattice")
                if not args.no_svg:
                    cut = os.path.join(args.out_dir, f"{block}_{panel['name']}_cut.svg")
                    ink = PALETTES[panel.get("palette", "sumi")]["fill"]
                    write_svg_fn(cut, {"paper": cut_paths_fn(paper)},
                                 width_mm=PAGE_MM[0], height_mm=PAGE_MM[1], background=None,
                                 styles={"paper": {"fill": ink, "fill-rule": "evenodd",
                                                   "stroke": "none"}},
                                 precision=6, title=f"{block}_{panel['name']}_cut")
                    stencil["cut_svg"] = cut
            record = {"block": block, "name": panel["name"], "note": panel["note"],
                      "layout": panel["layout"], "sheet": {**SHEET, **LAYOUTS[panel["layout"]]["sheet"]},
                      "field": {**FIELD, **LAYOUTS[panel["layout"]]["field"]},
                      "spec": panel["spec"], "ground_spec": panel.get("ground"),
                      "crop_mm": list(crop), "png": png,
                      "n_elements": report["n_elements"], "stencil": stencil}
            if not args.no_svg:
                svg = os.path.join(args.out_dir, f"{block}_{panel['name']}.svg")
                written = write_panel_svg_fn(placed, specs, svg,
                                             title=f"{block}_{panel['name']}",
                                             colours=panel_colours_fn(panel),
                                             web=panel.get("web", 0.0))
                record["svg"] = written["path"]
                record["n_paths"] = written["n_paths"]
            manifest.append(record)
            print(f"[panel]  {block}/{panel['name']:20s} -> {png}")

        sheet = montage_fn(panels, os.path.join(args.out_dir, f"block_{block}.png"),
                           title=f"katagami inked leaf -- {block}")
        print(f"[block]  {sheet}")

    manifest.sort(key=lambda p: (BLOCK_ORDER.index(p["block"]), p["name"]))
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump({"page_mm": list(PAGE_MM), "panels": manifest}, handle, sort_keys=False)
    print(f"[manifest] {path}  {len(manifest)} panels")


if __name__ == "__main__":
    main()
