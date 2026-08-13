"""Render a labelled sweep around the inked-leaf katagami, one block of variations per sheet.

The element is a pale leaf carrying a nested dark core. Four things are varied, each in its own
block: how much of the element is ink, where the core sits inside it, whether a finer second
population lies under the figures, and what the sheet's own deformation does to the flow.

Panels are grouped by LAYOUT. Everything that changes only the ink is drawn over one packing, so a
tonal comparison is of the tone alone and not of a different scatter of leaves. Every recipe is
written to a manifest beside the images, because a variation nobody can rebuild is not a choice.

    PYTHONPATH=$(pwd) JAX_PLATFORMS=cpu python3 nff/scripts/figures/katagami_contact_sheet.py
"""

import argparse
import os

import numpy as np
import yaml

from nff.closed.katagami import load_flat_tiles_fn
from nff.closed.scale_field import (
    PALETTES,
    build_scale_field_fn,
    render_scale_preview_fn,
    transform_elements_fn,
)
from nff.closed.net_field import (
    build_net_sheet_fn,
    net_join_residual_fn,
    net_region_fn,
    render_net_preview_fn,
)
from nff.closed.stencil import (
    cut_paths_fn,
    measure_stencil_fn,
    prune_dust_fn,
    resolve_paper_fn,
)
from nff.topology.quad_warp import transform_path_fn
from nff.topology.leaf import ElementSpec, element_layers_fn, leaf_path_fn
from nff.topology.svg_export import fit_to_page_fn, write_svg_fn

PAGE_MM = (420.0, 297.0)
PALE, DARK = "#F2EFE6", "#141414"
BLOCK_ORDER = ["tone", "core", "spread", "ground", "flow", "cut", "calm", "net"]

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
    wanted = args.blocks or sorted({p["block"] for p in PANELS}, key=BLOCK_ORDER.index)
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

    for block in [b for b in wanted if b != "net"]:
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
