"""Draw Ise-katagami artwork on a flat closed-pipeline sheet and write it as exact vector paths.

One motif, adapted into every tile of an irregular quadrilateral tiling produced by inverse design.
Because each tile is a different quad, a regular motif becomes an irregular repeat governed by a
rule. Flat state only; nothing here is meant to be cut.

    PYTHONPATH=$(pwd) JAX_PLATFORMS=cpu conda run -n kgnn_mac python \
      nff_paper/figures/export_katagami.py \
      --run-dir data/outputs/runs/run_20260806_183646_figmat_1_deploy \
      --motif asanoha --fit-mm 420x297 --preview
"""

import argparse
import os

import numpy as np

from nff_paper.closed_support.katagami import (
    build_vine_fn,
    compose_katagami_fn,
    corner_join_residual_fn,
    layer_bounds_fn,
    load_flat_tiles_fn,
    render_preview_fn,
    shared_corners_fn,
    transform_layers_fn,
)
from nff_paper.closed_support.scale_field import (
    build_scale_field_fn,
    render_scale_preview_fn,
    transform_elements_fn,
)
from nff_paper.closed_support.stencil import (
    cut_paths_fn,
    measure_stencil_fn,
    prune_dust_fn,
    resolve_paper_fn,
)
from nff_paper.topology_art.leaf import ElementSpec, element_layers_fn, leaf_path_fn
from nff_paper.closed_support.katagami import R_FIELDS
from nff_paper.topology_art.motif import MOTIFS
from nff_paper.topology_art.quad_warp import warp_polyline_fn
from nff_paper.topology_art.svg_export import fit_to_page_fn, write_svg_fn
from nff.utils import figstyle

DEFAULT_RUN = "data/outputs/runs/run_20260806_183646_figmat_1_deploy"


def _page(text):
    try:
        width, height = (float(v) for v in text.lower().split("x"))
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"expected WIDTHxHEIGHT in mm, got {text!r}") from error
    return width, height


def _grid(text):
    try:
        cols, rows = (int(v) for v in text.lower().split("x"))
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"expected MxN, got {text!r}") from error
    return cols, rows


def _parse_overrides(items):
    out = {}
    for item in items or []:
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"expected key=value, got {item!r}")
        key, value = item.split("=", 1)
        try:
            out[key] = int(value)
        except ValueError:
            try:
                out[key] = float(value)
            except ValueError:
                out[key] = value
    return out


def element_specs_fn(args) -> dict:
    """How each population is inked, keyed by population name."""
    main = ElementSpec(
        rings=args.rings, rim=args.inner, rim_abs=args.rim_abs, duty=args.duty,
        core_shift=args.core_shift, core_turn=args.core_turn,
        core_width_delta=args.core_width_delta, core_bow_delta=args.core_bow_delta,
        invert=args.invert)
    ground = {
        "same": main,
        "pale": ElementSpec(rings=1, rim=0.0),
        "dark": ElementSpec(rings=1, rim=0.0, invert=True),
    }[args.ground_ink]
    return {"main": main, "ground": ground}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--run-dir", help="trained run folder with config.yaml + best_params.pkl")
    source.add_argument("--design-npz", help="cached shield design carrying flat_tiles")
    source.add_argument("--synthetic", type=_grid, metavar="MxN",
                        help="build a sheet from a smooth random field, no run needed")
    parser.add_argument("--seed", type=int, default=0, help="synthetic field seed")
    parser.add_argument("--field", default="smooth", choices=sorted(R_FIELDS),
                        help="which r-field builds the synthetic sheet; changes the FLOW")
    parser.add_argument("--amplitude", type=float, default=2.0,
                        help="synthetic field logit amplitude; larger is more irregular")

    parser.add_argument("--motif", default="vine", choices=sorted(MOTIFS) + ["vine", "scales"],
                        help="'vine' follows the tiling's own diagonal chains as tapered ribbons")
    parser.add_argument("--density", type=float, default=0.24,
                        help="packing radius as a fraction of element length; <0.5 overlaps")
    parser.add_argument("--size-frac", type=float, default=2.10)
    parser.add_argument("--smoothing", type=int, default=8,
                        help="fewer passes lets the irregular tiling show as direction play")
    parser.add_argument("--aspect", type=float, default=1.0,
                        help="tile pitch y/x; above 1 makes tall tiles and an upright field")
    parser.add_argument("--size-gamma", type=float, default=1.0,
                        help="exaggerate the element size range; 2 roughly squares the spread")
    parser.add_argument("--size-smoothing", type=int, default=None,
                        help="passes on the size field; fewer keeps big and small apart")
    parser.add_argument("--rings", type=int, default=2,
                        help="nested contours per element; 2 is a pale shape with an inked core, "
                             "6 or more turns each element into a comb of parallel arcs")
    parser.add_argument("--hide-tips", action="store_true",
                        help="cover each point with the neighbouring base; smooths a spiky field")
    parser.add_argument("--duty", type=float, default=0.5,
                        help="fraction of each nested band that is pale; below 0.5 gives thin "
                             "pale slots on a dark ground, as a cut stencil reads")
    parser.add_argument("--inner", type=float, default=0.075,
                        help="pale rim width as a fraction of leaf length; the rest of the "
                             "interior is inked. 0 leaves the leaf solid pale")
    parser.add_argument("--rim-abs", type=float, default=0.0,
                        help="rim width in page mm instead of per leaf; one slot width across the "
                             "sheet, so size reads as scale and not as line weight")
    parser.add_argument("--core-shift", type=float, default=0.0,
                        help="slide the inked core toward the base, as a fraction of the rim; the "
                             "pale rim then widens into a highlight at the tip")
    parser.add_argument("--core-turn", type=float, default=0.0,
                        help="rotate the core off the leaf axis, as a fraction of the rim")
    parser.add_argument("--core-width-delta", type=float, default=0.0,
                        help="thin or fatten the core relative to its leaf; negative gives a spine")
    parser.add_argument("--core-bow-delta", type=float, default=0.0)
    parser.add_argument("--invert", action="store_true",
                        help="dark leaves holding pale cores, on the pale ground")
    parser.add_argument("--ground-scale", type=float, default=0.0,
                        help="add a finer second population at this fraction of the main element "
                             "size, painted underneath; 0 for none")
    parser.add_argument("--ground-density", type=float, default=0.34)
    parser.add_argument("--ground-ink", default="pale", choices=("same", "pale", "dark"))
    parser.add_argument("--aniso-power", type=float, default=0.0,
                        help="how strongly leaf width follows the local tile elongation")
    parser.add_argument("--placement", default="poisson", choices=("poisson", "lattice"),
                        help="'lattice' puts one element per tile, so which elements fuse when cut "
                             "is decided rather than left to the dart-throwing")
    parser.add_argument("--bond", type=float, default=1.0,
                        help="lattice row offset; 0 a square grid, 1 a half-cell shingle")
    parser.add_argument("--regularity", type=float, default=1.0,
                        help="1 places every element exactly on its lattice site; lower drifts")
    parser.add_argument("--edge-looseness", type=float, default=0.0,
                        help="regularity given up at the sheet boundary, so the edges break up")
    parser.add_argument("--cut-svg", default=None,
                        help="also resolve the painting into the region a blade would leave, and "
                             "write it as one even-odd path per connected piece")
    parser.add_argument("--min-piece-mm2", type=float, default=1.0,
                        help="fragments smaller than this are dropped from the cut file")
    parser.add_argument("--leaf-width", type=float, default=0.62)
    parser.add_argument("--leaf-bow", type=float, default=0.22)
    parser.add_argument("--angle-offset", type=float, default=0.0, help="degrees off the local axis")
    parser.add_argument("--depth-axis", type=float, default=-90.0)
    parser.add_argument("--web-mm", type=float, default=0.5)
    parser.add_argument("--vine-width", type=float, default=0.10,
                        help="base half-width as a fraction of the median tile radius")
    parser.add_argument("--vine-swell", type=float, default=0.45)
    parser.add_argument("--vine-taper", type=float, default=0.7)
    parser.add_argument("--vine-handle", type=float, default=0.42)
    parser.add_argument("--motif-param", action="append", metavar="KEY=VALUE")
    parser.add_argument("--node", default="none", help="corner ornament, or 'none'")
    parser.add_argument("--node-param", action="append", metavar="KEY=VALUE")
    parser.add_argument("--node-scale", type=float, default=0.17)
    parser.add_argument("--no-lod", action="store_true",
                        help="use one detail level everywhere instead of varying it by tile area")
    parser.add_argument("--list-motifs", action="store_true")

    parser.add_argument("--out", help="output .svg (default alongside the source)")
    parser.add_argument("--fit-mm", type=_page, default=(420.0, 297.0), metavar="WxH")
    parser.add_argument("--margin-mm", type=float, default=12.0)
    parser.add_argument("--stroke-mm", type=float, default=0.4)
    parser.add_argument("--show-tiles", action="store_true", help="draw the tile edges too")
    parser.add_argument("--background", default=None, help="page fill, e.g. white")
    parser.add_argument("--curve", default="q", choices=("q", "c"))
    parser.add_argument("--precision", type=int, default=6)
    parser.add_argument("--preview", action="store_true", help="also write a PNG preview")
    args = parser.parse_args()

    if args.list_motifs:
        print("motifs: " + ", ".join(sorted(MOTIFS)))
        return

    if not any((args.run_dir, args.design_npz, args.synthetic)):
        args.run_dir = DEFAULT_RUN

    tiles, meta = load_flat_tiles_fn(
        run_dir=args.run_dir, design_npz=args.design_npz, synthetic=args.synthetic,
        seed=args.seed, amplitude=args.amplitude, field_kind=args.field, aspect=args.aspect,
    )
    corners = shared_corners_fn(tiles)
    areas = meta["areas"]
    interior = int((corners["valence"] >= 2).sum())
    print(f"[sheet]  {meta['kind']}  {meta['n_tiles']} tiles  from {meta['source']}")
    print(f"[sheet]  tile area min/median/max {areas.min():.4g} / {np.median(areas):.4g} / "
          f"{areas.max():.4g}  -> spread {areas.max() / max(areas.min(), 1e-12):.1f}x")
    print(f"[sheet]  {len(corners['positions'])} shared corners, {interior} joining 2+ tiles  "
          f"| corner join residual {corner_join_residual_fn(tiles):.2e}")

    if args.motif == "scales":
        grid = (meta["M"], meta["N"]) if "M" in meta else None
        placed, field = build_scale_field_fn(
            tiles, smoothing=args.smoothing, size_smoothing=args.size_smoothing,
            size_gamma=args.size_gamma, density=args.density, size_frac=args.size_frac,
            hide_tips=args.hide_tips,
            leaf_width=args.leaf_width, leaf_bow=args.leaf_bow,
            ground_scale=args.ground_scale, ground_density=args.ground_density,
            aniso_power=args.aniso_power, placement=args.placement, grid=grid,
            bond=args.bond, regularity=args.regularity, edge_looseness=args.edge_looseness,
            angle_offset=np.deg2rad(args.angle_offset), depth_axis=args.depth_axis, seed=args.seed)
        print(f"[field]  {field['n_elements']} elements, {field['elements_per_tile']:.1f} per tile  "
              f"| length min/median/max {field['length_min']:.3g}/{field['length_median']:.3g}/"
              f"{field['length_max']:.3g}  (spread {field['length_spread']:.1f}x)  | axis p95 {field['neighbour_angle_p95_deg']:.1f} deg"
              f"  | opposed neighbour pairs {field['opposed_neighbours']}")
        if field["n_ground"]:
            print(f"[ground] {field['n_ground']} fine elements at {args.ground_scale:g}x")

        # Fit the page to the SHEET, not to the elements: those deliberately overhang, and the
        # overhang is what removes the bare border.
        matrix, offset = fit_to_page_fn(field["sheet_bounds"], args.fit_mm, margin_mm=0.0)
        placed = transform_elements_fn(placed, scale=float(matrix[0, 0]), offset=offset)

        specs = element_specs_fn(args)
        pale = ({"fill": "#F2EFE6", "stroke": "#141414",
                 "stroke-width": f"{args.web_mm:g}", "stroke-linejoin": "round"}
                if args.web_mm > 0.0 else {"fill": "#F2EFE6", "stroke": "none"})
        dark = {"fill": "#141414", "stroke": "none"}
        # Only the silhouette carries the web; the nested bands are pure fill, or every edge would
        # be drawn twice and the pattern would thicken as rings are added.
        styling = {"pale": pale, "dark": dark}
        elements = []
        for element in placed:
            layers = element_layers_fn(element.leaf, specs[element.population])
            for depth, (leaf, tag) in enumerate(layers):
                style = styling[tag] if depth == 0 else {**styling[tag], "stroke": "none"}
                elements.append((leaf_path_fn(leaf), style))

        out_svg = args.out or "data/outputs/katagami/scales.svg"
        os.makedirs(os.path.dirname(os.path.abspath(out_svg)), exist_ok=True)
        report = write_svg_fn(
            out_svg, {"scales": elements},
            width_mm=args.fit_mm[0], height_mm=args.fit_mm[1], background="#141414",
            styles={"scales": pale},
            precision=args.precision, title="katagami scale field")
        print(f"[svg]    {report['path']}  {report['n_paths']} elements  "
              f"{report['n_arc']} exact arcs, {report['n_quad']} quadratics  "
              f"page {report['page_mm'][0]:g}x{report['page_mm'][1]:g} mm")
        page = (0.0, 0.0, args.fit_mm[0], args.fit_mm[1])
        if args.cut_svg:
            paper, pieces = resolve_paper_fn(placed, specs, page=page)
            paper, dust = prune_dust_fn(paper, min_area=args.min_piece_mm2)
            stencil = measure_stencil_fn(paper, pieces, page=page, grid=grid,
                                         tile_of=[e.tile for e in placed])
            print(f"[cut]    {stencil['n_components']} pieces, largest holds "
                  f"{100 * stencil['largest_share']:.2f}%  | {stencil['n_holes']} holes, "
                  f"{stencil['cut_length_mm'] / 1000:.1f} m of cut  | ink {100 * stencil['coverage']:.0f}%")
            print(f"[cut]    narrowest bridge {stencil['min_bridge_mm']:.2f} mm "
                  f"({1e3 * stencil['min_bridge_per_width']:.2f} per m of sheet width)  | contacts "
                  f"{stencil['degree_min']}/{stencil['degree_median']:.0f}/{stencil['degree_max']} "
                  f"per element, {100 * stencil.get('lattice_incidence', 0.0):.0f}% on the lattice"
                  f"  | {dust['dust_dropped']} fragments dropped")
            report = write_svg_fn(
                args.cut_svg, {"paper": cut_paths_fn(paper)},
                width_mm=args.fit_mm[0], height_mm=args.fit_mm[1], background=None,
                styles={"paper": {"fill": "#141414", "fill-rule": "evenodd", "stroke": "none"}},
                precision=args.precision, title="katagami stencil, paper region")
            print(f"[cut]    {report['path']}  {report['n_paths']} components")
        if args.preview:
            png = os.path.splitext(out_svg)[0] + "_preview.png"
            render_scale_preview_fn(placed, png, specs=specs, web_mm=args.web_mm, crop=page)
            print(f"[png]    {png}")
        return

    filled = args.motif == "vine"
    if filled:
        vine_paths, vine = build_vine_fn(
            tiles, width_frac=args.vine_width, swell=args.vine_swell,
            taper=args.vine_taper, handle=args.vine_handle)
        layers = {"motif": vine_paths, "nodes": [], "tiles": [
            warp_polyline_fn(quad, np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
                             closed=True) for quad in tiles]}
        print(f"[chains] {vine['n_chains']} chains ({vine['n_closed_chains']} closed)  "
              f"cells min/median/max {vine['chain_cells_min']}/"
              f"{vine['chain_cells_median']:.0f}/{vine['chain_cells_max']}  "
              f"| {vine['loose_ends']} loose ends in the whole sheet")
        print(f"[motif]  vine  {len(vine_paths)} filled ribbons")
    else:
        layers = compose_katagami_fn(
            tiles,
            motif_name=args.motif,
            node_name=None if args.node.lower() == "none" else args.node,
            node_scale=args.node_scale,
            lod=not args.no_lod,
            motif_params=_parse_overrides(args.motif_param),
            node_params=_parse_overrides(args.node_param),
        )
        print(f"[motif]  {args.motif}  {len(layers['motif'])} paths  | node {args.node} "
              f"{len(layers['nodes'])} paths  | lod {'off' if args.no_lod else 'on'}")

    matrix, offset = fit_to_page_fn(layer_bounds_fn(layers), args.fit_mm,
                                    margin_mm=args.margin_mm)
    layers = transform_layers_fn(layers, matrix, offset)

    if args.out:
        out_svg = args.out
    elif args.run_dir:
        out_svg = os.path.join(args.run_dir, f"katagami_{args.motif}.svg")
    elif args.design_npz:
        out_svg = os.path.join(os.path.dirname(args.design_npz), f"katagami_{args.motif}.svg")
    else:
        out_svg = f"data/outputs/katagami_{args.motif}_{args.synthetic[0]}x{args.synthetic[1]}.svg"
    os.makedirs(os.path.dirname(os.path.abspath(out_svg)), exist_ok=True)

    emit = {"motif": layers["motif"], "nodes": layers["nodes"]}
    if args.show_tiles:
        emit = {"tiles": layers["tiles"], **emit}
    styles = {
        "tiles": {"stroke": figstyle.GREY, "stroke-width": f"{0.3 * args.stroke_mm:g}",
                  "stroke-opacity": "0.35"},
        "motif": ({"fill": figstyle.INK, "fill-rule": "evenodd", "stroke": "none"} if filled
                  else {"stroke": figstyle.INK, "stroke-width": f"{args.stroke_mm:g}"}),
        "nodes": {"stroke": figstyle.INK, "stroke-width": f"{args.stroke_mm:g}"},
    }
    report = write_svg_fn(
        out_svg, emit, width_mm=args.fit_mm[0], height_mm=args.fit_mm[1], styles=styles,
        background=args.background, curve=args.curve, precision=args.precision,
        title=f"katagami {args.motif} on {meta['n_tiles']} tiles",
    )
    print(f"[svg]    {report['path']}  {report['n_paths']} paths  "
          f"{report['n_line']} lines + {report['n_quad']} quadratics  "
          f"page {report['page_mm'][0]:g}x{report['page_mm'][1]:g} mm")

    if args.preview:
        png = os.path.splitext(out_svg)[0] + "_preview.png"
        render_preview_fn(layers, png, stroke_mm=args.stroke_mm, show_tiles=args.show_tiles,
                          background=args.background or "white", filled=filled)
        print(f"[png]    {png}")


if __name__ == "__main__":
    main()
