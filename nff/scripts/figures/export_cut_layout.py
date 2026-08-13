"""Workshop cutting package for a closed run: dimensioned drawing + coordinate tables + scaled DXF.

The DXF exporter (``export_cut_dxf.py``) answers "what does the machine cut". This one answers
"where do I put the tape measure", for a sheet laid out and cut by hand: a datum corner, a numbered
marking point at every cut crossing with both ordinates and two tape distances, a cut schedule with
the ligament setbacks, and a hole schedule.

The pattern may be scaled to stock of a different size than the design was optimised at. The lattice,
the ligaments and the fillets all scale together, so the flat pattern stays exactly the designed
pattern and the deployed SHAPE is unchanged. The sheet thickness does not scale, so this is not a
similarity scaling: the run's forces and damage margins belong to the design size only.

Run:
    JAX_PLATFORMS=cpu conda run -n kgnn_mac python nff/scripts/figures/export_cut_layout.py \
        --run-dir data/outputs/runs/run_<ts>_<config> [--stock-mm 1219.2 2438.4]
"""

import os
import pickle
import argparse
import datetime

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np

from nff.config.experiment import load_and_parse_config
from nff.closed.cut_export import reconstruct_run_geometry
from nff.closed.cut_layout import build_layout_schedule, write_layout_csv, verify_schedule
from nff.closed.cut_layout_render import render_overview_a3, render_tables_a4, render_detail_a4
from nff.topology.cut_dxf import export_cut_geometry_dxf
from nff.topology.cut_pattern import measure_cut_geometry


def _centre_faces(config):
    """(hold_face, pull_face): the middle tile of the bottom row and of the top row.

    Face index is ``column * N + row``, so column ``c`` runs from ``c*N`` (bottom) to ``c*N + N-1``
    (top). The middle column of an odd grid is ``M // 2``.
    """
    M, N = int(config.topology['M']), int(config.topology['N'])
    c = M // 2
    return c * N, c * N + (N - 1)


def _grip_matches_design(config, hold, pull):
    """Whether the marked HOLD/PULL tiles ARE the boundary condition the design was optimised under.

    A centred single-tile config already clamps and pulls exactly one tile, so the hand grip and the
    simulated grip coincide and the deployed shape is the one that was designed. A distributed
    config does not, and the drawing has to say so.
    """
    clamped = config.topology.get('bc_clamped', None)
    if not isinstance(clamped, list):
        return False
    loaded = set()
    for s in config.topology.get('loads', []) or []:
        f = s.get('face')
        loaded.update(f if isinstance(f, (list, tuple)) else [f])
    return set(int(c) for c in clamped) == {int(hold)} and loaded == {int(pull)}


def _resolve_scale(config, stock_mm):
    """Uniform pattern scale that lands the design on the requested stock, or 1.0 if unset.

    Both stock dimensions must agree on the same factor: a stock of a different aspect ratio cannot
    hold this pattern without distorting it, and a distorted kirigami pattern is a different design.
    """
    if stock_mm is None:
        return 1.0, None
    spacing = float(config.topology.get('spacing', 1.0))
    spacing_y = float(config.topology.get('spacing_y', spacing))
    M, N = int(config.topology['M']), int(config.topology['N'])
    sheet_w = float(config.topology.get('sheet_width_mm', 0.0) or 0.0)
    if sheet_w <= 0.0:
        raise SystemExit("--stock-mm needs the config to declare sheet_width_mm")
    design_w = sheet_w
    design_h = sheet_w * (N * spacing_y) / (M * spacing)
    sx, sy = stock_mm[0] / design_w, stock_mm[1] / design_h
    if abs(sx - sy) > 1e-6 * max(sx, sy):
        raise SystemExit(
            f"stock {stock_mm[0]:.1f} x {stock_mm[1]:.1f} mm has aspect "
            f"{stock_mm[0] / stock_mm[1]:.4f}, the design has {design_w / design_h:.4f}. "
            f"Scaling to it would distort the pattern (x{sx:.4f} vs x{sy:.4f}).")
    return sx, (design_w, design_h)


def _readme(path, *, schedule, meta, args, scale, design_size, length_scale, run_label):
    W, H = schedule["stock"]
    sm = schedule["summary"]
    act = {a["role"]: a for a in schedule.get("activation", [])}
    lines = [
        f"# Cutting package — {run_label}",
        "",
        f"Generated {meta['date']}. All dimensions in millimetres.",
        "",
        "## What you are cutting",
        "",
        f"- Stock: **{W:.1f} x {H:.1f} mm**, {meta['material']}.",
        f"- Panel pitch: **{meta['pitch']}**.",
        f"- {sm['n_cuts']} cuts, {sm['n_points']} marking points, {sm['n_holes']} relief holes.",
        f"- Total cut length {sm['cut_len_total'] / 1000.0:.2f} m.",
        f"- Ligaments {sm['w_lig_min']:.1f}-{sm['w_lig_max']:.1f} mm.",
        (f"- Relief holes: **{len(sm['hole_drills'])} different drills** "
         f"({', '.join(f'Ø{d:.1f}' for d in sm['hole_drills'])}). The hole tracks its own "
         f"ligament, so a single drill sized to the average would over-cut the narrowest hinge "
         f"-- the one with least material to spare. Take each size from `holes.csv`."
         if len(sm['hole_drills']) > 1 else
         f"- Relief holes: one Ø{sm['hole_drills'][0]:.1f} mm drill covers every one."),
        "",
    ]
    if abs(scale - 1.0) > 1e-9:
        lines += [
            "## The pattern is scaled, and what that costs",
            "",
            f"The design was optimised on a {design_size[0]:.1f} x {design_size[1]:.1f} mm sheet. "
            f"Here it is scaled by **{scale:.4f}** to fit your stock.",
            "",
            "- **The flat pattern and the deployed shape are exact.** Lattice, ligaments and "
            "fillets all scale together, and deployment is a kinematic mechanism, so the sheet "
            "opens to the same shape at the same scale factor.",
            "- **The forces and the damage margin do not transfer.** The stock is still 0.5 mm "
            "thick while the in-plane geometry shrank, so this is not a similarity scaling: "
            f"t/w_lig moves and the ligament strain at a given hinge rotation rises by about "
            f"1/{scale:.2f} = {1 / scale:.2f}x. Any calibrated grip force belongs to the design "
            "size only.",
            "- Getting honest forces for this stock means re-running the optimisation at the "
            f"scaled pitch ({length_scale:.2f} mm/unit) -- a new design, not a redrawn one.",
            "",
        ]
    if act:
        lines += [
            "## How to activate it",
            "",
            "Two tiles only:",
            "",
            f"- **HOLD** tile {act['HOLD']['face']} at ({act['HOLD']['x']:.0f}, "
            f"{act['HOLD']['y']:.0f}) -- middle tile of the bottom row."
            if "HOLD" in act else "",
            f"- **PULL** tile {act['PULL']['face']} at ({act['PULL']['x']:.0f}, "
            f"{act['PULL']['y']:.0f}) -- middle tile of the top row."
            if "PULL" in act else "",
            "",
            "Pull along the long axis of the sheet. Nothing else is gripped, and nothing else is "
            "restrained.",
            "",
            ("> This IS the boundary condition the design was optimised under: the config clamps "
             "exactly this tile and pulls exactly that one, with no other restraint. The sheet "
             "you deploy by hand is the sheet that was simulated."
             if meta.get("grip_is_design") else
             "> This two-tile grip is the physical test protocol, not the boundary condition the "
             "design was optimised under. The optimisation used the config's own clamp and load "
             "set, so the deployed shape under a two-tile pinch will not match the simulated one "
             "tile for tile."),
            "",
        ]
    lines += [
        "## Order of work",
        "",
        "1. **Square the stock.** Measure the diagonal: it must be "
        f"{schedule['checks'][0]['dist']:.1f} mm. If it is not, trim the sheet before anything "
        "else -- every coordinate below is measured from its corner.",
        "2. **Mark the datum** at the lower-left corner with the short edge horizontal. "
        "The datum is fixed and cannot be moved to another corner: the pattern is chiral, and a "
        "mirrored layout deploys to the mirrored shape.",
        "3. **Mark every point** from `points.csv` / the POINTS table. Use the x/y ordinates, then "
        "confirm each with the `from A` and `from B` tape distances. Points flagged `on edge` sit "
        "on the factory edge at the given distance and need only one measurement.",
        "4. **Check the layout** against the PRE-CUT CHECKS panel before any blade touches the "
        "sheet.",
        "5. **Scribe the construction lines** point to point. Most lines also run through further "
        "marks, listed in the `through` column: the line has to hit every one of them. A line that "
        "misses a mark indicts the mark, not the line, so fix it before cutting.",
        "6. **Drill the relief holes** at the coordinates in `holes.csv`.",
        "7. **Cut hole to hole.** Never cut past a hole: the uncut ligament between two holes is "
        "the hinge, and cutting through one is not recoverable.",
        "",
        "## Files",
        "",
        "| File | What it is |",
        "|---|---|",
        "| `cut_layout_A3.pdf` | the drawing, 1:%g, with the datum, point IDs and grip marks |"
        % args.drawing_scale,
        "| `cut_layout_tables_A4.pdf` | points, cuts and holes, with tick boxes |",
        "| `hinge_detail_1to1_A4.pdf` | hinge detail, 1:1 gauge, print calibration bar |",
        "| `points.csv` `cuts.csv` `holes.csv` | the same tables, machine readable |",
        "| `cut_layout.dxf` | 1:1 CAD of this exact scaled pattern |",
        "",
        "Print the detail sheet at **Actual size / 100%**, never *fit to page*, and measure the "
        "150 mm bar before trusting the gauge.",
        "",
    ]
    with open(path, "w") as f:
        f.write("\n".join(l for l in lines if l is not None) + "\n")
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--run-dir", help="run folder with config.yaml + best_params.pkl (trained design)")
    src.add_argument("--config-name", help="closed config name -> untrained flat design")
    ap.add_argument("--config-dir", default="closed")
    ap.add_argument("--stock-mm", type=float, nargs=2, default=[1219.2, 2438.4],
                    metavar=("W", "H"),
                    help="stock sheet size [mm]; the pattern is scaled uniformly onto it "
                         "(default: a 4ft x 8ft sheet). Pass 0 0 to keep the design size.")
    ap.add_argument("--drawing-scale", type=float, default=8.0,
                    help="N in 1:N for the A3 overview (default 8)")
    ap.add_argument("--hole-dia-mm", type=float, default=None,
                    help="single nominal relief-hole drill [mm] (default: nearest preferred size)")
    ap.add_argument("--hold-face", type=int, default=None, help="tile to hold (default: middle of the bottom row)")
    ap.add_argument("--pull-face", type=int, default=None, help="tile to pull (default: middle of the top row)")
    ap.add_argument("--kerf", type=float, default=0.2,
                    help="kerf [mm]; affects ONLY the companion DXF and the width of the slots "
                         "drawn on the detail sheet. No tabulated number depends on it, because a "
                         "hand cut is a slit on the line. Default is the simulated 0.2; use 0.5 if "
                         "the DXF is going to a laser.")
    ap.add_argument("--no-grid", dest="grid", action="store_false", help="drop the reference grid")
    ap.add_argument("--no-point-labels", dest="point_labels", action="store_false")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    if args.run_dir:
        config = load_and_parse_config(os.path.join(args.run_dir, "config.yaml"))
        with open(os.path.join(args.run_dir, "best_params.pkl"), "rb") as f:
            params_override = pickle.load(f)
        run_label = os.path.basename(os.path.normpath(args.run_dir))
        out_dir = args.out_dir or os.path.join(args.run_dir, "cut_layout")
    else:
        config = load_and_parse_config(f"data/configs/{args.config_dir}/{args.config_name}.yaml")
        params_override = None
        run_label = f"{args.config_name} (untrained)"
        out_dir = args.out_dir or f"data/outputs/{args.config_name}_cut_layout"
    os.makedirs(out_dir, exist_ok=True)

    stock = None if (args.stock_mm[0] <= 0 or args.stock_mm[1] <= 0) else tuple(args.stock_mm)
    scale, design_size = _resolve_scale(config, stock)

    geom, w_lig, length_scale, state = reconstruct_run_geometry(
        config, params_override, w_c_mm=args.kerf, geom_scale=scale)

    # ── the geometry must be what it claims before anything is drawn from it ──
    rt = measure_cut_geometry(geom)
    assert rt["max_w_lig_err"] < 1e-6, f"ligament round-trip failed: {rt['max_w_lig_err']:.2e} mm"
    minx, miny, maxx, maxy = geom["sheet"].bounds
    got_w, got_h = maxx - minx, maxy - miny
    if stock is not None:
        assert abs(got_w - stock[0]) < 1e-6 and abs(got_h - stock[1]) < 1e-6, (
            f"built sheet {got_w:.4f} x {got_h:.4f} mm != stock {stock[0]} x {stock[1]}")

    hold = _centre_faces(config)[0] if args.hold_face is None else args.hold_face
    pull = _centre_faces(config)[1] if args.pull_face is None else args.pull_face
    face_mm = np.asarray(state.face_centroids) * length_scale

    schedule = build_layout_schedule(geom, hole_dia_mm=args.hole_dia_mm, face_centroids_mm=face_mm,
                                     hold_faces=[hold], pull_faces=[pull])
    worst = verify_schedule(schedule, geom)
    sm = schedule["summary"]

    pitch_mm = length_scale * float(config.topology.get('spacing', 1.0))
    material = (f"{getattr(config.hinge_model, 'material', 'PET')}  "
                f"t={getattr(config.hinge_model, 'thickness_mm', 0.5)} mm")
    caveat = []
    if abs(scale - 1.0) > 1e-9:
        caveat = [f"Pattern scaled x{scale:.4f} from the",
                  f"{design_size[0]:.0f} x {design_size[1]:.0f} mm design sheet.",
                  "Flat pattern and deployed SHAPE are",
                  "exact. Forces and damage margins are",
                  "NOT: the stock thickness did not scale,",
                  f"so hinge strain rises about {1 / scale:.2f}x.",
                  "Do not quote simulated loads for this",
                  "sheet."]
    grip_is_design = _grip_matches_design(config, hold, pull)
    grip_note = (["This IS the boundary condition the",
                  "design was optimised under: the",
                  "sheet you deploy is the sheet that",
                  "was simulated."] if grip_is_design else
                 ["This is the TEST protocol, not the",
                  "boundary condition the design was",
                  "optimised under."])
    meta = dict(title="KIRIGAMI CUT LAYOUT",
                subtitle=f"{run_label}   ·   hand layout, knife + drilled relief holes",
                material=material, pitch=f"{pitch_mm:.2f} mm",
                source=(args.config_name or run_label.split("_", 3)[-1])[:26],
                date=datetime.date.today().isoformat(), caveat=caveat,
                grip_is_design=grip_is_design, grip_note=grip_note)

    a3 = render_overview_a3(schedule, geom, os.path.join(out_dir, "cut_layout_A3.pdf"),
                            draw_scale=args.drawing_scale, meta=meta,
                            grid_mm=100.0 if args.grid else 1e9,
                            label_points=args.point_labels)
    tab = render_tables_a4(schedule, os.path.join(out_dir, "cut_layout_tables_A4.pdf"), meta=meta)
    det = render_detail_a4(schedule, geom, os.path.join(out_dir, "hinge_detail_1to1_A4.pdf"),
                           meta=meta)
    csvs = write_layout_csv(schedule, out_dir)
    dxf = export_cut_geometry_dxf(geom, os.path.join(out_dir, "cut_layout.dxf"))
    rdm = _readme(os.path.join(out_dir, "README.md"), schedule=schedule, meta=meta, args=args,
                  scale=scale, design_size=design_size, length_scale=length_scale,
                  run_label=run_label)

    print(f"  design      : {run_label}")
    print(f"  scale       : x{scale:.4f}  ->  pitch {pitch_mm:.2f} mm, "
          f"length_scale {length_scale:.2f} mm/unit")
    print(f"  stock       : {got_w:.1f} x {got_h:.1f} mm   ({material})")
    print(f"  geometry    : {len(geom['hinge_info'])} hinges, w_lig "
          f"{w_lig.min():.2f}-{w_lig.max():.2f} mm, round-trip {rt['max_w_lig_err']:.1e} mm")
    print(f"  schedule    : {sm['n_points']} points, {sm['n_cuts']} cuts "
          f"({sm['n_perimeter']} perimeter segments excluded), {sm['n_holes']} holes, "
          f"{sm['cut_len_total'] / 1000.0:.2f} m of cutting")
    print(f"  table check : reproduces geometry to {worst:.1e} mm; "
          f"marks lie on their lines to {sm['max_collinearity_dev']:.1e} mm")
    print(f"  activation  : HOLD tile {hold}, PULL tile {pull}")
    print(f"  drill       : {sm['hole_drill_label']} "
          f"({', '.join(f'{d:.1f}' for d in sm['hole_drills'])}; "
          f"exact Ø{sm['hole_dia_min']:.2f}-{sm['hole_dia_max']:.2f})")
    for p in [a3, tab, det, *csvs, dxf["path"], rdm]:
        print(f"    -> {p}")


if __name__ == "__main__":
    main()
