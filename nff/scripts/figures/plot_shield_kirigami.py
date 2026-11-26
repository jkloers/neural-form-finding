"""Inverse-design a closed A4 kirigami sheet that deploys into a logo outline.

Starts from a flat, exactly closed A4 sheet of rigid RDPQK tiles joined by perfect
pin hinges, and optimises the per-cut aspect ratios so that opening the mechanism
carries the outer silhouette onto the target. The artwork is then projected onto
the deployed tiles and pulled back onto the flat sheet, which therefore looks
scrambled until it is deployed.

Deployment is closed-form, not simulated: the mechanism is one degree of freedom,
so pin closure holds to machine precision at every iterate and the flat sheet is a
gapless zero-void tiling of the rectangle by construction.

Outputs (per run directory): the flat sheet in both colour modes, the deployed
result, an outline-versus-target overlay, a convergence curve, a deployment
animation, and a metrics JSON.

Run:
  JAX_PLATFORMS=cpu conda run -n kgnn_mac python nff/scripts/figures/plot_shield_kirigami.py \
      --svg ~/Desktop/Princeton_seal.svg --grid 28x40 --steps 3000
"""

import argparse
import datetime as _dt
import json
import os

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np

from nff.utils.figstyle import use_agg, apply_charter, ORANGE, INK, GREY, TEAL, RED

use_agg()
import matplotlib.pyplot as plt

from nff.closed.shield_design import ShieldDesignConfig, optimize_shield
from nff.closed.shield_kinematics import tile_areas, polygon_area
from nff.closed.shield_target import load_shield_outline, load_shield_raster, load_shield_sdf
from nff.closed.shield_render import (
    draw_tiles, tile_colours, render_tiles, render_texture_pullback,
    render_texture_deployed, render_deployment_strip, render_frames)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--svg", required=True, help="Target logo (SVG, or any raster PIL can read)")
    p.add_argument("--grid", default="28x40", help="Panel grid MxN; A4 is 1:1.414")
    p.add_argument("--steps", type=int, default=3000, help="Total Adam steps across the schedule")
    p.add_argument("--psi-max", type=float, default=0.75, help="Opening-angle bound in radians")
    p.add_argument("--w-quality", type=float, default=5.0,
                   help="Weight on the tile-regularity floor; 0 allows sliver tiles")
    p.add_argument("--q-min", type=float, default=0.30, help="Quad regularity floor")
    p.add_argument("--w-barrier", type=float, default=0.0,
                   help="Manufacturability barriers (min edge / cut separation); 0 = off")
    p.add_argument("--w-homog", type=float, default=0.06,
                   help="Two-sided pull toward a uniform tessellation; trades against shape fit")
    p.add_argument("--w-outside", type=float, default=20000.0,
                   help="Keeps every tile inside the target, so no tile can hide as background")
    p.add_argument("--target-points", type=int, default=480)
    p.add_argument("--raster-px", type=int, default=2000)
    p.add_argument("--frames", type=int, default=48, help="Animation frames; 0 to skip")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--pipeline-steps", type=int, default=4,
                   help="Panels in the deployment strip, closed to open")
    p.add_argument("--export-dpi", type=int, default=600,
                   help="Resolution for the two publication figures")
    p.add_argument("--paper-dir", default=None,
                   help="If set, also write the publication figures into <paper-dir>/Images")
    return p.parse_args()


def main():
    args = parse_args()
    M, N = (int(v) for v in args.grid.lower().split("x"))
    apply_charter()

    out_dir = args.out_dir or os.path.join(
        "data/outputs/shield",
        f"{_dt.datetime.now():%Y%m%d_%H%M%S}_{M}x{N}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Run directory: {out_dir}")

    target_unit = load_shield_outline(args.svg, args.target_points)
    rgb, extent = load_shield_raster(args.svg, args.raster_px)
    sdf, sdf_extent = load_shield_sdf(args.svg, 800)
    print(f"  target: {target_unit.shape[0]} outline points, "
          f"aspect {np.ptp(target_unit[:, 0]) / np.ptp(target_unit[:, 1]):.4f}")

    half = args.steps // 2
    cfg = ShieldDesignConfig(
        M=M, N=N, psi_max=args.psi_max, w_barrier=args.w_barrier,
        w_quality=args.w_quality, w_uniform=args.w_quality, q_min=args.q_min,
        w_homog=args.w_homog, w_outside=args.w_outside,
        schedule=((half, 3e-2, 0.0), (args.steps - half, 8e-3, 1.0)))
    print(f"  grid {M}x{N} = {M * N} tiles, "
          f"pitch {cfg.spacing:.2f} x {cfg.spacing_y:.2f} mm on {cfg.sheet_w:.0f}x{cfg.sheet_h:.0f} mm")

    result = optimize_shield(cfg, target_unit, seed=args.seed, log_every=max(1, args.steps // 20),
                             sdf=sdf, sdf_extent=sdf_extent)

    params, forward, plan = result["params"], result["forward"], result["plan"]
    scale = float(np.exp(params["log_scale"]))
    shift = np.asarray(params["target_shift"])
    outline = np.asarray(forward["outline"])
    offset = outline.mean(axis=0)
    flat_tiles = np.asarray(forward["flat_tiles"])
    deployed = np.asarray(forward["deployed"])
    psi = float(forward["psi"])

    colours = tile_colours(deployed, rgb, extent, target_unit, scale, shift, offset)
    target_mm = (target_unit - target_unit.mean(axis=0)) * scale + shift + offset

    np.savez_compressed(os.path.join(out_dir, "design.npz"),
                        flat_tiles=flat_tiles, deployed=deployed, outline=outline,
                        sigma=np.asarray(plan["sigma"]), psi=psi, scale=scale,
                        shift=shift, offset=offset, target_unit=target_unit,
                        target_mm=target_mm, colours=colours)

    # Primary pair: the precise artwork printed on the deployed tessellation (voids and
    # all), and the closed state reverse-engineered from it by transporting each
    # fragment back through its tile's rigid motion.
    fig, ax = plt.subplots(figsize=(7.6, 9.6))
    fig.patch.set_facecolor("white")
    render_texture_deployed(ax, deployed, rgb, extent, target_unit, scale, shift, offset,
                            edge=None)
    _save_publication(fig, out_dir, "deployed_texture", args.export_dpi, args.paper_dir,
                      "shield_frontispiece")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 8.5))
    fig.patch.set_facecolor("white")
    render_texture_pullback(ax, flat_tiles, deployed, rgb, extent, target_unit,
                            scale, shift, offset, plan["sigma"], psi, edge=None)
    fig.savefig(os.path.join(out_dir, "flat_texture_pullback.png"), dpi=240,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Secondary: flat per-tile colour, useful for reading the tessellation itself.
    render_tiles(flat_tiles, colours, os.path.join(out_dir, "flat_closed_sheet.png"))
    render_tiles(deployed, colours, os.path.join(out_dir, "deployed.png"))

    _plot_pipeline_steps(plan, flat_tiles, deployed, rgb, extent, target_unit, scale,
                         shift, offset, psi, out_dir, args.export_dpi, args.paper_dir,
                         args.pipeline_steps)

    _plot_overlay(outline, target_mm, os.path.join(out_dir, "outline_vs_target.png"))
    _plot_convergence(result["history"], os.path.join(out_dir, "convergence.png"))
    _plot_summary(flat_tiles, deployed, colours, outline, target_mm,
                  os.path.join(out_dir, "summary.png"))

    if args.frames:
        render_frames(plan, flat_tiles, psi, colours,
                      os.path.join(out_dir, "frames"), n_frames=args.frames)

    metrics = _audit(result, flat_tiles, deployed, outline, target_mm, cfg, psi, scale,
                     colours)
    with open(os.path.join(out_dir, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)

    print("\n  " + "\n  ".join(f"{k:28s} {v}" for k, v in metrics.items()))
    print(f"\nWrote {out_dir}")


def _audit(result, flat_tiles, deployed, outline, target_mm, cfg, psi, scale, colours):
    """Independent, non-differentiable verification of the two hard requirements."""
    from shapely.geometry import Point, Polygon
    from shapely.ops import unary_union
    import jax.numpy as jnp
    from nff.closed.shield_kinematics import pin_residual

    plan = result["plan"]
    rect = cfg.sheet_w * cfg.sheet_h
    flat_area = float(jnp.sum(jnp.abs(tile_areas(flat_tiles))))
    flat_union = unary_union([Polygon(flat_tiles[k]) for k in range(flat_tiles.shape[0])]).area
    dep_area = float(jnp.sum(jnp.abs(tile_areas(deployed))))
    dep_union = unary_union([Polygon(deployed[k]) for k in range(deployed.shape[0])]).area
    hull = abs(float(polygon_area(outline)))

    d2 = np.sum((outline[:, None, :] - target_mm[None, :, :]) ** 2, axis=-1)
    fwd = np.sqrt(d2.min(axis=1))
    bwd = np.sqrt(d2.min(axis=0))
    tip_t = target_mm[np.argmin(target_mm[:, 1])]
    tip_o = outline[np.argmin(outline[:, 1])]

    shield = Polygon(target_mm)
    got = Polygon(outline).buffer(0)
    iou = (shield.intersection(got).area / shield.union(got).area
           if shield.is_valid and got.is_valid else float("nan"))

    # Does the silhouette come from material, or from tiles hiding outside as background?
    # `white_tiles_pct` alone cannot answer that: tiles sitting on the book are
    # legitimately white. `spill` and `tiles_outside` are the unambiguous measures.
    material = unary_union([Polygon(deployed[k]) for k in range(deployed.shape[0])])
    spill_pct = 100 * material.difference(shield).area / material.area
    coverage_pct = 100 * material.intersection(shield).area / shield.area
    white_pct = 100 * float((np.asarray(colours) > 0.93).all(axis=1).mean())
    centroids = deployed.mean(axis=1)
    outside_tiles_pct = 100 * float(np.mean(
        [not shield.contains(Point(c)) for c in centroids]))

    m = result["metrics"]
    return {
        "grid": f"{cfg.M}x{cfg.N}", "n_tiles": int(cfg.M * cfg.N),
        "n_pins": int(plan["pin_pairs"].shape[0]),
        "psi_rad": round(psi, 6), "psi_deg": round(np.degrees(psi), 3),
        "target_scale_mm": round(scale, 3),
        "max_pin_gap_mm": float(f"{float(jnp.max(jnp.abs(pin_residual(plan, deployed)))):.3e}"),
        "flat_area_mm2": round(flat_area, 9),
        "flat_area_rel_err": float(f"{abs(flat_area - rect) / rect:.3e}"),
        "flat_union_gap_rel": float(f"{abs(flat_union - flat_area) / rect:.3e}"),
        "deployed_overlap_pct": round(100 * (dep_area - dep_union) / dep_area, 6),
        "deployed_hull_mm2": round(hull, 1),
        "area_expansion": round(hull / flat_area, 4),
        "fill_fraction_pct": round(100 * dep_union / hull, 2),
        "chamfer_rel_rms_pct": round(100 * float(m["rel_rms"]), 3),
        "outline_to_target_mean_mm": round(float(fwd.mean()), 3),
        "outline_to_target_p95_mm": round(float(np.percentile(fwd, 95)), 3),
        "outline_to_target_max_mm": round(float(fwd.max()), 3),
        "target_to_outline_mean_mm": round(float(bwd.mean()), 3),
        "bottom_tip_error_mm": round(float(np.linalg.norm(tip_o - tip_t)), 3),
        "iou": round(float(iou), 4),
        "material_spill_outside_pct": round(float(spill_pct), 4),
        "max_tile_overshoot_mm": round(float(m["max_outside_mm"]), 3),
        "white_tiles_pct": round(white_pct, 3),
        "tiles_outside_shield_pct": round(outside_tiles_pct, 3),
        "shield_coverage_pct": round(float(coverage_pct), 2),
        "tile_area_cv": round(float(m["area_cv"]), 4),
        "min_tile_area_mm2": round(float(m["min_area_mm2"]), 3),
        "min_tile_edge_mm": round(float(m["min_edge_mm"]), 3),
        "min_quad_quality": round(float(m["min_quality"]), 4),
        "mean_quad_quality": round(float(m["mean_quality"]), 4),
    }


def _save_publication(fig, out_dir, stem, dpi, paper_dir, paper_stem):
    """Write a figure at publication resolution, optionally into the paper's Images.

    PNG only. These figures are built from one clipped raster per panel -- several
    hundred of them -- so a PDF is not vector in any useful sense; it just wraps the
    same pixels and came out at 109 MB against 1.2 MB for the PNG.
    """
    written = []
    for directory, name in [(out_dir, stem)] + (
            [(os.path.join(paper_dir, "Images"), paper_stem)] if paper_dir else []):
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"{name}.png")
        fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.01, facecolor="white")
        written.append(path)
    return written


def _plot_pipeline_steps(plan, flat_tiles, deployed, rgb, extent, target_unit, scale,
                         shift, offset, psi, out_dir, dpi, paper_dir, n_steps):
    """Deployment strip from closed to open at publication resolution.

    ⚠ This writes a PNG, and the manuscript's Figure 4 is the VECTOR PDF written by
    ``render_shield_strip.py`` -- authored at ``\\textwidth`` with the tile layer flattened to one
    image per panel. A full run of this script therefore leaves a ``shield_pipeline.png`` beside a
    ``shield_pipeline.pdf`` that the paper does not use; re-run the strip script to update the
    figure itself.
    """
    fig = render_deployment_strip(plan, flat_tiles, deployed, rgb, extent, target_unit,
                                  scale, shift, offset, psi, n_steps=n_steps)
    written = _save_publication(fig, out_dir, "pipeline_steps", dpi, paper_dir,
                                "shield_pipeline")
    plt.close(fig)
    return written


def _plot_overlay(outline, target_mm, path):
    fig, ax = plt.subplots(figsize=(5.5, 7.0))
    ax.plot(*_closed(target_mm), "-", color=RED, lw=2.0, label="target outline")
    ax.plot(*_closed(outline), "-", color=TEAL, lw=1.1, label="deployed outline")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(frameon=False, loc="center")
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _closed(pts):
    return np.append(pts[:, 0], pts[0, 0]), np.append(pts[:, 1], pts[0, 1])


def _plot_convergence(history, path):
    steps = [h["step"] for h in history]
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.plot(steps, [100 * h["rel_rms"] for h in history], color=ORANGE, lw=1.8)
    ax.set_xlabel("step")
    ax.set_ylabel("Chamfer / target RMS radius  (%)")
    ax.set_yscale("log")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_summary(flat_tiles, deployed, colours, outline, target_mm, path):
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 6.6))
    fig.patch.set_facecolor("white")
    draw_tiles(axes[0], flat_tiles, colours, lw=0.2)
    axes[0].set_title("flat closed A4 sheet")
    draw_tiles(axes[1], deployed, colours, lw=0.18)
    axes[1].set_title("deployed")
    axes[2].plot(*_closed(target_mm), "-", color=RED, lw=1.8)
    axes[2].plot(*_closed(outline), "-", color=TEAL, lw=1.0)
    axes[2].set_aspect("equal")
    axes[2].axis("off")
    axes[2].set_title("outline vs target")
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
