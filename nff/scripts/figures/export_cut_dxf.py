"""Standalone DXF export of a closed-run cut pattern — true 1:1, units = mm, for laser cutting.

Rebuilds the exact flat cut geometry a closed run produced (learned per-hinge ``w_lig`` + fillet at
the physical scale) and writes a laser-ready DXF, without re-training. Two sources:

- ``--run-dir DIR``: reload ``config.yaml`` + ``best_params.pkl`` from a run folder -> the TRAINED
  design (faithful per-hinge ligament widths). This is the usual case.
- ``--config-name NAME``: build the UNTRAINED (initial) flat design straight from a config, for a
  pre-training look at the pattern.

Toolpath convention via ``--mode``:
- ``outline``  (default): exact kerf-slot / fillet / sheet-edge loops (what we simulated).
- ``centerline``: sheet outline + slot centrelines + hinge-tip relief circles (thin-kerf).

Run:
    JAX_PLATFORMS=cpu conda run -n kgnn_mac python nff/scripts/figures/export_cut_dxf.py \
        --run-dir data/outputs/runs/run_<ts>_<config> [--mode outline] [--out path.dxf]
"""

import os
import pickle
import argparse

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp

from nff.config.experiment import load_and_parse_config
from nff.closed.setup import (build_closed_initial_state, init_closed_les_params,
                                      build_surrogate_energy)
from nff.closed.cut_export import build_run_cut_geometry
from nff.topology.closed_builder_jax import solve_cut_vertices_jax, boundary_flat_from_logits
from nff.topology.cut_dxf import export_cut_geometry_dxf, render_dxf_png
from nff.topology.cut_pattern import measure_cut_geometry


def _reconstruct_geometry(config, params_override=None):
    """(geom, w_lig, length_scale, hinge_model) for a config's design (override params = trained)."""
    initial_state, _ = build_closed_initial_state(config)
    params, static_features = init_closed_les_params(config)
    _, _, geometry_fn, _, w_lig_logit0 = build_surrogate_energy(
        config, static_features, initial_state, params)
    if w_lig_logit0 is not None:
        params = {**params, 'w_lig_logit': w_lig_logit0}
    if params_override is not None:                     # trained design from best_params.pkl
        params = {**params, **{k: jnp.asarray(v) for k, v in params_override.items()}}

    geom_hinge = geometry_fn(params) if geometry_fn is not None else None
    hinge_w_lig = geom_hinge.w_lig if geom_hinge is not None else None

    struct, sliders = static_features['struct'], static_features['sliders']
    cut_coords = np.asarray(solve_cut_vertices_jax(
        struct, boundary_flat_from_logits(sliders, params['bnd_logits']),
        jax.nn.sigmoid(params['z'])))
    geom, w_lig, length_scale = build_run_cut_geometry(
        initial_state, cut_coords, struct, config, config.hinge_model, hinge_w_lig)
    return geom, w_lig, length_scale


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--run-dir", help="run folder with config.yaml + best_params.pkl (trained design)")
    src.add_argument("--config-name", help="closed config name -> untrained flat design")
    ap.add_argument("--config-dir", default="closed")
    ap.add_argument("--mode", choices=["outline", "centerline"], default="outline")
    ap.add_argument("--out", default=None, help="output .dxf (default: cut_pattern.dxf next to the source)")
    ap.add_argument("--preview", action="store_true", help="also write a PNG preview beside the DXF")
    args = ap.parse_args()

    if args.run_dir:
        cfg_path = os.path.join(args.run_dir, "config.yaml")
        config = load_and_parse_config(cfg_path)
        with open(os.path.join(args.run_dir, "best_params.pkl"), "rb") as f:
            params_override = pickle.load(f)
        out = args.out or os.path.join(args.run_dir, "cut_pattern.dxf")
    else:
        cfg_path = f"data/configs/{args.config_dir}/{args.config_name}.yaml"
        config = load_and_parse_config(cfg_path)
        params_override = None
        out = args.out or f"data/outputs/{args.config_name}_cut_pattern.dxf"

    geom, w_lig, length_scale = _reconstruct_geometry(config, params_override)
    rt = measure_cut_geometry(geom)
    print(f"  geometry: {len(geom['hinge_info'])} hinges, w_lig {w_lig.min():.1f}-{w_lig.max():.1f}mm, "
          f"scale {length_scale:.1f} mm/unit, round-trip err {rt.get('max_w_lig_err', 0.0):.1e}mm")

    res = export_cut_geometry_dxf(geom, out, mode=args.mode)
    sw, sh = res["sheet_mm"]
    print(f"  DXF ({args.mode}): {res['n_loops']} loops, {res['n_lines']} lines, "
          f"sheet {sw:.1f} x {sh:.1f} mm @ 1:1 (units mm) -> {out}")

    if args.preview:
        png = os.path.splitext(out)[0] + "_preview.png"
        render_dxf_png(out, png)
        print(f"  preview -> {png}")


if __name__ == "__main__":
    main()
