"""Draw the load path of a deployed tessellation.

Tension is drawn as routes because it is path-like, compression as a shaded tile field because its
links form no chains, and shear as SLIP PLANES normal to the hinge axis -- three channels, three
representations. The figure itself lives in :mod:`nff.analysis.load_path_viz` so ``run_closed``
can emit it alongside the run's other visuals -- this is only the standalone entry point.

⚠ Hold the sheet against rotation before reading a load path from it. A single pin leaves a
zero-energy rigid rotation; the sheet then swings bodily (36 deg measured) and the geometry stops
aligning with the force, which makes the reading incoherent.

Run:
    PYTHONPATH=$(pwd) JAX_PLATFORMS=cpu conda run -n kgnn_mac \
        python nff/scripts/diagnostics/plot_load_paths.py \
            --config data/configs/closed/sheet_15x10_rom_d70_hpull.yaml
"""

import argparse
import copy
import os
from types import SimpleNamespace

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np                                                    # noqa: E402

from nff.utils.figstyle import use_agg                                # noqa: E402
use_agg()

from nff.config.experiment import load_and_parse_config               # noqa: E402
from nff.analysis.load_path_viz import plot_load_path                   # noqa: E402
from nff.analysis.load_paths import (extract_force_field, format_checks, load_field,  # noqa: E402
                                   orientation_confound, save_field, solve_deployed_state)


def _config_from_meta(stem):
    """Recover (config, tessellation) from the sidecar a saved field carries, if it is still there."""
    import json

    side = stem + '.json'
    if not os.path.exists(side):
        return None, None
    try:
        path = (json.load(open(side)).get('meta') or {}).get('config')
        if not path or not os.path.exists(path):
            return None, None
        config = copy.deepcopy(load_and_parse_config(path))
        from nff.closed.setup import build_closed_initial_state
        _, tessellation = build_closed_initial_state(config)
        return config, tessellation
    except Exception:
        return None, None                       # BCs are a bonus on a replot, never a hard failure


def deployed_tessellation(tessellation, field):
    """An UN-WELDED tessellation whose faces carry their own deployed corners.

    ⚠ ``write_deformed_into`` writes SHARED vertices, and in a deployed kirigami one shared vertex
    belongs to up to four tiles whose corners have pulled apart by a void width. Last-write-wins then
    puts three of them in the wrong place -- measured median error 360% of a tile, max 581% -- so the
    supports and the load arrow were being drawn on badly wrong geometry. Giving every face its own
    four vertices reproduces the deployed sheet exactly, and the shared BC drawing needs nothing more
    than ``.vertices`` and ``.faces[i].vertex_indices``.
    """
    if tessellation is None:
        return None
    nodes = np.asarray(field.face_nodes)
    n_faces, n_nodes, _ = nodes.shape
    faces = [SimpleNamespace(vertex_indices=np.arange(k * n_nodes, (k + 1) * n_nodes))
             for k in range(n_faces)]
    return SimpleNamespace(vertices=nodes.reshape(-1, 2), faces=faces)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None, help="config YAML; re-solves the deployment")
    p.add_argument("--npz", default=None, help="stem written by extract_load_paths.py (offline)")
    p.add_argument("--params", default=None, help="best_params.pkl; omit for the untrained design")
    p.add_argument("--no-zones", dest="zones", action="store_false",
                   help="drop the compression/shear shading, leaving the pure tension route")
    p.add_argument("--threshold", type=float, default=None,
                   help="share of F0 below which a link is faint (default: the mean)")
    p.add_argument("--steps", type=int, default=None, help="override physics.num_load_steps")
    p.add_argument("--maxiter", type=int, default=20000, help="override physics.solver_maxiter")
    p.add_argument("--out", default=None)
    p.add_argument("--label", default=None)
    args = p.parse_args()

    if not args.config and not args.npz:
        p.error("pass --config to re-solve, or --npz to replot an extracted field")

    config, tessellation = None, None
    if args.npz:
        field = load_field(args.npz)
        default_out = args.npz + ".png"
        # The saved sidecar records which config produced the field; reload it so an offline replot
        # still draws the supports and the load. Without it the figure silently omits how the sheet
        # is held, which is half of what a load path means.
        config, tessellation = _config_from_meta(args.npz)
    else:
        config = copy.deepcopy(load_and_parse_config(args.config))
        if args.steps:
            object.__setattr__(config.physics, 'num_load_steps', int(args.steps))
        if args.maxiter and args.maxiter != config.physics.solver_maxiter:
            print(f"  solver_maxiter {config.physics.solver_maxiter} -> {args.maxiter}")
            object.__setattr__(config.physics, 'solver_maxiter', int(args.maxiter))
        results, tessellation, bond_energy, hinge_geometry, cut_axis = solve_deployed_state(
            config, args.params)
        field = extract_force_field(results, config.physics, bond_energy_fn=bond_energy,
                                    hinge_geometry=hinge_geometry, cut_axis=cut_axis)
        name = os.path.splitext(os.path.basename(args.config))[0]
        default_out = os.path.join("data", "outputs", "load_paths", name, "load_paths.png")
        save_field(field, os.path.splitext(default_out)[0], meta={'config': args.config})

    out = args.out or default_out
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)

    print(format_checks(field.checks))
    written, stats = plot_load_path(
        field, deployed_tessellation(tessellation, field), out, config=config,
        threshold=args.threshold, zones=args.zones, label=args.label)

    print("  load path:")
    for key, value in stats.items():
        print(f"    {key:18s} {value if not isinstance(value, float) else f'{value:.4g}'}")
    confound = orientation_confound(field)
    if np.isfinite(confound):
        print(f"    {'orient. confound':18s} {confound:+.3f}"
              + ("   ⚠ the mode tint largely tracks HINGE ORIENTATION" if abs(confound) > 0.5
                 else ""))
    print(f"  wrote {written}")


if __name__ == "__main__":
    main()
