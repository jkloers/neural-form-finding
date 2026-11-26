"""Recompute the start-offset column of an ensemble run in place.

The start geometry is a deterministic, physics-free function of ``(init_seed, init_noise)``, so
a run whose offsets were recorded against the wrong reference can be repaired without retraining.
Idempotent: rerunning it recomputes the same values.

    JAX_PLATFORMS=cpu conda run -n kgnn_mac python nff/scripts/diagnostics/fix_start_offsets.py \
        --config closed/sheet_4x8ft_rom
"""

import argparse
import os
from copy import deepcopy

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np

from nff.config.experiment import load_and_parse_config
from nff.closed.chart_ablation import (closed_start_geometry, geom_shape, rms_displacement)
from nff.closed.hinge_path_ensemble import resolve_length_scale

START_OFFSET_COL = 6      # see ensemble_charts.py 'cols'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="closed/sheet_4x8ft_rom")
    ap.add_argument("--data", default="data/outputs/chart_ablation")
    args = ap.parse_args()

    config = load_and_parse_config(f"data/configs/{args.config}.yaml")
    length_scale = resolve_length_scale(config)
    y_ref, _, _, state, _ = closed_start_geometry(config)
    n_faces, max_nodes = geom_shape(state)
    n_nodes = n_faces * max_nodes
    y_ref = np.asarray(y_ref)

    cache = {}

    def offset(seed, noise):
        key = (int(seed), float(noise))
        if key not in cache:
            cfg = deepcopy(config)
            cfg.topology = {**cfg.topology, 'init_seed': key[0], 'init_noise': key[1]}
            y, _, _, _, _ = closed_start_geometry(cfg)
            cache[key] = rms_displacement(np.asarray(y), y_ref, n_nodes, length_scale)
        return cache[key]

    for name in ("ensemble_main.npz", "ensemble_basin.npz"):
        path = os.path.join(args.data, name)
        if not os.path.exists(path):
            print(f"  skip {name}: not present")
            continue
        z = dict(np.load(path, allow_pickle=True))
        meta = z['meta'].copy()
        before = meta[:, START_OFFSET_COL].copy()
        for i in range(meta.shape[0]):
            meta[i, START_OFFSET_COL] = offset(meta[i, 1], meta[i, 2])
        z['meta'] = meta
        np.savez_compressed(path, **z)
        print(f"  {name}: {meta.shape[0]} rows, start offset "
              f"{before.min():.1f}-{before.max():.1f} -> "
              f"{meta[:, START_OFFSET_COL].min():.1f}-{meta[:, START_OFFSET_COL].max():.1f} mm")

    for (seed, noise), v in sorted(cache.items(), key=lambda kv: (kv[0][1], kv[0][0]))[:8]:
        print(f"    seed {seed:2d}  noise {noise:5}  ->  {v:7.2f} mm")


if __name__ == "__main__":
    main()
