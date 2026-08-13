"""SUPERSEDED by `validity_ladder.py` -- kept only to reproduce the 5x5 six-rung numbers.

⚠ Do not run this for new work. `nff/scripts/diagnostics/validity_ladder.py` is the live driver:
it takes `--config`, runs the four rungs the figure now draws, splits voids from overhang, and
captures the deployment. This file is retained because it is the only record of how the 5x5
six-rung result was produced -- in particular the soft-penalty rung ("+ hinge-gap penalty",
worst gap 213 mm -> 11.6 mm at weight 1e3), which the four-rung ladder no longer runs.

The validity ladder on a square 5x5 sheet: five mechanisms, one task, six seeds each.

The task is identical for every rung -- match the target, subject to the flat sheet being a perfect
closed cut pattern. Only the mechanism that is supposed to deliver the constraint changes:

    1  free vertices, nothing enforced
    2  free vertices + a hinge-gap PENALTY                     (scored)
    3  free vertices + a Stage-1 PROJECTION                    (repaired)
    4  free vertices + projection + a non-overlap penalty
    5  free vertices + projection + non-overlap + tiling penalty
    6  the closed chart                                        (structural)

Each rung is expected to fix the previous defect and expose the next, so what is recorded per run is
not a scalar but the GEOMETRY of the failure: per-hinge gaps, the polygons where panels overlap, and
the polygons where the sheet fails to partition the stock. The figure needs those to draw a magnified
callout at the worst defect, which is the only way a sub-millimetre failure on a 1219 mm sheet can be
seen at all.

⚠ A run that never improves on its starting point is DISCARDED, not averaged in. The starting point
is the closed chart's own geometry, which is exactly closed and exactly tiled, so a stalled run
reports a flawless sheet it did no work to produce. That artefact has already been mistaken for a
result twice in this project.

    JAX_PLATFORMS=cpu conda run -n kgnn_mac python nff/scripts/diagnostics/ladder_5x5.py
"""

import argparse
import math
import os
import pickle
import time

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
from copy import deepcopy
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from nff.closed.chart_ablation import (arm_extra_loss, arm_training_cfg, arm_validity_cfg,
                                       closed_start_geometry, fixed_target, geom_coords,
                                       geom_shape, hinge_gap_operator, init_direct_from_closed,
                                       project_if_needed)
from nff.config.experiment import load_and_parse_config
from nff.stages.mapping import apply_closed_les_mapping, apply_direct_mapping
from nff.training.trainer import TrainState, create_train_step

# (key, label, validity arm, hinge-gap weight, overlap weight, tiling weight, epoch budget)
#
# ⚠ WHY THE EPOCH BUDGETS DIFFER. Measured per training step on this problem: a rung WITHOUT the
# Stage-1 projection costs 21.4 s, one WITH it costs 0.44 s -- 48x, in favour of the rung carrying
# MORE machinery. Torn geometry makes the Stage-2 solve badly conditioned, so L-BFGS runs to
# `solver_maxiter` on every one of the 24 load steps. The un-projected rungs are therefore given a
# shorter budget, which costs nothing that matters: their defect is STRUCTURAL and present from the
# first epoch (a torn sheet cannot be un-torn by a penalty), and their physics is meaningless
# regardless -- `build_reference_bond_vectors` takes the bond vector from the design itself, so a
# 200 mm hinge gap becomes a 200 mm spring at ZERO strain energy. The rungs whose numbers carry the
# argument -- projection onward, where the sheet is genuinely closed -- all get the full budget.
RUNGS = [
    ('naive',      'free vertices',                    'B', 0.0,   0.0,   0.0,  150),
    ('scored',     '+ hinge-gap penalty',              'C', 1.0e3, 0.0,   0.0,  150),
    ('projected',  '+ projection',                     'D', 0.0,   0.0,   0.0, 1000),
    ('nonoverlap', '+ projection + non-overlap',       'E', 0.0,   1.0e3, 0.0, 1000),
    ('tiled',      '+ projection + overlap + tiling',  'E', 0.0,   1.0e3, 1.0e3, 1000),
    ('chart',      'closed chart',                     'A', 0.0,   0.0,   0.0, 1000),
]


def defect_geometry(y, state, B, stock, scale):
    """Everything the figure needs to draw this sheet's failure, in physical units."""
    F, K = geom_shape(state)
    n = np.asarray(y).reshape(F, K, 2)
    pairs = np.asarray(state.hinge_node_pairs)

    gaps, seg = [], []
    for (fi, li), (fk, lk) in pairs:
        p, q = n[int(fi), int(li)], n[int(fk), int(lk)]
        gaps.append(np.linalg.norm(p - q) * scale)
        seg.append((p, q))
    gaps = np.array(gaps)

    ps = [Polygon(v) for v in n]
    ps = [v if v.is_valid else v.buffer(0) for v in ps]
    u = unary_union(ps)
    total = sum(v.area for v in ps)
    overlap_frac = max((total - u.area) / max(total, 1e-12), 0.0)
    # where panels lie on top of one another
    ov_polys = []
    for i in range(len(ps)):
        for j in range(i + 1, len(ps)):
            inter = ps[i].intersection(ps[j])
            if not inter.is_empty and inter.area > 1e-14:
                ov_polys.append(inter)
    # where the sheet fails to partition the stock, both directions
    mismatch = [stock.difference(u), u.difference(stock)]
    tiling = 1.0 - u.intersection(stock).area / max(u.union(stock).area, 1e-12)

    def coords(g):
        out = []
        for piece in (g.geoms if hasattr(g, 'geoms') else [g]):
            if not piece.is_empty and piece.area > 1e-14:
                out.append(np.array(piece.exterior.coords))
        return out

    return dict(nodes=n, gaps=gaps, hinge_seg=np.array(seg),
                overlap_frac=float(overlap_frac),
                overlap_polys=[np.array(p.exterior.coords) for p in ov_polys if p.geom_type == 'Polygon'],
                mismatch_polys=[c for g in mismatch for c in coords(g)],
                tiling=float(tiling),
                area=float(total), stock_area=float(stock.area))


def run_one(cfg0, rung, seed, epochs):
    key, label, arm, w_gap, w_ovl, w_til = rung[:6]
    cfg = deepcopy(cfg0)
    cfg.topology = {**cfg.topology, 'init_seed': seed, 'init_noise': 0.5}
    target, cloud = fixed_target(cfg)

    if arm == 'A':
        _, params, sf, state, _ = closed_start_geometry(cfg, seed)
        map_type = 'closed_les'
    else:
        params, state, _ = init_direct_from_closed(cfg, seed)
        sf, map_type = None, 'direct_vertices'
    tcfg = arm_training_cfg(cfg, hinge_gap=w_gap)
    cfg = arm_validity_cfg(cfg, arm)
    xfn = arm_extra_loss(cfg, arm, state, w_overlap=w_ovl, w_tiling=w_til) if arm == 'E' else None

    opt, step = create_train_step(state, target, cfg.validity, cfg.physics, tcfg,
                                  map_type=map_type, use_jit=True,
                                  load_specs=cfg.topology.get('loads', []),
                                  static_features=sf, target_cloud=cloud, extra_loss_fn=xfn)
    ts = TrainState(params=params, opt_state=opt.init(params), rng=jax.random.PRNGKey(0))
    best, L0 = (math.inf, params), None
    for _ in range(epochs):
        prev = ts.params
        ts, loss, _ = step(ts)
        L = float(loss)
        if L0 is None:
            L0 = L
        if math.isfinite(L) and L < best[0]:
            best = (L, prev)
    stalled = abs(best[0] - L0) < 1e-9

    mapped = (apply_closed_les_mapping(state, best[1], sf) if arm == 'A'
              else apply_direct_mapping(state, best[1]))
    y = np.asarray(geom_coords(project_if_needed(mapped, cfg)))
    jax.clear_caches()
    return y, best[0], stalled


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="closed/sheet_5x5_square")
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=0,
                    help="cap on every rung's own budget; 0 = use each rung's budget")
    ap.add_argument("--out", default="data/outputs/ladder_5x5/ladder.pkl")
    args = ap.parse_args()

    cfg0 = load_and_parse_config(f"data/configs/{args.config}.yaml")
    y0, _, _, S0, _ = closed_start_geometry(cfg0)
    F, K = geom_shape(S0)
    B = hinge_gap_operator(S0)
    scale = float(cfg0.hinge_model.length_scale)
    n0 = np.asarray(y0).reshape(F, K, 2)
    stock = box(n0[:, :, 0].min(), n0[:, :, 1].min(), n0[:, :, 0].max(), n0[:, :, 1].max())

    # True interior cut length, for converting an area deficit into an equivalent uniform gap.
    per = sum(Polygon(v).length for v in n0) * scale
    cut_len = (per - stock.length * scale) / 2.0
    out_cut = cut_len
    print(f"  square 5x5: {F} panels, {B.shape[0]//2} hinges, {F*K*2} geometric DOF")
    print(f"  interior cut length {cut_len:.0f} mm")
    print(f"  stock {(n0[:,:,0].max()-n0[:,:,0].min())*scale:.0f} x "
          f"{(n0[:,:,1].max()-n0[:,:,1].min())*scale:.0f} mm\n")
    print(f"  {'rung':<34}{'seeds':>7}{'gap mm':>10}{'overlap':>10}{'tiling':>11}{'gap mm':>10}")

    out = {'stock': np.array(stock.exterior.coords), 'scale': scale, 'shape': (F, K),
           'cut_len_mm': None, 'stock_area_mm2': float(stock.area) * scale ** 2, 'rungs': []}
    for rung in RUNGS:
        key, label, budget = rung[0], rung[1], rung[6]
        ep = min(budget, args.epochs) if args.epochs else budget
        runs = []
        for s in range(1, args.seeds + 1):
            t0 = time.time()
            y, loss, stalled = run_one(cfg0, rung, s, ep)
            g = defect_geometry(y, S0, B, stock, scale)
            g.update(seed=s, loss=loss, stalled=stalled, epochs=ep)
            runs.append(g)
            print(f"      {label:<32} seed {s}  {ep:>4} ep  {time.time()-t0:6.0f}s   "
                  f"gap {g['gaps'].max():8.2e} mm  overlap {100*g['overlap_frac']:5.2f}%  "
                  f"tiling {g['tiling']:.3e}{'   STALLED' if stalled else ''}", flush=True)
        live = [r for r in runs if not r['stalled']]
        use = live if live else runs
        gm = np.median([r['gaps'].max() for r in use])
        ov = np.median([r['overlap_frac'] for r in use])
        ti = np.median([r['tiling'] for r in use])
        # the drawn exemplar: the seed whose tiling deficit is the median of the live runs
        med_i = int(np.argsort([r['tiling'] for r in use])[len(use) // 2])
        out['rungs'].append(dict(key=key, label=label, runs=runs, exemplar=use[med_i],
                                 n_live=len(live), n=len(runs),
                                 med_gap=gm, med_overlap=ov, med_tiling=ti))
        gap_mm = ti * stock.area * scale ** 2 / cut_len
        print(f"  {label:<34}{len(live):>3}/{len(runs):<3}{gm:10.2e}{100*ov:9.2f}%"
              f"{ti:11.3e}{gap_mm:10.2f}\n", flush=True)
        # Written after EVERY rung, so an interrupted run still yields a usable figure.
        out['cut_len_mm'] = float(cut_len)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, 'wb') as fh:
            pickle.dump(out, fh)

    out['cut_len_mm'] = float(cut_len)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'wb') as fh:
        pickle.dump(out, fh)
    print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
