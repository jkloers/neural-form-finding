"""The validity ladder: four mechanisms, one task, several seeds each.

The task is identical for every rung -- match the target, subject to the flat sheet being a perfect
closed cut pattern. Only the mechanism that is supposed to deliver the constraint changes:

    1  free vertices, nothing enforced
    2  free vertices + Stage-1 PROJECTION + a non-overlap penalty
    3  ... + a tiling penalty
    4  the closed chart                                        (structural)

Each rung is expected to fix the previous defect and expose the next, so what is recorded per run is
not a scalar but the GEOMETRY of the failure: the polygons where the sheet leaves a VOID, the
polygons where panels OVERLAP, and the deployed shape against the target. The figure needs those to
draw a magnified callout at the worst defect, which is the only way a sub-millimetre failure on a
2.8 m sheet can be seen at all.

⚠ A run that never improves on its starting point is DISCARDED, not averaged in. The starting point
is the closed chart's own geometry, which is exactly closed and exactly tiled, so a stalled run
reports a flawless sheet it did no work to produce. That artefact has already been mistaken for a
result twice in this project. For the same reason every run reports how far it TRAVELLED from its
own start: a design that did not move has not been tested.

    JAX_PLATFORMS=cpu python nff/scripts/diagnostics/validity_ladder.py \\
        --config closed/sheet_7x7_circle --seeds 3
"""

import argparse
import math
import os
import pickle
import time

import equinox as eqx
import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
from copy import deepcopy
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from nff.closed.chart_ablation import (arm_extra_loss, arm_training_cfg, arm_validity_cfg,
                                       closed_start_geometry, deploy_validity, fixed_target,
                                       geom_coords, geom_shape, hinge_gap_operator,
                                       init_direct_from_closed, project_if_needed,
                                       rms_displacement)
from nff.config.experiment import load_and_parse_config
from nff.stages.mapping import apply_closed_les_mapping, apply_direct_mapping
from nff.training.trainer import TrainState, create_train_step

# (key, label, validity arm, hinge-gap weight, overlap weight, tiling weight, epoch budget)
#
# ★ EVERY RUNG GETS THE SAME EPOCH BUDGET. This is the only protocol under which the columns can be
# read side by side, and it is set by the SLOWEST rung, not the fastest: `free vertices` costs about
# 68 s per training step here against the chart's 2.1 s -- a 38x gap measured at an identical
# `solver_maxiter`, caused by the torn geometry wrecking the Stage-2 solve rather than by any
# setting. Equalising upward (1000 epochs everywhere) would mean ~19 h per seed for the free-vertex
# arm, so the budget equalises DOWNWARD to what that arm can afford.
#
# What this costs is worth stating plainly: at 150 epochs no arm has converged in SHAPE, the chart
# included, so the shape error is a fair comparison at a fixed budget and not a converged one. What
# it does not cost is the ladder's actual claim -- voids and overlap are STRUCTURAL, present from
# the first epoch, and no epoch count closes a hinge that the parameterisation cannot close.
BUDGET = 150

# ★ PER-ARM LEARNING RATE, chosen from the SAME candidate set {0.01, 0.1} evaluated on EVERY arm.
# No single rate serves all four columns, and that is itself a measurement rather than an
# inconvenience: at lr = 0.1 the closed chart is exact and reaches 8.7 mm, while `free vertices`
# explodes in 40 s to a 2.4 m hinge gap, `+ tiling penalty` STALLS at its starting point, and
# `+ overlap penalty` degrades (voids 73 -> 159 mm). At lr = 0.01 every baseline trains sensibly
# but the chart stalls at 74 mm. Forcing one rate would therefore either cripple our own column or
# sabotage the baselines; giving each arm its better rate is the conservative choice, since the
# baselines are handed their best result and we are handed ours.
# ⚠ Judge an LR on VALIDITY, not on shape error alone. lr = 0.2 gives the chart a BETTER shape
# (6.65 mm) while driving r to [0.018, 0.922] and breaking exactness (voids 0.05 mm, overlap
# 0.02%) -- the one claim the figure exists to make. See feedback_retune_lr_before_blaming_geometry.
LR_BASELINE, LR_CHART = 0.01, 0.1

# (key, label, validity arm, hinge-gap w, overlap w, tiling w, epochs, learning rate)
RUNGS = [
    ('naive',      'free vertices',                 'B', 0.0, 0.0,   0.0,   BUDGET, LR_BASELINE),
    ('nonoverlap', '+ projection, overlap penalty', 'E', 0.0, 1.0e3, 0.0,   BUDGET, LR_BASELINE),
    ('tiled',      '+ tiling penalty',              'E', 0.0, 1.0e3, 1.0e3, BUDGET, LR_BASELINE),
    ('chart',      'closed chart',                  'A', 0.0, 0.0,   0.0,   BUDGET, LR_CHART),
]



def defect_geometry(y, state, stock, scale):
    """Everything the figure needs to draw this sheet's failure, in physical units.

    The stock mismatch is split rather than pooled. ``void_polys`` is stock area NOT covered by any
    panel -- a hole in the sheet, and the only thing the figure paints red. ``overhang_polys`` is
    panel material lying outside the stock, which is still material and is drawn like any other
    panel; it is kept only so the shared frame can be sized to contain it.
    """
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

    ov_polys = []
    for i in range(len(ps)):
        for j in range(i + 1, len(ps)):
            inter = ps[i].intersection(ps[j])
            if not inter.is_empty and inter.area > 1e-14:
                ov_polys.append(inter)
    tiling = 1.0 - u.intersection(stock).area / max(u.union(stock).area, 1e-12)

    def coords(g):
        out = []
        for piece in (g.geoms if hasattr(g, 'geoms') else [g]):
            if not piece.is_empty and piece.area > 1e-14:
                out.append(np.array(piece.exterior.coords))
        return out

    return dict(nodes=n, gaps=gaps, hinge_seg=np.array(seg),
                overlap_frac=float(overlap_frac),
                overlap_polys=[np.array(p.exterior.coords) for p in ov_polys
                               if p.geom_type == 'Polygon'],
                void_polys=coords(stock.difference(u)),
                overhang_polys=coords(u.difference(stock)),
                tiling=float(tiling),
                area=float(total), stock_area=float(stock.area))


def outline_error_mm(boundary, cx, cy, r, scale, n_ring=720):
    """Symmetric distance between the deployed OUTLINE and the target circle, in mm.

    Two earlier versions of this were wrong in opposite directions, both instructively:

    * A one-sided radial RMS over the boundary VERTICES is gameable. Vertices can sit within 20 mm
      of the circle while clustering into an arc that leaves an 86 deg gap: measured 8.7 mm on a
      design whose outline enclosed only 69% of the disc, beating a genuinely round one (74 mm) by
      8x while being far worse by hull IoU (0.689 against 0.937).
    * Scoring coverage from ring points to the nearest VERTEX replaces that with a discretisation
      floor. 28 vertices on a 12.8 m circumference lie ~456 mm apart, so the term cannot go below
      ~130 mm however round the sheet is, and reported ~162 mm for outlines of IoU 0.97.

    Measuring to the outline POLYGON fixes both: clustering still opens a real gap in the polygon,
    while an evenly sampled round outline scores near zero (the inscribed-polygon sagitta, ~13 mm
    here). Vertices are ordered by angle about the centre, which is well defined because a deployed
    sheet is star-shaped about its clamped centre tile.
    """
    from shapely.geometry import LinearRing, Point

    b = np.asarray(boundary)
    c = np.array([cx, cy])
    order = np.argsort(np.arctan2(b[:, 1] - c[1], b[:, 0] - c[0]))
    ring_poly = LinearRing(b[order])

    t = np.linspace(0.0, 2.0 * np.pi, n_ring, endpoint=False)
    circle = c + r * np.stack([np.cos(t), np.sin(t)], axis=-1)
    coverage = np.array([ring_poly.distance(Point(p)) for p in circle])   # circle -> outline
    precision = np.abs(np.linalg.norm(b - c, axis=1) - r)                 # outline -> circle
    return float(max(np.sqrt((precision ** 2).mean()),
                     np.sqrt((coverage ** 2).mean())) * scale)


def run_one(cfg0, rung, seed, epochs, scale):
    key, label, arm, w_gap, w_ovl, w_til = rung[:6]
    cfg = deepcopy(cfg0)
    cfg.topology = {**cfg.topology, 'init_seed': seed, 'init_noise': 0.5}
    target, cloud = fixed_target(cfg)

    if arm == 'A':
        y_start, params, sf, state, _ = closed_start_geometry(cfg, seed)
        map_type = 'closed_les'
    else:
        params, state, _ = init_direct_from_closed(cfg, seed)
        sf, map_type = None, 'direct_vertices'
        y_start = geom_coords(apply_direct_mapping(state, params))
    tcfg = arm_training_cfg(cfg, hinge_gap=w_gap)
    tcfg = eqx.tree_at(lambda t: t.learning_rate, tcfg, float(rung[7]))
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

    F, K = geom_shape(state)
    travel = rms_displacement(y, np.asarray(y_start), F * K, scale)
    dep = deploy_validity(cfg, best[1], state, map_type, static_features=sf)
    jax.clear_caches()
    return y, best[0], stalled, travel, dep


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="closed/sheet_7x7_circle")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=0,
                    help="cap on every rung's own budget; 0 = use each rung's budget")
    ap.add_argument("--out", default="data/outputs/validity_ladder/ladder_7x7.pkl")
    ap.add_argument("--only", nargs="+", default=None,
                    help="run only these rung keys; used to gate on the chart before "
                         "committing to the full ladder")
    ap.add_argument("--append", action="store_true",
                    help="extend an existing --out instead of replacing it. Rungs already in the "
                         "file are kept; a rung recomputed now replaces its old entry. This is what "
                         "makes a single expensive rung re-runnable without redoing the others.")
    args = ap.parse_args()

    cfg0 = load_and_parse_config(f"data/configs/{args.config}.yaml")
    y0, _, _, S0, _ = closed_start_geometry(cfg0)
    F, K = geom_shape(S0)
    B = hinge_gap_operator(S0)
    scale = float(cfg0.hinge_model.length_scale)
    n0 = np.asarray(y0).reshape(F, K, 2)
    stock = box(n0[:, :, 0].min(), n0[:, :, 1].min(), n0[:, :, 0].max(), n0[:, :, 1].max())

    topo = cfg0.topology
    cx, cy = float(topo['target_cx']), float(topo['target_cy'])
    r_t = float(topo['target_r'])

    # True interior cut length, for converting an area deficit into an equivalent uniform gap.
    per = sum(Polygon(v).length for v in n0) * scale
    cut_len = (per - stock.length * scale) / 2.0
    print(f"  {int(round(math.sqrt(F)))}x{int(round(math.sqrt(F)))}: {F} panels, "
          f"{B.shape[0]//2} hinges, {F*K*2} geometric DOF")
    print(f"  interior cut length {cut_len:.0f} mm")
    print(f"  stock {(n0[:,:,0].max()-n0[:,:,0].min())*scale:.0f} x "
          f"{(n0[:,:,1].max()-n0[:,:,1].min())*scale:.0f} mm")
    print(f"  target circle r = {r_t*scale:.0f} mm at ({cx}, {cy})\n")

    out = {'stock': np.array(stock.exterior.coords), 'scale': scale, 'shape': (F, K),
           'cut_len_mm': float(cut_len), 'stock_area_mm2': float(stock.area) * scale ** 2,
           'target': {'cx': cx, 'cy': cy, 'r': r_t}, 'rungs': []}
    if args.append and os.path.exists(args.out):
        with open(args.out, 'rb') as fh:
            prior = pickle.load(fh)
        out['rungs'] = list(prior.get('rungs', []))
        print(f"  appending to {args.out}: kept "
              f"{[r['key'] for r in out['rungs']]}\n", flush=True)
    for rung in RUNGS:
        if args.only and rung[0] not in args.only:
            continue
        key, label, budget = rung[0], rung[1], rung[6]
        ep = min(budget, args.epochs) if args.epochs else budget
        runs = []
        for s in range(1, args.seeds + 1):
            t0 = time.time()
            y, loss, stalled, travel, dep = run_one(cfg0, rung, s, ep, scale)
            g = defect_geometry(y, S0, stock, scale)
            err = (outline_error_mm(dep['boundary'], cx, cy, r_t, scale)
                   if dep.get('finite') else float('nan'))
            g.update(seed=s, loss=loss, stalled=stalled, epochs=ep, travel_mm=travel,
                     deployed_nodes=dep.get('nodes'), deployed_boundary=dep.get('boundary'),
                     outline_err_mm=err, deployed_finite=bool(dep.get('finite')))
            runs.append(g)
            void_mm = g['tiling'] * stock.area * scale ** 2 / cut_len
            print(f"      {label:<30} seed {s}  {ep:>4} ep  {time.time()-t0:6.0f}s   "
                  f"voids {void_mm:7.2f} mm  overlap {100*g['overlap_frac']:5.2f}%  "
                  f"fit {err:7.2f} mm  moved {travel:6.1f} mm"
                  f"{'   STALLED' if stalled else ''}", flush=True)
        live = [r for r in runs if not r['stalled']]
        use = live if live else runs
        gm = np.median([r['gaps'].max() for r in use])
        ov = np.median([r['overlap_frac'] for r in use])
        ti = np.median([r['tiling'] for r in use])
        fits = [r['outline_err_mm'] for r in use if np.isfinite(r['outline_err_mm'])]
        fit = float(np.median(fits)) if fits else float('nan')
        # the drawn exemplar: the seed whose tiling deficit is the median of the live runs
        med_i = int(np.argsort([r['tiling'] for r in use])[len(use) // 2])
        entry = dict(key=key, label=label, runs=runs, exemplar=use[med_i], epochs=ep,
                     lr=float(rung[7]),
                     n_live=len(live), n=len(runs), med_gap=gm, med_overlap=ov,
                     med_tiling=ti, med_fit_mm=fit,
                     med_travel_mm=float(np.median([r['travel_mm'] for r in use])))
        out['rungs'] = [e for e in out['rungs'] if e['key'] != key] + [entry]
        print(f"  {label:<32}{len(live):>3}/{len(runs):<3} gap {gm:9.2e} mm  "
              f"overlap {100*ov:6.2f}%  voids {ti*stock.area*scale**2/cut_len:7.2f} mm  "
              f"fit {fit:7.2f} mm\n", flush=True)
        # Written after EVERY rung, so an interrupted run still yields a usable figure.
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, 'wb') as fh:
            pickle.dump(out, fh)

    print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
