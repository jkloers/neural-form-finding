"""Run each Stage-0 chart arm ONCE and draw what it actually produces.

The sanity check behind the chart ablation: before any condition number or ensemble statistic
means anything, each arm has to be shown doing its job on one seed -- the flat sheet it starts
from, the flat sheet it trains to, and the deployed equilibrium it is scored on.

Every panel is in millimetres, and every arm sees the same target, the same physics and the same
starting geometry; only the design parameterization differs.

    JAX_PLATFORMS=cpu conda run -n kgnn_mac python nff/scripts/figures/plot_chart_arms.py \
        --config closed/sheet_4x8ft_rom --arms A B --seed 1 --epochs 200
"""

import argparse
import math
import os
import time

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Rectangle
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from nff.closed.chart_ablation import (
    arm_extra_loss, arm_training_cfg, arm_validity_cfg, closed_start_geometry, deploy_validity,
    fixed_target, flat_validity, geom_coords, geom_shape, hinge_gap_operator,
    init_direct_from_closed, project_if_needed,
)
from nff.closed.hinge_path_ensemble import resolve_length_scale
from nff.config.experiment import load_and_parse_config
from nff.stages.mapping import apply_closed_les_mapping, apply_direct_mapping
from nff.training.trainer import TrainState, create_train_step
from nff.utils.figstyle import GREY, INK, ORANGE, RED, TEAL, apply_charter

ARM_LABEL = {'A': "closed chart  (closed_les)",
             'B': "free vertices  (direct_vertices, no Stage 1)",
             'C': "free vertices + hinge-gap penalty in the loss",
             'D': "free vertices + Stage-1 PROJECTION",
             'E': "free vertices + projection + OVERLAP PENALTY"}
HINGE_GAP_W = 1.0e3
OVERLAP_W = 1.0e3


def setup_arm(config, arm, seed, noise):
    """Starting params and everything the trainer needs, for one arm."""
    from copy import deepcopy
    cfg = deepcopy(config)
    cfg.topology = {**cfg.topology, 'init_seed': int(seed), 'init_noise': float(noise)}
    target, cloud = fixed_target(cfg)
    if arm == 'A':
        _, params, sf, state, tess = closed_start_geometry(cfg, seed)
        map_type, training_cfg = 'closed_les', arm_training_cfg(cfg, hinge_gap=0.0)
    else:
        params, state, tess = init_direct_from_closed(cfg, seed)
        sf, map_type = None, 'direct_vertices'
        training_cfg = arm_training_cfg(cfg, hinge_gap=HINGE_GAP_W if arm == 'C' else 0.0)
        cfg = arm_validity_cfg(cfg, arm)
    extra_loss_fn = arm_extra_loss(cfg, arm, state, w_overlap=OVERLAP_W)
    return cfg, params, sf, state, map_type, training_cfg, target, cloud, extra_loss_fn


def run_arm(config, arm, seed, noise, epochs):
    """Train one arm for ``epochs`` and return its geometry at start, at the end, and deployed."""
    (cfg, params, sf, state, map_type, training_cfg, target, cloud,
     extra_loss_fn) = setup_arm(config, arm, seed, noise)
    n_faces, max_nodes = geom_shape(state)
    B = hinge_gap_operator(state)

    def nodes_of(p):
        mapped = apply_closed_les_mapping(state, p, sf) if arm == 'A' else apply_direct_mapping(state, p)
        # The sheet that reaches Stage 2 is the POST-Stage-1 one, so that is what gets audited.
        if cfg.validity.validity_method == 'alternating_projection':
            from nff.stages.projection import solve_alternating_projections
            mapped = solve_alternating_projections(mapped, n_iters=cfg.validity.n_proj_iters)
        return np.asarray(geom_coords(mapped)).reshape(n_faces, max_nodes, 2)

    flat_start = nodes_of(params)
    # The control the whole ablation rests on: arms B/C are seeded from arm A's own geometry, so
    # every arm must START connected and valid. Verified, not assumed.
    fv_start = flat_validity(flat_start.reshape(-1), state, B=B)
    optimizer, step = create_train_step(
        state, target, cfg.validity, cfg.physics, training_cfg, map_type=map_type, use_jit=True,
        load_specs=cfg.topology.get('loads', []), static_features=sf, target_cloud=cloud,
        extra_loss_fn=extra_loss_fn)
    ts = TrainState(params=params, opt_state=optimizer.init(params), rng=jax.random.PRNGKey(0))

    # `step` returns the loss at the params BEFORE the update, so pair it with those -- not with
    # ts.params, which is one gradient step further on.
    best, hist, t0 = (math.inf, params), [], time.time()
    for ep in range(epochs):
        prev = ts.params
        ts, loss, _ = step(ts)
        tot = float(loss)
        hist.append(tot)
        if math.isfinite(tot) and tot < best[0]:
            best = (tot, prev)
    best_loss, best_params = best

    flat_end = nodes_of(best_params)
    fv = flat_validity(flat_end.reshape(-1), state, B=B)
    dv = deploy_validity(cfg, best_params, state, map_type, sf)
    jax.clear_caches()
    return {'arm': arm, 'flat_start': flat_start, 'flat_end': flat_end,
            'deployed': dv.get('nodes'), 'loss': best_loss, 'hist': np.array(hist),
            'loss_start': hist[0] if hist else float('nan'),
            'flat': fv, 'flat_start_audit': fv_start, 'deploy': dv, 'secs': time.time() - t0}


def draw_sheet(ax, nodes, scale, color=ORANGE, alpha=0.55, lw=0.8, gaps=None):
    """Draw the panels of a flat or deployed sheet, in mm.

    ``add_patch`` does not update the data limits, so the caller must set them -- see
    ``set_common_limits``, which also keeps every panel at ONE scale so the arms are comparable
    by eye and not just by their titles.
    """
    for f in range(nodes.shape[0]):
        ax.add_patch(MplPolygon(nodes[f] * scale, closed=True, facecolor=color, alpha=alpha,
                                edgecolor=INK, linewidth=lw, zorder=2))
    if gaps is not None and len(gaps):
        for p, q in gaps:
            ax.plot([p[0] * scale, q[0] * scale], [p[1] * scale, q[1] * scale],
                    color=RED, lw=1.6, zorder=5, solid_capstyle="round")
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("#D3D6DB"); s.set_linewidth(0.7)


def set_common_limits(axes, clouds, pad=0.06):
    """One shared mm window over every point that will be drawn in the given axes."""
    pts = np.concatenate([np.asarray(c).reshape(-1, 2) for c in clouds if c is not None
                          and np.size(c)])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    span = (hi - lo).max()
    cen, half = 0.5 * (lo + hi), 0.5 * span * (1 + 2 * pad)
    for ax in axes:
        ax.set_xlim(cen[0] - half, cen[0] + half)
        ax.set_ylim(cen[1] - half, cen[1] + half)


def tiling_iou(nodes, stock):
    """Overlap between the union of the flat panels and the stock sheet they must be cut from."""
    ps = [Polygon(q) for q in np.asarray(nodes)]
    ps = [q if q.is_valid else q.buffer(0) for q in ps]
    u = unary_union(ps)
    return u.intersection(stock).area / max(u.union(stock).area, 1e-12)


def hinge_gap_segments(nodes, state):
    """The two bonded corner positions of every hinge -- drawn as a segment when they separate."""
    pairs = np.asarray(state.hinge_node_pairs)
    return [(nodes[int(a[0]), int(a[1])], nodes[int(b[0]), int(b[1])]) for a, b in pairs]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="closed/sheet_4x8ft_rom_28deg")
    ap.add_argument("--arms", nargs="+", default=["A", "B"], choices=["A", "B", "C", "D", "E"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--noise", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--out", default="data/outputs/chart_ablation/chart_arms.png")
    ap.add_argument("--dpi", type=int, default=260)
    args = ap.parse_args()

    apply_charter()
    config = load_and_parse_config(f"data/configs/{args.config}.yaml")
    scale = resolve_length_scale(config)
    topo = config.topology
    cx, cy = float(topo['target_cx']), float(topo['target_cy'])
    hw, hh = float(topo['target_half_w']), float(topo['target_half_h'])
    if float(topo.get('target_rot_deg', 0.0) or 0.0):
        raise SystemExit(f"config carries target_rot_deg={topo['target_rot_deg']}, but this figure "
                         "draws an axis-aligned Rectangle. It would show a target the run never "
                         "trained against. Add the rotation to the patch before using it here.")
    y0_ref, _, _, state0, _ = closed_start_geometry(config)
    _F, _K = geom_shape(state0)
    _n0 = np.asarray(y0_ref).reshape(_F, _K, 2)
    stock = box(_n0[:, :, 0].min(), _n0[:, :, 1].min(), _n0[:, :, 0].max(), _n0[:, :, 1].max())

    print(f"\n  config {args.config}   seed {args.seed}   noise {args.noise}   "
          f"{args.epochs} epochs   length_scale {scale:.1f} mm/unit")
    print(f"  target rectangle  {2*hw*scale:.0f} x {2*hh*scale:.0f} mm  "
          f"centred ({cx*scale:.0f}, {cy*scale:.0f}) mm\n")

    results = [run_arm(config, a, args.seed, args.noise, args.epochs) for a in args.arms]

    fig, axes = plt.subplots(len(results), 3, figsize=(9.2, 3.5 * len(results)), squeeze=False)
    for row, r in zip(axes, results):
        arm = r['arm']
        segs = hinge_gap_segments(r['flat_end'], state0)
        torn = [(p, q) for p, q in segs if np.linalg.norm(p - q) * scale > 1.0]

        draw_sheet(row[0], r['flat_start'], scale, color=GREY, alpha=0.35)
        g0 = r['flat_start_audit']['max_hinge_gap'] * scale
        row[0].set_title(f"{ARM_LABEL[arm]}\nflat sheet at start   (gap {g0:.2g} mm, "
                         f"loss {r['loss_start']:.4f})", fontsize=8.5, color=INK)

        draw_sheet(row[1], r['flat_end'], scale, color=ORANGE, gaps=torn)
        gap = r['flat']['max_hinge_gap'] * scale
        fov = r['flat']['overlap_fraction']
        # The stock sheet, drawn on both flat panels. A cut pattern is a PARTITION of it, so any
        # panel outside the dashes -- or any dashed area left uncovered -- is material that cannot
        # be cut from this sheet. This is the criterion the four kinematic ones miss.
        for c in (0, 1):
            sx, sy = stock.exterior.xy
            row[c].plot(np.array(sx) * scale, np.array(sy) * scale, color=INK, lw=1.1,
                        ls='--', dashes=(5, 3), zorder=8)
        tiling = tiling_iou(r['flat_end'], stock)
        row[1].set_title(f"flat design after {args.epochs} epochs\n"
                         f"gap {gap:.2g} mm   self-overlap {100*fov:.1f}%\n"
                         f"TILES THE STOCK: {tiling:.4f}"
                         + ("  ✓" if tiling > 0.999 else "  ✗ not a cut pattern"),
                         fontsize=8.5, color=INK)

        if r['deployed'] is not None:
            draw_sheet(row[2], r['deployed'], scale, color=TEAL, alpha=0.45)
        row[2].add_patch(Rectangle(((cx - hw) * scale, (cy - hh) * scale), 2 * hw * scale,
                                   2 * hh * scale, fill=False, edgecolor=RED, linewidth=1.3,
                                   linestyle="--", zorder=6))
        ov = r['deploy'].get('overlap_fraction', float('nan'))
        row[2].set_title(f"deployed vs target (red)\nloss {r['loss']:.4f}   "
                         f"panel overlap {100*ov:.1f}%", fontsize=8.5, color=INK)

        print(f"  arm {arm}  loss {r['loss']:.5f}  max hinge gap {gap:9.2e} mm  "
              f"flat connected {int(r['flat']['connected'])}  "
              f"flat overlap {100*fov:5.1f}%  "
              f"deploy overlap {100*ov:5.1f}%  ({r['secs']:.0f}s)")

    # ONE mm window for every flat panel and one for every deployed panel, shared across arms:
    # a chart that tears the sheet must not be flattered by its own tighter autoscale.
    tgt = np.array([[cx - hw, cy - hh], [cx + hw, cy + hh]])
    set_common_limits([row[c] for row in axes for c in (0, 1)],
                      [r[k] * scale for r in results for k in ('flat_start', 'flat_end')])
    set_common_limits([row[2] for row in axes],
                      [r['deployed'] * scale for r in results if r['deployed'] is not None]
                      + [tgt * scale])

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
