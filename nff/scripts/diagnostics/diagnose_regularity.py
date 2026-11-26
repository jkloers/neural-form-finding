"""E1 -- regularity of the design objective at the closed-hinge set.

The ligament reference frame is selected by a threshold. ``build_reference_bond_vectors``
(``nff/stages/geometry.py``) returns a FIXED fallback vector while the flat hinge gap satisfies
``|b| < min_len`` and returns ``b`` itself once ``|b| >= min_len``. The two branches do not agree in
the limit: just below the threshold the frame points along a canonical axis, just above it points
wherever the gap happened to open. The strain measure is resolved against that frame, so the energy,
the equilibrium it produces and the design objective all inherit the switch.

The consequence is geometric. Let ``M = {y : By = 0}`` be the closed-hinge set. The singular locus is
not ``M`` itself but a SHELL at ``|b| = min_len`` around it. A chart whose image lies in ``M``, with
tangent image in ``TM``, stays strictly inside the fallback branch where the frame is constant and
never meets the shell. A chart with free panel vertices crosses it.

This module measures that, rather than assuming it:

  A. locate the shell -- per-hinge crossing points along a transverse ray;
  B. evaluate the objective across the crossing and test for a jump in VALUE (a discontinuity is a
     stronger statement than a kink, and is what a discontinuous frame predicts);
  C. contrast transverse rays (``Bd != 0``), tangent rays (``Bd = 0``) and the closed chart's own
     directions, which are tangent by construction.

    JAX_PLATFORMS=cpu conda run -n kgnn_mac python nff/scripts/diagnostics/diagnose_regularity.py
"""

import argparse
import json
import os

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from nff.analysis.chart_ablation import (build_geom_loss_fn, chart_basis, chart_jacobian,
                                       closed_start_geometry, geom_shape, hinge_gap_operator)
from nff.config.experiment import load_and_parse_config

MIN_LEN = 1e-4          # the threshold in build_reference_bond_vectors; the shell radius


def hinge_gaps(y, B):
    """Per-hinge gap magnitude |b_h| of a flat design."""
    return np.linalg.norm((np.asarray(B) @ np.asarray(y)).reshape(-1, 2), axis=1)


def direction_set(B, Q_A, rng, n_geom):
    """Transverse, tangent and chart directions, all unit norm in the geometric metric."""
    P_row = B.T @ np.linalg.solve(B @ B.T, B)          # orthogonal projector onto row(B)
    v = rng.normal(size=n_geom)
    d_trans = P_row @ v
    d_tan = v - P_row @ v
    out = {'transverse': d_trans / np.linalg.norm(d_trans),
           'tangent': d_tan / np.linalg.norm(d_tan)}
    for k in range(min(3, Q_A.shape[1])):
        out[f'chart mode {k}'] = Q_A[:, k] / np.linalg.norm(Q_A[:, k])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="closed/sheet_4x8ft_rom_28deg")
    ap.add_argument("--out", default="data/outputs/chart_ablation_28deg/regularity.npz")
    ap.add_argument("--n-scan", type=int, default=161, help="samples per side of the crossing")
    ap.add_argument("--span", type=float, default=40.0,
                    help="scan spans crossing/span .. crossing*span")
    args = ap.parse_args()

    cfg = load_and_parse_config(f"data/configs/{args.config}.yaml")
    y0, _, _, state, _ = closed_start_geometry(cfg)
    y0 = np.asarray(y0)
    F, K = geom_shape(state)
    B = hinge_gap_operator(state)
    J, _, _, _ = chart_jacobian(cfg)
    Q_A, k_A, _ = chart_basis(J)
    L_MM = float(cfg.hinge_model.length_scale)

    loss_fn, _, _, _ = build_geom_loss_fn(cfg)
    lf = jax.jit(loss_fn)
    L0 = float(lf(jnp.asarray(y0)))
    g0 = hinge_gaps(y0, B)
    print(f"  reference design: L0 = {L0:.10e}   max |b| = {g0.max():.3e} "
          f"(shell at |b| = {MIN_LEN:g})\n")

    rng = np.random.default_rng(0)
    dirs = direction_set(B, Q_A, rng, y0.size)
    results = {}

    for name, d in dirs.items():
        # A. where does this ray first cross the shell? |b_h(t)| = |t| * |B_h d|
        rate = np.linalg.norm((B @ d).reshape(-1, 2), axis=1)      # d|b_h|/dt
        crossing = MIN_LEN / rate.max() if rate.max() > 0 else np.inf
        tangent = rate.max() < 1e-12

        if tangent:
            # No crossing exists: the ray never leaves the fallback branch. Scan a comparable
            # geometric range so the two cases are sampled at the same physical scale.
            ts = np.concatenate([-np.logspace(-6, -2, args.n_scan)[::-1],
                                 np.logspace(-6, -2, args.n_scan)])
        else:
            lo, hi = crossing / args.span, crossing * args.span
            ts = np.concatenate([-np.logspace(np.log10(lo), np.log10(hi), args.n_scan)[::-1],
                                 np.logspace(np.log10(lo), np.log10(hi), args.n_scan)])

        # B. THE LIMIT TEST. A discontinuity is a statement about a limit, not about one sample.
        # Bracket the crossing symmetrically and shrink the bracket over decades:
        #     Delta(delta) = L(t* + delta) - L(t* - delta).
        # A jump keeps Delta bounded away from zero as delta -> 0; a continuous function drives it
        # to zero (linearly in delta if C^1). Sampling on a log grid and taking the largest
        # consecutive difference does NOT test this -- the spacing grows with t, so the largest
        # step lands at the largest t for any function whatsoever.
        anchor = crossing if not tangent else 1e-3
        deltas = anchor * np.logspace(-1, -6, 12)
        jumps = np.array([float(lf(jnp.asarray(y0 + (anchor + dd) * d)))
                          - float(lf(jnp.asarray(y0 + (anchor - dd) * d))) for dd in deltas])
        # Control: the same bracket at a point with no threshold crossing anywhere near it.
        ctrl_anchor = anchor * 7.0
        ctrl = np.array([float(lf(jnp.asarray(y0 + (ctrl_anchor + dd) * d)))
                         - float(lf(jnp.asarray(y0 + (ctrl_anchor - dd) * d))) for dd in deltas])

        vals = np.array([float(lf(jnp.asarray(y0 + t * d))) for t in ts])
        gmax = np.array([hinge_gaps(y0 + t * d, B).max() for t in ts])

        # Decay exponent of |Delta| against delta: ~1 for a C^1 function, ~0 for a jump.
        good = np.abs(jumps) > 0
        slope = (np.polyfit(np.log10(deltas[good]), np.log10(np.abs(jumps[good])), 1)[0]
                 if good.sum() > 2 else np.nan)
        slope_c = np.polyfit(np.log10(deltas), np.log10(np.abs(ctrl) + 1e-300), 1)[0]

        results[name] = dict(ts=ts, vals=vals, gmax=gmax, crossing=float(crossing),
                             tangent=bool(tangent), deltas=deltas, jumps=jumps, ctrl=ctrl,
                             slope=float(slope), slope_ctrl=float(slope_c))
        cross_mm = crossing / np.sqrt(y0.size // 2) * L_MM if np.isfinite(crossing) else np.inf
        print(f"  {name:<16} shell crossing at t = "
              f"{'never (tangent)' if tangent else f'{crossing:.3e} u = {cross_mm:.4f} mm RMS'}")
        print(f"  {'':16} |Delta| across the bracket: {abs(jumps[0]):.3e} -> {abs(jumps[-1]):.3e}"
              f"  as delta shrinks 1e5x")
        print(f"  {'':16} decay exponent {slope:+.2f}   (1 = continuous, 0 = JUMP)"
              f"   control {slope_c:+.2f}")
        print(f"  {'':16} max |b| over the scan: {gmax.max():.3e}\n")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, min_len=MIN_LEN, L0=L0, length_scale=L_MM,
                        **{f"{k}__{f}": v[f] for k, v in results.items()
                           for f in ('ts', 'vals', 'gmax', 'deltas', 'jumps', 'ctrl')},
                        summary=json.dumps({k: {f: v[f] for f in
                                                ('crossing', 'tangent', 'slope', 'slope_ctrl')}
                                            for k, v in results.items()}))
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
