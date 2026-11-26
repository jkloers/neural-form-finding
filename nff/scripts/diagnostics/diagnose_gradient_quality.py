"""E2/E3 -- is the gradient usable, and is it the right gradient? Both with their controls.

E1 established that the objective is discontinuous across a shell at ``|b| = 1e-4`` and that the
closed chart's tangent image lies in ``ker(B)``, so it never meets that shell. E2 and E3 ask what
that costs an optimiser. Each carries the control that makes it a measurement rather than an
artefact of how it was estimated:

E2  SMOOTHNESS MODULUS. beta = sup ||grad L(y1) - grad L(y2)|| / ||y1 - y2||, distances in mm.
    Gradient descent converges only below 2/beta, so beta becomes a predicted maximum step.
    ⚠ CONTROL: beta is a SUPREMUM OVER A SUBSPACE, so a 72-dimensional chart will exceed a
    12-dimensional one for the same reason Cauchy interlacing flatters any restriction. The chart's
    beta is therefore compared against beta measured on RANDOM subspaces of the SAME dimension.

E3  GRADIENT BIAS (Suh et al., ICML 2022). AD returns a gradient wherever the objective happens to
    be differentiable; on a discontinuous landscape the meaningful object is the gradient of the
    smoothed objective L_sigma(y) = E_xi[L(y + sigma xi)].
    ⚠ CONTROL: a zeroth-order estimator of grad L_sigma in k dimensions with N samples has relative
    error ~ sqrt(k/N), which for modest N is the same size as the bias being looked for. Estimating
    it once measures that variance and nothing else. Independent REPLICAS are therefore run, the
    bias is taken against the replica MEAN, and the standard error is reported beside it. A bias
    smaller than its own standard error is not a result.
"""

import argparse
import os

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from nff.closed.chart_ablation import (arm_validity_cfg, build_geom_loss_fn, chart_basis,
                                       chart_jacobian, closed_start_geometry, geom_shape)
from nff.config.experiment import load_and_parse_config


def beta_hat(gf, y0, basis, rng, n_pair, radius, scale_mm, n_nodes):
    """Gradient-Lipschitz constant over pairs sampled in ``basis`` (ambient if None), per mm."""
    k = basis.shape[1] if basis is not None else y0.size
    pts, grads = [], []
    for _ in range(n_pair):
        v = rng.normal(size=k)
        v /= np.linalg.norm(v)
        y = y0 + radius * (basis @ v if basis is not None else v)
        g = np.asarray(gf(jnp.asarray(y)))
        pts.append(y)
        grads.append(basis.T @ g if basis is not None else g)
    pts, grads = np.array(pts), np.array(grads)
    best = 0.0
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            dy = np.linalg.norm(pts[i] - pts[j]) / np.sqrt(n_nodes) * scale_mm
            if dy > 0:
                best = max(best, np.linalg.norm(grads[i] - grads[j]) / dy)
    return best


def zeroth_order_grad(lf, y0, basis, sigma, n_samp, rng):
    """One antithetic zeroth-order estimate of grad L_sigma restricted to ``basis``."""
    k = basis.shape[1] if basis is not None else y0.size
    acc = np.zeros(k)
    for _ in range(n_samp // 2):
        xi = rng.normal(size=k)
        step = sigma * (basis @ xi if basis is not None else xi)
        acc += (float(lf(jnp.asarray(y0 + step)))
                - float(lf(jnp.asarray(y0 - step)))) / (2.0 * sigma) * xi
    return acc / (n_samp // 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="closed/sheet_4x8ft_rom_28deg")
    ap.add_argument("--out", default="data/outputs/chart_ablation_28deg/gradient_quality.npz")
    ap.add_argument("--n-pair", type=int, default=14)
    ap.add_argument("--radius", type=float, default=2e-3)
    ap.add_argument("--n-null", type=int, default=24, help="random subspaces for the E2 control")
    ap.add_argument("--n-samp", type=int, default=200, help="E3 samples per replica")
    ap.add_argument("--n-rep", type=int, default=6, help="E3 independent replicas")
    args = ap.parse_args()

    cfg = load_and_parse_config(f"data/configs/{args.config}.yaml")
    y0, _, _, state, _ = closed_start_geometry(cfg)
    y0 = np.asarray(y0)
    F, K = geom_shape(state)
    n_nodes, scale_mm = F * K, float(cfg.hinge_model.length_scale)
    J, _, _, _ = chart_jacobian(cfg)
    Q_A, k_A, _ = chart_basis(J)

    sigmas = np.logspace(-2, -5, 4)
    charts = [('closed chart', 'A', Q_A), ('free + projection', 'D', None),
              ('free, naive', 'B', None)]
    res = {}

    for label, arm, basis in charts:
        loss_fn, _, _, _ = build_geom_loss_fn(arm_validity_cfg(cfg, arm))
        lf, gf = jax.jit(loss_fn), jax.jit(jax.grad(loss_fn))

        b = beta_hat(gf, y0, basis, np.random.default_rng(0), args.n_pair, args.radius,
                     scale_mm, n_nodes)
        g_ad = np.asarray(gf(jnp.asarray(y0)))
        g_ad = basis.T @ g_ad if basis is not None else g_ad

        bias, se_rel = [], []
        for s in sigmas:
            reps = np.array([zeroth_order_grad(lf, y0, basis, float(s), args.n_samp,
                                               np.random.default_rng(100 + r))
                             for r in range(args.n_rep)])
            gbar = reps.mean(axis=0)
            # standard error of the MEAN estimate, as a norm relative to that mean
            se = np.linalg.norm(reps.std(axis=0, ddof=1)) / np.sqrt(args.n_rep)
            nb = max(np.linalg.norm(gbar), 1e-300)
            bias.append(np.linalg.norm(g_ad - gbar) / nb)
            se_rel.append(se / nb)
        res[label] = dict(beta=b, eta=2.0 / b, bias=np.array(bias), se=np.array(se_rel))
        print(f"  {label:<20} beta-hat {b:10.3e}/mm   max stable step {2.0/b:9.3e} mm")
        print(f"  {'':20} bias +- SE:  " + "   ".join(
            f"s={s:.0e}: {bi:.2f}+-{sei:.2f}" for s, bi, sei in zip(sigmas, bias, se_rel)))
        print(f"  {'':20} significant (bias > 2 SE): " + "  ".join(
            f"{'YES' if bi > 2 * sei else 'no ':>3}" for bi, sei in zip(bias, se_rel)) + "\n")
        jax.clear_caches()

    # ---- E2 CONTROL: beta on random subspaces of the chart's own dimension ------------------
    loss_fn, _, _, _ = build_geom_loss_fn(arm_validity_cfg(cfg, 'D'))
    gf = jax.jit(jax.grad(loss_fn))
    rng = np.random.default_rng(7)
    null = []
    for _ in range(args.n_null):
        Q, _ = np.linalg.qr(rng.normal(size=(y0.size, k_A)))
        null.append(beta_hat(gf, y0, Q, np.random.default_rng(0), args.n_pair, args.radius,
                             scale_mm, n_nodes))
    null = np.array(null)
    bc = res['closed chart']['beta']
    pct = 100.0 * (null < bc).mean()
    print(f"  E2 CONTROL -- beta on {args.n_null} random {k_A}-dim subspaces of the same Hessian")
    print(f"    random {k_A}-dim:  median {np.median(null):.3e}/mm"
          f"   [p5 {np.percentile(null,5):.3e}, p95 {np.percentile(null,95):.3e}]")
    print(f"    THE CHART:      {bc:.3e}/mm   <- percentile {pct:.1f}  (lower is smoother)")
    print(f"    {'the chart is SMOOTHER than a random subspace of its own dimension'if pct<50 else 'the chart is NOT smoother than a random subspace of its dimension'}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, sigmas=sigmas, null_beta=null, chart_pct=pct,
                        **{f"{k}__{f}": v[f] for k, v in res.items() for f in ('bias', 'se')},
                        **{f"{k}__beta": v['beta'] for k, v in res.items()})
    print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
