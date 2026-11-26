"""Taylor remainder convergence test -- is the design gradient actually a gradient?

The verification standard of PDE-constrained optimisation (Farrell et al., ``dolfin-adjoint``).
Writing the objective as a pure function of the design parameters,

    R1(h) = | L(t + h d) - L(t) |                          must be O(h)
    R2(h) = | L(t + h d) - L(t) - h <grad L, d> |          must be O(h^2)

so halving h must divide R2 by four and the observed order log2(R2(h)/R2(h/2)) must converge to 2.
A gradient that fails this is not the gradient of the implemented objective, whatever automatic
differentiation returned.

Two properties make this the right instrument here, where every previous one failed:

  * the observed ORDER IS DIMENSIONLESS, so a 28-parameter chart and a 72-parameter chart are
    compared with no metric, unit or dimension argument available -- the Cauchy-interlacing
    objection that invalidated the condition-number comparison cannot be raised against it;
  * each method is tested IN ITS OWN PARAMETERISATION, the one it is actually optimised in, rather
    than pushed into a shared coordinate that belongs to one of them.

⚠ SEPARATING A WRONG ADJOINT FROM A NON-SMOOTH OBJECTIVE. A failing Taylor order has two possible
causes, and they must be told apart. The usual instrument -- a dot-product test comparing the
reverse-mode gradient against a forward-mode JVP -- is IMPOSSIBLE here: the Stage-2 solver
differentiates implicitly through a ``custom_vjp``, which JAX supports in reverse mode only, so
``jax.jvp`` raises outright.

The substitute is stronger than the test it replaces. BOTH METHODS SHARE THE IDENTICAL ADJOINT
MACHINERY -- the same solver, the same implicit-differentiation rule, the same energy. If either
method attains order 2, the adjoint implementation is thereby verified, and the other method's
failure to attain it cannot be attributed to the adjoint. The shared machinery does the separating.

    JAX_PLATFORMS=cpu conda run -n kgnn_mac python nff/scripts/diagnostics/taylor_test.py
"""

import argparse
import json
import os

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from nff.analysis.chart_ablation import (arm_extra_loss, arm_training_cfg, arm_validity_cfg,
                                       build_geom_loss_fn, closed_start_geometry, fixed_target,
                                       geom_shape, hinge_gap_operator, init_direct_from_closed,
                                       project_if_needed)
from nff.closed.setup import init_closed_les_params
from nff.hinge import build_hinge_model
from nff.config.experiment import load_and_parse_config
from nff.stages.mapping import apply_direct_mapping
from nff.training.loss import compute_end_to_end_loss

OVERLAP_W = 1.0e3

SHIPPING = 'shipping objective (surrogate)'


def physics_at(cfg, maxiter=None, tol=None):
    """``cfg`` with the Stage-2 solver settings overridden, every other field carried over.

    Rebuilt field by field from the annotations rather than mutated in place, so a field added to
    ``PhysicsConfig`` later cannot be silently dropped back to its default here.
    """
    if maxiter is None and tol is None:
        return cfg
    import equinox as eqx
    from nff.config.experiment import PhysicsConfig
    p = cfg.physics
    fields = {name: getattr(p, name) for name in p.__annotations__}
    if maxiter is not None:
        fields['solver_maxiter'] = int(maxiter)
    if tol is not None:
        fields['solver_tol'] = float(tol)
    return eqx.tree_at(lambda c: c.physics, cfg, PhysicsConfig(**fields))


def shipping_objective(cfg, maxiter=None, tol=None):
    """L as a pure function of theta, under the SURROGATE hinge law the paper designs with.

    The other three arms run the ROM ligament. This one is the objective actually minimised by
    ``run_closed.py``, and it is the one the differentiability claim is about, so it is built the
    way the driver builds it rather than the way the ablation builds its arms:

      * the surrogate energy, probe and design-tracked hinge geometry are threaded in, so
        ``hinge_model.type: surrogate`` is read. Without them the call falls back to the ROM
        springs and the test reports a ROM result under a surrogate config's name;
      * ``w_lig_logit`` joins the design vector when the surrogate makes the ligament width
        learnable. Omitting it would certify a smaller design space than is optimised;
      * ``config.training`` is passed unchanged, keeping the damage, trust-box and compression
        weights that ``arm_training_cfg`` is entitled to strip for a cross-chart comparison.
    """
    from nff.closed.setup import build_closed_initial_state
    cfg = physics_at(cfg, maxiter, tol)
    params0, sf = init_closed_les_params(cfg)
    state, _ = build_closed_initial_state(cfg)
    bond_energy, probe_fn, geometry_fn, _, w_lig_logit0 = build_hinge_model(
        cfg, sf, state, params0)
    if w_lig_logit0 is not None:
        params0 = {**params0, 'w_lig_logit': w_lig_logit0}
    target, cloud = fixed_target(cfg)
    theta0, unravel = ravel_pytree(params0)

    def L(theta):
        return compute_end_to_end_loss(
            unravel(theta), state, target, cfg.validity, cfg.physics, cfg.training,
            map_type='closed_les', static_features=sf,
            load_specs=cfg.topology.get('loads', []), target_cloud=cloud,
            bond_energy_fn=bond_energy, hinge_probe_fn=probe_fn,
            hinge_geometry_fn=geometry_fn)[0]

    return L, np.asarray(theta0)


def closed_chart_objective(cfg):
    """L as a pure function of the NATIVE design vector theta = (z, bnd_logits)."""
    params0, sf = init_closed_les_params(cfg)
    state, _ = __import__('nff.closed.setup', fromlist=['x']).build_closed_initial_state(cfg)
    target, cloud = fixed_target(cfg)
    tcfg = arm_training_cfg(cfg, hinge_gap=0.0)
    theta0, unravel = ravel_pytree(params0)

    def L(theta):
        return compute_end_to_end_loss(
            unravel(theta), state, target, cfg.validity, cfg.physics, tcfg,
            map_type='closed_les', static_features=sf,
            load_specs=cfg.topology.get('loads', []), target_cloud=cloud)[0]

    return L, np.asarray(theta0)


def baseline_objective(cfg, w_overlap=OVERLAP_W):
    """L as a pure function of the baseline's native design vector y, WITH its full mechanism.

    The strongest competitor: free panel vertices, Stage-1 projection closing the hinge gap, and a
    global non-overlap penalty. Both parts of its constraint machinery are inside the objective
    being differentiated, which is what it is actually optimised through.
    """
    cfg_e = arm_validity_cfg(cfg, 'E')
    params0, state, _ = init_direct_from_closed(cfg, None)
    target, cloud = fixed_target(cfg)
    tcfg = arm_training_cfg(cfg, hinge_gap=0.0)
    F, K = geom_shape(state)
    extra = arm_extra_loss(cfg_e, 'E', state, w_overlap=w_overlap)
    y0, unravel = ravel_pytree(params0)

    def L(y):
        p = unravel(y)
        return compute_end_to_end_loss(
            p, state, target, cfg_e.validity, cfg_e.physics, tcfg,
            map_type='direct_vertices', static_features=None,
            load_specs=cfg.topology.get('loads', []), target_cloud=cloud)[0] \
            + (extra(p) if extra is not None else 0.0)

    return L, np.asarray(y0)


def fd_cross_check(L, x0, rng, h=1e-3, n=4):
    """<grad L, d> against a central difference at a step drawn from INSIDE the quadratic band.

    NOT a substitute for the adjoint test -- a central difference is itself only as good as the
    smoothness of the objective, which is the thing under examination. Reported so that a gradient
    that is grossly wrong is visible immediately, while the actual separation of "wrong adjoint"
    from "non-smooth objective" rests on the shared-machinery argument in the module docstring.

    The pairs are returned, not just their worst ratio. Read as a scatter against the diagonal
    this is the one instrument here that never takes a limit: ``h`` is fixed inside the band the
    Taylor sweep certifies, so the cancellation that ends that band cannot reach it. The observed
    order, by contrast, differences neighbouring remainders and therefore amplifies their noise
    the moment the remainder stops falling.

    ⚠ ``h`` must lie inside the arm's own quadratic band. At the band edge the linear
    contamination below the band is already the same size as the quadratic signal, and the
    deviation reported here is then a property of the step, not of the gradient.

    Returns:
        ``(worst, fd, ref)``. ``worst`` is the largest relative deviation, kept for the printed
        line; ``fd`` and ``ref`` are the ``(n,)`` central-difference and adjoint directional
        derivatives, in the order drawn.
    """
    Lf, gf = jax.jit(L), jax.jit(jax.grad(L))
    g = np.asarray(gf(jnp.asarray(x0)))
    fds, refs = np.zeros(n), np.zeros(n)
    for i in range(n):
        d = rng.normal(size=x0.size)
        d /= np.linalg.norm(d)
        fds[i] = (float(Lf(jnp.asarray(x0 + h * d)))
                  - float(Lf(jnp.asarray(x0 - h * d)))) / (2 * h)
        refs[i] = float(g @ d)
    worst = float(np.max(np.abs(fds - refs) / np.maximum(np.abs(refs), 1e-300)))
    return worst, fds, refs


def taylor(L, x0, rng, n_steps, h0, n_dir, ratio=2.0):
    """Remainders and observed orders over a geometric step sequence, averaged over directions.

    ``ratio`` is the factor between consecutive steps. Two is the textbook halving; a smaller
    factor samples the same range of ``h`` more finely, which is what resolves where the
    quadratic band ends and the solver floor begins. The observed order is therefore taken
    against the actual step ratio rather than assuming a halving.
    """
    Lf = jax.jit(L)
    gf = jax.jit(jax.grad(L))
    L0 = float(Lf(jnp.asarray(x0)))
    g0 = np.asarray(gf(jnp.asarray(x0)))
    hs = h0 * float(ratio) ** (-np.arange(n_steps))
    R1 = np.zeros((n_dir, n_steps))
    R2 = np.zeros((n_dir, n_steps))
    for i in range(n_dir):
        d = rng.normal(size=x0.size)
        d /= np.linalg.norm(d)
        gd = float(g0 @ d)
        for j, h in enumerate(hs):
            Lh = float(Lf(jnp.asarray(x0 + h * d)))
            R1[i, j] = abs(Lh - L0)
            R2[i, j] = abs(Lh - L0 - h * gd)
    r1, r2 = R1.mean(axis=0), R2.mean(axis=0)
    # Each order spans a FACTOR OF TWO in h, whatever the sampling ratio, and is padded back to
    # full length. An adjacent-pair order divides by log(ratio), so fine sampling would inflate the
    # noise of the very quantity the test reads: at sqrt(2) spacing an arm that is cleanly
    # quadratic scatters between 0.8 and 3.0. Spanning a fixed factor keeps the classical
    # signal-to-noise while giving one estimate per sample.
    lag = max(1, int(round(np.log(2.0) / np.log(float(ratio)))))
    pad = np.full(lag, np.nan)
    step = np.log(hs[:-lag] / hs[lag:])
    o1 = np.concatenate([pad, np.log(r1[:-lag] / np.maximum(r1[lag:], 1e-300)) / step])
    o2 = np.concatenate([pad, np.log(r2[:-lag] / np.maximum(r2[lag:], 1e-300)) / step])
    return hs, r1, o1, r2, o2


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="closed/sheet_4x8ft_rom_28deg",
                    help="the three CHART arms. Its load is re-bisected for the ROM ligament.")
    ap.add_argument("--config-shipping", default="closed/sheet_4x8ft_pet_28deg",
                    help="the surrogate arm. A separate config because the PET baseline runs "
                         "2.50 N/face against the ROM config's 13.20, and each is calibrated to "
                         "its own hinge law.")
    ap.add_argument("--out", default="data/outputs/chart_ablation_28deg/taylor.npz")
    ap.add_argument("--halvings", type=int, default=21, dest="n_steps",
                    help="number of steps in the geometric sequence.")
    ap.add_argument("--ratio", type=float, default=2.0 ** 0.5,
                    help="factor between consecutive steps. sqrt(2) samples the classical "
                         "halving sequence twice as finely over the same range of h.")
    ap.add_argument("--h0", type=float, default=1e-1)
    ap.add_argument("--n-dir", type=int, default=8,
                    help="random directions averaged at each step. The observed order is a ratio "
                         "of two small differences, so its variance is what this buys down.")
    ap.add_argument("--maxiter", type=int, action="append", default=[],
                    help="extra solver_maxiter values for the shipping arm, one series each. The "
                         "forward-solve sensitivity: the adjoint is only as good as the state it "
                         "is taken at, so a failing order here localises the fault in the solve.")
    ap.add_argument("--tol", type=float, action="append", default=[],
                    help="extra solver_tol values for the shipping arm, one series each. The "
                         "remainder is a difference of converged objective values, so the solve "
                         "tolerance sets the floor R2 cannot fall below.")
    ap.add_argument("--gradcheck-h", type=float, default=4e-3,
                    help="step for the adjoint-vs-central-difference cross-check. It must lie "
                         "INSIDE every arm's quadratic band: 4e-3 clears the surrogate's "
                         "1.1e-3..1.25e-2 and the ROM chart's 1.4e-4..2.5e-2. At the band edge "
                         "the linear contamination below the band already matches the quadratic "
                         "signal, and the deviation then measures the step, not the gradient.")
    ap.add_argument("--gradcheck-n", type=int, default=48,
                    help="random directions in the cross-check. Each costs two forward solves.")
    args = ap.parse_args()

    cfg = load_and_parse_config(f"data/configs/{args.config}.yaml")
    cfg_ship = load_and_parse_config(f"data/configs/{args.config_shipping}.yaml")
    methods = [(SHIPPING, lambda _c: shipping_objective(cfg_ship)),
               ('closed chart', closed_chart_objective),
               ('free vertices + projection + overlap penalty', baseline_objective),
               # Isolates WHICH part of the baseline's mechanism breaks the gradient: the same
               # chart with the projection but WITHOUT the non-overlap penalty.
               ('free vertices + projection only',
                lambda c: baseline_objective(c, w_overlap=0.0))]
    methods += [(f"{SHIPPING} maxiter={m}",
                 (lambda m: lambda _c: shipping_objective(cfg_ship, maxiter=m))(m))
                for m in args.maxiter]
    methods += [(f"{SHIPPING} tol={t:g}",
                 (lambda t: lambda _c: shipping_objective(cfg_ship, tol=t))(t))
                for t in args.tol]
    out = {}
    for label, builder in methods:
        L, x0 = builder(cfg)
        rng = np.random.default_rng(0)
        adj, fd, ref = fd_cross_check(L, x0, rng, h=args.gradcheck_h, n=args.gradcheck_n)
        rel = np.abs(fd - ref) / np.maximum(np.abs(ref), 1e-300)
        print(f"\n  {label}   ({x0.size} design parameters)")
        print(f"    grad.d vs central FD at h={args.gradcheck_h:g} over {args.gradcheck_n} "
              f"directions:  median {np.median(rel):.3e}  worst {adj:.3e}")
        hs, r1, o1, r2, o2 = taylor(L, x0, np.random.default_rng(1), args.n_steps,
                                    args.h0, args.n_dir, ratio=args.ratio)
        print(f"    {'h':>10}{'R1':>13}{'order':>8}{'R2':>13}{'order':>8}")
        for j in range(len(hs)):
            s1 = f"{o1[j]:8.2f}" if np.isfinite(o1[j]) else f"{'':>8}"
            s2 = f"{o2[j]:8.2f}" if np.isfinite(o2[j]) else f"{'':>8}"
            print(f"    {hs[j]:10.2e}{r1[j]:13.3e}{s1}{r2[j]:13.3e}{s2}")
        tail = o2[len(o2) // 2:]
        print(f"    observed order of R2 over the finest half: {np.nanmedian(tail):.2f}"
              f"   (2.00 = the gradient is verified)")
        out[label] = dict(h=hs, R1=r1, o1=o1, R2=r2, o2=o2, fd=fd, ref=ref,
                          adjoint=adj, n_param=int(x0.size))
        jax.clear_caches()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out,
                        **{f"{k}__{f}": v[f] for k, v in out.items()
                           for f in ('h', 'R1', 'o1', 'R2', 'o2', 'fd', 'ref')},
                        meta=json.dumps({k: {'adjoint': v['adjoint'], 'n_param': v['n_param'],
                                             'gradcheck_h': args.gradcheck_h}
                                         for k, v in out.items()}))
    print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
