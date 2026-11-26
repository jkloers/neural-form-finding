"""The design gradient must agree with finite differences THROUGH the Stage-2 solve.

This is the regression guard for a silent gradient bug that cost five wrong diagnoses. The old
``setup_static_solver`` selected the adjoint linear solver like this::

    _ids = (functools.partial(solve_cg, ridge=backward_reg) if backward_reg > 0 else None)

so ``backward_reg = 0`` did not mean "no ridge" — it meant "fall back to jaxopt's default
``solve_normal_cg``", which solves the NORMAL equations and squares the condition number. With
cond(H) = 4.25e4 on the closed sheet the adjoint system carried ~1.8e9, CG did not converge, and
``d(compression)/d(design)`` came back 124x too large and sign-flipped, with no warning.

The failure was RHS-dependent, which is what made it so misleading: ``bending`` and ``chamfer`` were
exact (their cotangents live in the stiff rotational subspace, where ``normal_cg`` still converges)
while ``compression``, ``shearing`` and ``stretching`` were 90-97% wrong. That pattern reads as a
physics bug, not a linear-algebra one.

``backward_reg = 0`` is therefore the load-bearing case below: it is the setting that used to fail.
"""
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)   # a gradient check at 1e-3 is meaningless in float32
import pytest

from nff.config.experiment import load_and_parse_config
from nff.closed.setup import (build_closed_initial_state, init_closed_les_params,
                              build_surrogate_energy)
from nff.closed.gradcheck import directional_gradcheck, worst_relative_error, format_report
from nff.training.loss import compute_end_to_end_loss


CFG = "data/configs/closed/sheet_4x8ft_rom.yaml"

# Every additive weight off except the one under test, so a failure names a single term.
_ZERO = ('chamfer', 'regularization', 'material_area', 'hinge_gap', 'void_closure',
         'closure_delta', 'stretching', 'shearing', 'bending', 'contact', 'compression',
         'shear_path')


def _loss_and_params(term, weight, backward_reg):
    cfg = load_and_parse_config(CFG)
    object.__setattr__(cfg.physics, 'backward_reg', float(backward_reg))
    # Tight, because the finite difference must resolve a change the solver is not itself making.
    object.__setattr__(cfg.physics, 'solver_tol', 1e-12)
    object.__setattr__(cfg.physics, 'solver_maxiter', 20000)
    for k in _ZERO:
        object.__setattr__(cfg.training.loss_weights, k, 0.0)
    object.__setattr__(cfg.training.loss_weights, term, float(weight))

    state, _ = build_closed_initial_state(cfg)
    cfg.topology['init_noise'] = 0.0
    cfg.topology['init_seed'] = 0
    params, static_features = init_closed_les_params(cfg)
    bond_energy, probe_fn, geometry_fn, _, _ = build_surrogate_energy(
        cfg, static_features, state, params)

    def loss_fn(p):
        value, _ = compute_end_to_end_loss(
            p, state, cfg.target, cfg.validity, cfg.physics, cfg.training,
            map_type=cfg.mapping.type, static_features=static_features,
            load_specs=cfg.topology['loads'], bond_energy_fn=bond_energy,
            hinge_probe_fn=probe_fn, hinge_geometry_fn=geometry_fn)
        return value

    return loss_fn, params


@pytest.mark.parametrize("backward_reg", [0.0, 1e-6, 1e-2])
@pytest.mark.parametrize("term,weight", [('compression', 1.0), ('shear_path', 1.0),
                                         ('chamfer', 50.0)])
def test_design_gradient_matches_finite_differences(term, weight, backward_reg):
    """The analytic gradient tracks a central difference through Stage 2, at every ridge.

    ``backward_reg = 0`` is the case that used to return a 124x-wrong gradient. It must now pass on
    the same footing as any other ridge, because the adjoint solver is no longer chosen by whether
    the ridge happens to be positive.
    """
    loss_fn, params = _loss_and_params(term, weight, backward_reg)
    rows = directional_gradcheck(loss_fn, params, n_dir=2, seed=0)
    worst = worst_relative_error(rows)
    assert worst < 1e-2, (
        f"{term} at backward_reg={backward_reg:g}: relative error {worst:.3e}\n"
        + format_report(rows))


def test_zero_ridge_is_no_longer_a_different_solver():
    """A vanishing ridge must be continuous with a tiny one.

    Under the old code these two differed by orders of magnitude, because crossing zero swapped
    ``solve_cg`` for ``solve_normal_cg``. Continuity here is the property that was missing.

    The tolerance is 1e-4, not machine epsilon: both gradients come from an ITERATIVE adjoint solve,
    so they agree to its convergence level rather than exactly (measured 5.5e-06 apart). That is
    still four orders of magnitude tighter than the discontinuity being guarded against, which was a
    factor of ~124.
    """
    grads = []
    for backward_reg in (0.0, 1e-12):
        loss_fn, params = _loss_and_params('compression', 1.0, backward_reg)
        flat, _ = jax.flatten_util.ravel_pytree(jax.grad(loss_fn)(params))
        grads.append(np.asarray(flat))
    denom = max(np.abs(grads[1]).max(), 1e-30)
    assert np.abs(grads[0] - grads[1]).max() / denom < 1e-4


@pytest.mark.parametrize("term,weight", [('shear_path', 1.0), ('compression', 1.0)])
def test_path_term_gradient_is_finite_under_jit(term, weight):
    """A path term's gradient must be finite when JIT-COMPILED, not merely when traced eagerly.

    This guards a bug that eager checks structurally cannot catch. ``jnp.abs`` inside the shear path
    reduction returned an exact gradient eagerly and 176 of 226 NaN parameter gradients under
    ``jax.jit`` -- and the trainer JITs its step, so every ``shear_path`` run silently froze: the
    NaN-safe update zeroed the whole gradient and 200 epochs reproduced the initial design to six
    significant figures while the logs looked merely uninteresting.

    The failure was invisible to every other guard in the suite. It did not raise, the loss stayed
    finite, the finite-difference check above passed (it runs eagerly), and only ``grad_norm = nan``
    in the run history gave it away. Assert on the compiled function, because that is what trains.
    """
    loss_fn, params = _loss_and_params(term, weight, backward_reg=1e-6)
    grad_jit = jax.jit(jax.grad(loss_fn))(params)
    flat, _ = jax.flatten_util.ravel_pytree(grad_jit)
    flat = np.asarray(flat)
    n_nan = int(np.isnan(flat).sum())
    assert n_nan == 0, f"{term}: {n_nan}/{flat.size} JIT gradient entries are NaN"

    # And the compiled gradient must agree with the eager one, not merely be finite -- "finite but
    # different" would be the same class of silent corruption.
    #
    # 1e-3, not machine epsilon: both paths run an ITERATIVE adjoint solve and XLA fuses the
    # compiled one differently, so they converge to slightly different points (measured 1e-5 for
    # shear_path, 1e-4 for compression -- the same order as the ridge-continuity check above). The
    # bug being guarded made 78% of the entries NaN, so any threshold in this range catches it while
    # leaving room for solver noise.
    eager, _ = jax.flatten_util.ravel_pytree(jax.grad(loss_fn)(params))
    eager = np.asarray(eager)
    denom = max(np.abs(eager).max(), 1e-30)
    assert np.abs(flat - eager).max() / denom < 1e-3
