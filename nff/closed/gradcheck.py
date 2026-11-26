"""Analytic design gradient versus central finite differences, swept over the step size.

The check is directional: a random unit direction ``d`` in design space, the analytic directional
derivative ``<grad L, d>``, and the central difference ``(L(t+eps) - L(t-eps)) / 2eps`` at several
``eps``. Sweeping ``eps`` is what makes the result readable — a genuine gradient bug holds its
relative error flat as ``eps`` shrinks, while finite-difference noise grows and truncation error
shrinks, so a single step size cannot tell the two apart.

Why this exists as a library rather than a figure panel: the same recipe already lived inside
``nff/scripts/diagnostics/diagnose_chart.py`` (plotted by ``plot_chart_conditioning.py``), but it was
only ever pointed at the GEOMETRIC loss. It therefore never crossed the Stage-2 solve, and so never
exercised jaxopt's implicit-diff adjoint — which is exactly where the gradient was silently wrong:
``physics.backward_reg = 0`` used to select ``solve_normal_cg``, squaring the tangent-stiffness
condition number and returning a gradient off by 124x. A gradient check that stops short of the
implicit solve does not check the part that breaks.

Anything differentiable works; ``loss_fn`` need not involve physics at all.
"""

from typing import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree


DEFAULT_EPS = (1e-4, 1e-3, 1e-2)


def directional_gradcheck(loss_fn: Callable, params, n_dir: int = 4,
                          eps_values: Sequence[float] = DEFAULT_EPS,
                          seed: int = 0) -> np.ndarray:
    """Compare the analytic directional derivative to a central difference.

    Args:
        loss_fn: Scalar function of the params PyTree. Must be differentiable by ``jax.grad``.
        params: PyTree of design parameters; the point the check is performed at.
        n_dir: Random unit directions to test.
        eps_values: Finite-difference step sizes to sweep.
        seed: Seed for the direction draw.

    Returns:
        (n_dir * len(eps_values), 4) array of rows ``(eps, analytic, finite_difference,
        relative_error)``.
    """
    theta0, unravel = ravel_pytree(params)
    theta0 = np.asarray(theta0)
    grad_fn = jax.jit(jax.grad(lambda t: loss_fn(unravel(t))))
    loss_flat = jax.jit(lambda t: loss_fn(unravel(t)))
    g0 = np.asarray(grad_fn(jnp.asarray(theta0)))

    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_dir):
        d = rng.normal(size=theta0.size)
        d /= np.linalg.norm(d)
        analytic = float(np.dot(g0, d))
        for eps in eps_values:
            plus = float(loss_flat(jnp.asarray(theta0 + eps * d)))
            minus = float(loss_flat(jnp.asarray(theta0 - eps * d)))
            fd = (plus - minus) / (2.0 * eps)
            rows.append((eps, analytic, fd, abs(fd - analytic) / max(abs(analytic), 1e-30)))
    return np.array(rows)


def worst_relative_error(rows: np.ndarray) -> float:
    """Smallest achievable relative error per direction, then the worst of those.

    Taking the best ``eps`` per direction first is deliberate: a direction whose derivative is tiny
    is dominated by truncation at a large step and by round-off at a small one, and penalising it
    for both would report a failure that is a property of the step size rather than of the gradient.
    A real gradient bug survives every step size, so it survives this reduction too.
    """
    out = []
    for analytic in np.unique(rows[:, 1]):
        sel = rows[rows[:, 1] == analytic]
        out.append(float(sel[:, 3].min()))
    return float(max(out)) if out else float('nan')


def format_report(rows: np.ndarray) -> str:
    """One line per (direction, eps), plus the reduced verdict."""
    lines = [f"{'eps':>10}  {'analytic':>16}  {'central FD':>16}  {'rel err':>10}"]
    for eps, analytic, fd, rel in rows:
        lines.append(f"{eps:10.1e}  {analytic:+16.8e}  {fd:+16.8e}  {rel:10.3e}")
    lines.append(f"worst-direction best-eps relative error: {worst_relative_error(rows):.3e}")
    return "\n".join(lines)
