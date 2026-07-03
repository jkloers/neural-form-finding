"""Learned condensed hinge-energy surrogate ``W(u; g)``.

The hinge as a differentiable energy: given the relative-tile kinematics ``u = (a, s, theta)``
(mm, mm, rad) and the hinge geometry ``g = (w_lig, alpha)`` (mm, rad), return the stored
energy ``W`` [N.mm]. Its autodiff gradient ``dW/du`` is the internal force ``(F_a, F_s, M_theta)``
the Stage-2 solver balances; the pipeline differentiates this directly. Trained on the CalculiX
dataset (``sofa/output/hinge_dataset.npz``) with a Sobolev (energy + force) loss.

Architecture (locked 2026-07-03) -- the SQUARED energy form:

    W(u; g) = W_scale * || h(u, g) - h(0, g) ||^2 ,   h : R^5 -> R^m  (tanh MLP)

which gives, BY CONSTRUCTION and with no penalty terms:
  * W >= 0                          (norm squared) -- the global minimizer cannot exploit a
                                     spurious negative-energy state,
  * W(0, g) = 0, dW/du(0, g) = 0     (rigid-body motion is free), exactly,
  * C^2 in u                        (tanh) -- the L-BFGS forward solve + IFT backward solve
                                     need a well-behaved tangent stiffness.

Why NOT an additive analytic elastic term ``1/2 u^T H u``: the data's W is ~40-80x softer than
beam/membrane theory (the ligament relieves in-plane load out-of-plane), so an analytic H would
be far too stiff; and the true elastic stiffness governs only a sub-yield sliver (steel yields
by ~0.6 deg). The unavoidable quadratic flattening at the origin -- intrinsic to ANY W>=0 energy
with a zero there -- only weakens identification of that (negligible) origin curvature, so we
accept it and keep the simplest robust form.

A separate small head predicts the ductile-failure margin ``peeq_p99 / eps_f`` (the validity
barrier), kept distinct so the hard failure boundary does not distort the smooth W.

Follows the repo ``init_*`` / ``apply_*`` raw-JAX convention (no flax), float64.
"""

import pickle

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

# u = (a, s, theta); g = (w_lig, alpha[rad]). Network features (standardized):
#   [a, s, theta, log(w_lig), alpha]
FEAT_DIM = 5


# ── input / target normalization ────────────────────────────────────────────────

def compute_norm_stats(a, s, theta, w_lig, alpha_rad, W) -> dict:
    """Standardization stats from the training arrays (static; closed over at inference)."""
    feats = np.stack([a, s, theta, np.log(w_lig), alpha_rad], axis=-1)
    return dict(
        feat_mean=jnp.asarray(feats.mean(0), dtype=jnp.float64),
        feat_std=jnp.asarray(feats.std(0) + 1e-8, dtype=jnp.float64),
        # a positive energy scale so the net's ||dh||^2 ~ O(1) maps to physical N.mm
        W_scale=jnp.asarray(np.sqrt((W ** 2).mean()) + 1e-12, dtype=jnp.float64),
    )


def _features(u: Float[Array, "... 3"], g: Float[Array, "... 2"], stats: dict) -> Float[Array, "... 5"]:
    a, s, th = u[..., 0], u[..., 1], u[..., 2]
    w_lig, alpha = g[..., 0], g[..., 1]
    raw = jnp.stack([a, s, th, jnp.log(w_lig), alpha], axis=-1)
    return (raw - stats["feat_mean"]) / stats["feat_std"]


# ── parameters ──────────────────────────────────────────────────────────────────

def _init_mlp(key, sizes, scale=0.1):
    """List of (W, b) for a tanh MLP with the given layer sizes (float64)."""
    keys = jax.random.split(key, len(sizes) - 1)
    layers = []
    for k, din, dout in zip(keys, sizes[:-1], sizes[1:]):
        W = jax.random.normal(k, (din, dout), dtype=jnp.float64) * (scale / np.sqrt(din))
        layers.append((W, jnp.zeros(dout, dtype=jnp.float64)))
    return layers


def init_hinge_surrogate(key: jax.Array, hidden=(128, 128, 128), m_out: int = 32,
                         fail_hidden=(64, 64)) -> dict:
    """Parameter PyTree: the energy net ``h`` (R^5 -> R^m) and the failure-margin head."""
    k_e, k_f = jax.random.split(key)
    return {
        "energy": _init_mlp(k_e, [FEAT_DIM, *hidden, m_out]),
        "fail": _init_mlp(k_f, [FEAT_DIM, *fail_hidden, 1]),
    }


def _mlp(x, layers):
    """Tanh MLP; last layer linear."""
    for W, b in layers[:-1]:
        x = jnp.tanh(x @ W + b)
    Wl, bl = layers[-1]
    return x @ Wl + bl


# ── energy + force + failure ──────────────────────────────────────────────────────

def apply_hinge_energy(params: dict, u: Float[Array, "... 3"], g: Float[Array, "... 2"],
                       stats: dict) -> Float[Array, "..."]:
    """Condensed energy ``W(u; g)`` [N.mm]. Batched over the leading axes of ``u``/``g``."""
    z_u = _features(u, g, stats)
    z_0 = _features(jnp.zeros_like(u), g, stats)            # same geometry, zero motion
    dh = _mlp(z_u, params["energy"]) - _mlp(z_0, params["energy"])
    return stats["W_scale"] * jnp.sum(dh ** 2, axis=-1)


def apply_hinge_force(params, u, g, stats):
    """Internal force ``dW/du = (F_a, F_s, M_theta)``. Per-sample grad, vmapped over the batch."""
    grad_one = jax.grad(lambda uu, gg: apply_hinge_energy(params, uu, gg, stats).squeeze())
    flat_u = u.reshape(-1, 3)
    flat_g = g.reshape(-1, 2)
    F = jax.vmap(grad_one)(flat_u, flat_g)
    return F.reshape(u.shape)


def apply_hinge_failure(params, u, g, stats):
    """Ductile-failure margin ``peeq_p99 / eps_f`` (>=0 via softplus). >=1 => past fracture."""
    m = _mlp(_features(u, g, stats), params["fail"])[..., 0]
    return jax.nn.softplus(m)


def build_hinge_energy_fn(params: dict, stats: dict):
    """Pure ``energy_fn(u, g) -> W`` for pipeline integration (JAX-differentiable, jittable)."""
    def energy_fn(u, g):
        return apply_hinge_energy(params, u, g, stats)
    return energy_fn


def load_hinge_surrogate(path: str):
    """Load a trained surrogate checkpoint -> (net_params, stats, eps_f)."""
    with open(path, "rb") as f:
        ck = pickle.load(f)
    return ck["params"], ck["stats"], ck.get("eps_f", 0.25)


# ── pipeline adapter: the surrogate as a Stage-2 bond energy ───────────────────────

def build_hinge_bond_energy_fn(net_params, stats, *, alpha, w_lig, sec_dir,
                               length_scale: float = 1.0, energy_scale: float = 1.0):
    """Wrap the trained surrogate as a bond-energy fn for the Stage-2 solver.

    From the two connected face node-DOFs it (1) forms the relative tile motion, (2) PROJECTS it
    onto the hinge's cut frame -- axial ``a`` along ``sec_dir``, shear ``s`` perpendicular --
    COROTATED by the mean face rotation, (3) converts to physical mm via ``length_scale``,
    (4) evaluates ``W(a, s, theta; w_lig, alpha)`` and rescales to pipeline energy via
    ``energy_scale``. Linear projection (not ligament_strains' length/angle) matches how the RVE
    imposed ``(a, s)``. Ignores ``k_stretch/k_shear/k_rot`` -- the surrogate replaces the springs.

    Args:
        net_params, stats: from ``load_hinge_surrogate``.
        alpha:      (n_hinges,) cut angle [rad], per-hinge constant (from compute_hinge_descriptors).
        w_lig:      (n_hinges,) ligament width [mm].
        sec_dir:    (n_hinges, 2) unit secondary-cut (axial) direction, oriented opening-positive.
        length_scale: mm per pipeline length-unit (Gap 2).
        energy_scale: pipeline-energy per N.mm (Gap 2).

    Returns:
        ``bond_energy(nodal_DOFs, reference_vector=None, **kwargs) -> (n_hinges,)`` [pipeline units].
    """
    alpha = jnp.asarray(alpha, dtype=jnp.float64)
    w_lig = jnp.asarray(w_lig, dtype=jnp.float64)
    sec = jnp.asarray(sec_dir, dtype=jnp.float64)
    sec = sec / (jnp.linalg.norm(sec, axis=-1, keepdims=True) + 1e-12)

    def bond_energy(nodal_DOFs, reference_vector=None, **kwargs):
        DOFs1, DOFs2 = nodal_DOFs
        dU = DOFs2[..., :2] - DOFs1[..., :2]                    # relative tile translation
        dRot = DOFs2[..., 2] - DOFs1[..., 2]
        mean_rot = 0.5 * (DOFs1[..., 2] + DOFs2[..., 2])
        c, s = jnp.cos(mean_rot), jnp.sin(mean_rot)
        ux = c * sec[:, 0] - s * sec[:, 1]                      # corotated axial unit
        uy = s * sec[:, 0] + c * sec[:, 1]
        a = (dU[..., 0] * ux + dU[..., 1] * uy) * length_scale          # axial [mm]
        sh = (dU[..., 0] * (-uy) + dU[..., 1] * ux) * length_scale      # shear (perp) [mm]
        u = jnp.stack([a, sh, dRot], axis=-1)
        g = jnp.stack([w_lig, alpha], axis=-1)
        return energy_scale * apply_hinge_energy(net_params, u, g, stats)

    return bond_energy


# ── training loss ─────────────────────────────────────────────────────────────────

def sobolev_loss(params, batch, stats, lam: float = 0.8, w_fail: float = 0.1):
    """lambda-weighted Sobolev loss (energy-priority) + failure-margin term.

    L = lam * ||W - W*||^2 / sigma_W^2  +  (1 - lam) * ||dW/du - F*||^2 / sigma_F^2
        + w_fail * ||margin - margin*||^2

    Both Sobolev terms are variance-normalized so ``lam`` is a clean priority knob (energy
    matters for the loss regularizers; force sets where the solver's equilibrium lands, so it
    is kept, not dropped). ``batch`` holds u, g, W, F=(F_a,F_s,M_theta), and margin=peeq_p99/eps_f.
    """
    u, g = batch["u"], batch["g"]
    W_pred = apply_hinge_energy(params, u, g, stats)
    F_pred = apply_hinge_force(params, u, g, stats)

    e_W = jnp.mean((W_pred - batch["W"]) ** 2) / (jnp.var(batch["W"]) + 1e-12)
    e_F = jnp.mean((F_pred - batch["F"]) ** 2) / (jnp.var(batch["F"]) + 1e-12)
    e_fail = jnp.mean((apply_hinge_failure(params, u, g, stats) - batch["margin"]) ** 2)

    loss = lam * e_W + (1.0 - lam) * e_F + w_fail * e_fail
    return loss, dict(energy=e_W, force=e_F, failure=e_fail)
