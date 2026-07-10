"""Learned condensed hinge-energy surrogate ``W(u; g)``.

The hinge as a differentiable energy: given the relative-tile kinematics ``u = (a, s, theta)``
(mm, mm, rad) and the hinge geometry ``g = (w_lig, alpha)`` (mm, rad), return the stored
energy ``W`` [N.mm]. Its autodiff gradient ``dW/du`` is the internal force ``(F_a, F_s, M_theta)``
the Stage-2 solver balances; the pipeline differentiates this directly. Trained on the CalculiX
dataset (``data/fea/hinge_dataset.npz``) with a Sobolev (energy + force) loss.

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
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from nff.utils.linalg import corotated_bond_deformation

# u = (a, s, theta); geometry g = (w_lig, alpha[rad]) or (w_lig, alpha, fillet_ratio) for the swept
# 3rd DOF. Standardized network features: [a, s, theta, log(w_lig), alpha (, fillet)] -> 5-D or 6-D.
FEAT_DIM = 5


class HingeGeometry(NamedTuple):
    """Per-hinge geometry threaded to the Stage-2 solver -- all DESIGN-dependent and differentiable.

    Bundled (rather than three loose fields) so the design -> geometry map, the control_params
    threading, and the surrogate all speak ONE type. ``fillet_ratio`` is a fixed scalar config baked
    into the energy closure, so it is deliberately NOT here.

        w_lig:   (n_hinges,)    ligament width [mm]
        alpha:   (n_hinges,)    cut angle [rad]
        sec_dir: (n_hinges, 2)  axial (secondary-cut) unit direction
    """
    w_lig: Any
    alpha: Any
    sec_dir: Any


def w_lig_from_logit(logit):
    """Learnable ligament width in [1, 10] mm: ``1 + 9*sigmoid(logit)``. The single definition of the
    map_params `w_lig_logit` -> physical width (used by the loss and the visual deploys)."""
    return 1.0 + 9.0 * jax.nn.sigmoid(logit)


# ── input / target normalization ────────────────────────────────────────────────

def _robust_std(std):
    """Feature scale for standardization, robust to (near-)constant features.

    Following sklearn's StandardScaler convention, a zero-variance feature gets scale 1.0 -- its
    normalized input stays ~0 with a UNIT gradient -- instead of `std + eps` (~1e-8), which makes
    `(raw - mean)/std` and its gradient explode to 1e8 -> inf/NaN. This is not cosmetic: it bites the
    moment that feature is DIFFERENTIATED (e.g. the design -> (a, s) -> energy path in Phase 2b), where
    the 1/std blow-up overflows under XLA fusion even though the forward value looks finite. A dataset
    that fails to exercise a feature (e.g. a spine-only run with a=s=0) is the usual cause.
    """
    std = np.asarray(std)
    return np.where(std < 1e-6, 1.0, std)


def compute_norm_stats(a, s, theta, w_lig, alpha_rad, W, fillet_ratio=None) -> dict:
    """Standardization stats from the training arrays (static; closed over at inference).

    ``fillet_ratio`` optional: when given, the geometry input ``g`` is 3-D (w_lig, alpha, fillet) and
    the feature vector is 6-D; otherwise the legacy 2-D g / 5-D features. The chosen width is baked
    into ``feat_mean``/``feat_std`` length, so inference reads it back from the checkpoint.
    """
    cols = [a, s, theta, np.log(w_lig), alpha_rad]
    if fillet_ratio is not None:
        cols.append(fillet_ratio)
    feats = np.stack(cols, axis=-1)
    return dict(
        feat_mean=jnp.asarray(feats.mean(0), dtype=jnp.float64),
        feat_std=jnp.asarray(_robust_std(feats.std(0)), dtype=jnp.float64),
        # a positive energy scale so the net's ||dh||^2 ~ O(1) maps to physical N.mm
        W_scale=jnp.asarray(np.sqrt((W ** 2).mean()) + 1e-12, dtype=jnp.float64),
    )


def _features(u: Float[Array, "... 3"], g: Float[Array, "... _"], stats: dict) -> Float[Array, "... _"]:
    """Standardized features. Includes the fillet DOF ``g[...,2]`` iff the checkpoint's stats are 6-D."""
    a, s, th = u[..., 0], u[..., 1], u[..., 2]
    cols = [a, s, th, jnp.log(g[..., 0]), g[..., 1]]
    if stats["feat_mean"].shape[-1] >= 6:                    # 3rd geometry DOF: fillet_ratio
        cols.append(g[..., 2])
    raw = jnp.stack(cols, axis=-1)
    return (raw - stats["feat_mean"]) / stats["feat_std"]


def _feat6(stats: dict) -> bool:
    """Whether the checkpoint carries the swept fillet DOF (6-D features / 3-D geometry ``g``)."""
    return int(stats["feat_mean"].shape[-1]) >= 6


def _geom_vector(w_lig, alpha, fillet_ratio, stats: dict):
    """The surrogate geometry input ``g``: ``(w_lig, alpha)``, plus ``fillet_ratio`` iff 6-D checkpoint.

    THE single place geometry becomes the network's ``g`` (both the bond energy and the stability
    margin use it), so the 5-D/6-D split lives in exactly one function.
    """
    w_lig = jnp.asarray(w_lig, dtype=jnp.float64)
    cols = [w_lig, jnp.asarray(alpha, dtype=jnp.float64)]
    if _feat6(stats):
        cols.append(jnp.broadcast_to(jnp.asarray(fillet_ratio, dtype=jnp.float64), jnp.shape(w_lig)))
    return jnp.stack(cols, axis=-1)


def _unit(v):
    """Row-wise unit vector, safe for ~0 rows (closed hinges may pass a near-zero direction)."""
    v = jnp.asarray(v, dtype=jnp.float64)
    return v / (jnp.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


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
                         fail_hidden=(64, 64), feat_dim: int = FEAT_DIM) -> dict:
    """Parameter PyTree: the energy net ``h`` (R^feat_dim -> R^m) and the damage head.

    ``feat_dim`` = 5 (legacy 2-DOF geometry) or 6 (with the swept fillet DOF).
    """
    k_e, k_f = jax.random.split(key)
    return {
        "energy": _init_mlp(k_e, [feat_dim, *hidden, m_out]),
        "fail": _init_mlp(k_f, [feat_dim, *fail_hidden, 1]),
    }


def _mlp(x, layers):
    """Tanh MLP; last layer linear."""
    for W, b in layers[:-1]:
        x = jnp.tanh(x @ W + b)
    Wl, bl = layers[-1]
    return x @ Wl + bl


# ── energy + force + failure ──────────────────────────────────────────────────────

def apply_hinge_energy(params: dict, u: Float[Array, "... 3"], g: Float[Array, "... 2"],
                       stats: dict, h0=None) -> Float[Array, "..."]:
    """Condensed energy ``W(u; g)`` [N.mm]. Batched over the leading axes of ``u``/``g``.

    ``h0`` optionally supplies the precomputed zero-state embedding ``h(0, g)`` -- constant for a
    fixed geometry, so a Stage-2 solve can pass it once instead of recomputing it every evaluation
    (halves the MLP forward passes). ``None`` recomputes it (default, geometry-agnostic callers).
    """
    h_u = _mlp(_features(u, g, stats), params["energy"])
    if h0 is None:
        h0 = _mlp(_features(jnp.zeros_like(u), g, stats), params["energy"])   # same geometry, zero motion
    return stats["W_scale"] * jnp.sum((h_u - h0) ** 2, axis=-1)


def apply_hinge_force(params, u, g, stats):
    """Internal force ``dW/du = (F_a, F_s, M_theta)``. Per-sample grad, vmapped over the batch."""
    grad_one = jax.grad(lambda uu, gg: apply_hinge_energy(params, uu, gg, stats).squeeze())
    flat_u = u.reshape(-1, 3)
    flat_g = g.reshape(-1, g.shape[-1])   # 2 or 3 geom features (fillet DOF swept -> 3)
    F = jax.vmap(grad_one)(flat_u, flat_g)
    return F.reshape(u.shape)


def apply_hinge_failure(params, u, g, stats):
    """Continuous ductile DAMAGE ``D`` (>=0 via softplus). 0 = intact, >=1 = fractured.

    Trained on the triaxiality-based damage label (``nff.rve.damage.ductile_damage``); the old
    ``peeq_p99 / eps_f`` margin is the tension-only special case of it.
    """
    m = _mlp(_features(u, g, stats), params["fail"])[..., 0]
    return jax.nn.softplus(m)


# sharpness of the break barrier: softplus(_BREAK_SHARP*(D-1)) ~ 0 below fracture, linear above
_BREAK_SHARP = 8.0


def load_hinge_surrogate(path: str):
    """Load a trained surrogate checkpoint -> (net_params, stats, eps_f).

    Defensively re-floors ``feat_std`` (see ``_robust_std``) so older checkpoints trained before the
    floor — e.g. spine-only runs whose a/s variance was ~0 — cannot explode the normalization when a
    feature is differentiated. Physically such a checkpoint is still under-trained; this only keeps it
    numerically finite.
    """
    with open(path, "rb") as f:
        ck = pickle.load(f)
    stats = ck["stats"]
    stats = {**stats, "feat_std": jnp.asarray(_robust_std(stats["feat_std"]), dtype=jnp.float64)}
    return ck["params"], stats, ck.get("eps_f", 0.25)


def calibrate_scales(net_params, stats, *, alpha, w_lig, k_stretch, k_rot,
                     a_ref_frac: float = 0.1, theta_ref: float = 0.1, fillet_ratio: float = 0.16):
    """Gap-2 (length_scale, energy_scale) matching BOTH the pipeline's axial and rotational springs.

    The abstract pipeline is anisotropic (k_stretch >> k_rot enforces near-rigid translations, soft
    rotation), so one scalar cannot match it — matching only k_rot leaves translations far too soft
    and they blow out of domain. We co-solve for the two scales from the surrogate's small-strain
    axial and rotational stiffness at a representative hinge:

        energy_scale * k_theta            = k_rot        (theta is frame-free, no length_scale)
        energy_scale * length_scale^2 * k_axial = k_stretch

    =>  energy_scale = k_rot / k_theta ,   length_scale = sqrt(k_stretch*k_theta / (k_rot*k_axial)).

    This keeps translations near-rigid (in-domain) while rotation stays the soft deployment mode.
    ``k_stretch``/``k_rot`` are the pipeline's abstract stiffnesses (e.g. ``valid_state.k_*``).
    """
    a_m = float(jnp.mean(jnp.asarray(alpha)))
    w_m = float(jnp.mean(jnp.asarray(w_lig)))
    a_ref = a_ref_frac * w_m
    g = _geom_vector(jnp.array([w_m]), jnp.array([a_m]), fillet_ratio, stats)
    W_th = float(apply_hinge_energy(net_params, jnp.array([[0., 0., theta_ref]]), g, stats)[0])
    W_ax = float(apply_hinge_energy(net_params, jnp.array([[a_ref, 0., 0.]]), g, stats)[0])
    k_theta = 2.0 * W_th / (theta_ref ** 2) + 1e-30            # [N.mm/rad]
    k_axial = 2.0 * W_ax / (a_ref ** 2) + 1e-30               # [N/mm]
    energy_scale = float(k_rot) / k_theta
    length_scale = float(jnp.sqrt(k_stretch * k_theta / (k_rot * k_axial)))
    return length_scale, energy_scale


# ── pipeline adapter: the surrogate as a Stage-2 bond energy ───────────────────────

# training-box bounds (from the campaign), in the surrogate's normalized units, for the OOD barrier
DOMAIN = dict(eta_a_max=1.0, eta_s_max=0.7, theta_max=0.66)   # a/w_lig, s/w_lig, theta [rad]


def _domain_barrier(a, sh, dRot, w_lig, dom):
    """One-sided cubic (C^2) penalty (0 in-domain) making W COERCIVE outside the training box.

    The squared-form W saturates (tanh h), so it is not coercive: a load can push hinges out of
    the trustworthy region where W extrapolates arbitrarily and the Stage-2 minimizer runs off to
    infinity. This penalty grows as ||u|| leaves the box, guaranteeing a bounded minimizer AND
    acting as the validity barrier that keeps the solve in the domain (brief section 10). Tension
    only: a < 0 (compression) is out of domain."""
    r = lambda x: jnp.maximum(x, 0.0) ** 3     # C^2 (was **2 = C^1: jump in 2nd deriv at boundary
                                               # -> stiffness-matrix jumps in the IFT backward solve)
    eta_a, eta_s = a / w_lig, sh / w_lig
    return (r(eta_a - dom["eta_a_max"]) + r(-eta_a)
            + r(jnp.abs(eta_s) - dom["eta_s_max"])
            + r(jnp.abs(dRot) - dom["theta_max"]))


def hinge_kinematics(nodal_DOFs, sec, length_scale, reference_vector=None):
    """Project the two connected face DOFs -> per-hinge ``(a, s, theta)`` in the COROTATED cut frame.

    Uses the SHARED frame-invariant reduction ``corotated_bond_deformation`` (same core as the ROM's
    ``ligament_strains_linearized``): the current bond is corotated by the mean face rotation and the
    reference is subtracted, so the result is exactly zero for any rigid-body motion. The surrogate's
    own normalization then differs from the ROM's only in the final step -- it projects onto the
    DESIGN cut frame (``sec`` = axial unit, its 90-deg rotation = shear) and scales to mm by
    ``length_scale`` (NO ``1/||ref||`` -- closed hinges have ``||ref||~0``).

    Shared by the Stage-2 bond energy and the design-loss stability terms so the ``(a, s, theta)``
    mapping is byte-identical between the forward solve and the loss.

    Args:
        nodal_DOFs: ``(DOFs1, DOFs2)`` each (n, 3) node DOFs of the two connected faces.
        sec: (n, 2) unit axial (secondary-cut) directions.
        length_scale: mm per pipeline length-unit.
        reference_vector: (n, 2) rest bond vector (node2 - node1). ``None`` -> zero (closed hinge);
            passing it makes ``(a, s)`` frame-invariant for OPEN bonds / net-rotating deployments.
    """
    DOFs1, DOFs2 = nodal_DOFs
    if reference_vector is None:
        reference_vector = jnp.zeros_like(DOFs1[..., :2])
    deformation, dRot = corotated_bond_deformation(DOFs1, DOFs2, reference_vector)
    a = (deformation[..., 0] * sec[..., 0] + deformation[..., 1] * sec[..., 1]) * length_scale     # axial [mm]
    sh = (deformation[..., 0] * (-sec[..., 1]) + deformation[..., 1] * sec[..., 0]) * length_scale  # shear [mm]
    return a, sh, dRot


def build_hinge_bond_energy_fn(net_params, stats, *, length_scale: float = 1.0,
                               energy_scale: float = 1.0, domain: dict = DOMAIN,
                               barrier: float = 0.0, fillet_ratio=None):
    """Wrap the trained surrogate as a bond-energy fn for the Stage-2 solver.

    The per-hinge ``HingeGeometry`` (w_lig, alpha, sec_dir) is supplied PER CALL via the solver's
    ``control_params`` (``bond_params.hinge_geometry``), NOT closed over, so jaxopt's implicit diff
    carries ``d/d(design)``. From the two connected face node-DOFs it (1) forms the relative tile
    motion, (2) PROJECTS it onto the cut frame -- axial ``a`` along ``sec_dir``, shear ``s``
    perpendicular -- COROTATED by the mean face rotation, (3) converts to mm via ``length_scale``,
    (4) evaluates ``W(a, s, theta; w_lig, alpha[, fillet])`` and rescales via ``energy_scale``,
    (5) adds ``barrier`` * OOD penalty (set ``barrier > 0`` or the solve can diverge out of domain).
    Ignores ``k_stretch/k_shear/k_rot`` -- the surrogate replaces the springs.

    Returns ``bond_energy(nodal_DOFs, reference_vector=None, hinge_geometry, **kwargs) -> (n_hinges,)``.
    """
    fr = jnp.asarray(fillet_ratio if fillet_ratio is not None else 0.16, dtype=jnp.float64)

    def bond_energy(nodal_DOFs, reference_vector=None, hinge_geometry=None, **kwargs):
        geo = hinge_geometry
        a, sh, dRot = hinge_kinematics(nodal_DOFs, _unit(geo.sec_dir), length_scale,
                                       reference_vector=reference_vector)
        u = jnp.stack([a, sh, dRot], axis=-1)
        W = energy_scale * apply_hinge_energy(net_params, u, _geom_vector(geo.w_lig, geo.alpha, fr, stats), stats)
        if barrier:
            W = W + barrier * _domain_barrier(a, sh, dRot, geo.w_lig, domain)
        return W

    return bond_energy


def build_hinge_stability_fn(net_params, stats, *, bond_pairs,
                             length_scale: float = 1.0, domain: dict = DOMAIN,
                             w_damage: float = 0.0, w_fail: float = 0.0, w_ood: float = 0.0,
                             m_safe: float = 1.0, fail_line: float = 1.0, fillet_ratio=None):
    """Damage / stability penalty at the DEPLOYED state, for the design loss:

        w_damage * mean(D^2)                          (reduce ductile damage on EVERY hinge)
      + w_fail   * sum softplus(K*(D - m_safe))/K     (legacy one-sided break barrier; 0 = off)
      + w_ood    * sum domain_barrier(u)              (hinges leaving the trustworthy training box)

    ``D`` is the surrogate's continuous ductile-damage prediction. The CalculiX ductile-damage law it
    was trained on calls ``D=1`` fracture-INITIATION, but that is a modeling choice with real calibration
    slack -- a thin plastic-hinge ligament can micro-crack and still fold -- so the PRIMARY term
    ``w_damage * mean(D^2)`` bakes NO failure threshold into the gradient: it just pushes every hinge's
    damage down, weighting the worst hinges (grip concentration) hardest, decoupled from where physical
    failure actually is. ``fail_line`` is a REPORTING overlay ONLY (the count of hinges above it) --
    calibrate it against the real printed experiment; it never enters the loss. The legacy ``w_fail``
    softplus barrier (onset ``m_safe``) is kept for older configs.

    ``u = (a, s, theta)`` is the SAME corotated projection the bond energy uses (``hinge_kinematics``).
    The per-hinge ``HingeGeometry`` is passed PER CALL (same design-tracked value as the bond energy).
    ``bond_pairs`` (n_hinges, 2): the two connected NODES per hinge (= Stage-2 bond_connectivity); the
    caller MUST pass NODE displacements (face displacements mapped through the rigid-tile kinematics).

    Returns ``fn(node_displacements, hinge_geometry, reference_vectors) -> (penalty, aux)``, or ``None``
    if all weights are 0.
    """
    if w_damage == 0.0 and w_fail == 0.0 and w_ood == 0.0:
        return None
    fi = jnp.asarray(np.asarray(bond_pairs)[:, 0])
    fj = jnp.asarray(np.asarray(bond_pairs)[:, 1])
    fr = jnp.asarray(fillet_ratio if fillet_ratio is not None else 0.16, dtype=jnp.float64)

    def stability(node_displacements, hinge_geometry, reference_vectors=None):
        # ``reference_vectors`` (per-hinge rest bond) makes the margin's (a, s) frame-invariant,
        # exactly matching the bond energy; None -> zero (closed-hinge fast path, identical result).
        geo = hinge_geometry
        a, sh, dRot = hinge_kinematics((node_displacements[fi], node_displacements[fj]),
                                       _unit(geo.sec_dir), length_scale, reference_vector=reference_vectors)
        u = jnp.stack([a, sh, dRot], axis=-1)
        gg = _geom_vector(geo.w_lig, geo.alpha, fr, stats)
        D = apply_hinge_failure(net_params, u, gg, stats)     # continuous ductile damage (>=0)
        damage_pen = w_damage * jnp.mean(D ** 2)              # reduce damage on ALL hinges, threshold-free
        break_pen = w_fail * jnp.sum(jax.nn.softplus(_BREAK_SHARP * (D - m_safe)) / _BREAK_SHARP)
        ood_pen = w_ood * jnp.sum(_domain_barrier(a, sh, dRot, geo.w_lig, domain))
        aux = {"stab_damage": damage_pen, "stab_fail": break_pen, "stab_ood": ood_pen,
               "hinge_max_D": jnp.max(D), "hinge_mean_D": jnp.mean(D),
               "hinge_p90_D": jnp.percentile(D, 90.0),
               "hinge_n_over": jnp.sum((D >= fail_line).astype(jnp.float64))}  # display line only
        return damage_pen + break_pen + ood_pen, aux

    return stability


def build_hinge_damage_fn(net_params, stats, *, bond_pairs, length_scale: float = 1.0,
                          fillet_ratio=None):
    """Per-hinge ductile damage ``D`` at a deployed state, in bond order -- for diagnostics/visuals.

    Same kinematics + geometry as the bond energy / stability term, but returns the raw per-hinge ``D``
    (n_hinges,) instead of a scalar penalty. ``bond_pairs`` = Stage-2 bond_connectivity (node pairs), so
    the returned order matches the bond-energy hinges (and the same node indices place the dots).
    """
    fi = jnp.asarray(np.asarray(bond_pairs)[:, 0])
    fj = jnp.asarray(np.asarray(bond_pairs)[:, 1])
    fr = jnp.asarray(fillet_ratio if fillet_ratio is not None else 0.16, dtype=jnp.float64)

    def damage(node_displacements, hinge_geometry, reference_vectors=None):
        geo = hinge_geometry
        a, sh, dRot = hinge_kinematics((node_displacements[fi], node_displacements[fj]),
                                       _unit(geo.sec_dir), length_scale, reference_vector=reference_vectors)
        u = jnp.stack([a, sh, dRot], axis=-1)
        return apply_hinge_failure(net_params, u, _geom_vector(geo.w_lig, geo.alpha, fr, stats), stats)

    return damage


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
