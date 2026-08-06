"""Inverse design: a closed A4 sheet whose rigid-tile deployment matches a target outline.

Design variables are the per-cut aspect ratios of the collinearity LES, the
boundary sliders, and the single opening angle. Because the deployment is
closed-form (``shield_kinematics``) and the flat sheet tiles the rectangle exactly
by construction, both hard requirements -- machine-precision pin closure and a
gapless closed sheet -- hold at every iterate. The optimisation therefore carries
only the shape objective and optional manufacturability barriers.

The target's scale and position are free variables: the deployed size follows from
the conserved tile area, so it is an output of the design, not a constraint on it.
"""

from dataclasses import dataclass, field

import numpy as np
import jax
import jax.flatten_util
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float
from scipy.optimize import minimize

from nff.topology.closed_builder_jax import (
    build_deploy_structure, build_boundary_edges, boundary_flat_from_logits,
    solve_cut_vertices_jax)
from nff.closed.shield_kinematics import (
    build_deploy_plan, deploy_tiles, deployed_outline, tile_areas, polygon_area)


@dataclass
class ShieldDesignConfig:
    """Grid, sheet, and loss settings for one inverse-design run."""
    M: int = 28
    N: int = 40
    sheet_w: float = 210.0
    sheet_h: float = 297.0
    psi_max: float = 0.45
    r_lo: float = 0.06
    r_hi: float = 0.94
    n_sub: int = 6
    w_barrier: float = 0.0            # manufacturability: off by default
    l_edge_min: float = 4.0           # mm, inert while w_barrier == 0
    l_cut_min: float = 3.0            # mm, inert while w_barrier == 0
    w_degen: float = 0.0              # tile-area collapse guard
    a_min: float = 4.0                # mm^2
    w_quality: float = 0.0            # keep quads well-shaped rather than slivers
    w_uniform: float = 0.0            # keep tile areas near the grid average
    q_min: float = 0.30               # quad regularity floor, one-sided
    area_frac_min: float = 0.25       # tile area floor as a fraction of the mean
    w_homog: float = 0.0              # two-sided pull toward a uniform tessellation
    w_outside: float = 60.0           # keep every tile inside the target (near-constraint)
    n_tile_samples: int = 4           # k x k lattice per tile for the inside test
    w_precision: float = 1.0          # deployed outline -> target
    w_recall: float = 2.5             # target -> deployed outline; coverage matters more
    w_worst: float = 3.0              # soft-max on recall: no stretch of target may be abandoned
    tau_frac: float = 0.04            # soft-max temperature, as a fraction of the target radius
    outside_tol_frac: float = 0.30    # corners may reach out this fraction of a tile pitch
    w_centroid_outside: float = 4.0   # centroids are held hard: no tile may sit outside
    schedule: tuple = field(default_factory=lambda: (
        # (steps, learning_rate, barrier_fraction)
        (800, 3e-2, 0.0), (1500, 3e-2, 1.0), (700, 5e-3, 1.0)))

    @property
    def spacing(self) -> float:
        return self.sheet_w / self.M

    @property
    def spacing_y(self) -> float:
        return self.sheet_h / self.N


# ══════════════════════════════════════════════════════════════════════════════
# Forward map
# ══════════════════════════════════════════════════════════════════════════════

def init_params(cfg: ShieldDesignConfig, n_logits: int, seed: int = 0) -> dict:
    """Uniform-ish start: r ~ 0.79, uniform sliders, a partly opened mechanism."""
    rng = np.random.default_rng(seed)
    return {
        "z": jnp.asarray(0.8 + 0.02 * rng.normal(size=(cfg.M + 1, cfg.N + 1))),
        "bnd_logits": jnp.zeros(n_logits),
        "psi_raw": jnp.asarray(-0.4055),               # sigmoid -> psi ~ -0.30 at psi_max 0.45
        "log_scale": jnp.asarray(np.log(1.55 * cfg.sheet_h)),
        "target_shift": jnp.zeros(2),
    }


def build_forward(struct, bnd, plan, cfg: ShieldDesignConfig):
    """Design parameters to flat tiles, deployed tiles, and the deployed outline."""
    corner_pid = jnp.asarray(np.asarray(struct["corner_pid"]))
    half = 0.5 * (cfg.r_hi - cfg.r_lo)

    def forward_fn(params):
        r = 0.5 + half * jnp.tanh(params["z"])
        boundary_flat = boundary_flat_from_logits(bnd, params["bnd_logits"])
        coords = solve_cut_vertices_jax(struct, boundary_flat, r)
        flat_tiles = coords[corner_pid]

        psi = -cfg.psi_max * jax.nn.sigmoid(params["psi_raw"])
        deployed = deploy_tiles(plan, flat_tiles, psi)
        outline = deployed_outline(plan, deployed, cfg.n_sub)
        return {"coords": coords, "flat_tiles": flat_tiles, "deployed": deployed,
                "outline": outline, "psi": psi, "r": r}

    return forward_fn


# ══════════════════════════════════════════════════════════════════════════════
# Loss terms
# ══════════════════════════════════════════════════════════════════════════════

def _chamfer_sq(a: Float[Array, "n 2"], b: Float[Array, "m 2"]):
    """Squared nearest-neighbour distances both ways: (per-a, per-b)."""
    d2 = jnp.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1)
    return jnp.min(d2, axis=1), jnp.min(d2, axis=0)


def _soft_max(values, tau):
    """Smooth maximum. Averages let the optimiser abandon the few hard stretches of the
    target while scoring well on the easy ones; this makes the worst stretch expensive."""
    return tau * jax.nn.logsumexp(values / tau)


def _edge_lengths_sq(tiles: Float[Array, "n_tiles 4 2"]):
    """Squared length of every tile edge. Squared throughout -- norm() is NaN at zero."""
    d = tiles - jnp.roll(tiles, -1, axis=1)
    return jnp.sum(d ** 2, axis=-1).reshape(-1)


def _cut_separations_sq(coords, interior_base):
    """Squared distance between the two endpoints of each interior cut."""
    d = coords[interior_base] - coords[interior_base + 1]
    return jnp.sum(d ** 2, axis=-1)


def _quad_quality(tiles: Float[Array, "n_tiles 4 2"]):
    """Per-tile regularity in (0, 1]: ``4 * area / sum(edge^2)``, unity for a square.

    Squared throughout, so it stays finite as a quad collapses -- unlike any
    formulation that normalises by a length.
    """
    d = tiles - jnp.roll(tiles, -1, axis=1)
    perimeter_sq = jnp.sum(jnp.sum(d ** 2, axis=-1), axis=-1)
    return 4.0 * jnp.abs(tile_areas(tiles)) / (perimeter_sq + 1e-12)


def _soft_min_barrier(values_sq, limit: float):
    """One-sided quadratic barrier on squared lengths, normalised to be scale-free."""
    slack = jax.nn.relu(limit ** 2 - values_sq) / (limit ** 2)
    return jnp.mean(slack ** 2)


def _bilinear_sample(field, extent, points: Float[Array, "n 2"]):
    """Bilinear lookup into a regular grid; row 0 sits at ``y_max``. Clamped at the edges."""
    x_min, x_max, y_min, y_max = extent
    H, W = field.shape
    col = (points[:, 0] - x_min) / (x_max - x_min) * (W - 1)
    row = (y_max - points[:, 1]) / (y_max - y_min) * (H - 1)
    col = jnp.clip(col, 0.0, W - 1 - 1e-6)
    row = jnp.clip(row, 0.0, H - 1 - 1e-6)

    c0 = jnp.floor(col).astype(int)
    r0 = jnp.floor(row).astype(int)
    fc, fr = col - c0, row - r0
    f00 = field[r0, c0]
    f01 = field[r0, c0 + 1]
    f10 = field[r0 + 1, c0]
    f11 = field[r0 + 1, c0 + 1]
    return ((1 - fr) * ((1 - fc) * f00 + fc * f01)
            + fr * ((1 - fc) * f10 + fc * f11))


def _tile_sample_points(tiles: Float[Array, "n_tiles 4 2"], k: int = 4):
    """A k x k bilinear lattice over each quad, corners included."""
    t = jnp.linspace(0.0, 1.0, k)
    u, v = jnp.meshgrid(t, t, indexing="ij")
    u, v = u.reshape(-1), v.reshape(-1)
    bl, br, tr, tl = tiles[:, 0], tiles[:, 1], tiles[:, 2], tiles[:, 3]
    bottom = bl[:, None, :] * (1 - u)[None, :, None] + br[:, None, :] * u[None, :, None]
    top = tl[:, None, :] * (1 - u)[None, :, None] + tr[:, None, :] * u[None, :, None]
    return (bottom * (1 - v)[None, :, None] + top * v[None, :, None]).reshape(-1, 2)


def build_loss(forward_fn, target_unit, plan, cfg: ShieldDesignConfig, interior_base,
               sdf=None, sdf_extent=None):
    """Shape objective, the inside-the-target constraint, and homogeneity terms."""
    target_unit = jnp.asarray(target_unit)
    rms_unit = jnp.sqrt(jnp.mean(jnp.sum(target_unit ** 2, axis=1)))
    mean_tu = jnp.mean(target_unit, axis=0)
    sdf_j = None if sdf is None else jnp.asarray(sdf)

    def loss_fn(params, barrier_frac: float = 1.0):
        out = forward_fn(params)
        outline = out["outline"]
        centred = outline - jnp.mean(outline, axis=0)

        scale = jnp.exp(params["log_scale"])
        target = target_unit * scale
        target = target - jnp.mean(target, axis=0) + params["target_shift"]
        D = rms_unit * scale

        prec_d2, rec_d2 = _chamfer_sq(centred, target)
        precision, recall = jnp.mean(prec_d2), jnp.mean(rec_d2)
        tau = (cfg.tau_frac * D) ** 2
        worst_rec = _soft_max(rec_d2, tau)
        worst_prec = _soft_max(prec_d2, tau)
        shape = (cfg.w_precision * precision + cfg.w_recall * recall
                 + cfg.w_worst * (worst_rec + 0.5 * worst_prec)) / (D ** 2)

        e2 = _edge_lengths_sq(out["flat_tiles"])
        s2 = _cut_separations_sq(out["coords"], interior_base)
        barrier = (_soft_min_barrier(e2, cfg.l_edge_min)
                   + _soft_min_barrier(s2, cfg.l_cut_min))

        areas = jnp.abs(tile_areas(out["flat_tiles"]))
        degen = jnp.mean(jax.nn.relu(cfg.a_min - areas) ** 2) / (cfg.a_min ** 2)

        # One-sided floors: tiles may vary freely above them. A two-sided pull toward
        # a perfect square would recreate the uniform pattern, which deploys back into
        # a rectangle and cannot make a shield at all.
        quality = _quad_quality(out["flat_tiles"])
        irregularity = jnp.mean(jax.nn.relu(cfg.q_min - quality) ** 2) / (cfg.q_min ** 2)
        a_bar = jnp.mean(areas)
        nonuniformity = jnp.mean(jax.nn.relu(cfg.area_frac_min - areas / a_bar) ** 2) / (
            cfg.area_frac_min ** 2)

        # Homogeneity: two-sided, so tiles are pulled toward a common size and shape.
        # This directly opposes the shape objective -- a perfectly uniform sheet deploys
        # self-similarly -- so the weight sets a point on that Pareto, it is not free.
        homogeneity = (jnp.mean((areas / a_bar - 1.0) ** 2)
                       + jnp.mean((1.0 - quality) ** 2))

        # Every tile must land inside the target. Otherwise the optimiser parks boundary
        # tiles outside the silhouette, where they sample background white and vanish
        # against a white page: the outline scores well while the visible shape is faked.
        if sdf_j is None:
            outside = jnp.asarray(0.0)
            frac_outside = jnp.asarray(0.0)
            max_outside_mm = jnp.asarray(0.0)
            max_centroid_out_mm = jnp.asarray(0.0)
        else:
            offset = jnp.mean(outline, axis=0)

            def depth_of(p):
                unit = (p - offset - params["target_shift"]) / scale + mean_tu
                return _bilinear_sample(sdf_j, sdf_extent, unit) * scale

            # Corners get a tolerance of a fraction of a tile: without it the outline can
            # only ever approach the target from inside, never reach it, and coverage
            # is capped by the scallop depth.
            pitch = min(cfg.spacing, cfg.spacing_y)
            tol = cfg.outside_tol_frac * pitch
            over = jax.nn.relu(depth_of(_tile_sample_points(out["deployed"],
                                                            cfg.n_tile_samples)) - tol)
            # Centroids are held hard, so no tile can sit outside and hide as background.
            over_c = jax.nn.relu(depth_of(out["deployed"].mean(axis=1)))

            outside = (jnp.mean(over ** 2) + cfg.w_centroid_outside * jnp.mean(over_c ** 2)) / (D ** 2)
            frac_outside = jnp.mean(over > 1e-9)
            max_outside_mm = jnp.max(over)
            max_centroid_out_mm = jnp.max(over_c)

        total = (shape + barrier_frac * cfg.w_barrier * barrier + cfg.w_degen * degen
                 + cfg.w_quality * irregularity + cfg.w_uniform * nonuniformity
                 + cfg.w_homog * homogeneity + cfg.w_outside * outside)
        metrics = {
            "loss": total, "shape": shape, "barrier": barrier, "degen": degen,
            "irregularity": irregularity, "nonuniformity": nonuniformity,
            "homogeneity": homogeneity, "outside": outside,
            "frac_pts_outside": frac_outside, "max_outside_mm": max_outside_mm,
            "max_centroid_out_mm": max_centroid_out_mm,
            "worst_recall_mm": jnp.sqrt(jnp.max(rec_d2)),
            "p95_recall_mm": jnp.sqrt(jnp.percentile(rec_d2, 95)),
            "area_cv": jnp.std(areas) / a_bar,
            "min_quality": jnp.min(quality), "mean_quality": jnp.mean(quality),
            "precision_mm": jnp.sqrt(precision), "recall_mm": jnp.sqrt(recall),
            "rel_rms": jnp.sqrt(shape), "psi": out["psi"], "scale": scale,
            "min_edge_mm": jnp.sqrt(jnp.min(e2)), "min_cut_mm": jnp.sqrt(jnp.min(s2)),
            "min_area_mm2": jnp.min(areas), "hull_area": polygon_area(centred),
        }
        return total, metrics

    return loss_fn


# ══════════════════════════════════════════════════════════════════════════════
# Driver
# ══════════════════════════════════════════════════════════════════════════════

def optimize_shield(cfg: ShieldDesignConfig, target_unit, seed: int = 0,
                    verbose: bool = True, log_every: int = 100,
                    sdf=None, sdf_extent=None):
    """Run the staged Adam schedule, then an L-BFGS-B polish.

    Returns a dict with the optimised parameters, the forward output, the metric
    history, and the static structures needed to render the result.
    """
    struct = build_deploy_structure(cfg.M, cfg.N)
    bnd = build_boundary_edges(struct, cfg.spacing, cfg.spacing_y)
    plan = build_deploy_plan(struct, cfg.M, cfg.N, cfg.spacing, cfg.spacing_y)
    interior_base = _interior_cut_bases(struct)

    forward_fn = build_forward(struct, bnd, plan, cfg)
    loss_fn = build_loss(forward_fn, target_unit, plan, cfg, interior_base,
                         sdf=sdf, sdf_extent=sdf_extent)
    value_and_grad_fn = jax.jit(jax.value_and_grad(loss_fn, has_aux=True),
                                static_argnums=())

    params = init_params(cfg, bnd["n_logits"], seed)
    history = []

    step = 0
    for stage, (n_steps, lr, barrier_frac) in enumerate(cfg.schedule):
        optimizer = optax.adam(lr)
        opt_state = optimizer.init(params)

        @jax.jit
        def train_step_fn(params, opt_state, bf):
            (_, metrics), grads = value_and_grad_fn(params, bf)
            grads = jax.tree.map(lambda g: jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0),
                                 grads)
            updates, opt_state = optimizer.update(grads, opt_state)
            return optax.apply_updates(params, updates), opt_state, metrics

        for _ in range(n_steps):
            params, opt_state, metrics = train_step_fn(params, opt_state, barrier_frac)
            if step % log_every == 0:
                rec = {k: float(v) for k, v in metrics.items()}
                rec["step"] = step
                history.append(rec)
                if verbose:
                    print(f"  [{stage}] step {step:5d}  rel_rms {rec['rel_rms']:.4f}"
                          f"  prec {rec['precision_mm']:6.2f} mm  rec {rec['recall_mm']:6.2f} mm"
                          f"  worstRec {rec['worst_recall_mm']:6.2f} mm"
                          f"  p95Rec {rec['p95_recall_mm']:6.2f} mm"
                          f"  cOut {rec['max_centroid_out_mm']:5.2f}"
                          f"  areaCV {rec['area_cv']:.3f}  psi {rec['psi']:+.3f}", flush=True)
            step += 1

    if verbose:
        print("  L-BFGS-B polish ...", flush=True)
    params = _lbfgs_polish(params, loss_fn)

    _, metrics = loss_fn(params, 1.0)
    rec = {k: float(v) for k, v in metrics.items()}
    rec["step"] = step
    history.append(rec)
    if verbose:
        print(f"  final       rel_rms {rec['rel_rms']:.4f}  prec {rec['precision_mm']:.2f} mm"
              f"  rec {rec['recall_mm']:.2f} mm  outside {100 * rec['frac_pts_outside']:.2f}%"
              f"  max_out {rec['max_outside_mm']:.2f} mm  areaCV {rec['area_cv']:.3f}"
              f"  psi {rec['psi']:+.3f}", flush=True)

    return {"params": params, "forward": forward_fn(params), "metrics": rec,
            "history": history, "struct": struct, "bnd": bnd, "plan": plan,
            "cfg": cfg, "forward_fn": forward_fn, "loss_fn": loss_fn}


def _lbfgs_polish(params, loss_fn, maxiter: int = 300):
    """Flatten to a vector, run scipy L-BFGS-B with exact gradients, unflatten."""
    flat0, unravel = jax.flatten_util.ravel_pytree(params)
    grad_fn = jax.jit(jax.value_and_grad(lambda v: loss_fn(unravel(v), 1.0)[0]))

    def objective(v):
        val, grad = grad_fn(jnp.asarray(v))
        return float(val), np.asarray(grad, dtype=float)

    res = minimize(objective, np.asarray(flat0, dtype=float), jac=True,
                   method="L-BFGS-B", options={"maxiter": maxiter})
    return unravel(jnp.asarray(res.x))


def _interior_cut_bases(struct) -> np.ndarray:
    """Flat index of endpoint ``x`` for every interior cut (``T > 0``)."""
    T, cols = struct["T"], struct["cols"]
    idx = [2 * (i * cols + j) for i in range(T.shape[0]) for j in range(T.shape[1])
           if T[i, j] > 0]
    return np.array(idx, dtype=np.int64)
