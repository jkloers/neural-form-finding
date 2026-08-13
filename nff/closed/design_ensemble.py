"""Multi-start ensembles for the kinematic closed-sheet inverse design.

One target, many random starting cut patterns, the same solve from each. The question is
whether the target determines the design, and three outcomes have to be told apart:

    same shape error, same design        the inverse problem is well posed
    same shape error, DIFFERENT design   a flat solution set: the target underdetermines it
    different shape error                genuine bad local minima

so every record keeps the converged DESIGN, not only its score.

Design distance is measured on flat cut-vertex coordinates in mm and quotiented by the cut
family's own symmetry group. Skipping that quotient would report a design and its own
relabelled image as maximally different.
"""

from dataclasses import replace

import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np
from jaxtyping import Float
from scipy.optimize import linear_sum_assignment

from nff.closed.analytic_targets import circle_outline_unit
from nff.closed.shield_design import (ShieldDesignConfig, build_forward, init_params,
                                      optimize_shield)
from nff.closed.shield_kinematics import build_deploy_plan, pin_residual, tile_areas
from nff.topology.closed_builder_jax import build_boundary_edges, build_deploy_structure

# The rotations of the square. Reflections are deliberately absent: the RDPQK cut family is
# CHIRAL, so a mirrored pattern is not a relabelling of the original. Measured on the uniform
# reference, the rotations close to 1e-12 mm while every reflection leaves a 308 mm residual,
# which `r -> 1-r` does not repair either (352 mm). `symmetry_closure` re-checks this on the
# ensemble itself rather than trusting the claim.
ROTATIONS = {
    "e":    ((1.0, 0.0), (0.0, 1.0)),
    "r90":  ((0.0, -1.0), (1.0, 0.0)),
    "r180": ((-1.0, 0.0), (0.0, -1.0)),
    "r270": ((0.0, 1.0), (-1.0, 0.0)),
}
REFLECTIONS = {
    "mx": ((-1.0, 0.0), (0.0, 1.0)),
    "my": ((1.0, 0.0), (0.0, -1.0)),
    "d":  ((0.0, 1.0), (1.0, 0.0)),
    "ad": ((0.0, -1.0), (-1.0, 0.0)),
}


# ══════════════════════════════════════════════════════════════════════════════
# The problem
# ══════════════════════════════════════════════════════════════════════════════

def circle_problem(M: int = 7, N: int = 7, sheet_mm: float = 2844.8,
                   radius_mm: float = 2171.0, steps: int = 10000) -> ShieldDesignConfig:
    """The Fig. 5(d) task, deployed kinematically: a square closed sheet onto a fixed circle.

    The circle is fixed a priori, as in the figure. Its radius is the size the design
    reaches on its own -- calibrated by running with a free scale, whose median over random
    starts was 2171 mm -- so the task is "be exactly circular at your natural size" rather
    than "be circular AND contract", which trades regularity against match along one Pareto
    line instead of resolving it.

    ``w_outside`` is off. The shield's inside-constraint exists to stop tiles hiding outside
    a silhouette; here the target sits at the mean deployed radius, so roughly half the
    outline must lie outside it and the constraint would make the target unreachable.
    """
    return ShieldDesignConfig(
        M=M, N=N, sheet_w=sheet_mm, sheet_h=sheet_mm,
        psi_max=0.80,                       # full rotating-squares opening is 0.785 rad
        w_outside=0.0,
        w_quality=5.0, w_uniform=5.0, q_min=0.30, area_frac_min=0.25,
        w_homog=0.0,
        fixed_scale=2.0 * radius_mm,        # target_unit has unit HEIGHT, so scale = diameter
        fixed_shift=(0.0, 0.0),
        schedule=((steps // 2, 1e-1, 0.0), (steps - steps // 2, 1e-2, 1.0)),
    )


def problem_statics(cfg: ShieldDesignConfig):
    """Topology, boundary sliders, deployment plan and forward map -- built once per grid."""
    struct = build_deploy_structure(cfg.M, cfg.N)
    bnd = build_boundary_edges(struct, cfg.spacing, cfg.spacing_y)
    plan = build_deploy_plan(struct, cfg.M, cfg.N, cfg.spacing, cfg.spacing_y)
    return {"struct": struct, "bnd": bnd, "plan": plan,
            "forward_fn": build_forward(struct, bnd, plan, cfg),
            "centre": np.array([cfg.sheet_w / 2.0, cfg.sheet_h / 2.0])}


# ══════════════════════════════════════════════════════════════════════════════
# Design vector and its symmetry quotient
# ══════════════════════════════════════════════════════════════════════════════

def design_coords(forward_out, statics) -> Float[np.ndarray, "n_nodes 2"]:
    """Flat cut vertices in mm, centred on the sheet.

    The design is compared as geometry, not as latents: ``z`` and ``bnd_logits`` are chart
    coordinates, and a distance in them has no length scale.
    """
    return np.asarray(forward_out["coords"]) - statics["centre"]


def _match(a: Float[np.ndarray, "n 2"], b: Float[np.ndarray, "n 2"]):
    """Bijection from ``a`` onto ``b`` minimising total squared displacement.

    Hungarian rather than nearest-neighbour: the two cut vertices of a boundary node are
    exactly coincident, so a greedy match is not a permutation there.
    """
    cost = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1)
    rows, cols = linear_sum_assignment(cost)
    return cols, float(np.sqrt(cost[rows, cols].max()))


def build_symmetry(cfg: ShieldDesignConfig, statics, include_reflections: bool = False,
                   tol_mm: float = 1e-6) -> dict:
    """Index relabelling for every symmetry, derived from the uniform reference pattern.

    A symmetry is a rigid map of the plane that carries the cut pattern onto itself; the
    permutation it induces on cut-vertex indices is what makes two designs comparable. It is
    read off the uniform design, which is the most symmetric member of the family, and each
    candidate is REJECTED if the map does not close on it to ``tol_mm``.

    Returns:
        {name: (2x2 matrix, permutation)} for the maps that survive.
    """
    ref = design_coords(statics["forward_fn"](
        init_params(cfg, statics["bnd"]["n_logits"], seed=0, z_sigma=0.0, z_mean=0.8)),
        statics)
    candidates = dict(ROTATIONS)
    if include_reflections:
        candidates.update(REFLECTIONS)

    group = {}
    for name, mat in candidates.items():
        matrix = np.asarray(mat, dtype=float)
        perm, resid = _match(ref @ matrix.T, ref)
        if resid <= tol_mm:
            group[name] = (matrix, perm)
    if "e" not in group:
        raise RuntimeError("the identity failed to close on the reference pattern")
    return group


def apply_symmetry(coords: Float[np.ndarray, "n 2"], element) -> Float[np.ndarray, "n 2"]:
    """Rotate a design and relabel its vertices, giving the same design in a new frame."""
    matrix, perm = element
    return (coords @ matrix.T)[np.argsort(perm)]


def canonical_distance(a: Float[np.ndarray, "n 2"], b: Float[np.ndarray, "n 2"],
                       group: dict) -> float:
    """RMS cut-vertex displacement in mm, minimised over the symmetry group.

    Same quantity as ``chart_ablation.rms_displacement``, in mm and quotiented.
    """
    return min(float(np.sqrt(np.mean(np.sum((a - apply_symmetry(b, el)) ** 2, axis=-1))))
               for el in group.values())


def distance_matrix(designs, group: dict) -> Float[np.ndarray, "n n"]:
    """Symmetric pairwise canonical distances."""
    n = len(designs)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = canonical_distance(designs[i], designs[j], group)
    return D


def split_families(D: Float[np.ndarray, "n n"], k: int = 2):
    """Cluster the ensemble on its own canonical distances.

    Independent starts need not land in one basin, and a covariance pooled over several is a
    statistic of the gap between them rather than of any design set. Everything local -- the
    spread a Hessian at one design can explain -- has to be measured inside a single family.

    Returns:
        (labels, report) with labels in 0..k-1 ordered by descending family size.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    if len(D) <= k:
        return np.zeros(len(D), dtype=int), {"n_families": 1}
    raw = fcluster(linkage(squareform(D, checks=False), method="average"), k,
                   criterion="maxclust")
    order = sorted(range(1, k + 1), key=lambda c: -(raw == c).sum())
    labels = np.zeros(len(D), dtype=int)
    for new, old in enumerate(order):
        labels[raw == old] = new

    within, sizes = [], []
    for c in range(k):
        m = labels == c
        sizes.append(int(m.sum()))
        sub = D[np.ix_(m, m)]
        within.append(float(sub[np.triu_indices(len(sub), 1)].mean()) if m.sum() > 1 else 0.0)
    between = [[float(D[np.ix_(labels == a, labels == b)].mean()) for b in range(k)]
               for a in range(k)]
    off = [between[a][b] for a in range(k) for b in range(k) if a != b]
    return labels, {
        "n_families": k, "sizes": sizes, "within_mean_mm": within,
        "between_mean_mm": between,
        "separation_ratio": float(min(off) / max(max(within), 1e-12)) if off else float("nan"),
    }


def symmetry_closure(designs, cfg, statics) -> dict:
    """Does adding reflections collapse any pair of converged designs?

    The chiral claim is checked against the data instead of being assumed: if mirrored
    designs were realisable by this chart, some pair would sit near zero under the enlarged
    group and far from zero under the rotations alone.
    """
    rot = build_symmetry(cfg, statics, include_reflections=False)
    full = dict(rot)
    ref = design_coords(statics["forward_fn"](
        init_params(cfg, statics["bnd"]["n_logits"], seed=0, z_sigma=0.0, z_mean=0.8)), statics)
    for name, mat in REFLECTIONS.items():          # forced in, WITHOUT the closure test
        matrix = np.asarray(mat, dtype=float)
        perm, resid = _match(ref @ matrix.T, ref)
        full[name] = (matrix, perm)
        rot[f"_resid_{name}"] = resid
    d_rot = distance_matrix(designs, {k: v for k, v in rot.items() if not k.startswith("_")})
    d_full = distance_matrix(designs, full)
    off = ~np.eye(len(designs), dtype=bool)
    return {
        "reflection_reference_residual_mm": {n: rot[f"_resid_{n}"] for n in REFLECTIONS},
        "min_pair_rotations_mm": float(d_rot[off].min()),
        "min_pair_with_reflections_mm": float(d_full[off].min()),
        "median_ratio": float(np.median(d_full[off] / np.maximum(d_rot[off], 1e-12))),
    }


# ══════════════════════════════════════════════════════════════════════════════
# One run, and the ensemble
# ══════════════════════════════════════════════════════════════════════════════

def draw_start(cfg: ShieldDesignConfig, statics, seed: int, sigma: float) -> dict:
    """A random starting cut pattern. Valid by construction, so no draw is ever rejected."""
    return init_params(cfg, statics["bnd"]["n_logits"], seed=seed,
                       z_mean=0.0, z_sigma=sigma, bnd_sigma=0.5 * sigma,
                       psi_sigma=0.3 * sigma)


def run_one(cfg: ShieldDesignConfig, statics, target_unit, seed: int, sigma: float,
            snapshot_every=None) -> dict:
    """One start to one converged design, with the geometry audited independently."""
    start = draw_start(cfg, statics, seed, sigma)
    start_out = statics["forward_fn"](start)
    result = optimize_shield(cfg, target_unit, seed=seed, verbose=False, params0=start,
                             snapshot_every=snapshot_every)
    out = result["forward"]
    metrics = {k: float(v) for k, v in result["metrics"].items()}

    # Scattered endpoints mean nothing unless each one is a stationary point: an
    # under-converged run would also leave designs spread out.
    grads = jax.grad(lambda p: result["loss_fn"](p, 1.0)[0])(result["params"])
    flat_grad, _ = jax.flatten_util.ravel_pytree(grads)
    grad_norm = float(jnp.linalg.norm(flat_grad))

    coords_end = design_coords(out, statics)
    coords_start = design_coords(start_out, statics)
    outline = np.asarray(out["outline"])
    outline = outline - outline.mean(axis=0)
    radius = np.hypot(outline[:, 0], outline[:, 1])
    areas = np.abs(np.asarray(tile_areas(out["flat_tiles"])))

    snaps = result.get("snapshots") or []
    traj = {}
    if snaps:
        traj = {
            "traj_steps": np.array([s["step"] for s in snaps], dtype=int),
            "traj_params": np.stack([s["params"] for s in snaps]),
            # The same symmetric metric the endpoints are scored with, so a path and its
            # endpoint can be read on one axis.
            "traj_error": np.array([max(s["precision_mm"], s["recall_mm"]) for s in snaps]),
        }

    return {
        **traj,
        "seed": seed, "sigma": sigma,
        "params": {k: np.asarray(v) for k, v in result["params"].items()},
        "start_params": {k: np.asarray(v) for k, v in start.items()},
        "coords": coords_end, "coords_start": coords_start,
        "flat_tiles": np.asarray(out["flat_tiles"]),
        "flat_tiles_start": np.asarray(start_out["flat_tiles"]),
        "deployed": np.asarray(out["deployed"]),
        "outline": outline, "r": np.asarray(out["r"]),
        "psi": float(out["psi"]),
        "shape_error_mm": max(metrics["precision_mm"], metrics["recall_mm"]),
        "precision_mm": metrics["precision_mm"], "recall_mm": metrics["recall_mm"],
        "worst_recall_mm": metrics["worst_recall_mm"],
        "loss": metrics["loss"], "shape": metrics["shape"], "grad_norm": grad_norm,
        "area_cv": metrics["area_cv"], "min_quality": metrics["min_quality"],
        "radius_mean_mm": float(radius.mean()),
        "radius_min_mm": float(radius.min()), "radius_max_mm": float(radius.max()),
        # Independent audits: the kinematic exactness claim must survive every start.
        "max_pin_gap_mm": float(np.abs(np.asarray(pin_residual(statics["plan"],
                                                               out["deployed"]))).max()),
        "flat_area_rel_err": float(abs(areas.sum() / (cfg.sheet_w * cfg.sheet_h) - 1.0)),
        # The one-sided floors are only a guard if they are slack; report how often they bite.
        "quality_floor_active_frac": float(np.mean(_quad_quality_np(
            np.asarray(out["flat_tiles"])) < cfg.q_min + 1e-6)),
    }


def _quad_quality_np(tiles: Float[np.ndarray, "n 4 2"]) -> Float[np.ndarray, "n"]:
    """``4 * area / sum(edge^2)`` per tile; unity for a square. NumPy twin of the loss term."""
    d = tiles - np.roll(tiles, -1, axis=1)
    perim_sq = np.sum(np.sum(d ** 2, axis=-1), axis=-1)
    return 4.0 * np.abs(np.asarray(tile_areas(tiles))) / (perim_sq + 1e-12)


def run_ensemble(cfg: ShieldDesignConfig, seeds, sigma: float = 1.0,
                 target_points: int = 480, progress=None, snapshot_every=None):
    """Solve the same problem from each seed's random start.

    Failures are collected rather than raised: one diverged start must not cost the ensemble.
    """
    statics = problem_statics(cfg)
    target_unit = circle_outline_unit(target_points)
    records, failures = [], []
    for seed in seeds:
        try:
            rec = run_one(cfg, statics, target_unit, seed, sigma, snapshot_every)
            records.append(rec)
            if progress is not None:
                progress(rec)
        except Exception as exc:                                   # noqa: BLE001
            failures.append({"seed": seed, "error": repr(exc)})
    return records, failures, statics


# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════

def ensemble_summary(records, group: dict) -> dict:
    """The pair of spreads that separates the three outcomes, plus the audits."""
    ends = [r["coords"] for r in records]
    starts = [r["coords_start"] for r in records]
    d_end = distance_matrix(ends, group)
    d_start = distance_matrix(starts, group)
    off = ~np.eye(len(records), dtype=bool)
    err = np.array([r["shape_error_mm"] for r in records])
    travel = np.array([canonical_distance(r["coords"], r["coords_start"], group)
                       for r in records])
    medoid = int(np.argmin(d_end.sum(axis=1)))
    labels, families = split_families(d_end)
    theta = np.array([2 * abs(np.degrees(r["psi"])) for r in records])
    families["per_family"] = [
        {"n": int((labels == c).sum()),
         "theta_deg": _stats(theta[labels == c]),
         "shape_error_mm": _stats(err[labels == c]),
         "area_cv": _stats(np.array([r["area_cv"] for r in records])[labels == c]),
         "seeds": [int(records[i]["seed"]) for i in np.flatnonzero(labels == c)]}
        for c in range(families["n_families"])]

    return {
        "n": len(records),
        "families": families,
        "shape_error_mm": _stats(err),
        "shape_error_spread_pct_of_median": float(100 * (err.max() - err.min())
                                                  / np.median(err)),
        "design_spread_end_mm": _stats(d_end[off]),
        "design_spread_start_mm": _stats(d_start[off]),
        "spread_ratio_end_over_start": float(np.median(d_end[off]) / np.median(d_start[off])),
        "travel_mm": _stats(travel),
        "grad_norm": _stats(np.array([r["grad_norm"] for r in records])),
        "psi_deg": _stats(np.array([np.degrees(r["psi"]) for r in records])),
        "theta_deg": _stats(np.array([2 * abs(np.degrees(r["psi"])) for r in records])),
        "area_cv": _stats(np.array([r["area_cv"] for r in records])),
        "radius_mean_mm": _stats(np.array([r["radius_mean_mm"] for r in records])),
        "max_pin_gap_mm": float(max(r["max_pin_gap_mm"] for r in records)),
        "max_flat_area_rel_err": float(max(r["flat_area_rel_err"] for r in records)),
        "quality_floor_active_frac": _stats(
            np.array([r["quality_floor_active_frac"] for r in records])),
        "medoid_seed": int(records[medoid]["seed"]),
        "group_order": len(group),
    }


def _stats(x) -> dict:
    x = np.asarray(x, dtype=float)
    return {"median": float(np.median(x)), "min": float(x.min()), "max": float(x.max()),
            "mean": float(x.mean()), "std": float(x.std())}


# ══════════════════════════════════════════════════════════════════════════════
# Persistence
# ══════════════════════════════════════════════════════════════════════════════

def save_ensemble(out_dir, records, cfg: ShieldDesignConfig):
    """One npz per design, so a run folder can be re-analysed without re-solving."""
    import os
    os.makedirs(os.path.join(out_dir, "designs"), exist_ok=True)
    for rec in records:
        payload = {k: v for k, v in rec.items()
                   if isinstance(v, np.ndarray) or np.isscalar(v)}
        for prefix in ("params", "start_params"):
            for k, v in rec[prefix].items():
                payload[f"{prefix}__{k}"] = v
        np.savez_compressed(
            os.path.join(out_dir, "designs", f"seed{rec['seed']:03d}.npz"),
            M=cfg.M, N=cfg.N, sheet_w=cfg.sheet_w, sheet_h=cfg.sheet_h,
            psi_max=cfg.psi_max, fixed_scale=cfg.fixed_scale, **payload)


def load_ensemble(out_dir):
    """Records as saved, ordered by seed."""
    import glob
    import os
    records = []
    for path in sorted(glob.glob(os.path.join(out_dir, "designs", "seed*.npz"))):
        with np.load(path) as z:
            rec = {k: z[k] for k in z.files}
        for k in ("seed",):
            rec[k] = int(rec[k])
        for k in ("psi", "shape_error_mm", "area_cv", "loss", "radius_mean_mm"):
            if k in rec:
                rec[k] = float(rec[k])
        records.append(rec)
    return records


def config_from_records(records) -> ShieldDesignConfig:
    """Rebuild the problem config from a saved run folder."""
    z = records[0]
    return replace(circle_problem(M=int(z["M"]), N=int(z["N"]),
                                  sheet_mm=float(z["sheet_w"]),
                                  radius_mm=float(z["fixed_scale"]) / 2.0),
                   psi_max=float(z["psi_max"]))
