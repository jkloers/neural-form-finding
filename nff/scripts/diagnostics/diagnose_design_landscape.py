"""Why the ensemble lands where it does: curvature of the objective at a converged design.

The ensemble says independent starts reach different patterns. That is only degeneracy if
the objective is genuinely flat along the directions they differ in -- otherwise it is slow
convergence wearing a disguise. Two measurements settle it:

    the Hessian spectrum at the medoid   how many directions the target actually constrains
    principal angles                     whether the ensemble spreads along those flat ones

Both are computed on the SHAPE term alone and on the full posed objective, because the
regularity floors carry curvature of their own and answer a different question.

    JAX_PLATFORMS=cpu conda run -n kgnn_mac python \
        nff/scripts/diagnostics/diagnose_design_landscape.py \
        --run data/outputs/design_cloud/circle_7x7_n64_s1
"""

import argparse
import json
import os
import sys
from dataclasses import replace

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp                                              # noqa: E402
import numpy as np                                                   # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

from nff.closed.analytic_targets import circle_outline_unit          # noqa: E402
from nff.closed.design_ensemble import (apply_symmetry, build_symmetry,  # noqa: E402
                                        circle_problem, distance_matrix, load_ensemble,
                                        problem_statics, split_families)
from nff.closed.design_landscape import (coords_jacobian, effective_dimension,  # noqa: E402
                                         flat_subspace, hessian_spectrum,
                                         principal_angles, push_to_geometry)
from nff.closed.shield_design import build_loss, _interior_cut_bases  # noqa: E402


def params_from_record(rec) -> dict:
    """The design vector as saved by the ensemble."""
    return {k[len("params__"):]: jnp.asarray(v)
            for k, v in rec.items() if k.startswith("params__")}


def align_to(reference, coords, group):
    """Rotate a design into the reference's frame."""
    best = min(group.values(),
               key=lambda el: np.mean(np.sum((reference - apply_symmetry(coords, el)) ** 2,
                                             axis=-1)))
    return apply_symmetry(coords, best), best[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--rel-tol", type=float, default=1e-6,
                    help="curvature below this fraction of the stiffest direction is flat")
    ap.add_argument("--n-modes", type=int, default=8,
                    help="ensemble PCA directions compared against the flat subspace")
    args = ap.parse_args()

    records = load_ensemble(args.run)
    with open(os.path.join(args.run, "summary.json")) as fh:
        summary = json.load(fh)
    meta = summary["config"]

    cfg = circle_problem(M=meta["M"], N=meta["N"], sheet_mm=meta["sheet_mm"],
                         radius_mm=meta["radius_mm"])
    statics = problem_statics(cfg)
    group = build_symmetry(cfg, statics)
    target_unit = circle_outline_unit(480)
    interior_base = _interior_cut_bases(statics["struct"])

    ends = [r["coords"] for r in records]
    D = distance_matrix(ends, group)
    medoid = int(np.argmin(D.sum(axis=1)))
    labels, fam = split_families(D)
    family = int(labels[medoid])
    members = np.flatnonzero(labels == family)
    theta = np.array([2 * abs(np.degrees(float(r["psi"]))) for r in records])
    print(f"  medoid: seed {records[medoid]['seed']} (index {medoid})")
    print(f"  families: sizes {fam['sizes']}, within {['%.1f' % w for w in fam['within_mean_mm']]} mm, "
          f"separation ratio {fam['separation_ratio']:.1f}")
    for c in range(fam["n_families"]):
        m = labels == c
        print(f"    family {c}: n={m.sum():2d}  theta {theta[m].mean():5.2f} deg  "
              f"error {np.mean([float(records[i]['shape_error_mm']) for i in np.flatnonzero(m)]):.2f} mm")
    print(f"  the local analysis below uses family {family} only "
          f"({len(members)} designs): a Hessian at one design cannot explain a spread that "
          f"jumps between basins.")

    # ── the two objectives ────────────────────────────────────────────────────────
    loss_total = build_loss(statics["forward_fn"], target_unit, statics["plan"], cfg,
                            interior_base)
    cfg_shape = replace(cfg, w_quality=0.0, w_uniform=0.0, w_homog=0.0, w_degen=0.0,
                        w_barrier=0.0)
    loss_shape = build_loss(statics["forward_fn"], target_unit, statics["plan"], cfg_shape,
                            interior_base)

    params = params_from_record(records[medoid])
    spec_total = hessian_spectrum(lambda p: loss_total(p, 1.0)[0], params)
    spec_shape = hessian_spectrum(lambda p: loss_shape(p, 1.0)[0], params)

    n_par = len(spec_shape["evals"])
    for name, spec in (("full objective", spec_total), ("shape term only", spec_shape)):
        ev = spec["evals"]
        flat = flat_subspace(spec, args.rel_tol).shape[1]
        print(f"\n  {name}: {n_par} design parameters, |grad| {spec['grad_norm']:.2e}")
        print(f"    curvature spans {np.abs(ev).max():.3e} down to {np.abs(ev).min():.3e} "
              f"({np.abs(ev).max() / max(np.abs(ev).min(), 1e-300):.1e} condition)")
        print(f"    negative eigenvalues: {int((ev < -args.rel_tol * np.abs(ev).max()).sum())}")
        print(f"    flat directions (< {args.rel_tol:g} of stiffest): {flat} of {n_par}")
        print(f"    effective dimension: {effective_dimension(ev):.1f}")

    # ── the ensemble's own spread, in the same geometric space ────────────────────
    ref = ends[medoid]
    aligned_all = np.stack([align_to(ref, c, group)[0].reshape(-1) for c in ends])
    aligned = aligned_all[members]
    centred = aligned - aligned.mean(axis=0)
    _, S, _ = np.linalg.svd(centred, full_matrices=False)
    pca_evals = (S ** 2) / max(len(members) - 1, 1)
    pca_dirs = np.linalg.svd(centred.T, full_matrices=False)[0][:, :args.n_modes]
    print(f"\n  ensemble spread: effective dimension {effective_dimension(pca_evals):.1f}, "
          f"top-{args.n_modes} keeps "
          f"{100 * pca_evals[:args.n_modes].sum() / pca_evals.sum():.1f}%")

    jac = coords_jacobian(statics["forward_fn"], params, statics["centre"])
    flat_par = flat_subspace(spec_shape, args.rel_tol)
    angles_flat = None
    if flat_par.shape[1] > 0:
        flat_geom = push_to_geometry(flat_par, jac)
        angles_flat = principal_angles(pca_dirs, flat_geom)
        print(f"  principal angles, ensemble spread vs FLAT subspace (deg): "
              f"{np.array2string(angles_flat[:6], precision=1)}")

    # The softest directions are informative even when nothing is exactly flat.
    k = min(args.n_modes, n_par)
    soft_geom = push_to_geometry(spec_shape["evecs"][:, :k], jac)
    angles_soft = principal_angles(pca_dirs, soft_geom)
    stiff_geom = push_to_geometry(spec_shape["evecs"][:, -k:], jac)
    angles_stiff = principal_angles(pca_dirs, stiff_geom)
    print(f"  principal angles vs the {k} SOFTEST directions (deg): "
          f"{np.array2string(angles_soft[:6], precision=1)}")
    print(f"  principal angles vs the {k} STIFFEST directions (deg): "
          f"{np.array2string(angles_stiff[:6], precision=1)}")

    # The direct test, one point per design direction: how much does the ensemble vary along
    # a direction, against how much curvature the objective puts there? A basin the optimiser
    # merely failed to descend would show no relation; a genuinely flat set spreads where the
    # curvature is smallest.
    all_geom, kept = push_to_geometry(spec_shape["evecs"], jac, return_mask=True)
    var_along = np.var(centred @ all_geom, axis=0)
    curvature = np.abs(spec_shape["evals"])[kept]      # same filter, or the pairing is wrong
    ok = (curvature > 0) & (var_along > 0)
    rho = float(np.corrcoef(np.log10(curvature[ok]), np.log10(var_along[ok]))[0, 1])
    print(f"\n  variance-vs-curvature over {int(ok.sum())} directions: "
          f"log-log correlation {rho:+.3f}")
    soft_half = curvature < np.median(curvature)
    print(f"    softest half carries "
          f"{100 * var_along[soft_half].sum() / var_along.sum():.1f}% of the ensemble variance")

    # ── determinacy field: per cut node, how much does the ensemble disagree? ─────
    # Over the WHOLE ensemble: the question is what the target leaves free, and the family
    # split is part of that freedom rather than something to average away.
    per_node = aligned_all.reshape(len(records), -1, 2)
    node_std = np.sqrt(np.mean(np.var(per_node, axis=0), axis=-1))     # mm, per cut vertex
    grid = node_std.reshape(-1, 2).mean(axis=1)                        # vertex pair -> node
    print(f"\n  determinacy: cut-node scatter {grid.min():.1f} to {grid.max():.1f} mm "
          f"(median {np.median(grid):.1f})")

    out = os.path.join(args.run, "landscape.npz")
    np.savez_compressed(
        out,
        evals_total=spec_total["evals"], evals_shape=spec_shape["evals"],
        grad_norm_total=spec_total["grad_norm"], grad_norm_shape=spec_shape["grad_norm"],
        pca_evals=pca_evals, node_std=node_std, node_std_grid=grid,
        medoid_index=medoid, medoid_seed=records[medoid]["seed"],
        labels=labels, family=family, family_sizes=np.asarray(fam["sizes"]),
        theta=theta,
        angles_soft=angles_soft, angles_stiff=angles_stiff,
        angles_flat=np.array([]) if angles_flat is None else angles_flat,
        n_flat=flat_par.shape[1], rel_tol=args.rel_tol,
        curvature=curvature, var_along=var_along, var_curv_logcorr=rho,
        medoid_coords=ends[medoid],
        medoid_tiles=np.asarray(records[medoid]["flat_tiles"]) - statics["centre"],
    )
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
