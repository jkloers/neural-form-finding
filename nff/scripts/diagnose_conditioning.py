"""Stage-2 physics-solve conditioning diagnostic.

Measures, at the converged static-equilibrium solution, whether the gradient
risk lives in the FORWARD solve (non-convergence) or the BACKWARD pass
(ill-conditioned implicit-diff linear solve).

It reconstructs the exact total-potential-energy functional that
`setup_static_solver` minimizes — using only public building blocks, so this
script touches none of the plan-gated solver files — and reports, per problem:

  forward residual   ‖∇E(x*)‖           → is the forward solve actually at a
                                           stationary point? (compare to solver_tol)
  Hessian spectrum    λmin, λmax, cond   → conditioning of the energy Hessian H
  null/indefinite     #|λ|≈0, #λ<0       → mechanism soft modes / saddle
  backward gain        ‖H⁻¹ b‖ for unit b → gradient amplification in the IFT solve
  contact split       λ range w/ vs w/o  → how much contact barrier adds

Because the canonical config uses updated_lagrangian=false, the Hessian is
independent of the load factor t (the external-work term is linear), so the
reconstruction at the original reference frame is EXACT for that regime.

Usage:
    JAX_PLATFORMS=cpu conda run -n kgnn_mac python nff/scripts/diagnose_conditioning.py \
        --run-dir data/outputs/runs/run_20260611_124535_mpnn_best_2x2
    # optionally restrict:  --problem-ids c001
"""

import os
import glob
import pickle
import argparse

import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
import jax.numpy as jnp
import numpy as np

from nff.config.experiment import load_and_parse_config
from nff.scripts.train import _build_initial_state, _init_map_params
from nff.stages.pipeline import forward_pipeline
from nff.stages.physics.params import ReferenceGeometry, build_control_params
from nff.stages.physics.energy import build_potential_energy, constrain_energy
from nff.stages.physics.kinematics import build_constrained_kinematics, DOFsInfo
from nff.stages.physics.loading import build_loading
from nff.stages.physics.force_types import (
    has_geometry_dependent_loads, build_geometry_dependent_loading,
)


def _build_energy_over_free_dofs(valid_state, physics_cfg, load_specs, use_contact):
    """Reconstruct E(free_DOFs) exactly as setup_static_solver does, at t=1.

    Returns (energy_free_fn, free_DOF_ids, force_ext_vector).
    The returned energy_free_fn is a pure function of the free-DOF vector.
    """
    geometry = ReferenceGeometry.from_centroidal_state(valid_state)

    energy_fn = build_potential_energy(
        bond_connectivity=geometry.bond_connectivity,
        linearized_strains=physics_cfg.linearized_strains,
        use_contact=use_contact,
    )

    if has_geometry_dependent_loads(load_specs):
        loaded_pairs, loading_fn, force_vals_jax = build_geometry_dependent_loading(
            load_specs, valid_state.face_centroids)
        loading_params = {"force_values": force_vals_jax}
    else:
        loading_fn = valid_state.get_loading_function()
        loaded_pairs = valid_state.loaded_face_DOF_pairs if loading_fn else None
        loading_params = {}

    control_params = build_control_params(
        geometry=geometry,
        k_stretch=valid_state.k_stretch,
        k_shear=valid_state.k_shear,
        k_rot=valid_state.k_rot,
        density=valid_state.density,
        k_contact=physics_cfg.k_contact,
        min_angle=physics_cfg.min_angle,
        cutoff_angle=physics_cfg.cutoff_angle,
        use_contact=use_contact,
    )

    constrained_pairs = valid_state.constrained_face_DOF_pairs
    kinematics_fn = build_constrained_kinematics(geometry, constrained_pairs)
    constrained_energy_fn = constrain_energy(energy_fn, kinematics_fn)
    free_DOF_ids, _ = DOFsInfo(geometry.n_faces, constrained_pairs)

    if loaded_pairs is not None and loading_fn is not None and len(loaded_pairs) > 0:
        global_loading_fn = build_loading(
            geometry, loaded_pairs, loading_fn, constrained_pairs)
        F_ext = global_loading_fn(None, 1.0, loading_params)   # at full load
    else:
        F_ext = jnp.zeros_like(free_DOF_ids, dtype=float)

    def energy_free_fn(free_DOFs):
        U_int = constrained_energy_fn(free_DOFs, 1.0, control_params)
        W_ext = jnp.dot(F_ext, free_DOFs)
        return U_int - W_ext

    return energy_free_fn, free_DOF_ids, F_ext


def _spectrum(H):
    """Symmetric-eigenvalue summary of a Hessian matrix."""
    w = np.linalg.eigvalsh(np.asarray(H))            # ascending
    lam_max = float(w[-1])
    abs_w = np.abs(w)
    lam_min_abs = float(abs_w.min())
    tol = 1e-10 * lam_max
    n_zero = int(np.sum(abs_w < tol))
    n_neg = int(np.sum(w < -tol))
    pos = w[w > tol]
    lam_min_pos = float(pos.min()) if pos.size else float("nan")
    cond = lam_max / lam_min_pos if pos.size else float("inf")
    return dict(lam_max=lam_max, lam_min_abs=lam_min_abs, lam_min_pos=lam_min_pos,
                cond=cond, n_zero=n_zero, n_neg=n_neg, w=w)


def diagnose_problem(label, merged_cfg_path, ckpt_path):
    config = load_and_parse_config(merged_cfg_path)
    initial_state, _ = _build_initial_state(config)
    _, static_features = _init_map_params(config, initial_state)

    with open(ckpt_path, "rb") as f:
        map_params = pickle.load(f)

    load_specs = config.topology.get("loads", []) or []

    results = forward_pipeline(
        initial_state, config.target, config.validity, config.physics,
        map_type=config.mapping.type, map_params=map_params,
        use_shirley_chiu=config.mapping.use_shirley_chiu,
        strict_boundary_fit=config.mapping.strict_boundary_fit,
        static_features=static_features, load_specs=load_specs,
    )
    valid_state = results["valid_state"]
    final_disp = results["solution"].fields[-1]                  # (n_faces, 3)

    phys = config.physics
    # Full energy (matches the solver: strain + contact if enabled)
    E_full, free_ids, F_ext = _build_energy_over_free_dofs(
        valid_state, phys, load_specs, use_contact=phys.use_contact)
    final_free = final_disp.reshape(-1)[free_ids]

    grad = jax.grad(E_full)(final_free)
    H_full = jax.hessian(E_full)(final_free)
    res = float(jnp.linalg.norm(grad))
    f_norm = float(jnp.linalg.norm(F_ext)) + 1e-30
    sp = _spectrum(H_full)

    # Strain-only Hessian to attribute the contact contribution
    E_strain, _, _ = _build_energy_over_free_dofs(
        valid_state, phys, load_specs, use_contact=False)
    H_strain = jax.hessian(E_strain)(final_free)
    sp_strain = _spectrum(H_strain)

    # Backward amplification: worst-case ‖H⁻¹ b‖ over unit b == 1/λ_min_abs.
    # Also do a concrete solve with a random unit RHS and report its residual.
    rng = np.random.default_rng(0)
    b = rng.standard_normal(final_free.shape[0])
    b /= np.linalg.norm(b)
    Hn = np.asarray(H_full)
    try:
        v = np.linalg.solve(Hn, b)
        solve_res = np.linalg.norm(Hn @ v - b)
        v_norm = float(np.linalg.norm(v))
    except np.linalg.LinAlgError:
        v_norm, solve_res = float("inf"), float("nan")

    print(f"\n── {label}  ({free_ids.shape[0]} free DOFs) ──")
    print(f"  FORWARD  residual ‖∇E(x*)‖ = {res:.3e}   "
          f"(solver_tol={phys.solver_tol:.0e}, ‖F_ext‖={f_norm:.2e}, "
          f"rel={res/f_norm:.2e})")
    conv = "OK (at stationary point)" if res < 10 * phys.solver_tol * max(1.0, f_norm) \
        else "NOT converged — IFT gradient is biased"
    print(f"           verdict: {conv}")
    print(f"  HESSIAN  λmax={sp['lam_max']:.3e}  λmin(pos)={sp['lam_min_pos']:.3e}  "
          f"|λ|min={sp['lam_min_abs']:.3e}")
    print(f"           cond(H) = {sp['cond']:.3e}   "
          f"near-zero modes={sp['n_zero']}   negative(saddle)={sp['n_neg']}")
    print(f"  BACKWARD worst-case gradient gain 1/|λ|min = {1.0/sp['lam_min_abs']:.3e}   "
          f"‖H⁻¹·unit‖={v_norm:.3e}  (solve_res={solve_res:.1e})")
    print(f"  CONTACT  split:  cond(strain only)={sp_strain['cond']:.3e}  "
          f"→ full={sp['cond']:.3e}   "
          f"(contact {'ACTIVE' if sp['cond'] > 3*sp_strain['cond'] else 'mild/inactive'})")
    print(f"  ANISO    k_stretch/k_rot ≈ {float(jnp.max(valid_state.k_stretch))/float(jnp.max(valid_state.k_rot)):.1f}  "
          f"(theoretical cond floor from stiffness ratio)")

    return dict(label=label, n_free=int(free_ids.shape[0]), residual=res,
                rel_residual=res / f_norm, converged=res < 10 * phys.solver_tol * max(1.0, f_norm),
                cond=sp["cond"], cond_strain=sp_strain["cond"],
                lam_max=sp["lam_max"], lam_min_abs=sp["lam_min_abs"],
                n_zero=sp["n_zero"], n_neg=sp["n_neg"],
                backward_gain=1.0 / sp["lam_min_abs"])


def main():
    ap = argparse.ArgumentParser(description="Stage-2 conditioning diagnostic.")
    ap.add_argument("--run-dir", required=True,
                    help="A data/outputs/runs/run_* directory with per-problem subdirs.")
    ap.add_argument("--problem-ids", default=None,
                    help="Comma-separated substrings to filter problem labels.")
    args = ap.parse_args()

    ckpts = sorted(glob.glob(os.path.join(args.run_dir, "*", "best_params.pkl")))
    if args.problem_ids:
        keys = args.problem_ids.split(",")
        ckpts = [c for c in ckpts if any(k in c for k in keys)]
    if not ckpts:
        raise SystemExit(f"No best_params.pkl found under {args.run_dir}")

    print(f"Diagnosing {len(ckpts)} problem(s) under {args.run_dir}")
    rows = []
    for ckpt in ckpts:
        prob_dir = os.path.dirname(ckpt)
        label = os.path.basename(prob_dir)
        merged = os.path.join(prob_dir, "merged_config.yaml")
        if not os.path.exists(merged):
            merged = os.path.join(prob_dir, "config.yaml")
        if not os.path.exists(merged):
            print(f"  skip {label}: no config yaml")
            continue
        try:
            rows.append(diagnose_problem(label, merged, ckpt))
        except Exception as exc:
            import traceback
            print(f"  ✗ {label} FAILED: {exc}")
            traceback.print_exc()

    if rows:
        print(f"\n{'═'*92}\n  SUMMARY\n{'═'*92}")
        print(f"  {'problem':38s} {'conv':>5} {'cond(H)':>11} {'cond(str)':>11} "
              f"{'#0':>3} {'#neg':>4} {'bwd_gain':>11}")
        for r in rows:
            print(f"  {r['label']:38s} {('yes' if r['converged'] else 'NO'):>5} "
                  f"{r['cond']:>11.2e} {r['cond_strain']:>11.2e} "
                  f"{r['n_zero']:>3d} {r['n_neg']:>4d} {r['backward_gain']:>11.2e}")
        print(f"{'═'*92}")
        print("\nReading the table:")
        print("  conv=NO            → forward solve not at ∇E=0; IFT gradient biased (raise maxiter / better solver)")
        print("  cond(H) huge       → backward IFT solve amplifies gradients; regularize the backward (Fix 1)")
        print("  #0 / #neg > 0      → mechanism soft modes / saddle; grounding stiffness or LM damping (Fix 2/3)")
        print("  cond≈cond(str)     → contact not the driver; anisotropy/mechanism is")


if __name__ == "__main__":
    main()
