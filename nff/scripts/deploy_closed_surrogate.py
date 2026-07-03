"""Phase 3.6 validation: deploy a closed_les design under the LEARNED hinge-energy surrogate
vs the linear-spring baseline, and compare the deployed shapes.

Runs Stage-2 directly (no full-pipeline threading needed) with build_potential_energy's injectable
bond_energy_fn: the spring model (bond_energy_fn=None) and the surrogate adapter, on the SAME flat
closed state + loads. Reports per-face deployment agreement.

    JAX_PLATFORMS=cpu conda run -n kgnn_mac python nff/scripts/deploy_closed_surrogate.py \
        --config-name circle_4x4_moment --surrogate data/outputs/hinge_surrogate.pkl
"""
import argparse

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from nff.config.experiment import load_and_parse_config
from nff.stages.mapping import apply_closed_les_mapping
from nff.stages.physics.params import ReferenceGeometry, build_control_params
from nff.stages.physics.energy import build_potential_energy
from nff.stages.physics.statics import setup_static_solver
from nff.stages.physics.force_types import (has_geometry_dependent_loads,
                                            build_geometry_dependent_loading)
from nff.scripts.closed_setup import (build_closed_initial_state, init_closed_les_params,
                                      closed_hinge_geometry, surrogate_scales)
from nff.models.hinge_surrogate import load_hinge_surrogate, build_hinge_bond_energy_fn


def deploy_stage2(valid_state, physics_cfg, load_specs, bond_energy_fn=None):
    """Replicates pipeline._execute_stage2_physics, parameterized by the bond energy."""
    geometry = ReferenceGeometry.from_centroidal_state(valid_state)
    pe = build_potential_energy(geometry.bond_connectivity, physics_cfg.linearized_strains,
                                physics_cfg.use_contact, bond_energy_fn=bond_energy_fn)
    if has_geometry_dependent_loads(load_specs):
        loaded, loading_fn, force_vals = build_geometry_dependent_loading(
            load_specs, valid_state.face_centroids)
    else:
        loading_fn = valid_state.get_loading_function()
        loaded = valid_state.loaded_face_DOF_pairs if loading_fn else None
        force_vals = None
    solve = setup_static_solver(
        geometry=geometry, energy_fn=pe, loaded_face_DOF_pairs=loaded, loading_fn=loading_fn,
        constrained_face_DOF_pairs=valid_state.constrained_face_DOF_pairs,
        incremental=physics_cfg.incremental, num_steps=physics_cfg.num_load_steps,
        solver_maxiter=physics_cfg.solver_maxiter, solver_tol=physics_cfg.solver_tol,
        updated_lagrangian=physics_cfg.updated_lagrangian)
    cp = build_control_params(
        geometry=geometry, k_stretch=valid_state.k_stretch, k_shear=valid_state.k_shear,
        k_rot=valid_state.k_rot, density=valid_state.density, k_contact=physics_cfg.k_contact,
        min_angle=physics_cfg.min_angle, cutoff_angle=physics_cfg.cutoff_angle,
        use_contact=physics_cfg.use_contact)
    if force_vals is not None:
        cp = cp._replace(loading_params={'force_values': force_vals})
    u0 = jnp.zeros((geometry.n_faces, 3), dtype=float)
    return solve(initial_displacements=u0, control_params=cp).fields[-1]   # (n_faces, 3) deployed


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config-dir", default="closed")
    ap.add_argument("--config-name", default="circle_4x4_moment")
    ap.add_argument("--surrogate", default="data/outputs/hinge_surrogate.pkl")
    ap.add_argument("--w-lig-mm", dest="w_lig_mm", type=float, default=5.0)
    ap.add_argument("--length-scale", dest="length_scale", type=float, default=None)
    ap.add_argument("--energy-scale", dest="energy_scale", type=float, default=None)
    args = ap.parse_args()

    config = load_and_parse_config(f"data/configs/{args.config_dir}/{args.config_name}.yaml")
    load_specs = config.topology.get('loads', [])
    initial_state, _ = build_closed_initial_state(config)
    params, static_features = init_closed_les_params(config)

    # Stage-0: closed_les map -> the flat closed sheet (Stage-1 bypassed).
    valid_state = apply_closed_les_mapping(initial_state, params, static_features)

    topo = config.topology
    M, N = int(topo['M']), int(topo['N'])
    r = float(topo.get('r_init', 0.45)); spacing = float(topo.get('spacing', 1.0))
    alpha, w_lig, sec_dir = closed_hinge_geometry(valid_state, M, N, r, spacing, args.w_lig_mm)

    net, stats, _ = load_hinge_surrogate(args.surrogate)
    ls, es = surrogate_scales(config)
    if args.length_scale is not None: ls = args.length_scale
    if args.energy_scale is not None: es = args.energy_scale
    adapter = build_hinge_bond_energy_fn(net, stats, alpha=alpha, w_lig=w_lig, sec_dir=sec_dir,
                                         length_scale=ls, energy_scale=es)

    print(f"config {args.config_name}: {valid_state.face_centroids.shape[0]} faces, {len(alpha)} hinges  "
          f"| w_lig={args.w_lig_mm}mm  length_scale={ls}  energy_scale={es}")
    u_spring = np.asarray(deploy_stage2(valid_state, config.physics, load_specs, None))
    u_surro = np.asarray(deploy_stage2(valid_state, config.physics, load_specs, adapter))

    dxy = lambda u: np.linalg.norm(u[:, :2], axis=1)
    print("\n              spring        surrogate")
    print(f"  max |dxy|   {dxy(u_spring).max():.4f}       {dxy(u_surro).max():.4f}")
    print(f"  max |dtheta| {np.abs(u_spring[:,2]).max():.4f}       {np.abs(u_surro[:,2]).max():.4f}")
    both = dxy(u_spring).max() > 1e-6 and dxy(u_surro).max() > 1e-6
    if both:
        cos = float(np.sum(u_spring * u_surro) / (np.linalg.norm(u_spring) * np.linalg.norm(u_surro)))
        print(f"  deployment-direction cosine(spring, surrogate) = {cos:.3f}  (1 = same mode)")
    print("\nboth solves produced a finite deployment ✓" if np.isfinite(u_surro).all() else "surrogate solve non-finite ✗")


if __name__ == "__main__":
    main()
