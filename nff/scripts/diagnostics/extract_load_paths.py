"""Extract the load path of a deployed tessellation and report whether it can be trusted.

Numbers before pictures. A load-path drawing is only worth looking at if the forces behind it are
exact and the solve that produced them actually converged, so this prints the exactness assertions
and the equilibrium residual before it writes anything.

⚠ ``solver_maxiter`` matters far more here than it does for a shape loss. An L-BFGS solve that stops
on its ITERATION CAP rather than its tolerance still lands a plausible chamfer while its per-hinge
quantities are badly wrong, so the default below raises the cap and the residual check catches what
is left.

Run:
    PYTHONPATH=$(pwd) JAX_PLATFORMS=cpu conda run -n kgnn_mac \
        python nff/scripts/diagnostics/extract_load_paths.py \
            --config data/configs/closed/sheet_4x8ft_pet_28deg.yaml
"""

import argparse
import copy
import os

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np                                                   # noqa: E402

from nff.config.experiment import load_and_parse_config              # noqa: E402
from nff.analysis.load_paths import (extract_force_field, force_budget, format_checks,  # noqa: E402
                                   interface_decomposition, orientation_confound, save_field,
                                   solve_deployed_state)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--params", default=None, help="best_params.pkl; omit for the untrained design")
    p.add_argument("--out", default=None, help="output stem (default data/outputs/load_paths/<name>)")
    p.add_argument("--steps", type=int, default=None, help="override physics.num_load_steps")
    p.add_argument("--maxiter", type=int, default=20000,
                   help="override physics.solver_maxiter; the config default is often too low")
    p.add_argument("--tol", type=float, default=1e-6, help="equilibrium residual tolerance")
    p.add_argument("--gamma", type=float, default=None,
                   help="community-detection resolution; default picks it per channel")
    p.add_argument("--seed", type=int, default=0, help="fixes the community sweep order")
    p.add_argument("--cut-rings", type=int, default=0,
                   help="make interfaces within this many steps of a grip uncuttable")
    args = p.parse_args()

    config = copy.deepcopy(load_and_parse_config(args.config))
    if args.steps:
        object.__setattr__(config.physics, 'num_load_steps', int(args.steps))
    if args.maxiter and args.maxiter != config.physics.solver_maxiter:
        print(f"  solver_maxiter {config.physics.solver_maxiter} -> {args.maxiter}")
        object.__setattr__(config.physics, 'solver_maxiter', int(args.maxiter))

    results, _, bond_energy, hinge_geometry, cut_axis = solve_deployed_state(config, args.params)
    field = extract_force_field(results, config.physics, bond_energy_fn=bond_energy,
                                hinge_geometry=hinge_geometry, cut_axis=cut_axis, tol=args.tol)

    name = os.path.splitext(os.path.basename(args.config))[0]
    stem = args.out or os.path.join("data", "outputs", "load_paths", name, "load_paths")
    save_field(field, stem, meta={'config': args.config, 'params': args.params,
                                  'label': 'trained' if args.params else 'untrained'})

    budget = force_budget(field)
    magnitude = np.linalg.norm(field.t_total, axis=-1)
    print(f"  {field.n_faces} tiles, {field.n_hinges} hinges")
    print(format_checks(field.checks))
    print("  force budget (shares sum to 1):")
    for key in ('compression_share', 'tension_share', 'shear_share', 'bearing_share'):
        print(f"    {key:20s} {budget[key]:6.1%}")
    print("  structure:")
    print(f"    {'active_fraction':20s} {budget['active_fraction']:6.1%}  "
          f"(hinges above 10% of peak)")
    print(f"    {'gini':20s} {budget['gini']:6.3f}  (0 = even, 1 = one chain carries all)")
    print(f"    {'couple_share':20s} {budget['couple_share']:6.3f}  (Cosserat signature, over the "
          f"{budget['bulk_tiles']} bulk tiles)")
    print(f"    {'peak_force':20s} {budget['peak_force']:6.4g}")
    print(f"  |t| median {np.median(magnitude):.4g}   max {magnitude.max():.4g}")

    _report_network(field, args)
    print(f"  wrote {stem}.npz / .json")


def _report_network(field, args):
    """The graph-level reading: chains, the interfaces most paths use, and the bottleneck."""
    from nff.analysis import force_network as fnet

    dec = interface_decomposition(field)
    summary = fnet.network_summary(field, gamma=args.gamma, seed=args.seed,
                                   protect_rings=args.cut_rings)
    stress, cut = summary['self_stress'], summary['min_cut']

    print("  force-chain network:")
    for channel in ('total', 'normal', 'shear'):
        community = summary[f'communities_{channel}']
        sizes = np.bincount(community['labels'])
        print(f"    {channel + ' chains':20s} {community['n_communities']:4d} communities "
              f"(largest {sizes.max()} tiles, {(sizes >= 3).sum()} of size 3+), "
              f"Q = {community['modularity']:.3f}, gamma = {community['gamma']:g}")
    print(f"    {'shear-dominated':20s} {summary['shear_dominated_fraction']:6.1%} of interfaces"
          f"   ⚠ branch-frame label")
    confound = orientation_confound(field)
    print(f"    {'orientation confound':20s} r = {confound:+.3f} between 'branch lies across the "
          f"load' and 'shear-dominated'")
    if np.isfinite(confound) and abs(confound) > 0.5:
        print("      ⚠ the normal/shear label is largely reporting HINGE ORIENTATION, not force")
        print("        character. Report it; never partition a figure or dataset on it.")

    top = np.flatnonzero(summary['betweenness_top_decile'])
    print(f"    {'betweenness':20s} top decile is {top.size} interfaces "
          f"(distance = F0/|f|, so strong links are short)")

    if np.isfinite(cut['cut_value']):
        print(f"    {'bottleneck':20s} {cut['cut_value']:.4g} N = "
              f"{cut['cut_value'] / field.f0:.2f} x F0 on {len(cut['cut_hinges'])} interfaces"
              + ("  <- the collar around a grip" if cut['at_grip'] else ""))
        if cut['at_grip'] and args.cut_rings == 0:
            print("      (true but trivial: the weakest point of a structure is where you hold it."
                  " Use --cut-rings 2.)")
        print("      the VALUE is pinned by equilibrium at >= F0; what it locates is the interfaces")

    print("  states of self-stress (Maxwell-Calladine):")
    print(f"    {'s / m':20s} {stress['n_self_stress']} states of self-stress, "
          f"{stress['n_mechanisms']} mechanisms  (index residual {stress['maxwell_calladine']})")
    print(f"    {'locked-in share':20s} {stress['share']:6.1%} of the interface force state carries "
          f"no external load")
    if stress['equilibrium_defect'] > 1e-4:
        print(f"    ⚠ {'closure defect':18s} {stress['equilibrium_defect']:.2e} of F0 -- the pair "
              f"interactions do not close as")
        print("      (force at the hinge + one couple). That is the l0_nom frame dependence of the")
        print("      ENERGY MODEL on closed hinges, not an extraction error.")


if __name__ == "__main__":
    main()
