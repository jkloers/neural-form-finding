"""Measure the DESIGN-GRADIENT weight at which each loss term matches a reference term.

A loss weight is not a preference, it is a gradient budget. Two terms with equal weights do not
contribute equally: what the optimizer sees is ``w * d(term)/d(design)``, and the raw gradient norms
of ``chamfer``, ``compression`` and ``damage`` differ by an order of magnitude. So the only
meaningful statement about a weight is the ratio it bears to PARITY --

    parity(term) = ||w_ref * grad(ref)||  /  ||grad(term at weight 1)||

the weight at which ``term`` pushes the design exactly as hard as the reference does.

⚠ PARITY IS PROBLEM-SPECIFIC AND MUST BE RE-MEASURED WHENEVER THE LOAD CHANGES. On the PET 45 deg
sheet the inherited ``compression: 1.0`` turned out to be 8.7x BELOW parity: the value was carried
over from an over-driven parent where the worst ``eta_a`` was -1.76, and at the correctly calibrated
load the worst is -0.223, so the same weight carries ~8x less gradient. Reusing a weight across a
re-calibration silently disables the term.

The gradient is taken THROUGH the Stage-2 solve (jaxopt implicit diff), so this is the real design
gradient, not a geometric proxy.

Run:
    PYTHONPATH=$(pwd) JAX_PLATFORMS=cpu conda run -n kgnn_mac \
        python nff/scripts/diagnostics/gradient_parity.py \
            --config data/configs/closed/pet45_baseline.yaml --terms compression,damage
"""

import argparse
import json
import os
import pickle

import jax
jax.config.update("jax_enable_x64", True)   # a gradient ratio in float32 is not worth reporting

import numpy as np

from nff.config.experiment import load_and_parse_config, TargetConfig
from nff.config.targets import get_target_points
from nff.closed.setup import (build_closed_initial_state, init_closed_les_params,
                              build_surrogate_energy)
from nff.training.loss import compute_end_to_end_loss, TERMS


# Every additive weight, so a measurement isolates exactly one term.
_ADDITIVE = tuple(name for name, _attr, _fn, _probe in TERMS)


def _fixed_rect_target(config):
    """The SAME explicit rectangle cloud ``run_closed`` trains against, or (None, cfg.target).

    ⚠ Without this the chamfer reference would be measured against the config's fallback
    ``target:`` circle, which is a different objective from the one that actually trains -- and
    every parity ratio derived from it would be wrong.
    """
    topo = config.topology
    if config.target.type != 'rectangle' or 'target_half_w' not in topo:
        return None, config.target
    cen = np.array([float(topo.get('target_cx', config.target.center[0])),
                    float(topo.get('target_cy', config.target.center[1]))])
    hw, hh = float(topo['target_half_w']), float(topo['target_half_h'])
    rot = float(topo.get('target_rot_deg', 0.0) or 0.0)
    cloud = get_target_points({'type': 'rectangle', 'center': cen, 'radius': max(hw, hh),
                               'half_w': hw, 'half_h': hh, 'rot_deg': rot}, n_points=400)
    target_eff = TargetConfig(type='circle', center=(float(cen[0]), float(cen[1])),
                              radius=float(max(hw, hh)))
    return cloud, target_eff


def _resolve(config_arg: str) -> str:
    if os.path.exists(config_arg):
        return config_arg
    return f"data/configs/closed/{config_arg}.yaml"


def build_loss_fn(cfg, term: str, weight: float, state, static_features, bond_energy, probe_fn,
                  geometry_fn, target_cloud, target_eff):
    """A loss closure with EVERY additive weight zeroed except ``term``."""
    for k in _ADDITIVE:
        if hasattr(cfg.training.loss_weights, k):
            object.__setattr__(cfg.training.loss_weights, k, 0.0)
    object.__setattr__(cfg.training.loss_weights, term, float(weight))

    def loss_fn(p):
        value, _ = compute_end_to_end_loss(
            p, state, target_eff, cfg.validity, cfg.physics, cfg.training,
            map_type=cfg.mapping.type, static_features=static_features,
            load_specs=cfg.topology['loads'], target_cloud=target_cloud,
            bond_energy_fn=bond_energy, hinge_probe_fn=probe_fn, hinge_geometry_fn=geometry_fn)
        return value

    return loss_fn


def grad_norm(cfg_path, term, weight, params_path=None, init_noise=None, init_seed=None):
    """||d(weight * term)/d(design)||_2 at the given design point."""
    cfg = load_and_parse_config(cfg_path)
    if init_noise is not None:
        cfg.topology['init_noise'] = float(init_noise)
    if init_seed is not None:
        cfg.topology['init_seed'] = int(init_seed)

    state, _ = build_closed_initial_state(cfg)
    params, static_features = init_closed_les_params(cfg)
    bond_energy, probe_fn, geometry_fn, _, w_lig_logit0 = build_surrogate_energy(
        cfg, static_features, state, params)
    if w_lig_logit0 is not None:
        params = {**params, 'w_lig_logit': w_lig_logit0}
    if params_path:                                  # measure at a TRAINED design instead
        with open(params_path, 'rb') as fh:
            params = pickle.load(fh)

    target_cloud, target_eff = _fixed_rect_target(cfg)
    loss_fn = build_loss_fn(cfg, term, weight, state, static_features, bond_energy, probe_fn,
                            geometry_fn, target_cloud, target_eff)
    flat, _ = jax.flatten_util.ravel_pytree(jax.grad(loss_fn)(params))
    return float(np.linalg.norm(np.asarray(flat)))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="path, or a bare name under data/configs/closed/")
    p.add_argument("--terms", default="compression,damage",
                   help="comma-separated terms to price against the reference")
    p.add_argument("--ref", default="chamfer", help="reference term (measured at its CONFIG weight)")
    p.add_argument("--ref-weight", type=float, default=None,
                   help="override the reference weight; defaults to the config's own")
    p.add_argument("--params", default=None,
                   help="best_params.pkl -- measure at a TRAINED design instead of the untrained one")
    p.add_argument("--init-noise", type=float, default=None)
    p.add_argument("--init-seed", type=int, default=None)
    p.add_argument("--out", default=None, help="write the result as JSON here")
    args = p.parse_args()

    cfg_path = _resolve(args.config)
    cfg0 = load_and_parse_config(cfg_path)
    ref_w = (args.ref_weight if args.ref_weight is not None
             else float(getattr(cfg0.training.loss_weights, args.ref)))

    where = f"TRAINED design ({args.params})" if args.params else "UNTRAINED design"
    print(f"\n  {cfg_path}")
    print(f"  gradient measured at the {where}")
    noise = (args.init_noise if args.init_noise is not None
             else float(cfg0.topology.get('init_noise', 0.0)))
    seed = args.init_seed if args.init_seed is not None else int(cfg0.topology.get('init_seed', 0))
    print(f"  start: init_noise = {noise:g}  init_seed = {seed}\n")

    kw = dict(params_path=args.params, init_noise=args.init_noise, init_seed=args.init_seed)
    g_ref = grad_norm(cfg_path, args.ref, ref_w, **kw)
    print(f"    {args.ref:<14} w = {ref_w:<10g} ||w*g|| = {g_ref:.4e}   <- reference")

    out = {"config": cfg_path, "ref": args.ref, "ref_weight": ref_w, "ref_grad_norm": g_ref,
           "params": args.params, "init_noise": noise, "init_seed": seed, "terms": {}}

    for term in [t.strip() for t in args.terms.split(",") if t.strip()]:
        g1 = grad_norm(cfg_path, term, 1.0, **kw)
        parity = g_ref / g1 if g1 > 0 else float('inf')
        current = float(getattr(cfg0.training.loss_weights, term, 0.0))
        rel = (current / parity) if parity not in (0.0, float('inf')) else 0.0
        verdict = ("BELOW parity" if rel < 0.5 else
                   "above parity" if rel > 2.0 else "near parity")
        print(f"    {term:<14} w = 1          ||g||    = {g1:.4e}   "
              f"parity = {parity:9.3f}   config has {current:g} ({rel:.2f}x, {verdict})")
        out["terms"][term] = {"grad_norm_at_1": g1, "parity_weight": parity,
                              "config_weight": current, "ratio_to_parity": rel}

    print("\n  parity(term) = ||w_ref*g_ref|| / ||g_term at w=1||  -- the weight at which this term\n"
          "  moves the design as hard as the reference. Scale it deliberately (e.g. 10x parity to\n"
          "  make the term dominate); never inherit a weight across a re-calibration.\n")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"  wrote {args.out}\n")


if __name__ == "__main__":
    main()
