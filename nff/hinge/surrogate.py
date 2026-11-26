"""The learned hinge energy: the CalculiX-trained condensed surrogate as a Stage-2 bond energy.

The network itself lives in ``nff/models/hinge_surrogate.py``; this module wires a checkpoint into
the pipeline: Gap-2 scale calibration, the out-of-domain barrier, the design-tracked per-hinge
geometry ``HingeGeometry(w_lig, alpha, sec_dir)``, and the probe and damage readers the design loss
uses.
"""

import jax.numpy as jnp
import numpy as np

from nff.hinge.frame import _alpha_sec_bond_order, _bond_order_perm, _flat_coords_from_design
from nff.hinge.model import HingeModel


def build_surrogate_hinge(config, static_features, state, init_map_params) -> HingeModel:
    """Surrogate hinge model with DESIGN-TRACKED hinge geometry.

    The energy and probe are built ONCE (they close over only the net, the Gap-2 scales and the fixed
    fillet). The design-dependent ``HingeGeometry(w_lig, alpha, sec_dir)`` is supplied PER STEP as data
    by ``geometry_fn(map_params)``, and the caller threads it into the solver via
    ``forward_pipeline(hinge_geometry=)`` and into ``probe_fn``. Because it is an explicit solver
    input (not closed over), jaxopt's implicit diff carries the full ``d/d(design)``.
    """
    hm = config.hinge_model
    from nff.models.hinge_surrogate import (load_hinge_surrogate, build_hinge_bond_energy_fn,
                                            build_hinge_probe_fn, build_hinge_damage_fn,
                                            calibrate_scales, DOMAIN, HingeGeometry, W_LIG_BOUNDS,
                                            w_lig_from_logit, w_lig_logit_from_width)
    from nff.topology.hinge_descriptor import build_hinge_descriptor_structure

    topo = config.topology
    M, N = int(topo['M']), int(topo['N'])
    # The net is dimensional, so a checkpoint condensed from a different material or thickness is
    # silently wrong rather than an error. `expect` makes the mismatch print.
    net, stats, eps_f = load_hinge_surrogate(
        hm.checkpoint, expect={'material': getattr(hm, 'material', None),
                               'thickness': getattr(hm, 'thickness_mm', None)})
    fr = float(getattr(hm, 'fillet_ratio', 0.16))     # design cut-tip fillet (3rd g DOF for 6-feat nets)
    dom = stats.get("domain", DOMAIN)                 # data-driven OOD box (wide v2 auto-widens it)
    fail_line = float(getattr(hm, 'fail_line', 1.0))
    bond_pairs = np.asarray(state.bond_connectivity)
    w_lig_arr = jnp.full(state.hinge_node_pairs.shape[0], float(hm.w_lig_mm))

    # static topology (once): hinge structure + the design-invariant bond-order permutation.
    # `state` is ALWAYS the uniform-r_init sheet (build_closed_tessellation), so the position-based
    # perm matching must use the UNIFORM design -- not init_map_params, which may be a RANDOM start
    # that no longer aligns with `state`. The perm is topological, so uniform-vs-random is irrelevant
    # to its correctness; alpha0/sec0 (header + calibration) then reflect the actual starting design.
    hs = build_hinge_descriptor_structure(M, N, ref_r=float(topo.get('r_init', 0.45)))
    ru = float(topo.get('r_init', 0.45))
    uni_params = {'z': jnp.full_like(jnp.asarray(init_map_params['z']), float(np.log(ru / (1.0 - ru)))),
                  'bnd_logits': jnp.zeros_like(jnp.asarray(init_map_params['bnd_logits']))}
    perm = _bond_order_perm(state, hs, _flat_coords_from_design(static_features, uni_params))
    alpha0, sec0 = _alpha_sec_bond_order(hs, perm, _flat_coords_from_design(static_features, init_map_params))

    # calibrate the Gap-2 scales ONCE at the initial design (fixed normalization bridges)
    if hm.calibrate:
        kst = float(np.mean(np.asarray(state.k_stretch))); krt = float(np.mean(np.asarray(state.k_rot)))
        ls, es = calibrate_scales(net, stats, alpha=alpha0, w_lig=w_lig_arr, k_stretch=kst, k_rot=krt,
                                  fillet_ratio=fr)
    else:
        ls, es = hm.length_scale, hm.energy_scale

    a0 = np.degrees(np.asarray(alpha0))
    print(f"[hinge_model] SURROGATE  material={hm.material} t={hm.thickness_mm}mm w_lig={hm.w_lig_mm}mm "
          f"eps_f={eps_f}  |  {len(alpha0)} hinges  alpha {a0.min():.0f}-{a0.max():.0f}deg (RVE-frame)"
          f"  length_scale={ls:.3g}mm/u  energy_scale={es:.3g}  barrier={hm.barrier}")
    lw = getattr(getattr(config, 'training', None), 'loss_weights', None)
    if lw is not None and (lw.damage or lw.ood or lw.compression):
        print(f"[hinge_model]   + design loss  damage={lw.damage} (path-max, D/D_scale, "
              f"D_scale={float(stats.get('D_scale', 1.0)):.3g})  compression={lw.compression}  "
              f"ood={lw.ood}  fail_line(report only)={fail_line}")
        if float(dom.get('eta_a_min', 0.0)) < 0.0 and not lw.compression:
            print(f"[hinge_model]   NOTE this checkpoint's box allows compression "
                  f"(eta_a_min={dom['eta_a_min']:.3g}), so the OOD barrier no longer resists it; "
                  "loss_weights.compression is the only handle left")

    w_lig_logit0 = None
    if bool(getattr(hm, 'learn_w_lig', False)):
        w_lig_logit0 = w_lig_logit_from_width(w_lig_arr)
        lo, hi = W_LIG_BOUNDS
        got = float(np.mean(np.asarray(w_lig_from_logit(w_lig_logit0))))
        print(f"[hinge_model]   + LEARNABLE w_lig (per-hinge, [{lo:g},{hi:g}]mm; "
              f"asked {float(hm.w_lig_mm):.1f}mm -> init {got:.1f}mm)")
        if abs(got - float(hm.w_lig_mm)) > 1e-3:
            print(f"[hinge_model]   WARNING w_lig_mm={hm.w_lig_mm} is outside the learnable box "
                  f"[{lo:g},{hi:g}] -- the optimizer starts at {got:.1f}mm while the cut export "
                  "writes the config value")

    # Energy + probe built ONCE; they close over only the net / scales / fillet. The per-hinge
    # geometry is threaded per step (see hinge_geometry_from_design below).
    bond_energy = build_hinge_bond_energy_fn(net, stats, length_scale=ls, energy_scale=es,
                                             barrier=hm.barrier, domain=dom, fillet_ratio=fr)
    probe_fn = build_hinge_probe_fn(net, stats, bond_pairs=bond_pairs, length_scale=ls,
                                    domain=dom, fillet_ratio=fr)
    damage_fn = build_hinge_damage_fn(net, stats, bond_pairs=bond_pairs, length_scale=ls, fillet_ratio=fr)

    def hinge_geometry_from_design(map_params):
        """The per-hinge HingeGeometry(w_lig, alpha, sec_dir) for the CURRENT design -- TRACED in
        map_params. alpha/sec_dir = f(r, boundary) via the flat sheet; w_lig from the learnable logit
        (or the fixed manufacturing width). Deterministic, so autodiff carries d/d(design) through it."""
        alpha, sec_dir = _alpha_sec_bond_order(hs, perm, _flat_coords_from_design(static_features, map_params))
        w_lig = (w_lig_from_logit(map_params['w_lig_logit'])
                 if isinstance(map_params, dict) and 'w_lig_logit' in map_params else w_lig_arr)
        return HingeGeometry(w_lig=w_lig, alpha=alpha, sec_dir=sec_dir)

    return HingeModel(bond_energy, probe_fn, hinge_geometry_from_design, damage_fn, w_lig_logit0)
