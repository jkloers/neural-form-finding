"""Shared setup for the closed-state (closed_les) inverse-design pipeline.

Builds the flat closed-sheet tessellation and the differentiable design
parameters from a parsed config. Imported by both ``train.py`` (generic
dispatch) and ``run_closed.py`` (the closed-state driver) so neither script
imports the other.

The flat sheet is the Dang et al. (2021) RDPQK construction; the deployed shape
is produced later by Stage-2 physics. ``map_type: closed_les`` optimizes the
per-cut aspect ratios and the boundary-slider positions through the pipeline.
"""

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from nff.topology.closed_builder import build_closed_tessellation
from nff.topology.closed_builder_jax import build_deploy_structure, build_boundary_edges
from nff.config.conditions import configure_tessellation
from nff.stages.state import CentroidalState


def build_closed_initial_state(config):
    """Build a flat closed-state RDPQK tessellation + CentroidalState from config.

    Geometry comes from the closed builder (M×N panels); material, clamps, and
    loads are applied with the standard config machinery.
    """
    topo = config.topology
    M = int(topo['M'])
    N = int(topo['N'])
    r_init = float(topo.get('r_init', 0.45))
    spacing = float(topo.get('spacing', 1.0))

    tessellation = build_closed_tessellation(M, N, r=r_init, spacing=spacing)
    configure_tessellation(tessellation, SimpleNamespace(**topo))  # material, clamps, loads

    # Optional: restrict which DOFs the clamp fixes (default all 3 = [x, y, theta]).
    # e.g. clamped_dofs: [0, 1] pins translation but leaves rotation free.
    clamp_dofs = topo.get('clamped_dofs', None)
    clamped = topo.get('bc_clamped')
    if clamp_dofs is not None and isinstance(clamped, list):
        for f in clamped:
            tessellation.set_face_dofs(int(f), list(clamp_dofs))
    # Additionally pin y (DOF 1) on specific faces — e.g. anchor one corner so a
    # net vertical load has support while the rest of the edge stays x-only.
    for f in topo.get('y_pin_faces', []) or []:
        tessellation.set_face_dofs(int(f), [0, 1])

    state = CentroidalState.from_tessellation(tessellation, target_cfg=config.target)
    return state, tessellation


def init_closed_les_params(config):
    """Init design params and static LES structure for the closed_les map type.

    Params: {'z': (rows, cols) latent for r = sigmoid(z), init r_init;
             'bnd_logits': (n_logits,) per-edge logits -> ordered boundary sliders}.
    The ordered-boundary + r∈(0,1) parameterization guarantees a valid (non-self-
    intersecting) flat tessellation, so no validity loss is needed.
    """
    topo = config.topology
    M, N = int(topo['M']), int(topo['N'])
    r_init = float(topo.get('r_init', 0.45))
    spacing = float(topo.get('spacing', 1.0))

    struct = build_deploy_structure(M, N)
    sliders = build_boundary_edges(struct, spacing=spacing)

    z_init = float(np.log(r_init / (1.0 - r_init)))         # sigmoid(z_init) = r_init
    params = {
        'z': jnp.full((struct['rows'], struct['cols']), z_init),
        'bnd_logits': jnp.asarray(sliders['init_logits'], dtype=float),
    }
    static_features = {'struct': struct, 'sliders': sliders, 'closed_les': True}
    return params, static_features


def surrogate_scales(config):
    """Gap-2 unit bridges for the closed_les hinge-energy surrogate (from config, with defaults).

    The surrogate is trained in physical units (mm, N.mm) but the closed pipeline runs in abstract
    units. Two scalars reconcile them at integration:
      length_scale : mm per pipeline length-unit — pick a physical tile size so w_lig ~ 1/10 tile,
      energy_scale : pipeline-energy per N.mm — so the surrogate energy is commensurate with the
                     chamfer loss (neither swamps nor vanishes).
    Passed to ``nff.models.hinge_surrogate.build_hinge_bond_energy_fn`` at setup.
    """
    phys = getattr(config, 'physics', None) or {}
    get = phys.get if isinstance(phys, dict) else (lambda k, d: getattr(phys, k, d))
    return float(get('length_scale_mm', 1.0)), float(get('energy_scale', 1.0))


def closed_hinge_geometry(state, M, N, r, spacing, w_lig_mm, ref_r=None):
    """Per-hinge ``(alpha[rad], w_lig[mm], sec_dir[2])`` in BOND order for a closed design.

    The surrogate adapter needs geometry aligned with the Stage-2 bond order. The analytic hinge
    descriptor is computed in its own hinge order; we align it to the bonds by POSITION-matching
    each state hinge-vertex to the nearest descriptor pivot (verified: same count, identity order,
    face-pairs agree — but we match + assert to stay robust to any config). ``alpha`` and
    ``sec_dir`` are scale-free so they transfer directly; ``w_lig`` is a manufacturing constant [mm].
    """
    import numpy as np
    from nff.stages.geometry import hinge_vertex_positions
    from nff.topology.hinge_descriptor import (build_hinge_descriptor_structure,
                                               hinge_descriptors_from_design)
    from nff.topology.closed_builder_jax import boundary_points_flat, solve_cut_vertices_jax

    hs = build_hinge_descriptor_structure(M, N, ref_r=ref_r if ref_r is not None else r)
    ds = hs['deploy_struct']
    bf = jnp.asarray(boundary_points_flat(ds, spacing))
    r_arr = jnp.full((ds['rows'], ds['cols']), r)
    out = hinge_descriptors_from_design(hs, bf, r_arr)
    coords = np.asarray(solve_cut_vertices_jax(ds, bf, r_arr))
    piv_desc = coords[np.asarray(hs['pivot_pid'])]                          # (H, 2)

    p1, _ = hinge_vertex_positions(state.face_centroids, state.centroid_node_vectors,
                                   state.hinge_node_pairs)
    D = np.linalg.norm(np.asarray(p1)[:, None, :] - piv_desc[None, :, :], axis=-1)
    perm = D.argmin(1)
    max_dist = float(D[np.arange(len(perm)), perm].max())
    assert max_dist < 1e-3, f'descriptor<->bond alignment failed (max {max_dist:.2e})'

    alpha = jnp.asarray(np.asarray(out['alpha'])[perm])
    sec_dir = jnp.asarray(np.asarray(out['sec_dir'])[perm])
    w_lig = jnp.full(len(perm), float(w_lig_mm))
    return alpha, w_lig, sec_dir


def build_surrogate_bond_energy(config, state):
    """Build the Stage-2 hinge-energy override from ``config.hinge_model``; ``None`` for the ROM.

    Config-driven selection (faithful to the project's config-first spirit): reads
    ``config.hinge_model`` (type / checkpoint / w_lig / scales / barrier), computes per-hinge
    geometry from the closed design, loads the trained net, calibrates the Gap-2 scales, and
    returns the injectable ``bond_energy_fn`` for ``forward_pipeline(bond_energy_fn=...)``. Prints a
    one-line material/model header so every run is self-documenting.
    """
    import numpy as np
    hm = getattr(config, 'hinge_model', None)
    if hm is None or getattr(hm, 'type', 'rom') != 'surrogate':
        print("[hinge_model] ROM (linear-spring ligament energy)")
        return None, None, None

    from nff.models.hinge_surrogate import (load_hinge_surrogate, build_hinge_bond_energy_fn,
                                            build_hinge_stability_fn, calibrate_scales, DOMAIN)
    topo = config.topology
    M, N = int(topo['M']), int(topo['N'])
    r = float(topo.get('r_init', 0.45)); spacing = float(topo.get('spacing', 1.0))
    alpha, w_lig, sec_dir = closed_hinge_geometry(state, M, N, r, spacing, hm.w_lig_mm)
    net, stats, eps_f = load_hinge_surrogate(hm.checkpoint)
    fr = float(getattr(hm, 'fillet_ratio', 0.16))   # design cut-tip fillet (3rd g DOF for 6-feat nets)

    if hm.calibrate:
        kst = float(np.mean(np.asarray(state.k_stretch)))
        krt = float(np.mean(np.asarray(state.k_rot)))
        ls, es = calibrate_scales(net, stats, alpha=alpha, w_lig=w_lig, k_stretch=kst, k_rot=krt,
                                  fillet_ratio=fr)
    else:
        ls, es = hm.length_scale, hm.energy_scale

    print(f"[hinge_model] SURROGATE  material={hm.material} t={hm.thickness_mm}mm "
          f"w_lig={hm.w_lig_mm}mm eps_f={eps_f}  |  {len(alpha)} hinges  "
          f"length_scale={ls:.3g}mm/u  energy_scale={es:.3g}  barrier={hm.barrier}")
    # Physical-stability design-loss term (failure-margin + OOD), so the chamfer-only objective
    # stops trading structural safety for shape. None unless a weight is set.
    w_fail = float(getattr(hm, 'w_fail', 0.0))
    w_ood = float(getattr(hm, 'w_ood', 0.0))
    m_safe = float(getattr(hm, 'm_safe', 0.8))
    bond_pairs = np.asarray(state.bond_connectivity)

    # Trust region: the surrogate carries its own (data-driven) domain in stats; a wider v2 dataset
    # auto-widens the OOD barrier here. Legacy checkpoints without it fall back to the hardcoded box.
    dom = stats.get("domain", DOMAIN)

    def _build(w_lig_arr):
        """Bond energy + stability fn for a given per-hinge ligament width [mm] (scales fixed)."""
        bond = build_hinge_bond_energy_fn(net, stats, alpha=alpha, w_lig=w_lig_arr, sec_dir=sec_dir,
                                          length_scale=ls, energy_scale=es, barrier=hm.barrier,
                                          domain=dom, fillet_ratio=fr)
        stab = build_hinge_stability_fn(net, stats, alpha=alpha, w_lig=w_lig_arr, sec_dir=sec_dir,
                                        bond_pairs=bond_pairs, length_scale=ls, domain=dom,
                                        w_fail=w_fail, w_ood=w_ood, m_safe=m_safe, fillet_ratio=fr)
        return bond, stab

    bond_energy_fn, stability_fn = _build(jnp.asarray(w_lig))
    if stability_fn is not None:
        print(f"[hinge_model]   + stability loss  w_fail={w_fail}  w_ood={w_ood}  m_safe={m_safe}")

    # Optional per-hinge LEARNABLE ligament width (option A): w_lig = 1 + 9*sigmoid(logit) in
    # [1,10]mm flows to the solver as an EXPLICIT control_params input (differentiable via jaxopt
    # implicit diff), threaded by the loss/pipeline as forward_pipeline(hinge_w_lig=...). Here we
    # only emit the init logit; the single bond_energy_fn already reads w_lig from control_params
    # when present (else its fixed closure). length_scale/energy_scale stay at the init calibration.
    w_lig_logit0 = None
    if bool(getattr(hm, 'learn_w_lig', False)):
        frac = np.clip((np.asarray(w_lig) - 1.0) / 9.0, 1e-3, 1.0 - 1e-3)
        w_lig_logit0 = jnp.asarray(np.log(frac / (1.0 - frac)))
        print(f"[hinge_model]   + LEARNABLE w_lig (per-hinge, [1,10]mm; init {float(np.mean(w_lig)):.1f}mm)")
    return bond_energy_fn, stability_fn, w_lig_logit0
