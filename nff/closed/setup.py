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
    spacing_y = float(topo.get('spacing_y', spacing))

    tessellation = build_closed_tessellation(M, N, r=r_init, spacing=spacing, spacing_y=spacing_y)
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
    # y-roller: pin y ONLY (DOF 1) so the face slides freely in x under an x-load. Removes the
    # rigid rotation mode of a single-tile pinch without blocking the pull (an x-force face can
    # still be y-rollered — different DOFs).
    for f in topo.get('y_roller_faces', []) or []:
        tessellation.set_face_dofs(int(f), [1])

    state = CentroidalState.from_tessellation(tessellation, target_cfg=config.target)
    return state, tessellation


def init_closed_les_params(config):
    """Init design params and static LES structure for the closed_les map type.

    Params: {'z': (rows, cols) latent for r = sigmoid(z), init r_init;
             'bnd_logits': (n_logits,) per-edge logits -> ordered boundary sliders}.
    The ordered-boundary + r∈(0,1) parameterization guarantees a valid (non-self-
    intersecting) flat tessellation, so no validity loss is needed.

    Optional RANDOM starting design (topology ``init_noise`` > 0, seeded by ``init_seed``): adds
    Gaussian noise to z and bnd_logits. Still valid by construction (r = sigmoid stays in (0,1);
    ordered-softmax boundary stays convex), so different random starts probe whether the optimizer
    is init-sensitive / stuck at the uniform symmetric attractor. ``init_noise`` = 0 (default) is the
    deterministic uniform init -- backward-compatible.
    """
    topo = config.topology
    M, N = int(topo['M']), int(topo['N'])
    r_init = float(topo.get('r_init', 0.45))
    spacing = float(topo.get('spacing', 1.0))
    spacing_y = float(topo.get('spacing_y', spacing))

    struct = build_deploy_structure(M, N)
    sliders = build_boundary_edges(struct, spacing=spacing, spacing_y=spacing_y)

    z_init = float(np.log(r_init / (1.0 - r_init)))         # sigmoid(z_init) = r_init
    z = np.full((struct['rows'], struct['cols']), z_init)
    bnd = np.asarray(sliders['init_logits'], dtype=float)
    noise = float(topo.get('init_noise', 0.0))
    if noise > 0.0:
        rng = np.random.default_rng(int(topo.get('init_seed', 0)))
        z = z + rng.normal(0.0, noise, size=z.shape)
        bnd = bnd + rng.normal(0.0, noise, size=bnd.shape)
    params = {'z': jnp.asarray(z, dtype=float), 'bnd_logits': jnp.asarray(bnd, dtype=float)}
    static_features = {'struct': struct, 'sliders': sliders, 'closed_les': True}
    return params, static_features


# ── the simple logic, factored into three small steps ──────────────────────────────
#   {r, boundary} --(1)--> flat cut vertices --(2)--> per-hinge frame --(3)--> bond order

def _flat_coords_from_design(static_features, map_params):
    """(1) The flat closed-sheet cut vertices for the CURRENT design (differentiable in z/boundary)."""
    import jax
    from nff.topology.closed_builder_jax import solve_cut_vertices_jax, boundary_flat_from_logits
    boundary = boundary_flat_from_logits(static_features['sliders'], map_params['bnd_logits'])
    return solve_cut_vertices_jax(static_features['struct'], boundary, jax.nn.sigmoid(map_params['z']))


def _bond_order_perm(state, hs, coords):
    """(3) The design-INVARIANT permutation descriptor-order -> Stage-2 bond order (by pivot position)."""
    import numpy as np
    from nff.stages.geometry import hinge_vertex_positions
    piv = np.asarray(coords)[np.asarray(hs['pivot_pid'])]
    p1, _ = hinge_vertex_positions(state.face_centroids, state.centroid_node_vectors,
                                   state.hinge_node_pairs)
    D = np.linalg.norm(np.asarray(p1)[:, None, :] - piv[None, :, :], axis=-1)
    perm = D.argmin(1)
    assert float(D[np.arange(len(perm)), perm].max()) < 1e-3, 'descriptor<->bond alignment failed'
    return jnp.asarray(perm)


def _alpha_sec_bond_order(hs, perm, coords):
    """(2)+(3) Per-hinge ``(alpha, sec_dir)`` in bond order from flat vertices (RVE frame, in-graph)."""
    from nff.topology.hinge_descriptor import compute_hinge_frame
    frame = compute_hinge_frame(hs, coords)
    return frame['alpha'][perm], frame['sec_dir'][perm]


def build_surrogate_energy(config, static_features, state, init_map_params):
    """Config-driven Stage-2 surrogate energy with DESIGN-TRACKED hinge geometry.

    The energy + probe are built ONCE (they close over only the net, the Gap-2 scales, and the
    fixed fillet). The design-dependent ``HingeGeometry(w_lig, alpha, sec_dir)`` is supplied PER STEP
    as data by ``hinge_geometry_from_design(map_params)`` and the caller threads it into the solver via
    ``forward_pipeline(hinge_geometry=)`` (control_params) and into ``probe_fn``. Because it is an
    explicit solver input (not closed over), jaxopt's implicit diff carries the full ``d/d(design)``.

    Returns ``(bond_energy_fn, probe_fn, hinge_geometry_from_design, damage_fn, w_lig_logit0)``;
    ``hinge_geometry_from_design(map_params) -> HingeGeometry``, ``probe_fn(fields, cnv, geom, ref) ->
    per-step per-hinge surrogate readings`` (the design loss aggregates them; the weights live in
    ``loss_weights:``), ``damage_fn(node_disp, geom, ref) -> per-hinge D`` (diagnostics/visuals).
    ``(None,)*5`` for the ROM.
    """
    import numpy as np
    hm = getattr(config, 'hinge_model', None)
    if hm is None or getattr(hm, 'type', 'rom') != 'surrogate':
        print("[hinge_model] ROM (linear-spring ligament energy)")
        return None, None, None, None, None

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

    return bond_energy, probe_fn, hinge_geometry_from_design, damage_fn, w_lig_logit0
