"""Per-hinge cut frame of the closed sheet, shared by both hinge energies.

The design ``{r, boundary}`` fixes the flat cut vertices; from them each hinge gets its RVE frame
``(alpha, sec_dir)``, reordered into Stage-2 bond order. Both the ROM probe and the surrogate read
hinge kinematics in this frame, so both are built on these helpers.

    {r, boundary} --(1)--> flat cut vertices --(2)--> per-hinge frame --(3)--> bond order
"""

import jax.numpy as jnp


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


def hinge_cut_axis(config, state, static_features, map_params):
    """(n_hinges, 2) per-hinge axial (secondary-cut) unit direction, in Stage-2 BOND order.

    The ligament's OWN axis, in the reference configuration. Public because a diagnostic needs the
    same frame the physics uses: ``eta_a``/``eta_s`` are read off this frame, so anything that talks
    about "axial vs shear" and does not use it is talking about something else.

    ⚠ Available for a ROM run exactly as for a surrogate one -- see :func:`build_rom_probe_fn`. It is
    a property of the CUT GEOMETRY, not a surrogate prediction. (Not to be confused with the
    reference bond vector Stage 2 actually integrates, which for a closed sheet is the degenerate
    ``l0_nom`` fallback along global x for every hinge.)

    ⚠ The bond-order permutation must be matched on the UNIFORM ``r_init`` design, because ``state``
    is always the uniform sheet; matching it on a trained (or randomly initialised) design aligns
    positions that no longer correspond. The axis itself then comes from the ACTUAL design. This
    mirrors ``nff.hinge.surrogate.build_surrogate_hinge`` exactly -- deliberately, so both read the same frame.
    """
    import numpy as np
    from nff.topology.hinge_descriptor import build_hinge_descriptor_structure

    topo = config.topology
    ru = float(topo.get('r_init', 0.45))
    hs = build_hinge_descriptor_structure(int(topo['M']), int(topo['N']), ref_r=ru)

    uniform = {'z': jnp.full_like(jnp.asarray(map_params['z']),
                                  float(np.log(ru / (1.0 - ru)))),
               'bnd_logits': jnp.zeros_like(jnp.asarray(map_params['bnd_logits']))}
    perm = _bond_order_perm(state, hs, _flat_coords_from_design(static_features, uniform))
    _, sec = _alpha_sec_bond_order(hs, perm, _flat_coords_from_design(static_features, map_params))

    sec = np.asarray(sec, dtype=float)
    return sec / np.maximum(np.linalg.norm(sec, axis=-1, keepdims=True), 1e-30)

