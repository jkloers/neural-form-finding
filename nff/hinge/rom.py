"""The linear-spring (ROM) hinge energy.

Stage 2 integrates ``ligament_energy_linearized`` (``nff/stages/physics/energy.py``) whenever no
learned bond energy is injected, so the ROM *is* the absence of a ``bond_energy_fn``: its constants
are the per-face ``k_stretch``, ``k_shear``, ``k_rot`` of the config's ``material:`` block. What
this module adds is the optional KINEMATIC probe, so ROM runs can still penalise compression or
shear along the hinge paths.
"""

import jax
import jax.numpy as jnp
import numpy as np

from nff.hinge.model import HingeModel


def build_rom_probe_fn(config, state, length_scale: float):
    """Per-hinge, per-load-step KINEMATIC readings for a ROM run: ``eta_a``, ``eta_s``, ``theta``.

    The linear-spring branch has no surrogate, so it used to have no probe at all and every
    probe-backed loss term was skipped no matter what its weight said. But axial compression is not
    a surrogate prediction -- ``eta_a = a / w_lig`` is a pure KINEMATIC fact read off the corotated
    local hinge frame, so it is available to the ROM exactly as it is to the surrogate, and it is
    the quantity a compression penalty needs.

    Deliberately returns ONLY the three kinematic keys. ``D`` and ``ood`` are surrogate predictions
    with no ROM counterpart; emitting zeros for them would silently report a damage-free sheet.
    ``nff/training/loss.py`` skips a term whose key is absent, so the omission is what disables
    ``damage`` and ``ood`` here.

    Shares ``hinge_kinematics`` with the surrogate probe and the offline path extractor, so all
    three reduce a deployment to the same coordinates.

    Args:
        length_scale: mm per pipeline length-unit; use ``hinge_paths.resolve_length_scale``.

    Returns:
        ``probe(face_fields, centroid_node_vectors, hinge_geometry, reference_vectors) -> dict`` with
        ``eta_a``, ``eta_s``, ``theta``, each ``(n_steps, n_hinges)``.
    """
    import numpy as np
    from nff.models.hinge_surrogate import hinge_kinematics, _unit
    from nff.utils.linalg import gather_bond_node_dofs

    pairs = jnp.asarray(np.asarray(state.bond_connectivity))

    def probe(face_fields, centroid_node_vectors, hinge_geometry, reference_vectors=None):
        geo = hinge_geometry
        sec = _unit(geo.sec_dir)

        def one_step(face_disp):
            dofs1, dofs2 = gather_bond_node_dofs(face_disp, centroid_node_vectors, pairs)
            a, sh, dRot = hinge_kinematics((dofs1, dofs2), sec, length_scale,
                                           reference_vector=reference_vectors)
            return dict(eta_a=a / geo.w_lig, eta_s=sh / geo.w_lig, theta=dRot)

        # A single (n_faces, 3) state is promoted to a one-step history, so callers get the same
        # (n_steps, n_hinges) shape whether they pass one frame or the whole scan output.
        fields = face_fields if face_fields.ndim == 3 else face_fields[None]
        return jax.vmap(one_step)(fields)

    return probe


def build_rom_hinge(config, static_features, state) -> HingeModel:
    """The ROM hinge model: no learned energy, and a kinematic probe only if a loss term reads one.

    Returns:
        ``HingeModel`` whose ``bond_energy_fn`` is ``None`` (Stage 2 falls back to the springs);
        ``probe_fn``/``geometry_fn`` are set only when ``loss_weights.compression`` or
        ``loss_weights.shear_path`` is non-zero, else every field is ``None``.
    """
    hm = getattr(config, 'hinge_model', None)
    print("[hinge_model] ROM (linear-spring ligament energy)")
    # The ROM has no learned energy, but the KINEMATIC probe is still available and is what a
    # compression penalty needs. Built only when a probe-backed weight actually asks for it, so
    # every ROM config that does not use one keeps returning (None,)*5 and is bit-identical to
    # before this branch existed.
    lw = getattr(getattr(config, 'training', None), 'loss_weights', None)
    # The guard runs BEFORE the early return below. Ordering it the other way made the raise
    # unreachable for the exact config that needs it -- damage/ood non-zero with compression
    # zero returned (None,)*5 first, the probe stayed None, and loss.py then skipped both terms
    # in silence.
    for surrogate_only in ('damage', 'ood'):
        if lw is not None and float(getattr(lw, surrogate_only, 0.0)):
            raise ValueError(
                f"loss_weights.{surrogate_only} is non-zero under hinge_model.type: rom, but "
                f"'{surrogate_only}' is a SURROGATE prediction with no linear-spring "
                "counterpart. It was silently ignored before the ROM probe existed. Set it to "
                "0, or switch to hinge_model.type: surrogate.")
    # Every KINEMATIC term the ROM probe can serve. `compression` reads eta_a, `shear_path`
    # reads eta_s; both come out of the same corotated hinge frame, so one probe covers them
    # and any future sibling only has to be named here.
    kinematic = {name: float(getattr(lw, name, 0.0)) for name in ('compression', 'shear_path')} \
        if lw is not None else {}
    if not any(kinematic.values()):
        return HingeModel(None, None, None, None, None)
    from nff.hinge.paths import build_hinge_geometry_fn, resolve_length_scale
    ls = resolve_length_scale(config)
    geometry_fn = build_hinge_geometry_fn(config, static_features, state)
    probe_fn = build_rom_probe_fn(config, state, ls)
    n_h = np.asarray(state.bond_connectivity).shape[0]
    active = "  ".join(f"{k}={v:g}" for k, v in kinematic.items() if v)
    print(f"[hinge_model]   + KINEMATIC probe  {n_h} hinges  length_scale={ls:.1f}mm/u  "
          f"w_lig={float(getattr(hm, 'w_lig_mm', 5.0)) if hm is not None else 5.0:g}mm  "
          f"-> {active}  (eta_a, eta_s = a/w_lig, s/w_lig; path extremum)")
    return HingeModel(None, probe_fn, geometry_fn, None, None)

