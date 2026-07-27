"""Observe the ``(a, s, theta)`` displacement paths that real hinges ride during a deployment.

WHY THIS EXISTS. The surrogate is trained on the CalculiX oracle, and each oracle job is one
``DeploymentRay`` -- a *straight, proportional* path ``u(lambda) = lambda * u1`` whose direction is
LHS-sampled over ``(theta1, eta_a, eta_s)`` (``nff/rve/dataset.py:sample_jobs``). That sampling is a
guess: it covers a generous box because we did not know which paths a real tessellation actually
traverses. Training a surrogate on paths the pipeline never rides wastes oracle budget on one side
and leaves the ridden region under-resolved on the other.

This module closes that loop by MEASURING the paths instead of guessing them. A Stage-2 deployment
already computes every hinge's kinematics at every load step -- ``SolutionData.fields`` is
``(n_steps, n_faces, 3)`` -- but the reduction to ``(a, s, theta)`` is consumed inside the bond
energy and then discarded. Here we replay that history through the SAME reduction the surrogate
uses (``hinge_surrogate.hinge_kinematics``) and keep the whole trajectory.

The output feeds back into ``sample_jobs``: ``fit_deployment_rays`` turns observed paths into the
``(theta1_deg, eta_a, eta_s)`` triples the oracle already understands, and ``path_diagnostics``
reports how well that straight-ray idealization actually holds (see ``straightness``).

WORKS WITHOUT A SURROGATE -- which is the point. ``build_surrogate_energy`` returns ``(None,)*5`` on
the ROM branch, so a ROM run has no ``HingeGeometry`` and hence no cut frame to project onto.
``build_hinge_geometry`` below reconstructs it from the design alone (the three factored helpers in
``nff.closed.setup`` are surrogate-independent), so a plain linear-spring ROM deployment -- no
trained net, no checkpoint -- yields paths in exactly the coordinates the oracle consumes.

CAVEAT. These paths inherit the physics of whatever hinge model produced them. A ROM path is only
as trustworthy as the ROM's ``k_stretch/k_shear/k_rot`` calibration: linear springs never yield, so
they do not reproduce the load-redistribution that plasticity causes near failure. Read the paths
as a first-order map of WHICH REGION of ``(a, s, theta)`` space the tessellation visits and in what
proportion -- not as ground truth for the path shape at high load.
"""

from dataclasses import dataclass, asdict
from typing import Optional

import json
import numpy as np
import jax
import jax.numpy as jnp

from nff.models.hinge_surrogate import HingeGeometry, hinge_kinematics, DOMAIN
from nff.stages.physics.kinematics import face_to_node_kinematics_fn


# ── hinge geometry, independent of the hinge model ────────────────────────────────

def build_hinge_geometry(config, static_features, state, map_params,
                         w_lig_mm: Optional[float] = None) -> HingeGeometry:
    """Per-hinge ``HingeGeometry(w_lig, alpha, sec_dir)`` for the CURRENT design.

    This is the ROM-compatible twin of ``setup.build_surrogate_energy``'s inner
    ``hinge_geometry_from_design``: identical construction, but it never touches a surrogate
    checkpoint, so it also works for ``hinge_model.type: rom`` runs (where ``build_surrogate_energy``
    returns ``(None,)*5``). ``tests/test_hinge_paths.py`` pins the two to agree exactly.

    ``w_lig`` comes from the learnable per-hinge logit when the design carries one, else from
    ``w_lig_mm`` / ``config.hinge_model.w_lig_mm`` / 5.0 mm as a uniform fallback -- the ROM has no
    learnable ligament width, so its hinges are all the manufactured width.
    """
    from nff.closed.setup import (_flat_coords_from_design, _bond_order_perm, _alpha_sec_bond_order)
    from nff.topology.hinge_descriptor import build_hinge_descriptor_structure
    from nff.models.hinge_surrogate import w_lig_from_logit

    topo = config.topology
    M, N = int(topo['M']), int(topo['N'])
    r_init = float(topo.get('r_init', 0.45))
    hs = build_hinge_descriptor_structure(M, N, ref_r=r_init)

    # The bond-order permutation must be matched on the UNIFORM-r sheet, because `state` is always
    # built from build_closed_tessellation at r_init -- matching against a trained (non-uniform)
    # design would misalign the position-based nearest-pivot search. Same argument as setup.py:181.
    z_uniform = float(np.log(r_init / (1.0 - r_init)))
    uni = {'z': jnp.full_like(jnp.asarray(map_params['z']), z_uniform),
           'bnd_logits': jnp.zeros_like(jnp.asarray(map_params['bnd_logits']))}
    perm = _bond_order_perm(state, hs, _flat_coords_from_design(static_features, uni))

    alpha, sec_dir = _alpha_sec_bond_order(hs, perm, _flat_coords_from_design(static_features, map_params))

    n_hinges = np.asarray(state.bond_connectivity).shape[0]
    if isinstance(map_params, dict) and 'w_lig_logit' in map_params:
        w_lig = w_lig_from_logit(map_params['w_lig_logit'])
    else:
        if w_lig_mm is None:
            hm = getattr(config, 'hinge_model', None)
            w_lig_mm = float(getattr(hm, 'w_lig_mm', 5.0)) if hm is not None else 5.0
        w_lig = jnp.full((n_hinges,), float(w_lig_mm))
    return HingeGeometry(w_lig=w_lig, alpha=alpha, sec_dir=sec_dir)


# ── the extraction itself ─────────────────────────────────────────────────────────

@dataclass
class HingePaths:
    """Per-hinge ``(a, s, theta)`` trajectories over a deployment.

    ``u`` is physical: ``a``/``s`` in **mm**, ``theta`` in **rad**. ``eta`` is the dimensionless form
    the oracle samples in -- ``(a/w_lig, s/w_lig, theta)`` -- so it is directly comparable to
    ``DeploymentRay(theta1_deg, eta_a, eta_s)`` and to ``DOMAIN``.

    Step 0 is the UNDEFORMED state (all zeros), prepended so every path starts at the origin like an
    oracle ray does; ``solution.fields`` itself only holds the solved increments.
    """
    u: np.ndarray                 # (n_steps+1, n_hinges, 3)  (a [mm], s [mm], theta [rad])
    eta: np.ndarray               # (n_steps+1, n_hinges, 3)  (a/w_lig, s/w_lig, theta [rad])
    w_lig: np.ndarray             # (n_hinges,) [mm]
    alpha: np.ndarray             # (n_hinges,) [rad] RVE-frame hinge opening angle
    length_scale: float           # mm per pipeline unit used for the reduction
    load_fraction: np.ndarray     # (n_steps+1,) pseudo-time lambda in [0, 1]

    @property
    def n_hinges(self) -> int:
        return self.u.shape[1]

    @property
    def n_steps(self) -> int:
        return self.u.shape[0] - 1


def extract_hinge_paths(solution, state, geometry: HingeGeometry, length_scale: float,
                        reference_bond_vectors=None) -> HingePaths:
    """Replay a solved deployment into per-hinge ``(a, s, theta)`` paths.

    Uses the EXACT gather the Stage-2 strain energy uses -- ``face_to_node_kinematics_fn`` then
    reshape to ``(n_faces * n_nodes, 3)`` then index by ``bond_connectivity`` (``energy.py:320``) --
    and then the surrogate's own ``hinge_kinematics`` reduction, so the numbers are byte-identical
    to what a surrogate bond energy would have seen at each step. Vectorized over steps with ``vmap``,
    mirroring ``energy.compute_ligament_strains_history``.

    Args:
        solution: ``SolutionData`` with ``.fields`` (n_steps, n_faces, 3).
        state: the ``CentroidalState`` the solve ran on (for connectivity + reference geometry).
        geometry: per-hinge ``HingeGeometry``; ``sec_dir`` defines the axial/shear split.
        length_scale: mm per pipeline length-unit (converts the reduction to physical mm).
        reference_bond_vectors: (n_hinges, 2) rest bond vectors. ``None`` -> recomputed from
            ``state`` exactly as ``ReferenceGeometry`` does. Passing the solve's own array (from
            ``forward_pipeline``'s result) is preferred when available.

    Returns:
        ``HingePaths`` with the undeformed origin prepended.
    """
    from nff.stages.geometry import build_reference_bond_vectors

    fields = jnp.asarray(solution.fields)                      # (n_steps, n_faces, 3)
    cnv = jnp.asarray(state.centroid_node_vectors)
    bond = np.asarray(state.bond_connectivity)
    ref = jnp.asarray(build_reference_bond_vectors(state) if reference_bond_vectors is None
                      else reference_bond_vectors)
    n_faces, n_nodes, _ = cnv.shape

    sec = geometry.sec_dir / jnp.linalg.norm(geometry.sec_dir, axis=-1, keepdims=True)

    def step_kin(face_disp):
        nodes = face_to_node_kinematics_fn(face_disp, cnv).reshape((n_faces * n_nodes, 3))
        a, s, th = hinge_kinematics((nodes[bond[:, 0]], nodes[bond[:, 1]]), sec, length_scale,
                                    reference_vector=ref)
        return jnp.stack([a, s, th], axis=-1)                  # (n_hinges, 3)

    u = np.asarray(jax.vmap(step_kin)(fields))                 # (n_steps, n_hinges, 3)
    u = np.concatenate([np.zeros_like(u[:1]), u], axis=0)      # prepend the undeformed origin

    w_lig = np.asarray(geometry.w_lig, dtype=float)
    eta = u.copy()
    eta[:, :, 0] /= w_lig[None, :]
    eta[:, :, 1] /= w_lig[None, :]

    n_steps = u.shape[0] - 1
    return HingePaths(u=u, eta=eta, w_lig=w_lig, alpha=np.asarray(geometry.alpha, dtype=float),
                      length_scale=float(length_scale),
                      load_fraction=np.linspace(0.0, 1.0, n_steps + 1))


# ── diagnostics: is the oracle's straight-ray idealization actually valid? ────────

def straightness(eta_path: np.ndarray) -> float:
    """Max deviation from the best straight ray through the origin, as a fraction of path extent.

    The oracle can only produce PROPORTIONAL paths ``u(lambda) = lambda * u1``. This measures how
    badly a real path violates that: project every point onto the unit vector towards the endpoint,
    take the largest perpendicular residual, and normalize by the endpoint norm. ``0`` = perfectly
    proportional (reproducible by one ``DeploymentRay``); ``0.1`` = the path bows 10% of its own
    length away from the straight chord and a single ray cannot represent it.

    Args:
        eta_path: (n_pts, 3) one hinge's path in ``(eta_a, eta_s, theta)``.
    """
    end = eta_path[-1]
    L = float(np.linalg.norm(end))
    if L < 1e-12:
        return 0.0                                             # hinge never moved: trivially straight
    d = end / L
    proj = eta_path @ d                                        # (n_pts,) coordinate along the chord
    perp = eta_path - proj[:, None] * d[None, :]
    return float(np.linalg.norm(perp, axis=1).max() / L)


def path_diagnostics(paths: HingePaths, domain: dict = DOMAIN) -> dict:
    """Summarize the observed paths: extent, straightness, monotonicity, domain coverage.

    ``domain`` is the surrogate's trust box (``DOMAIN``); the fractions report how much of the
    observed motion the currently-trained surrogate can actually be trusted on, and conversely how
    much of the sampled box the tessellation ever visits.
    """
    eta = paths.eta
    ends = eta[-1]                                             # (n_hinges, 3)
    per_hinge = [straightness(eta[:, h, :]) for h in range(paths.n_hinges)]

    # monotonic in pseudo-time? a non-monotonic component means the hinge REVERSES during
    # deployment -- something no proportional ray can ever reproduce, not even approximately.
    d = np.diff(eta, axis=0)
    mono = [float(np.mean(np.all(np.sign(d[:, :, k]) >= -1e-12, axis=0) |
                          np.all(np.sign(d[:, :, k]) <= 1e-12, axis=0))) for k in range(3)]

    ea, es, th = np.abs(eta[..., 0]), np.abs(eta[..., 1]), np.abs(eta[..., 2])
    in_box = ((eta[..., 0] >= 0.0) & (eta[..., 0] <= domain['eta_a_max']) &
              (es <= domain['eta_s_max']) & (th <= domain['theta_max']))
    return {
        'n_hinges': paths.n_hinges,
        'n_steps': paths.n_steps,
        'eta_a_range': [float(eta[..., 0].min()), float(eta[..., 0].max())],
        'eta_s_range': [float(eta[..., 1].min()), float(eta[..., 1].max())],
        'theta_range_deg': [float(np.degrees(eta[..., 2].min())), float(np.degrees(eta[..., 2].max()))],
        'straightness_max': float(np.max(per_hinge)),
        'straightness_mean': float(np.mean(per_hinge)),
        'straightness_per_hinge': [float(x) for x in per_hinge],
        'monotonic_frac': {'eta_a': mono[0], 'eta_s': mono[1], 'theta': mono[2]},
        'in_domain_frac': float(np.mean(in_box)),
        'endpoint_in_domain_frac': float(np.mean(in_box[-1])),
        # how much of the SAMPLED box the deployment actually occupies (wasted-budget indicator)
        'box_occupancy': {
            'eta_a': float(ends[:, 0].max() / domain['eta_a_max']) if domain['eta_a_max'] else 0.0,
            'eta_s': float(np.abs(ends[:, 1]).max() / domain['eta_s_max']) if domain['eta_s_max'] else 0.0,
            'theta': float(np.abs(ends[:, 2]).max() / domain['theta_max']) if domain['theta_max'] else 0.0,
        },
        'domain': dict(domain),
    }


# ── the bridge back into the oracle ───────────────────────────────────────────────

def fit_deployment_rays(paths: HingePaths, n_steps: int = 20, tag_prefix: str = "obs"):
    """Turn observed paths into ``DeploymentRay`` objects the CalculiX oracle can run.

    Each hinge's ENDPOINT defines one proportional ray, which is the best a single ray can do. How
    good that is depends entirely on ``straightness`` -- check ``path_diagnostics`` before trusting
    these. Returned as ``(HingeGeometry_rve, DeploymentRay)`` job pairs matching ``sample_jobs``'
    output, so they drop straight into ``run_jobs``/``generate_dataset``.

    Note the two ``HingeGeometry`` types are DIFFERENT classes: the RVE one
    (``nff.rve.hinge_function``) is ``(w_lig, alpha_deg, fillet_ratio)`` describing the FEA specimen;
    the pipeline one (``nff.models.hinge_surrogate``) is ``(w_lig, alpha, sec_dir)`` describing a
    hinge in the tessellation. This converts pipeline -> RVE.
    """
    from nff.rve.hinge_function import HingeGeometry as RVEGeometry, DeploymentRay

    jobs = []
    for h in range(paths.n_hinges):
        end = paths.eta[-1, h]
        geo = RVEGeometry(float(paths.w_lig[h]), float(np.degrees(paths.alpha[h])), 0.16)
        ray = DeploymentRay(float(np.degrees(end[2])), float(end[0]), float(end[1]),
                            n_steps, f"{tag_prefix}{h:03d}")
        jobs.append((geo, ray))
    return jobs


# ── persistence ───────────────────────────────────────────────────────────────────

def save_paths(paths: HingePaths, out_path: str, meta: Optional[dict] = None) -> dict:
    """Write ``<out_path>.npz`` (arrays) + ``<out_path>.json`` (diagnostics + meta).

    Mirrors the ``nff/rve/dataset.py`` convention (compressed npz beside a human-readable json) so
    the observed paths sit naturally next to the oracle datasets they will steer.
    """
    import os
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez_compressed(out_path + ".npz", u=paths.u, eta=paths.eta, w_lig=paths.w_lig,
                        alpha=paths.alpha, load_fraction=paths.load_fraction,
                        length_scale=np.asarray(paths.length_scale))
    summary = {'diagnostics': path_diagnostics(paths),
               'length_scale_mm_per_unit': paths.length_scale,
               'w_lig_mm': [float(x) for x in paths.w_lig],
               'alpha_deg': [float(np.degrees(a)) for a in paths.alpha],
               **(meta or {})}
    with open(out_path + ".json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def load_paths(out_path: str) -> HingePaths:
    """Inverse of ``save_paths`` (arrays only)."""
    z = np.load(out_path + ".npz")
    return HingePaths(u=z['u'], eta=z['eta'], w_lig=z['w_lig'], alpha=z['alpha'],
                      length_scale=float(z['length_scale']), load_fraction=z['load_fraction'])
