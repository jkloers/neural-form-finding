"""Where force travels through a deployed tessellation -- the structural load path.

Stage 2 treats every tile as a RIGID body ``(dx, dy, dtheta)``, so a tile stores no energy and every
newton crossing the sheet crosses a tile-to-tile interaction. There are exactly two such channels --
the hinge LIGAMENT (ROM springs or the learned surrogate) and the void-angle BEARING contact, which
acts on the same face pairs at the two void corners of each bond. The load path is therefore a
weighted graph on tile adjacency, exactly, with no interpolation anywhere.

Forces come from ``jax.grad`` of the SOLVER'S OWN pair energy rather than hand-derived algebra, so
ROM, surrogate and contact are covered by one mechanism that cannot drift from what the solve saw.
(Re-deriving strains here is how ``compute_ligament_strains`` ended up hardcoding the nonlinear
measure even for linearized solves.)

Sign conventions, fixed once:
  * ``f_on_face = -dE/du_face`` -- the force the interaction exerts ON that tile.
  * ``t`` is the force exerted on face ``j``, i.e. what ``i`` pushes into ``j``.
  * ``axial = -(t . unit(c_j - c_i))``, so **positive means TENSION** (the link resists separation).
  * ``sigma`` follows the usual continuum sign: positive normal stress is tension.

Drawn by :mod:`nff.closed.load_path_viz`. Companion to :mod:`nff.closed.hinge_paths`, which reads the
same solve at the opposite scale -- what each hinge experiences, in its own cut frame.
"""

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from nff.stages.physics.energy import (build_potential_energy, contact_energy, ligament_energy,
                                       ligament_energy_linearized)
from nff.stages.physics.params import ReferenceGeometry, build_control_params
from nff.utils.linalg import _face_to_node_displacement, face_to_node_kinematics_fn, void_angles


# ── Container ─────────────────────────────────────────────────────────────────────────────────

@dataclass
class ForceField:
    """Final-state load path of one deployed tessellation under one set of BCs.

    Every per-hinge array is in ``bond_connectivity`` order, the same order used by
    ``hinge_paths`` and the per-hinge damage arrays, so they can be indexed together.

    Attrs:
        face_xy:     (n_faces, 2)  deployed tile centroids.
        face_nodes:  (n_faces, n_nodes, 2) deployed tile polygons.
        face_areas:  (n_faces,)    deployed tile areas.
        hinge_xy:    (n_hinges, 2) deployed hinge midpoints.
        face_pairs:  (n_hinges, 2) the two tiles each hinge joins, ``(i, j)``.
        t_lig:       (n_hinges, 2) ligament force exerted on face j, global frame.
        m_lig:       (n_hinges, 2) ligament moment on face i and on face j.
        t_con:       (n_hinges, 2) bearing force on face j (both void corners summed).
        m_con:       (n_hinges, 2) bearing moment on face i and on face j.
        hinge_axis:  (n_hinges, 2) COROTATED cut axis per hinge; zeros when unavailable.
        axial:       (n_hinges,)   signed on the centroid-centroid axis, + = tension.
        transverse:  (n_hinges,)   perpendicular component of the same force.
        energy:      (n_hinges,)   stored ligament energy.
        sigma:       (n_faces, 2, 2) virial tile stress, UNSYMMETRISED (see ``tile_stress``).
        f_ext:       (n_faces, 3)  applied external generalized load.
        residual:    (n_faces, 3)  equilibrium residual; the convergence guard.
        free_mask:   (n_faces, 3)  True where the DOF is free (residual must vanish there).
        boundary_xy: (n_b, 2)      deployed outline, for masking the continuum field.
        checks:      dict          the exactness assertions and their measured values.
    """

    face_xy: Any
    face_nodes: Any
    face_areas: Any
    hinge_xy: Any
    face_pairs: Any
    t_lig: Any
    m_lig: Any
    t_con: Any
    m_con: Any
    hinge_axis: Any
    axial: Any
    transverse: Any
    energy: Any
    sigma: Any
    f_ext: Any
    residual: Any
    free_mask: Any
    boundary_xy: Any
    checks: Dict[str, Any]

    @property
    def n_hinges(self) -> int:
        return int(self.face_pairs.shape[0])

    @property
    def n_faces(self) -> int:
        return int(self.face_xy.shape[0])

    @property
    def t_total(self):
        """(n_hinges, 2) total force crossing each hinge, both channels."""
        return self.t_lig + self.t_con

    @property
    def m_total(self):
        """(n_hinges, 2) total moment on face i and on face j, about their own centroids."""
        return self.m_lig + self.m_con

    @property
    def f0(self) -> float:
        """Magnitude of the applied LOADING -- the normaliser every share is quoted against.

        The sum of the per-tile force magnitudes, NOT the resultant of the load set. A
        self-equilibrated load case -- a couple, +F on one half of an edge and -F on the other --
        has a resultant of zero while loading the sheet perfectly well, and normalising by it
        destroys the whole reading. Measured on the shear-couple row of the inverse-design matrix:
        the resultant came out at 5.7e-14 rather than exactly zero, sailed past the old `> 0`
        guard, and inflated every normalised force by ~1e15 (f_norm max 8.1e15). `draw_tapered`
        then asked for stroke widths of order 1e17 points and the tension channel silently stopped
        rendering, on a field whose mode census was perfectly healthy.

        For a load set whose forces are parallel -- every other row of that figure -- this is
        identical to the resultant, so the change is invisible wherever the old definition was
        already sound.
        """
        applied = float(np.linalg.norm(np.asarray(self.f_ext)[:, :2], axis=1).sum())
        # A tolerance, not `> 0`: the degenerate case arrives as float noise, never as a clean zero.
        return applied if applied > 1e-9 else 1.0

    @property
    def tile_scale(self) -> float:
        """Representative tile edge length, for non-dimensionalising moments."""
        return float(np.sqrt(np.mean(np.asarray(self.face_areas))))


# ── Deployed geometry ─────────────────────────────────────────────────────────────────────────

def deployed_nodes(face_centroids, centroid_node_vectors, disp):
    """(n_faces, n_nodes, 2) deployed tile vertices from the rigid-tile kinematics."""
    fc = np.asarray(face_centroids) + np.asarray(disp[:, :2])
    cnv = np.asarray(centroid_node_vectors)
    th = np.asarray(disp[:, 2])
    ct, st = np.cos(th)[:, None], np.sin(th)[:, None]
    rx = ct * cnv[:, :, 0] - st * cnv[:, :, 1]
    ry = st * cnv[:, :, 0] + ct * cnv[:, :, 1]
    return fc[:, None, :] + np.stack([rx, ry], axis=-1)


def polygon_areas(nodes):
    """(n_faces,) shoelace areas of the deployed tile polygons."""
    x, y = nodes[:, :, 0], nodes[:, :, 1]
    return 0.5 * np.abs(np.sum(x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y, axis=1))


# ── The two force channels ────────────────────────────────────────────────────────────────────

def ligament_pair_forces(face_disp, centroid_node_vectors, bond_pairs, bond_energy_fn, bond_kwargs):
    """Per-hinge generalized force the LIGAMENT exerts on each of its two tiles.

    The bond energy is a function of NODE DOFs, so the differentiation must chain through the
    rigid-tile kinematics to land on the tile's ``(dx, dy, dtheta)`` -- that chain is exactly what
    puts the lever arm into the moment component.

    Each hinge's energy depends only on its own row of ``(u_i, u_j)``, so the gradient of the SUM is
    already the per-hinge gradient: one ``jax.grad`` call, no vmap, no approximation.

    Returns:
        ``(f_on_i, f_on_j, energy)`` -- forces (n_hinges, 3), stored energy (n_hinges,).
    """
    n_faces, n_nodes, _ = centroid_node_vectors.shape
    cnv_flat = centroid_node_vectors.reshape(n_faces * n_nodes, 2)
    i_nodes, j_nodes = bond_pairs[:, 0], bond_pairs[:, 1]
    cnv_i, cnv_j = cnv_flat[i_nodes], cnv_flat[j_nodes]

    node_dofs = jax.vmap(_face_to_node_displacement)

    def energies(u_i, u_j):
        return bond_energy_fn((node_dofs(u_i, cnv_i), node_dofs(u_j, cnv_j)), **bond_kwargs)

    face_i, face_j = np.asarray(i_nodes) // n_nodes, np.asarray(j_nodes) // n_nodes
    u_i, u_j = face_disp[face_i], face_disp[face_j]

    grad_i, grad_j = jax.grad(lambda a, b: jnp.sum(energies(a, b)), argnums=(0, 1))(u_i, u_j)
    return -grad_i, -grad_j, energies(u_i, u_j)


def contact_pair_forces(face_disp, geometry, bond_pairs, contact_params):
    """Per-hinge generalized force the BEARING contact exerts on each of its two tiles.

    The void-angle barrier couples the same two faces as the bond, at the bond's two void corners,
    so it attributes onto the same graph edges. ``jax.jacrev`` over the per-corner energy vector
    keeps the attribution exact without re-deriving the void geometry.

    Returns ``(f_on_i, f_on_j)`` each (n_hinges, 3); exactly zero when no void has closed.
    """
    n_faces, n_nodes, _ = geometry.centroid_node_vectors.shape
    n_hinges = bond_pairs.shape[0]
    face_i = np.asarray(bond_pairs[:, 0]) // n_nodes
    face_j = np.asarray(bond_pairs[:, 1]) // n_nodes
    zeros = jnp.zeros((n_hinges, 3))

    if contact_params is None:
        return zeros, zeros

    fc = geometry.face_centroids
    cnv = geometry.centroid_node_vectors

    def corner_energies(u):
        node_disp = face_to_node_kinematics_fn(u, cnv)[:, :, :2]
        nodes_xy = fc[:, None] + cnv + node_disp
        return contact_energy(void_angles(nodes_xy, bond_pairs), **contact_params._asdict())

    # No void has closed -> the whole channel is off and the (expensive) Jacobian is pure zeros.
    if float(jnp.max(jnp.abs(corner_energies(face_disp)))) == 0.0:
        return zeros, zeros

    jac = jax.jacrev(corner_energies)(face_disp)                 # (2*n_hinges, n_faces, 3)
    jac = jac.reshape(2, n_hinges, n_faces, 3).sum(axis=0)       # both void corners of each bond
    rows = np.arange(n_hinges)
    return -jac[rows, face_i], -jac[rows, face_j]


# ── Interface decomposition ───────────────────────────────────────────────────────────────────

def _cross(a, b):
    """z-component of the 2D cross product, row-wise."""
    a, b = np.asarray(a), np.asarray(b)
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def branch_frame(field: 'ForceField'):
    """``(n_hat, t_hat)`` per hinge -- the frame the interface force is resolved in.

    These hinges are POINT contacts: every face pair is joined by exactly one bond node pair, so
    there is no contact plane and hence no normal perpendicular to one. The frame is therefore the
    BRANCH vector (centroid to centroid) and its perpendicular, which is the granular-contact
    convention -- the normal direction of a contact between two bodies is the line of centres.

    (The masonry rigid-block convention, where ``n_hat`` is perpendicular to a planar interface and
    the transmitted moment is implicit in the distribution of normal force across it, does not apply
    to a point hinge. Here the moment is explicit instead; see :func:`transmitted_moment`.)

    ⚠⚠ **THE FALLBACK, not the frame.** Use :func:`resolve_frame`, which prefers the hinge's own
    corotated cut axis. This one rotates with each hinge, so normal-vs-tangential becomes a property
    of the force AND the hinge's orientation in the lattice: under a near-uniaxial load the same
    physical force reads as "normal" at a hinge aligned with it and "shear" at a hinge across it
    (r = 0.855 on the 70-degree sheet; see :func:`orientation_confound`). Against the real cut frame
    it disagrees on ~47% of hinges.

    **Never partition a figure or a dataset on it.** Splitting the load-path figure into normal and
    tangential panels cut one connected load path in half and made a lattice artifact look like two
    orthogonal force networks.
    """
    face_xy = np.asarray(field.face_xy)
    pairs = np.asarray(field.face_pairs)
    span = face_xy[pairs[:, 1]] - face_xy[pairs[:, 0]]
    n_hat = span / np.maximum(np.linalg.norm(span, axis=-1, keepdims=True), 1e-30)
    t_hat = np.stack([-n_hat[:, 1], n_hat[:, 0]], axis=-1)
    return n_hat, t_hat


def transmitted_moment(field: 'ForceField', side: int = 1):
    """``(n_hinges,)`` the couple transmitted across each hinge, taken about the hinge point.

    The generalized force on a tile is ``(fx, fy, M)`` with ``M`` about that tile's OWN centroid, so
    ``M`` already contains the torque of the transmitted force about the centroid. Removing it
    leaves the pure couple the interface carries:

        ``m_e = M_j - (x_h - c_j) x t_j``

    A point hinge genuinely transmits one, and it is not small: ``couple_share`` is 0.35 on the
    9-tile ``circle_4x4``. Evaluating it from the other tile must give exactly ``-m_e`` (zero net
    moment about the hinge point), which :func:`run_checks` measures as ``moment_balance``.

    Args:
        side: 0 to read it off face i, 1 off face j.
    """
    pairs = np.asarray(field.face_pairs)
    centroid = np.asarray(field.face_xy)[pairs[:, side]]
    lever = np.asarray(field.hinge_xy) - centroid
    t_on_side = np.asarray(field.t_total) * (1.0 if side == 1 else -1.0)
    return np.asarray(field.m_total)[:, side] - _cross(lever, t_on_side)


def edge_modes(dec) -> np.ndarray:
    """(n_hinges,) 'compression' | 'tension' | 'shear' -- shear wins where it exceeds the normal.

    ⚠ A BRANCH-FRAME label: see :func:`branch_frame` and :func:`orientation_confound`. Report it,
    tint with it, never partition on it.
    """
    mode = np.where(dec['f_n'] > 0, 'tension', 'compression').astype(object)
    mode[dec['shear_dominated']] = 'shear'
    return mode


def orientation_confound(field: 'ForceField') -> float:
    """How much the normal/tangential split is really a split by HINGE ORIENTATION.

    Correlation between "this hinge's branch runs across the load axis" and "this hinge came out
    shear-dominated". Near 1 means the classification is reporting where a hinge sits in the lattice
    rather than what its force is doing, which happens whenever the force field is close to
    uniaxial -- the common case under a single applied load.

    Exists as a measured number rather than a comment because a comment did not stop the figure from
    being split on exactly this. d70 scores 0.855.
    """
    dec = interface_decomposition(field)
    axis = load_axis_of(field)
    across = np.abs(np.sum(dec['n_hat'] * axis, axis=-1)) < 0.5      # branch across the load axis
    shear = dec['shear_dominated'].astype(float)
    if np.ptp(across.astype(float)) == 0 or np.ptp(shear) == 0:
        return float('nan')                                          # one class is empty
    return float(np.corrcoef(across.astype(float), shear)[0, 1])


def load_axis_of(field: 'ForceField') -> np.ndarray:
    """Unit vector of the total applied load; falls back to +x when nothing is applied."""
    total = np.asarray(field.f_ext)[:, :2].sum(axis=0)
    norm = float(np.linalg.norm(total))
    return total / norm if norm > 0 else np.array([1.0, 0.0])


def resolve_frame(field: 'ForceField'):
    """``(n_hat, t_hat, source)`` -- the frame the interface force is resolved in.

    Prefers the hinge's OWN corotated cut axis. Falls back to the branch vector only when that is
    unavailable, because the branch frame is a geometric proxy that answers a different question:
    measured on the r7 runs the two frames disagree on ~47% of hinges, and **63-83% of the interfaces
    the branch frame called SHEAR are AXIAL in the frame the physics actually uses**. The cut frame
    is what ``eta_a``/``eta_s`` -- and therefore the ``compression`` loss term -- are read off, so a
    figure using anything else is not describing the quantity that was optimised.
    """
    axis = np.asarray(getattr(field, 'hinge_axis', None) if field.hinge_axis is not None
                      else np.zeros((field.n_hinges, 2)))
    if axis.shape == (field.n_hinges, 2) and np.linalg.norm(axis, axis=-1).min() > 1e-12:
        n_hat = axis / np.linalg.norm(axis, axis=-1, keepdims=True)
        return n_hat, np.stack([-n_hat[:, 1], n_hat[:, 0]], axis=-1), 'cut'
    n_hat, t_hat = branch_frame(field)
    return n_hat, t_hat, 'branch'


def interface_decomposition(field: 'ForceField') -> Dict[str, Any]:
    """The per-interface force state in its own frame, raw and normalised by the applied load.

    ``f_n`` and ``f_t`` are the existing ``axial`` / ``transverse`` under their structural names:
    normal (positive = tension) and tangential (shear). Normalised quantities are what the figure
    encodes, so that two sheets carrying different absolute loads stay comparable.
    """
    n_hat, t_hat, source = resolve_frame(field)
    t = np.asarray(field.t_total)
    # Resolve the transmitted force in whichever frame won. `axial`/`transverse` on the field are
    # branch-frame by construction, so they are recomputed here rather than reused.
    f_n = -np.sum(t * n_hat, axis=-1)                    # + = tension (resists separation)
    f_t = -np.sum(t * t_hat, axis=-1)
    m_e = transmitted_moment(field)
    f0, scale = field.f0, field.tile_scale

    return {
        'n_hat': n_hat, 't_hat': t_hat,
        'f_n': f_n, 'f_t': f_t, 'm_e': m_e,
        'f_mag': np.linalg.norm(np.asarray(field.t_total), axis=-1),
        'f_n_norm': f_n / f0,
        'f_t_norm': f_t / f0,
        'm_e_norm': m_e / (f0 * scale),
        'f_norm': np.linalg.norm(np.asarray(field.t_total), axis=-1) / f0,
        'shear_dominated': np.abs(f_t) > np.abs(f_n),
        'f0': f0, 'tile_scale': scale, 'frame': source,
    }


# ── Virial stress ─────────────────────────────────────────────────────────────────────────────

def virial_stress(face_xy, face_areas, face_pairs, hinge_xy, f_on_i, f_on_j):
    """(n_faces, 2, 2) Cauchy stress of the discrete network, per tile.

    ``sigma_a = (1/A_a) * sum_k (x_k - c_a) (x) f_k_on_a`` over the interactions touching tile ``a``.
    Internal interactions only -- the external load is kept out so loaded and clamped tiles are not
    silently contaminated by their own applied traction.

    NOT symmetric: hinges transmit moments, so this sheet is a micropolar (Cosserat) medium. Use
    :func:`tile_stress` to split it rather than symmetrising silently.
    """
    n_faces = face_xy.shape[0]
    sigma = np.zeros((n_faces, 2, 2))
    fi, fj = np.asarray(face_pairs[:, 0]), np.asarray(face_pairs[:, 1])

    for faces, forces in ((fi, np.asarray(f_on_i)[:, :2]), (fj, np.asarray(f_on_j)[:, :2])):
        lever = np.asarray(hinge_xy) - face_xy[faces]                       # (n_hinges, 2)
        np.add.at(sigma, faces, lever[:, :, None] * forces[:, None, :])     # outer product

    return sigma / np.maximum(face_areas, 1e-30)[:, None, None]


def tile_stress(field: 'ForceField'):
    """Split the virial stress into its symmetric part and its Cosserat couple part.

    Returns ``(sym, skew)``. Principal directions must come from ``sym``; ``skew`` is the
    moment-transmission signature a textbook Cauchy stress plot would silently discard.
    """
    sigma = np.asarray(field.sigma)
    sym = 0.5 * (sigma + np.transpose(sigma, (0, 2, 1)))
    return sym, sigma - sym


def bulk_tiles(field: 'ForceField') -> np.ndarray:
    """(n_faces,) True for tiles that carry no external load and no support reaction.

    ``virial_stress`` sums INTERNAL interactions only, so a clamped or loaded tile is not in internal
    equilibrium: its virial skew is simply the reaction torque that was deliberately left out. Any
    bulk material property has to be measured away from those tiles or it reports the grip instead.
    """
    external = np.abs(np.asarray(field.f_ext)).sum(axis=1) > 0
    constrained = ~np.asarray(field.free_mask).all(axis=1)
    return ~(external | constrained)


# ── Metrics ───────────────────────────────────────────────────────────────────────────────────

def _gini(x):
    """Gini coefficient of a non-negative array; 0 = perfectly even, 1 = one edge carries all."""
    x = np.sort(np.asarray(x, dtype=float).ravel())
    n = x.size
    total = x.sum()
    if n == 0 or total <= 0:
        return 0.0
    return float((2.0 * np.arange(1, n + 1) - n - 1).dot(x) / (n * total))


def force_budget(field: 'ForceField') -> Dict[str, float]:
    """The scalar scorecard: how much load rides each KIND of force, and how channelled it is.

    The four kind-shares sum to 1 by construction, so they read directly as a stacked bar and are
    comparable between tessellations of different size.

    ⚠ ``compression_share`` / ``tension_share`` / ``shear_share`` are BRANCH-FRAME readings, so they
    inherit the orientation confound documented on :func:`branch_frame`: on a lattice under a
    near-uniaxial load, "shear" partly means "hinges lying across the load". Quote them with
    :func:`orientation_confound` beside them, and never split anything on them.
    """
    # Read the RESOLVED frame, not the stored branch-frame arrays. `axial`/`transverse` on the
    # field are branch-frame by construction; using them here would report shares in a different
    # frame from every other number the tool produces.
    dec = interface_decomposition(field)
    axial = dec['f_n']
    transverse = np.abs(dec['f_t'])
    bearing = np.linalg.norm(np.asarray(field.t_con), axis=-1)
    total = np.abs(axial).sum() + transverse.sum() + bearing.sum()
    total = total if total > 0 else 1.0

    mag = np.linalg.norm(np.asarray(field.t_total), axis=-1)
    peak = mag.max() if mag.size and mag.max() > 0 else 1.0

    # ⚠ Measured over BULK tiles only. Including grip tiles reports the reaction torque that
    # `virial_stress` intentionally omits, not the material's Cosserat content -- on a 16-tile sheet
    # where half the tiles are gripped that artifact is the entire signal (0.35 vs 0.007).
    sym, skew = tile_stress(field)
    bulk = bulk_tiles(field)
    if not bulk.any():
        bulk = np.ones(field.n_faces, dtype=bool)

    return {
        'compression_share': float(np.maximum(-axial, 0).sum() / total),
        'tension_share':     float(np.maximum(axial, 0).sum() / total),
        'shear_share':       float(transverse.sum() / total),
        'bearing_share':     float(bearing.sum() / total),
        'couple_share':      float(np.linalg.norm(skew[bulk]) / max(np.linalg.norm(sym[bulk]), 1e-30)),
        'bulk_tiles':        int(bulk.sum()),
        'active_fraction':   float((mag > 0.1 * peak).mean()) if mag.size else 0.0,
        'gini':              _gini(mag),
        'peak_force':        float(peak),
        'frame':             dec['frame'],
    }


# ── Extraction ────────────────────────────────────────────────────────────────────────────────

def corotate_axis(axis, disp, face_pairs):
    """Rotate each hinge's reference cut axis by the MEAN rotation of the two tiles it joins.

    The cut axis is a reference-configuration quantity; the force it is projected on is deployed.
    Corotating by the mean face rotation is exactly what ``hinge_kinematics`` does before reading
    ``(a, s)``, so the figure and the loss end up in the same frame rather than two frames that
    happen to share a name.
    """
    if axis is None:
        return None
    theta = np.asarray(disp)[:, 2]
    mean_rot = 0.5 * (theta[np.asarray(face_pairs)[:, 0]] + theta[np.asarray(face_pairs)[:, 1]])
    c, s_ = np.cos(mean_rot), np.sin(mean_rot)
    axis = np.asarray(axis, dtype=float)
    return np.stack([c * axis[:, 0] - s_ * axis[:, 1],
                     s_ * axis[:, 0] + c * axis[:, 1]], axis=-1)


def extract_force_field(results, physics_cfg, *, bond_energy_fn=None, hinge_geometry=None,
                        cut_axis=None, step: int = -1, tol: float = 1e-6) -> ForceField:
    """Final-state load path from a ``forward_pipeline`` result.

    ``control_params`` is rebuilt here with the same public ``build_control_params`` factory Stage 2
    uses, from the ``valid_state`` the pipeline returned -- the pipeline does not expose its own copy
    and is not worth widening for a diagnostic.

    Args:
        results: the dict from ``forward_pipeline`` (needs ``valid_state`` and ``solution``).
        physics_cfg: the config's ``physics`` block.
        bond_energy_fn: the surrogate bond energy the solve used, or None for the ROM springs.
        hinge_geometry: the ``HingeGeometry`` threaded into the solve, or None for the ROM.
        cut_axis: (n_hinges, 2) reference cut axis in bond order, from
            ``setup.hinge_cut_axis``. Corotated here. Omit and the branch frame is used instead.
        step: load step to read; -1 is the deployed final state.
        tol: relative tolerance for the exactness assertions.

    Returns:
        ForceField, with every assertion's measured value recorded in ``.checks``.
    """
    if getattr(physics_cfg, 'updated_lagrangian', False):
        raise NotImplementedError(
            "physics.updated_lagrangian rotates the reference bond vectors per load step, so the "
            "post-solve reference_vector is not the one the solve saw. Re-run with it disabled.")

    valid_state = results['valid_state']
    disp = jnp.asarray(results['solution'].fields[step])
    geometry = ReferenceGeometry.from_centroidal_state(valid_state)
    bond_pairs = np.asarray(geometry.bond_connectivity)

    control_params = build_control_params(
        geometry=geometry,
        k_stretch=valid_state.k_stretch, k_shear=valid_state.k_shear, k_rot=valid_state.k_rot,
        density=valid_state.density,
        k_contact=physics_cfg.k_contact, min_angle=physics_cfg.min_angle,
        cutoff_angle=physics_cfg.cutoff_angle, use_contact=physics_cfg.use_contact,
        hinge_geometry=hinge_geometry,
    )
    bond_params = control_params.mechanical_params.bond_params
    contact_params = control_params.mechanical_params.contact_params

    # ── the two channels ──
    lig_i, lig_j, energy = ligament_pair_forces(
        disp, geometry.centroid_node_vectors, bond_pairs,
        bond_energy_fn if bond_energy_fn is not None else _rom_bond_energy(physics_cfg),
        bond_params._asdict())
    con_i, con_j = contact_pair_forces(disp, geometry, bond_pairs, contact_params)

    # ── deployed geometry ──
    nodes = deployed_nodes(geometry.face_centroids, geometry.centroid_node_vectors, disp)
    face_xy = np.asarray(geometry.face_centroids) + np.asarray(disp[:, :2])
    areas = polygon_areas(nodes)
    n_nodes = geometry.centroid_node_vectors.shape[1]
    face_pairs = np.stack([bond_pairs[:, 0] // n_nodes, bond_pairs[:, 1] // n_nodes], axis=-1)
    flat = nodes.reshape(-1, 2)
    hinge_xy = 0.5 * (flat[bond_pairs[:, 0]] + flat[bond_pairs[:, 1]])

    # ── axial / transverse split on the centroid-centroid axis ──
    span = face_xy[face_pairs[:, 1]] - face_xy[face_pairs[:, 0]]
    axis = span / np.maximum(np.linalg.norm(span, axis=-1, keepdims=True), 1e-30)
    t_lig = np.asarray(lig_j)[:, :2]
    t_con = np.asarray(con_j)[:, :2]
    t_tot = t_lig + t_con
    axial = -np.sum(t_tot * axis, axis=-1)                       # + = tension
    transverse = t_tot[:, 0] * axis[:, 1] - t_tot[:, 1] * axis[:, 0]

    # ── external load and the equilibrium residual ──
    f_ext, free_mask = _external_load(valid_state)
    potential_energy_fn = build_potential_energy(
        bond_connectivity=geometry.bond_connectivity,
        linearized_strains=physics_cfg.linearized_strains,
        use_contact=physics_cfg.use_contact,
        bond_energy_fn=bond_energy_fn,
    )
    f_int = -np.asarray(jax.grad(potential_energy_fn)(disp, control_params))
    residual = f_int + f_ext

    sigma = virial_stress(face_xy, areas, face_pairs,
                          hinge_xy, np.asarray(lig_i) + np.asarray(con_i),
                          np.asarray(lig_j) + np.asarray(con_j))

    field = ForceField(
        face_xy=face_xy, face_nodes=nodes, face_areas=areas,
        hinge_xy=hinge_xy, face_pairs=face_pairs,
        t_lig=t_lig, m_lig=np.stack([np.asarray(lig_i)[:, 2], np.asarray(lig_j)[:, 2]], axis=-1),
        t_con=t_con, m_con=np.stack([np.asarray(con_i)[:, 2], np.asarray(con_j)[:, 2]], axis=-1),
        hinge_axis=(corotate_axis(cut_axis, disp, face_pairs) if cut_axis is not None
                    else np.zeros((face_pairs.shape[0], 2))),
        axial=axial, transverse=transverse, energy=np.asarray(energy),
        sigma=sigma, f_ext=f_ext, residual=residual, free_mask=free_mask,
        boundary_xy=_boundary_xy(valid_state, nodes),
        checks={},
    )
    field.checks = run_checks(field, lig_i, lig_j, con_i, con_j, f_int, results, tol,
                              is_rom=bond_energy_fn is None)
    return field


def _rom_bond_energy(physics_cfg):
    """The spring bond energy the solver would have used, picked by the same flag the solve read."""
    return ligament_energy_linearized if physics_cfg.linearized_strains else ligament_energy


def _external_load(valid_state):
    """(n_faces, 3) applied generalized load at full deployment, and the free-DOF mask."""
    n_faces = valid_state.face_centroids.shape[0]
    f_ext = np.zeros((n_faces, 3))
    pairs = np.asarray(valid_state.loaded_face_DOF_pairs)
    if pairs.size:
        f_ext[pairs[:, 0], pairs[:, 1]] = np.asarray(valid_state.load_values)

    free_mask = np.ones((n_faces, 3), dtype=bool)
    constrained = np.asarray(valid_state.constrained_face_DOF_pairs)
    if constrained.size:
        free_mask[constrained[:, 0], constrained[:, 1]] = False
    return f_ext, free_mask


def _boundary_xy(valid_state, nodes):
    """(n_b, 2) deployed outline in the state's CCW boundary order."""
    b = np.asarray(valid_state.boundary_face_node_ids)
    return nodes[b[:, 0], b[:, 1]] if b.size else np.zeros((0, 2))


# ── Exactness assertions ──────────────────────────────────────────────────────────────────────

def run_checks(field, lig_i, lig_j, con_i, con_j, f_int, results, tol, is_rom=True) -> Dict[str, Any]:
    """Measure the identities that must hold, and say plainly whether they do.

    A load-path picture is only worth drawing if the forces behind it are exact. The equilibrium
    residual doubles as the Stage-2 convergence guard: an L-BFGS solve that stopped on its iteration
    cap rather than on its tolerance produces per-hinge quantities that are badly wrong while the
    global chamfer still looks fine, so it must be caught here rather than drawn.
    """
    scale = max(float(np.abs(field.f_ext).max()), float(np.linalg.norm(field.t_lig, axis=-1).max()), 1e-30)

    newton = float(np.abs(np.asarray(lig_i)[:, :2] + np.asarray(lig_j)[:, :2]).max()
                   + np.abs(np.asarray(con_i)[:, :2] + np.asarray(con_j)[:, :2]).max())

    # Every channel accounted for: scattering the per-hinge forces back onto the tiles must
    # reproduce the global internal force field exactly.
    scattered = np.zeros_like(f_int)
    np.add.at(scattered, field.face_pairs[:, 0], np.asarray(lig_i) + np.asarray(con_i))
    np.add.at(scattered, field.face_pairs[:, 1], np.asarray(lig_j) + np.asarray(con_j))
    attribution = float(np.abs(scattered - f_int).max())

    equilibrium = float(np.abs(field.residual[field.free_mask]).max())

    # Translational invariance of the internal energy: summed over every tile the internal forces
    # must self-cancel. This is the only independent global identity worth asserting -- the
    # applied-load / support-reaction balance follows from it, so checking THAT would just
    # re-measure the applied load (a naive "sum of all forces is zero" reads as a failure of size
    # |F_applied|, which is a mis-stated check rather than a broken solve).
    translation = float(np.abs(f_int[:, :2].sum(axis=0)).max())

    # Rotational invariance -- dU/d(global rotation), via the generator dtheta = w, du = w z x r.
    # This one is NOT an extraction check: it measures a property of the ENERGY MODEL. Closed
    # hinges have a degenerate reference bond, so build_reference_bond_vectors substitutes a
    # nominal vector of length l0_nom (1e-4) along the cut axis. That substituted offset does not
    # rotate with the configuration, so the corotational cancellation is inexact at order l0_nom
    # and the sheet's energy is very slightly frame-dependent. Reported, not asserted.
    lever = field.face_xy - field.face_xy.mean(axis=0)
    moment = f_int[:, 2] + lever[:, 0] * f_int[:, 1] - lever[:, 1] * f_int[:, 0]
    span = max(float(np.abs(lever).max()), 1e-30)
    rotation = float(np.abs(moment.sum()))

    # Moment balance about the hinge point. Newton's third law for a two-body interaction needs the
    # net MOMENT to vanish too, not just the net force, so the couple read off face i must be minus
    # the one read off face j. Same standing as `frame_dependence` below: the l0_nom substitution on
    # closed hinges makes the energy model itself slightly frame-dependent, so this lands near 1e-6
    # rather than at machine precision. Reported, not asserted.
    m_from_i, m_from_j = transmitted_moment(field, side=0), transmitted_moment(field, side=1)
    span_scale = max(float(np.linalg.norm(field.hinge_xy - field.face_xy.mean(axis=0), axis=-1).max()),
                     1e-30)
    moment_balance = float(np.abs(m_from_i + m_from_j).max()) / (scale * span_scale)

    reaction = -field.residual.copy()
    reaction[field.free_mask] = 0.0

    # Only meaningful under the ROM. `build_decompose_energy_fn` reports the SPRING stretch/shear/rot
    # split, which a surrogate run does not compute at all -- comparing against it would flag every
    # surrogate deployment as broken.
    energies = getattr(results.get('solution'), 'energies', None)
    closure = float('nan')
    if is_rom and isinstance(energies, dict):
        stored = sum(float(np.asarray(energies[k])[-1]) for k in ('stretch', 'shear', 'rot')
                     if k in energies)
        if stored > 0:
            closure = abs(float(field.energy.sum()) - stored) / stored

    checks = {
        'force_scale': scale,
        'newton_third_law': newton / scale,
        'attribution': attribution / scale,
        'equilibrium_residual': equilibrium / scale,
        'internal_translation': translation / scale,
        'frame_dependence': rotation / (scale * span),
        'moment_balance': moment_balance,
        'energy_closure': closure,
        'applied_total': [float(v) for v in field.f_ext[:, :2].sum(axis=0)],
        'reaction_total': [float(v) for v in reaction[:, :2].sum(axis=0)],
        'tol': tol,
    }
    checks['converged'] = bool(checks['equilibrium_residual'] < tol)
    checks['exact'] = bool(max(checks['newton_third_law'], checks['attribution'],
                               checks['internal_translation']) < 1e-8)
    return checks


def format_checks(checks: Dict[str, Any]) -> str:
    """Human-readable check report, with the failure spelled out rather than implied."""
    ax, ay = checks['applied_total']
    rx, ry = checks['reaction_total']
    lines = [
        "  exactness (must be ~0):",
        f"    Newton's third law      {checks['newton_third_law']:.2e}",
        f"    channel attribution     {checks['attribution']:.2e}",
        f"    internal translation    {checks['internal_translation']:.2e}",
        f"    energy closure          {checks['energy_closure']:.2e}"
        + ("" if np.isfinite(checks['energy_closure']) else "   (ROM only)"),
        "  convergence:",
        f"    equilibrium residual    {checks['equilibrium_residual']:.2e}  "
        f"(tol {checks['tol']:.0e})",
        "  energy-model properties (not extraction checks):",
        f"    rotational frame-dep.   {checks['frame_dependence']:.2e}  "
        f"(l0_nom on closed hinges)",
        f"    hinge moment balance    {checks.get('moment_balance', float('nan')):.2e}  "
        f"(same l0_nom origin)",
        "  equilibrium of the whole sheet:",
        f"    applied   ({ax:+.4g}, {ay:+.4g})",
        f"    reaction  ({rx:+.4g}, {ry:+.4g})",
    ]
    if not checks['exact']:
        lines.append("  !! FORCE EXTRACTION IS NOT EXACT -- a channel is unaccounted for.")
    if not checks['converged']:
        lines.append("  !! STAGE 2 DID NOT CONVERGE -- the load path below is NOT trustworthy.")
        lines.append("     Raise physics.solver_maxiter (use 20000) and re-run.")
    return "\n".join(lines)


# ── Driver ────────────────────────────────────────────────────────────────────────────────────

def solve_deployed_state(config, params_path: Optional[str] = None):
    """Deploy one design once, so both CLIs reach the load path the same way.

    No solution history is persisted by ``run_closed``, so a deployed state has to be re-solved from
    the config plus its ``best_params.pkl``. Shared here rather than copied into each CLI, because
    the surrogate branch has a live trap: ``hinge_geometry`` may only be threaded when a surrogate
    bond energy is actually reading it.

    Returns:
        ``(results, tessellation, bond_energy_fn, hinge_geometry, cut_axis)``.
    """
    import pickle

    from nff.closed.setup import (build_closed_initial_state, build_surrogate_energy,
                                  init_closed_les_params)
    from nff.stages.pipeline import forward_pipeline

    state, tessellation = build_closed_initial_state(config)
    config.topology.setdefault('init_noise', 0.0)
    config.topology.setdefault('init_seed', 0)
    params, static_features = init_closed_les_params(config)

    bond_energy, _, geometry_fn, _, w_lig_logit0 = build_surrogate_energy(
        config, static_features, state, params)
    if w_lig_logit0 is not None:
        params = {**params, 'w_lig_logit': w_lig_logit0}
    if params_path:
        with open(params_path, 'rb') as fh:
            params = {k: np.asarray(v) for k, v in pickle.load(fh).items()}

    hinge_geometry = (geometry_fn(params)
                      if (geometry_fn is not None and bond_energy is not None) else None)
    results = forward_pipeline(
        state, config.target, config.validity, config.physics, map_type=config.mapping.type,
        map_params=params, static_features=static_features, load_specs=config.topology['loads'],
        bond_energy_fn=bond_energy, hinge_geometry=hinge_geometry)

    from nff.closed.setup import hinge_cut_axis
    try:
        cut_axis = hinge_cut_axis(config, state, static_features, params)
    except Exception:
        cut_axis = None                       # no cut frame: the caller falls back to the branch

    return results, tessellation, bond_energy, hinge_geometry, cut_axis


# ── IO ────────────────────────────────────────────────────────────────────────────────────────

_ARRAYS = ('face_xy', 'face_nodes', 'face_areas', 'hinge_xy', 'face_pairs', 't_lig', 'm_lig',
           't_con', 'm_con', 'hinge_axis', 'axial', 'transverse', 'energy', 'sigma', 'f_ext',
           'residual', 'free_mask', 'boundary_xy')

# Fields added after the first archives were written. Zero-filling them is exact rather than an
# approximation for every field saved so far: the contact barrier is switched off for any OPEN hinge
# (``cutoff_angle: 0.0``), so those runs carry no bearing force and hence no bearing moment either.
_ADDED_LATER = ('m_con', 'hinge_axis')


def save_field(field: ForceField, out_path: str, meta: Optional[dict] = None):
    """Write ``<out_path>.npz`` (arrays) and ``<out_path>.json`` (budget, checks, meta)."""
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    np.savez_compressed(out_path + '.npz', **{k: np.asarray(getattr(field, k)) for k in _ARRAYS})
    with open(out_path + '.json', 'w') as fh:
        json.dump({'budget': force_budget(field), 'checks': field.checks, 'meta': meta or {}},
                  fh, indent=2)


def load_field(out_path: str) -> ForceField:
    """Inverse of :func:`save_field`; ``out_path`` is the stem, without extension."""
    with np.load(out_path + '.npz') as data:
        arrays = {k: data[k] for k in _ARRAYS if k in data.files}
    for key in _ADDED_LATER:
        arrays.setdefault(key, np.zeros_like(arrays['m_lig']))   # (n_hinges, 2) for both
    checks = {}
    if os.path.exists(out_path + '.json'):
        with open(out_path + '.json') as fh:
            checks = json.load(fh).get('checks', {})
    return ForceField(checks=checks, **arrays)
