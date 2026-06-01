"""
Geometry-dependent force types for neural form-finding.

Replaces global-frame Neumann BCs with physically meaningful alternatives:

  tile_to_tile  — force on source_face pointing toward/away from target_face.
                  Direction is computed from live centroid positions (differentiable).
                  Physical analogue: cable, spring or actuator between two tiles.

  tess_frame    — force along the major or minor principal axis of the mapped
                  tessellation. Direction is computed from the PCA of live
                  centroids (differentiable). Physical analogue: a load applied
                  relative to the structure's own orientation rather than the lab.

  global_frame  — legacy: fixed direction in the world frame. Included for
                  completeness; prefer the above two for new experiments.

All three types produce a (loaded_face_DOF_pairs, loading_fn) pair compatible
with setup_static_solver. The loading_fn closes over JAX expressions computed
from face_centroids at call time, so gradients flow back through the force
directions to the GNN parameters.

YAML spec format
----------------
Old format (global_frame, backward-compatible — no 'type' key needed):
    - face: 3
      dof: 0        # 0=Fx, 1=Fy, 2=Mz
      value: -1.0

New formats (require a 'type' key; handled here, NOT by conditions.py):

  tile_to_tile:
    - type: tile_to_tile
      source_face: 3
      target_face: 7
      magnitude: -1.0   # <0 → push source toward target (compression/closing)
                         # >0 → pull source away from target (tension/opening)

  tess_frame:
    - type: tess_frame
      face: 5
      tess_dof: 0    # 0 = major axis (largest spatial extent)
                      # 1 = minor axis (smallest spatial extent)
      value: 1.0

  global_frame (explicit, same as legacy):
    - type: global_frame
      face: 3
      dof: 0
      value: -1.0
"""

import numpy as np
import jax.numpy as jnp


# ── Helpers ───────────────────────────────────────────────────────────────────

# Small asymmetric regularizer added to the scatter matrix before eigh.
# Makes eigenvalues always distinct (2e-6 vs 1e-6) even for a perfectly
# isotropic point cloud, so eigenvectors and their gradients are well-defined.
# Negligible compared to the real variance of any non-degenerate tessellation.
_TESS_REG = jnp.array([[2e-6, 0.0], [0.0, 1e-6]])


def _tess_principal_axis(face_centroids: jnp.ndarray, axis_index: int) -> jnp.ndarray:
    """Return the major (0) or minor (1) principal axis of the centroid cloud.

    Uses jnp.linalg.eigh so gradients flow back through the axis direction to
    the GNN parameters that produced face_centroids.

    Args:
        face_centroids: (n_faces, 2) JAX array — may be a tracer inside JIT.
        axis_index:     0 → major axis (largest variance direction),
                        1 → minor axis (smallest variance direction).

    Returns:
        (2,) unit vector along the requested axis.
    """
    c = face_centroids - jnp.mean(face_centroids, axis=0)   # centre the cloud
    cov = c.T @ c + _TESS_REG   # regularise: keeps eigenvalues distinct → no NaN grad
    _, eigvecs = jnp.linalg.eigh(cov)   # columns sorted ascending by eigenvalue
    # col 0 = minor axis, col 1 = major axis
    # axis_index=0 → major → col 1,  axis_index=1 → minor → col 0
    return eigvecs[:, 1 - axis_index]                        # (2,)


def _unit_vec(v: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    """Safely normalise a 2-vector."""
    return v / jnp.maximum(jnp.linalg.norm(v), eps)


# ── Public API ────────────────────────────────────────────────────────────────

def has_geometry_dependent_loads(load_specs: list) -> bool:
    """Return True if any spec requires geometry-dependent force directions."""
    return any('type' in s for s in (load_specs or []))


def build_geometry_dependent_loading(
        load_specs: list,
        face_centroids: jnp.ndarray,
) -> tuple:
    """Build (loaded_face_DOF_pairs, loading_fn) from typed load specs.

    All force directions that depend on geometry are computed here from
    face_centroids (a live JAX array), making them differentiable end-to-end.
    The returned loading_fn is a pure closure over JAX expressions — it is
    fully compatible with setup_static_solver and jax.lax.scan.

    Args:
        load_specs:     List of load dicts (see module docstring for format).
                        May include both old-format (no 'type') and new-format.
        face_centroids: (n_faces, 2) JAX array from valid_state — may be a
                        tracer inside JIT.

    Returns:
        loaded_face_DOF_pairs: (n_loaded, 2) NumPy int32 — static indices.
        loading_fn:            Callable (state, t, **kwargs) → (n_loaded,).
                               Returns t-scaled force values at load step t.
                               Returns None if load_specs is empty.
    """
    if not load_specs:
        return np.zeros((0, 2), dtype=np.int32), None

    dof_pairs    = []  # will become the static NumPy array
    force_values = []  # JAX scalar expressions — differentiable

    for spec in load_specs:
        load_type = spec.get('type', 'global_frame')

        if load_type == 'global_frame':
            # Fixed direction in the world frame — static value, no geometry dependence.
            face  = int(spec['face'])
            dof   = int(spec['dof'])
            value = float(spec['value'])
            dof_pairs.append((face, dof))
            force_values.append(jnp.array(value, dtype=jnp.float64))

        elif load_type == 'tile_to_tile':
            # Force on source_face directed toward (magnitude < 0) or away from
            # (magnitude > 0) target_face. Direction is differentiable.
            #
            # Convention: diff = source - target points OUTWARD (away from target).
            # force = magnitude * diff_normalised
            #   magnitude < 0 → force opposes outward = pushes source TOWARD target
            #   magnitude > 0 → force along outward   = pulls source AWAY from target
            source    = int(spec['source_face'])
            target    = int(spec['target_face'])
            magnitude = float(spec['magnitude'])

            diff      = face_centroids[source] - face_centroids[target]  # (2,) outward from target
            direction = _unit_vec(diff)                                   # (2,) JAX

            # Decompose into x and y DOFs (moment is not meaningful here)
            dof_pairs.append((source, 0))
            dof_pairs.append((source, 1))
            force_values.append(magnitude * direction[0])
            force_values.append(magnitude * direction[1])

        elif load_type == 'tess_frame':
            # Force along a principal axis of the mapped tessellation.
            # Both direction components → two DOF entries per load.
            face     = int(spec['face'])
            tess_dof = int(spec['tess_dof'])  # 0=major, 1=minor
            value    = float(spec['value'])

            direction = _tess_principal_axis(face_centroids, tess_dof)  # (2,) JAX

            dof_pairs.append((face, 0))
            dof_pairs.append((face, 1))
            force_values.append(value * direction[0])
            force_values.append(value * direction[1])

        else:
            raise ValueError(
                f"Unknown load type '{load_type}'. "
                f"Valid types: 'global_frame', 'tile_to_tile', 'tess_frame'."
            )

    loaded_face_DOF_pairs = np.array(dof_pairs, dtype=np.int32)  # static
    force_vals_jax = jnp.stack(force_values)                     # (n_loaded,) — differentiable

    # force_vals_jax is returned separately so the caller can place it inside
    # control_params.loading_params.  The loading_fn reads it from kwargs
    # rather than closing over it — this is required because jaxopt's custom_vjp
    # solver can only differentiate through explicit arguments, not closed-over
    # JAX tracers.
    loading_fn = lambda state, t, **kwargs: t * kwargs['force_values']

    return loaded_face_DOF_pairs, loading_fn, force_vals_jax
