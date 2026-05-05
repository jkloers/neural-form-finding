"""
Parameter structures and utility functions for the JAX physics solver.

This module defines:
  - Data classes (NamedTuples) for all solver parameters, registered as JAX
    Pytrees so they can carry Tracers through jit/grad boundaries.
  - GeometricalParams: the single geometry container for the pipeline,
    replacing the former TessellationGeometry class. It handles the
    static/dynamic split required by JAX for the bond connectivity.
  - build_control_params: standalone factory bridging geometry + state → solver.
"""

import jax
import jax.numpy as jnp
from typing import Any, Dict, NamedTuple, Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from jax_backend.state import CentroidalState


# ── Geometry ──────────────────────────────────────────────────────────────────

@jax.tree_util.register_pytree_node_class
class GeometricalParams(NamedTuple):
    """JAX-compatible geometry container for the centroidal physics solver.

    Holds the four arrays that describe the tessellation reference configuration:
      - face_centroids          (n_faces, 2)            — centroid positions
      - centroid_node_vectors   (n_faces, n_nodes, 2)   — face shape vectors
      - bond_connectivity       (n_hinges, 2) [STATIC]  — global node indices
      - reference_bond_vectors  (n_hinges, 2)           — rest ligament vectors

    JAX Pytree layout:
      children  → dynamic arrays (differentiated during training)
      aux_data  → bond_connectivity (static — must not be a Tracer for smap.bond)
    """

    face_centroids: Any
    centroid_node_vectors: Any
    bond_connectivity: Any = None
    reference_bond_vectors: Any = None

    # ── JAX Pytree protocol ───────────────────────────────────────────────────

    def tree_flatten(self):
        # face_centroids, centroid_node_vectors, reference_bond_vectors are dynamic
        children = (self.face_centroids, self.centroid_node_vectors,
                    self.reference_bond_vectors)
        # bond_connectivity is static topology (pre-computed NumPy array)
        aux_data = {'bond_connectivity': self.bond_connectivity}
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(children[0], children[1],
                   aux_data['bond_connectivity'], children[2])

    # ── Convenience properties ────────────────────────────────────────────────

    @property
    def n_faces(self) -> int:
        return self.face_centroids.shape[0]

    # ── Factory methods ───────────────────────────────────────────────────────

    @classmethod
    def from_centroidal_state(cls, state: 'CentroidalState') -> 'GeometricalParams':
        """Builds the geometry from a centroidal state.

        bond_connectivity is read from the state where it was pre-computed
        as a static NumPy array — it will never be a JAX Tracer.
        """
        from jax_backend.geometry import build_reference_bond_vectors

        return cls(
            face_centroids=state.face_centroids,
            centroid_node_vectors=state.centroid_node_vectors,
            bond_connectivity=state.bond_connectivity,
            reference_bond_vectors=build_reference_bond_vectors(state),
        )

    @classmethod
    def from_dict(cls, d: dict) -> 'GeometricalParams':
        """Builds from the dict returned by Tessellation.to_centroidal_state()."""
        return cls(
            face_centroids=jnp.array(d['face_centroids']),
            centroid_node_vectors=jnp.array(d['centroid_node_vectors']),
            bond_connectivity=d['bond_connectivity'],
            reference_bond_vectors=jnp.array(d['reference_bond_vectors']),
        )


# ── Mechanical parameters ─────────────────────────────────────────────────────

@jax.tree_util.register_pytree_node_class
class LigamentParams(NamedTuple):
    """Stiffness parameters for ligament bonds.

    Attrs:
        k_stretch:        scalar or (n_hinges,) — axial stiffness
        k_shear:          scalar or (n_hinges,) — shear stiffness
        k_rot:            scalar or (n_hinges,) — rotational stiffness
        reference_vector: (n_hinges, 2)         — rest ligament vectors
    """

    k_stretch: Any
    k_shear: Any
    k_rot: Any
    reference_vector: Any

    def tree_flatten(self):
        return (self.k_stretch, self.k_shear, self.k_rot, self.reference_vector), ()

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


BondParams = LigamentParams


@jax.tree_util.register_pytree_node_class
class ContactParams(NamedTuple):
    """Contact energy parameters.

    Attrs:
        min_angle:    void angle below which contact activates
        cutoff_angle: void angle at which contact energy saturates
        k_contact:    contact stiffness coefficient
    """

    min_angle: Any
    cutoff_angle: Any
    k_contact: Any

    def tree_flatten(self):
        return (self.min_angle, self.cutoff_angle, self.k_contact), ()

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


@jax.tree_util.register_pytree_node_class
class MechanicalParams(NamedTuple):
    """Mechanical parameters of the assembly.

    Attrs:
        bond_params:    LigamentParams for all hinges
        density:        scalar or (n_faces,) — face mass density
        contact_params: ContactParams or None if contact is disabled
    """

    bond_params: BondParams
    density: Any
    contact_params: Optional[ContactParams] = None

    def tree_flatten(self):
        return (self.bond_params, self.density, self.contact_params), ()

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


# ── Solver control ────────────────────────────────────────────────────────────

@jax.tree_util.register_pytree_node_class
class ControlParams(NamedTuple):
    """Top-level parameter bundle for the static solver.

    Groups all inputs required by the solver into a single JAX-compatible
    structure that can be passed through jit/grad transparently.

    Attrs:
        geometrical_params: GeometricalParams — geometry in current config
        mechanical_params:  MechanicalParams  — stiffness, density, contact
        loading_params:     dict              — extra kwargs for loading_fn
        constraint_params:  dict              — extra kwargs for constraint_fn
    """

    geometrical_params: GeometricalParams
    mechanical_params: MechanicalParams
    loading_params: Dict = dict()
    constraint_params: Dict = dict()

    def tree_flatten(self):
        return (self.geometrical_params, self.mechanical_params,
                self.loading_params, self.constraint_params), ()

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


# ── Solution ──────────────────────────────────────────────────────────────────

@jax.tree_util.register_pytree_node_class
class SolutionData(NamedTuple):
    """Output of the static solver across all load steps.

    Attrs:
        face_centroids:         (n_faces, 2)          — reference centroids
        centroid_node_vectors:  (n_faces, n_nodes, 2) — reference shapes
        bond_connectivity:      (n_hinges, 2)         — static topology
        fields:                 (n_steps, n_faces, 3) — displacement history
        energies:               (n_steps,) or dict    — energy history
    """

    face_centroids: Any
    centroid_node_vectors: Any
    bond_connectivity: Any
    fields: Any
    energies: Any = None

    def tree_flatten(self):
        return (self.face_centroids, self.centroid_node_vectors,
                self.bond_connectivity, self.fields, self.energies), ()

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


# ── Standalone factory ────────────────────────────────────────────────────────

def build_control_params(geometry: GeometricalParams,
                         state: 'CentroidalState',
                         k_contact: float = 1.0,
                         min_angle: float = 0.0,
                         cutoff_angle: float = 0.1,
                         use_contact: bool = True) -> ControlParams:
    """Assembles ControlParams from a geometry object and a centroidal state.

    Bridges (GeometricalParams, CentroidalState) → ControlParams without
    coupling either class to the other's internal structure.

    Args:
        geometry:     GeometricalParams with reference positions and bond vectors.
        state:        CentroidalState with mechanical properties.
        k_contact:    Contact stiffness coefficient.
        min_angle:    Minimum void angle before contact activates.
        cutoff_angle: Void angle at which contact energy saturates.
        use_contact:  Whether to include contact energy.

    Returns:
        ControlParams ready to be passed to the static solver.
    """
    return ControlParams(
        geometrical_params=geometry,
        mechanical_params=MechanicalParams(
            bond_params=LigamentParams(
                k_stretch=state.k_stretch,
                k_shear=state.k_shear,
                k_rot=state.k_rot,
                reference_vector=geometry.reference_bond_vectors,
            ),
            density=state.density,
            contact_params=ContactParams(
                k_contact=k_contact,
                min_angle=min_angle,
                cutoff_angle=cutoff_angle,
            ) if use_contact else None,
        ),
        constraint_params=dict(),
    )
