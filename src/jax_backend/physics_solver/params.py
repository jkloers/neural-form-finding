"""
Parameter structures for the JAX physics solver.

All classes are NamedTuples, which JAX automatically registers as Pytrees
(all fields as dynamic children). The exception is ReferenceGeometry, which
needs a custom Pytree registration to keep bond_connectivity static.
"""

import jax
import jax.numpy as jnp
from typing import Any, Dict, NamedTuple, Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from jax_backend.state import CentroidalState


# ── Geometry ──────────────────────────────────────────────────────────────────

@jax.tree_util.register_pytree_node_class
class ReferenceGeometry(NamedTuple):
    """Physics-solver view of the tessellation reference geometry.

    This is a focused, 4-field subset of CentroidalState, designed specifically
    for use inside ControlParams (the physics solver's parameter bundle).

    Relationship to CentroidalState:
      - face_centroids, centroid_node_vectors, bond_connectivity are copied
        directly from a *validated* CentroidalState.
      - reference_bond_vectors is computed from the validated geometry via
        build_reference_bond_vectors() — it does not exist in CentroidalState
        because it depends on the post-validity-optimization geometry.

    Why not use CentroidalState directly here?
      - CentroidalState carries 14 fields; the physics solver needs only 4.
      - bond_connectivity must be in JAX aux_data (static) for smap.bond.
        CentroidalState is a plain NamedTuple (all fields dynamic), which is
        correct for the validity solver but incompatible with the physics solver.

    JAX Pytree layout:
      children  → face_centroids, centroid_node_vectors, reference_bond_vectors
      aux_data  → bond_connectivity (static — must not be a Tracer for smap.bond)
    """

    face_centroids: Any
    centroid_node_vectors: Any
    bond_connectivity: Any = None
    reference_bond_vectors: Any = None

    def tree_flatten(self):
        children = (self.face_centroids, self.centroid_node_vectors,
                    self.reference_bond_vectors)
        aux_data = {'bond_connectivity': self.bond_connectivity}
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(children[0], children[1],
                   aux_data['bond_connectivity'], children[2])

    @property
    def n_faces(self) -> int:
        return self.face_centroids.shape[0]

    @classmethod
    def from_centroidal_state(cls, state: 'CentroidalState') -> 'ReferenceGeometry':
        """Builds geometry from a CentroidalState.

        bond_connectivity is read from the state where it was pre-computed
        as a static NumPy array by CentroidalState.from_tessellation() — it
        will never be a JAX Tracer.
        """
        from jax_backend.geometry import build_reference_bond_vectors

        return cls(
            face_centroids=state.face_centroids,
            centroid_node_vectors=state.centroid_node_vectors,
            bond_connectivity=state.bond_connectivity,
            reference_bond_vectors=build_reference_bond_vectors(state),
        )

    @classmethod
    def from_dict(cls, d: dict) -> 'ReferenceGeometry':
        """Builds from the raw dict returned by Tessellation._to_dict()."""
        return cls(
            face_centroids=jnp.array(d['face_centroids']),
            centroid_node_vectors=jnp.array(d['centroid_node_vectors']),
            bond_connectivity=d['bond_connectivity'],
            reference_bond_vectors=jnp.array(d['reference_bond_vectors']),
        )


# ── Mechanical parameters ─────────────────────────────────────────────────────
# These are plain NamedTuples. JAX registers NamedTuples as Pytrees natively,
# with all fields as dynamic children — no boilerplate needed.

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


BondParams = LigamentParams


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


class ControlParams(NamedTuple):
    """Top-level parameter bundle for the static solver.

    Groups all inputs required by the solver into a single JAX-compatible
    structure that can be passed through jit/grad transparently.

    Attrs:
        reference_geometry: ReferenceGeometry — geometry in current config
        mechanical_params:  MechanicalParams  — stiffness, density, contact
        loading_params:     dict              — extra kwargs for loading_fn
        constraint_params:  dict              — extra kwargs for constraint_fn
    """
    reference_geometry: ReferenceGeometry
    mechanical_params: MechanicalParams
    loading_params: Dict = dict()
    constraint_params: Dict = dict()


class SolutionData(NamedTuple):
    """Output of the static solver across all load steps.

    Only carries what the solver produces — geometry is available via
    ControlParams and does not need to be duplicated here.

    Attrs:
        fields:   (n_steps, n_faces, 3) — displacement history [dx, dy, dtheta]
        energies: (n_steps,) or dict    — total potential energy per step,
                                          or decomposed dict after post-processing
    """
    fields: Any
    energies: Any = None


# ── Standalone factory ────────────────────────────────────────────────────────

def build_control_params(geometry: ReferenceGeometry,
                         k_stretch: Any,
                         k_shear: Any,
                         k_rot: Any,
                         density: Any,
                         k_contact: float = 1.0,
                         min_angle: float = 0.0,
                         cutoff_angle: float = 0.1,
                         use_contact: bool = True) -> ControlParams:
    """Assembles ControlParams from explicit parameters.

    Takes the geometry container and all mechanical properties as direct
    arguments — no implicit duck-typing or proxy objects needed.

    Args:
        geometry:     ReferenceGeometry with reference positions and bond vectors.
        k_stretch:    Axial stiffness per hinge, shape (n_hinges,).
        k_shear:      Shear stiffness per hinge, shape (n_hinges,).
        k_rot:        Rotational stiffness per hinge, shape (n_hinges,).
        density:      Mass density per face, shape (n_faces,).
        k_contact:    Contact stiffness coefficient.
        min_angle:    Minimum void angle before contact activates.
        cutoff_angle: Void angle at which contact energy saturates.
        use_contact:  Whether to include contact energy.

    Returns:
        ControlParams ready to be passed to the static solver.
    """
    return ControlParams(
        reference_geometry=geometry,
        mechanical_params=MechanicalParams(
            bond_params=LigamentParams(
                k_stretch=k_stretch,
                k_shear=k_shear,
                k_rot=k_rot,
                reference_vector=geometry.reference_bond_vectors,
            ),
            density=density,
            contact_params=ContactParams(
                k_contact=k_contact,
                min_angle=min_angle,
                cutoff_angle=cutoff_angle,
            ) if use_contact else None,
        ),
        constraint_params=dict(),
    )
