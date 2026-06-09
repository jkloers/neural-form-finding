"""
tesseract_api.py — Tesseract entry point for the NFF-SOFA physics oracle.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ARCHITECTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  JAX pipeline (kgnn_mac, Python 3.10)
       │
       │  HTTP — Tesseract client (tesseract_core JAX integration)
       ▼
  THIS FILE — Tesseract API layer (Python 3.12, Docker/linux/amd64)
       │  imports
       ├── mesh_builder.py — build_mesh_from_centroidal_state (CS → hex mesh)
       └── simulate_cell.py — evaluate_unit_cell (SOFA FEM)
              │  calls
              ▼
         SOFA v25.12 shared libs

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GRADIENT STRATEGY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SOFA is a black-box simulator — no adjoint or AD is available. Gradients are
computed by Tesseract calling apply() with central finite differences:
  cost = 2 × n_differentiable_inputs SOFA simulations per gradient call
       = 4 simulations for arm_width_physical + fold_length.

Each ±ε call fully rebuilds the hex mesh (build_mesh_from_centroidal_state)
then re-runs the SOFA simulation. This is intentional: the mesh is a function
of the hinge design variables, so the geometry changes with each perturbation.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INPUTS / OUTPUTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CS topology (non-differentiable, passed as nested lists):
  face_centroids, centroid_node_vectors, hinge_node_pairs, hinge_adj_info

Differentiable hinge design params (Phase 3d k_rot calibration):
  arm_width_physical  — hinge strip gap [m] (bending stiffness ∝ 1/w³)
  fold_length         — hinge strip depth along face edge [m]

Outputs (all scalar, all Differentiable[Float64]):
  strain_energy, max_von_mises_stress, max_xy_displacement,
  max_z_displacement, first_yield_fraction
"""

import types
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from tesseract_core.runtime import Differentiable, Float64, ShapeDType
from tesseract_core.runtime.experimental import (
    finite_difference_jacobian,
    finite_difference_jvp,
    finite_difference_vjp,
)

# Mesh builder and SOFA core are copied into the container root via package_data.
import mesh_builder  as _mb
import simulate_cell as _sofa


# ── Schemas ────────────────────────────────────────────────────────────────────

class InputSchema(BaseModel):
    """
    Inputs to the SOFA unit-cell simulation (CS-mesh, Phase 3d).

    CS topology arrays are passed as nested Python lists (JSON-serialisable).
    They are not differentiable — the JAX pipeline only differentiates with
    respect to the two hinge design scalars below.
    """

    # ── CS topology (non-differentiable) ──────────────────────────────────────

    face_centroids: list = Field(
        description="(n_faces, 2) float — face centroid XY positions [m].",
    )
    centroid_node_vectors: list = Field(
        description="(n_faces, max_nodes, 2) float — node offset vectors from centroid [m].",
    )
    hinge_node_pairs: list = Field(
        description="(n_hinges, 2, 2) int — [[face_i, local_i], [face_k, local_k]] per hinge.",
    )
    hinge_adj_info: list = Field(
        description=(
            "(n_hinges, 5) int — [face_i, face_k, pivot_local_i, adj_local_i, adj_local_k]. "
            "pivot_local_i: which corner of face_i is the hinge pivot. "
            "adj_local_i/k: adjacent (next) local node inside each face — sets fold direction."
        ),
    )
    clamped_faces: list = Field(
        default=[],
        description="Face indices to clamp (fixed). If empty, defaults to cs-derived BCs.",
    )
    loaded_faces: list = Field(
        default=[],
        description="Face indices to drive (rotation/moment applied). If empty, defaults to cs-derived BCs.",
    )

    # ── Differentiable hinge design params ────────────────────────────────────

    arm_width_physical: Differentiable[Float64] = Field(
        default=0.005,
        description=(
            "Physical width of the hinge strip (gap between panel edges) [m]. "
            "Bending stiffness ∝ 1/arm_width³. "
            "For closed hinges (Stage-1 altproj output), this overrides the near-zero natural gap. "
            "Typical range: 0.002–0.020 m."
        ),
    )
    fold_length: Differentiable[Float64] = Field(
        default=0.003,
        description=(
            "Depth of the hinge strip along the face edge at each corner [m]. "
            "Defines how far into the face the hinge material extends. "
            "Total stiffness scales roughly linearly with fold_length. "
            "Typical range: 0.002–0.010 m."
        ),
    )

    # ── Mesh resolution (non-differentiable) ──────────────────────────────────

    sheet_thickness: float = Field(
        default=0.001,
        description="Material thickness in z [m]. Default: 1 mm PLA sheet.",
    )
    n_face: int = Field(
        default=4,
        description="Hex elements per face edge (in-plane). 4 = qualitative; 6–8 = k_rot calibration.",
    )
    n_hinge: int = Field(
        default=2,
        description="Hex elements across hinge width. 2 = qualitative; 4–6 = k_rot calibration.",
    )
    n_z: int = Field(
        default=2,
        description="Hex layers through thickness. 2 is sufficient with PartialFixed(z).",
    )

    # ── Loading ───────────────────────────────────────────────────────────────

    rotation_angle_deg: float = Field(
        default=5.0,
        description=(
            "In-plane rotation of loaded face [deg]. "
            "Negative = CW = correct RDQK opening direction. "
            "Used when loading_mode='rotation'."
        ),
    )
    applied_moment: float = Field(
        default=0.0,
        description="In-plane torque on loaded face [N·m]. Used when loading_mode='moment'.",
    )
    loading_mode: str = Field(
        default='rotation',
        description="'rotation' (displacement-controlled) | 'moment' (force-controlled).",
    )

    # ── Material ──────────────────────────────────────────────────────────────

    young_modulus: float = Field(
        default=3.5e9,
        description="Young's modulus [Pa]. Default: PLA 3.5 GPa.",
    )
    poisson_ratio: float = Field(
        default=0.36,
        description="Poisson ratio. Default: PLA 0.36.",
    )
    yield_strength: float = Field(
        default=50e6,
        description="Yield stress [Pa] for first_yield_fraction. Default: PLA 50 MPa.",
    )

    # ── Gradient hyperparameter ───────────────────────────────────────────────

    fd_eps: float = Field(
        default=1e-5,
        description=(
            "Finite-difference perturbation size. Not differentiable — tune per experiment. "
            "Central differences: accuracy ∝ eps², cost = 2 × n_diff_inputs simulations."
        ),
    )


class OutputSchema(BaseModel):
    """Scalar mechanical quantities at SOFA static equilibrium."""

    strain_energy: Differentiable[Float64] = Field(
        description="Total SvK elastic energy at static equilibrium [J].",
    )
    max_von_mises_stress: Differentiable[Float64] = Field(
        description="Peak von Mises stress across all tetrahedral elements [Pa].",
    )
    max_xy_displacement: Differentiable[Float64] = Field(
        description="Peak in-plane |XY| displacement on free (non-clamped) nodes [m].",
    )
    max_z_displacement: Differentiable[Float64] = Field(
        description="Peak |z| displacement (out-of-plane buckling) [m].",
    )
    first_yield_fraction: Differentiable[Float64] = Field(
        description="max_von_mises_stress / yield_strength. > 1 means plastic yielding.",
    )


# ── Required endpoint ──────────────────────────────────────────────────────────

def apply(inputs: InputSchema) -> OutputSchema:
    """
    Build the hex mesh from CS topology + hinge design params, then run SOFA.

    The mesh is fully rebuilt on every call — including ±ε perturbations for
    finite-difference gradient estimation. This is necessary because arm_width
    and fold_length directly affect the hex mesh geometry.
    """
    # Reconstruct CS as a plain namespace (no JAX dependency in container)
    cs = types.SimpleNamespace(
        face_centroids        = np.array(inputs.face_centroids,        dtype=np.float64),
        centroid_node_vectors = np.array(inputs.centroid_node_vectors, dtype=np.float64),
        hinge_node_pairs      = np.array(inputs.hinge_node_pairs,      dtype=np.int32),
        hinge_adj_info        = np.array(inputs.hinge_adj_info,        dtype=np.int32),
    )

    nodes, hexes, bc_masks = _mb.build_mesh_from_centroidal_state(
        cs,
        fold_length        = float(inputs.fold_length),
        sheet_thickness    = float(inputs.sheet_thickness),
        n_face             = int(inputs.n_face),
        n_hinge            = int(inputs.n_hinge),
        n_z                = int(inputs.n_z),
        arm_width_physical = float(inputs.arm_width_physical),
    )

    # Override clamped/loaded masks from explicit face index lists if provided
    if inputs.clamped_faces:
        mask = np.zeros(len(nodes), dtype=bool)
        for fi in inputs.clamped_faces:
            key = f'f{fi}'
            if key in bc_masks:
                mask |= bc_masks[key]
        bc_masks['clamped'] = mask

    if inputs.loaded_faces:
        mask = np.zeros(len(nodes), dtype=bool)
        for fi in inputs.loaded_faces:
            key = f'f{fi}'
            if key in bc_masks:
                mask |= bc_masks[key]
        bc_masks['loaded'] = mask

    result = _sofa.evaluate_unit_cell(
        nodes, hexes, bc_masks,
        rotation_angle_deg = float(inputs.rotation_angle_deg),
        applied_moment     = float(inputs.applied_moment),
        loading_mode       = str(inputs.loading_mode),
        sheet_thickness    = float(inputs.sheet_thickness),
        young_modulus      = float(inputs.young_modulus),
        poisson_ratio      = float(inputs.poisson_ratio),
        yield_strength     = float(inputs.yield_strength),
    )

    return OutputSchema(
        strain_energy        = float(result['strain_energy']),
        max_von_mises_stress = float(result['max_von_mises_stress']),
        max_xy_displacement  = float(result['max_xy_displacement']),
        max_z_displacement   = float(result['max_z_displacement']),
        first_yield_fraction = float(result['first_yield_fraction']),
    )


# ── Optional endpoints — gradients via finite differences ──────────────────────

def jacobian(
    inputs: InputSchema,
    jac_inputs: set[str],
    jac_outputs: set[str],
) -> dict[str, dict[str, Any]]:
    """Full Jacobian of outputs w.r.t. differentiable inputs via central FD."""
    return finite_difference_jacobian(
        apply, inputs, jac_inputs, jac_outputs,
        algorithm="central",
        eps=inputs.fd_eps,
    )


def jacobian_vector_product(
    inputs: InputSchema,
    jvp_inputs: set[str],
    jvp_outputs: set[str],
    tangent_vector: dict[str, Any],
) -> dict[str, Any]:
    """Forward-mode AD proxy via finite differences."""
    return finite_difference_jvp(
        apply, inputs, jvp_inputs, jvp_outputs, tangent_vector,
        algorithm="central",
        eps=inputs.fd_eps,
    )


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
) -> dict[str, Any]:
    """Reverse-mode AD proxy via finite differences."""
    return finite_difference_vjp(
        apply, inputs, vjp_inputs, vjp_outputs, cotangent_vector,
        algorithm="central",
        eps=inputs.fd_eps,
    )


def abstract_eval(abstract_inputs: InputSchema) -> dict[str, Any]:
    """Return output shapes without running the simulation."""
    scalar = ShapeDType(shape=(), dtype="Float64")
    return {
        "strain_energy":        scalar,
        "max_von_mises_stress": scalar,
        "max_xy_displacement":  scalar,
        "max_z_displacement":   scalar,
        "first_yield_fraction": scalar,
    }
