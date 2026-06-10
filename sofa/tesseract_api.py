"""
tesseract_api.py — Tesseract entry point for the NFF-SOFA physics oracle.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ARCHITECTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  hinge_optimizer/ or JAX pipeline (kgnn_mac, Python 3.10)
       │
       │  HTTP — requests / tesseract_core JAX client
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
  cost = 2 × n_diff_inputs × n_modes SOFA sims per gradient call
       = 10 perturbations × 3 modes = 30 simulations for 5 Bézier params.

Each ±ε call fully rebuilds the hex mesh then re-runs all three loading modes.
This is necessary because the hinge design variables directly affect mesh geometry.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INPUTS / OUTPUTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CS topology (non-differentiable, passed as nested lists):
  face_centroids, centroid_node_vectors, hinge_node_pairs, hinge_adj_info

Differentiable hinge design params (5-param Bézier hinge profile):
  arm_width_physical  — hinge strip gap [m]
  fold_top            — far Bézier anchor depth along face edge [m]
  fold_bot            — near Bézier anchor depth [m] (0 = corner; must be < fold_top)
  waist_top           — fold-dir offset of the far Bézier control point [m]
  waist_bot           — fold-dir offset of the near Bézier control point [m]

Outputs (all scalar, all Differentiable[Float64]):
  strain_energy, max_von_mises_stress, max_xy_displacement,
  max_z_displacement, first_yield_fraction  — from rotation mode
  energy_shear                               — from shear loading mode
  energy_tension                             — from tension loading mode
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

# mesh_builder.py and simulate_cell.py live alongside this file in sofa/.
import mesh_builder  as _mb
import simulate_cell as _sofa


# ── Schemas ────────────────────────────────────────────────────────────────────

class InputSchema(BaseModel):
    """Inputs to the SOFA unit-cell simulation (CS-mesh, 5-param Bézier hinge)."""

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
            "pivot_local_i: local node index of the hinge pivot in face_i. "
            "adj_local_i/k: adjacent node inside each face — sets fold direction."
        ),
    )
    clamped_faces: list = Field(
        default=[],
        description="Face indices to clamp (fixed DOFs). If empty, uses CS-derived BCs.",
    )
    loaded_faces: list = Field(
        default=[],
        description="Face indices to drive (rotation/moment applied). If empty, uses CS-derived BCs.",
    )

    # ── Differentiable hinge design params (5-param Bézier) ───────────────────

    arm_width_physical: Differentiable[Float64] = Field(
        default=0.005,
        description=(
            "Physical width of the hinge strip (gap between panel edges) [m]. "
            "Bending stiffness ∝ 1/arm_width³. Typical range: 0.002–0.020 m."
        ),
    )
    fold_top: Differentiable[Float64] = Field(
        default=0.003,
        description=(
            "Far Bézier anchor depth along the face edge [m]. "
            "Depth of the hinge strip at the far end (away from the panel corner). "
            "Must be > fold_bot. Typical range: 0.002–0.010 m."
        ),
    )
    fold_bot: Differentiable[Float64] = Field(
        default=0.0,
        description=(
            "Near Bézier anchor depth [m]. "
            "Depth at the panel-corner end of the hinge strip (0 = corner exactly). "
            "Must be < fold_top."
        ),
    )
    waist_top: Differentiable[Float64] = Field(
        default=0.0,
        description=(
            "Bézier control-point fold-dir offset for the far curve [m]. "
            "Positive = pinched (dogbone); negative = bulging. 0 = bilinear hinge."
        ),
    )
    waist_bot: Differentiable[Float64] = Field(
        default=0.0,
        description="Bézier control-point fold-dir offset for the near curve [m].",
    )

    # ── Mesh resolution (non-differentiable) ──────────────────────────────────

    n_ctrl: int = Field(
        default=3,
        description="Bézier control points per hinge curve. 3 = quadratic.",
    )
    sheet_thickness: float = Field(
        default=0.001,
        description="Material thickness in z [m]. Default: 1 mm PLA sheet.",
    )
    n_face: int = Field(
        default=4,
        description="Hex elements per face edge (in-plane). 4 = qualitative; 6–8 = calibration.",
    )
    n_hinge: int = Field(
        default=2,
        description="Hex elements across hinge width. 2 = qualitative; 4–6 = calibration.",
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
            "Used for the rotation loading mode."
        ),
    )
    applied_moment: float = Field(
        default=0.0,
        description="In-plane torque on loaded face [N·m]. Used when loading_mode='moment'.",
    )
    loading_mode: str = Field(
        default='rotation',
        description=(
            "Primary load mode: 'rotation' (displacement-controlled) | "
            "'moment' (force-controlled torque). "
            "Shear and tension modes always run alongside the primary mode."
        ),
    )
    shear_displacement_m: float = Field(
        default=0.005,
        description="Shear loading: tangential displacement of loaded face [m].",
    )
    tension_displacement_m: float = Field(
        default=0.005,
        description="Tension loading: normal pull-apart displacement of loaded face [m].",
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
            "Central differences: cost = 2 × n_diff_inputs × n_modes SOFA simulations."
        ),
    )


class OutputSchema(BaseModel):
    """Scalar mechanical quantities at SOFA static equilibrium under three load cases."""

    # ── Rotation / moment mode ────────────────────────────────────────────────
    strain_energy: Differentiable[Float64] = Field(
        description="Total SvK elastic energy under rotation loading [J].",
    )
    max_von_mises_stress: Differentiable[Float64] = Field(
        description="Peak von Mises stress under rotation loading [Pa].",
    )
    max_xy_displacement: Differentiable[Float64] = Field(
        description="Peak in-plane |XY| displacement on free nodes (rotation mode) [m].",
    )
    max_z_displacement: Differentiable[Float64] = Field(
        description="Peak |z| out-of-plane displacement (rotation mode) [m].",
    )
    first_yield_fraction: Differentiable[Float64] = Field(
        description="max_von_mises_stress / yield_strength (rotation mode). > 1 = plastic.",
    )
    # ── Shear mode ────────────────────────────────────────────────────────────
    energy_shear: Differentiable[Float64] = Field(
        description="Total SvK elastic energy under shear loading [J].",
    )
    # ── Tension mode ──────────────────────────────────────────────────────────
    energy_tension: Differentiable[Float64] = Field(
        description="Total SvK elastic energy under tension loading [J].",
    )


# ── Required endpoint ──────────────────────────────────────────────────────────

def apply(inputs: InputSchema) -> OutputSchema:
    """
    Build the hex mesh from CS topology + 5-param Bézier hinge design, then run
    SOFA under three load cases: rotation (primary mode), shear, and tension.

    The mesh is fully rebuilt on every call — including ±ε FD perturbations —
    because all five hinge design variables directly affect the hex geometry.
    """
    # Reconstruct CS as a plain namespace (no JAX dependency in container).
    # constrained/loaded_face_DOF_pairs are left empty; BCs come from clamped/loaded_faces.
    cs = types.SimpleNamespace(
        face_centroids             = np.array(inputs.face_centroids,        dtype=np.float64),
        centroid_node_vectors      = np.array(inputs.centroid_node_vectors, dtype=np.float64),
        hinge_node_pairs           = np.array(inputs.hinge_node_pairs,      dtype=np.int32),
        hinge_adj_info             = np.array(inputs.hinge_adj_info,        dtype=np.int32),
        constrained_face_DOF_pairs = np.empty((0, 2), dtype=np.int32),
        loaded_face_DOF_pairs      = np.empty((0, 2), dtype=np.int32),
    )

    fold_top  = float(inputs.fold_top)
    fold_bot  = float(inputs.fold_bot)
    waist_top = float(inputs.waist_top)
    waist_bot = float(inputs.waist_bot)

    use_bezier = fold_bot > 1e-9 or abs(waist_top) > 1e-9 or abs(waist_bot) > 1e-9
    bezier_params = (
        {'waist_top': waist_top, 'waist_bot': waist_bot, 'n_ctrl': int(inputs.n_ctrl)}
        if use_bezier else None
    )

    nodes, hexes, bc_masks = _mb.build_mesh_from_centroidal_state(
        cs,
        fold_top           = fold_top,
        fold_bot           = fold_bot,
        bezier_params      = bezier_params,
        sheet_thickness    = float(inputs.sheet_thickness),
        n_face             = int(inputs.n_face),
        n_hinge            = int(inputs.n_hinge),
        n_z                = int(inputs.n_z),
        arm_width_physical = float(inputs.arm_width_physical),
    )

    # Override clamped/loaded masks from explicit face index lists.
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

    mat_kw = dict(
        sheet_thickness = float(inputs.sheet_thickness),
        young_modulus   = float(inputs.young_modulus),
        poisson_ratio   = float(inputs.poisson_ratio),
        yield_strength  = float(inputs.yield_strength),
    )

    # ── Rotation / moment (primary load) ──────────────────────────────────────
    res_rot = _sofa.evaluate_unit_cell(
        nodes, hexes, bc_masks,
        rotation_angle_deg = float(inputs.rotation_angle_deg),
        applied_moment     = float(inputs.applied_moment),
        loading_mode       = str(inputs.loading_mode),
        **mat_kw,
    )

    # ── Shear ─────────────────────────────────────────────────────────────────
    res_shear = _sofa.evaluate_unit_cell(
        nodes, hexes, bc_masks,
        loading_mode         = 'shear',
        shear_displacement_m = float(inputs.shear_displacement_m),
        **mat_kw,
    )

    # ── Tension ───────────────────────────────────────────────────────────────
    res_tension = _sofa.evaluate_unit_cell(
        nodes, hexes, bc_masks,
        loading_mode           = 'tension',
        tension_displacement_m = float(inputs.tension_displacement_m),
        **mat_kw,
    )

    return OutputSchema(
        strain_energy        = float(res_rot['strain_energy']),
        max_von_mises_stress = float(res_rot['max_von_mises_stress']),
        max_xy_displacement  = float(res_rot['max_xy_displacement']),
        max_z_displacement   = float(res_rot['max_z_displacement']),
        first_yield_fraction = float(res_rot['first_yield_fraction']),
        energy_shear         = float(res_shear['strain_energy']),
        energy_tension       = float(res_tension['strain_energy']),
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
        "energy_shear":         scalar,
        "energy_tension":       scalar,
    }
