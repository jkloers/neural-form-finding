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

Differentiable hinge design params (11-param corner-hinge cubic Bézier):
  arm_width_physical  — hinge strip gap [m]
  fold_top            — far Bézier anchor depth along face edge [m]
  fold_bot            — near Bézier anchor depth [m] (≈0 for corner; must be < fold_top)
  bc1_x, bc1_y        — interior Bézier CP 1 for upper wing far boundary [m] (absolute XY)
  bc2_x, bc2_y        — interior Bézier CP 2 for upper wing far boundary [m] (absolute XY)
  bc1l_x, bc1l_y      — interior Bézier CP 1 for lower wing far boundary [m] (absolute XY)
  bc2l_x, bc2l_y      — interior Bézier CP 2 for lower wing far boundary [m] (absolute XY)
  Lower wing CPs default to the earm-mirrored values of the upper wing (symmetric start).

FD gradient cost: 2 × 11 params × 3 modes = 66 SOFA sims per gradient call.

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

    # ── Differentiable hinge design params (7-param corner-hinge cubic Bézier) ─

    arm_width_physical: Differentiable[Float64] = Field(
        default=0.003,
        description=(
            "Physical width of the hinge strip (gap between panel edges) [m]. "
            "Bending stiffness ∝ 1/arm_width³. Typical range: 0.002–0.010 m."
        ),
    )
    fold_top: Differentiable[Float64] = Field(
        default=0.008,
        description=(
            "Far Bézier anchor depth along the face edge [m]. "
            "Depth of the hinge strip at the far end (away from the panel corner). "
            "Must be > fold_bot. Typical range: 0.004–0.012 m."
        ),
    )
    fold_bot: Differentiable[Float64] = Field(
        default=0.00005,
        description=(
            "Near Bézier anchor depth [m]. "
            "Near-zero for corner hinge (≈50 μm closes the corner void). "
            "Must be < fold_top."
        ),
    )
    bc1_x: Differentiable[Float64] = Field(
        default=0.198,
        description=(
            "Interior Bézier CP 1 x-coordinate for upper wing far boundary [m]. "
            "Absolute world coordinate. Optimizer moves this freely."
        ),
    )
    bc1_y: Differentiable[Float64] = Field(
        default=0.10283,
        description=(
            "Interior Bézier CP 1 y-coordinate for upper wing far boundary [m]. "
            "Default ≈ corner_y + fold_top/2 * sin(45°)."
        ),
    )
    bc2_x: Differentiable[Float64] = Field(
        default=0.13859,
        description="Interior Bézier CP 2 x-coordinate for upper wing far boundary [m].",
    )
    bc2_y: Differentiable[Float64] = Field(
        default=0.07937,
        description="Interior Bézier CP 2 y-coordinate for upper wing far boundary [m].",
    )
    bc1l_x: Differentiable[Float64] = Field(
        default=0.14159,
        description=(
            "Interior Bézier CP 1 x-coordinate for lower wing far boundary [m]. "
            "Defaults to the earm-mirror of bc1 (symmetric start). "
            "Set independently for asymmetric hinge shapes."
        ),
    )
    bc1l_y: Differentiable[Float64] = Field(
        default=0.07937,
        description="Interior Bézier CP 1 y-coordinate for lower wing far boundary [m].",
    )
    bc2l_x: Differentiable[Float64] = Field(
        default=0.14725,
        description="Interior Bézier CP 2 x-coordinate for lower wing far boundary [m].",
    )
    bc2l_y: Differentiable[Float64] = Field(
        default=0.07937,
        description="Interior Bézier CP 2 y-coordinate for lower wing far boundary [m].",
    )

    # ── Mesh resolution (non-differentiable) ──────────────────────────────────

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
            "Shear and tension modes run alongside unless skip_secondary_modes=True."
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
    skip_secondary_modes: bool = Field(
        default=False,
        description=(
            "When True, skip shear and tension simulations. "
            "Reduces apply() cost from 3 to 1 SOFA sim → 3× fewer FD Jacobian sims. "
            "energy_shear and energy_tension are returned as 0.0."
        ),
    )
    n_steps: int = Field(
        default=500,
        description=(
            "Incremental load steps for SOFA AnimationLoop. "
            "Reduce to 100 for fast optimizer runs; increase to 500 for accurate large-rotation results."
        ),
    )
    fem_method: str = Field(
        default='polar',
        description="HexahedronFEMForceField method: 'polar' (co-rotational, large rotation) or 'small' (linear, valid to ~10°).",
    )
    rotation_pivot_auto: bool = Field(
        default=True,
        description=(
            "When True, automatically use the hinge corner vertex as rotation pivot "
            "(computed from hinge_node_pairs[0]). "
            "Required for 2-face corner-hinge experiments. "
            "Set False to use loaded face centroid (standard multi-face behavior)."
        ),
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
    Build the hex mesh from CS topology + 7-param corner-hinge cubic Bézier design,
    then run SOFA under three load cases: rotation (primary mode), shear, and tension.

    The mesh is fully rebuilt on every call — including ±ε FD perturbations —
    because all seven hinge design variables directly affect the hex geometry.
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

    bezier_params = {
        'bc1_upper_xy': [float(inputs.bc1_x),  float(inputs.bc1_y)],
        'bc2_upper_xy': [float(inputs.bc2_x),  float(inputs.bc2_y)],
        'bc1_lower_xy': [float(inputs.bc1l_x), float(inputs.bc1l_y)],
        'bc2_lower_xy': [float(inputs.bc2l_x), float(inputs.bc2l_y)],
    }

    nodes, hexes, bc_masks = _mb.build_mesh_from_centroidal_state(
        cs,
        fold_top           = float(inputs.fold_top),
        fold_bot           = float(inputs.fold_bot),
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

    # Compute rotation pivot: use corner vertex from hinge_node_pairs[0] when requested.
    rotation_pivot = None
    if inputs.rotation_pivot_auto and len(cs.hinge_node_pairs) > 0:
        fi = int(cs.hinge_node_pairs[0, 0, 0])
        lj = int(cs.hinge_node_pairs[0, 0, 1])
        corner_xy = cs.face_centroids[fi] + cs.centroid_node_vectors[fi, lj]
        rotation_pivot = (float(corner_xy[0]), float(corner_xy[1]))

    mat_kw = dict(
        sheet_thickness = float(inputs.sheet_thickness),
        young_modulus   = float(inputs.young_modulus),
        poisson_ratio   = float(inputs.poisson_ratio),
        yield_strength  = float(inputs.yield_strength),
    )
    sim_kw = dict(
        n_steps    = int(inputs.n_steps),
        fem_method = str(inputs.fem_method),
    )

    # ── Rotation / moment (primary load) ──────────────────────────────────────
    res_rot = _sofa.evaluate_unit_cell(
        nodes, hexes, bc_masks,
        rotation_angle_deg = float(inputs.rotation_angle_deg),
        applied_moment     = float(inputs.applied_moment),
        loading_mode       = str(inputs.loading_mode),
        rotation_pivot     = rotation_pivot,
        **mat_kw, **sim_kw,
    )

    # ── Shear and tension (skipped when skip_secondary_modes=True) ────────────
    if inputs.skip_secondary_modes:
        e_shear   = 0.0
        e_tension = 0.0
    else:
        res_shear = _sofa.evaluate_unit_cell(
            nodes, hexes, bc_masks,
            loading_mode         = 'shear',
            shear_displacement_m = float(inputs.shear_displacement_m),
            **mat_kw, **sim_kw,
        )
        res_tension = _sofa.evaluate_unit_cell(
            nodes, hexes, bc_masks,
            loading_mode           = 'tension',
            tension_displacement_m = float(inputs.tension_displacement_m),
            **mat_kw, **sim_kw,
        )
        e_shear   = float(res_shear['strain_energy'])
        e_tension = float(res_tension['strain_energy'])

    return OutputSchema(
        strain_energy        = float(res_rot['strain_energy']),
        max_von_mises_stress = float(res_rot['max_von_mises_stress']),
        max_xy_displacement  = float(res_rot['max_xy_displacement']),
        max_z_displacement   = float(res_rot['max_z_displacement']),
        first_yield_fraction = float(res_rot['first_yield_fraction']),
        energy_shear         = e_shear,
        energy_tension       = e_tension,
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
