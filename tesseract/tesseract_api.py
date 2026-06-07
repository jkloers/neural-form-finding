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
       ▼
  simulate_cell.py — SOFA physics core (copied into container via package_data)
       │  calls
       ▼
  SOFA v25.12 shared libs (installed at /opt/sofa via custom_build_steps)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GRADIENT STRATEGY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SOFA is a black-box simulator — no adjoint or AD is available. Gradients are
computed by Tesseract calling apply() with central finite differences:
  cost = 2 × n_differentiable_inputs SOFA simulations per gradient call
       = 6 simulations for the 3 geometric diff inputs.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 3a — HexahedronFEM inputs and outputs
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Differentiable inputs (Phase 3a):
  hinge_arm_width    — physical gap between panels [m], stiffness ∝ 1/w³
  hinge_fold_length  — hinge strip length along shared edge [m]
  applied_displacement — rigid x-displacement of face F2 [m]

Non-differentiable inputs (fixed geometry / material constants):
  face_size, sheet_thickness, young_modulus, poisson_ratio, yield_strength

Outputs (all differentiable, all scalar):
  strain_energy, max_von_mises_stress, max_z_displacement, first_yield_fraction

Phase 4 extension: use strain_energy as a reward/fine-tuning signal in the NFF
training loop. Add face_centroids: Array[(4,2)] to InputSchema when connecting
to CentroidalState from the JAX pipeline.

Design rule: add new fields with defaults so existing callers never break.
"""

from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from tesseract_core.runtime import Differentiable, Float64, ShapeDType
from tesseract_core.runtime.experimental import (
    finite_difference_jacobian,
    finite_difference_jvp,
    finite_difference_vjp,
)

# The SOFA physics core is copied into the container root via package_data.
# simulate_cell.evaluate_unit_cell() acquires _SOFA_LOCK internally — no
# additional locking is needed here.
import simulate_cell as _sofa


# ── Schemas ────────────────────────────────────────────────────────────────────

class InputSchema(BaseModel):
    """
    Inputs to the SOFA unit-cell simulation (Phase 3a HexahedronFEM).

    The three geometric parameters are differentiable so the JAX pipeline can
    compute gradients of any output with respect to hinge design variables.
    Material constants and face_size are non-differentiable — treat as fixed
    for a given material/scale combination.

    fd_eps controls the finite-difference perturbation for gradient estimation.
    It is NOT differentiable — treat as a hyperparameter.
    """

    hinge_arm_width: Differentiable[Float64] = Field(
        default=0.005,
        description=(
            "Physical gap between adjacent face panels (hinge arm length) [m]. "
            "Bending stiffness ∝ 1/arm_width³. Typical range: 0.002–0.020."
        ),
    )
    hinge_fold_length: Differentiable[Float64] = Field(
        default=0.020,
        description=(
            "Length of the uncut hinge strip along the shared edge [m]. "
            "Total stiffness scales linearly with fold_length. Typical range: 0.010–0.050."
        ),
    )
    applied_displacement: Differentiable[Float64] = Field(
        default=0.010,
        description=(
            "Rigid x-displacement imposed on face F2 [m]. "
            "Keep ≤ 0.3 × face_size to remain in the quasi-static regime."
        ),
    )
    face_size: float = Field(
        default=0.100,
        description="Square face panel side length [m]. Default: 100 mm.",
    )
    sheet_thickness: float = Field(
        default=0.001,
        description="Material thickness in z [m]. Bending stiffness ∝ t³. Default: 1 mm.",
    )
    young_modulus: float = Field(
        default=3.5e9,
        description="Young's modulus [Pa]. Default: PLA 3.5 GPa.",
    )
    poisson_ratio: float = Field(
        default=0.36,
        description="Poisson ratio. Default: PLA 0.36.",
    )
    yield_strength: float = Field(
        default=55e6,
        description="Yield stress [Pa] used to compute first_yield_fraction. Default: PLA 55 MPa.",
    )
    fd_eps: float = Field(
        default=1e-5,
        description=(
            "Finite-difference perturbation size for gradient estimation. "
            "Not differentiable — tune per experiment if energy scale changes. "
            "Central differences: accuracy ∝ eps², cost = 2 × n_diff_inputs simulations."
        ),
    )


class OutputSchema(BaseModel):
    """
    Outputs from the SOFA unit-cell simulation (Phase 3a).

    All four quantities are computed analytically (SvK) from SOFA equilibrium
    positions — see simulate_cell._svk_energy and _vm_stress_per_tet.
    """

    strain_energy: Differentiable[Float64] = Field(
        description=(
            "Total Saint Venant-Kirchhoff elastic energy across all 4 hinge "
            "volumes at static equilibrium [J]."
        )
    )
    max_von_mises_stress: Differentiable[Float64] = Field(
        description=(
            "Peak von Mises (Cauchy) stress across all tetrahedral elements "
            "in all 4 hinges [Pa]."
        )
    )
    max_z_displacement: Differentiable[Float64] = Field(
        description=(
            "Peak |z| displacement of interior hinge nodes [m]. "
            "Proxy for out-of-plane buckling magnitude."
        )
    )
    first_yield_fraction: Differentiable[Float64] = Field(
        description=(
            "max_von_mises_stress / yield_strength. "
            "Values > 1 indicate the material has plastically yielded."
        )
    )


# ── Required endpoint ──────────────────────────────────────────────────────────

def apply(inputs: InputSchema) -> OutputSchema:
    """
    Run the SOFA unit-cell simulation and return all mechanical QoIs.

    Stateless: simulate_cell.evaluate_unit_cell() creates a fresh SOFA root node
    and unloads it in a finally block on every call. The SOFA singleton lock is
    managed inside evaluate_unit_cell — no additional locking needed here.
    """
    result = _sofa.evaluate_unit_cell(
        hinge_arm_width      = float(inputs.hinge_arm_width),
        hinge_fold_length    = float(inputs.hinge_fold_length),
        applied_displacement = float(inputs.applied_displacement),
        face_size            = float(inputs.face_size),
        sheet_thickness      = float(inputs.sheet_thickness),
        young_modulus        = float(inputs.young_modulus),
        poisson_ratio        = float(inputs.poisson_ratio),
        yield_strength       = float(inputs.yield_strength),
    )
    return OutputSchema(
        strain_energy        = float(result["strain_energy"]),
        max_von_mises_stress = float(result["max_von_mises_stress"]),
        max_z_displacement   = float(result["max_z_displacement"]),
        first_yield_fraction = float(result["first_yield_fraction"]),
    )


# ── Optional endpoints — gradients via finite differences ──────────────────────
#
# Tesseract calls these when the JAX pipeline requests gradients.
# Central differences: (f(x+ε) − f(x−ε)) / 2ε — accuracy O(ε²), cost 2 runs
# per differentiable input (6 total for 3 diff params × 2 sides).

def jacobian(
    inputs: InputSchema,
    jac_inputs: set[str],
    jac_outputs: set[str],
) -> dict[str, dict[str, Any]]:
    """Full Jacobian via central finite differences."""
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
    """Jacobian-vector product (forward-mode AD proxy) via finite differences."""
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
    """Vector-Jacobian product (reverse-mode AD proxy) via finite differences."""
    return finite_difference_vjp(
        apply, inputs, vjp_inputs, vjp_outputs, cotangent_vector,
        algorithm="central",
        eps=inputs.fd_eps,
    )


def abstract_eval(abstract_inputs: InputSchema) -> dict[str, Any]:
    """
    Return output shapes without running the simulation.

    JAX uses this to trace the computation graph before any simulation runs.
    All four outputs are always scalar Float64 regardless of input shapes.
    """
    scalar = ShapeDType(shape=(), dtype="Float64")
    return {
        "strain_energy":        scalar,
        "max_von_mises_stress": scalar,
        "max_z_displacement":   scalar,
        "first_yield_fraction": scalar,
    }
