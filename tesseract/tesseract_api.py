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
computed by Tesseract calling apply() (N_differentiable_inputs + 1) times with
central finite differences. The fd_eps field controls the perturbation size and
can be tuned per experiment without code changes.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 2 → PHASE 3 EXTENSION POINTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

InputSchema is intentionally minimal (3 scalars) for Phase 2. Phase 3 will
extend it — likely adding:
  - face_centroids: Array[(4, 2), Float64]  from NFF CentroidalState
  - hinge_properties: Array[(4,), Float64]  per-hinge stiffness

OutputSchema will grow to include:
  - equilibrium_positions: Array[(12, 3), Float64]
  - per_hinge_energy: Array[(4,), Float64]

Design rule: add new fields with defaults so existing callers never break.
"""

import threading
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from tesseract_core.runtime import Differentiable, Float64, ShapeDType
from tesseract_core.runtime.experimental import (
    finite_difference_jacobian,
    finite_difference_jvp,
    finite_difference_vjp,
)

# Import the SOFA physics core (copied into the container root via package_data).
# On macOS dev machines, run via ./sofa/run_sofa.sh which sets PYTHONPATH so
# both simulate_cell and Sofa are importable.
import simulate_cell as _sofa


# ── SOFA global-state lock ─────────────────────────────────────────────────────
#
# SOFA uses a process-level singleton registry. Concurrent apply() calls in the
# same process corrupt this state. The lock serializes all simulation calls.
#
# Phase 3 migration path: replace with a subprocess-per-call pool for true
# parallelism (each subprocess has its own SOFA singleton).
_SOFA_LOCK = threading.Lock()


# ── Schemas ────────────────────────────────────────────────────────────────────

class InputSchema(BaseModel):
    """
    Inputs to the SOFA unit-cell simulation.

    All three geometric parameters are differentiable so the JAX pipeline can
    compute gradients of strain_energy with respect to filament design.

    fd_eps controls the finite-difference perturbation for gradient estimation.
    It is NOT differentiable — treat it as a hyperparameter, not a design variable.
    """

    filament_thickness: Differentiable[Float64] = Field(
        default=0.01,
        description=(
            "Cross-section radius of each hinge filament [m]. "
            "Bending stiffness scales as r^4. Typical range: 0.005–0.1."
        ),
    )
    filament_length: Differentiable[Float64] = Field(
        default=0.20,
        description=(
            "Natural (rest) length of each hinge filament [m]. "
            "Sets the gap between adjacent face panels in the reference state. "
            "Typical range: 0.02–0.5."
        ),
    )
    applied_displacement: Differentiable[Float64] = Field(
        default=0.10,
        description=(
            "Rigid x-displacement imposed on face F2 [m]. "
            "Keep ≤ 0.3 × face_size for quasi-linear regime."
        ),
    )
    fd_eps: float = Field(
        default=1e-5,
        description=(
            "Finite-difference perturbation size for gradient estimation. "
            "Not differentiable — tune per experiment if energy scale changes. "
            "Central differences: accuracy ∝ eps^2, cost = 2 × n_diff_inputs simulations."
        ),
    )


class OutputSchema(BaseModel):
    """
    Outputs from the SOFA unit-cell simulation.
    """

    strain_energy: Differentiable[Float64] = Field(
        description=(
            "Total Euler-Bernoulli strain energy stored in the 4 hinge filaments "
            "at static equilibrium [J]. Computed analytically from SOFA equilibrium "
            "positions using the 6-DOF beam stiffness matrix."
        )
    )


# ── Required endpoint ──────────────────────────────────────────────────────────

def apply(inputs: InputSchema) -> OutputSchema:
    """
    Run the SOFA unit-cell simulation and return the filament strain energy.

    Stateless: each call creates a fresh SOFA root node and unloads it in a
    finally block, regardless of success or failure. The SOFA lock ensures
    only one simulation runs at a time in this process.
    """
    with _SOFA_LOCK:
        energy = _sofa.evaluate_unit_cell(
            filament_thickness=float(inputs.filament_thickness),
            filament_length=float(inputs.filament_length),
            applied_displacement=float(inputs.applied_displacement),
        )

    return OutputSchema(strain_energy=float(energy))


# ── Optional endpoints — gradients via finite differences ──────────────────────
#
# Tesseract calls these when the JAX pipeline requests ∂strain_energy/∂inputs.
# We delegate entirely to Tesseract's built-in finite-difference helpers, which
# call apply() with perturbed inputs automatically.
#
# algorithm="central" uses (f(x+ε) - f(x-ε)) / 2ε — accuracy O(ε²), costs 2 runs
# per differentiable input (6 total for our 3 differentiable params).

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
    strain_energy is always a scalar Float64 regardless of input shapes.
    """
    return {
        "strain_energy": ShapeDType(shape=(), dtype="Float64"),
    }
