# Neural Form-Finding

A fully differentiable, end-to-end framework for the inverse design and simulation of deployable Kirigami structures, powered by JAX.

Developed at the **Princeton Form Finding Lab**.

## Overview

This project implements a multi-stage, gradient-based pipeline to seamlessly bridge the gap between flat Kirigami tessellations and target 3D/2D deployed geometries. The pipeline guarantees physical validity by passing gradients backward through an implicit static equilibrium solver.

### The Forward Pipeline

1. **Stage 0: Initial Mapping**  
   Maps a flat tessellation to a target geometry using complex polynomial conformal mapping.
   *Parameters:* Rigid body degrees of freedom (translation $t_x, t_y$, rotation $\theta$) + scaling factor + polynomial coefficients. 
2. **Stage 1: Geometric Validity**  
   Optimizes face shapes and positions to preserve connectivity, enforce arm symmetry, and prevent face intersections.
3. **Stage 2: Static Solver**  
   Computes the physical equilibrium state of the tessellation under external loads. Formulated as a gradient-based minimization of mechanical energy (stretch, shear, rotational, and contact penalties).

### End-to-End Optimization (`train.py`)

Using `jax.value_and_grad`, the pipeline supports inverse design by propagating gradients through the entire solver chain to optimize the initial mapping parameters.

The **Dual Loss Function** ensures optimal and physically valid designs:
*   **Geometric Loss (Chamfer Distance):** Penalizes the distance from the outer boundary faces to the target shape (e.g., a circle or heart).
*   **Physical Loss (Strain Energy Penalty):** Penalizes internal strain energy (stretch, shear, rot, contact) to automatically discard physically invalid solutions (e.g., severe intersections or material tearing).

## Project Structure

```text
.
├── train.py                # End-to-End Inverse Design Optimizer
├── main.py                 # Forward Pipeline Execution & Visualization
├── src/                    # Source code
│   ├── jax_backend/        # JAX physics, centroidal states, and solvers
│   │   ├── physics_solver/ # Energy functionals and static equilibrium
│   │   └── training/       # Loss definitions and Optax training loop
│   ├── topology/           # Tessellation and Unit Cell logic
│   ├── problem/            # Config schemas, BCs, and loading logic
│   └── utils/              # Visualization (Princeton theme) & Math helpers
├── data/                   # Data-centric assets
│   ├── library/            # Pattern library (YAML)
│   ├── configs/            # Simulation configurations (YAML)
│   └── outputs/            # Results (Animations, Plots, JSON backups)
└── notebooks/              # Interactive exploration
```

## Configuration (YAML)

Experiments are driven by YAML configurations (e.g., `data/configs/complex_mapping/2_cs_asy_complex.yaml`).

```yaml
mapping:
  type: "conformal_polynomial"
  params: [0.0, 0.0, 0.0, 1.0, 0.0]  # [tx, ty, theta, scale, c1, ...]
```
*The initial parameters encode translation, rotation, scale, and high-order complex deformation coefficients.*
