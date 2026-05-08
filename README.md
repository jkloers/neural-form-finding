# Neural Form-Finding

A fully differentiable, end-to-end framework for the inverse design and simulation of deployable Kirigami structures, powered by JAX.

Developed at the **Princeton Form Finding Lab**.

## Overview

This project implements a multi-stage, gradient-based pipeline to bridge the gap between flat Kirigami tessellations and target geometries. The pipeline is fully differentiable, passing gradients through both geometric and physical equilibrium solvers.

### The Forward Pipeline

1. **Stage 0: Initial Mapping**  
   Maps a flat tessellation to a target geometry using parameterized mappings (e.g., conformal polynomial or root-based asymmetric maps).
2. **Stage 1: Geometric Validity**  
   A differentiable optimization stage that ensures face connectivity, enforces arm symmetry, and prevents intersections.
3. **Stage 2: Static Physics Solver**  
   Computes the equilibrium state by minimizing potential energy. Supports **incremental load stepping** for large deformations via `jax.lax.scan`.

## Development Standards

### Nomenclature Convention
To ensure clarity in the JAX-based functional pipeline, we follow a strict naming convention:
*   **Suffix `_fn`**: Applied to all callable functions produced by factories or passed as arguments (e.g., `potential_energy_fn`, `solve_statics_fn`, `mapping_fn`).
*   **Data vs. Logic**: Variables representing JAX arrays use descriptive names (e.g., `initial_displacements` instead of `state0`) to distinguish them from `CentroidalState` objects.

## Project Structure

```text
.
├── train.py                # End-to-End Inverse Design Optimizer
├── src/                    # Source code
│   ├── jax_backend/        # JAX physics, centroidal states, and solvers
│   │   ├── physics_solver/ # Energy functionals and static equilibrium (L-BFGS)
│   │   ├── training/       # Loss definitions and Optax training loop
│   │   ├── pipeline.py     # Orchestration of the 3 stages
│   │   └── initial_map.py  # Mapping factories and Shirley-Chiu projection
│   ├── topology/           # Tessellation and Unit Cell logic
│   ├── problem/            # Config schemas (Equinox), BCs, and loading
│   └── utils/              # Visualization (Princeton theme) & Math helpers
├── data/                   # Data-centric assets
│   ├── library/            # Pattern library (YAML)
│   ├── configs/            # Simulation configurations (YAML)
│   └── outputs/            # Results (Animations, Plots, JSON backups)
```

## Configuration (YAML)

Experiments are driven by structured YAML configurations. The mapping stage is now grouped for better modularity:

```yaml
mapping:
  map_type: "asymmetric_roots"
  use_shirley_chiu: true
  map_params: 
    tx: 0.0
    ty: 0.0
    s_val: 1.0
    roots: [10.0, 0.0]
    weights: [1.0]

physics:
  incremental: true
  num_load_steps: 20
  solver_maxiter: 1000
```

## Optimization

Using `jax.value_and_grad`, the pipeline supports inverse design by propagating gradients through the entire solver chain to optimize mapping parameters against a **Dual Loss Function** (Geometric Chamfer Distance + Physical Strain Energy Penalty).
