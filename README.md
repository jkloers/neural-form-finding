# Neural Form-Finding

Unified centroidal form-finding pipeline for Kirigami tessellations using JAX.

## Overview

This project implements a multi-stage pipeline to find the equilibrium state of deployable Kirigami structures:
1.  **Stage 0: Initial Mapping** - Maps a flat tessellation to a target geometry.
2.  **Stage 1: Geometric Validity** - Optimizes face shapes to ensure connectivity and prevent overlap.
3.  **Stage 2: Static Solver** - Finds the physical equilibrium state using gradient-based minimization of mechanical energy.

## Project Structure

```text
.
├── main.py                 # Main entry point (run from here)
├── src/                    # Source code
│   ├── jax_backend/        # JAX physics and geometry engines
│   ├── topology/           # Tessellation and Unit Cell logic
│   ├── problem/            # Config schemas, BCs, and loading logic
│   └── utils/              # Visualization and math helpers
├── data/                   # Data-centric assets
│   ├── library/            # Pattern library (YAML)
│   ├── configs/            # Simulation configurations (YAML)
│   └── outputs/            # Results (Plots, Config backups)
└── notebooks/              # Interactive exploration
```