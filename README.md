# Neural Form-Finding

End-to-end differentiable pipeline for the inverse design of deployable Kirigami structures. Combines a geometric mapping stage, a kirigami validity solver, and a physical simulator — all in JAX, fully differentiable.

Developed at the **Princeton Form Finding Lab**.

---

## Pipeline overview

The forward pipeline is a chain of three independent, swappable stages. Gradients flow end-to-end through all three via `jax.value_and_grad`.

```
Flat tessellation  (CentroidalState)
        │
        ▼  Stage 0 — Initial Mapping
        │   Any function CentroidalState → CentroidalState.
        │   Current options: GNN (EGNN, MPNN), analytical polynomial maps,
        │   asymmetric root maps, direct vertex optimisation.
        │
        ▼  Stage 1 — Kirigami Validity Solver
        │   Enforces the geometric laws of valid Kirigami (hinge connectivity,
        │   face non-intersection, symmetry). Applied fully or partially
        │   depending on the configuration.
        │   Methods: L-BFGS (jaxopt) or alternating projections.
        │
        ▼  Stage 2 — Physical Simulator
        │   Minimises total potential energy (elastic strain + contact − work)
        │   under the applied loads. Incremental load stepping via jax.lax.scan.
        │   Solver: L-BFGS (jaxopt), with optional Updated Lagrangian.
        │
        ▼  Loss
            Chamfer distance (boundary vertices vs. target shape)
          + physical energy penalties (stretch, shear, bending, contact)
          + optional: hinge gap, openness, deformation, regularisation
```

Each stage is defined by a single interface. Swapping an implementation only requires changing the routing inside `forward_pipeline` — nothing else.

---

## Installation

```bash
conda env create -f environment.yml
conda activate kgnn_mac

# Install the nff package in editable mode
pip install -e .
```

JAX is forced to CPU and float64 globally inside `train.py`. Do not set `JAX_PLATFORM_NAME` yourself.

---

## Running experiments

### Single config (legacy analytical maps)

```bash
python train.py --config-dir asymmetric_roots --config-name 2_free
python train.py --config-dir legacy/gnn/benchmark --config-name exp1_rot_clamp0
```

Config files live under `data/configs/<config-dir>/<config-name>.yaml`.

### Architecture + problem suite (recommended for GNNs)

```bash
# Full suite
python train.py --arch architectures/mpnn_base --suite problems/suite_2x2_rdqk

# Selected problems
python train.py --arch architectures/egnn_base --suite problems/suite_compressive \
                --problem-ids p001,p005,p010
```

Outputs are written to `data/outputs/runs/run_<timestamp>_<arch>/`.

### Programmatic

```python
import jax
jax.config.update("jax_enable_x64", True)

from nff.config.experiment import load_and_parse_config
from nff.topology.builder import build_tessellation
from nff.config.conditions import configure_tessellation
from nff.stages.state import CentroidalState
from nff.stages.pipeline import forward_pipeline
from nff.training.trainer import train_pipeline, TrainState, create_train_step

config = load_and_parse_config("data/configs/architectures/mpnn_base.yaml")
# ... build initial_state, call train_pipeline(map_params, initial_state, ...)
```

---

## Package structure

```
nff/
├── topology/               Tessellation geometry — pure NumPy, no JAX
│   ├── core.py             UnitPattern, Tessellation, IndexedFace, Hinge
│   └── builder.py          build_tessellation(pattern, nx, ny) → Tessellation
│
├── config/                 Experiment configuration — no JAX compute
│   ├── experiment.py       ExperimentConfig + YAML parsers
│   │                         load_and_parse_config()
│   │                         load_arch_config(), load_problem_suite()
│   │                         merge_arch_problem(), parse_map_params()
│   ├── conditions.py       Apply BCs and loads to a Tessellation
│   └── targets.py          Target shape point clouds (circle, heart, …)
│
├── models/                 Stage 0 — GNN architectures
│   ├── egnn.py             E(2)-equivariant GNN: init_egnn, apply_egnn
│   ├── mpnn.py             Non-equivariant MPNN: init_mpnn, apply_mpnn
│   └── graph_builder.py    Tessellation → jraph.GraphsTuple
│                             build_static_features(), state_to_graph()
│
├── stages/                 The three pipeline stages
│   ├── state.py            CentroidalState — the central data structure
│   ├── geometry.py         Pure JAX geometry primitives
│   ├── constraints.py      Geometric penalty functions for Stage 1
│   │
│   ├── mapping.py          Stage 0: analytical and GNN mapping engines
│   │                         build_mapping_fn(), apply_mapping()
│   │                         apply_gnn_mapping()
│   │
│   ├── validity.py         Stage 1 (L-BFGS): solve_geometric_validity()
│   ├── projection.py       Stage 1 (alternating): solve_alternating_projections()
│   │
│   ├── pipeline.py         forward_pipeline() — orchestrates all three stages
│   │
│   └── physics/            Stage 2 internals
│       ├── params.py         ReferenceGeometry, ControlParams, SolutionData
│       ├── energy.py         build_potential_energy() (elastic + contact)
│       ├── kinematics.py     Constrained DOF mapping (Dirichlet BCs)
│       ├── loading.py        Neumann BC assembly
│       ├── force_types.py    Geometry-dependent load types
│       └── statics.py        setup_static_solver() → solve_statics_fn()
│
├── training/               Optimisation loop
│   ├── loss.py             evaluate_physical_loss(), compute_end_to_end_loss()
│   └── trainer.py          TrainState, create_train_step(), train_pipeline()
│
├── utils/
│   ├── linalg.py           JAX linear algebra (vdot, rotation_matrix, …)
│   ├── visualization.py    Tessellation and animation plotting
│   └── pipeline_viz.py     Per-stage visualisation, loss curves
│
└── scripts/
    └── train.py            CLI entry point logic (imported by root train.py)
```

---

## Central data structure: `CentroidalState`

`CentroidalState` is an immutable `NamedTuple` that flows through all three stages. Fields split into two groups with different JAX semantics:

| Group | Storage | Fields |
|---|---|---|
| **Optimizable** | `jnp.array` float64 | `face_centroids (n,2)`, `centroid_node_vectors (n, max_nodes, 2)` + mechanical properties |
| **Fixed topology** | `np.array` int32 | `hinge_face_pairs`, `bond_connectivity`, `constrained_face_DOF_pairs`, … |

Topology arrays are **never** JAX Tracers. They are used as static indices inside JIT-compiled functions. Converting them to `jnp.array` breaks the solver.

Built via: `CentroidalState.from_tessellation(tessellation)`.

---

## Config YAML structure

**Architecture files** (`data/configs/architectures/`) define the model, mapping type, and training hyper-parameters.  
**Problem suite files** (`data/configs/problems/`) define BCs, loads, material, and physics per problem.  
Both are merged at runtime by `merge_arch_problem()`.

```yaml
tessellation:         # pattern name, grid size (width × height), total_area
mapping:              # map_type, map_params, domain_restriction, …
target:               # type (circle | heart), center, radius
optimization_weights: # Stage 1 penalty weights per constraint
physics:              # incremental, num_load_steps, stiffnesses, contact
material:             # k_stretch, k_shear, k_rot, density
boundary_conditions:  # clamped_faces (list of face ids or "boundary")
loads:                # [{face, dof, value}] or typed [{type, source_face, …}]
training:             # num_epochs, learning_rate, lr_schedule, grad_clip
loss_weights:         # chamfer, coverage, stretching, shearing, hinge_gap, …
visualization:        # stage0/1/2, save_outputs, show_hinges, …
```

---

## Naming conventions

- **`_fn` suffix** — every callable produced by a factory or passed as a callback: `potential_energy_fn`, `solve_statics_fn`, `mapping_fn`.
- **Descriptive array names** — `initial_displacements`, not `u0`; `face_centroids`, not `x`.
- **JAX arrays** — `jnp.array` for anything that may receive gradients.
- **Index arrays** — `np.array` (int32) for topology; never traced by JAX.
