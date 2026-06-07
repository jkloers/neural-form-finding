# AGENT_MEMORY.md — System directives for AI agents

Mandatory reading before modifying any file in this repository. Encodes architectural decisions, JAX invariants, and workflow rules. Violations produce silent numerical errors or JIT crashes.

---

## 0. Before you touch anything

1. Read this file completely.
2. Read `README.md` for the module map and stage definitions.
3. Read `CLAUDE.md` for the three-stage pipeline summary.
4. If your change touches `pipeline.py`, `statics.py`, `validity.py`, or any message-passing code (`egnn.py`, `mpnn.py`, `graph_builder.py`): **propose a written plan and wait for explicit approval before editing.**

---

## 1. The three stages — conceptual model

Understanding this is prerequisite to touching any file.

**Stage 0 — Initial Mapping**
Maps a flat `CentroidalState` to a new `CentroidalState`. This is the learnable part of the pipeline. The current implementations are:
- Analytical polynomial maps (conformal, asymmetric roots)
- GNNs: EGNN (E(2)-equivariant), MPNN (non-equivariant)

The stage is deliberately open-ended. Other implementations (direct vertex optimisation, diffusion models, etc.) plug in through the same interface: `CentroidalState → CentroidalState`. Adding a new mapping engine requires only a new branch in `apply_gnn_mapping()` or a new factory in `build_mapping_fn()`. Nothing else in the pipeline changes.

**Stage 1 — Kirigami Validity Solver**
Enforces the geometric laws of valid Kirigami on the mapped configuration. The constraints include hinge connectivity (vertex coincidence), face non-intersection, arm symmetry, and boundary fitting. They can be applied fully or partially depending on the experiment. Two solver implementations exist:
- L-BFGS via `jaxopt` (`validity.py`) — differentiable through `custom_vjp`; optimises both centroids and CNVs
- Alternating projections (`projection.py`) — closed-form; projects CNVs only, enforces hinge gap = 0 exactly

**Stage 2 — Physical Simulator**
Given a geometrically valid configuration, computes the static equilibrium by minimising total potential energy (elastic strain + contact − external work) under the applied loads. This is a physics solver, not a learning module. It knows nothing about the mapping or the GNN.

---

## 2. JAX conventions — non-negotiable

### 2.1 Functional purity

Every function that touches JAX arrays must be **pure**: no mutation of external state, no global variables, no side effects.

```python
# WRONG — mutates a dict that came from inside jax.value_and_grad
aux['grad_norm'] = global_grad_norm

# CORRECT — produces a new dict
aux = {**base_aux, 'grad_norm': global_grad_norm}
```

Impure functions silently produce wrong results under `jax.jit` because JAX traces Python execution, not runtime values.

### 2.2 PRNG key management

Every stochastic operation consumes a **fresh derived key**. Keys must never be reused.

```python
# WRONG — correlated samples
result1 = jax.random.normal(key, shape)
result2 = jax.random.normal(key, shape)

# CORRECT
key, subkey = jax.random.split(key)
result1 = jax.random.normal(subkey, shape)
key, subkey = jax.random.split(key)
result2 = jax.random.normal(subkey, shape)
```

`train_step` carries the current `rng` inside `TrainState`, splits it at the start of each step, and returns the updated key in the new state. Do not add stochastic operations that bypass this.

### 2.3 PyTree state — use `TrainState`

```python
class TrainState(NamedTuple):
    params:    Any       # map_params PyTree (GNN weights or analytical coefficients)
    opt_state: Any       # optax optimizer state
    rng:       jax.Array # PRNG key
```

`train_step_fn: TrainState → (TrainState, float, dict)` is a pure function. Never split `params` and `opt_state` into separate arguments — that defeats the PyTree contract and complicates checkpointing.

### 2.4 Tensor typing — always use jaxtyping

All public functions accepting or returning JAX arrays must carry `jaxtyping` annotations:

```python
from jaxtyping import Array, Float, Int
import numpy as np

def apply_egnn(
    params: dict,
    h_raw: Float[Array, "n_faces node_feat_dim"],
    x: Float[Array, "n_faces 2"],
    senders_np: Int[np.ndarray, "n_edges"],
    num_layers: int,
) -> tuple[Float[Array, "n_faces 2"], ...]:
```

Use `Float[Array, "..."]` for JAX arrays, `Int[np.ndarray, "..."]` for static NumPy index arrays.

### 2.5 Static vs dynamic — the JIT boundary

| Category | Storage | Rule |
|---|---|---|
| Optimizable geometry | `jnp.array` float64 | Traced; receives gradients |
| Topology index arrays | `np.array` int32 | Must stay NumPy; used as scatter indices |
| Hyperparameters, layer counts | Python int/str | Capture in closure before JIT |
| GNN static features | numpy dict | Precompute before JIT; close over as XLA constant |
| Target point cloud | `jnp.array` | Precompute in `create_train_step`, not inside the loss |

Anything closed over before `jax.jit` becomes a compile-time XLA constant. This is intentional for `_static_features` and `_target_cloud` in `create_train_step`. Recompilation happens only if those values change — they don't during a training run.

### 2.6 Reductions over PyTree leaves

```python
# WRONG — Python sum builds a sequential add tree in XLA
total = sum(jnp.sum(g**2) for g in leaves)

# CORRECT — XLA reduces in parallel
total = jnp.sum(jnp.stack([jnp.sum(g**2) for g in leaves]))
```

---

## 3. Architectural boundaries — do not cross

### 3.1 Dependency graph (strictly downward)

```
nff/topology/   →  nothing in nff/
nff/config/     →  nothing in nff/
nff/models/     →  nff/stages/state  (CentroidalState type only)
nff/stages/     →  nff/config/, nff/models/
nff/training/   →  nff/stages/, nff/config/
nff/scripts/    →  nff/training/, nff/stages/, nff/config/, nff/topology/
```

The cross-layer dependency `config/ → stages/` was eliminated during refactoring by inlining `parse_map_params` into `nff/config/experiment.py`. Do not reintroduce any upward dependency.

### 3.2 Models know nothing about physics

`nff/models/` contains only message-passing operations. A GNN receives node features (density, area, boundary flag, loads) and outputs centroid displacements and local transformation matrices. It must **not**:
- import from `nff/stages/physics/`
- know about hinge stiffnesses, contact forces, or energy minimisation
- accept `CentroidalState` directly — it receives plain arrays `h_raw` and `x`

The GNN↔stage interface is `apply_gnn_mapping()` in `nff/stages/mapping.py`. That function is the only bridge.

### 3.3 Stages do not contain learning logic

`nff/stages/` computes physical and geometric quantities. It does not:
- reference loss weights
- call optimisers
- implement training loops

### 3.4 CentroidalState field groups

```python
# JAX arrays (float64) — may receive gradients
face_centroids:         Float[Array, "n_faces 2"]
centroid_node_vectors:  Float[Array, "n_faces max_nodes 2"]
load_values, k_stretch, k_shear, k_rot, density, initial_face_areas

# NumPy arrays (int32) — topology indices, NEVER a JAX Tracer
hinge_face_pairs, hinge_node_pairs, bond_connectivity,
hinge_adj_info, boundary_face_node_ids, void_opposite_node_pairs,
constrained_face_DOF_pairs, loaded_face_DOF_pairs
```

Topology arrays are used as indices in scatter operations inside JIT (`jnp.zeros(...).at[hinge_face_pairs].add(...)`). Converting them to `jnp.array` is wrong — they would be traced instead of used as static indices.

---

## 4. Stage-specific invariants

### Stage 0 — Mapping (`nff/stages/mapping.py`)

- `static_features` must be precomputed **before** the JIT boundary (in `create_train_step`), then closed over. Do not call `build_static_features()` inside any JIT-compiled function.
- `num_layers` is a static Python `int`. Pass it explicitly to `apply_egnn` / `apply_mpnn`. Do not derive it by inspecting param dict keys at runtime — that was the old pattern, now removed.
- Adding a new mapping engine: implement `init_<name>` and `apply_<name>` in `nff/models/`, add a routing branch in `apply_gnn_mapping()`, and add the corresponding `init_<name>` call in `nff/scripts/train.py:_init_gnn_params()`.

### Stage 1 — Validity solver (`nff/stages/validity.py`, `nff/stages/projection.py`)

- `validity_cfg.validity_method` selects the implementation. Both are valid; neither is deprecated.
- The L-BFGS solver uses `jaxopt`'s `custom_vjp` — gradients flow through it. Do not wrap or re-implement the solver without understanding this mechanism.
- The target cloud passed to Stage 1 is `n_points=200`. The Chamfer loss target cloud is `n_points=500`. They are separate arrays.
- **Stage 1 must not match the target shape.** Its only objective is geometric validity: closed hinges, non-intersecting faces, symmetry. Target shape matching belongs exclusively to Stage 0 via the Chamfer loss. Introducing any target-fitting term into `optimization_weights` collapses the separation of concerns and corrupts the gradient signal.
- **Validity weight ratio — critical:** `connectivity` must always dominate `anchoring`. A ratio of 5:1 to 10:1 in favour of connectivity is appropriate. Reversing this (anchoring > connectivity) prevents Stage 1 from closing hinges — the solver accepts hinge gaps because the cost of moving faces exceeds the cost of tolerating gaps. Void constraints (`void_length`, `void_collinear`) need values around 500 to have any effect. The `DEFAULT_GEOMETRIC_WEIGHTS` in `validity.py` encode a good baseline; configs that override these should not violate the connectivity > anchoring invariant.

### Stage 2 — Physics simulator (`nff/stages/physics/`)

- `bond_connectivity` is an `np.int32` array precomputed in `Tessellation._to_dict()`. It must never become a JAX Tracer. It is used as a flat index array: `face_id * n_nodes + local_node_id`.
- `incremental=True` uses `jax.lax.scan` for load stepping — JIT-compilable. `incremental=False` applies the full load in one step.
- `force_vals_jax` for geometry-dependent loads must be an **explicit solver argument**, not a closure. `jaxopt`'s `custom_vjp` cannot differentiate through closed-over JAX Tracers.
- `updated_lagrangian=True` recomputes `ReferenceGeometry` from the current deformed state at each load step.

---

## 5. Loss function components

```
total = chamfer + coverage * chamfer_coverage
      + w_stretch * u_stretch + w_shear * u_shear + w_bend * u_bend + w_contact * u_contact
      + w_hinge_gap * hinge_gap
      + w_material * (mapped_area - initial_area)²
      - w_openness * log1p(void_area)      ← reward: large void area at Stage 1
      - w_deformation * log1p(u_bend)      ← reward: bending energy at Stage 2
      + w_reg * Σ param²
```

The `target_cloud` (n_points=500) is precomputed before JIT in `create_train_step` and closed over in `loss_fn`. Do not call `get_target_points()` inside the loss — it would be rebaked into XLA on every retrace.

**`chamfer` and `coverage` weights must always be equal.** They are the two halves of the bidirectional Chamfer distance (precision and recall toward the target shape). Setting them to different values breaks the symmetry of the fitting signal. Always verify `loss_weights.chamfer == loss_weights.coverage` when writing or editing a config.

**`openness` and `deformation` rewards must stay ≤ 20 % of the `chamfer` signal.** These secondary rewards encourage a good initial configuration (open void area) and meaningful deformation. If they exceed ~20 % of the chamfer weight, they dominate training and the network stops learning the shape.

---

## 6. Style conventions

**Language** — English only. No French in comments, docstrings, or variable names.

**Docstrings**

```python
def my_function(x: Float[Array, "n 2"]) -> Float[Array, "n"]:
    """One-line summary.

    Longer explanation only when the WHY is non-obvious: a hidden constraint,
    a workaround for a specific bug, a non-obvious invariant.

    Args:
        x: (n, 2) centroid positions.

    Returns:
        (n,) scalar per face.
    """
```

Do not write: what the code does (identifiers say that), which PR motivated it, or "used by X".

**Naming**

- `_fn` suffix on every factory-produced or callback callable: `potential_energy_fn`, `solve_statics_fn`, `mapping_fn`.
- Descriptive array names: `initial_displacements` not `u0`, `face_centroids` not `x`.
- Factory prefix: `build_*` (e.g., `build_mapping_fn`, `build_potential_energy`).
- Private helpers: `_` prefix.

**Imports**

- No circular imports. The §3.1 dependency graph is the authority.
- No lazy imports inside function bodies except where documented (Metal crash avoidance).
- All imports at the top of the file.

---

## 7. When to propose a plan (mandatory)

Do not edit these files without first proposing a written plan and receiving explicit approval:

| File | Reason |
|---|---|
| `nff/stages/pipeline.py` | Orchestrates all three stages; failures are silent |
| `nff/stages/physics/statics.py` | jaxopt custom_vjp differentiation contract |
| `nff/stages/physics/energy.py` | Energy errors corrupt gradients numerically |
| `nff/stages/validity.py` | jaxopt integration; changes break gradient flow |
| `nff/models/egnn.py` | E(2) equivariance invariant — easy to break silently |
| `nff/models/mpnn.py` | Same; non-equivariant but directional edge features are load-bearing |
| `nff/models/graph_builder.py` | JIT boundary for static features; order of fields is load-bearing |
| `nff/stages/state.py` | Every downstream function depends on the field layout |
| `nff/config/experiment.py` | Silent field name mismatches produce wrong experiments |

---

## 9. SOFA Physics Oracle & Tesseract Integration

**Branch:** `Tesseract_SOFA`  
**Status:** Phase 3b complete — unified mesh, correct hinge geometry, kirigami loading. Docker not yet built/tested.

---

### 9.1 Module map and roles

```
sofa/
├── simulate_cell.py     Thin entry point — evaluate_unit_cell() public API.
│                        Imports from geometry/materials/scene_builder.
│                        Run: ./sofa/run_sofa.sh sofa/simulate_cell.py
├── geometry.py          Unified hex mesh builder — build_unified_mesh().
│                        Face panels + hinge strips in one MechanicalObject.
│                        Hinge positions from patterns.yaml vertex pairs (corners).
├── materials.py         SvK energy + von Mises stress post-processing.
├── scene_builder.py     SOFA scene — FixedConstraint on F0, rotation loading on F1.
│                        Loading: F1 rotates about H0 fold axis at x=a (not shear).
├── dump_results.py      Saves nodes_nat/cur + hexes + bc_masks + QoIs to .npz.
│                        No matplotlib (avoids SOFA+Qt crash on macOS).
│                        Args: --mode rotation|moment --angle [deg] --moment
│                              --arm-width --fold-length --thickness
│                        Full 3D rotation formula (no small-angle approx); works 0-90+°
│                        Run: ./sofa/run_sofa.sh sofa/dump_results.py [args]
├── visualize.py         Two-panel "kirigami opening" figure:
│                        LEFT = flat natural state top-down (hatched void cuts, hinge labels)
│                        RIGHT = 3D deformed butterfly fold (boundary surface, isometric)
│                        Run: conda run -n kgnn_mac python sofa/visualize.py --npz ...
├── run_viz.sh           Combined wrapper: dump → visualize.
│                        Usage: ./sofa/run_viz.sh [--save] [--fold-length 0.010 ...]
│                        Output: sofa/output/sofa_result.png
└── run_sofa.sh          macOS launcher — sets SOFA env vars, calls Homebrew Python 3.12

tesseract/              (kept in sync with sofa/ — copy shipped in Docker)
├── tesseract_api.py     Tesseract API — InputSchema/OutputSchema, apply(), FD grads.
├── tesseract_config.yaml
├── tesseract_requirements.txt
├── simulate_cell.py    } Mirror of sofa/*.py — keep in sync manually after sofa/ changes
├── geometry.py         }
├── materials.py        }
└── scene_builder.py    }

data/configs/sofa/
└── sandbox_1x1.yaml     NFF config for 1×1 unit_RDQK_0 tessellation (4 faces,
                         total_area=4.0, direct_transform mapping, 200 epochs).
                         Use with: python train.py --config-dir sofa --config-name sandbox_1x1
```

---

### 9.2 Two-environment architecture

| Environment | Python | SOFA | Purpose |
|---|---|---|---|
| `kgnn_mac` (conda) | 3.10 | — | JAX / NFF pipeline |
| Homebrew + `kgnn_sofa` | 3.12 | macOS ARM64 binary | Local SOFA development |
| Docker (Tesseract) | 3.12 | Linux x86_64 binary | Deployed oracle |

The Docker container is **self-contained**. It communicates with the JAX pipeline exclusively via the Tesseract HTTP API — no shared Python env, no shared filesystem.

---

### 9.3 Critical invariants — do not violate

**SOFA global state:** SOFA uses a process-level singleton registry. Concurrent `evaluate_unit_cell()` calls in the same process corrupt state. The `_SOFA_LOCK` in `tesseract_api.py` serialises all calls. Do not remove it or call `evaluate_unit_cell()` from multiple threads without this lock.

**Stateless `apply()`:** Every call to `apply()` must create a fresh `Sofa.Core.Node("root")` and call `Sofa.Simulation.unload(root)` in a `finally` block. Never cache a root node between calls.

**Energy extraction:** `Sofa.Simulation.getPotentialEnergy()` does not exist in SOFA v25.12. Energy is computed analytically via `_euler_bernoulli_energy()` in `simulate_cell.py`, applied to the equilibrium positions returned by SOFA. This is intentional — do not attempt to re-introduce SOFA API energy extraction without verifying the v25.12 API.

**SOFA plugin names (v25.12):** Component names changed between versions. The working set for our scene is:
- `DefaultAnimationLoop` (required for `Sofa.Simulation.animate()`)
- `CGLinearSolver` — from `Sofa.Component.LinearSolver.Iterative`
- `CollisionResponse` (NOT `DefaultContactManager` — removed in v24.12)
- `Sofa.Component.Visual` (NOT `Sofa.Component.Visual.Style`)

**BeamFEMForceField template:** Requires `Rigid3d` MechanicalObject. Does not support `listRadius` or `useShearStressComputation` in v25.12. Use separate child nodes (`mech.addChild("arms")`, `mech.addChild("hinges")`) each with their own `EdgeSetTopologyContainer` + `BeamFEMForceField` to achieve per-group radius.

---

### 9.4 Gradient strategy

SOFA has no adjoint. Gradients come from Tesseract's `finite_difference_jacobian` (central differences, `eps=fd_eps`, default `1e-5`). Cost: 2 × n_differentiable_inputs SOFA simulations per gradient call (6 total for Phase 2's 3 inputs).

---

### 9.5 Phase roadmap

| Phase | Status | Description |
|---|---|---|
| 1 | Done | Standalone `simulate_cell.py` — 1×1 unit cell, BeamFEM, analytical energy |
| 2 | Done | Tesseract wrapper — Docker build recipe, HTTP API, finite-diff gradients (BeamFEM, superseded) |
| 3a | Done | HexahedronFEM physics + Tesseract wrapper synced (superseded by 3b) |
| 3b | Done | Unified mesh (faces+hinges), correct hinge corners, rotation loading on F1 |
| 3c | Planned | Face-face contact (FreeMotionAnimationLoop) |
| 3d | Planned | Plasticity (d_plasticYieldThreshold) |
| 4 | Planned | Use Tesseract strain_energy as reward signal / fine-tuning in NFF training loop |

---

### 9.6 Files that require a written plan before editing

| File | Reason |
|---|---|
| `sofa/simulate_cell.py` | Public API — changes affect Tesseract callers |
| `sofa/geometry.py` | Any geometry change invalidates the hex mesh layout and SvK reference positions |
| `tesseract/simulate_cell.py` | Must stay in sync with `sofa/simulate_cell.py` |
| `tesseract/geometry.py` | Must stay in sync with `sofa/geometry.py` — shipped in Docker |
| `tesseract/tesseract_api.py` | Schema changes break existing JAX callers |
| `tesseract/tesseract_config.yaml` | Wrong SOFA download URL or ENV will silently break the Docker build |

---

## 8. Quick reference — critical functions

| Function | File | Role |
|---|---|---|
| `forward_pipeline` | `stages/pipeline.py` | Single entry point for inference; runs all three stages |
| `compute_end_to_end_loss` | `training/loss.py` | `forward_pipeline` + loss; input to `jax.value_and_grad` |
| `create_train_step` | `training/trainer.py` | Builds JIT-compiled step closure; precomputes static features and target cloud |
| `CentroidalState.from_tessellation` | `stages/state.py` | Converts `Tessellation` → JAX-ready state |
| `build_static_features` | `models/graph_builder.py` | Precomputes graph topology for GNNs; must be called before JIT |
| `setup_static_solver` | `stages/physics/statics.py` | Returns `solve_statics_fn`; wraps jaxopt L-BFGS |
| `solve_geometric_validity` | `stages/validity.py` | Stage 1 L-BFGS; differentiable via jaxopt custom_vjp |
| `apply_gnn_mapping` | `stages/mapping.py` | Routes GNN output back into CentroidalState; GNN↔stage bridge |
| `merge_arch_problem` | `config/experiment.py` | Merges architecture YAML and problem YAML at runtime |
