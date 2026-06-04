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
total = chamfer_w * (precision + coverage_w * recall)          ← shape matching
      + w_stretch * u_stretch + w_shear * u_shear + w_contact * u_contact
      + w_hinge_gap * hinge_gap                                 ← connectivity (at Stage 0)
      + w_material * (mapped_area - initial_area)²
      - w_openness * log1p(void_area_stage1)                    ← reward: large void at Stage 1
      + w_void_closure * log1p(void_area_stage2)                ← penalty: remaining void after load
      - w_closure_delta * log1p(void_stage1 - void_stage2)      ← reward: void closed by loads
      - w_deformation * log1p(mean_sq_disp)                     ← DEPRECATED
      + w_reg * Σ param²
```

The `target_cloud` (n_points=500) is precomputed before JIT in `create_train_step` and closed over in `loss_fn`. Do not call `get_target_points()` inside the loss — it would be rebaked into XLA on every retrace.

### Coverage formula — critical

**`coverage` must be 1.0 for symmetric Chamfer.** The formula is `chamfer_w × (precision + coverage_w × recall)`. Setting `coverage_w = chamfer_w` inflates recall by `chamfer_w²` while precision only gets `chamfer_w` — a factor-of-chamfer_w asymmetry. Training immediately plateaus on recall; precision stagnates. Always `coverage: 1.0`, adjust `chamfer` alone for global scaling.

### Kirigami closing terms

Two independent closing terms can be used separately or together:

**`void_closure` — absolute penalty on Stage 2 void area:**
```
loss = +void_closure * log1p(void_area_stage2)   ← penalty
```
No Stage 1 reference. The model is penalised purely for open voids after loading. Cannot cheat by inflating the starting void. Strong, direct signal.

**`closure_delta` — reward for void area decrease Stage 1 → Stage 2:**
```
delta = max(0, void_area_stage1 - void_area_stage2)
loss  = -closure_delta * log1p(delta)             ← reward
```
Rewards the decrease in void area caused by the loads. A rigid-body swing leaves void area invariant (delta=0, no reward). The delta gradient is two-sided: encourages a reference that is open (large void_stage1) AND loads that close (small void_stage2). Can be gamed by inflating void_stage1, but `void_closure` together prevents this.

**Using both simultaneously** gives complementary signals: `void_closure` ensures the final state is closed; `closure_delta` ensures it is the loads doing the closing (not just a pre-collapsed reference).

**`deformation` — DEPRECATED.** `mean_sq_disp` rewards large absolute displacement, trivially satisfied by rigid-body swing around the clamp. Set `deformation: 0.0`.

### Validated MPNN + alternating-projection config (2026-06-04)

Canonical reference: `data/configs/architectures/mpnn_proj_base.yaml`

```yaml
chamfer: 5000.0
coverage: 1.0        # symmetric Chamfer — never set equal to chamfer
hinge_gap: 200.0     # prevents centroid gap explosion during chamfer optimisation
void_closure: 100.0  # penalty on remaining Stage 2 void area
closure_delta: 50.0  # reward for void closed by loads
deformation: 0.0     # deprecated
openness: 0.0        # redundant when closure_delta is active
contact: 0.0         # off — destabilises early training
num_layers: 3        # 3-hop message passing needed for hub face gradient
```

**Why hinge_gap=200:** With chamfer=5000 and clean symmetric gradients, the MPNN freely moves centroids to fit the circle, opening hinge gaps >1 face-length. This makes alternating projections unstable (large CNV corrections → face intersections → physics NaN). Balance formula: `hinge_gap × 24 × (0.1-unit gap)² ≈ chamfer × convergence_chamfer` → ~270. Safe value: 200.

**Why coverage=1.0 not equal to chamfer:** With coverage=500, chamfer=500: recall gets 250,000× weight, precision gets 500×. Training plateau on recall at epoch ~5; loss stays at 10³–10⁴. With coverage=1.0, chamfer=5000: symmetric, loss drops to 10¹–10², chamfer ≈ 0.013 at convergence.

**Key finding:** `void_closure=50` (delta-only) improved the hardest problems dramatically and was complementary to chamfer — not competitive. c009 improved 29× (0.409→0.014). Kirigami closing aligns with the circle target because a closed kirigami contracts inward.

### MPNN inner_depth parameter (2026-06-04)

The MPNN has two inner MLPs per layer: `phi_e` (edge update) and `phi_h` (node update). Each defaults to 2-layer (`inner_depth=2`). With `inner_depth=1`, each MLP is single-layer, halving the non-linear depth per step.

**Why inner_depth=1 matters for deep networks:** Each inner layer adds tanh activations to the gradient path. With `inner_depth=2` and L message-passing layers: 4L tanh ops per gradient path. At L=7: 28 tanh ops → severe gradient vanishing. With `inner_depth=1`: 2L ops. At L=6: 12 tanh ops → manageable.

**Critical bug fixed (2026-06-04):** `create_train_step` in `trainer.py` previously rebuilt `_static_features` from scratch (no `inner_depth`). This caused `apply_mpnn` to be called with `inner_depth=2` (default) during training, even when params were initialised with `inner_depth=1` (no W2/b2 keys) → KeyError at JAX trace time → NaN. Fix: thread `static_features` through `train_pipeline` → `create_train_step` so `inner_depth` reaches the JIT closure.

**Validated 5×5 config:** `data/configs/architectures/mpnn_proj_5x5.yaml`
- `num_layers: 6, inner_depth: 1` — 6 layers with shallow MLPs = 12 tanh ops.
  - 8 layers caused gradient vanishing (chamfer 0.286 → 0.526 regression).
  - 6 layers is optimal for 5×5 (center hub is ~5 hops from boundary).
- `hinge_gap: 200` — same as 2×2 in absolute units; stable for 5×5.
- Moment direction: **same-sign CCW for "iris" closing** (all 4 neighbors of center hub).
  - Alternating ±5 works but activates swirl, not contraction.
  - Same-sign +10 caused physics explosion (energy=433k). Safe value: **+5**.

### 5×5 tessellation geometry

- `width: 5, height: 5` → 100 faces, radius=2.5, total_area=18.75
- Center hub: **face 50** (centroid ≈ 7.78, 7.78), degree-4
- Center hub neighbors: **49** (below), **51** (left), **55** (right), **69** (above) — all degree-4 hubs
- For iris closing: clamp face 50, apply same-sign moments to {49, 51, 55, 69}

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
