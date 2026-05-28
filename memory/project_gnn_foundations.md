---
name: project-gnn-foundations
description: GNN integration foundations — graph structure, dummy GNN, gradient validation, JAX-Metal JIT bug
metadata:
  type: project
---

GNN module added at src/jax_backend/gnn/ with three components:
- graph_builder.py: build_static_graph_features() (precomputed outside JIT), state_to_graph() (jraph.GraphsTuple per forward pass)
- dummy_gnn.py: init_dummy_gnn() + apply_dummy_gnn() — flat dict PyTree, scatter-add aggregation, near-identity init (scale=0.01)
- Node features (7-dim): density, initial_area, is_boundary, is_clamped, load_x, load_y, load_theta
- Edges: bidirectional from hinge_face_pairs, feature = euclidean distance (SO(2)-equivariant scalar)

**Why:** Foundational step before implementing EGNN. Validates that the differentiable pipeline can optimize GNN weights.

**Key constraint — JAX-Metal JIT bug:** When map_type starts with 'gnn_', JIT is disabled (use_jit=False in train_pipeline). The Metal XLA backend on Apple M4 crashes during compilation of programs combining scatter-add + nested LBFGS backward passes. The gradient flow IS correct (validated without JIT: 5 epochs, loss 3.72→3.63, all gradients non-zero non-NaN). Re-enable JIT when the Metal bug is fixed or when running on CPU-only/Linux.

**Architecture pattern:** build_static_graph_features() must be called OUTSIDE the JIT boundary (in create_train_step, not inside forward_pipeline) because calling NumPy operations on closure-constant JAX arrays inside XLA compilation triggers the Metal crash.

**Config:** data/configs/gnn/dummy_test.yaml — map_type: gnn_dummy, map_params: {hidden_dim: 16, seed: 0}

**How to apply:** When implementing the real EGNN, keep the same interface (apply_gnn_mapping in initial_map.py, static_features precomputed in trainer.py). Re-enable JIT by removing the gnn_ check in train.py once Metal is fixed.
