---
name: feedback-validity-solver-role
description: The validity solver's sole job is geometric validity, NOT target shape matching
metadata:
  type: feedback
---

The validity solver (Stage 1) must NOT attempt to match the target shape. Its only goal is to make the kirigami geometrically valid: closed hinges, non-intersecting faces, arm symmetry. Target shape matching is the exclusive responsibility of the GNN (Stage 0) via the Chamfer loss.

**Why:** Confusing these two responsibilities breaks the gradient signal — if Stage 1 pulls centroids toward the target, it fights Stage 0 and the separation of concerns collapses.

**How to apply:** When tuning Stage 1 optimization weights (`optimization_weights` in YAML), never introduce a target-fitting term there. Any target-related objective belongs only in the loss function evaluated after Stage 2.
