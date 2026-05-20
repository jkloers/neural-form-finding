---
name: validity-solver-diagnosis
description: Stage 1 validity solver diagnosis and fix: anchoring dominated connectivity 20:1, hinges never closed
metadata:
  type: project
---

**Diagnosis:** In configs v1–v3_s2, `optimization_weights.anchoring: 2000` vs `connectivity: 100` gave a 20:1 ratio favoring anchoring. The L-BFGS stayed within ~0.01 of the Stage 0 output and could not close hinges. `void_length: 1.0` and `void_collinear: 1.0` were negligible vs all other terms, so voids were never enforced as parallelograms.

**Fix (v5 config):** Rebalanced to `connectivity: 3000, anchoring: 300` (10:1 in favor of connectivity) and `void_length: 500, void_collinear: 500`. Also `face_area: 10` (was 1).

**Why:** `hinge_connectivity` computes `sum((p1-p2)^2)`. With hinge gaps ~0.1 and ~20 hinges, `e_connect ~ 0.2`. At `connectivity=100`: penalty = 20. Meanwhile `initial_map_anchoring` computes `sum(c_diff^2) + sum(s_diff^2)`: for 16 faces × 4 nodes with deviation ~0.01, `e_anchor ~ 0.8`. At `anchoring=2000`: penalty = 1600. Solver accepted hinge gaps because the cost of moving faces was 80× the cost of tolerating gaps.

**Key file:** `validity_solver.py:111` — `LBFGS(fun=objective_fn).run(x0, ...)` — no maxiter/tol set (uses jaxopt defaults: maxiter=500, tol=1e-3). Weight ratio is the sole lever.

**DEFAULT_GEOMETRIC_WEIGHTS** in `validity_solver.py:27`: `{connectivity: 700, anchoring: 100, ...}` — these defaults are good (7:1 ratio favoring connectivity). The configs override them in the wrong direction.

**How to apply:** Never set `anchoring > connectivity` in `optimization_weights`. A 5:1 to 10:1 ratio favoring connectivity is appropriate. Void constraints need ~500 each to have any effect.
