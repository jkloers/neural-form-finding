# legacy/ — archived approaches (frozen, not maintained)

Everything here was tried, worked to some degree, and was superseded by the differentiable
closed pipeline in `nff/`. It is kept for the record, **not** imported by anything in `nff/` or
`nff_paper/`, not collected by pytest, and not updated when the core moves on.

**To run any of it, check out the tag that froze it:**

```bash
git checkout pre-cleanup-2026-11      # every script below ran unchanged at this commit
```

The import paths inside these files were rewritten to their new locations on a best-effort
basis only; from the current tree they will generally not import.

| Folder | What it was | Why it was parked |
|---|---|---|
| `gnn/` | Stage-0 GNN mappings (EGNN, MPNN) on the open/deployed RDQK pattern, trained end to end through Stage-1 validity and Stage-2 physics by `train.py`. `topology_builder.py` built the open tessellations; `configs/` holds the architecture/problem YAMLs (incl. the 205-trial `c001` search). | Validity had to be *repaired* after every mapping (Stage 1), and the learned map never beat the closed chart, which guarantees validity by construction. |
| `open_pipeline/` | Notebooks for the analytic open-pattern maps (conformal / elliptical / polynomial) and the original prototype. | Same reason; the analytic and `direct_*` maps still live in `nff/stages/mapping.py` because the validity-ladder ablation uses them. |
| `sofa/` | SOFA FEM hinge oracle + Tesseract HTTP wrapper (`runtime/`, Homebrew Python 3.12 + Docker) and its JAX-side Bézier hinge optimiser (`nff_sofa/`, `scripts/`). | Replaced by the CalculiX oracle in `nff/rve/` and the learned hinge surrogate. The Docker images were deleted in a 2026-07 disk cleanup. |
| `experiments/` | One-off probes from the article period: grip-C sweeps, force-pocket probes, localisation and compliance sweeps, the superseded 5×5 ladder, early PET figure drafts. | Each answered one question; the answers are recorded in the agent memory and the article. |

## Invariants that held for the SOFA oracle (if it is ever revived)

- SOFA is a process-level singleton: serialise every `evaluate_unit_cell()` call (`_SOFA_LOCK`).
- Every `apply()` builds a fresh root node and unloads it in `finally`.
- `HexahedronFEM` must use `method="small"` with `PartialFixedConstraint(z)`; `method="large"`
  diverges under displacement control.
- The mesh is built from the `CentroidalState` (`face_centroids + cnv`, `hinge_node_pairs`), never
  from a parametric face size; `runtime/mesh_builder_gmsh.py` is a symlink to `nff_sofa/`.
- The moment pivot is the loaded-face centroid, so the applied load is a pure torque.

## Invariants that held for the GNN pipeline

- `num_layers` is a static Python int; `static_features` are built before the JIT boundary.
- Stage-1 validity weights: `connectivity` dominates `anchoring` 5:1 to 10:1; `n_proj_iters: 20`.
- `coverage_w` must never equal `chamfer_w` (recall weight becomes `chamfer_w²` and plateaus).
- At most 500 epochs on the canonical config; `contact: 0.0`.
