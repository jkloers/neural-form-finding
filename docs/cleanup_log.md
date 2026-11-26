# Cleanup log — 2026-11 restructure

One line per judgement call made during the restructure, so it can be audited after the fact.
Tag `pre-cleanup-2026-11` is the last commit where every script (GNN, SOFA, paper) ran unchanged.

## Decisions

- **Repo is public.** Manuscript prose (`docs/*_reapply.py`, the editing audits, `pending_measurements.md`,
  `ume_closing_figure.md`, `length_review_plan.md`, `vocab_audit_*.patch`, `pet_table_block*.txt`, paper/slide/figure
  backups) is archived LOCALLY under `nff_paper/manuscript/` and gitignored, never pushed. `paper/` stays the
  Overleaf-synced, gitignored source of truth.
- **`slides/assets/` is not versioned**: third-party kirigami images, logos and lab photographs. The deck source is.
- **Closed configs are versioned** (`data/configs/closed/*.yaml`, scratch configs prefixed `_` stay ignored).
- **The `.gitignore` lab-manifest exception was dead** (`data/*` excluded the parent dir); fixed, the three manifests
  and the deployment README are now tracked.
- **Stage 1 (`validity`, `projection`, `constraints`) and the `direct_*` maps stay in core for now**: they feed
  `chart_ablation`, which produces the article's validity-ladder figure. Their exit is roadmap session 3.
- **`rve/materials/paper.py` stays in core**: it is paper the *material*, not the article.
- **Merging into `main` was blocked by the session's permission policy**, so the cleanup branch was cut from
  `hinge-quarantine-and-oracle-fixes` (PR #25) instead. Merge PR #25, then the cleanup PR.
- **`stash@{0}` kept**: labelled redundant with 318266f, but dropping it gains nothing.
- **Deleted**: root LaTeX artefacts (`sn-article.*`, `texput.log`, `b4.log`, all from a failed root-level build),
  empty `p.txt`, every `__pycache__/`, and the remote branches `worktree-pet-surrogate-campaign` and
  `claude/refine-local-plan-dhoqnt` (both verified fully contained in the pushed branch).
- **Archived to `legacy/`** (`git mv`, history kept): see `legacy/README.md`. Legacy configs that were gitignored
  under `data/configs/` are now tracked inside `legacy/*/configs/`, so the archive is complete.
- **Legacy data moved, not deleted**: GNN runs (7 `*_egnn_*`/`*_mpnn_*`/`c001`), `plots2_cs_asy_centr`, `search/`,
  `mpnn_baseline/` → `data/archive/legacy_gnn/`; `outputs/hinge_opt/` (SOFA optimiser) → `data/archive/legacy_sofa/`.
  `figmat_4_asym` runs are closed-pipeline (asymmetric figure-matrix case), kept.
- **Path-dependence and return-memory tooling stays in core** (`audit_return_memory`, `replay_hinge_path`, ...):
  the return-memory gap is an open surrogate problem, not paper decoration.
- **Fixed in passing**: `rom_best_fit.py` used `np` without importing numpy.
- **Test baseline before any move**: 877 passed (CPU, 8 min 19 s). Smoke baseline (`circle_4x4_surrogate`,
  15 epochs): ROM best loss 2.7544e-01, surrogate 2.7365e-01.
