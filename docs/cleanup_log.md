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
- **Paper isolation** (`nff_paper/`): figure generators, cited analyses, closed-sheet support modules (shield,
  katagami, cell families, design ensembles, one-DOF, training film), katagami topology art, lab photo analysis,
  the deck and the manuscript build scripts. 21 tests that exercise them moved to `tests/paper/`.
- **Kept in core although the paper also uses them**: `utils/figstyle.py` (house style, imported by core
  run visualisation), `closed/force_network.py` (load paths), `closed/path_dataset.py` (harvest),
  `closed/chart_ablation.py` + `closed/gradcheck.py` (gradient verification; `taylor_test` needs them),
  `calibration/` tensile modules (they fix the PET material law the oracle uses).
- **Core scripts regrouped**: `scripts/surrogate/` (oracle, harvest, audits, path dependence, campaign control),
  `scripts/design/` (target calibration, cut export), `scripts/calibration/`, `scripts/diagnostics/`. The three
  entry points stay at `nff/scripts/`.
- **`run_pet_campaign.sh`** now runs from the repo it lives in; it pointed at a deleted worktree copy.
- **Review agents** (`.claude/agents/relecture-*`, local) now read and write `nff_paper/review/`.
- **README assets**: `closing_mpnn.gif` and the open-pipeline diagram moved to `legacy/gnn/`.
- **`pyproject.toml`**: entry points `nff-closed`, `nff-train-surrogate`, `nff-hinge-dataset` replace `nff-train`;
  `jraph`/`flax` moved to a `legacy` extra (core imports neither); `jax-md` (energy.py) and `pillow` added.
- **Verified after the move**: 218/218 modules import; ROM and surrogate smoke losses identical to the baseline;
  `validity_ladder` and `taylor_convergence` regenerate with the published numbers.
- **`nff/rve/sweep.py` archived** to `legacy/experiments/rve_sweep.py`: an orphan batch runner (nothing imported it), superseded by `rve/dataset.py`. The author confirmed: everything SOFA/gmsh/Bézier-hinge goes to legacy. **gmsh stays in core only as the CalculiX RVE mesher** (`rve/ccx_solver.py`), which the surrogate dataset depends on.
- **Session 2 (data):** 128 of 307 run folders staged in `data/archive/to_delete/` (475 MB: 74 superseded reruns, 44 crashed with only a config, 10 scratch configs); 179 kept (referenced anywhere, newest of each config, or hand-named). June runs without `loss_history.json` are NOT failures (they predate it) and were kept. All 47 MB of surrogate checkpoints kept: the ablations back memory results. Deletion awaits the author's yes.
