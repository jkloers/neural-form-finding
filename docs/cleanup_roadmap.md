# Cleanup roadmap — what remains after the 2026-11 restructure

The restructure (core `nff/`, article `nff_paper/`, archive `legacy/`) is done; decisions are in
`docs/cleanup_log.md`. These sessions finish the job. Each runs without the author except where
marked **(needs the author)**.

## Session 2 — prune `data/` (needs the author: one yes)

`data/` is gitignored (about 3.9 GB), so nothing in it has a git safety net.

1. Score the ~305 closed run folders in `data/outputs/runs/`:
   - **keep**: referenced by a tracked config, by `paper/` (sidecar `.txt` provenance in `paper/Images/`),
     or by an agent-memory entry;
   - **candidate**: `_probe*` / `_fast*` / `_smoke*` / `sweep*` runs, and failed runs (NaN loss or no
     `loss_history.json`).
2. Score the ~30 checkpoints in `data/surrogates/`: keep `hinge_surrogate_v2`, `_pet_v3`, `_pet_merged`,
   `_originray` and anything a tracked config points to.
3. Move every candidate to `data/archive/to_delete/` with a `MANIFEST.tsv` (path, size, reason).
   Ask the author once; delete only on that yes.
4. Write `data/README.md` describing every surviving folder.

## Session 3 — close Stage 1 — DONE 2026-11-26

Finding: the chart ablation does not just *read* Stage 1, it runs its free-vertex arms through the
live `forward_pipeline` / `compute_end_to_end_loss` with `map_type: direct_vertices` and
`validity_method: alternating_projection`, and four core diagnostics (Taylor test, gradient quality,
regularity, landscape spectra) are built on it. Moving that path out would have broken the evidence
for the closed chart, so the **ablation baseline stays live** and only the dead parts went:

- archived to `legacy/open_pipeline/`: the L-BFGS validity solver, the open-pattern penalties
  (`constraints.py` keeps only `hinge_connectivity`), the analytic open maps and `direct_transform`;
- `pipeline.py` Stage 0 accepts `closed_les | direct_vertices`, Stage 1 `none | alternating_projection`;
  anything else raises with a pointer to `legacy/`;
- `tests/test_chart_ablation_path.py` now guards arm D (it had no test before).

## Session 4 — core hardening

1. Build a fresh conda env (`kgnn_core`) from a pruned `environment.yml` (drop flax, jraph,
   e3nn-jax, wandb; add ezdxf, pillow). Leave `kgnn_mac` intact.
2. Follow the README quick start from a fresh shell, verbatim.
3. Add a `slow` pytest marker so a fast suite runs in under a minute.
4. GitHub Actions: fast suite + `tests/test_layering.py` on every push.

## Session 5 — research restart (needs the author)

Write `docs/research_restart.md`: what the pipeline can do today, and the open problems the
article left (surrogate return-memory gap, trust region / out-of-domain deployments, path
dependence of the hinge energy). The direction of the next phase is the author's call.
