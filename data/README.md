# data/

Everything here is gitignored except this file, `configs/closed/*.yaml` and the lab
`experiments/*/manifest.json` / `README.md`. Nothing else has a git safety net: **move, never
delete**, unless the author has said yes.

| Folder | What it holds | Status |
|---|---|---|
| `configs/closed/` | Closed-pipeline experiment configs (tracked). Scratch configs start with `_` and stay local. | core |
| `library/` | `patterns.yaml` (unit-pattern definitions every config loads) and unit-cell helper scripts. | core |
| `fea/` | CalculiX hinge datasets (`hinge_dataset_*.npz` + `.json` provenance + campaign `.log`), oracle reproducibility audits, `path_priors/` (deployment-path envelopes that aim a campaign). The PET datasets took days of CPU: **irreplaceable**. | core |
| `surrogates/` | Trained hinge surrogates (`hinge_surrogate_*.pkl` + `.json` sidecar with stats, provenance and metrics), capacity sweeps, path-dependence and return-memory audits, parity/training plots. Model of record: `hinge_surrogate_pet_v3.pkl`. The ablation checkpoints back measured results in the agent memory, so all are kept. | core |
| `experiments/` | The physical PET campaign: tensile coupons (`raw/`, `processed/`, `video/`) and the two 3×3 deployment sessions (photographs + tracked `manifest.json`). **Irreplaceable.** | core |
| `outputs/runs/` | One folder per `run_closed.py` run: `config.yaml`, `loss_history.json`, `design_traj.npz`, best params, deployed-sheet and cut-pattern figures. 179 kept: every run referenced by code, the paper, docs or memory, plus the newest run of each config. | core |
| `outputs/<study>/` | Derived data behind specific analyses and article plates (validity ladder, chart ablation, design cloud, hinge paths, load paths, shield, ume probes, ...). | article / research |
| `archive/legacy_gnn/` | GNN/MPNN runs and the 205-trial `c001` hyper-parameter search. | legacy |
| `archive/legacy_sofa/` | SOFA Bézier hinge-optimiser outputs. | legacy |
| `archive/katagami/` | Katagami stencil artwork outputs (contact sheets, SVGs). | legacy |
| `archive/to_delete/` | Runs staged for deletion (superseded reruns, crashed runs, scratch configs), with `MANIFEST.tsv` giving each one's size and reason and `ALL_RUNS_SCORED.tsv` giving the verdict for every run. Delete only on the author's yes. | pending |

Where code expects things:
- `hinge_model.checkpoint` in a config points into `surrogates/` (the default in `ExperimentConfig` is
  `data/surrogates/hinge_surrogate.pkl`, the original steel surrogate).
- `generate_hinge_dataset.py` writes `fea/<name>.npz`; `train_hinge_surrogate.py` reads it and writes
  `surrogates/<name>.pkl`.
- `run_closed.py` writes `outputs/runs/run_<timestamp>_<config>/`.
