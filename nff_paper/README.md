# nff_paper/ — the article and the defense deck

Everything that exists to produce the PFE article (`paper/`, Overleaf-synced and gitignored)
and the 2026 defense slides. The research pipeline lives in `nff/`; this package **imports
`nff`, and nothing in `nff` imports it** (enforced by `tests/test_layering.py`).

| Folder | Contents |
|---|---|
| `figures/` | One generator per plate in `paper/Images/` (plus the slide-width variants). Regenerate a plate with the flags it was published with: the sidecar `.txt` next to each image in `paper/Images/` records them. |
| `analysis/` | Scripts whose printed numbers the article cites (validity ladder, design ensembles, cell-family corpus, force transmission, ROM best fit, ume collapse, ...). |
| `closed_support/` | Library modules only the article uses: closed-form shield kinematics and rendering, unit-cell families, design ensembles, one-DOF and perturbation analyses, the training film. |
| `lab/` | Photograph analysis of the physical 3×3 PET deployment and the hinge buckle close-up. |
| `slides/` | Beamer source of the defense deck. `build.sh` stages plates from `paper/Images/`; `make_slide_figures.sh` re-renders the slide-width ones. `assets/` (third-party images, lab photos) is local only. |
| `manuscript/` | Build scripts for the article and the French extended summary. `manuscript/private/` (gitignored) holds the editing history: the `*_reapply.py` patch scripts, review audits and `.tex` backups. |
| `review/` | Night-review agent reports (gitignored, regenerated). |

Run anything here from the repo root, e.g.

```bash
JAX_PLATFORMS=cpu conda run -n kgnn_mac python nff_paper/figures/plot_taylor.py --help
```
