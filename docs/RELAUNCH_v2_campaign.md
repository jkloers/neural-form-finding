# RELAUNCH PROMPT — v2 hinge-surrogate campaign (paste to a fresh session after restart)

> Copy the block below into a new chat. It is self-contained.

---

Relaunch the v2 hinge-surrogate dataset campaign. Context: the code is committed (fillet is now a
swept 3rd geometry DOF; continuous ductile-damage `D` is the failure label; campaign CLI is
parameterized; surrogate is backward-compatible 5-or-6 feature). CalculiX is in the `ccx` conda env
(`nff` is NOT pip-installed there → use `python -m nff.scripts.X`, never `python nff/scripts/X.py`).
Retrain runs in `kgnn_mac`. Work from repo root `/Users/julienkloers/Documents/Code2/princeton/neural-form-finding`.

**Step 1 — 30-second pre-flight (confirm ccx + fillet + damage still work after restart):**
```bash
conda run -n ccx python -c "
from nff.rve.hinge_function import evaluate_hinge, HingeGeometry, DeploymentRay, HingeConstants
import numpy as np
r = evaluate_hinge(HingeGeometry(5.0, 90.0, 0.25), DeploymentRay(60.0, 0.3, 0.2, 15),
                   HingeConstants(thickness=1.5, r_win=45.0, n_through=3, lc_fillet_frac=0.3))
print('damage_p99 finite:', bool(np.isfinite(r.damage_p99).any()), '| n_samples:', r.n_samples)
"
```
Proceed only if `damage_p99 finite: True`.

**Step 2 — launch the campaign (caffeinated, ~6000 hinges, background + monitor):**
```bash
caffeinate -i conda run --no-capture-output -n ccx python -m nff.scripts.generate_hinge_dataset \
  --n 6000 --out sofa/output/hinge_dataset_v2 --parallel 8 --timeout 600 --batch-size 50 \
  --w-lig-min 1 --w-lig-max 20 --thickness 1.5 --angle 90 --steps 30 \
  --eta-a-max 1.5 --eta-s-max 1.0 --fracture-margin 2.5 \
  --n-through 3 --r-win 45 --lc-fillet-frac 0.3 \
  --fillet-min 0.10 --fillet-max 0.30
```
Run it in the BACKGROUND and monitor the per-batch prints. **Verify the FIRST checkpoint** (~10 min):
`sofa/output/hinge_dataset_v2.npz` must exist and contain a `damage_p99` column with finite values
and a swept `fillet_ratio` column — if not, stop and debug before burning the night. Expected
throughput ~50 hinges / 4 min → ~6000 in ~8 h.

**Step 3 — retrain once the campaign finishes (kgnn_mac):**
```bash
conda run -n kgnn_mac python -m nff.scripts.train_hinge_surrogate \
  --data sofa/output/hinge_dataset_v2 --out data/outputs/hinge_surrogate_v2 \
  --lam 0.65 --epochs 400 --hidden 64,64
```
It auto-detects the fillet DOF (→ 6 features) and trains the head on `D`. It prints the data-driven
`trust region (p99)` and saves it in the checkpoint (`stats["domain"]`).

**Step 4 — smoke-test the closed pipeline with the new surrogate:**
```bash
JAX_PLATFORMS=cpu conda run -n kgnn_mac python -m nff.scripts.run_closed \
  --config-dir closed --config-name rect_a4_beam_surrogate_v2 --every 100
```
(`rect_a4_beam_surrogate_v2.yaml` already points at `hinge_surrogate_v2.pkl`, thickness 1.5; `DOMAIN`
auto-loads from the checkpoint.) Then retune `loads.value` / `target_half_*` to explore the deeper
feasible deployment range the new `D` criterion + fillet unlock.

**Numbers rationale:** fillet `[0.10, 0.30]` = rehearsal-safe (≥1 degenerate, >0.3 eats the
ligament); `--n 6000` = ~3× the 2-DOF baseline for the added dimension; caffeinate `-i` prevents
idle sleep. Full details: `docs/hinge_dataset_v2_runbook.md`.
