# Hinge surrogate v2 — deeper campaign runbook

Goal: retrain the hinge-energy surrogate on a **wider, deeper** dataset now that failure is a
continuous ductile-damage `D` (not the conservative `peeq_p99 ≥ eps_f` stop). This lifts the
angular ceiling and covers a bigger geometry/displacement envelope.

**What changed in code (already committed):** continuous damage `D` (`nff/rve/damage.py`), wired
through the oracle → dataset → head label → the loss break-barrier; the campaign CLI is
parameterized; the surrogate's trust region (`DOMAIN`) is now data-driven from the checkpoint, so a
wider dataset auto-widens the OOD barrier with no manual edit.

Environments: campaign runs in `ccx` (CalculiX + gmsh), retrain in `kgnn_mac` (JAX).

**Invocation note:** `nff` is not pip-installed in the `ccx` env, so run scripts with the **`-m`
module form** (`python -m nff.scripts.X`) from the repo root, NOT `python nff/scripts/X.py` (the
latter drops the repo root from the path → `ModuleNotFoundError: nff`).

---

## 0. Validate the D extraction FIRST (one hinge, ~1 min)

`ccx` is not on the dev machine, so `_damage_pct`'s `.frd` field handling is untested against live
output. Confirm `damage_p99` is finite before spending hours:

```bash
conda run -n ccx python -c "
from nff.rve.hinge_function import evaluate_hinge, HingeGeometry, DeploymentRay, HingeConstants
import numpy as np
r = evaluate_hinge(HingeGeometry(5.0, 90.0), DeploymentRay(60.0, 0.3, 0.2, 20),
                   HingeConstants(thickness=1.5))
print('peeq_p99  :', np.round(r.peeq_p99, 3))
print('damage_p99:', np.round(r.damage_p99, 3))   # must be finite; ~tracks peeq, LOWER in shear
"
```
If `damage_p99` is all-NaN → the `.frd` stress/PEEQ field names differ from assumption; fix
`nff/rve/ccx_solver.py::_damage_pct` (the `('PEEQ','PE')` / `'STRESS'` keys) before the campaign.

## 1. Full dataset campaign (overnight; batched + checkpointed)

```bash
conda run -n ccx python -m nff.scripts.generate_hinge_dataset \
  --n 2000 --out data/fea/hinge_dataset_v2 --parallel 8 --timeout 600 \
  --w-lig-min 1 --w-lig-max 20 \
  --thickness 1.5 \
  --angle 90 --steps 30 \
  --eta-a-max 1.5 --eta-s-max 1.0 \
  --fracture-margin 2.5 \
  --n-through 3 --r-win 45 --lc-fillet-frac 0.3
```

| flag | why |
|---|---|
| `--w-lig-min/max 1 20` | wider ligament range (stiffness lever) |
| `--thickness 1.5` | standard laser-cut gauge; stiffer, higher strain/rotation |
| `--fracture-margin 2.5` | run PAST first fracture into the ductile-tearing regime `D` measures |
| `--eta-a-max/eta-s-max 1.5/1.0` | deeper displacement envelope |
| `--angle 90 --steps 30` | larger folds at ~3°/increment |
| `--n-through 3` | smoother plastic bending through the thickness |
| `--r-win 45` | Saint-Venant window ≥2×w_lig at the wide end (smoother, less boundary drift) |
| `--lc-fillet-frac 0.3` | finer fillet → less energy jitter across geometries |

Writes `data/fea/hinge_dataset_v2.npz` + `.json`. Safe to interrupt (checkpoints every 50 jobs).

## 2. Retrain (auto-picks up `D`)

```bash
conda run -n kgnn_mac python -m nff.scripts.train_hinge_surrogate \
  --data data/fea/hinge_dataset_v2 --out data/surrogates/hinge_surrogate_v2 \
  --lam 0.65 --epochs 400 --hidden 64,64
```
`load_dataset` detects `damage_p99` → trains the head on `D` (else legacy margin). Prints the
data-driven `trust region (p99)` and saves it inside the checkpoint's `stats["domain"]`.

## 3. Use it — the v2 config is already staged

`data/configs/closed/rect_a4_beam_surrogate_v2.yaml` points at `hinge_surrogate_v2.pkl`,
`thickness_mm: 1.5`, and relies on the auto-loaded `DOMAIN`. Once the checkpoint exists:

```bash
JAX_PLATFORMS=cpu conda run -n kgnn_mac \
  python -m nff.scripts.run_closed --config-dir closed --config-name rect_a4_beam_surrogate_v2 --every 100
```
Then retune `loads.value` / `target_half_*` to explore the now-larger feasible deployment range.

---

## Caveats
- **Large-angle arc accuracy:** `ccx_solver` applies the rotation as a straight-line ramp per
  increment (docstring TODO: multi-step arc for large-angle production). At 90°/30 steps that's
  ~3°/step, but sanity-check the arc error before trusting ≥90° data; drop to `--angle 75` if off.
- **Compute:** `--fracture-margin 2.5` = deeper plastic solves (slower); `--timeout 600` guards
  stalls, and partial data up to the kill is still parsed.
