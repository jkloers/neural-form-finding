# Handoff: the PET hinge surrogate now drives the closed pipeline, and it extrapolates

**Date:** 2026-08-01 · **Branch:** `hinge-quarantine-and-oracle-fixes` · **Audience:** the next agent

This is a status + investigation brief. Part 1 is settled fact with numbers. Part 2 is the open
problem and is where the work should go. Part 3 lists traps, including four things I asserted and
then had to retract — read it before repeating them.

---

## 0. One-paragraph summary

The PET surrogate was retrained successfully (held-out energy **2.65% val / 1.70% test**), its
architecture was swept and settled for the first time, and it now **drives the closed `closed_les`
pipeline end to end** — 200 epochs converge to a **9.04 mm** mean boundary match. But the deployment
rides states that are **~6× (median) to 28× (p90) further from training data than a held-out
validation row**, so the physics it reports there is extrapolation, not interpolation. The binding
constraint is no longer the model or the training procedure. **It is the FEA campaign's sampling
envelope.** That is the thing to investigate.

---

## 1. Settled — do not re-derive

### 1.1 The surrogate trains well, and why it used to not

`W` is a **path functional** for an elastoplastic oracle, so on a campaign mixing prescribed
`origin_ray` jobs with solver-chosen `free_path` jobs the regression target is **multi-valued**.
Filtering to `origin_ray` (`--route-family`) took held-out energy from **22.6% → 2.7%** with no
model change. See `memory/project_route_family_multivaluedness.md`. The internal-variable rescue
`W(u,z;g)` was tested and **rejected** (`memory/project_internal_variable_infeasible.md`).

**Shipping checkpoint:** `data/surrogates/hinge_surrogate_pet_v3.pkl` — 3×256 trunk, `m_out` 128,
`origin_ray`, `exclude_unconverged both`, `feat_kind eta`, lam 0.5, 26 009 updates.
energy **2.65% val / 1.70% test**, force 11.93/14.00% (per-component), damage 12.15%.

### 1.2 Architecture — swept, and the answer is "keep it big"

`nff/scripts/diagnostics/sweep_surrogate_capacity.py` (new). 10 arms budget-matched via
`--min-updates`, plus a 4-arm × 5-seed study.

| arm | energy params | hvp | test E | test F (5-seed) |
|---|---|---|---|---|
| 32×32·32 | 2 304 | 43 µs | 2.32 ± 0.30% | 17.05 ± **3.99**% |
| 64×64·32 | 6 624 | 52 µs | 2.27 ± 0.39% | 16.88 ± **3.06**% |
| 128×128·64 | 25 536 | 135 µs | 2.05 ± 0.34% | 15.60 ± **2.61**% |
| **256×256×256·128** | 166 016 | 834 µs | **1.82 ± 0.16%** | **13.36 ± 0.92%** |

- ★ **Every arm from 1 776 params up clears the 5% energy target.** Capacity never bound the VALUE.
- ★ It binds the **GRADIENT** and above all the gradient's **VARIANCE** (force sd 3.99 → 0.92, 4×).
  Force is `∂W/∂u` = the Stage-2 equilibrium residual, so reliability there matters more than the
  mean. Cohen's *d* ≈ 1.5–1.7 smallest-vs-largest, reproduced across 5 seeds.
- ★ **Measured pipeline cost makes the size question moot at this scale**: 0.038 vs 0.352 s/epoch →
  **34 s vs 102 s** for a 200-epoch run. ⚠ Only true at 12 hinges; surrogate cost scales with
  `n_hinges` while much L-BFGS overhead does not. Re-run the sweep before scaling the tessellation.
- **The energy and damage MLPs are DISJOINT.** Energy/force are bit-identical across `--fail-hidden`
  arms; sweep the damage head separately (`--stage fail`). Measured: `f64x64` D 12.15%,
  `f128x128` 11.71%, **`f256x256` 10.46%**.

Details: `memory/project_surrogate_architecture.md`. Figure:
`nff/scripts/figures/plot_surrogate_capacity.py`.

### 1.3 The pipeline runs

**`data/configs/closed/sheet_4x8ft_pet.yaml`** (new) — forked from `sheet_4x8ft_rom.yaml`, NOT from
the steel `sheet_4x8ft.yaml`, so the two are a controlled A/B of the hinge model alone. Before this,
all 59 `type: surrogate` configs still pointed at the dead S235 checkpoint.

```bash
JAX_PLATFORMS=cpu conda run -n kgnn_mac \
  python nff/scripts/run_closed.py --config-name sheet_4x8ft_pet --every 20
```

Run `data/outputs/runs/run_20260801_195025_sheet_4x8ft_pet`: chamfer precision 0.0297 → 0.000714
(42×), grad_norm 1.62 → 0.021, full artifact set, finite throughout.

**A/B against the ROM twin, same target, same load, 200 epochs:**

| | boundary mean | max | deployment angle |
|---|---|---|---|
| ROM (`sheet_4x8ft_rom`) | **5.46 mm** | 14.06 mm | 45.36° [43.79, 45.97] |
| Surrogate (`sheet_4x8ft_pet`) | 9.04 mm | 17.56 mm | 54.39° [48.63, **64.73**] |

The surrogate matches the target *worse* and **over-opens by ~9°** with a 16° hinge spread against
the ROM's 2°. A plausible reading — UNVERIFIED, and a good first thing to test — is that the
surrogate is too soft in the compressive/high-η region because that region is extrapolated (§2).

---

## 2. ★ THE OPEN PROBLEM — coverage, not accuracy

### 2.1 The measurement that matters

Box-exceedance against `stats["domain"]` is a **weak** diagnostic: that box is only the p1/p99
axis-aligned marginal hull of the training rows, and being inside it does not mean being near data
(the ridden set is a thin curved sheet, not a filled box). **Use nearest-neighbour distance in the
net's whitened feature space instead**, with held-out rows as the yardstick for healthy
interpolation. **Implemented** in `nff/scripts/diagnostics/score_deployment_ood.py` — pass `--data`
to enable it (it rebuilds the checkpoint's own train/held-out split so the yardstick is the data the
net was actually validated on).

| query set | NN p50 | NN p90 | % beyond held-out p90 |
|---|---|---|---|
| held-out val rows (**the yardstick**) | 0.345 | 0.688 | — |
| deployment, trained, vs `origin_ray` train | **2.00** (5.8×) | **19.0** (28×) | **83.7%** |
| deployment, trained, vs **ALL** rows (incl. `free_path`) | 1.75 | **8.65** | 76.0% |
| deployment, untrained, vs `origin_ray` train | 1.63 | 4.98 | 74.2% |

Units are training feature-std. ⚠ The "ALL rows" reference set includes val rows, so that row is
slightly optimistic; the `origin_ray` train comparison is the clean one.

**Two conclusions, both load-bearing:**

1. **The route-family filter costs real coverage.** Including `free_path` cuts the p90 gap
   19.0 → 8.65 (2.2×), because `free_path` is exactly where the compressive states live
   (`eta_a` p1..p99 −0.670…+0.016 vs `origin_ray`'s −0.145…+0.242). We bought accuracy by
   discarding coverage the campaign had deliberately built. **This is a genuine tension, not a bug.**
2. **Even the full campaign does not cover the deployment.** 76% of states remain beyond the
   interpolation scale. So recovering `free_path` helps materially and does **not** solve it.

Supporting numbers (box metric, for continuity with older notes): trained deployment **36.4%**
in-box vs untrained **45.5%** — training moves *away* from the data. η_a reaches **−1.76** (axial
closing of 1.76× the ligament width) against an oracle that sampled η_a ≥ 0 only; 12/12 hinges go
compressive over 47.4% of path samples.

### 2.2 Leads, roughly in order of expected value

1. **Re-aim the FEA campaign at the ridden manifold.** This is the main lead. The tooling to measure
   the manifold already exists (`nff/closed/hinge_paths.py`, `hinge_path_ensemble.py`,
   `sampling_spec()` emits `sample_jobs` kwargs verbatim). ⚠ `memory/project_path_harvest_dataset.md`
   says the harvested priors are **STALE** (harvested at the wrong contact angle) — re-harvest first.
   The specific gap is **compressive `eta_a`**, which `dataset.py::sample_jobs` does not sample.
2. **Can the compressive region be sampled as `origin_ray`?** This is the crux question and I do not
   know the answer. `free_path` is unlearnable because solver-chosen curves cross the ray foliation
   — but a *prescribed* ray into the compressive quadrant is still an origin ray and should stay
   single-valued. If yes, the accuracy-vs-coverage tension largely dissolves. **Test this before
   anything expensive.**
3. **Make `_term_ood` path-aggregated.** `nff/training/loss.py::_term_ood` returns `sum(ood[-1])` —
   the FINAL load step only — while `_term_damage` uses path-max and `_term_compression` path-min,
   and both docstrings argue explicitly why the endpoint is insufficient. The optimizer exploits the
   gap: `ood_raw` fell 178 → 8.81 over training while path exposure got *worse*. Fix candidate: reuse
   the existing `_softmax_max`. ⚠ `loss.py` is on the propose-a-plan list in `CLAUDE.md` — propose
   before editing.
4. ★★ **`loss_weights.compression` — MEASURED, and it is the single most effective lever found.**
   Path-min on `eta_a`; once the trust box is itself compressive (`eta_a_min = −0.14`) the OOD
   barrier stops resisting compression, so this is the only handle. Configs
   `data/configs/closed/pet_comp{01,10}.yaml`.

   | run | boundary mean | hinge spread | in-box | NN p50 | NN p90 | NN max |
   |---|---|---|---|---|---|---|
   | ROM baseline | **5.46 mm** | 2.18° | — | — | — | — |
   | compression 0.0 | 9.04 mm | 16.10° | 36.4% | 2.00 | 19.02 | 28.44 |
   | compression 0.1 | **6.73 mm** | 10.31° | 42.1% | 1.95 | 10.28 | 20.16 |
   | compression 1.0 | 7.67 mm | **1.84°** | **46.7%** | **1.31** | **3.25** | 8.15 |

   ★ **There is no trade-off between coverage and accuracy here** — I expected one and was wrong.
   Compression 1.0 cuts the extrapolation p90 **5.9×** (19.0 → 3.25), nearly matches the ROM's
   hinge-angle discipline (1.84° vs 2.18°), AND beats the unconstrained surrogate on boundary match.
   Interpretation (plausible, still UNVERIFIED): the surrogate is too soft in the compressive region
   it never sampled, the optimizer exploits that fictitious compliance, and forbidding it recovers
   both physical plausibility and accuracy. **Testing this interpretation directly — e.g. compare
   surrogate vs a compressive FEA point — is a high-value, cheap next step.**

   ⚠ Note the `%` beyond-yardstick column barely moves (83.7 → 78.7%) while the MAGNITUDE collapses.
   With a yardstick p90 of 0.688 and a deployment median of 1.31, most states exceed it by
   construction. **Report the magnitude (p50/p90), not the fraction** — the fraction saturates.

5. **Raise `hinge_model.barrier`** (currently 0.05). Cheaper than 3–4 but blunter: it shapes the
   Stage-2 solve rather than the design.

### 2.3 What would count as success

The deployment's NN-distance p90 falling to within ~2–3× the held-out p90 (i.e. ≲2 in the table
above), *without* the boundary match degrading past the ROM's 5.46 mm. Report both or neither.

**Current best: p90 3.25 (4.7× the yardstick) at boundary 7.67 mm** — `compression: 1.0`. Down from
19.0 (27.6×) unconstrained. So most of the gap closed with a loss weight, and the remaining factor of
~2 is what a re-aimed campaign (lead 1) would have to buy. A weight between 0.1 and 1.0 is unexplored
and may sit better on both axes.

---

## 3. ⚠ Traps — four things I asserted and had to retract

1. **"`energy_scale` must be re-derived when swapping checkpoints."** WRONG. It is the pure unit
   conversion `1/length_scale` that makes `loads.value` newtons (`sheet_4x8ft_rom.yaml:156`), and it
   is checkpoint-independent. The PET net's `W_scale` 52.6 vs steel's 1814.7 (34.5×) is **real
   physics** — plastic hinge moment ∝ σ_y·t², giving (235/48.8)·(1.5/0.5)² ≈ 43×.
2. **"The load is mis-calibrated; the surrogate over-drives it."** WRONG.
   `calibrate_target.py --angle 45` on the surrogate gives **55.85 N** vs the ROM's 55.33 N — within
   0.9%. At the *untrained* design the two hinge models are equally stiff. The over-opening is the
   optimizer moving the design. **Never read a trained run's deployment angle as evidence about load
   calibration.**
3. **"`ood_raw` fell 20×, so the optimizer pulled the design into the trust region."** WRONG, and it
   is the most instructive error here — see §2.2 lead 3. The endpoint-only term fell while the path
   got worse. **Always check the path, not the endpoint.**
4. **"The single-seed force gap is a lucky test split."** WRONG. The 5-seed study reproduced it at
   *d* ≈ 1.6. My reasoning (best on test, only 2nd on val ⇒ noise) was plausible and wrong; the
   seed study is what settles these, not inference.

Two more, inherited:

5. `nff/scripts/diagnostics/extract_hinge_paths.py` reports in-domain against the **hard-coded
   fallback `DOMAIN`** (η_a 0…1, θ ≤ 37.8°), not the loaded checkpoint's `stats["domain"]`. Its
   "3.7% in domain" is not the checkpoint's number. Worth fixing.
6. `memory/project_surrogate_pet_retrain.md` used to conclude "5% is not reachable by retraining this
   dataset". Falsified — 1.70% test on the *same* dataset. The file is corrected but its
   inert-levers table is still valid and is now *explained* by multi-valuedness.

---

## 4. Reproduction

```bash
# capacity sweep (10 arms, ~25 min) and the seed study (20 runs, ~35 min)
JAX_PLATFORMS=cpu conda run -n kgnn_mac python -m nff.scripts.diagnostics.sweep_surrogate_capacity
JAX_PLATFORMS=cpu conda run -n kgnn_mac python -m nff.scripts.diagnostics.sweep_surrogate_capacity \
    --arms h64x64_m32,h256x256x256_m128 --seeds 0,1,2,3,4 --out-dir data/surrogates/capacity_seeds
# damage head, on the winning trunk
... --stage fail --hidden 256,256,256 --m-out 128 --w-damage 0.5 --out-dir data/surrogates/capacity_fail

# deployment + its error bar
JAX_PLATFORMS=cpu conda run -n kgnn_mac python nff/scripts/run_closed.py --config-name sheet_4x8ft_pet
JAX_PLATFORMS=cpu PYTHONPATH=$(pwd) conda run -n kgnn_mac python \
    nff/scripts/diagnostics/extract_hinge_paths.py --config data/configs/closed/sheet_4x8ft_pet.yaml \
    --params <run_dir>/best_params.pkl --steps 40 --out data/outputs/hinge_paths/pet_trained_deploy
JAX_PLATFORMS=cpu conda run -n kgnn_mac python -m nff.scripts.diagnostics.score_deployment_ood \
    --paths data/outputs/hinge_paths/pet_trained_deploy/hinge_paths \
    --surrogate data/surrogates/hinge_surrogate_pet_v3
```

New files this session: `sweep_surrogate_capacity.py`, `score_deployment_ood.py`,
`plot_surrogate_capacity.py`, `sheet_4x8ft_pet.yaml`, `hinge_surrogate_pet_v3.pkl`,
`memory/project_surrogate_architecture.md`.

**State at handoff.** All 281 tests pass. `hinge_surrogate_pet_v3.pkl` is final: 3×256·128 trunk +
`f256x256` damage head, energy **2.65% val / 1.70% test**, force 11.93/14.00%, damage 10.46/7.05%,
float64, loads with no provenance warning. `sheet_4x8ft_pet.yaml` defaults to `compression: 1.0` with
the measured sweep in its comments. The throwaway `pet_comp{01,10}.yaml` variants were deleted (the
table lives in the main config).

⚠ **`data/*` is gitignored** and the other closed configs are force-added, so
`data/configs/closed/sheet_4x8ft_pet.yaml` and `data/surrogates/hinge_surrogate_pet_v3.pkl` are
currently UNTRACKED. They need `git add -f` to match convention, or they will be lost. Nothing was
committed this session.

**Damage-head caveat:** val and test disagree (`f64x64` has the best TEST damage at 6.91% but nearly
the worst val at 12.15%; `f256x256` is best on val at 10.46%). Selection used val, which is correct,
and it independently reproduces an earlier session's 10.43%. But with one seed per arm and a 1.25 pt
spread, treat the damage-head ordering as suggestive, not established — the energy seed study showed
this dataset needs multiple seeds to separate 0.5 pt effects.
