# Hinge Condensation Surrogate — Implementation Plan

> Companion to `docs/hinge_surrogate_condensation.md` (the WHAT/WHY brief). This is the
> HOW: phases, deliverables, decision gates, file/environment touch-points. Approval is
> required before Phase 3 (touches `nff/stages/physics/energy.py`, a plan-gated file).

## Status (2026-07-03)

- **Phase 0 — DONE**, engine = **CalculiX**, not FEniCSx (FEniCSx abandoned: hand-coded J2
  intractable). Oracle = `nff/rve/{geometry,ccx_solver,hinge_function}.py`. Fillet `c=0.16`,
  `n_through=2`, stop-at-fracture.
- **Phase 1 (data) — DONE.** 1000-hinge campaign (`nff/rve/dataset.py`,
  `nff/scripts/generate_hinge_dataset.py`): 1000/1000 usable, 0 errored, 63,644 Sobolev samples
  (`data/fea/hinge_dataset.npz`).
- **Phase 2 (surrogate) — DONE.** Squared-form energy `nff/models/hinge_surrogate.py`, trained
  (`nff/scripts/train_hinge_surrogate.py`): held-out (150 unseen geometries) **energy_rel 1.8%,
  force_rel 5.9%, fail_rmse 2.5%** (λ=0.9). Params `data/surrogates/hinge_surrogate.pkl`.
- **Phase 3 (integration) — NEXT (this doc's "Known integration gaps").**

## Scientific figures — ongoing workstream

Publication-quality figures are a first-class, recurring deliverable of this program (not an
afterthought). Every figure:
- uses the **project charter** (palette from `nff/scripts/diagnostics/validate_hinge.py` —
  orange `#F58025` / teal `#2A9D8F` / red `#D62828` / grey `#6C757D` / ink `#1A1A1A`; clean 1×N;
  recessive axes; **never** numeric values in titles; no hardcoded fonts),
- follows the **dataviz skill** method (pick the form; assign color by job; **run the palette
  validator**; thin marks; legend for ≥2 series),
- is **rendered and eyeballed, then iterated** (open the PNG, fix collisions / scale / geometry —
  do not ship the first render),
- lives in `nff/scripts/figures/plot_*.py`, modular (separable fit / bin / plot helpers).
- Reference implementation: `nff/scripts/figures/plot_hinge_energy_regimes.py`.

## Environment split (no import coupling)

- **FEniCSx process** (new, standalone — like the old SOFA process): builds one hinge RVE,
  runs the 3D elasto-plastic FEM, writes labeled data to disk. Its own conda/venv.
- **kgnn_mac (JAX/Flax)**: the analytic descriptor map, the surrogate training, and the
  pipeline integration. Reads the on-disk dataset; never imports FEniCS.
- Data crosses the boundary as files (e.g. parquet/npz), exactly as SOFA outputs did.

---

## Phase 0 — Foundations & de-risking (ONE hinge; decides the architecture)

Goal: produce the three decision numbers (brief §11) and the two reusable core objects
(descriptor map + RVE), before any campaign.

**0.1 Analytic descriptor map** — `descriptor = f(r, x_bound, manuf_params)` in JAX.
- In-plane part from `closed_builder_jax.solve_cut_vertices_jax` → per-hinge
  `(L_main, L_sec, α, g)` as distances/angles between solved cut vertices.
- Manufacturing part: `(w_c, t, ρ)` injected as global/per-cut params.
- Emit the dimensionless descriptor (brief §5) + a **crowding/validity flag** (min
  inter-pivot distance vs. window budget) — analytic, no meshing.
- **This is also the on-the-fly inference evaluator** used at integration (Phase 3).
- Deliverable: `nff/topology/hinge_descriptor.py` (+ unit tests vs. hand-computed hinges).

**0.2 Standalone single-hinge RVE builder** (FEniCS side) — `descriptor → meshed domain`.
- Two rigid tile-side patches + the ligament between; boundary placed per Saint-Venant
  (brief §8). Define the **shared kinematic-driving map** `(a,s,θ) → rigid patch transform`
  (same definition used online in Phase 3).
- Deliverable: `fenics/hinge_rve/builder.py`, `fenics/hinge_rve/kinematics.py`.

**0.3 RVE solver** (FEniCS) — 3D geometric-nonlinear **J2 elasto-plastic** static solve.
- Impose a monotonic `(a,s,θ)` ray; `z` free; harvest per step
  `(ū, W [total work potential], ∇W [reaction on driven patch], σ_vM/σ_y, ε_p/ε_f)`.
- Energy and reaction are **native** in the variational form (the reason we left SOFA).
- Deliverable: `fenics/hinge_rve/solve.py` + a writer to the on-disk schema.

**0.4 The three probes** (on one representative steel hinge):
1. **Mesh/element gate** — RVE out-of-plane bending stiffness vs. analytical thin-plate.
   Decides solid-vs-shell element + resolution. (Pass criterion: within ~5–10%.)
2. **Yield + failure sweep** — `σ_vM/σ_y`, `ε_p/ε_f` vs. deployment → elastic-domain size,
   confirm plastic model is exercised, locate the failure boundary.
3. **Smoothness / branch check** — sweep one DOF; plot `W(ū)` + z-amplitude → confirm
   smooth/single-valued (no snap); decide whether an imperfection seed is needed.

**0.5 Frame-invariance test** — pure rigid-body `(a,s,θ)` ⇒ `W = 0` (guards the shared map).

**Gate out of Phase 0:** element type fixed; elastic-vs-plastic domain understood; seeding
decision made; descriptor map validated. Only then start the campaign.

---

## Phase 1 — Data campaign

**1.1 Descriptor sampling** — LHS over the dimensionless descriptor ranges (brief §5), aimed
at **maximal coverage** (user goal: most complete hinge-geometry dataset). Sanity-bound the
ranges by running the Phase-0 analytic map on a handful of sample sheets so the box brackets
what real `{r, x_bound}` produce. Pre-filter crowded/invalid descriptors analytically.

**1.2 Kinematic sampling** — a fan of **monotonic proportional rays** in `(a,s,θ)` around the
dominant opening mode; each run to **full deployment / failure**, all steps recorded.

**1.3 Execution** — embarrassingly parallel FEniCS runs (one process per descriptor). Robust
per-run try/except; write partial paths. Storage schema:
`(descriptor[9], ū[3], W, ∇W[3], σ_vM/σ_y, ε_p/ε_f, regime_flag)` per load step.

**1.4 Descriptor-completeness test** — for a few real harvested hinges, compare standalone-RVE
`W` to a windowed full-sheet `W`. Mismatch ⇒ a missing descriptor parameter.

**Deliverable:** on-disk dataset + a loader; a coverage/quality report.

---

## Phase 2 — Surrogate model (JAX/Flax)

**2.1 Architecture** — `W_θ = W_baseline(ū; k(geom)) + W_NN(ū, geom)`; `W_NN`
Taylor-residualized at the origin (exact elastic small-strain limit); **C² smooth
activations** (softplus/tanh). Inputs: 3 dimensionless kinematics + descriptor. Outputs:
`W` and a **validity/failure margin** head.

**2.2 Training** — Sobolev loss, variance-normalized:
`L = ‖W−W*‖²/σ_W² + ‖∇_ū W − F*‖²/σ_F² + λ·validity_loss`, with `F_pred = jax.grad(W_fn, kin)`
vmapped. Optional 2nd-order term if the FEM emits tangent stiffness.

**2.3 Validation** — held-out hinges: energy/force error, validity-domain accuracy,
extrapolation behavior (must fall back to baseline OOD). Plots per viz preferences.

**Deliverable:** `nff/models/hinge_surrogate.py` + trained params + a pure
`hinge_energy_surrogate_fn(ū, descriptor, params) -> W`.

---

## Phase 3 — Pipeline integration  ⚠ REQUIRES APPROVAL (touches `energy.py`)

**3.1 Wire the analytic descriptor map** into the closed pipeline: per-hinge descriptor from
the live `{r, x_bound}` + manufacturing params, precomputed at the JIT boundary where static.

**3.2 Replace `ligament_energy_linearized`** — swap its last line for `W_θ` (residual form),
preserving the `ligament_strains` interface. Use the **shared kinematic map** so labeling and
inference are identical.

**3.3 Global-solver stability** — LM damping on the backward IFT solve (per
`project_stage2_conditioning`), keep load stepping (`lax.scan`), add the **validity barrier**
to the training loss to hold hinges in the trustworthy domain.

**3.4 End-to-end** — retrain a `closed_les` deployment through the surrogate; compare deployed
shape / energy balance against the linear-spring baseline; verify stability and validity
coverage.

**Deliverable:** surrogate-backed Stage-2 for `closed_les`, behind a config flag so the
spring baseline stays available.

### Known integration gaps (surfaced by the 2026-07-02 compatibility trace)

The interface fits: bond energies are evaluated batched, so per-hinge `(alpha, w_lig)` thread
in by adding two `(n_hinges,)` fields to `LigamentParams` + populating them in
`build_control_params` (both `params.py`, not gated); `alpha` comes from
`compute_hinge_descriptors` (differentiable in `r`), `w_lig` is a manufacturing constant; the
only gated edit is routing `build_potential_energy` to `ligament_energy_surrogate`. Two gaps
must be closed at integration — **neither corrupts the dataset, both are recoverable, so
nothing needs fixing before the campaign** — but they are logged here so they are not
rediscovered:

- **Gap 1 — point-ROM ↔ physical-hinge frame bridge.** The ROM is point-based: for closed
  hinges `build_reference_bond_vectors` collapses to `~[1e-4, 0]` (`l0≈0`), so
  `ligament_strains` splits `(a, s)` along an arbitrary (global-x fallback) axis. The RVE
  dataset defines `(a, s)` in a hinge-local frame tied to the cut (pivot at the main-cut tip,
  `a` along the secondary cut, `s` perpendicular, `θ = dRot`). `θ` and the coupling are
  frame-free; the `(a, s)` *decomposition* is not. Fix: give each closed hinge a
  `reference_bond_vector` carrying the **cut orientation** (`main_vec` from the descriptor ×
  nominal length) so `ligament_strains` resolves `(a, s)` in the same frame the RVE imposed.
  This IS the bridge from the finite physical hinge to the point ROM.
  `build_reference_bond_vectors` is in `geometry.py` (not gated).
  - REQUIRED NOW (documentation only, not a code fix): the dataset must record its exact
    kinematic convention (pivot = main-cut tip; `a` along the secondary cut; `s` perpendicular;
    `θ` = relative rotation; frame set by the secondary-cut orientation) so the bridge is
    buildable later. Captured in `hinge_function.py`'s module docstring.

- **Gap 2 — length + energy scale.** The surrogate returns physical N·mm at physical mm; the
  closed pipeline runs in abstract units (target radius 2.5–5, `k=0.5/1000`, `W~0.25`).
  Integration needs (i) a **length scale** [mm per pipeline-unit] — pick a physical tile size so
  `w_lig ≈ 1/10` tile — to map the ROM's abstract `dU` into mm, and (ii) an **energy scale** so
  physical `W` neither swamps nor vanishes against the chamfer loss. Both live in
  setup/config, not the gated core.

---

## Decision gates (must clear before advancing)

| Gate | After | Question answered |
|---|---|---|
| G0 | 0.4 probe 1 | solid vs shell element; resolution |
| G1 | 0.4 probe 2 | elastic-only vs plastic domain reach; failure boundary |
| G2 | 0.4 probe 3 | imperfection seeding needed? |
| G3 | 1.4 | is the descriptor complete? |
| G4 | 2.3 | does `W_θ` meet energy/force/validity accuracy + graceful OOD? |
| G5 | 3.4 | does the surrogate-backed solve match/beat the baseline & stay stable? |

## Open inputs still needed (brief §13)

- Steel: grade → `E, ν, σ_y`, hardening law, `ε_fracture`; sheet thickness/gauge range.
- Deployment magnitude target (bounds the kinematic ray extent).
- Manufacturing params: cut kerf `w_c`, fillet `ρ` ranges (new DOF, not in `{r, x_bound}`).
