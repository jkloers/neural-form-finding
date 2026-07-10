# Hinge surrogate — validation & dataset plan (CalculiX era)

Status after the 2026-07-02 tool pivot: **CalculiX is the physics oracle** (built-in
finite-strain von-Mises plasticity + buckling; we write no constitutive code). This
document is the plan to (A) validate that the oracle measures the *right* physics, then
(B) generate the dataset. **No dataset is generated until the validation suite passes.**

---

## 1. Re-anchor — what we are building

Replace the analytic linear-spring hinge energy
`nff/stages/physics/energy.py::ligament_energy_linearized` (½k_a a² + ½k_s s² + ½k_θ θ²)
with a **learned condensed energy** `W(a, s, θ; geometry)`. The global closed-kirigami
model stays rigid in-plane tiles (3 DOF/face); the surrogate carries the **local
out-of-plane ligament physics** the linear springs miss. One CalculiX deployment per
(geometry, kinematic ray) yields `W` **and its gradient** along the whole path.

---

## 2. What each simulation measures (per load increment = per kinematic state)

| Quantity | Source in CalculiX | Role |
|---|---|---|
| **Stored energy `W`** | `*EL PRINT, TOTALS=ONLY, ELSE` (.dat) | primary label |
| **Generalized forces `∇W = (F_a, F_s, M_θ)`** | reaction forces/moment on the driven arc (`RF`) | **Sobolev labels, free** |
| **Out-of-plane `u_z,max`** | `U` field (.frd) | buckling amplitude / flag |
| **Equivalent plastic strain `PEEQ`, von-Mises σ** | `PEEQ`, `S` (.frd) | plasticity, yield check |
| **Failure margin `ε_p/ε_f`** | max equiv. strain vs ε_f≈0.25 | validity: elastic/plastic/**failed** |

**On the forces idea (you raised it — it's exactly right and important).** By the
**envelope theorem**, for a state reached by prescribed kinematics the reaction generalized
force equals `∂W/∂(kinematic DOF)`. So the reaction forces/moment on the driven face
**are** `∇W` — obtained from the *same* run at no extra cost. Training on `(W, ∇W)`
(Sobolev / gradient-enhanced regression) gives far more signal per sim and a smoother,
physically-consistent surrogate. This is already the locked plan; CalculiX gives us the
labels directly.

---

## 3. Dataset representation

**Inputs**
- *Geometry* (dimensionless descriptor): `L_sec/L_main, α, w_lig/L_main, w_c/L_main,
  t/L_main, ρ/L_main` + material `ν, σ_y/E`. Nondimensional ⇒ scale collapses to `t/L`.
- *Kinematics*: relative in-plane rigid motion of tile B vs A = **(a, s, θ)** (axial,
  shear, rotation) — the 3 DOF the global model needs (so the surrogate must be `W(a,s,θ)`,
  not `W(θ)` alone; see Open Decision O1).

**Labels (per sample = per increment on a ray)**
- `W`, `∇W=(F_a,F_s,M_θ)`, `u_z,max`, `ε_p/ε_f`, validity flag.

**Sampling**
- LHS over the descriptor space (crowding-filtered, OOD-flagged).
- Per geometry: a **fan of monotonic proportional kinematic rays** `(a,s,θ)=λ·dir`, each
  run to failure; **every converged increment is a sample** (dense `W`, `∇W` coverage).

**Surrogate** (from memory): `W = W_baseline + W_NN` (residualized at origin), C²
activations, Sobolev loss = energy + force + validity-margin, variance-normalized.

---

## 4. Validation suite — the critical gate (run on 1–3 representative hinges, BEFORE any dataset)

Each item is a **visual + a quantitative check** answering "are we measuring the right
physics?" If any fails, we fix before generating data.

**A · Geometry & mesh**
- **A1** Render the RVE prism mesh — neck refinement, thickness layers, `rigid_A/rigid_B`
  tags. Confirm the meshed region is the intended ligament + Saint-Venant window.
- **A2** Mesh convergence: `W(θ)` vs `lc_min` and `n_through`. Pick the coarsest mesh
  within ~2–3 % of converged (this sets the speed budget).

**B · Kinematics**
- **B1** Initial (flat) vs final (deployed) 3-D state: tiles rotate rigidly about the
  pivot, faces stay coplanar, only the ligament deforms.
- **B2** Arc-path check: single-step straight-line ramp vs multi-step cumulative arc path
  — does `W(θ)` differ at large θ? If yes, switch the solver to the multi-step path.
- **B3** Pivot location: sweep along the primary-cut axis; confirm the energy-minimising
  pivot (near the tip) and quantify its effect on `W`.

**C · Out-of-plane buckling (the entire reason the surrogate exists)**
- **C1** 3-D `u_z` field at several θ + a fold animation — crest on the compression side,
  faces flat. This is the headline visual.
- **C2** Buckling engaged & beneficial: `W(θ)` with vs without out-of-plane freedom
  (imperfection on/off) — confirm the buckled path is **lower energy** (the condensation
  premise) and that the imperfection seeds the *ligament* mode, not a spurious tile tilt.
- **C3** Imperfection insensitivity: `W(θ)` vs imperfection amplitude — post-buckling
  result should be ~independent of the seed.

**D · Energy & forces**
- **D1** `W(θ)`: elastic `θ²` at small θ, then the plastic knee. Order-of-magnitude sanity
  vs analytic bending energy of the ligament.
- **D2** **Envelope-theorem check (critical for Sobolev labels)**: reaction moment `M_θ`
  vs finite-difference `dW/dθ` — they must agree. Validates every force label.
- **D3** Energy balance: CalculiX internal energy = external work (reaction·displacement).

**E · Plasticity & failure**
- **E1** von-Mises field capped at ~σ_y(+hardening); `PEEQ` localizes at the neck.
  Confirm plasticity actually engages (not spuriously elastic).
- **E2** Failure: max equivalent strain vs ε_f; the θ at which it would tear → validity
  colouring of the `W(θ)` curve.
- **E3** Path check: confirm the imposed ray is monotonic/proportional (deformation-theory
  / single-valued `W` assumption holds).

**F · Robustness**
- **F1** Determinism: same input → identical `W` (CalculiX is deterministic).
- **F2** A spread of descriptors across the design box run to completion without solver
  failure (pipeline robustness before committing to the campaign).

---

## 5. Interface & campaign (built only after §4 passes)

- `deploy(descriptor, kinematic_ray) -> {W[:], forces[:], u_z_max[:], strain_margin[:],
  validity[:]}` — wraps geometry → mesh → deck → ccx → parse (energy from `.dat`,
  fields from `.frd`).
- Parallel batch runner: independent `ccx` jobs across cores (N cores ⇒ N× throughput).
- Dataset writer: one row per sample (inputs + labels + flags), plus provenance.
- Speed budget from A2; target ≈ tens of seconds/run × parallelism.

---

## 6. Open decisions to resolve during validation

- **O1 — kinematic dimensionality.** The global model needs `W(a,s,θ)` (3 DOF). But if the
  translation `(a,s)` is effectively set by energy minimisation for a given deployment (the
  "pivot minimises energy" observation), the *deployment path* is ~1-parameter even though
  the *surrogate* stays 3-DOF. Decide the sampling design after B2/B3/D2.
- **O2 — arc-path multi-step** (B2): implement if it changes `W` materially.
- **O3 — mesh/speed budget** (A2).
- **O4 — descriptor ranges + crowding filter** for the LHS box.

---

## 7. This session vs next

- **This session (done):** purged all FEniCSx/hand-coded-physics code; CalculiX is the sole
  oracle; wrote this plan; material hypotheses in `docs/hinge_material_hypotheses.md`.
- **Next session:** implement the validation suite (§4) on 1–3 hinges, produce the visuals,
  confirm the physics, resolve O1–O4. **Only then** build the campaign (§5) and generate data.
