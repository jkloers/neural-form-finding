# Hinge Condensation Surrogate — Design Brief

> Status: **design / pre-implementation**. This document captures WHAT we are building
> and WHY, precisely enough to restart the conversation from scratch. It is deliberately
> not an implementation plan (step ordering comes later, once the de-risking probes fix
> the open decisions).

---

## 1. Purpose (one sentence)

Replace the analytic linear-spring hinge energy in the closed-kirigami Stage-2 physics
with a **learned, differentiable, condensed energy surrogate** `W_θ` that captures the
local **out-of-plane deformation, plasticity, and failure** of the material ligament
between cuts — while the global model keeps **rigid, in-plane tiles** (3 DOF/face).

This is **kinematic condensation / nonlinear homogenization**: 3D micro-physics learned
offline, exposed to the global 2D solver as a smooth scalar energy.

---

## 2. Context — where this plugs in

- **Closed framework (`closed_les`):** a flat cut sheet is deployed by the Stage-2
  physics solver to match a target shape. Structure = rigid tiles + hinges.
- **Current hinge model** — `nff/stages/physics/energy.py::ligament_energy_linearized`:
  `W = ½·k_stretch·(a·l0)² + ½·k_shear·(s·l0)² + ½·k_rot·θ²`.
  `ligament_strains` already produces the **3 corotational, frame-invariant objective
  strains** `(axial a, shear s, dRot θ)`.
- **Important:** hinges are currently idealized **point pivots**; the material ligament
  strip is NOT represented anywhere. We are introducing a genuinely new physical object
  — its parametrization is ours to define.
- **Integration target:** *full replacement* of the quadratic with `W_θ(a, s, θ; geom_local)`.
  The 3 strains are already frame-invariant, so there is no new corotational work; the
  swap is at the last line of `ligament_energy_linearized`.

---

## 3. Physical model

- **Condensation:** `W_2D(ū) = min_z W_3D(ū, z)`, where `ū = (a·l0, s·l0, θ)` are the 3
  in-plane relative rigid DOF of two adjacent tiles and `z` is all out-of-plane micro-DOF,
  relaxed freely. The global solver never sees `z`.
- **Envelope theorem (Danskin):** at the relaxed configuration, the reaction on the driven
  boundary equals `∇_ū W_2D`. So **energy and force both come from one sim** — no need to
  differentiate through the relaxation. This gives free Sobolev (force) labels.
- **Regime — DEPLOYMENT ONLY:** opening / tension / shear / moderate rotation. No
  compression (that would need a deployed reference state we do not have). Loading is
  **monotonic**.
- **Tiles stay COPLANAR** — only the ligament bulges out of plane. This is what keeps the
  input at exactly 3 DOF.
- **Out-of-plane behavior:** smooth geometric flexure forced by the cut topology, **not a
  snap/bifurcation** (per user experience — to be VERIFIED in probe 3). The ±(up/down)
  symmetry is quotiented out automatically because we learn energy, not the `z`-field, so
  the dataset stays single-valued. A tiny imperfection seed is optional insurance.
- **Material — STEEL:** finite-strain **J2 (von Mises) elasto-plasticity**, isotropic
  hardening, geometrically nonlinear. Monotonic loading ⇒ **deformation-theory
  equivalence** ⇒ a **single-valued potential `W(ū)`** whose gradient is the reaction.
  (Exact for proportional loading; approximate for non-proportional — see §9.)
- **Failure / validity domains** (colored into the dataset, not just elastic/plastic):
  - **Elastic:** `σ_vM < σ_y`.
  - **Plastic:** `σ_vM ≥ σ_y`; still single-valued under monotonic loading; **modeled**.
  - **Failed:** ductile fracture when equivalent plastic strain `ε_p ≥ ε_f` ("thin hinge
    breaks"). Marked as an **invalid domain** via a scalar margin — NOT crack-propagation
    mechanics.
  - The surrogate outputs `W` **and** a validity/failure margin; the global training loss
    uses the margin as a **soft barrier** to keep hinges in the trustworthy region.

---

## 4. Nondimensionalization (the key simplification)

One thickness `t`, one material ⇒
`W = E·t·L_ref² · f(ū/L_ref ; dimensionless geometry)`.
"**Scale** is not a free input" — it enters only through the analytic prefactor `E·t·L_ref²`
and through the ratio `t/L`. The surrogate learns the dimensionless `f`. Plasticity adds
`σ_y/E` (and the hardening ratio) as extra dimensionless groups. This collapses the
sampling problem and lets one surrogate transfer across physical scales.

---

## 5. Local-geometry descriptor

Per-hinge dimensionless vector (candidate, confirmed with user):

| symbol | meaning | dimensionless |
|---|---|---|
| `L_main` | main cut length | reference `L_ref` |
| `L_sec` | secondary cut length | `L_sec/L_main` |
| `α` | angle between the two cuts | `α` |
| `g` | gap (secondary cut ↔ main cut) | `g/L_main` |
| `w_c` | cut width (slit) | `w_c/L_main` |
| `t` | sheet thickness | `t/L_main` |
| — | material | `ν`, `σ_y/E`, hardening ratio |
| — | fillet/tip radius (if cuts not sharp) | `ρ/L_main` |

≈ 7–9 dimensionless inputs (plus the 3 kinematics).

- **Scale-separation design rule (user):** ligament width ≤ ~**1/10** of the tile face
  length — an order-of-magnitude tile↔hinge separation. This is what justifies rigid faces
  + isolated-hinge condensation. Must stay within steel's working range.
- **Descriptor-completeness test:** build a standalone RVE from a *harvested* descriptor and
  compare its `W` to the windowed full-sheet `W`. Match ⇒ descriptor complete; mismatch ⇒
  a missing parameter.

---

## 6. Data pipeline — TWO DECOUPLED PROCESSES

We only control the cut pattern (`{r, x_bound}`); the hinge descriptor is emergent. But the
in-plane part is **derivable analytically** (no meshing) — see below.

**Descriptor derivability (resolved):** the in-plane cut geometry `(L_main, L_sec, α, g)` is
an analytic, differentiable function of the LES-solved cut vertices
(`closed_builder_jax.solve_cut_vertices_jax`), hence of `{r, x_bound}`. The current model has
**zero-width point-pivot hinges**, so the finite-width parameters `(w_c cut kerf, ligament
width, t thickness, ρ fillet)` are **new manufacturing DOF** specified separately (global or
per-cut), NOT in `{r, x_bound}`. So:
`descriptor = f(r, x_bound [→ L, α, g] , manufacturing_params [w_c, t, ρ])` — a JAX function,
no meshing. This is ALSO the **required on-the-fly inference evaluator** (§10).

**Process A — analytic descriptor extraction** (replaces the earlier "mesh + window" idea):
1. Apply the analytic map above → per-hinge descriptor. For training we sample the
   descriptor space **directly by LHS** (user goal: maximal coverage), bounding ranges by
   running the map on a few sample sheets.
2. **Crowding filter (analytic):** reject hinges whose window would overlap a neighbor (min
   inter-pivot distance vs. window budget) → flagged OOD at inference.

**Process B — RVE simulation** (separate builder / mesher / solver):
1. Standalone single-hinge builder from a descriptor → mesh → 3D geometrically-nonlinear
   **elasto-plastic** FEM under an imposed **monotonic** `(a,s,θ)` path, `z` free, tile-side
   boundaries rigid.
2. Harvest along the path: `(ū, W, ∇W = boundary reaction, σ_vM/σ_y, ε_p/ε_f)`.
   One run = an entire path of labels (sample paths, not endpoints).

---

## 7. Engine (recommendation — GATED by probe 1)

Requirements: 3D geometric nonlinearity, J2 plasticity, **thin-structure accuracy**,
native energy + reaction extraction, monotonic loading.

- **Leading candidate: a variational FEM where energy is native + plasticity + thin
  structures — FEniCSx** (shell or refined solid + J2). Directly solves SOFA's "no energy
  exposed" annoyance; the RVE is a standalone process anyway.
- **SOFA = fallback.** It is installed, but reuse buys little here: we would still need a
  new closed-cut mesh builder + a plasticity force field + thin-bending-suitable elements +
  post-hoc energy. Solid tetrahedra **shear-lock** in thin bending — the exact regime we
  care about.
- **GATE:** benchmark the chosen engine's thin-plate bending stiffness vs. the analytical
  value before committing. If it locks → use **shell elements**.

---

## 8. RVE boundary — "what counts as a hinge"

The truncation boundary is artificial and `W` depends on it.
- Place it where the deformation field has decayed toward rigid (**Saint-Venant**, ~a few
  ligament-widths beyond the cut tips), and make the global model's node / `reference_vector`
  placement **consistent** with that same boundary.
- The order-of-magnitude separation (§5) guarantees the window fits without touching
  neighbors.
- The **window-size sweep** (Process A step 4) confirms a `W`-plateau (window independence).
  If not converged within the tile budget, include the window ratio as a descriptor input.

---

## 9. Surrogate model

- **Residual form:** `W_θ = W_baseline(ū; k(geom)) + W_NN(ū; geom)`, with `W_NN`
  Taylor-residualized at the origin (`W=0, F=0, [H=0]` at `ū=0`) so the elastic small-strain
  limit is exact **by construction**, and OOD queries degrade gracefully to the baseline.
- **C² smooth activations** (softplus/tanh) — the global L-BFGS forward solve and the IFT
  backward solve need a well-behaved tangent stiffness `∇²W`. No ReLU.
- **Inputs:** 3 dimensionless kinematics + ~7–9 dimensionless geometry.
  **Outputs:** `W` + validity/failure margin(s).
- **Sobolev loss:** variance-normalized `energy-MSE + force(∇W)-MSE + validity-margin`
  loss. Optional 2nd-order (Hessian) term if the FEM emits tangent stiffness cheaply.

---

## 10. Integration into the pipeline

- Swap into `ligament_energy_linearized`'s last line; upstream (`ligament_strains`) and
  downstream (`statics` L-BFGS, IFT backward) are unchanged in shape.
- **Frame consistency:** ONE shared mapping `(relative tile DOF) → (RVE boundary rigid
  transform)`, used both to drive the RVE (labeling) and to feed the surrogate online.
  Unit test: pure rigid-body motion ⇒ `W = 0`.
- **Global-solver stability** with a non-convex softening `W`: residual form (near-convex at
  origin) + smooth activations + LM damping on the backward IFT solve (per existing
  conditioning notes) + load stepping (`lax.scan`) + validity barrier keeping the solver in
  the trustworthy domain.

---

## 11. De-risking probes (decide BEFORE the campaign; each ≈ one afternoon on ONE hinge)

1. **Mesh/element gate** — FEM thin-plate bending stiffness vs. analytical ⇒ engine/element
   decision.
2. **Yield + failure sweep** — `σ_vM/σ_y` and `ε_p/ε_f` vs. deployment ⇒ size of the elastic
   domain, how early plasticity/failure onsets, confirms the plastic model is needed.
3. **Smoothness / branch check** — sweep one DOF, plot `W(ū)` and z-amplitude ⇒ confirm
   smooth & single-valued (no snap), whether an imperfection seed is required.

---

## 12. Assumptions & known limitations

- Deployment only (no compression); monotonic loading.
- Scale separation: no hinge crowding (order-of-magnitude tile↔hinge separation); crowded
  configs filtered + flagged OOD at inference.
- Tiles coplanar (only ligament out-of-plane) ⇒ 3-DOF input.
- Broadly flat sheet (no macroscopic pop-up). Condensation is local — add a post-hoc global
  check (buckling eigenvalue on the assembled condensed tangent, or verify FEA `z` stays
  hinge-local).
- Single-valued `W` is EXACT only for proportional loading; non-proportional per-hinge paths
  introduce mild path-dependence (deformation-theory approximation error).
- Failure modeled as a scalar ductile-strain margin, not crack propagation.

---

## 13. Open questions

RESOLVED: **engine = FEniCSx** ✔. **Descriptor distribution** = sample descriptor space by
LHS for maximal coverage (no target sheets to harvest); realistic ranges bounded via the
analytic map ✔. **Deployment** = run each RVE monotonically to full deployment/failure
(magnitude free), sample kinematic ray *directions* ✔. Proportional rays for v1 ✔.

Still needed (inputs, not design forks):
1. **Steel properties:** grade → `E, ν, σ_y`, hardening law, `ε_fracture`; sheet
   thickness/gauge range.
2. **Deployment magnitude target:** bounds the kinematic ray extent.
3. **Manufacturing params:** cut kerf `w_c`, fillet `ρ` ranges (new DOF, not in `{r, x_bound}`).

---

## 14. Glossary of the key decisions already made

- Condensation, not a 3D global rewrite. ✔
- Full energy replacement of `ligament_energy_linearized`, not just k-extraction. ✔
- Deployment-only, monotonic, coplanar tiles ⇒ 3-DOF input. ✔
- Nondimensionalized (scale → `t/L`). ✔
- Two decoupled data processes (harvest descriptors from sheets → standalone RVE sim). ✔
- Elasto-plastic steel + ductile-strain failure domain; single-valued via monotonic
  deformation theory. ✔ (physical-model choice delegated to and made by the assistant)
- Engine leaning FEniCSx (variational, native energy, plasticity, shells), SOFA fallback;
  gated by the thin-bending benchmark. ✔ (delegated to and recommended by the assistant)
