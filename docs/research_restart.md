# Research restart — where the pipeline stands (2026-11)

A one-page statement of what works, what was measured, and what is open, written from the
code and the agent memory after the PFE. The last section is for the author: the direction of
the next phase is not something to infer.

## What the pipeline does today

- **Inverse design of a closed kirigami sheet, end to end differentiable.** Design parameters
  (per-cut aspect ratios + boundary sliders) build a flat RDPQK sheet that is valid by construction
  (Dang et al. 2021 collinearity system, Tutte embedding); Stage-2 statics deploys it under forces,
  moments or prescribed displacements; the shape error is back-propagated through the equilibrium.
  Verified: Taylor remainder converges at order 2.05 (learned hinge, tight forward solve).
- **Two hinge energies behind one switch.** The linear-spring ROM (constants condensed from FEA) and a
  learned condensed energy `W(a, s, θ; w_lig, α)` with ductile damage and an out-of-domain barrier,
  trained on a CalculiX RVE campaign (PET: 15 948 states over 1 370 jobs, `hinge_surrogate_pet_v3`).
- **From design to a cut sheet.** Target calibration from tile kinematics, cut patterns as DXF and
  A3/A4 hand-cutting sheets; a 3×3 PET sheet was cut, deployed and measured against the prediction.
- **Why the chart matters, measured.** The validity ladder: free vertices, projection and penalties
  leave voids or overlaps; the closed chart leaves neither and fits best (Chamfer/radius 0.47%).

## What was measured and is now settled

- The PET surrogate's ~29% error was never capacity, features, or data volume: the campaign mixed two
  route families, making `W` multi-valued. On proportional (origin-ray) routes it scores ~6%.
- `W` is a path functional for an elastoplastic hinge: same endpoint, different route, +65% energy
  and 3.2× damage. The deployed routes are close to proportional (second-order effect, exponent 2.3).
- Rotation drives hinge damage, not compression; compression depth and damage are anti-correlated.
- In grip-C (single-tile diagonal pull), the binding trust-box axis is `θ < 0`, not compression.
- The geometric inverse problem is under-constrained: many cut patterns reach the same outline.

## Open problems (from the article's own limits)

1. **Return memory.** A hinge opened 40° and closed back to flat accumulates 1.211× the damage of the
   fold alone, while a state-function surrogate reads zero. Any cyclic or service-load design needs a
   history variable or a path-aware energy. (An internal-variable surrogate `W(u, z)` with `z` =
   damage was tested and found infeasible: sufficiency saturates at 13%.)
2. **Trust region of the surrogate.** 2.1% of deployed routes leave the region any training pair
   covers; the out-of-domain barrier bounds them but the designs are then extrapolations.
3. **Scale.** The differentiable LES has a compile wall around 20×20 panels.
4. **Physical validation depth.** One 3×3 PET sheet: the mean opening is verified (per-hinge RMSE 2.4°
   exceeds the 1.5° spread), its distribution is not.
5. **Coverage.** The RDPQK chart reaches a subset of rigidly deployable patterns, not all of them.

## Candidate directions (for discussion, not a plan)

- **Path-aware hinge model:** a recurrent or incremental surrogate trained on mixed-route data, so
  open-close cycles and service loads become designable (attacks 1 and, partly, 2).
- **Active learning on the trust region:** harvest the routes optimised designs actually take, run
  CalculiX there, retrain; close the loop automatically (attacks 2).
- **Larger sheets:** a sparse or multigrid LES, or a domain-decomposed chart (attacks 3).
- **A second physical campaign:** more sheets, per-hinge measurement, another material (attacks 4).
- **Beyond the flat plane:** out-of-plane deployment (the hinge already buckles out of plane), or
  multi-material sheets.

## For the author

- What is the next deliverable: a journal paper from the PFE, a new problem, a demonstrator?
- Which of the open problems matters for it, and is there a deadline?
- Is a new physical campaign possible (lab access, material)?
