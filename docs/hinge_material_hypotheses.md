# Hinge RVE — material & modelling hypotheses

Assumptions behind the single-hinge condensation RVE, as of the plastic-deployment
step. These are deliberate, first-pass choices; everything here is meant to be swept.

## Material — mild structural steel (S235 / A36)

| Quantity | Symbol | Value | Note |
|---|---|---|---|
| Young's modulus | E | 210 GPa | standard steel |
| Poisson ratio | ν | 0.30 | |
| Yield strength | σ_y | 235 MPa | S235; ε_y = σ_y/E ≈ 0.11 % |
| Hardening modulus | E_t | E/100 ≈ 2.1 GPa | mild **linear isotropic** hardening |
| Fracture strain | ε_f | ≈ 0.25 | ductile tearing threshold (validity flag) |
| Thickness | t | 0.5 mm | thin foldable gauge (1 mm was too stiff) |

Isotropic, homogeneous, room temperature, rate-independent.

## Constitutive model — J2 (von Mises) plasticity, **deformation theory**

- Bilinear σ_eq–ε_eq law: elastic slope 3μ up to σ_y, then a plateau + hardening E_t.
- **Deformation theory is used, and is justified only because the deployment is
  MONOTONIC** (one opening, no unloading). For proportional monotonic loading it
  coincides with incremental J2, but with no history variables — a standard Newton
  solve suffices. It does **not** model unloading / springback / residual stress.
- Finite strain (geometric nonlinearity) — required to capture the out-of-plane buckle.

## Kinematics — the closed-kirigami mechanism

- The two faces are **rigid** and move in 2-D; their relative motion is a **rotation
  about a pivot**. Faces stay **coplanar** (z = 0 on the arc handles).
- Only the **hinge (ligament) bends out of plane** — a compression-driven buckle that
  lowers the energy. This is the mechanism the surrogate condenses.
- **Pivot** is not guessed: it is the point (near the primary-cut tip) that
  **minimises the hinge energy** for the imposed rotation — i.e. the relative
  translation is relaxed, so `W = W(θ, geometry)` alone.
- **One deployment only**, quasi-static.

## Geometry (first RVE values, all swept later)

| Quantity | Value | Meaning |
|---|---|---|
| Ligament length w_lig | 5 mm | primary-cut tip → secondary cut |
| Kerf w_c | 0.2 mm | standard laser cut width |
| Tip fillet ρ | 0.4 mm | stress-concentration relief |
| Saint-Venant window r | 12 mm | ≈ 2.4·w_lig |

Bending strain to fold 90°: ε ≈ t·θ/(2·w_lig) ≈ 8 % — plastic, but < ε_f, so the hinge
takes a permanent set without tearing.

## Numerics

- **P2 (quadratic) tets**, 3 layers through the thickness (P1 locks — see probe 1).
- Small out-of-plane geometric imperfection (crest along the primary-cut axis) seeds
  the buckle; faces pinned coplanar on the arcs, interior free.
- FEniCSx / dolfinx 0.11, MUMPS direct solve, Newton line search.

## Not modelled yet (known limitations)

- Unloading / springback / residual stress (deformation theory is monotonic-only).
- Self-contact of the faces at large folds.
- Anisotropy, rolling texture, strain-rate, temperature, heat-affected zones.
- Fatigue / cyclic deployment (single deployment only).
- The failure flag (ε > ε_f) is post-hoc, not a damage model.
