---
name: jacobian-cnv-transform
description: Why the per-face Jacobian for CNV transformation beats independent scale/theta, and what det-normalization breaks
metadata:
  type: project
---

**The Jacobian approach:** For each face i, fit 2×2 deformation gradient F_i from neighbor centroid displacements: `F_i = (X^T X + εI)^{-1} X^T Y`. Apply to CNVs: `new_cnv = F_i @ cnv`. This is the same formula as the polynomial Jacobian (`jax.jacfwd`), just evaluated numerically.

**Why better than independent scale/theta:** Independent per-face R(θ_i)*s_i has no coupling between faces — two adjacent faces can compute their CNV transforms completely differently and leave large hinge gaps. The Jacobian uses the *collective motion of all neighbors*, so adjacent faces get similar Jacobians near their shared hinge → automatic hinge pre-closure.

**v3 vs v4 (det-normalization):** v4 divided F by `sqrt(|det(F)|)` to remove area, leaving only rotation/shear. Intended: GNN's `local_scale` to control face size uniformly. Problem: at init `local_scale≈1` (zero weights → exp(0)=1), but the flat tessellation needs to be scaled up to fill the unit circle — F already carries this scale. Det-normalization strips it → faces start too small → chamfer 0.022–0.067 vs v3's 0.019–0.035.

**Current code (v5+):** Uses full Jacobian `F`, same as v3_s2 (best result). GNN's `local_scale` and `local_theta` provide fine-tuning corrections on top.

**About individual node vectors:** Jacobian applies same F_i to all nodes of face i. Per-node corrections would add a GNN output head predicting `δ_{f,n}` per node — more expressive, but harder to train and no geometric coordination between nodes of adjacent faces. Deferred for later.
