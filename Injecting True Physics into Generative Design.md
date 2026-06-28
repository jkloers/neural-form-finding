> A non-differentiable finite element solver, wrapped in Tesseract, lets our kirigami pipeline stop guessing the physics of its own hinges.
---

While origami relies solely on folding, kirigami extends the concept by introducing cuts into the flat sheet. This ancient Japanese technique has evolved into a wide class of mechanical metamaterials where the layout of rigid panels and hinge networks entirely dictates the final geometric and physical behavior.

To bridge the gap between idealized kinematics and real-world fabrication, we introduce a [physics-informed form-finding pipeline](https://github.com/jkloers/neural-form-finding).

## I - Fast Geometry, Naive Physics

The pipeline designs kirigami the way you'd hope: by gradient descent. On a deliberately compact representation. Each face a rigid body with three degrees of freedom, each hinge a handful of springs ($k_\theta,\ k_\text{stretch},\ k_\text{shear}$) — three differentiable stages map network weights to a folded shape:

- **Stage 0 — Mapping:** a learned conformal map displaces and resizes the tiles.
- **Stage 1 — Validity:** a differentiable projection enforces the geometric constraints.
- **Stage 2 — Physics:** incremental load-stepping relaxes the cell to equilibrium.
    

We score the result against a target shape, and `jax.value_and_grad` carries the gradient back through all three stages, energy minimization included, to update the network. This works well.

![training_tess](https://canada1.discourse-cdn.com/flex009/uploads/si_tesseract/original/1X/f32b5c136e70482deaf33586da8187fcc9a295d6.gif)

_Figure 1: We show the output of Stage 2 (after the physical simulation has been applied). The optimization landscape is tightly constrained by the geometric rules that enforce a valid Kirigami mechanism. The gradient spikes come from a near-diverging Stage 2 solver (faces beginning to intersect), whose large equilibrium sensitivities are backpropagated through the implicit function theorem ._

**Our sim-to-real gap**  
But the whole edifice rests on one assumption, and it sits on the only component that does any real work — the hinge. Each hinge is a reduced-order energy model with no geometry.$$U_\text{hinge} = \tfrac{1}{2}k_\text{stretch}\,\delta_n^2 + \tfrac{1}{2}k_\text{shear}\,\delta_t^2 + \tfrac{1}{2}k_\text{rot}\,\Delta\theta^2$$,three spring constants we picked by hand. Before this tessellation can ever be 3D-printed, those hinges must be designed. And their design feeds back into the physics of the whole cell.  
It's a loop, and right now it's open.

---

## II - Wrapping the FEM Oracle

The real mechanics live in a continuum FEM solver. Hand [SOFA](https://github.com/sofa-framework/sofa), the true 3D hinge and it returns the true stress, strain, and energy. It is made for soft robotics, ideal for Kirigami simulation.

Problem solved? No — SOFA is everything our pipeline is not: slow, a black box with no gradients, and living in a runtime you cannot drop into a `value_and_grad` call.

![pipeline](https://canada1.discourse-cdn.com/flex009/uploads/si_tesseract/original/1X/3df381e598c944a9f8cc6883c80f66912034662b.jpeg)

_Figure 2 : The global pipeline of this project. SOFA is used to model the physical interactions between the tiles of a valid Kirigami (hinge design)._  
**

Tesseract** supplies the one missing piece. `/apply` runs SOFA; `/jacobian` returns ∂(outputs)/∂(inputs) by central finite differences, server-side. Wrap those two calls and the black box becomes a differentiable function — one that could occupy the _exact_ slot Stage 2 holds today, composing with `value_and_grad` without changing how the rest of the pipeline takes gradients.

```python
# ── The differentiable design loop: client → oracle → gradient step ───────────
#    nff/sofa/hinge_optimizer.py  (builds payloads via nff/sofa/oracle_payload.py; calls the oracle via the tesseract_core SDK)

params = initial_bezier_params()
for epoch in range(n_epochs):
    payload = tc.build_payload(cs, params)  # → InputSchema dict
    out  = tc.apply(url, payload)       
    jac  = tc.jacobian(url, payload, "smooth_principal_strain")
    
    eps_p = max(0.0, out.strain - yield_strain)  # plastic strain fatigues
    loss  = w_fatigue*eps_p + w_mat*hinge_area(params) + w_gap*gap**2
    params = adam_step(params, grad_of(loss, jac))  # reshape the hinge
```

A finite-difference Jacobian costs `2·n_inputs` SOFA solves per step — too slow to call inside every macro training step at full-tessellation scale. So we split the labor:

- **JAX owns the macrogeometry** (layout, fold target);
- **SOFA owns the microgeometry** (each hinge's cross-section, hence its stiffness and failure limit).

The whole game becomes: _how does the micro-loop talk to the macro-loop?_

---

## III — Micro-Optimization & Non-Linear Reality

At the hinge scale, the oracle is cheap enough to put directly in the loop. We parameterize the flexure as nine Bézier variables; `build_mesh_gmsh` turns them (plus the cell shape from Stage 1) into a conforming tetrahedral mesh — two faces pushed apart by a learnable gap, the hinge drawn as quadratic Bézier arcs whose endpoints slide along the face edges. SOFA folds it to equilibrium and returns elastic energy, peak stress, and peak strain — and, on request, their gradients.

A compact Adam loop reshapes the nine parameters to maximize Coffin–Manson fatigue life (minimize plastic strain), penalized for excess material and gap.

![training_hinge](https://canada1.discourse-cdn.com/flex009/uploads/si_tesseract/original/1X/e79f959f8ad9665789ca52200ae2491e3dc299e0.gif)

_Figure 3 :_ _The hinge itself is the thin strip of material spanning that gap, and its outline is drawn with two quadratic Bézier arcs — an upper one and a lower one. The gradient landscape is bumpy because we differentiate a max-over-elements quantity (the single hottest tet's strain) by finite differences through a black-box solver: tiny design changes flip which element is hottest._

This micro-optimization yielded two major breakthroughs:

**1 — We can now sculpt a hinge's cross-section to an elasto-plastic goal with full 3D FEM.**  
The headline design is a TPU flexure that folds a full 90° while holding peak strain to **15.4%**, safely inside the trustworthy range for **~2,000 fold cycles**.

**2 — We measured the hinge's true spring law .**  
Straight from SOFA energy: $k_{stretch} ≈ 8,500 N/m$ and $k_{shear} ≈ 905 N/m$ (a clean 1 : 9.4 ratio). And here is the twist: **stretch and shear are honest linear springs, but rotation is not** — its energy grows like $\theta^3$, its stiffness rising fivefold over the first few degrees. The fast model is faithful in two degrees of freedom and (with this hinge design at least) _structurally wrong in the third ._ That single measurement tells us exactly what to fix.

![closing-1](https://canada1.discourse-cdn.com/flex009/uploads/si_tesseract/original/1X/6c3d509678ff2d46e9beb16dbe0e4dd6ab441dad.gif)  
_Figure 4: The optimizer converges on a slender, concave hourglass flexure whose pinched waist lets the bending spread along the arc rather than concentrate at the corner, so the peak strain stays just at the edge of the elastic limit — accumulating only a small amount of plastic (permanent) strain per fold, which is what gives it a long fatigue life._

---

## Act IV — Where it goes: Towards Full Co-Design

Currently, the macro (JAX) and micro (SOFA) pipelines run side-by-side. In the coming weeks, the ultimate goal is to fuse them. While per-hinge co-design is a viable intermediate step, the true prize lies in building a **Differentiable Surrogate Model**.

By training a fast, lightweight neural model on high-fidelity SOFA data—mapping `(loading, shape)` to `(energy, stiffness, strain, cycle lifespan)` we could entirely replace the linear spring approximations. This unlocks the ability to train the hinge's micro-geometry directly within the main generative pipeline, at JAX speeds.

This also unlocks the potential for heterogeneous co-design: individually optimizing the energy landscape of every single hinge within the tessellation based on the precise local physics of the target deployment.

**Looking Forward**  
To make this approach more robust, future work will integrate hyperelastic material laws (Stable Neo-Hookean/TPU) to lift the current strain ceilings, and capture closure self-contact and out-of-plane buckling.