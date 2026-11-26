# Citation audit of `paper/sn-article.tex`

Audited 2026-08-09. Every citation site in the manuscript was checked against the cited
source's own text. Full-text PDFs were read where available (13 from `~/Zotero/storage/`,
the rest retrieved from arXiv, publisher open access, the NSF Public Access Repository,
or institutional mirrors). Metadata was checked against Crossref.

**Scope, as agreed with the author:** a citation fails when the cited paper does not
establish the claim attached to it. Preprint-versus-journal, survey-versus-primary and
missing-citation gaps are not failures. Objective bibliographic errors are collected
separately in Appendix A and are not counted as failures.

> **STATUS 2026-08-09, second pass: ALL FIXES APPLIED.** Every failing site below has
> been repaired in `sn-article.tex` and `sn-bibliography.bib`. Nine new sources were
> found and quote-verified; five keys were retired. The isolated build is clean at
> exit 0, 47 pages, 0 undefined references, 0 undefined citations, 41 keys cited
> against 41 rendered references. See "What was applied" immediately below. The
> verdicts in the body are the pre-fix state and are kept as the record of why each
> change was made.

## What was applied

**New sources, each verified by verbatim quote against the source text:**

| Key | Replaces | At |
|---|---|---|
| `shyu_kirigami_2015` + `isobe_softening_2016` | `rafsanjani2017` | linear-cut sheet |
| `dias_kirigami_2017` | `zhao_modular_2025` | "used as actuators" |
| `babaee_kirigami_2021` | (gap) | biomedical devices |
| `newbury_review_2024` | `lavin_simulation_2021` | definition of differentiable physics |
| `xue_jaxfem_2023` | `lavin_simulation_2021` | implicit function theorem, inverse design |
| `oed_kirigami` | `temko_kirigami_1962` | etymology, now 1958 |
| `coulais_characteristic_2018` | `schenk_origami_2011`, `filipov_bar_2017` | the three-spring model |
| `cerda_conical_1998` + `liang_crescent_2005` | `lobkovsky1996` | developable cone |

**Retired (now uncited; `lavin` and `temko` kept in the bib with a SUPERSEDED comment):**
`schenk_origami_2011`, `filipov_bar_2017`, `liu_nonlinear_2017`, `hong_boundary_2022`,
`lavin_simulation_2021`, `temko_kirigami_1962`.

**Claims reworded where no source supported them as written:** the Dang criterion now
carries its genus-0 and M,N >= 3 hypotheses; faces rotate rather than buckle; the
d-cone sentence now says curvature and stress rather than displacement; "decompose
generally" became "has been approximated as"; Zheng is now cited for single-cell
kinematics at leading order rather than homogenisation; Bordiga's second task is
shielding rather than absorption; Deng is scoped to a hinged-quadrilateral cell;
Yazdani is a conditional generative model fine-tuned by RL; Schek and Pastrana are
split so the differentiable-solver claim rests only on Pastrana; and the ligament
comparison is now the (t/w)^2 stiffness ratio, with "beams from flat stock" dropped
because no source supports it.

**Metadata corrected:** `vlassis_sobolev_2021` DOI (2020 -> 2021), `bradbury_jax_2018`
missing author Katariya, `dang_compatibility_2021` preprint -> PRL 128:035501 (2022),
`grima2000` pages and DOI, `zheng_modelling_2023` issue 2272, `adriaenssens_shell_2014`
author -> editor, `yazdani_reinforcement_2026` title and volume,
`frostig_compiling_2018` venue, `rafsanjani2017` DOI added, `dhondt_finite_2004`
version note.

**Two items deliberately NOT actioned, both flagged for the author:**

1. **The sign of `eq:dcone-log` may disagree with the literature.** The d-cone energy
   is E ~ kappa eps^2 ln(R_out/R_core), whose conjugate force *increases* with
   ln R_out at fixed cone angle. The manuscript's fit has a *negative* coefficient,
   F_s^max decreasing in ln(r_win). The citations support "logarithmic, no plateau,
   because the conical field is not screened"; they do not explain the direction.
   This is a question about the measurement, not the citation, so it was left alone.
2. **`adriaenssens_shell_2014`'s definition was read only as Google Books snippets.**
   Verbatim and consistent, but confirm against a physical copy before submission.

**Line numbers** are from the file as read at 2026-08-09 15:09. Three other background
sessions were working on the manuscript concurrently (`restructure-appendix-a`,
`standardize-figure-captions`, `add-kirigami-citations`), so line numbers may have
drifted. Every entry quotes the claim verbatim, so sites can be relocated by grep.

---

## Verdict summary

54 claim-source pairs across 38 distinct keys and 42 `\cite` commands.

| Verdict | Count |
|---|---|
| SUPPORTED | 31 |
| PARTIAL | 13 |
| UNSUPPORTED | 4 |
| WRONG SOURCE | 6 |

Every pair carries a verbatim quote from the source. Nothing was left unverified.

**Nine sites need action before submission.** In descending order of severity:

1. **l.748** `filipov_bar_2017` and `schenk_origami_2011` do not state the three-spring
   model. `bordiga_automated_2024`, already co-cited, states it verbatim.
2. **l.326** `filipov_bar_2017` is cited for a kirigami claim in a paper that explicitly
   excludes kirigami.
3. **l.358** `lavin_simulation_2021` carries the manuscript's definition of differentiable
   physics and does not contain it.
4. **l.2100** `lobkovsky1996` analyses a different singularity from the one under discussion.
5. **l.300** `rafsanjani2017` is cited for the linear-cut sheet and studies orthogonal cuts.
6. **l.276** `temko_kirigami_1962` supports a first-attestation date the OED contradicts.
7. **l.351** `deoliveira_experimental_2023` supports neither half of its two-part claim.
8. **l.344** the Dang criterion is stated without its two hypotheses.
9. **l.2103** `chaudhary_geometric_2023` is followed by an inference the source resists.

A tenth, `l.369 schek_force_1974`, attributes gradient-based optimisation through a
differentiable solver to a 1974 paper containing zero instances of "gradient" or "optimiz*".

Two further defects are in the manuscript's own prose rather than in a citation, and
are recorded in Appendix C because they were found during this audit.

---

## 1. The Dang lineage

Source read: local PDF `NJZZXWMF/...pdf`, which is arXiv:2106.15891v2, plus
arXiv:2107.13744v2 and its supplement, plus local PDF `KSMZST8W/Dang et Paulino...pdf`.

### l.344 `dang_theorem_2021` -- PARTIAL

> "For rigid planar quadrilateral tessellations, Dang *et al.* prove that a pattern is
> rigidly deployable if and only if every void of cuts is a parallelogram."

The source result is **Lemma 1 (Genus-0 deployability)**: *"A genus-0 PQK tessellation is
rigidly deployable if and only if all the cuts form parallelogram voids."* The unqualified
sentence is true only under two hypotheses the source states and the manuscript omits here.

Both hypotheses are load-bearing, not decoration:

> "Note that the tessellations discussed here are restricted to having at least three
> columns and rows of panels (i.e., M, N >= 3). The PQK tessellations with two columns or
> rows are actually always rigidly deployable even if the voids of cuts are arbitrary
> quadrilaterals."

and for genus >= 1 the criterion changes shape entirely, becoming **Theorem 1**: *"A genus-n
(n >= 0) PQK tessellation is rigidly deployable if and only if there exists a deployed state
with all the cuts forming parallelogram voids."*

**Remedy.** Add the qualifier at l.344, as l.506 and l.2414 already do. Suggested:
"for a sheet with no interior hole and at least three panels in each direction". No source
change.

### l.506 `dang_theorem_2021` -- SUPPORTED

All three hypotheses the manuscript states (M x N panels, M,N >= 3, no interior hole) are
in the source. The flat-sheet readability claim is also supported: the void side lengths
are defined on the undeployed segment, *"Since these voids are straight segments at the
undeployed state ... we have the constraints a_i + b_i = c_i + d_i"* [Appendix A].

### l.2279 `dang_theorem_2021` -- SUPPORTED

Both halves of the Appendix A attribution hold. The collinearity system is the source's
Eqs. (3)-(4):

> "(1 - r_{i,j}) x'_{i+1,j} - x_{i,j} + r_{i,j} x_{i-1,j} = 0 ... The equations for
> t_{i,j} = 1 and 2 in Eq. (3) represent the collinear constraints of vertices on
> horizontal and vertical cuts, respectively."

and `r` is defined there as the manuscript uses it:

> "We denote the side lengths of the parallelogram voids by a_{i,j} and b_{i,j} ...; then
> the aspect ratio for each cut C_{i,j} is defined by r_{i,j} = b_{i,j}/a_{i,j}."

### l.2400 `dang_theorem_2021` -- SUPPORTED

The pointer to their Appendix A is exact:

> "This lemma was stated as a corollary of a compatibility theorem of spherical kirigami
> tessellations, and proved as the degeneracy of spherical geometry in Ref. [34]. Here we
> provide a straightforward proof in Appendix A directly from the viewpoint of planar
> geometry."

Their Appendix B holds the genus-n proof, so "Appendix A" points at this result and only
this result.

### l.2414 `dang_theorem_2021` -- SUPPORTED

The iff, the M,N >= 3 hypothesis, the genus-0 hypothesis and the opposite-sides-equal
rendering are all in the source. One point of slippage, harmless but worth knowing: the
manuscript defines "deploys" as *exactly one internal degree of freedom*, whereas Dang
defines rigid deployability as connection-preserving rigid-panel motion and *derives* the
single degree of freedom for genus-0 RDPQK. The source numbers this **Lemma 1**, not a
theorem; their Theorem 1 is the genus-n statement. The manuscript labels it "Criterion"
and claims no numbering, so nothing is misattributed.

### l.2399 `dang_compatibility_2021` -- SUPPORTED

The degeneration story is exactly as the manuscript tells it. From arXiv:2107.13744v2:

> "If the Gaussian curvature K is sufficiently small, we can expect that the SQK
> tessellations are approximately located on a plane, degenerating into planar
> quadrilateral kirigami (PQK) tessellations. When K = 0, Eq. (3) becomes g^e(x) = x ...
> **Corollary 2** A PQK tessellation is rigidly deployable between two compact
> configurations if and only if all the slits form parallelograms."

Independently corroborated by `dang_theorem_2021`'s own attribution sentence, quoted above.
See Appendix A: the entry is a superseded preprint.

### l.280 `dang_kirigami_2025` -- SUPPORTED

> "Since kirigami engineering is performed on a surface, the most basic deformation mode is
> the 2D deformation within the surface. In this case, the kirigami hinges only allow the
> connecting panels to rotate around axes perpendicular to the surface. In theory, the
> cutting geometry typically determines the deformation mechanics..."

The review also supports the "nearly" in "nearly rigid tiles", via its hinge-stiffness
caveat and its description of *"relatively thick ligaments at the intersection of cuts
(called living hinges or compliant hinges)"*.

---

## 2. Geometric inverse design

All seven pairs SUPPORTED. Metadata clean on all three entries.

### l.326 `choi_programming_2019` -- SUPPORTED, with a scope note

> "Here we pose and solve the inverse problem of determining the number, size and
> orientation of cuts that enables the deployment of a closed, compact regular kirigami
> tessellation to conform approximately to any prescribed target shape in two or three
> dimensions."

Two precisions the manuscript should be aware of. Their inverse design is **purely
geometric**, a constrained optimisation with no hinge energy; the spring model enters only
afterwards, to study the deployment path:

> "our method so far is agnostic to the path of deployment. To explore the deployment
> process of our patterns, we now extend our purely geometrical approach to a mechanical
> model by having linear springs along the edges and diagonals of the quads, and simple
> torsional springs at the nodal hinges to model the ligaments there."

And in that model the tiles are **not rigid**: springs sit along the edges *and diagonals*
of each quad, which is a bar-and-hinge network. The manuscript's own disjunction
("kinematics alone, or ... a bar-and-hinge network") absorbs both, so the citation stands.

### l.326 `dudte_additive_2023` -- SUPPORTED, ideal-joint limb only

Prescribed-shape design and rigid tiles are explicit. But a full-text search for
*energy / spring / torsional / stiffness / equilibrium* returns **no mechanical model
anywhere**: the design is kinematic linear algebra, with hinges as the pin joints of
parallelogram four-bar linkages. Only the "ideal joint" limb of the manuscript's sentence
applies to them, never the "linear torsional spring" limb.

### l.346 `choi_programming_2019`, `dudte_additive_2023` -- both SUPPORTED

Choi's hinge is a dimensionless interior node at which four tile angles close to 2pi.
Dudte's is a shared corner node, *"x0_{j,2} = x0_{j+1,0} denotes a shared node"*.

One caveat worth recording: Dudte *does* give the hinge a width at fabrication ("Slots of
0.75 mm in width are cut into the tiles"). "A sheet cut from stock has no such point" is
true of their *geometry*, but the paper is not innocent of the issue.

### l.1069 `choi_programming_2019`, `dudte_additive_2023` -- both SUPPORTED

Purely geometric optimisation to a prescribed target with no equilibrium solved is
genuinely the setting of both. Dudte is the closer match: their design matrix is a chart
whose image is the admissible set, and target matching is a distance-to-outline objective
over the offset field.

Choi's method is a constrained nonlinear program (`fmincon` with analytic gradients) that
enforces admissibility by *constraints*, which is the opposite of the manuscript's
parameterisation strategy. That is a contrast the manuscript could exploit, not a defect:
the sentence claims a shared setting, not a shared method.

### l.309 `liu_quasicrystal_2022` -- SUPPORTED

> "Our starting point is an aperiodic quasicrystal tiling pattern, which we seek to make
> deployable by cutting it along appropriate edges to articulate the structure while
> keeping it as a single connected whole."

Penrose, Ammann-Beenker and Stampfli tilings. Metadata clean; PRR uses article numbers,
so `pages = {033114}` is correct.

---

## 3. The bar-and-hinge cluster

This is the weakest group in the manuscript. Five pairs, none SUPPORTED.

### l.326 `schenk_origami_2011` -- UNSUPPORTED

The paper is origami only. "Kirigami" and "cut" never appear. It is a forward eigenmode
study, not an inverse design:

> "by studying its softest eigenmodes, important deformations of a partially folded sheet
> can be found, which aids in the understanding of Origami sheets for engineering
> applications."

"Inverse design", "target" and "prescribed shape" appear nowhere. It also rejects the
rigid-panel premise the manuscript's sentence attributes to it:

> "Our purpose is not to formulate an alternative method to describe rigid origami, as we
> aim to obtain different information. ... It should also not be limited to rigid origami
> as the out-of-plane kinematics of the sheets involves bending of the facets."

Supports only the "equilibrium on a bar-and-hinge network" clause, and only for origami.

### l.326 `filipov_bar_2017` -- UNSUPPORTED, and the source disclaims the application

The paper names kirigami exactly once, to exclude it:

> "From this simplification, our model ignores the local effect around the boundary edges
> (e.g. potential buckling), and thus in the current form, **it would be difficult to deal
> with kirigami models where cuts on surfaces produce higher compliance**."

Its three stated objectives are all forward: introduce the model, discuss thin-sheet
behaviours, demonstrate physical simulation. The only design language is a suitability
claim, never a demonstration: *"its efficiency makes it suitable for design problems such
as optimization"*. Panels are deformable by construction.

This is the single most exposed citation in the manuscript. A referee who opens it finds
the paper declining the job it is cited for.

### l.326 `liu_nonlinear_2017` -- UNSUPPORTED

"Kirigami" appears once, inside a reference title. Three forward simulations, no inverse
problem; "optimiz*" and "inverse design" occur only in the reference list. Non-rigid by
title and abstract.

### l.748 `schenk_origami_2011` -- WRONG SOURCE

The manuscript's `eq:rom` is three uncoupled quadratic springs in per-hinge relative tile
motion: opening `a`, shear `s`, rotation `theta`. Schenk's three stiffnesses are **bar
axial, facet bending, and fold-line bending**:

> "we introduce a bending stiffness along the fold line (K_fold) and across the facets
> (K_facet) ... we obtain a material stiffness matrix that incorporates the stiffness of
> the bars, as well as the bending stiffness of the facets and along the fold lines."

The DOFs are nodal displacements. There is no hinge-opening measure and no hinge-shear
measure anywhere in the paper. Only one of the three terms lives at the hinge at all. No
energy expression is ever written; the paper is stiffness-matrix-only.

### l.748 `filipov_bar_2017` -- WRONG SOURCE

> "The model simulates three distinct behaviors: stretching and shearing of thin sheet
> panels; bending of the initially flat panels; and bending along prescribed fold lines."

Filipov's "shear" is **panel** in-plane shear carried by bars inside the panel, not sliding
at a hinge. Panels are joined at shared nodes, so opening is not a degree of freedom.
And the panel-bending hinge is explicitly **nonlinear**, M_BL proportional to theta^(4/3),
so "quadratic" fails too.

### The fix is already in the cluster

`bordiga_automated_2024`, the third key at l.748, states the manuscript's model verbatim:

> "the strain energy of the ith ligament is assumed to be of the form
> V_i = 1/2 [ k_l (eps_i l0_i)^2 + k_theta dTheta_i^2 + k_s (psi_i l0_i)^2 ] + V_i^c,
> where k_l, k_theta and k_s denote the stiffness of the ligament upon stretching,
> bending, and shearing, respectively."

on *"a network of rigid units connected by flexible ligaments"*. Three uncoupled quadratic
terms in exactly the manuscript's three measures. Its additive `V_i^c` is a differentiable
contact term with a Heaviside cut-off and an asymptote at theta_min, which is the same
construction the manuscript describes two sentences later as its own barrier.

**Remedy.** At l.748, keep `bordiga_automated_2024` alone. If the bar-and-hinge lineage
must be acknowledged, phrase it as ancestry ("descended from the origami bar-and-hinge
models of ...") rather than as the source of the three-spring form. At l.326, remove all
three; `choi_programming_2019` and `dudte_additive_2023` already carry the claim.

One caveat for rigour: Bordiga et al. do not call their own system kirigami either, so
"the standard energetic model of **kirigami**" remains slightly stronger than any source
says. Their own antecedents for the discrete spring model are Coulais, Kettenis & van
Hecke, *Nat. Phys.* **14**:40-44 (2018) and Deng, Raney, Tournat & Bertoldi, *Phys. Rev.
Lett.* **118**:204102 (2017).

---

## 4. Buckling and the developable cone

### l.300 `rafsanjani2017` -- WRONG SOURCE

> "Parallel rows of straight slits give the linear-cut sheet, whose tiles are ribbons
> joined end to end."

The paper studies one topology, and it is not this one:

> "We investigate the mechanical response of thin sheets perforated with a **square array
> of mutually orthogonal cuts**, which leaves a network of squares connected by small
> ligaments."

It explicitly assigns the parallel-slit sheet to other people's work: *"instability-induced
kirigami designs subjected to tensile loading are limited to a single incision pattern
comprised of parallel cuts in a centered rectangular arrangement [19-23]."* "Ribbon",
"strip" and "linear cut" never appear. The only variable it explores is the loading
direction on the same orthogonal array.

**Remedy.** Replace with the sources Rafsanjani cites as [19-23]: Shyu et al., *Nat. Mater.*
**14**:785 (2015); Blees et al., *Nature* **524**:204 (2015); Lamoureux et al., *Nat.
Commun.* **6**:8092 (2015); Isobe & Okumura, *Sci. Rep.* **6**:24758 (2016). These were
identified from Rafsanjani's own reference list and have not been read in full; verify
before citing.

### l.349 `rafsanjani2017` -- PARTIAL

Ligament buckling and ligament yield are supported:

> "under uniaxial tension **the ligaments buckle out of plane**, inducing the formation of
> 3D patterns whose morphology is controlled by the load direction."
> "by largely stretching the buckled perforated sheets, **plastic strains develop in the
> ligaments**"

Face buckling is not. The faces *rotate*, driven by the hinge buckling, and the paper's own
analysis holds them rigid:

> "a sudden departure from linearity to a plateau stress caused by the out-of-plane
> **buckling of the hinges**. Such buckling **in turn induces out-of-plane rotations of both
> the square domains and the cuts**"
> "assuming that **the square domains remain rigid** and that the deformation localizes at
> the hinges"

**Remedy.** Drop "and the faces" from the claim, or say the faces rotate out of plane
rather than buckle.

### l.765 `rafsanjani2017` -- PARTIAL

"Almost immediately after opening" holds only in the thin-hinge limit, and the controlling
parameter is the ratio t/delta, not absolute thickness. The source's closed form is
eps_c ~= (1/2)(t/delta)^2, which gives ~0.004 for their thinnest specimen and ~0.13 for
their thickest.

The following sentence, "The fold moment then saturates", is uncited and is **not** in the
paper. The nearest statement is a macroscopic stress plateau, and that plateau ends:

> "for large enough values of the applied strain eps_x, **the stress sigma_x rises sharply
> again** ... the deformation mechanism of the hinges switches from bending dominated to
> stretching dominated."

**Remedy.** Qualify "almost immediately" with the thin-ligament condition, and either cite
the saturation claim to the manuscript's own oracle data or state its limit.

### l.2100 `lobkovsky1996` -- WRONG SOURCE

The strings "cone" and "conical" occur **zero** times in the paper. It analyses the ridge:

> "We examine a simple case of a plate bent by forces exerted along its boundary so as to
> create **a sharp crease** ... We find a separable boundary layer solution of the von
> Karman plate equations which is **valid along the ridge line**."

Its energy is a power law, not logarithmic: *"The elastic energy of the ridge scales as
E ~ kappa (X/h)^{1/3}"*. "Log", "ln" and "1/r" do not appear. Being a boundary-layer
analysis, it is if anything the opposite framing to "no boundary-layer decay".

### l.2100 `witten2007` -- PARTIAL

The logarithmic energy is supported exactly, in the manuscript's own form:

> "B = (1/2) kappa integral ... = (1/2) **kappa log(R_p/R*)** integral ds c(s)^2 ...
> Being proportional to the dimensionless material constant kappa, this energy cannot vary
> as a power of the system size. Instead, **a logarithmic growth with system size is
> dictated**."

The 1/r claim is supported for **curvature and stress**, not displacement (see Appendix C).
"Saint-Venant" does not appear in the paper, and Witten reports *exponential* decay for the
ridge flank: *"Any small edge displacement decays in a length of order L, so the decay is
exponential."* The substantive support for the manuscript's point is indirect but real: an
energy growing as log(R_p/R*) is by construction not confined to a boundary layer.

**Remedy.** Replace `lobkovsky1996` with **Liang, T. and Witten, T. A., "Crescent
singularities in crumpled sheets", *Phys. Rev. E* 71(1):016612 (2005), doi
10.1103/PhysRevE.71.016612**, which carries both propositions in one place:

> "The total curvature C has the form **C = phi(theta)/r**. Its bending energy B_0 is thus
> given by B_0 = (1/2) kappa integral ... **proportional to kappa log(R/R_c)**."
> "We have found out two types of stresses in the outer region of a single d-cone ... One
> type ... **scales as 1/r^2**. The other type of stress **scaling as 1/r** is needed to
> balance the external pushing force."

and, on non-locality, *"They persist up to the supporting container edge, where normal
force from the container counteracts the external pushing force."*

Not proposed, because no quotable full text could be reached: Cerda & Mahadevan *PRL*
**80**:2358 (1998); Cerda et al. *Nature* **401**:46 (1999); Ben Amar & Pomeau *Proc. R.
Soc. A* **453**:729 (1997).

Note for the record: `lobkovsky1995`, also in the bib and uncited, is likewise ridge-only.
Neither Lobkovsky entry covers a developable cone.

---

## 5. Kirigami mechanics

### l.310 `chaudhary_geometric_2023` -- SUPPORTED

> "we take a step in this direction by characterizing the geometry and mechanics of a
> single cut subject to deformation, and then generalize our study to kirigami sheets with
> randomly located cuts, but staying in the dilute limit where the cuts do not intersect."

### l.2103 `chaudhary_geometric_2023` -- PARTIAL, and the inference after it is unsupported

The four-item list is verbatim exact:

> "We show that the geometry of mechanically loaded sheets **can be approximated as** a
> composition of simple developable units: flats, cylinders, cones and compressed
> Elasticae."

Two problems. "Decompose **generally**" overstates "can be approximated as", and the result
holds only for dilute non-intersecting slits in a circular sheet under two extensile point
loads. The one thing they do claim for an arbitrary pattern is a geodesic theorem, not the
developable decomposition.

More seriously, the manuscript's *next* sentence -- "A long-ranged conical disturbance is
therefore the expected companion of a cut, not an artefact of the present window" -- carries
no citation of its own and will be read as Chaudhary's. The source resists it:

> "Indeed, regardless of the number of cuts, **the conical domains localize** to the ends of
> cuts that are nearest to the loading points while the sheet remains flat in between."

and their only kappa ~ 1/r statement sits inside the sentence justifying that the cones are
*negligible*:

> "We note that we have ignored the contribution of energy from the conical domains since
> the size of conical domains is much smaller than the cylinder-like domains just above the
> onset of out-of-plane deformation, and the mean curvature decays away from the cone tip
> as kappa ~ 1/r."

**Remedy.** Replace "decompose generally into" with "has been approximated as a composition
of". Then let the long-range argument rest on `witten2007` and the proposed Liang & Witten
2005, which is where it actually lives, and stop attributing it to Chaudhary. The
manuscript's own F_s^max = 303 - 41.2 ln(r_win) measurement is the strongest evidence for
it in any case.

### l.305 `zheng_modelling_2023` -- PARTIAL

Periodicity is a genuine hypothesis and the leading-order effective response is built from
single-cell mechanism kinematics by Cauchy-Born. But the paper's actual contribution is
that a single cell is *insufficient*:

> "The model accounts for three sources of elasticity: a strong preference for the effective
> fields to match those of a local mechanism, inter-panel stresses arising from gradients in
> slit actuation, and distributed hinge bending."
> "Domain walls cannot be captured with a model based only on cell-averaged deformations."

which is why the title says *generalized* continua. Also, the word
"homogenization"/"homogenisation" never appears; the mechanism is Cauchy-Born coarse-graining.

**Remedy.** Soften to "the leading-order effective response is set by the kinematics of a
single cell", or change "homogenisation" to "coarse-graining".

### l.326 `hong_boundary_2022` -- PARTIAL, recommend removal from the cluster

The inverse-design clause holds, though even it is off by a shade: they design boundary
curvature, explicitly *"rather than the complex cut patterns"*.

Both stated assumptions are contradicted. Not rigid:

> "Upon stretching, the boundary ribbon starts bending and compresses the enclosed discrete
> ribbons to induce their out-of-plane buckling"
> "our approach requires the application of external stretching forces to remain the
> deployed shapes ... due to the fully reversible elastic deformation"

And there is no hinge model at all. The sheet is continuous and the analysis is continuum
FEA: *"the PET sheets ... were modeled as linear elastic, isotropic material"*, meshed with
*"solid quadratic tetrahedral elements (C3D10H)"*.

**Remedy.** Remove from the l.326 cluster. If wanted, cite it separately as an
elastic-buckling counterexample, which strengthens the novelty framing rather than diluting
it: Hong et al. are among the few prior works that do go through elasticity.

---

## 6. Rotating tiles and etymology

### l.300 `grima2000` -- SUPPORTED, with a caveat

> "This is based on an arrangement involving rigid squares connected together at their
> vertices by hinges as illustrated in Fig. 1."

"Meet at their corners" is verbatim. "Quadrangular tiles" is fair because the paper
generalises: *"this type of behavior may also be achieved from the more general case
involving hinging parallelograms or in networks built with different sized squares"*.

Caveat: the paper contains **no sentence** stating that adjacent squares rotate in
*alternating* directions, and never uses "void" or "pore". Both rest on Fig. 1 and Fig. 2
and implicitly on Eq. 1. `grima_triangles_2006` states it in prose if a prose-level source
is wanted: *"the squares or triangles remain rigid but rotate (hinge) relative to each
other"*.

### l.301 `grima_triangles_2006` -- SUPPORTED

> "the simplest unit cell for this 'rotating triangles' tessellation is a parallelogram
> such as ABCD ... This unit cell ABCD contains two triangles (AEF and DEG) at an angle
> theta (=angle FED) to each other."

Nuance: this is a mechanics derivation, not the first proposal. The rotating-triangles
picture already appears as Fig. 3 of `grima2000`. The citation is still correct as placed.

### l.276 `temko_kirigami_1962` -- WRONG SOURCE

> "The English word dates from 1962."

The dictionary of record disagrees. From the OED entry for *kirigami, n.* (first published
2024):

> "What is the earliest known use of the noun kirigami? Earliest known use 1950s. The
> earliest known use of the noun kirigami is in the **1958**, in the writing of M. Joya."

and the entry is listed as "kirigami, n. **1958-**". Merriam-Webster has no entry at all.
Dictionary.com gives a range, "First recorded in 1960-65". American Heritage gives no date.
No dictionary credits Temko; the "Temko coined it" claim traces to Wikipedia citing a
hobbyist site.

The book itself is real and the bib entry is essentially correct (title "paper cutting",
two words, is right; Platt & Munk, New York, 1962, corroborated by three independent
listings, WorldCat OCLC 24461627). But a book published in 1962 shows the word existed *by*
1962, not that it dates from then.

**Remedy.** Either (i) change the claim to "The English word is attested from 1958" and cite
the OED, *s.v.* "kirigami (n.)", first published 2024, `https://www.oed.com/dictionary/kirigami_n`;
or (ii) keep Temko and credit popularisation rather than origin: "The word entered general
English use with Temko's 1962 book", which the book does support and which stays true
alongside the OED date.

Note: the OED quotation itself is behind the paywall, so the 1958 attribution to M. Joya
(probably Mock Joya, *Things Japanese*, Tokyo News Service, 1958) could not be read in full.
The date and the "1958-" range are on the public Factsheet tab.

---

## 7. Applications

### l.289 `zhao_modular_2025` -- PARTIAL

All eight occurrences of "actuator" in the paper denote the *external* device driving the
specimen ("the actuator rotates by -83 degrees", "Linear actuator and displacement
transducer"). The assembly is what gets actuated, and "actuator" appears nowhere in its own
list of applications.

Their own application is a robot body:

> "an untethered robotic metamaterial emerges, capable of both locomotion and transformation
> between various states, that is, locomotion mode, transformer mode and
> transformer-locomotion mode"

so this source supports the manuscript's *next* clause better than the one it is attached to.

**Remedy.** Recast as "modular assemblies driven by single-degree-of-freedom actuation", or
move the key to the soft-robot clause and double-cite it with `wu_modular_2024`.

### l.289 `wu_modular_2024` -- SUPPORTED

> "we report a modular soft Kresling origami crawling robot enabled by electrothermal
> actuation."

Origami, folded from a laser-cut PET sheet then assembled, which fits the paragraph's
"printed or assembled" framing exactly. Also the manuscript's own material.

### l.291 `li_origami_2024` -- PARTIAL

Two of the three attributed applications are there, in one sentence:

> "Deployable solar panels for spacecraft, kinetic building facades and paper packaging are
> good examples of origami and kirigami kinematics put to use."

The manuscript's "kinetic facades" is a near-verbatim lift from it. But **biomedical devices
are absent**: `biomed|medic|health|stent|implant|surgic|drug|tissue` returns zero hits
across the full text. The source is also a themed-issue *introduction*, not a review.

**Remedy.** Drop "biomedical devices", or add a source for it.

### l.351 `deoliveira_experimental_2023` -- UNSUPPORTED on both halves

> "Bending a thin ligament out of plane costs less than deploying it in plane, an effect
> used to manufacture beams from flat stock."

(a) There is no in-plane-versus-out-of-plane comparison anywhere. The paper's only force
comparison runs the other way, between deployment load and the *deployed* unit's buckling
capacity: *"The buckling load of the kirigami units ranged from 10 to 25 times the load
required to deploy the system."* Their system has no in-plane deployment mode to compare
against.

(b) The artefact is a **space frame**, not a beam:

> "Planar space frame versions of the rotational kirigami system have been evaluated as
> appropriate for sculptural and facade applications [26], but design changes are required
> for structural applications."
> "Such development could enable the use of this system for practical purposes, including
> civil engineering applications such as slabs and walls."

"Beam" occurs once in the paper, in an imperfection-sensitivity analogy.

There is also a mechanism-class mismatch. Their deployment *is* the out-of-plane mode, by
design, driven by pulling a hub out of plane, with hinges yielding plastically on purpose.
The manuscript's sheet deploys in plane and treats out-of-plane ligament bending as the
parasitic mode. Material is 1.897 mm A1008 carbon steel, waterjet cut; the pattern is
spin-valence, not rotating squares.

**Remedy.** Rewrite to what the paper shows, for example: "cut steel sheets can be pulled
out of plane into stiff space frames at a load one to two orders of magnitude below the
deployed frame's buckling capacity". If the energetic comparison is wanted, it needs a
different source or the manuscript's own oracle.

---

## 8. Differentiable physics

### l.358 `lavin_simulation_2021` -- WRONG SOURCE

This citation carries the manuscript's definition of its central method:

> "Differentiable physics denotes the implementation of a physical simulator whose output is
> differentiable with respect to its inputs. The derivative is obtained by automatic
> differentiation of the solver, or by the implicit function theorem applied to its
> solution."

The source never defines "differentiable physics". The term occurs twice in about 100 pages,
both non-definitional: once describing Brax, once inside a reference title. Its term of art
is differentiable *programming*, a property of programs generally:

> "Differentiable programming (DP) is a programming paradigm in which derivatives of a
> program are automatically computed and used in gradient-based optimization in order to
> tune program to achieve a given objective."

The implicit function theorem is not mentioned anywhere in the source.

### l.366 `lavin_simulation_2021` -- PARTIAL

Supports the weaker claim that differentiable simulators give gradients usable for design
optimisation and optimal control:

> "Both works point to the value of differentiable simulators, providing gradients that can
> be used for design optimization, optimal control tasks, and inverse problems."

But its dedicated inverse-design treatment enumerates Bayesian and variational methods, deep
generative models, invertible models and surrogate-based optimisation, never gradient descent
through a differentiable physics solver.

**Remedy for both sites.** Replace with **Xue, Liao, Gan, Park, Xie, Liu & Cao, "JAX-FEM: A
differentiable GPU-accelerated 3D finite element solver for automatic inverse design and
mechanistic data science", *Computer Physics Communications* 291:108802 (2023), doi
10.1016/j.cpc.2023.108802.** Peer-reviewed, in mechanics, and it contains both halves of the
l.358 definition plus the l.366 use:

> "Beyond efficiently solving forward problems, JAX-FEM employs the automatic differentiation
> technique so that inverse problems are solved in a fully automatic manner without the need
> to manually derive sensitivities."
> "where the existence of the derivative dU/dtheta is justified by the **implicit function
> theorem** [37] under certain mild conditions."
> "Inverse design problems are of great interest in various engineering applications. ...
> Successfully computing the 'sensitivity' ... is key to gradient-based optimization
> algorithms."

A PDF is already in the author's Zotero at `2E8NTKH3/`.

Correction to an earlier suggestion: Sapienza et al., "Differentiable Programming for
Differential Equations: A Review" (arXiv:2406.09699) is **also a preprint**, not the
peer-reviewed review it appeared to be. It is a good second cite for the definition, not a
replacement.

`lavin_simulation_2021` is itself never peer-reviewed (no journal-ref on arXiv; only a
DataCite DOI), and the local PDF is v2 (2022) while the bib says 2021.

### l.364 `kochkov_machine_2021` -- SUPPORTED

> "we design algorithms that accurately solve the equations on coarser grids by **replacing
> the components most affected by the resolution loss with better performing learned
> alternatives** ... We train the solver inside a standard numerical method for solving the
> underlying PDEs **as a differentiable program** ... This allows for **end-to-end gradient
> based optimization of the entire algorithm**."

Trained through 32 unrolled solver steps.

### l.373 `bordiga_automated_2024` -- PARTIAL

Three of four elements hold. The simulator is dynamic and differentiable, with adjoint
gradients; the design variables are the tile vertex coordinates; focusing is demonstrated.

"Absorbs" is not. The word **"absorb" occurs zero times** in the paper. The second task is
dynamic protection:

> "we deploy our design strategy to search for a metamaterial architecture with the ability
> to **maximize kinetic energy** at a target location Omega_t for an applied precompression
> of eps = eps_1 and **to minimize it** for eps = eps_2."

Also relevant to the lead-in "Two applications to kirigami exist": they never call their
system kirigami, and it is *"3D-printed polylactic acid (PLA) units and thin polyester
plastic shims"*, an assembly, not a cut sheet.

**Remedy.** "focuses an elastic pulse on a target region, or shields that region from it".
Consider "hinged-tile" instead of "kirigami" for their system.

### l.748 `bordiga_automated_2024` -- SUPPORTED

See section 3. This is the one key at that site that does state the model.

### l.288 and l.376 `wang_physics-aware_2023` -- both SUPPORTED

Every element of both claims is in the source in nearly the same words. Remote actuation:
*"A uniform magnetic field B = B e_B ... will then impart distributed magnetic torques in
kirigami to achieve shape morphing."* Analytical rather than FE, with FEA used only for
validation. Kinematics plus stored energy, differentiated by chain rule. And joint design
variables:

> "we only need to use the vertical component (y-component) d_i of a unit magnetization
> vector to determine the orientation of the magnetization panel, **which is combined with
> the coordinates (x_i, y_i) of the nodes as the design variables**"

---

## 9. Form finding

### l.319 `adriaenssens_shell_2014` -- SUPPORTED

> "Form finding denotes the determination of a shape as the equilibrium configuration of a
> structure under prescribed action."

The book's own definition, from the four editors' unnumbered Introduction, p. 2:

> "The process of designing form-found shapes is called form finding or shape finding, where
> the former term is used in this book."
> "We define the concept as: Form finding is a **forward process in which parameters are
> explicitly / directly controlled to find an 'optimal' geometry of a structure which is in
> static equilibrium with a design loading**."

"Static equilibrium with a design loading" is the manuscript's "equilibrium configuration
under prescribed action". The definitional core is the book's own.

The book also supports the manuscript's *next two* sentences, which the citation does not
currently reach. One page later:

> "Structural optimization is an **inverse process** in which parameters are implicitly /
> indirectly optimized to find the geometry of a structure such that an objective function
> or fitness criterion is minimized."

That forward/inverse contrast is exactly what the manuscript restates as "The forward
problem maps design parameters to an equilibrium state. The inverse problem selects those
parameters." The citation could reasonably cover all three sentences.

**Reachability caveat, recorded for honesty.** The text was obtained as Google Books
search-within-volume snippets (volume `PAIkAwAAQBAJ`), not as continuous page text; the OCR
spacing above is Google's. Each fragment is verbatim and the p. 2 folio is inferred from the
Taylor & Francis record (Introduction = pp. 1-4). No full text exists on Internet Archive,
and routledge.com and taylorfrancis.com refuse automated fetch. Confirm against a physical
copy before submission.

Note the definition is in the editors' front-matter Introduction, not in a numbered chapter.
Chapter 1 is "Exploring shell forms" (Ochsendorf and Block); Chapter 2 is "Shaping forces"
(Ney and Adriaenssens).

See Appendix A: the four names are **editors**, not authors.

### l.369 `schek_force_1974` -- PARTIAL

> "In form finding, inverse problems on bar and cable systems have been posed as
> gradient-based optimisation through a differentiable force-density solver
> [schek_force_1974, pastrana_jax_2023]."

Schek is the origin of the force-density solver, and the paper does contain an inverse
treatment, which is more than expected. From the abstract:

> "A new method for network analysis, the 'force density method', is presented based upon
> the force-length ratios or force densities which are defined for each branch of the net
> structure. The force densities are very suitable parameters for the description of the
> equilibrium state of any general network: the associated node coordinates of the structure
> are obtained by solving one single system of linear equations. **If the free choice of the
> force densities is restricted by further conditions, the linear method is extended to a
> nonlinear one, and a damped least squares approach is then introduced.**"

Two things the sentence claims are absent from the paper. A full-text search returns **zero
hits** for `gradient`, `optimiz`/`optimis`, `Newton`, `automatic differentiation`,
`objective function` and `inverse problem`. And the prescribed data are never a target
geometry: the constrained problems of his sections 3.3 to 3.6 prescribe **branch lengths,
branch forces and unstrained lengths** ("We suppose that the lengths of the first r branches
(bars or cables) have prescribed values"). Shape approximation to a target is not posed.

The co-cited `pastrana_jax_2023` characterises Schek the same way: *"The FDM is a **forward**
physics solver expressed as a function f(theta, G) = U."*

The one element genuinely close to the claim, and the honest hook if Schek is kept at this
site, is that he hand-derives implicit-function-theorem sensitivities of the equilibrium
solve with respect to the design parameters (section 3.2, p. 124):

> "The real problem is a representation of the Jacobian matrices of the coordinates
> referring to the force densities, namely dx/dq, dy/dq and dz/dq. ... **This bottleneck may
> be bypassed by using implicit derivatives:** We look at the equations of equilibrium (5)
> for the x-coordinates ... Any changes dq and dx should leave untouched the state of
> equilibrium."

giving dx/dq = -(C^t Q C)^{-1} C^t U. That is the same mathematical object a differentiable
solver produces by AD, but it is analytic and it feeds a damped least-squares step, not a
descent on a loss.

**Remedy.** Split the attribution. Cite Schek for the force-density method and its
damped-least-squares extension, and `pastrana_jax_2023` for the differentiable,
gradient-based form. For example: "In form finding, inverse problems on bar and cable
systems date to the force-density method [schek_force_1974] and have since been posed as
gradient-based optimisation through a differentiable solver [pastrana_jax_2023]."

One further caution: "the direct approach" is **not** Schek's term (zero hits). That is
Linkwitz's later retroactive naming, and should not be attributed to this paper.

### l.369 `pastrana_jax_2023` -- SUPPORTED

> "We introduce JAX FDM, a differentiable solver to design mechanically efficient shapes for
> 3D structures conditioned on target architectural, fabrication and structural properties.
> Examples of such structures are domes, cable nets and towers. JAX FDM solves these
> **inverse form-finding problems by combining the force density method, differentiable
> sparsity and gradient-based optimization**."

"Bar and cable systems" is exact: *"3D structures modeled as pin-jointed bar networks"*.
Every element of the claim is in one sentence of the abstract.

### l.371 `pastrana_realtime_2025` -- SUPPORTED

All three elements verify.

The network predicts the **parameters** of the solver, not the shape:

> "Mechanical decoder The decoder gives the latent space of our model a physical meaning
> since it represents the inputs of a mechanical simulator. ... the relationship between the
> stiffness values q, the shape X, and the loads P must satisfy K(q) X - P(X) = 0."

The solver is the same one, so "that solver" resolves correctly:

> "Here, we utilize the force density method as our simulator (Pastrana et al., 2023b)."

(their 2023b is `pastrana_jax_2023`). And the interactive rate is measured, not asserted:
their timing table reports **0.3 +/- 0.1 ms** for the model against **5570.4 +/- 1688.6 ms**
for direct optimisation, described as *"comparable accuracy to direct optimization but in
real time"* and *"up to four orders of magnitude faster"*.

---

## 10. Surrogates and machine learning

### l.402 `le_computational_2015` -- SUPPORTED

The potential, not the stress, is the fitted object, and they explicitly decline the
alternative:

> "the effective potential, which can be derived to obtain the macroscopic strain and
> tangent tensors, is replaced by a NN approximation"
> "An alternative, not pursued in this work, is to directly compute and store the effective
> stress components ... However ... this is not necessary as the effective potential is very
> smooth, and its derivatives are then very accurate."

Gradient and Hessian are the manuscript's pair: sigma-bar = d w-bar / d eps-bar (Eq. 16),
C-bar^tan = d^2 w-bar / d eps-bar d eps-bar (Eq. 20). And it replaces the nested solve:

> "in contrast with FE2 method, each new macroscopic simulation does not imply new
> microscopic calculations, as the effective potential is constructed once for all."

One divergence, which the manuscript already defers to `sec:truncation`: their fine scale is
nonlinear **elastic**, small-strain, path-independent. Worth checking that
`sec:truncation` names path-dependence explicitly, since that is precisely where Le et al.'s
guarantee lapses. No speed-up number is ever stated in the source; do not attribute one.

### l.1008 `vlassis_sobolev_2021` -- SUPPORTED

Their own justification for the H1/H2 loss is the manuscript's sentence:

> "the convergence of the stored energy measured by the L2 norm does not guarantee the
> quality and even the existence of the stress and elastic tangent operators stemmed from
> the learned energy function."

with the learned object an energy whose gradient is the stress and whose Hessian is the
tangent (their Eq. 2), and Eqs. (4)-(5) the direct precedent for `eq:sobolev`.
See Appendix A: the DOI is wrong.

### l.329 `deng_inverse_2022` -- SUPPORTED as written, misplaced in the paragraph

All three elements verify: 16-dimensional vertex-perturbation input, 100-dimensional
stress-strain output, custom (mu+lambda) evolution strategy run on the trained network.

But "kirigami" appears **zero** times in the paper, and the samples are cast in 3D-printed
molds, not cut: *"two elastomeric metamaterials comprising a 10 x 8 array of hinged
quadrilateral units"*, and *"J.W. printed the molds for samples fabrication."* The sentence
sits in a paragraph whose stated subject is kirigami.

**Remedy.** Qualify, e.g. "for a hinged-quadrilateral metamaterial", so a referee who opens
the reference is not surprised by the absence of a cut sheet.

### l.330 `yazdani_reinforcement_2026` -- PARTIAL, and under-weighted as a competitor

Cut patterns and prescribed silhouettes both verify, down to the cutter: *"Generated layouts
were exported to DXF and laser-cut in 50 micron polymeric sheets"*.

"A reinforcement-learning policy produces cut patterns" overstates the RL contribution. The
generator is a conditional flow-matching model; RL enters as a reweighting of the
flow-matching loss. The headline number comes from the supervised prior *before* any RL:
*"a single sample from the pretrained OT-CFM prior reached 94.2% sIoU"*, with GRPO lifting it
to 94.91%.

More importantly, this is a much closer competitor than its placement in an ML list suggests.
It works on the manuscript's own tessellation family:

> "Only compact reconfigurable parallelogram quad kirigami is considered in this study (Choi
> et al., 2019). In this case, the deployment angle field must follow a checkerboard pattern
> (Dudte et al., 2023) and is fully set by a global deployment parameter phi."

with a per-void side-length ratio field structurally equivalent to the manuscript's per-cut
`z`. And it disclaims mechanics in its own limitations section:

> "The geometry-based forward simulator does not model forces, material nonlinearity, contact
> along the full deployment path, or fabrication tolerances. ... the reported results should
> be read as strong evidence for fast inverse design within this constrained geometry class,
> simulator, and evaluator, rather than as a full mechanics-aware inverse design."

**Remedy.** Reword to "A conditional generative model, fine-tuned by reinforcement learning,
produces cut patterns for prescribed silhouettes." Then consider citing it again in the
differentiator sentence: a 2026 paper on the same tessellation family that explicitly
disclaims mechanics is the sharpest available foil for "inverse design through mechanics".

---

## 11. Tooling

All four SUPPORTED.

### l.643 `dhondt_finite_2004` -- SUPPORTED

This book is the CalculiX project's own recommended citation. From dhondt.de:

> "For a reference describing the theory behind CalculiX CrunchiX the user is referred to:
> Dhondt, G. The Finite Element Method for Three-Dimensional Thermomechanical Applications,
> Wiley, 2004."

C3D15 is CalculiX's quadratic 15-node wedge element (*"C3D15 (15-node quadratic triangular
prism element)"*), which is what `nff/rve/ccx_solver.py:136` emits. The manual attributes its
incremental-plasticity routine to this book, which is the physics the manuscript exercises.

Two notes. The C3D15 *shape functions* are attributed by the manual to the ABAQUS Theory
Manual, not to Dhondt, so do not later cite this book for them. And adding
`note = {Software version 2.23}` would make the version traceable, since a 2004 book cannot
document a 2025 release.

### l.1046 `bradbury_jax_2018` -- SUPPORTED as a citation, one metadata error

The entry matches JAX's own recommended BibTeX in title, year and URL, and `jax-ml/jax` is
the current canonical URL. JAX 0.6.2 exists (PyPI, 2025-06-17) and matches the installed
`kgnn_mac` environment. See Appendix A: an author is missing.

### l.1046 `frostig_compiling_2018` -- SUPPORTED

The correct and only peer-reviewed paper describing JAX's tracing and compilation, and
JAX's README points to it as the paper companion to the software entry. See Appendix A on
the venue name.

### l.1047 `schoenholz_jax_2021` -- SUPPORTED

The facility the manuscript uses is described in the paper:

> "To facilitate the construction of energy functions from pairwise interactions, JAX MD
> provides several higher-order functions that take functions that act on pairs of elements
> and transforms them to operate on an entire system ... U = smap.pair(d, u),
> U = smap.neighbors(d, u), and U = smap.bond(d, u)"

and the paper sanctions use outside molecular dynamics: *"These primitives are flexible
enough that they can be used to scale up workloads outside of molecular dynamics."* The
manuscript's phrasing scopes the credit to assembly, which is correct: `jax_md` is imported
in exactly one module, `nff/stages/physics/energy.py:9`.

Note: the JAX MD repo asks for the NeurIPS 2020 entry instead. IOP's own footnote confirms
the two are the same work, the JSTAT article being the updated version. Either is
defensible. The local Zotero PDF is the NeurIPS version, not the JSTAT article the entry
points at.

---

## Appendix A: bibliographic metadata errors

Objective errors, independent of claim support.

| Key | Error | Correction |
|---|---|---|
| `yazdani_reinforcement_2026` | Title is the arXiv preprint's; volume missing | Title: "Reinforcement learning for inverse **silhouette** design and rapid laser cutting of **compact parallelogram quad** kirigami prototypes"; add `volume = {268}` |
| `vlassis_sobolev_2021` | DOI `10.1016/j.cma.2020.113695` returns 404 | `10.1016/j.cma.2021.113695` (verified live) |
| `bradbury_jax_2018` | Author list has 11 of 12 names | Insert `Katariya, Yash and` after `Johnson, Matthew James and` |
| `dang_compatibility_2021` | Superseded preprint | Published as *Phys. Rev. Lett.* **128**(3):035501 (**2022**), doi 10.1103/PhysRevLett.128.035501. Published title drops the leading "A" |
| `grima2000` | Page range truncated; DOI missing | `pages = {1563--1565}`; add `doi = {10.1023/A:1006781224002}` |
| `zheng_modelling_2023` | Issue number missing | Add `number = {2272}` |
| `frostig_compiling_2018` | `booktitle` names a volume that does not exist | No *Proceedings of Machine Learning and Systems* 2018 volume exists; the paper's own header reads "SYSML'18". Use `booktitle = {SysML Conference}`, `address = {Stanford, CA}` |
| `dhondt_finite_2004` | Version not traceable | Optional: `note = {Software version 2.23}` |
| `lavin_simulation_2021` | Cited year is v1, local PDF is v2 | Moot if replaced per section 8 |
| `adriaenssens_shell_2014` | Uses `author`; the four names are **editors** | Use `editor = {...}`. Three independent authorities agree: Library of Congress MARC (LCCN 2013028638) 245$c reads "edited by Sigrid Adriaenssens, Philippe Block, Diederik Veenendaal, and Chris Williams", with all four in field 700 and none in 100; Crossref `10.4324/9781315849270` returns `author: None` with all four as `editor`; Routledge lists "Edited By". The manuscript's own cited `pastrana_realtime_2025` gets this right in its bibliography. Since the cited text is the Introduction (pp. 1-4, T&F chapter DOI `10.4324/9781315849270-1`, of which the four *are* the authors), `@incollection` with locator p. 2 is the precise form. Also: ISBN 978-0-415-84060-6 is the **paperback**; hardback is 978-0-415-84059-0 |

Verified correct, despite suspicion: `liu_quasicrystal_2022` (a report that it carried a
PNAS DOI was checked against the file and is false), `schenk_origami_2011` pages 291-304
(the widely circulated "293-305" is the wrong one), `choi_programming_2019`,
`dudte_additive_2023`, `dang_theorem_2021`, `dang_kirigami_2025`, `chaudhary_geometric_2023`,
`hong_boundary_2022`, `grima_triangles_2006`, `filipov_bar_2017`, `liu_nonlinear_2017`,
`rafsanjani2017`, `lobkovsky1996`, `witten2007`, `kochkov_machine_2021`,
`bordiga_automated_2024`, `wang_physics-aware_2023`, `zhao_modular_2025`, `wu_modular_2024`,
`li_origami_2024`, `deoliveira_experimental_2023`, `le_computational_2015`,
`deng_inverse_2022`, `schoenholz_jax_2021`, `temko_kirigami_1962`.

---

## Appendix B: entries in the bib with no `\cite`

`bendsoe_generating_1988`, `li_visualizing_2018`, `lobkovsky1995`, `suh_differentiable_2022`.

`lobkovsky1995` is deliberately uncited per `memory/project_manuscript.md`: the
crumpling-ridge reading was discarded in favour of the developable cone. Section 4 confirms
that decision was right, and extends it: `lobkovsky1996` is ridge-only too and should join
it in the uncited set.

---

## Appendix C: manuscript defects found during the audit

Not citation problems, but found while checking sources and worth fixing.

1. **l.2100 is wrong physics.** "its displacement field decays only as 1/r" misstates which
   field decays. For a developable cone the displacement is conical and grows *linearly*
   with r; the **curvature** goes as 1/r and the stresses as 1/r and 1/r^2. Both Witten 2007
   and Liang & Witten 2005 are explicit on this. The sentence should say curvature or stress.

2. **l.2100 "There is no Saint-Venant decay to appeal to"** is unsupported as literal
   wording. Neither cited source uses the term, and Witten reports exponential decay for the
   ridge flank. The defensible form is the energy statement itself: the d-cone bending
   energy grows as log(R_p/R_c) and therefore has no convergent outer limit, which is exactly
   what the manuscript's own F_s^max = 303 - 41.2 ln(r_win) measurement shows.

3. **l.300 typo and a sentence with no main verb.** "quadragular" should be "quadrangular".
   And "In rotating squares, quadragular tiles that meet at their corners and turn in
   alternating directions as the voids between them open." has no main clause; the relative
   clause never resolves.

## Addendum 2026-08-10: three sources added with the Discussion/Conclusion overhaul

Each verified against publisher metadata (title, authors, venue, volume, pages, DOI)
before insertion. None has yet been read in full text against its claim site at the
depth of the 2026-08-09 audit; the claims they carry were written to stay inside
their abstracts.

| Key | Claim it carries | At | Verified |
|---|---|---|---|
| `mozaffar_deep_2019` | recurrent networks represent path-dependent plasticity via an internal state | Sec. sec:path-dependence | PNAS 116(52):26414-26420, doi:10.1073/pnas.1911815116 |
| `deng_nonlinear_2021` | transition waves and solitons in flexible mechanical metamaterials, review | Sec. sec:extra-dof | J. Appl. Phys. 130(4):040901, doi:10.1063/5.0050271 (Deng, Raney, Bertoldi, Tournat) |
| `behler_generalized_2007` | learned interatomic potentials condense electronic-structure solves into an energy whose gradient is the force | Sec. sec:beyond-planar | PRL 98(14):146401, doi:10.1103/PhysRevLett.98.146401 |

Also newly cited at existing keys, sites not previously carrying them:
`vlassis_sobolev_2021` (thermodynamics-informed internal variables, sec:path-dependence),
`coulais_characteristic_2018` (characteristic length of mechanism metamaterials,
sec:extra-dof -- note this key was previously cited for the three-spring model, a
different claim of the same paper), `czajkowski_conformal_2022` (conformal limit,
sec:extra-dof), `block_thrust_2007`, `majmudar_contact_2005`,
`adriaenssens_shell_2014`, `schoenholz_jax_2021`, `xue_jaxfem_2023`,
`deoliveira_experimental_2023`, `dang_theorem_2021` (sec:beyond-planar).

Historical facts in the Conclusion tribute (1955 Living National Treasures and their
specialisms; 1993 collective designation; Suzuka districts) were verified against the
Ritsumeikan ARC "Ise Katagami Stencils" exhibit (Google Arts & Culture) and the
Wikipedia "Ise katagami" article, 2026-08-10. They are stated as history, uncited, in
the manner of the Introduction's history paragraph. The living-artist sentences
(Nahoko Kojima, Masayo Fukuda) were checked against the artists' public profiles.

---

## Addendum 2026-08-12 -- `sec:ideal-deployment-range` dissolved

The subsection "Range of validity of the ideal deployment" and its figure
`fig:ideal-deployment` were deleted. Three consequences for this audit.

**`schenk_origami_2011` is uncited again, as the 2026-08-09 audit intended.** The key was
reintroduced on 2026-08-12 at the single site `sec:ideal-deployment-range`, carrying "A
measurement of shape therefore cannot certify the energy." The review pass then cut the
one clause written to make it defensible ("The modes a mechanism is soft in are distinct
from the ones that store its energy"), leaving a bare citation on a claim the source does
not make. This audit had already recorded the key as UNSUPPORTED ("the paper is origami
only; 'kirigami' and 'cut' never appear") and WRONG SOURCE. The deletion repairs it. The
claim itself survives, unattributed and carried by our own measurement, at the close of
`sec:solution-set`. **Do not reintroduce this key.**

**`coulais_characteristic_2018` no longer carries the characteristic-length claim.** The
2026-08-10 addendum put that site on the not-yet-verified watchlist and recorded its home
as `sec:extra-dof`, a subsection since dissolved; the claim had migrated to
`sec:ideal-deployment-range`. The key now sits only on its two Methods sites (the
three-spring model), which are audited and SUPPORTED. This also settles a standing
tension: `memory/project_force_pocket.md` warns "Do not use this to adjudicate
coulais_characteristic_2018, which is about other systems."

**`czajkowski_conformal_2022` is cited again, at a new site.** It was recorded above as
cited at `sec:extra-dof` and fell out when that subsection went, leaving it a dead
bibliography entry. It now sits in `sec:methods` beside `zheng_modelling_2023`, on the
claim that a local disturbance decays within a few cells of an auxetic cell. That is the
paper's own result: for a boundary dilation `w0 cos(k0 x)` it gives
`alpha = exp[w0 e^{-k0 y} cos(k0 x)]`, decay length `1/k0` set by the boundary's variation
wavelength. **Not yet read in full text against this site** -- the claim was written to
stay inside the abstract and the published closed form. Watchlist.

**Why the subsection went, in one line:** every headline in it was a published prediction
cited to none of the papers that make it, it was referenced by nothing outside the
Discussion's own preamble, and at 1073 words plus a full-width plate it was the most
expensive item in Section 4. The measurements were sound and the data and scripts remain
on disk.

---

## 2026-08-17 — bibliography pruned to the cited set

`paper/sn-bibliography.bib` held 67 entries against 54 cited keys. It now holds exactly the 51
entries the document cites. Twenty entries were removed. **This section is the record; the
entries themselves are recoverable from
`docs/paper_backup/sn-bibliography.pre-refbalance-20260817-1327.bib`.**

**Removed because never cited** (13). Several were earlier deliberate removals recorded above,
and the instruction not to reintroduce them still stands, in particular `schenk_origami_2011`
and `filipov_bar_2017` (wrong source for `eq:rom`; see the entry above), plus
`lavin_simulation_2021`, `temko_kirigami_1962` and `lobkovsky1996`:

`bendsoe_generating_1988`, `deng_nonlinear_2021`, `filipov_bar_2017`, `grima_connectivity_2005`,
`hong_boundary_2022`, `lavin_simulation_2021`, `li_visualizing_2018`, `liu_nonlinear_2017`,
`lobkovsky1995`, `lobkovsky1996`, `schenk_origami_2011`, `suh_differentiable_2022`,
`temko_kirigami_1962`.

**Removed with the citation-balance pass** (7). Each was used exactly once, supporting an aside
the paper never returned to. Cut to concentrate the reference list on load-bearing sources:

| key | what it supported |
|---|---|
| `dias_kirigami_2017` | actuators exemplar, applications list |
| `babaee_kirigami_2021` | stents exemplar, applications list |
| `wu_modular_2024` | second soft-robot ref beside `zhao_modular_2025` |
| `grima_triangles_2006` | "rotating triangles on a triangular grid" sentence |
| `liu_quasicrystal_2022` | aperiodic-patterns sentence |
| `chaudhary_geometric_2023` | same sentence |
| `peters_characterization_2005` | second granular-contact ref beside `majmudar_contact_2005` |

⚠ `deoliveira_experimental_2023` and `pastrana_realtime_2025` were on the proposed cut list and
were **kept at the author's instruction** (2026-08-17). Do not re-propose them.

**Added** (4), each a standard method the manuscript uses that carried no attribution:
`liu_limited_1989` (L-BFGS, the forward solver), `hartley_multiple_2004` (planar homography and
rectification, Sec. 4.1, which previously had zero citations), `hill_elastic_1963` (the
representative volume element), `astm_d882` (the coupon standard the appendix names).

⚠ `paper/Article Primary sources.bib` is a five-entry orphan that the document does not load; all
five keys are duplicated in `sn-bibliography.bib`. Do not add references to it.
