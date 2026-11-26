# Unit-cell families against actuation: a structure–property map that comes back flat

**Status:** standalone analysis, 2026-08-21. Nothing in `paper/` was touched.
**Question asked:** in Fig. 6 (`fig:inverse-design-matrix`) the optimised tilings show obvious
families of repeating unit cells. Can we say which family suits which actuation, and thereby give
each family a property?

**Answer, in one line:** the families are real as *geometry* and real as *spatial regions*, but they
do not carry the load differently and they are not selected by the actuation. A unit cell's own
shape and size predict essentially nothing about the force it transmits.

---

## 1. What a unit cell is, and why the taxonomy is complete

With `A, B` the two neighbouring cut endpoints the collinearity system links a cut to, and
`L = ‖A − B‖`, the cut's own endpoints are `rA + (1−r)B` and `(1−r)A + rB`. The void's four sides
are therefore `rL, (1−r)L, rL, (1−r)L` in order, and deployment rotates rigid panels, so it opens
the angle between those sides without changing any of them:

> the deployed void is an **exact parallelogram of sides `rL` and `(1−r)L` at every opening angle.**

Gated in `tests/test_void_geometry.py:70-79` to 1e-11, and re-derived here in
`tests/test_cell_families.py`.

Two consequences:

- **A unit cell has exactly two geometric degrees of freedom**, `(r, L)`, equivalently the two side
  lengths `(long, short)`. The published descriptor is a rotated `(r, L)` plane. So a "family" is a
  region of a two-dimensional plane and the taxonomy is *complete*, not heuristic. Kites, darts and
  trapezoids are unreachable for the void: it can only ever be a parallelogram, or at `r = 1/2` a
  rhombus. This is a sharper footing than any comparable inverse-design study has (Deng et al. 2022
  parameterise their 2×2 cell with sixteen numbers).
- **`r` is a local design freedom; `L` is not.** Only `r_ij = sigmoid(z_ij)` is chosen per cell.
  `L_ij` falls out of the global collinearity solve, so it carries the whole sheet. The two are
  reported separately throughout and never summed into one number.

`r` is recoverable from geometry only up to `r ↔ 1−r`, since the two sides are interchangeable;
the branch in `(0, 1/2]` is used.

## 2. The prediction, stated before measuring

From the parallelogram lemma, a cell of parameters `(r, L)` at opening `ψ` has

```
A(ψ)   = r(1−r) L² sin ψ        void area
dA/dψ  = r(1−r) L² cos ψ        stroke
```

Every cell of a sheet shares one `ψ` to within half a degree (measured: 0.37–0.58° on four of the
five figmat sheets; see the caveat in §7). So **`gain = r(1−r)L²` is the whole kinematic gain of a
cell**, maximised at `r = 1/2`, vanishing as the cell degenerates to a slit.

The pre-registered prediction was: *the family an actuation recruits should be the one whose
gain and anisotropy match what that actuation needs* — high-gain cells where stroke is wanted,
low-gain cells where force must be transmitted.

The three published families, with their gains:

| family | centroid (mm) | `L` (mm) | `r` | gain (10³ mm²) |
|---|---|---|---|---|
| 0 | 725 × 297 | 1022 | 0.291 | 215.5 |
| 1 | 491 × 360 | 850 | 0.423 | 176.5 |
| 2 | 411 × 226 | 637 | 0.355 | 93.0 |

## 3. The corpus, and one gate that failed

`nff/scripts/diagnostics/cell_family_corpus.py` re-solves each design from `config.yaml` +
`best_params.pkl` and writes one row per (design, unit cell): the cell's geometry, its family, and
the response of the four panels and four hinges it owns.

**525 cells = 15 sheets × 35 disjoint unit cells** on a 7×5 lattice, 140 of 150 panels covered.

⚠ **The fifteen sheets are not one experiment.** The grip configs carry no `target_half_w`, so
`run_closed` takes the *auto-fit* branch: the target is the untrained deployed bounding box,
scaled, anchored at the **clamp centroid**. Measured across the ten grips:

| | target centre x | centre y | half-width | half-height |
|---|---|---|---|---|
| figmat (5) | spread **0.000** | 0.000 | 0.000 | 0.000 |
| grips (10) | spread **5.86** | 3.24 | 0.76 | 1.40 |

On a sheet of half-width ≈ 11, the grips' target centre moves by 5.86. Actuation and target are
entangled *by construction*, because the target follows the clamp. So:

- **figmat only** licenses any between-sheet claim ("actuation X recruits family Y").
- **all fifteen** are admissible for within-sheet claims ("family X carries more compression than
  family Y *of the same sheet*"), which the target cannot bias — and there the grips are ten extra
  independent replicates rather than a confound.

All fifteen do share `r_init = 0.45` and `init_noise = 0`, so every run starts from the identical
uniform design. Families are **fitted on the figmat block** and the grips **labelled** with that
fitted model, so a family means one shape everywhere and the definition is anchored on the block
whose target is controlled.

**Reproduction gate passed before anything new was measured.** On the five figmat rows the
refactored pipeline reprints the published plate exactly: centroids 725×297 (n=49), 491×360 (n=89),
411×226 (n=37); contiguity 1.40 / 1.11 / 1.39 / 1.19 / 1.11; clustered 30/29/23/25/26 of 35. The
regenerated figure is **pixel-identical** to the shipped PDF at 150 dpi (0 differing pixels).

## 4. Arm A — the actuation does not select the family

Composition of the five controlled-target sheets:

| sheet | family 0 | family 1 | family 2 |
|---|---|---|---|
| I Concentrated axial load | 10 | 16 | 9 |
| II Axial and transverse load | 8 | 23 | 4 |
| III Axial load and couple | 12 | 12 | 11 |
| IV Eccentric axial load | 11 | 15 | 9 |
| V Self-equilibrated shear couple | 8 | 23 | 4 |

**χ² = 12.42, p = 0.100** (20 000 draws). The null is that the five sheets are exchangeable **at
each site**: for each of the 35 cell sites the five sheets' labels are shuffled among the five
sheets. A label never moves to a different site, so the spatial coherence the families are already
known to have cannot inflate the test — a free label shuffle over all 175 cells would, and would
manufacture significance.

So the composition differences are **not significant**. Cases II and V have *identical*
compositions (8/23/4) despite being the two most dissimilar actuations in the set (a net transverse
pull against a self-equilibrated couple).

Every sheet does cluster at or above chance — figmat 1.11–1.40×, grips 1.00–1.64× — so the families
form spatial *regions* in all fifteen. The grips cluster slightly harder than figmat on average,
which rules out the region structure being an artefact of the controlled block.

## 5. Arms B and D — cell geometry does not predict what the cell carries

Arm B compares family means within each sheet, after removing a low-order polynomial in lattice
position and standardising inside the sheet, then tests across the fifteen sheets with a paired
sign test. Of 27 comparisons exactly one clears p < 0.05 (family 1 vs 2 on departure from 1-DOF,
p = 0.035), which is what chance alone predicts at that count.

Arm D asks the same question of the *continuous* descriptor, since three families are a coarsening
of two numbers and coarsening costs power. Median within-sheet Spearman over the fifteen sheets,
with the detrend order **swept** rather than chosen — the families occupy spatial regions, so a
high-order detrend can delete the very variation under test, and reporting one order would leave
that indistinguishable from a physical null:

| response | vs `gain` | vs `aspect` (r) | detrend 0 → 3 |
|---|---|---|---|
| normal force | +0.03 … −0.01 | +0.14 … −0.01 | no consistent sign |
| tangential force | +0.04 … −0.02 | +0.01 … +0.04 | – |
| transmitted force | −0.08 … +0.03 | +0.09 … 0.00 | – |
| shear-dominated share | −0.14 … −0.05 | −0.11 … −0.10 | – |
| tile pressure | −0.10 … −0.09 | +0.03 … −0.03 | – |
| **tile couple** | −0.01 … +0.04 | **−0.23 … −0.12** | **sign consistent, see below** |
| stored hinge energy | −0.08 … +0.06 | +0.08 … 0.00 | – |
| panel rotation | −0.18 … −0.05 | +0.03 … +0.06 | – |
| departure from 1-DOF | +0.18 … +0.05 | +0.04 … −0.02 | – |
| **void area** (wiring check) | **+1.00 … +0.52** | +0.05 … −0.17 | **identity, see below** |

**The wiring check.** At detrend degree 0, `void area` against `gain` returns **ρ = +1.0000 on all
15 of 15 sheets**. That is the closed form of §2 reproducing itself exactly, and it is the proof
that the descriptor, the reduction and the solve are all wired correctly. It is not a result.

**The one candidate.** `tile couple` against `aspect` is negative on 13/15, 13/15, 11/15 and 12/15
sheets at detrend orders 0, 1, 2, 3 (p = 0.007, 0.007, 0.119, 0.035): a cell with a more *elongated*
void (`r` far from 1/2) carries more couple. The sign and rough magnitude survive every detrend
order, which a spurious hit generally does not. But across 50 independent (predictor, response)
pairs one p ≈ 0.007 is roughly what chance yields, so this is **a candidate, not a result**. It
would be consistent with the already-published observation that case III's top-decile moment
carriers are 1.61× median panel area — moment carriage is the one channel with a known geometric
signature.

**This is a bounded null, not weak power.** The same test detects |ρ| ≈ 0.2 at p < 0.01 (the couple
result). Every other channel sits below that. So the honest statement is not "we found nothing" but
**"a cell's own geometry explains less than about 4% of the rank variance of what it carries."**

## 5b. The apparent division of labour is POSITIONAL, not geometric

Pooled over all 525 cells in raw physical units the families look strongly specialised:

| median, pooled | family 0 (725x297) | family 1 (491x360) | family 2 (411x226) |
|---|---|---|---|
| normal force `f_n` (N) | 5.97 | **11.21** | 5.38 |
| tangential force (N) | **8.75** | 3.13 | 6.63 |
| tile couple (N mm) | **0.525** | 0.004 | 0.211 |
| tile pressure (N/mm) | 6.11 | **17.30** | 10.33 |
| panel rotation (rad) | 0.6658 | 0.6696 | 0.6675 |
| void area (mm^2) | 2.10e5 | 1.70e5 | 0.91e5 |

That table invites the conclusion that family 1 is the axial carrier and family 0 the shear-and-
moment carrier. **It does not survive.** With each sheet standardised inside itself but no spatial
detrend, the split is still significant (normal force, families 0 vs 1, p = 0.013; tile pressure,
0 vs 1, p = 0.013). Remove a merely **linear** trend in lattice position and every one of them
vanishes. The chain is therefore: family membership correlates with *where* a cell sits, position
determines the force, and family adds nothing once position is known.

Panel rotation is identical across the three families to four significant figures, which is exactly
what a 1-DOF mechanism must do -- every panel turns by the same magnitude, so no family can be
"the one that rotates".

**Nor do the families occupy distinct positions relative to the actuation.** Mean distance to the
loaded tile is 6.92 / 7.09 / 5.98 panel pitches and to the clamp 7.70 / 7.17 / 8.84, and no pair
survives the sign test across the fifteen sheets. So they are neither load-path elements nor
boundary-layer elements.

**What the families *are* ordered by is exact and kinematic:** their gain,
215 / 176 / 93 x10^3 mm^2, and their realised void area follows `gain * sin(psi)` at rho = 1.000.
Family 2 contributes under half the opening of family 0. That is a real role, and it is the only
one the data support.

## 6. What the flat result means

The load path in this sheet is set by the global topology and the boundary conditions, not by local
cell geometry. A unit cell is a *kinematic* unit — its two numbers fix its area and stroke exactly,
by a closed form that reproduces at ρ = 1.000 — and it is **not** a *force* unit.

That is a substantive statement, and it cuts in a useful direction for the manuscript: the family
panel of Fig. 6 currently claims a geometric and spatial fact and nothing more, and that is exactly
what the data support. Upgrading it to a structure–property claim would not survive.

It also sits well with two references already in the bib and currently under-used:
`coulais_characteristic_2018` (a characteristic length scale in a rotating-square metamaterial, the
mechanism-level reason a *finite block* is the right descriptor at all) and `zheng_modelling_2023`
(which homogenises the cell away — the theoretical counterpoint, and the flat result here is a
point in its favour).

## 7. Limits, stated

- **One seed per case.** The paper already carries this as `[OPEN]`. Arm A's null could in
  principle be a null because a single seed's composition is noisy. The gate is
  `run_closed.py --init-noise` on three cases × three seeds: the recruitment claim (in either
  direction) only firms up if within-case composition is tighter than between-case.
- **Five controlled actuations.** Arm A rests on five sheets. The ten grips could not be promoted
  into it without re-running them against a pinned target (10 training runs).
- **`L` is not a local freedom**, so "family" is not a purely local design choice and a causal
  reading of any family effect would need the ablation (below).
- **Case V's opening angle is not uniform.** The claim "the void opening is constant over a sheet
  to under half a degree" holds at 0.37–0.58° on cases I–IV, but case V spreads **2.64°**, 4.5× the
  others. The descriptor's justification is weakest exactly where case V is already the outlier in
  kind (84% antisymmetric cut field, least efficient, rotating-rectangle motif). This does not
  affect the results above, which never divide by the opening, but it should not be repeated as an
  unqualified sheet-wide claim.
- **The ablation was not run.** Arm C (constrain a sheet to one family, re-optimise, measure the
  shape-error penalty) is the only arm that licenses the word "better". It is ~20 training runs at
  200 epochs. Arms A, B and D all predict a flat penalty matrix; the ablation is the test that
  would distinguish "no signal in observation" from "no causal role".

## 8. Reproduce

```bash
JAX_PLATFORMS=cpu conda run -n kgnn_mac python \
    nff/scripts/diagnostics/cell_family_corpus.py --figmat --grips
JAX_PLATFORMS=cpu conda run -n kgnn_mac python \
    nff/scripts/diagnostics/cell_family_map.py --n-perm 20000
JAX_PLATFORMS=cpu conda run -n kgnn_mac python -m pytest tests/test_cell_families.py -q
```

Outputs: `data/outputs/cell_families/{cells.csv, runs.json, family_map.json}`.

## 8b. WHERE OUR FAMILIES SIT IN THE LITERATURE, and why that explains the null

⚠ **Correction to an earlier draft of this memo.** It said Deng et al. have "no families". That came
from a text grep, and a grep cannot see a figure. **Their Figure 2c shows four distinct 2x2 unit
cells, each tiled across the whole specimen, each with its own rotation and shrinking field and its
own response class:** (i) near-square pinwheel, stiffest; (ii) splayed flat quads, most compliant;
(iii) vertically aligned elongated quads, softening plateau; (iv) compact bowtie with two rows that
snap, load drop. That is a motif-level structure-property correspondence and it is visible on the
page. The accurate narrower statement is that these are **four exemplars chosen from 7500 to
illustrate the range, not a partition** -- no boundaries, no rule assigning an arbitrary design to a
class. Archetypes, not a taxonomy.

**The taxonomy that does exist, and that our cells belong to, is Grima's rotating rigid units.**
That line names and solves family after family: rotating squares (Grima & Evans 2000), rotating
rectangles with different connectivity (2005, already cited here at case V), rotating rhombi and
rotating parallelograms (Attard & Grima 2008, 2009), connected different-sized squares and
rectangles (2011), rotating triangles. Each family has a closed-form Poisson's ratio; the rotating-
rhombi case is isotropic at `nu = -1`. And the defining property of the parallelogram sub-family is
exactly ours: **the empty space between the rotating units is always a parallelogram.**

Measured placement of our sheets in that scheme, from the cut pattern alone (no physics; panel shape
is design-fixed), over all 15 sheets, 2250 panels:

| | value |
|---|---|
| panel aspect ratio | median **1.35**, p99 **5.40**, max **6.90** |
| panels with aspect > 2.0 / > 3.0 | **20.9% / 8.1%** |
| \|corner angle - 90 deg\| | median **14.7 deg**, p90 **41.6 deg** |
| corners within 5 deg of square | **20.9%** |
| void `r` | median 0.392, p99 0.494 |

So the tiles are **irregular quadrilaterals**, not squares and not rectangles, and they do reach
strongly elongated shapes. Our system is therefore a *generalisation* of Grima's rotating-
rectangles-with-parallelogram-voids to irregular quadrilateral units. (g09 is the near-square
outlier at 6.9 deg median, and it is also the sheet whose families degenerate to 0/34/1 -- it is
close to a uniform rotating-squares sheet, and behaves like one.)

★★★ **THIS IS THE MECHANISM BEHIND THE NULL.** All three of our families sit inside **one** Grima
family. They differ in aspect and in size within it, never in kind. Deng's four archetypes span
*different* rotating-unit families, which is why theirs differ in response: their cell has sixteen
free parameters and **compliant spring hinges**, so it is free to leave a family. Ours cannot ---
rigid deployability forces parallelogram voids (Dang's Lemma 1), and that condition pins the whole
sheet inside a single rotating-rigid-unit family.

**We buy guaranteed deployability with the loss of family diversity.** Within one rotating-unit
family the mechanism is identical everywhere, so cells cannot differ in what they carry --- only in
how much they open. Which is exactly the gain, and exactly the one family property that did survive
measurement.

### Other tile-family analyses worth reading

- **Coulais, Teomy, de Reus, Shokef & van Hecke**, *Combinatorial design of textured mechanical
  metamaterials*, Nature 535 (2016). Anisotropic building blocks each carrying an orientation; the
  spatial arrangement of two block orientations sets the mode. The canonical "motif class determines
  deformation" result, and aperiodic like ours. Recent follow-ups: *Defect positioning in
  combinatorial metamaterials* and *Breaking mechanical holography in combinatorial metamaterials*
  (Phys. Rev. Research).
- **Zheng, Niloy, Tobasco, Celli & Plucinsky 2023** (already in the bib) names our family from the
  other side: planar kirigami as periodic arrays of unit cells with **four quadrilateral panels and
  four rhombi-slits**, the rotating-squares family generalised.
- **Sun et al. 2021** five geometric classes, actuation-blind. **Hanakata et al. 2020** learned
  parallel-vs-orthogonal cut classes. Both now in the bib.
- ML latent-space work that *discovers* structural classes rather than naming them: point-cloud
  inverse design of free-form metamaterials (arXiv:2411.19681, latent space splits into clusters
  each a structural class, with property gradients inside each), the generative truss framework
  unifying a design space (Nat. Commun. 14, 2023), and interpretable shape-frequency features with
  unit-cell templates.

## 9. On the choice of comparison paper

**Deng et al. 2022 cannot host this analysis.** A grep of the full text for
`motif|famil|taxonom|archetyp|cluster|classif|recurr` returns three hits, all the boilerplate phrase
"the considered class of metamaterials". They have no families, one actuation (uniaxial compression,
`ε ∈ [0, −0.1]`), a 16-parameter cell, and a single mechanism scalar — rotation-induced shrinking
`Δε_rot`, their Eq. 3 — through which all four of their optimised designs are rationalised. They
offer the black-box search *in place of* a structure–property classification. The manuscript's one
citation of them (`sn-article.tex:1109`, as a surrogate-plus-search competitor) is correct and
narrow, and should stay exactly as it is.

The three works that actually engage classification:

| ref | in bib? | why |
|---|---|---|
| `dang_theorem_2021` | yes, 9× | The RDPQK source. Classifies cuts into two types and encodes a tessellation as a `±1, ±2` topology matrix — a genuine discrete motif label on quad kirigami, parameterised by the same per-cut `r_ij = b/a`. The natural anchor. |
| Sun et al. 2021, *Structures* 33:3633 (`Zotero/QIXMD3H3`) | **no** | A five-class kirigami taxonomy (fractal cut, ribbon, lattice, zigzag, closed-loop) that is purely geometric and entirely **actuation-blind**. The precise citation for "classifications exist; none is conditioned on the load". |
| Hanakata et al. 2020, *PRR* 2:042006(R) (`Zotero/7JL34XEH`) | **no** | A supervised autoencoder that *learns* parallel-vs-orthogonal cut classes and interpolates between them — the closest existing "learned family structure in a kirigami design space". |

Also worth promoting: `coulais_characteristic_2018` (cited once), `wang_physics-aware_2023` (the
closest methodological twin that actually varies actuation), and — if the negative result is ever
written up — Felsch & Slesarenko 2024 (`Zotero/I48VPRK5`), which argues that naive similarity
metrics are unsuited to kirigami design spaces.

---

# Part II: the families as an effective medium — where the behaviour actually lives

**2026-08-21.** Part I asked which force channel a family carries and found nothing. Zheng, Niloy,
Tobasco, Celli & Plucinsky (*Proc. R. Soc. A* **479**:20220665, 2023 — `zheng_modelling_2023`,
already in the bib) say that is the wrong observable. In their coarse-graining each cell carries a
**shape tensor** (Eq. 4) fixing the deformation it prefers, and the sign of its **effective Poisson
ratio** (Eq. 17) sets the type of the PDE governing the actuation field (Eq. 16): negative is
elliptic, and a disturbance decays away from where it is applied; positive is hyperbolic, and it
persists into the bulk.

⚠ **The manuscript already said this and never computed it.** `paper/sn-article.tex:4671-4676`, in a
block deleted for length on 2026-08-12: *"Decay versus persistence, decided by the unit cell's
Poisson ratio, is the dichotomy of `zheng_modelling_2023`."* The subsection was cut for length, not
for being wrong, and no Poisson ratio of this sheet has ever existed. There is one now.

## 10. Method: the shape tensor read off our own kinematics

Their algebra is **not** ported. Their cell is three edge lengths and two sector angles with
congruent mirrored panels and rhombus slits; ours has irregular quadrilateral panels and
parallelogram voids. Instead, using the identity that Eq. 17 is algebraically

```
nu_21 = - (d ln lambda_2 / d psi) / (d ln lambda_1 / d psi)
```

with `lambda_1, lambda_2` the stretches along the two cell-lattice axes, the whole quantity follows
from `jax.grad` through the exact rigid kinematics (`shield_kinematics.deploy_tiles`, already
differentiable in `psi`). `alpha` and `beta` are then read back from Eq. 4. Physics-free: a sheet
needs only `config.yaml` and `best_params.pkl`. Module `nff/closed/effective_medium.py`.

⚠ The cell frame comes from **cell-lattice vectors**, never the void's sides: those two lengths are
deployment-invariant and in the flat sheet the void is a collapsed slit whose edges are parallel.

**Gate — the published case.** A uniform `r = 1/2` sheet is the rotating-squares pattern, for which
their Eq. 28 gives `nu_21 = -1` identically. Measured on 8x8, 10x8 and 15x10 sheets at three
openings: **`nu_21 = -1.000000000` on every cell**, `beta = -alpha` to 1e-15, zero lattice shear.
That is the licence for reading the same quantity off the optimised sheets.

## 11. The regime map, and what deployment does to it

| block | `nu_21` median | hyperbolic cells |
|---|---|---|
| figmat I-V (`psi` -28 to -34 deg) | -0.52 to -0.82 | **0.0 % to 14.3 %** |
| grips g00-g09 (`psi` -38 to -46 deg) | -0.29 to +0.98 | **40.0 % to 85.7 %** |
| pooled, 525 cells | -0.262 | 39.4 % |

★★★ **Spearman(hyperbolic fraction, `|psi|`) = +0.836 over 15 sheets.** Deploying further drives a
sheet from elliptic into hyperbolic. That is exactly the sign-changing behaviour of their second
example, whose `nu_21 ~ 1.11 xi` crosses zero as the actuation grows, and it is the single
strongest relation found anywhere in this study.

**The pre-registered prediction is confirmed.** Part I recorded that case V's void opening spreads
**2.64 deg** against 0.37-0.58 deg on cases I-IV, and I predicted before measuring that V is where
hyperbolic cells would appear. Case V has **14.3 %**, the highest of the figmat block; cases II and
IV have **0.0 %**. Across all fifteen sheets,
Spearman(hyperbolic fraction, opening-angle spread) = **+0.536**. A sheet whose cells disagree about
the regime cannot hold one opening angle, and that is what the spread measures.

⚠ **Their shape tensor is diagonal; ours is not.** Lattice shear: median **11.1 deg**, p90 **36.3**,
max **55.5**. The manuscript's own warning at `sn-article.tex:5445` (the `diag(lambda_x, lambda_y)`
picture reaching 32 deg of shear) is thereby confirmed from a second direction. The off-diagonal is
an O(1) part of the deformation that Eq. 4 has no slot for, so every number here is the diagonal
part of a tensor that is not diagonal. Stated, not hidden.

## 12. ★★★ The families DO differ — in the conformality defect

`nu_21 = -1` exactly when `beta = -alpha`, so **`alpha + beta` is a one-number conformality defect**,
and it is far better behaved than `nu_21` itself, which is a ratio and diverges wherever
`d lambda_1 / d psi` crosses zero.

| family | centroid (mm) | `L` (mm) | defect `alpha+beta`, pooled median |
|---|---|---|---|
| 0 | 725 x 297 | 1022 | **+0.193** |
| 1 | 491 x 360 | 850 | **+0.101** |
| 2 | 411 x 226 | 637 | **-0.443** |

Within-sheet, detrended, paired sign test over the fifteen sheets:

| detrend | family medians | significant pairs |
|---|---|---|
| 0 | +0.267 / +0.131 / -0.575 | 0v2 **p = 0.0001** (14/14), 1v2 **p = 0.0001** (15/15) |
| 1 | +0.306 / +0.127 / -0.621 | 0v2 **p = 0.0001** (14/14), 1v2 **p = 0.0001** (15/15) |
| 2 | +0.084 / +0.021 / -0.179 | 0v2 p = 0.013, 1v2 p = 0.007 |
| 3 | +0.053 / +0.037 / -0.121 | 1v2 p = 0.007 |

**Family 2 departs from conformality with the opposite sign to families 0 and 1**, on 14 of 14 and
15 of 15 sheets, surviving three of four detrend orders. This is the strongest and most consistent
family effect in the whole study, and it is in the quantity the theory nominates.

**What sets it is SIZE, not shape.** Within-sheet Spearman of the defect against cell geometry,
15/15 sheets in every case: `span L` **+0.709\*\*\***, `gain` **+0.703\*\*\***, `long` +0.591\*\*\*,
`short` +0.588\*\*\*, and against `aspect` (`r`) **+0.025, null**. Large cells depart from
conformality one way, small cells the other, and the aspect ratio does not enter. Families 0 and 1
(`L` = 1022, 850 mm) never separate from each other; both separate from family 2 (`L` = 637 mm).

So the answer to the original question is: **a family's behaviour is how far, and in which
direction, it departs from the conformal rotating-squares limit — and that is a property of the
cell's size.**

## 13. ⛔ Refuted: frustration does not sit at family boundaries

Pre-registered in the plan: since the families form spatial regions, and Celli et al. 2018 define
geometric frustration as *"the desired deformation mode is impeded by the geometric incompatibility
between neighbouring cells"*, departure from the mechanism should concentrate on the domain walls
between family regions.

| test | result |
|---|---|
| departure from 1-DOF, boundary - interior | median **-1.8e-4**, 7/15 sheets positive, **p = 1.00** |
| stored hinge energy, boundary - interior | median +0.025, 9/15 positive, p = 0.61 |
| neighbour mismatch in `lambda_1` / `lambda_2` / `nu_21` vs departure, detrend 0-2 | all null, `|rho| <= 0.08` |

**The prediction was wrong.** Neither the family interface nor a continuous neighbour mismatch in
the shape tensor predicts where the sheet leaves its mechanism. Frustration in this sheet is not
localised on the family map. Recorded because it was pre-registered, and because it was the
mechanism I expected to find.

## 13b. THE DIRECT QUESTION: do the families map onto a distinct BEHAVIOUR of Zheng et al.?

**No.** They map onto a *coefficient* of that model, not onto one of its behaviours.

Their two distinct behaviours are the regimes: elliptic (`nu_21 < 0`, actuation decays) and
hyperbolic (`nu_21 > 0`, actuation persists). Fraction of hyperbolic cells per family, per sheet,
over the fifteen sheets:

| | family 0 | family 1 | family 2 |
|---|---|---|---|
| mean hyperbolic fraction | 39.6 % | 35.3 % | 49.4 % |

and no pair survives a paired sign test: 0v1 p = 0.065, 0v2 p = 0.388, 1v2 p = 0.092.

The variance decomposition is the cleanest statement of it, and the two quantities partition the
opposite way round:

| share of variance explained (eta^2) | by which SHEET the cell is in | by which FAMILY it is in |
|---|---|---|
| regime label (hyperbolic vs elliptic) | **0.291** | **0.010** |
| conformality defect `alpha + beta` | 0.007 | **0.205** |

**The regime belongs to the sheet; the defect belongs to the family.** Which regime a cell is in is
set by how far its sheet deploys (Spearman with `|psi|` = +0.836), and the family label adds
essentially nothing to it. Conversely the conformality defect is almost a pure family property and
is nearly independent of which sheet the cell sits in.

So the mapping that exists is: **family -> position in the shape tensor's parameter plane**, not
**family -> one of the paper's two behaviours**. A family is a value of `alpha + beta`, and
`alpha + beta` does not by itself decide the regime: `nu_21` depends on the actuation angle too, and
here the angle is what dominates.

## 14. Where Part I and Part II leave the question

- A unit cell's geometry says **nothing** about the force it carries (Part I, bounded at ~4 % of
  rank variance).
- It says a **great deal** about the deformation it prefers: the families differ decisively in
  their conformality defect, that defect is set by cell size, and how far the sheet deploys decides
  whether its cells are elliptic or hyperbolic.
- The unit cell is a **kinematic** object throughout. Part I found its exact stroke
  (`r(1-r)L^2`, reproducing at rho = 1.000); Part II finds its regime. Neither found a force role.

If any of this ever enters the manuscript, the natural home is the subsection that was deleted on
2026-08-12, which already framed the dichotomy and already cited Zheng for it. That is a proposal,
not a change: `paper/` is untouched.

## 15. Reproduce Part II

```bash
JAX_PLATFORMS=cpu conda run -n kgnn_mac python \
    nff/scripts/diagnostics/cell_regime_map.py
JAX_PLATFORMS=cpu conda run -n kgnn_mac python -m pytest tests/test_effective_medium.py -q
```

Outputs `data/outputs/cell_families/regime.json`. The rotating-squares gate is
`tests/test_effective_medium.py::test_rotating_squares_is_conformal`.

---

# Part III: do the tension routes prefer a unit-cell family?

**2026-08-21.** Asked because the author reads a correspondence off Fig. 6 between where a family
appears and where the tension routes run. Parts I and II do **not** answer it: they reduce force
onto each cell individually, and a family *region* can coincide with a *route* while no single
cell's geometry predicts what that cell carries. This is a spatial-overlap question and it needed
its own measurement. `nff/scripts/diagnostics/route_family_overlap.py`; no solve, the five runs
store their force field and the routes are the same objects the plate draws.

**The null.** Shuffling family labels would destroy the region structure and make a contiguous route
look significant for free. The family grid is instead **cyclically shifted on its own 7x5 torus**, so
every region keeps its size and shape and only its alignment with the routes changes. Thirty-four
non-trivial shifts, so no per-sheet p falls below 0.029.

**Result, weighting every route visit (the finer of the two statistics):**

| family | I | II | III | IV | V | Fisher over 5 |
|---|---|---|---|---|---|---|
| 0 (725x297) | 1.06 | 0.91 | 0.83 | 0.75 | 0.88 | p = 0.995 |
| 1 (491x360) | 1.13 | 1.18 | **1.92** | 1.10 | 0.61 | p = 0.375 |
| 2 (411x226) | 0.69 | **0.15** | 0.18 | 1.14 | **3.50** | **p = 0.044** |

A ratio of one is neutral. On the coarser statistic, which counts a cell once however many routes
touch it, nothing survives (p = 0.99, 0.26, 0.56).

★★ **There is a relation, and it is conditioned on the actuation.** Family 2, the small family, is
the one that deviates from its own shift null more than chance across the five designs. But it is
**avoided** under three of the four net-resultant actuations (0.69, 0.15, 0.18) and **strongly
preferred** under the self-equilibrated couple (3.50). Family 1 leans the other way in I-IV
(1.13, 1.18, 1.92, 1.10) and also reverses in V (0.61), though it does not clear chance.

⚠ **This is the same reversal the project has already measured elsewhere.** The two *cut-orientation*
families specialise almost completely and swap roles under the self-equilibrated couple. Case V
inverts a family relation here too, and it is already the outlier in kind on every other axis
(84% antisymmetric cut field, opening spread 2.64 deg, highest hyperbolic fraction).

⛔ **Five designs cannot settle it.** The corpus of Fig. 6 is five sheets, and a relation whose sign
depends on the actuation needs more actuations under a controlled target than exist. The honest
statement is that a correspondence is present and is not a simple preference: it changes direction
with how the sheet is driven. That is what goes in the paper, and it is why the question is open
rather than answered.

Reproduce: `JAX_PLATFORMS=cpu conda run -n kgnn_mac python nff/scripts/diagnostics/route_family_overlap.py`
