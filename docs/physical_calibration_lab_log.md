# PET physical calibration — lab log

Chronological record of the PET tensile/hinge calibration campaign (2026-07-20 → 2026-07-28),
moved out of Claude's memory store on 2026-07-28: it is a lab notebook, not a recall capsule.

**For the current state of the material model, read `nff/rve/materials/pet.py` and the terse
memory capsule `project_physical_calibration`. Where this log disagrees with either, they win** —
entries below are dated snapshots and several were superseded by later ones (most importantly
E ≈ 3.0 GPa, retired in favour of ≈ 2.3–2.5 GPa once the video ladder removed machine compliance).

Related docs: `physical_calibration_series1_tensile_protocol.md` (the protocol),
`physical_calibration_video_extensometer_plan.md` (the extensometer).

---


NEW CHAPTER (2026-07-20): move the [[project_hinge_surrogate_condensation]] surrogate from
simulated S235-steel / paper to a **real physical sheet** the user found — "probably PET",
**8×4 ft, 0.5 mm thick**. Goal: calibrate the model/surrogate against real experiments on the
user's **Instron test column**, then optimize + laser-cut real deployable patterns.

## The calibration chain (two rungs; a tension test hits each differently)
`real coupon → bulk constitutive law (Material class params) → CalculiX RVE → surrogate .pkl → run_closed`.
Plus a second validation arc: `real cut hinge → validates CalculiX→surrogate directly`.
- **Bulk law** lives in `nff/rve/materials/{steel.py STEEL dict, paper.py PAPER_80GSM}` behind the
  `Material` ABC (`base.py`) — `constitutive_cards`/`section_cards`/`el_file_fields`/`failure`.
- **Geometry knobs** (`w_lig, alpha_deg, fillet_ratio`) are surrogate INPUT FEATURES, chosen per cut,
  already spanned by the net — NOT measured. `thickness`(=1.0→set 0.5), `w_c` kerf on `HingeConstants`.

## KEY MODELING POINT: PET ≠ steel ≠ paper
PET is roughly-isotropic, ductile thermoplastic with a yield-then-cold-draw plateau. Architecturally
closest to `SteelJ2` (isotropic `*ELASTIC` 2-num + `failure_mode="ductile_D"`), NOT paper's 9-constant
orthotropic tensile-tear. Right home = a NEW `materials/pet.py::PETIsotropic` copying SteelJ2's
structure, but with a **multi-point `*PLASTIC` table** from the real σ-ε curve (steel is bilinear Et=E/100).
One-file addition behind the existing seam — no oracle/mesh/parser change. HOLD until Phase 0a IDs it.

## TENSION-ONLY ↔ operating regime (the crux)
Deployed closed hinges open almost entirely in ROTATION θ (a,s≈0 on closed hinges). The load cell
natively measures the AXIAL DOF `a` (pull tiles apart) — a DIFFERENT slice of `W(a,s,θ)`. So:
- dogbone + hinge-tension → bulk law + `a`-slice + failure. Necessary, not the dominant DOF.
- θ (the surrogate's operating DOF) needs a FIXTURE converting crosshead pull → moment about ligament.

## User's two hard constraints (answered 2026-07-20)
1. **Load cell + crosshead ONLY** (no DIC, no extensometer). ⇒ crosshead disp = strain + machine
   compliance + grip slip → absolute E untrustworthy. Fixes: **(a) DIY phone DIC** (2 ink marks +
   Tracker/ImageJ/OpenCV) = ~$0, recovers axial strain + ν + θ = HIGHEST ROI; (b) compliance cal on a
   known-E Al/steel strip → subtract C_machine(F); (c) long/narrow gauge to dilute compliance.
2. **No material ID / no spec sheet.** ⇒ Phase 0a = cheap ID: density (PET≈1.38, PETG≈1.27, PC≈1.20,
   PMMA≈1.18) via mass/area/t; fold/snap test (PC folds white unbroken; acrylic snaps brittle;
   PET/PETG creases white then tears); optional acetone (crazes acrylic/PC).

## Recommended sequence
0a ID (density+snap) → 0b compliance cal + set up phone DIC → 2 anisotropy (0/45/90 coupons: >10%
E-spread ⇒ orthotropic + orientation_deg becomes a real cut-angle design var) → 1 dogbone-to-break
(E, σ_y, *PLASTIC table, eps_f/eps_tear0) → 3 cyclic + 2 rates (permanent set? rate-sensitive? H7) →
4 single cut hinge axial pull (end-to-end validate F_a(a)+failure incl fillet concentration) →
5 lever fixture for M(θ) (needs DIC for θ; = the rotation-regime figure with a real data point).

## ★ ROBUSTNESS WIN: inverse calibration against the hinge
Since the surrogate's only consumer is the hinge, DON'T trust absolute coupon E. Treat **Exp-4 hinge
force-displacement as the calibration TARGET** and run CalculiX with (E, σ_y, eps_f) as FIT variables
until the sim hinge matches. Shape + failure-point jointly constrain the params → robust to crosshead
noise. Coupons (Exp 1-3) become priors, not ground truth. Machinery already exists: `evaluate_hinge`
→ (F_a, M_theta, failure_theta_deg). A small inverse-fit script is a natural next build.

## ★ MATERIAL ID (Phase 0a done, 2026-07-20) — likely BIAXIALLY-ORIENTED PET (BOPET)
User measurements: **ρ≈1390 kg/m³** (whole-sheet mass on mg scale ÷ machine-cut 4×8ft area); feels
ductile with some stress whitening; **white rims**; **rainbow/iridescence across the entire roll at
certain angles**; **folds 180° without breaking**.
- ρ=1.39 ⇒ PET-or-PVC band (PETG 1.27 / PC 1.20 / PMMA 1.18 excluded). Rainbow = stress-birefringence
  from molecular ORIENTATION ⇒ breaks tie toward **oriented PET**; rigid PVC doesn't birefringe like that.
- ⚠ ρ alone can't separate PET vs PVC → **copper-wire (Beilstein) flame test**: green flame = chlorine =
  PVC; no green + drips + sweet = PET. DO before committing the class.
- Caveat: rainbow could be a surface hard-coat interference instead of bulk orientation → the 0/45/90
  test resolves it.

### Three consequences for the model (from the ID)
1. **Anisotropy check now MANDATORY** (rainbow = orientation evidence). Exp 2 (0/45/90) moves to front;
   **`orientation_deg` (cut-angle-to-MD) becomes a real design lever** (paper-H2 path already in code).
   FIRST task on the sheet: find + mark MD/roll direction.
2. **Folds 180° unbroken ⇒ deployment ceiling blows WIDE open.** Steel surrogate capped ~7-8% deploy /
   ~34° by FRACTURE (material, not geometry); real PET survives 180° fold ⇒ eps_f many× larger ⇒ design
   no longer fracture-limited. `failure_mode=ductile_D` (SteelJ2 template) CONFIRMED, with big eps_f.
3. **Stress whitening = FREE visible damage gauge.** Answers the long-open "is failure D=1 or D~3-7?" —
   photograph whitening onset vs fold angle/load ⇒ physical anchor for the damage variable D.
- Reframe: **permanent set** (hold-the-fold vs spring-back) is now central ⇒ Exp 3 (cyclic) elevated;
  for a shape-holding deployable, plastic set is a FEATURE.

### Provisional class call
Oriented-PET, ductile_D failure, ELEVATED yield, HIGH eps_f — structurally SteelJ2, but orientation may
force the orthotropic (paper-style) `*ELASTIC` card. HOLD writing the class until Exp 2 (isotropic-vs-
orthotropic fork). Immediate order: copper-wire test → mark MD → cut 0/45/90 coupons.

## ★ Instron 34SC-5 anisotropy test plan (Exp 2, 2026-07-20)
Machine: **Instron 34SC-5 = 5 kN FRAME** (the "-05" listings online are the 0.5 kN version — different).
Bluehill Universal software. User cut **10 rectangular strips 202.5 × 36.8 × 0.50 mm**, 5 per direction.
- **⚠ LOAD-CELL GATE (do first):** A=36.8×0.50=18.4 mm². Est. peak force: amorphous PET yield ~1.0 kN;
  oriented PET yield ~2.0 kN; BOPET break (~200 MPa) **~3.7 kN**. Verify the FITTED load cell rating
  (frame≠cell) and expected peak < ~80% of it, correct cell selected in Bluehill. If cell too small →
  slit strips to ~15 mm width (force ~2.4× lower, also brings into D882 spec).
- **Standard = ASTM D882** (sheet ≤1 mm; rectangular strips — user's approach is correct; NOT D638
  dogbones). Their 36.8 mm width is WIDER than D882's 5–25.4 mm → fine for a screen but ↑force + edge/
  misalignment sensitivity.
- **Setup:** grips ≥37 mm wide, rubber/serrated faces or emery cloth (grip-slip = #1 failure on wide
  thin strips). Grip sep L0≈150 mm (clamp ~26 mm/end). Speed 0.1 min⁻¹ → **~15 mm/min**, same for all 10.
- **Strain (crosshead-only):** raw crosshead understates modulus → use Bluehill **"Corrected Displacement"**
  (built-in system-compliance correction) AND/OR **phone-dot video extensometer** (2 ink dots ~50 mm, track
  in Tracker/ImageJ — bypasses compliance+grip slip; tracking WIDTH change also gives ν). Video is Instron's
  own preferred thin-film strain method.
- **Record:** modulus (chord 0.05–0.25%), yield, UTS, strain-at-break (= eps_f for the model). Zero load+set
  gauge per specimen. **Break at/inside grip = INVALID** (re-run). Label groups by roll direction: **MD ≈ the
  8 ft long axis** — record which set was cut along it.
- **DECISION RULE:** per-direction mean±std of E; ratio within ~10% (bars overlap) → **PETIsotropic** class
  (SteelJ2 template, ductile_D); >~10-15% → **orthotropic** (paper `ENGINEERING CONSTANTS` route) +
  `orientation_deg` becomes a real cut-angle design var (stiffer dir = MD). Either way yields the σ-ε curve
  for the ccx `*PLASTIC` table + failure strain → unlocks the material class.

## ★ Series-1 execution decisions (2026-07-20) — grips, uncertainty, and the RICHER test scope
Hardware confirmed: **5 kN load cell** (matches force envelope); grips = flat/spiky × small/large,
**NO pneumatic**, **large flat faces span ≥37 mm**; **tape yes, no sandpaper**; **calipers + micrometer** yes.
- **GRIPS: large FLAT + tape-tabbed ends** (primary). Rationale: modulus+yield (the anisotropy metrics)
  occur at LOW force, captured cleanly before any high-force slip → flat's clean gauge behavior beats
  spiky's grip-line breaks (=invalid). Spiky-large + tape CUSHION only as slip fallback. Small grips
  rejected (won't span 36.8 mm). **Tape prep:** fold last ~25 mm of each end back on itself OR wrap in
  1-2 layers cloth/duct tape = friction + moves stress conc. off the jaw edge into the gauge (tabbing).
  Avoid slippery packing tape.
- **SETTINGS:** test to break; **15 mm/min** (≈0.1 min⁻¹ @ L0≈150 mm, D882 default), SAME for all;
  data rate ≥50 Hz; **preload 3 N then zero** (kills hand-cut toe/slack); modulus = Bluehill Automatic
  Young's (or chord 5-25 MPa); strain = **Corrected Displacement** (Bluehill compliance corr.) + phone-dot
  video (free, beats corrected crosshead, width-track → ν).
- **UNCERTAINTY (hand-cut):** area = biggest lever → measure EACH specimen w(×3, caliper) + t(×3,
  micrometer), enter PER-specimen (mean for modulus, min for strength). Measure actual free length per
  specimen = gauge. Edge nicks scatter UTS/elong but NOT modulus/yield → weight modulus/yield for the
  iso/ortho decision. Centerline + preload to avoid off-axis. 5/dir → mean±std; anisotropy real if
  |E_MD−E_CD| > ~2×pooled std. Burn 1-2 SCRAP strips first to tune grip pressure/slip.

### RICHER SCOPE (user's key redirect): measure whitening + plasticity + rupture to calibrate FEA/surrogate
User rightly reframed "to break?" → capture the MODEL-relevant physics. Mapping (phenomenon→code):
- **Rupture** (strain-at-break) → `eps_f`/`eps_f0` (damage.py); confirms ductile_D. [monotonic ✓]
- **Plastic hardening** (σ-ε shape) → multi-point `*PLASTIC` table in `constitutive_cards`. [monotonic ✓]
- **Permanent set** (recover vs permanent; "holds the fold?") → whether to model plastic set + unloading
  modulus. [needs LOAD-UNLOAD, not monotonic]
- **Stress-whitening onset** → the PHYSICAL D level for functional failure = resolves the long-open
  "is failure D=1 or D≈5?" (`m_safe` in stability loss). [monotonic + SYNCED VIDEO ✓]
PLAN: keep all **10 monotonic-to-break + synced video** (preserves 5+5 anisotropy stats; yields UTS,
eps_f, hardening, AND whitening-onset strain). **Cut ~4 EXTRA (2 MD/2 CD) → incremental load-unload**
(ramp to 1%,2%,3%… unload to ~5 N hold each time, measure residual = permanent set + unloading modulus +
does white fade on unload). Sheet is 8×4 ft → extra strips are free, don't spend anisotropy replicates.
- **Whitening capture:** sync video t=0 (Bluehill readout in frame or clap/LED); log timestamp of first
  faint white + saturated/necked white → map to σ,ε via synced curve = the D anchor.
- **MD label:** the strip set cut along the **8 ft roll length** = presumptive MD — record before testing.
STATUS: protocol fully specified, no blocking Qs. Awaiting first numbers (per-specimen E/yield/UTS/eps_break
+ whitening-onset strain) → then iso-vs-ortho call + stub the `PET` material class.

## ★ PROTOCOL DOC WRITTEN (2026-07-20): `docs/physical_calibration_series1_tensile_protocol.md`
Complete Series-1 tensile protocol, ASTM D882-based, for the 34SC-5 + Bluehill Universal. Sections:
objective→code-target table, equipment, specimen inventory (10 strips 202.5×36.8×0.50, MD=8ft roll dir),
per-specimen area measurement (caliper w×3 + micrometer t×3; A_mean for E, A_min for strength), frame
setup (5kN cell, large flat grips, limit stops, force limit ≈4.8kN, balance, jog/thumbwheel/`RESET GL`),
Bluehill method (D882 template, rectangular specimen, **15 mm/min** = 0.1min⁻¹@L0≈150, preload 3N+zero,
≥50Hz, **Corrected Displacement** compliance corr., break=40% load drop, Automatic Young's Modulus),
per-specimen run steps, validity rules (jaw break=INVALID; slip→keep E/yield only), a full **uncertainty
table** (area/compliance/slip/jaw-conc/edge-nicks/toe/misalign/rate + mitigations), stats rule (anisotropy
real if |E_MD−E_CD|>2×pooled std → PETIsotropic vs orthotropic), data-recording template, references.
NOTE: user kept RECTANGULAR strips for now (declined the dogbone/waist variant). Official 3400/Bluehill
manuals are SCANNED-IMAGE PDFs (WebFetch+pdftotext both failed; no pymupdf/pypdf on machine) → machine/
software step LABELS reconstructed from Instron product pages + compliance app-note + 3rd-party SOPs (UCR
notebook confirmed thumbwheel + `RESET GL`; Bluehill has a pre-configured D882 template + Corrected
Displacement + preload); WORKFLOW is standard, exact button names to confirm on-machine. Whitening =
semi-quantitative in series-1 (watch CENTRAL THIRD only, tie to yield/neck; jaw-edge whitening = artifact).

## ★ BLUEHILL HANDS-ON WORKFLOW (2026-07-21) — method-build gotchas learned live
User built the D882 method; several UI facts corrected against actual on-screen behavior (my earlier
doc-based descriptions of menu paths were WRONG — do not trust reconstructed Bluehill menu names):
- **Values are entered at TEST TIME in the Test view, not the method editor.** Home→Test→pick method→
  name a **Sample**→specimen input panel. Method editor only DEFINES fields.
- **A field only PROMPTS if it's added to Workspace / Operator Inputs.** There is NO per-column "prompt
  before test" checkbox. This applies to BOTH custom Number Inputs (`w1..t3`, `Direction` choice) AND the
  **standard dimensions (Width, Thickness, Length)** — the standard dims were invisible at test time until
  the user added them to Workspace/Operator Inputs. THIS was the blocker; that's the fix.
- **Custom `w1..t3` do NOT feed stress.** Native `σ=F/A` reads the standard Width×Thickness fields. So the
  raw 3-point inputs are RECORD-ONLY; user must hand-compute the mean (trivial arithmetic, no test run
  needed — Bluehill's `W_mean` calc only populates AFTER a run and is just a cross-check) and type it into
  the standard Width/Thickness. Fixture separation = grip-to-grip = the ~150 mm free length = gauge length
  (no extensometer). "Final width/thickness/length" fields = post-test broken-specimen dims for %elong/%RA →
  SKIP for this series (strain-at-break comes from the extension channel).
- **User Calculations populate during/after the test**, blank at data-entry time (normal). Empty-after-run =
  fail icon → click it for the reason.
- **Validation before real specimens (go/no-go on a SCRAP strip):** (1) known-value calc check w=10/t=0.5 ⇒
  A_mean=5.000 mm²; (2) live stress ≈ load ÷ (entered W×T), NOT ÷18.4 mm² default = confirms dims feed
  stress; (3) preload→zero, 15 mm/min, Corrected Displacement, 40% break; (4) export has all cols + raw σ-ε;
  (5) **modulus sanity ≈2-4 GPa** (PET) — ~200 MPa ⇒ strain overstated (compliance/slip).
STATUS 2026-07-21: method fully set up + dimension entry working. NEXT ACTION = one scrap validation run
(checks 1-5), then run MD1-5/CD1-5 at 15 mm/min, ≥4 valid/dir.

## ★ FIRST REAL PULLS — cold drawing, material ID, rate embrittlement (2026-07-21)
Live bench findings on scrap/trial strips (34SC-5, Bluehill, crosshead-only strain):
- **GRIPPING SOLVED:** wide 36.8 mm strip SLIPPED in flat grips even taped (too wide → too much force).
  Serrated grips then NOTCHED the thin PET → premature break at the jaw base (~0.8 kN ≈ 87 MPa on
  ~9 mm², well below PET's ~200 MPa ⇒ INVALID grip failure). WINNING RECIPE = **slit narrower (~half
  width) + cloth-tape tabs (both faces, past the jaw line) + FLAT grips + slower rate (lower peak
  force)**. Serrated = last resort only; it notches thin film. Tab must extend just past the jaw so the
  stress riser sits on reinforced material.
- **COLD DRAWING / NECK PROPAGATION observed at 5 mm/min** (thinner taped strip): stiff elastic rise to
  ~0.6 kN @ ~3 mm → small yield drop → **~60 mm quasi-constant "plateau"** = neck-shoulder propagating
  up the strip. Below the front the material thinned ~2× (concave/hourglass), oriented. **Front moved at
  ~2× crosshead speed ⇒ natural draw ratio λ ≈ 2 (front velocity = λ/(λ−1)·v)** (front vel = λ/(λ−1)·v; 2×⇒λ=2 ⇒ area ~halves —
  matches "twice as thin"). This is PLASTIC & PERMANENT (not elastic; drawn region stays thin =
  permanent set = the "holds-the-fold" feature). Mechanism = chain unfolding/alignment + strain-induced
  orientation/crystallization (user's "untangling" intuition = correct). Drawn region is **birefringent**
  (crossed-polarizer colors) — same physics as the sheet's rainbow; confirms orientation.
- **MATERIAL ID = PET, CONFIRMED (~97%, 2026-07-22 burn test).** Flame test: **melted, dripped, drew into
  LONG FILAMENTS, did not sustain burning.** Filament-forming molten drip = hallmark of a thermoplastic
  polyester (PET is a fiber-former) ⇒ **excludes PVC** (PVC chars/blackens, self-extinguishes sharply w/
  acrid HCl, does NOT filament) and the brittle amorphous ones (PMMA/PS burn readily, no filaments).
  Stacks with cold-draw+strain-crystallization + ρ1.39 + rainbow iridescence ⇒ **semi-crystalline /
  partially-oriented PET.** (Optional 100% nail: copper-wire green-flame = PVC — but filaments already
  exclude it; sweet-vs-acrid smell + sooty flame + hard bead residue would further confirm.) ⇒ **commit to
  the PET material class**; the PETIsotropic-vs-orthotropic fork still gated on the E(MD)/E(CD) result.
- **RATE EMBRITTLEMENT (design-critical):** 15 mm/min broke early (partly grips) w/ no draw; 5 mm/min drew
  fully. Deployment target is **a few cm/s (~20–50 mm/s) = ~2–3 DECADES faster** than the 5 mm/min test
  ⇒ stiffer, MORE BRITTLE, less draw ⇒ **slow-calibrated `eps_f` is OPTIMISTIC/UNCONSERVATIVE for fast
  deploy.** Add a rate margin on eps_f and/or run a bounding rate study (5/50/500 mm/min). ⚠ 34SC max
  crosshead ~1000 mm/min CANNOT reach cm/s → can't match deploy rate directly; bound + margin instead.
  (Also: hinge fails in BENDING and local strain-rate is geometry-set, so don't over-attribute.)
- **PROVISIONAL E ≈ 3 GPa (LOWER BOUND):** from 0.6 kN @ ~3 mm, A≈9 mm², L0≈150 mm → σ_y≈65 MPa, ε≈2%,
  E≈3.2 GPa. Crosshead-only ⇒ compliance+slip inflate displacement ⇒ TRUE E higher. Consistent w/ PET
  (2–4 GPa). Needs phone-DIC/TestCam or compliance-corrected strain to firm up + exact width/L0.
- **TRUE-STRESS caveat for `*PLASTIC`:** neck localizes strain ⇒ engineering plateau ≠ true material law;
  extract yield + draw stress + λ (true stress–strain), not the raw engineering plateau.
- **NEXT before precut/hinge coupons:** (1) fix strain measurement (phone-DIC/compliance) so E is trusted;
  (2) get a clean VALID coupon set at ONE fixed rate (5 mm/min) → E, yield, full σ-ε to break, MD+CD;
  (3) THEN a short bounding rate study (a faster run) to quantify embrittlement; (4) then the cut hinges.

## ★ ANISOTROPY, SHEAR BANDS, COLD-DRAW→*PLASTIC, EDGE QUALITY (2026-07-22)
- **ANISOTROPY: E_MD ≈ E_CD (similar in both directions)** ⇒ **balanced biaxially-oriented PET (BOPET)**,
  in-plane **near-isotropic**. Consistent w/ rainbow (oriented) + balanced modulus (equally both ways).
  Principal axes = MD/CD (films orient along MD/TD, ~never at 45°) ⇒ user's "axes at 45°?" hypothesis is
  physically unlikely; similar MD/CD = balanced biaxial, not a 45° artifact. **Leaning `PETIsotropic`.**
  ⚠ Outstanding tie-breaker = **45° off-axis strip**: it's the ONLY test sensitive to shear modulus G
  (0°/90° miss G). E_45≈E_0≈E_90 ⇒ TRULY isotropic (lock PETIsotropic, orientation_deg NOT a design var);
  E_45 differs ⇒ orthotropic w/ distinct G. **USER DEFERRED the 45° test 2026-07-22** (not cutting for now).
- **±45° STRESS-WHITENING SHEAR BANDS observed:** conjugate deformation bands on the max-shear planes
  (45° to load) = **ductile shear yielding**, NOT 90° crazes (which = brittle void/crack). ⇒ shear-driven
  yield ⇒ **validates the J2/von-Mises framework** the ccx RVE assumes. Band angle ≈45° ⇒ LOW pressure-
  sensitivity ⇒ **plain J2 OK, no Drucker-Prager needed yet** (measuring exact angle <45° would quantify
  pressure-sensitivity). Whitening onset = the physical **damage-D gauge** (ties to the m_safe/D question).
- **COLD-DRAW → `*PLASTIC` EXTRACTION (the modeling payload).** The propagating neck = a controlled
  LARGE-STRAIN test = exactly the hinge-ligament regime. λ = A0/A_drawn (≈2, area halves). **True strain in
  neck ε = ln λ ≈ 0.69; true stress σ_true = F_plateau/A_drawn = λ·σ_eng,plateau** (~2× eng plateau).
  Build true-σ vs true-ε_p `*PLASTIC` table: `(σ_y,0)`, `(λ·σ_plateau, ln λ)`, + post-draw rehardening pts
  → **replaces the crude bilinear Et=E/100** with a physically-grounded, large-strain-anchored curve.
  ALSO a validation target: a sim w/ the law must reproduce λ≈2 + STABLE propagating neck + plateau stress.
  Area-halving confirms **plastic incompressibility (vol conserved, ν_p≈0.5) ⇒ J2 valid.** TO BUILD IT,
  MEASURE on a drawn sample: **yield-peak force, plateau force, drawn cross-section w_d×t_d.** Then I compute
  the true points + stub PET `*PLASTIC`. BEST approach = **inverse-fit ccx (E,σ_y,hardening) to reproduce the
  cold-draw curve** (yield+plateau+λ jointly constrain it, robust to crosshead noise).
- **EDGE QUALITY drives brittle-vs-ductile.** Hand-cut (box-cutter+ruler) strips → **edge-notch-initiated
  BRITTLE angled breaks in the mid-third**, and SUPPRESS cold drawing (PET is notch-sensitive: a flaw
  fractures before yield can draw). Width variation ~0.3 mm ⇒ ~2% area ⇒ **negligible on E** (fine), not the
  problem — **edge notches are.** STRATEGY: **hand-cut strips = MODULUS ONLY** (E from the pre-fracture
  elastic slope is edge-insensitive; brittle breaks OK); get the **plastic/failure law from CLEAN or
  LASER-cut coupons** (match the real kirigami's laser edge — laser melt-edge is usually smoother, watch a
  thin HAZ) OR from the **cut-hinge test** (the real failure calibration anyway). More end-tape helped grip.
- **DATA FILE `20260721_212033_1.csv`:** wide **37 mm** strip @ **5 mm/min**, crosshead strain,
  Direction=`<None>`. **Premature grip/edge break at ~53 MPa** (below yield, NO draw) ⇒ INVALID for
  strength/ductility. **Usable E ≈ 2.75 GPa** (chord 0.3-1.2% strain; crosshead ⇒ lower bound). ⚠ Verify the
  `Length` field = **grip-to-grip free length**, NOT total strip length (210.3 mm looks like whole-strip) —
  it directly scales E/strain. Export = per-specimen results+raw, clean & parseable. Confirms: **wide strips
  break early → use narrow.** For the MD/CD **ratio**, crosshead strain is acceptable (systematic error
  cancels); absolute E needs phone-DIC/compliance.

## ★ EXPERIMENTAL-CALIBRATION CODE ORGANIZED (2026-07-22)
New standalone analysis stack (numpy; optional opencv/matplotlib; NOT part of the JAX pipeline):
- **`nff/calibration/`** (library): `bluehill_io.py` (parse the multi-block Bluehill results+raw CSV →
  per-specimen time/disp/force; geometry from Results Table is UNTRUSTED per §13), `summary_io.py`
  (parse the operator's manual summary sheet = SOURCE OF TRUTH for area/L/direction/speed, matched to
  curves BY ORDER of `included` rows), `stress_strain.py` (chord E, 0.2%-offset yield, UTS, draw plateau,
  strain-at-break; `crosshead_strain` w/ optional `C_machine` compliance subtraction; `draw_true_point`
  = the cold-draw `*PLASTIC` anchor σ_true=λ·σ_plateau @ ε=ln λ), `video_extensometer.py` (OpenCV 2-dot
  tracker `track_marks`→`marks_to_strain`→`sync_to_force`, +Poisson; cv2 imported lazily).
- **`nff/scripts/calibration/analyze_run.py`** (CLI): `python -m nff.scripts.calibration.analyze_run
  --csv <bluehill> --summary <sheet> [--compliance C --draw-window LO HI --plot P --processed-dir D]`.
  VALIDATED: summary_io reads the real sheet (included=2.1,3.2,3.1,7.1,6.1); full pipeline runs.
- **`data/experiments/{raw,video,processed}/`** (gitignored) — Bluehill CSVs + summary sheets in raw/,
  test videos in video/, extracted stress-strain + plots in processed/. Summary sheet copied in;
  ⚠ USB `20260721_212033_1.csv` raw curves NOT yet copied (drive unmounted) — copy when remounted.
- **Docs:** `docs/physical_calibration_video_extensometer_plan.md` (dot-placement protocol: 2 axial dots
  40-60mm apart mid-gauge, white paint-pen on matte-black backing / black on light, matte no-glare,
  ruler-in-frame scale, LED/clap t=0 sync; +2 lateral dots for ν) + software arch + roadmap.
- **NEXT code:** validate the dot tracker on a real video; compliance-cal on a known-E metal strip;
  then true-σ/ε + `*PLASTIC` → stub `nff/rve/materials/pet.py` (SteelJ2 template).

## ★★ KIRIGAMI TENSILE BATCH + PET MATERIAL CLASS BUILT (2026-07-23)
Best dataset yet: **10 strips (4 CD: 4.1/4.2/5.1/5.2; 6 MD: 8.1/8.2/9.1/9.2/10.1/10.2), 1 mm/min**
(the sheet's "1mm/sec" Speed label is a TYPO — actual rate from disp/time = 1 mm/min). **9 cold-drew,
4.2 broke** (premature edge flaw at 7.4mm/47MPa). Data: `data/experiments/raw/kirigami_20260723/`
(plain `Time,Displacement,Force` per-specimen files + metadata sheet w/ measured DRAWN dims + λ).
- **RESULTS (9 drawn):** yield **48.8±1.6 MPa** (ISOTROPIC: CD 49.5, MD 48.4) · eng plateau **33 MPa**
  (~32% yield drop) · draw ratio **λ=3.28** (CD 3.57 > MD 3.14 — mild; MD more pre-oriented) · true draw
  anchor **σ_true≈109 MPa @ ε_true≈1.19** (=λ·plateau, ε=ln λ). Very reproducible.
- **E RESOLVED:** user spotted **two linear sections** — steep initial (~1.2 GPa glassy tangent) then
  softer pre-yield (~0.7 GPa); the initial TANGENT is the right E (not a chord). BUT still a compliance
  lower bound. Web search: PET E = 2-4 GPa (unoriented), ~4 (BOPET), floor ~2 → **1.2 GPa is below the
  whole PET range = compliance artifact.** DECISION: **use E=3.0 GPa (literature placeholder)**, ν=0.4;
  pin later w/ video (initial-tangent on gauge strain). Our 1.2 stays only as the crosshead lower bound.
- **DECISIONS (user):** (1) **PETIsotropic (pooled mean)** — yield isotropic justifies it; log CD>MD draw
  anisotropy for later. (2) **Monotonic *PLASTIC** — folding = plastic "damage"/permanent set WITHOUT
  break (user's insight), a bending ligament forms no propagating neck → no softening branch needed;
  pair w/ large eps_f. (3) E=3.0 literature now.
- **BUILT (Phase 0-2 DONE):**
  * `nff/rve/materials/pet.py::PETIsotropic` (registered `"pet"`): `*ELASTIC 3000/0.40`, multi-point
    `*PLASTIC=[(48.8,0.0),(109.0,1.154)]` (true σ / true ε_p), **ductile_D**, **eps_f0=1.2 (LOWER BOUND**
    — stopped mid-plateau, not broken; refine w/ run-to-break/hinge test), k=1.5 placeholder. SteelJ2
    template; deck verified; **8 material tests still pass (steel byte-identical).**
  * `nff/calibration`: added `bluehill_io.load_raw_csv` (plain format), `stress_strain.initial_tangent_
    modulus` + `build_plastic_table`, `summary_io` drawn-dims + `.lam` (A0/A_drawn) + `.included` default.
  * CLI `nff/scripts/calibration/analyze_kirigami_set.py` → per-spec table + MD/CD aggregates + *PLASTIC +
    overlay figure `data/experiments/processed/kirigami_20260723_overlay.png` (textbook yield-drop+plateau).
- **NEXT:** (a) **video extensometer** → true E to replace the 3.0 placeholder; (b) 1-2 **run-to-break** →
  real eps_f + rehardening tail (extend *PLASTIC); (c) few **fast pulls** → rate-embrittlement bound
  (deploy is fast → slow eps_f optimistic); (d) **ccx RVE hinge trial** (reproduce a fold, compare to
  physical hinge test).
- **★ ccx IS AVAILABLE ON THIS MACHINE (corrected 2026-07-24):** `/opt/miniconda3/envs/ccx/bin/ccx`
  (CalculiX; run via `conda run -n ccx ccx -i job`), gmsh 4.15 in `kgnn_mac`. Earlier "ccx not on this
  machine" notes (here + [[project_paper_hinge_fea]]) were WRONG. `nff/rve/ccx_solver.deploy` already
  routes `material` through `coerce_material` + `constitutive_cards`, so passing a **`PETIsotropic()`**
  instance as `HingeConstants.material` emits the PET `*PLASTIC` deck directly (⚠ `descriptor()` still does
  `material["sigma_y"]` — dict-only — so skip it for PET; `evaluate_hinge` doesn't call it).

## ★★ PET ccx HINGE TRIAL — works; SHEAR chosen as Instron validation mode (2026-07-24)
Ran the calibrated `PETIsotropic` law through the single-hinge RVE (`nff.rve.hinge_function.
evaluate_hinge`) end-to-end in ccx. **Rotation smoke (w_lig=5mm, t=0.5mm, α=90, fold→10°):** ok,
3224 elems, **M_theta peaks ~93 N·mm at ~5°** then softens (buckling relief), ligament yields at
~2.5°, PEEQ 0.08≪eps_f 1.2 (survives). Confirms the material→hinge pipeline.
- **GEOMETRY MAP (user's real hinge = ~20cm-dia disk, cut 0.5-5cm, 0.5mm PET, 8×4ft sheets):** the RVE
  models the LIGAMENT neighbourhood only (Saint-Venant truncation) so the 20cm dia doesn't enter directly
  — inputs are `w_lig`∈[5,50]mm (=the 0.5-5cm cut), `thickness=0.5`, `w_c`=laser kerf ~0.2, `fillet_ratio`,
  `alpha`, `r_win≈2.4·w_lig`. First trial w_lig=5mm; sweep to 50mm later.
- **★ SHEAR = the validation mode (user's call, agreed).** On the Instron, ROTATION (the deploy mode)
  needs a moment fixture; SHEAR is fixture-free — just orient the hinge so the crosshead pull SLIDES the
  tiles parallel to the cut (shears the ligament); AXIAL opening tears the notch early. Shear is
  fixture-free AND probes the ±45° shear-yielding directly = a strong J2 test. FRAMING: validate in shear
  (easy F_s(s) measurement), PREDICT in rotation (same constitutive law transfers). RVE does both via the
  ray's `eta_s`/`eta_a`.
- **PHYSICAL HYPOTHESES (material in ccx):** isotropic `*ELASTIC` E=3.0GPa/ν=0.4 (yield-isotropic
  justified; E=lit. placeholder); J2 rate-indep multi-pt `*PLASTIC` (49→109 true, monotonic — ±45° bands
  ⇒ near-von-Mises, no propagating neck in bending); ductile_D eps_f≥1.2 (lower bound); plastic
  incompressibility (validated by λ area-loss); permanent set via plasticity (crease; H5 off); quasi-static
  (⚠ 1mm/min calib, fast deploy embrittles ⇒ optimistic). Structural: H1 rigid tiles (⚠ thin PET tiles
  bend — constrain near ligament), H3 NLGEOM+seeded buckling, solid C3D15 n_through≥2 (⚠ 0.5mm thin ⇒
  aspect ratio; shell S8R fallback).
- **SCRIPT:** `nff/scripts/calibration/run_pet_hinge_trial.py` (`--w-lig --theta --eta-s --eta-a --steps
  --out`; run in ccx env). Saves npz(theta_deg,a,s,M_theta,F_a,F_s,W,peeq,damage,uz_max,regime,failure).
  COMPARE-TO-EXPERIMENT map: M_theta↔fold torque, F_s(s)↔Instron shear force-disp, failure_theta↔tearing
  angle, damage↔whitening, uz↔buckling, residual after unload↔permanent set.
- **RESULTS (w_lig=5mm, t=0.5mm, α=90):**
  * **ROTATION** = **buckling SNAP-THROUGH**: M_theta peaks **+17 N·mm @ 4°** then REVERSES (buckles
    uz=6.2mm) to **−361 N·mm @ 40°** (solve stopped ~40° deep in buckled fold; survives, PEEQ 0.42).
    **Imperfection-sensitive** (10° smoke peaked 93, full run 17) ⇒ NOT a reliable calibration target.
  * **SHEAR** = **clean MONOTONIC** F_s(s): steep elastic → knee **~44 N @ ~0.5mm** → gentle harden to
    **~54 N @ 4mm**; PEEQ 0.96 near end (approaches eps_f 1.2 ⇒ may tear ~s=4-5mm); uz 6.4mm (shear-panel
    wrinkling = the ±45° field). Force ~50 N = trivially measurable on 5 kN Instron.
  * **CONCLUSION (validates user's shear call):** rotation buckling-dominated/fragile → shear robust →
    **CALIBRATE on shear F_s(s), PREDICT rotation.** Figure `data/experiments/processed/pet_hinge_trial.png`
    (`nff/scripts/figures/plot_pet_hinge_trial.py`). Physical shear setup: grip tiles sliding parallel to
    the cut, constrain near ligament (RVE=rigid tiles), watch tear ~4-5mm + ±45° whitening.

## ★★ HINGE CALIBRATION STRATEGY (2026-07-24, plan-mode approved)
Principle: **material law = from COUPONS, no hinge-specific fitting** (isotropic E/yield/hardening/eps_f/k);
hinge tests VALIDATE it and calibrate only genuinely-structural knobs. The ONLY hinge-specific fit is the
**imperfection amplitude imp_amp** (buckling seed). Keep material-vs-structure separate = trustworthy, not
a curve-fit.
- **KEY INSIGHT — a 1-DOF Instron sweeps the whole in-plane (a,s) plane via GRIP ORIENTATION φ:** φ=0 (pull
  ∥ cut)=pure SHEAR (eta_s), φ=90 (pull ⊥ cut)=pure OPENING (eta_a), 0<φ<90=mixed. Fixture-free. CANNOT
  reach: rotation θ (needs moment fixture) or shear-at-fixed-θ (2-DOF jig). Map φ→(eta_a=sinφ, eta_s=cosφ).
- **BONUS:** each φ = different triaxiality at the notch ⇒ running φ-sweep TO BREAK **calibrates the failure
  locus eps_f0 + k** (both currently placeholders: eps_f0=1.2, k=1.5).
- **MEASUREMENT CAVEAT (same as coupons):** load-cell FORCE is clean, crosshead DISPLACEMENT is compliance-
  corrupted ⇒ calibrate on **force vs VIDEO-measured tile displacement**, not force-vs-crosshead.
- **TIERS:** T1 shear φ=0 → validate material law (headline, no free params). T2 mixed-φ (0/30/60/90) to
  break → in-plane W(a,s,0) + failure locus (eps_f0,k). T3 buckling: fold + camera uz(θ) → tune imp_amp
  (pins the fragile rotation moment; report value+spread, imperfection varies per specimen). T4 (defer)
  shear-at-fixed-θ / rotation-coupled (2-DOF jig).
- **USER'S 2 IDEAS JUDGED:** (a) calibrate buckling magnitude for rotation = YES = T3 (uz→imp_amp).
  (b) shear at fixed θ = valuable but 2-DOF/hard ⇒ defer; mixed-φ covers (a,s) 1-DOF-free first.
- **SOLVER TODO for T3:** expose `imp_amp` in `evaluate_hinge`/`solver_kwargs` (currently default only) to
  sweep it. RVE already supports mixed (eta_a,eta_s) rays + stop-at-fracture.
- **TIER-2 φ-SWEEP DONE (predictions, w_lig=5mm):** projected force-along-pull vs disp, to break:
  φ=0 shear ~53 N plateau, ductile, SURVIVES to 4mm (PEEQ 0.96<1.2); φ=30 peak ~83 N, fractures ~2.8mm;
  φ=60 ~97 N, ~2.0mm; φ=90 opening ~106 N, fractures EARLIEST ~1.7mm (all PEEQ~1.35 at fracture). Clean
  triaxiality/failure-locus: **shear=most ductile, opening=fails earliest** → the 4 fracture points
  calibrate eps_f0 & k. Forces 50-106 N, disp 1.7-4mm = trivially Instron-measurable. Figure
  `data/experiments/processed/pet_hinge_phi_sweep.png` (`nff/scripts/figures/plot_pet_phi_sweep.py`;
  sequential blue ramp, × at predicted fracture). Physical: cut identical hinges, pull at φ=0/30/60/90 to
  break, force=load cell / disp=video, overlay on the prediction; match the × points → fit eps_f0,k.

## ★★ GRAVITY-COPLANAR REFRAME + CORRECTED GEOMETRY (2026-07-24)
**Corrected physical setup (supersedes earlier "20cm disk, cut 0.5-5cm"):** real **PET**, 0.5mm, **8×4ft
sheets, ~10 faces** (coarse → BIG tiles ~40-70cm). **Hinge strip width (=w_lig) = 0.5-10cm = 5-100mm**
(t/w_lig = 0.1 → 0.005, a 20× slenderness span). **Hinge/buckling domain ≤ 20cm**, scales w/ strip width
(≈2·w_lig) → sets r_win. Big tiles ⇒ RVE rigid-tile + Saint-Venant truncation MORE valid (earlier
"w_lig≤40mm" caveat RELAXED). Narrow end (5mm)=bending-stiff (our 1st trials); wide end (100mm)=film-like
membrane/wrinkling — mechanism TRANSITIONS across the sweep (that's the scientific content of the w_lig
sweep: 5/10/25/50/100mm).
- **★ GRAVITY KEEPS THE *FACES* COPLANAR — the *LIGAMENT* buckles out of plane (user's insight; I first had
  it BACKWARDS, corrected 2026-07-24).** The mechanism is a **weight/stiffness hierarchy:** the big heavy
  gravity-stabilized FACES stay flat = act as **rigid flat handles**; the thin light LIGAMENT **buckles out
  of plane** to absorb the in-plane deployment motion at low energy. **The out-of-plane buckling is the
  DESIRED mechanism (buckling relief = low-force fold), NOT an artifact to suppress.** ⇒ this **VALIDATES
  the RVE**: gravity physically enforces its rigid-tile BC, and the ligament out-of-plane buckling the RVE
  computes IS the real mechanism (the w_lig=5mm rotation snap-through is physical, not spurious).
  Scaling that supports "faces stay flat": gravity hold ~ρ·g·L³·t (grows L³) vs ligament out-of-plane
  stiffness ~E·t³ (L-indep) ⇒ crossover face size **L~(E t²/ρg)^(1/3)≈38cm**; user's faces 40-70cm >
  crossover ⇒ gravity dominant on the faces. ⚠ DO NOT say "gravity suppresses the buckling" — WRONG.
- **SANITY TEST DONE:** w_lig=50mm rotation (no gravity, full solid): **ligament buckles uz=41.6mm** at fold
  moment **265 N·mm, ~elastic** (PEEQ 0.013). Gravity's hold on a ~55cm face ≈**567 N·mm ≫ weak ligament
  265** ⇒ heavy faces stay flat, light ligament buckles = hierarchy CONFIRMED. (⚠ M-vs-gravity is a rough
  proxy — rigorous number = the out-of-plane HANDLE reaction, not extracted yet.) w_lig=100 rotation FAILED
  (mesh aspect 190:1); wide shear diverged at 1st increment — **wide solid mesh numerically hard** (needs
  finer thru-thick or a better mesh; user said NO membrane/shell). Figure `data/experiments/processed/
  pet_gravity_check.png` (`nff/scripts/figures/plot_gravity_check.py`).

## ★ EXPERIMENT 1 (hinge shear) DESIGN LOCKED (2026-07-24): w_lig = 15 mm
Best shear-validation coupon given Instron(5kN)+physics+sim: **w_lig=15mm, α=90°, fillet_ratio=0.16
(ρ≈2.4mm), t=0.5mm.** Logic: (1) force ~128N = 2.5% of 5kN cell = precise (vs ~50N@w=5mm resolution
floor); (2) sim-RELIABLE (129 incr, ran to fracture — wide 50/100mm shear DIVERGE at 1st increment,
mesh aspect); (3) representative + genuinely buckling (t/w_lig=0.033, uz~9.6mm = real mechanism); (4)
grips clean as single-hinge coupon. **Shear family predictions (all mid-range solve; sequential fig
`data/experiments/processed/pet_shear_family.png`):** peak F_s = 54/90/128/205 N for w=5/10/15/25mm
(≈linear in w_lig ∝ area); **fractures in shear at s ≈ w_lig (shear strain ≈1)** consistently → one
shear test gives stiffness + true E + a shear-FAILURE point. w=15 prediction: steep elastic → yield
knee → peak ~128N @ s≈13mm → fracture s≈13-15mm, uz~9.6mm. SPECIMEN: 0.5mm PET, two tiles ≥40mm beyond
ligament + ~25mm taped grip tabs, pull ∥ cut (φ=0). MEASURE: force=load cell, disp=front dots (→ true
E), side camera → uz. Overlay F_s(s) on the sim; match w/ no free params ⇒ material law validated.
OPEN: center w_lig on the real run_closed design distribution (offered to pull it); grip strategy for tiles.

## ★ EXP-1 GEOMETRY VISUAL + HARDENING-COMPARISON CONFIRMED (2026-07-25)
- **Hinge geometry visual** `data/experiments/processed/pet_hinge_geometry.png` (`nff/scripts/figures/
  plot_hinge_geometry.py`): Panel A = cut pattern TO SCALE (secondary cut, main slit + 2.4mm rounded tip,
  w_lig=15mm, α=90°, deformation domain Ø≈72mm); Panel B = **actual deformed state parsed from the ccx
  .frd** (`/tmp/hinge/w015.../hinge.frd`), 3D scatter colored by uz — ligament buckles ~10mm out of plane
  under shear (matches uz~9.6mm). Confirms mechanism: flat tiles + ligament buckling. (`.frd` files for all
  w exist in /tmp/hinge/; parse_frd reads 2C node block + last DISP block via regex on E-floats.)
- **EXP-1 CUT SPEC (0.5mm PET, laser):** w_lig=15mm, α=90°, fillet ρ=2.4mm, kerf~0.2mm; tiles extend ≥40mm
  beyond ligament (rigid past the 72mm domain) + ~25mm taped grip tabs; pull ∥ secondary cut (φ=0 shear).
- **★ STRAIN-HARDENING COMPARISON = doable & standard** (searched): matching the post-yield hardening
  region (not just yield) of the σ-ε / F-s curve between experiment & FE is standard model validation. In
  our shear test = the rise from ~90N knee → ~128N peak. **⚠ CRITICAL polymer caveat (confirmed): force-
  displacement ALONE is insufficient to calibrate/validate polymer constitutive models — need FULL-FIELD
  DIC.** ⇒ endorses the 2-camera plan (front dots = in-plane strain/E, side = uz). Refs: ScienceDirect
  S0734743X11000054, S0167663607001639, S0022509622003374 (polymer DIC+IR+FE).

## ★ EXP-1 w_lig = 18 mm — matches the coupon width (user decision, 2026-07-25)
⚠ **CORRECTION:** an earlier attempt to derive a "design w_lig" from a config comment ("tile = 10×w_lig
→ ~55 mm") was WRONG — I over-read a single config COMMENT as a universal design rule; it is NOT a real
constraint. (It never saved to memory; the only "55 cm" in memory is the legit gravity-calc face estimate.)
Do not repeat it. **Actual EXP-1 decision: w_lig = 1.8 cm = 18 mm**, chosen to **match the coupon width**
(the tensile strips were ~18 mm), so the ligament cross-section = the coupon cross-section (18 × 0.5 mm).
This holds cross-section FIXED so the only difference vs the plain coupon is the **hinge geometry** (cut,
notch, out-of-plane buckling) → a CONTROLLED experiment to see if geometry significantly changes the
behavior. Sim-reliable (like 15 mm), driven by instrument+solver ability. **Supersedes the 15 mm pick.**
Re-running the w=18 mm shear prediction; geometry/cut spec otherwise as EXP-1 (α=90°, ρ=0.16·w_lig≈2.9 mm,
t=0.5 mm, tiles ≥40 mm + tabs, pull ∥ cut).
- **w=18mm SHEAR PREDICTION (ran):** peak **F_s ≈ 150 N @ s≈16.6mm**, **fractures s≈16.6mm (s/w_lig=0.92**
  — same "fail at s≈w_lig" rule), uz_max ≈11.6mm, early notch-yield s≈0.5mm/~33N then bulk knee+hardening.
  `data/experiments/processed/pet_hinge_w18_shear.npz`. Comfortably on the 5kN cell.

## ★ ELASTO-GRAVITATIONAL LENGTH — the right size scale (user's concept, 2026-07-25)
User's intuition (correct): hinge sizing should use the **elasto-gravitational / gravito-bending length**
**L_eg = (B/(ρ_s g))^(1/3)**, B = E t³/12, ρ_s = ρt. For PET (E=3GPa, t=0.5mm, ρ=1390): B≈0.031 N·m,
ρ_s≈0.70 kg/m², **L_eg ≈ 17 cm** (≈ the "domain ≤20cm" from before; note w/o the /12 plate factor it's
~38cm — standard def uses B ⇒ ~17cm). **L_eg does TWO things at TWO scales (don't conflate):** (1) MIN
FACE size — face ≥ L_eg ⇒ gravity flattens it (coplanar); user's ~55cm faces ✓; (2) it CAPS the out-of-
plane deformation domain (gravity kills buckling beyond ~L_eg). ⇒ **deformation domain ≈ min(2.4·w_lig,
L_eg)**; for w_lig=18mm that's 2.4·18=4.3cm ≪ L_eg (ligament-set); only w_lig≳7cm hits the L_eg cap.
**⚠ CORRECTED (2026-07-25) — "grips substitute for gravity, small tiles OK" was WRONG.** See d-cone result.

## ★★ d-CONE CONFIRMED — hinge domain is LONG-RANGE (log), grip distance = domain (2026-07-25)
User's idea (buckled hinge = developable cone + relaxation length) = CORRECT. **Domain-convergence study
(w=18mm shear, r_win=27/43/72/108mm):** peak F_s = 166/150/127/110 N — **cleanly LOGARITHMIC:
F_s = 303 − 41.2·ln(r_win), R²=0.997.** = textbook d-cone signature: deformation decays ~1/r (LONG-RANGE),
NOT Saint-Venant exponential ⇒ **NO convergence plateau; the 2.4·w_lig heuristic was an arbitrary point on
a log curve.** Energy of a cone E ~ D·ψ²·ln(R_out/R_core), D=Et³/12; only the log of (outer/core) radii
matters.
- **VALUE:** extrapolate log law to the PHYSICAL outer radius: at L_eg=170mm → **F_s ≈ 92 N** (vs 150 N at
  the truncated 43mm). ⇒ **the 2.4× heuristic OVERESTIMATES the hinge force by ~60%.** Physical w=18mm
  prediction ≈ **90 N**, not 150 N.
- **★ CONSEQUENCE (corrects earlier claim): the GRIP DISTANCE = the effective domain.** Because the cone is
  long-range, gripping the coupon tiles CLOSE (~5cm) truncates the cone → measures a ~60%-too-stiff force.
  The real deployed hinge's cone is bounded by GRAVITY at L_eg≈17cm. ⇒ **coupon must have tiles ≥ ~17cm and
  be GRIPPED AT ≈ L_eg (~17cm from the ligament), NOT close** — else it over-stiffens. This is the "vital
  assumption" the user flagged; it IS vital (~60%), resolved by grip-at-L_eg.
- Caveat: peak F_s is plastic (at fracture) yet fits log R²=0.997; elastic stiffness drops even faster
  (in-plane shear compliance, messier). The log/d-cone is the right frame; ~90N is the L_eg-domain number.
- **TODO:** update EXP-1 coupon spec → tiles ≥17cm, grip at L_eg; re-state w=18 prediction as ~90N.
(a) draft `PETIsotropic` placeholder class; (b) inverse-calibration script (fit ccx material params to
a measured hinge curve); (c) spec exact coupon/hinge dims for the sheet + laser. Gate everything on
Phase 0a. `ccx` = CalculiX 2.23 in the `ccx` conda env (run via `conda run -n ccx --no-capture-output`).

## ★★ w_lig CALIBRATION SWEEP — 20cm domain, predicted shear family (2026-07-26)
User launched the validate-or-calibrate experiment: sweep `w_lig` = **1,2,3,4,5 cm (=10-50mm)**, **2 phys
replicates each**, cut real PET + match ccx; compare buckle height, cone width, shear tear curve.
- **DOMAIN CHOSEN = 20cm WIDE = 10cm RADIUS ⇒ `r_win=100mm`** (user's explicit call, supersedes the L_eg
  =170mm option). Coupon spec accordingly: tiles ≥100mm, **GRIP AT ~100mm** (grip dist MUST = sim r_win, per
  the d-cone/[[grip=domain]] finding), +25mm taped tabs, α=90°, fillet ρ=0.16·w_lig, kerf 0.2, t=0.5, pull ∥
  cut (φ=0 shear). Note: 10cm < gravity L_eg~17cm ⇒ reads a bit stiffer than the deployed hinge, but sim↔exp
  match is exact since both use 10cm (extrapolate log law 10→17cm for the deployed number).
- **★ GOTCHA — shear ray is ramped by `theta` (pseudo-time):** in `DeploymentRay`, `frac=theta_deg/theta1_deg`
  drives a,s. `--theta 0` FREEZES s at 0 (whole run useless, F_s flat/garbage). For shear mode set a SMALL
  nonzero `--theta` (2°) as the loading parameter + `--eta-s` for the shear magnitude (s1=eta_s·w_lig). The
  known-good w18 run used theta≈2. Fixed the sweep this way.
- **PREDICTIONS (r_win=100, eta_s=1.3, 30 steps, `data/experiments/processed/pet_shear_w{10..50}_rwin100.npz`,
  fig `pet_shear_sweep_rwin100.png`, script `nff/scripts/figures/plot_pet_shear_sweep.py`):**
  * w=10mm: SURVIVES to s=13mm (D=0.63, most ductile), F≈58N rising, uz 15.8mm.
  * w=20mm: tear s≈23mm, F≈137N, uz 22.9mm.  w=30mm: tear s≈28mm, F≈217N, uz 23.9mm.
  * w=40mm: tear s≈33mm, F≈285N, uz 22.3mm.  **w=50mm: DIVERGED** (n=4 incr, wide-solid mesh aspect; needs
    n_through=3 / shell fallback if that point is wanted).
  * LAWS: **force-at-tear ∝ w_lig (area), tear at s≈w_lig (shear strain≈1), buckle uz≈16-24mm collapsing onto
    ~one uz(s) curve to s≈10mm.** All forces trivially on the 5kN cell.
- **★ PERF GOTCHA — ccx ran SINGLE-THREADED:** `rve/ccx_solver.solve_job` defaults `ncpus=1`
  (OMP_NUM_THREADS=1) and `run_pet_hinge_trial.py`/`deploy` never override ⇒ **~9-12 MIN per solve** (w20
  11m51s, w30 8m45s; ~50min/sweep). To speed re-runs: thread the equation solver (pass ncpus=4-8) and/or run
  the 5 widths in parallel. Not yet wired into the trial CLI.
- **NEXT:** overlay real coupon F_s(s)+uz on the fig (force=load cell, s=front-dot video, uz=side cam); decide
  w=50 re-run vs cap at 40mm; then inverse-fit eps_f0/k if curves miss.

## ★★ FIRST REAL w=18mm HINGE VALIDATION — fold VALIDATES, opening over-predicts ~1.6× (2026-07-26)
User ran TWO physical experiments on ONE real w_lig=18mm PET hinge (t=0.5, α=90, fillet f=2.88mm=0.16·w):
**(1) FOLD/rotation OOP** (Lh=200mm coupon → r_win=100mm): buckle height `bh` + width `bw` vs θ, θ=0..90°,
file `Kirigami experiments - Hinge calibration OOP.csv` (decimal commas; bh baseline-corrected = raw−6.32).
**(2) TENSION/pure-opening** (⊥ cut confirmed, grip sep 50mm → r_win=25mm): Bluehill Time,Disp,Force(kN),
`/Volumes/KINGSTON/.../20260726_114859_1_2.csv`, 1mm/min to 9.27mm crosshead. Parsed → `real_w18_fold_bh.csv`,
`real_w18_fold_bw.csv`, `real_w18_shear.csv`. Figs: `pet_w18_calibration.png` (fold + tension 2-panel), script
`nff/scripts/figures/plot_pet_w18_calibration.py`.
- **FOLD = VALIDATED (no material tuning):** sim `uz(θ)` (imp_amp=0.5mm) sits ON the real `bh` through 50°,
  BOTH saturate ~25mm (sim 25.0 / real bh 25.5). imp_amp is the ONLY knob (T3). Thin-solid sim converges to
  ~50-58° (localization wall, as paper); real `bh` already plateaus by 40° so nothing lost. Real `bw` (cone
  base width) saturates at 151mm from θ=40° (= domain-bounded). **Visible damage first at θ=20°** (D-anchor).
- **TENSION/opening = model OVER-predicts ~1.6×:** sim peak F_a=**378N** @ a≈2mm, tear a=4.3mm, **uz=0.2mm
  (NO buckling — matches user's "no buckling in tension" ✓)**; real peak **233N** @ ~7mm crosshead, ductile
  plateau 3-7mm then 28% drop (tear). Sim 378N=0.86×full-section tensile yield (σ_y·A=48.8·9=439N); real
  233N=0.53× (eff. draw stress ~26MPa vs coupon 48.8). Same SHAPE (elastic→yield/draw plateau→tear), right
  mode. ⇒ GENUINE calibration target. Stiffness gap likely NOT real (real x=crosshead compliance; need video).
  Candidate causes ranked: (1) real min ligament width <18mm (laser kerf/fillet — MEASURE with calipers, ~0.6×
  factor); (2) laser HAZ edge weakening; (3) material failure law (eps_f0/k/yield). Opening peak is r_win-
  insensitive (notch-dominated) so 378N robust.
- **★ MODE LESSON:** the 1-DOF Instron pull was PURE OPENING (axial `a`, eta_a), NOT shear (eta_s) — I first
  ran shear by mistake. Also: FOLD coupon gripped at 200mm (r_win=100) but TENSION coupon at 50mm (r_win=25) —
  DIFFERENT domains per test; match each test's actual grip to its sim r_win.
- **NEXT:** (a) measure real min ligament width; (b) video-measure ligament opening → fix tension x-axis +
  true stiffness/disp-to-tear; (c) inverse-calibrate ccx (yield/eps_f0/k) to the opening curve; (d) confirm
  opening r_win insensitivity (25→100).

## ★ SESSION CODE CHANGES / GOTCHAS (2026-07-26)
- **FRD PARSER BUG FIXED** (`nff/rve/ccx_solver.py::_parse_frd` ~line 264): the stop-at-fracture kill leaves a
  TRUNCATED ` -4` field header → `ln.split()[1]` raised IndexError, silently killing ANY solve that tears.
  Guard added: `if len(parts)<3: break` (keep complete frames). This was corrupting most tearing solves.
- **evaluate_hinge + run_pet_hinge_trial now expose:** `ncpus` (thread the ccx eq. solver — solves were
  SINGLE-THREADED = 9-12min each; ncpus=4 ~halves it), `imp_amp`/`min_inc` (buckle seed + increment floor;
  imp_amp is the fold OOP calibration knob, default 0.3·t=0.15mm too small → rotation snap-diverges),
  `workdir` (⚠ geo/ray tag DOESN'T include imp_amp → parallel runs sharing geometry CLOBBER each other; pass
  distinct --workdir). HingeConstants gained `imp_amp`,`min_inc` fields; solver_kwargs forwards them.
- **SHEAR/AXIAL ray driven by θ pseudo-time:** `frac=theta_deg/theta1_deg` ⇒ `--theta 0` FREEZES a,s at 0
  (garbage). Set small `--theta 2` + `--eta-s`/`--eta-a` for shear/opening magnitude.
- **`*STATIC, STABILIZE` added** (`ccx_solver._write_deck` `stabilize` param → prepare_job → HingeConstants
  `stabilize` field → solver_kwargs → CLI `--stabilize`). Off by default (deck byte-identical). Adds viscous
  damping to walk past buckling snaps; ⚠ CONTAMINATES F/M (viscous reaction) → use for uz-only runs.

## ★★ TENSION GAP = PRE-DAMAGED SPECIMEN + bz(θ) BUCKLE-HEIGHT FORMULA (2026-07-26)
- **The tension over-prediction (378 vs 233N) is CONFOUNDED — same hinge was folded to 90° FIRST** (visible
  damage from θ=20°) so the ligament was pre-creased/micro-torn before the tension pull ⇒ real yields/tears
  LOW. **DO NOT calibrate eps_f/yield to that curve.** User redoing tension on a FRESH hinge (should pull
  higher, toward the sim). Prior-damage is the leading explanation, not a material-law error.
- **E is NOT the knob for the force gap (demonstrated):** re-ran opening at E=1.2GPa (crosshead lower-bound)
  vs 3.0: peak F_a 378→370N (−2%, UNCHANGED) while elastic stiffness halved 591→245 N/mm (tracks the 2.5× E).
  Peak force is set by yield·section+hardening+eps_f, E-INDEPENDENT. Using 1.2GPa would only bake machine
  compliance into the material (1.2 was itself a compliance artifact) — wrong fix. Slope fix = video disp.
- **★ FOLD-90° SIM: stalls ~57° even with STABILIZE** (st=1e-4 & 3e-4 both die ~56-57°, uz≈24.9mm; hard
  localization wall = the paper thin-solid wall). But uz already at the SATURATED PLATEAU by 57° = real
  plateau ⇒ buckle-height validated; literal 90° march needs the SHELL path (Phase 3). imp_amp=0.5 fits best.
- **★★ bz(θ) BUCKLE-HEIGHT FORMULA (fit to real w18 bh, θ 0-90°):**
  * PHYSICAL (1-param, recommended): **uz = 25.7·√(sin θ) mm, R²=0.97.** θ→0: uz≈25.7√θ = post-buckling √-law
    (lit. θ₀²≈ε−ε_c, [[Rafsanjani-Bertoldi Buckling-Induced Kirigami arXiv:1702.06470]]); θ→90: saturates as
    finite ligament runs out (e-cone/developable-cone geometry, arXiv:2109.03019 / JMPS S0022509621000612).
    Prefactor A≈25.7mm = ligament out-of-plane "reach" (should scale w/ w_lig — test via the sweep).
  * EMPIRICAL (2-param, tightest): **uz = 24.0·(1−e^(−θ/15.7°)) mm, R²=0.99.**
  * CalculiX independently tracks the fit to 57°. Fig `data/experiments/processed/pet_w18_fold_formula.png`
    (`nff/scripts/figures/plot_pet_fold_formula.py`); fit script inline (no scipy in ccx env → analytic
    1-param + grid 2-param). Real bw (cone base width) saturates 151mm from θ=40°.
- **★ EFFECTIVE LIGAMENT WIDTH = w_lig·(1−fillet_ratio) = w_lig − ρ** (fillet disk is centered on the main-cut
  TIP so its top edge rises ρ toward the secondary cut, shortening the neck). Verified vs `build_rve_domain`:
  w18/fr0.16 → narrowest throat **15.12mm** at x=0 (not 18) → area 7.56mm² (not 9). ⇒ sim opening peak 378N =
  full yield at 15.12mm (369N); my earlier 439N hand-estimate used nominal 18 (wrong). ALL session sims used
  this filleted geometry (fillet_ratio=0.16) — the solver was always right; only my hand number was off.
- **★ FILLET SWEEP (w18 fold, fr=0.08/0.16/0.24/0.32, `pet_w18_fold_fr*.npz`, fig `pet_w18_fillet_sweep.png`,
  `nff/scripts/figures/plot_pet_fillet_sweep.py`): BUCKLE HEIGHT ≈ FILLET-INDEPENDENT.** Prefactor A of
  uz=A√(sinθ) = 28.9/28.1/29.1/30.1mm → **A≈29±1mm over a 4× fillet range** (mild +7% w/ larger fillet).
  ⇒ the single formula uz≈29√(sinθ) is usable across fillets for w18; fillet does NOT set buckle amplitude.
  Fillet's real leverage = (a) SOLVABILITY (fr=0.08 sharp tip diverged at θ=6° — its A is unreliable), (b) the
  FAILURE/tear point (stress concentration). NEXT lever to test = w_lig scaling of A (expect A∝w_lig-ish).

## ★ FRESH (undamaged) w18 TENSION + VIDEO GRIP-TRACKING (2026-07-26)
- **Fresh hinge opening** (`/Volumes/KINGSTON/.../wlig18_nodamage.csv`, 1mm/min): peak **287N @ 3.9mm**
  crosshead, then only **3% drop, NO tear** to 9.66mm (vs damaged 233N + 28% tear). Prior damage cost ~19%
  peak + caused the tear. Parsed → `real_w18_open_nodamage.csv`. **Sim opening 379N** ⇒ gap now **~1.3×**.
- **OPENING FORCE IS r_win-INSENSITIVE:** peak F_a = 379/378/376N at r_win=25/50/100 ⇒ grip distance does NOT
  explain the gap (opening is notch/section-dominated, not the long-range d-cone). Gap = yield×section (+eps_f
  too low: real didn't tear). eps_f=1.2 ⇒ RAISE (real PET very ductile).
- **RATE (1mm/min):** NOT a confound for the sim (ccx is rate-independent J2; the *PLASTIC law was built at
  1mm/min) — same regime, so it doesn't explain the 1.3×. IS a real factor for FAST deploy (~cm/s = 2-3 decades
  faster → stiffer+brittler → slow eps_f optimistic). Handle later via bounding rate study (1/10/100/1000
  mm/min = Instron ceiling) + margin on eps_f/σ_y, or add rate-dependence. Not now.
- **INVERSE-CAL IDENTIFIABILITY:** force-only curve fixes **yield×effective-section** (degenerate — can't split
  "lower yield (HAZ)" from "smaller neck" without measuring the neck). Effective neck = **w_lig·(1−fillet_ratio)
  = 15.12mm** already (fillet). Minimal fit = 1-param yield-scale to the 287N plateau (~0.76×) + eps_f0 lower
  bound from no-tear; k + full shape need the φ-sweep-to-break + video. Principle: material from COUPONS; only
  imp_amp is a legit hinge-specific fit; a residual force knock-down = documented HAZ factor, NOT bulk-law edit.
- **★ VIDEO GRIP-TRACKING (pure-numpy ZNCC tracker, cv2 absent both envs):** 1.7GB 4K→ **low-res stored
  `data/experiments/video/w18_tension_lowres.mp4`** (720p, 3.4MB). Tracked grip collets + cut-dot from 960x540
  1fps raw frames (`scratchpad/track.py`+`sync.py`, fig `data/experiments/processed/w18_video_sync.png`).
  RESULT: **top grip moves (linear ramp, 1px-quantized), bottom grip FIXED, no toe/slip** ⇒ **crosshead is a
  faithful slip-free grip-separation signal** (axis NOT a machine artifact — earlier worry overturned). BUT
  **cut-dot tracking FAILED** (transparent film, wrinkling ligament → lost lock) ⇒ could NOT get pure ligament
  opening. Grip-vs-crosshead R²=0.998 only confirms both linear ramps (can't exclude CONSTANT compliance w/o an
  absolute in-frame scale). **KEY: grips sit ~25mm each side = sim r_win=25 ⇒ crosshead axis ≡ sim handle `a`
  basis** → the sim-vs-real overlay WAS fair; can calibrate force+stiffness on crosshead. **To ISOLATE ligament
  opening / pin E: NEXT experiment needs 2 high-contrast INK DOTS straddling the cut** (video-extensometer plan).
- **★ FILLET-HOLE OPENING (user's idea: track top/bottom of the fillet hole = differential opening + built-in
  scale) — CONCEPT SOUND, AUTO-TRACKING FAILED on this footage.** The laser fillet is a clear dark ~2.4-4.4mm
  hole (nominal 2ρ=5.76mm; physical smaller) that visibly opens vertically under the vertical pull (cut runs
  horizontal). Extracted 4K fillet crops (`fillet_crop_800x600_1fps.gray`), pure-numpy segmentation (fixed
  window + seed-following window, `scratchpad/fillet_open.py`/`fillet_track.py`) — BOTH break mid-test (opening
  reads NEGATIVE then jumps/sticks). Cause: transparent film (low/variable contrast), stress-whitening "lens"
  over the hole, dark machine background at the film edge, no scipy/cv2. Scale from grip-tracking = 0.060mm/px
  @4K (not the hole-diameter assumption). **NOT worth more CV on this clip.** Doesn't block calibration: the
  FORCE gap (287 vs 379N) is displacement-axis-INDEPENDENT; fillet opening would only remove ~1-2mm tile stretch
  to refine E. Paths: hand-digitize ~10 frames (±10%), OR ink-dot specimens next batch (clean), OR skip &
  do the force-level fit now (recommended). Low-res video: `data/experiments/video/w18_tension_lowres.mp4`.

## ★ MANUSCRIPT SECTION WRITTEN (2026-07-26): `docs/sn-article.tex` §Calibration
Added `\subsection{Calibration of the surrogate to a physical material}` (Methods, after the RVE
subsubsection) — the paper-voice writeup of this whole campaign. Covers: the chain eq. coupon→bulk
law→ccx RVE→W→deployment + the **material-vs-structure separation principle** (bulk law from coupons
ONLY; hinge tests VALIDATE; `imp_amp` = the sole free structural knob); PET ID; D882 campaign
(isotropic σ_y 48.8±1.6, cold-draw anchor σ_true=λσ_plateau @ ε=lnλ → 109 MPa @ 1.15, J2 justified by
plastic incompressibility + ±45° bands); `tab:pet-params`; the "what coupons can't give" para (E is a
lit. placeholder, peak force E-insensitive by 2%; eps_f0 = lower bound; damage locus
eps_f0·exp[−k(η−1/3)] — matches `damage.py::fracture_locus`, note the −1/3 normalization); the
**grip-angle map (η_a,η_s)=(sinφ,cosφ)** framing ("calibrate in-plane, predict rotation");
**d-cone log truncation** F_s=303−41.2·ln r_win ⇒ grip separation MUST equal sim r_win (=100mm);
w18 fold validation `u_z=A√(sinθ)`, A fillet-independent, + the opening pre-damage confound; rate margin.
- ⚠ `sn-jnl.cls` NOT installed on this machine → full manuscript can't be built; verified by compiling
  the section standalone in an `article` wrapper (clean, 4 pages).
- ⚠ `docs/sn-article.tex` was UNTRACKED → **PR #16** (`worktree-paper-calibration-section`, off main)
  adds the whole file. To get it into the working copy: copy from the worktree/branch.

## ★★ DECISION (2026-07-26): HINGE VALIDATES, DOES NOT CALIBRATE — HAZ knock-down REJECTED
User principle [[feedback_hinge_validates_not_calibrates]]: material params come from COUPONS ONLY; the
cut-hinge is a VALIDATION test, never a calibration knob. ⇒ the **×0.757 laser-HAZ yield knock-down is
NOT applied** to `PETIsotropic` (would be circular). The opening **287 (real) vs 379N (sim)** gap is now
an **OPEN VALIDATION DISCREPANCY** (section verified neck=15.11mm=design; mesh-converged n_through 2/3/4
all 379N; r_win-insensitive; E-independent → the 1.3× is a genuine effective-yield gap, cause TBD: real
HAZ vs coupon under-characterisation vs part-edge effect). `eps_f` must come from a coupon RUN-TO-BREAK,
not the hinge no-tear. Next question the user raised: **were the COUPONS fully exploited?** (E unmeasured/
=3.0 lit.; eps_f only a ≥1.2 lower bound, never run to break; *PLASTIC only 2 pts from full σ-ε curves;
rate/permanent-set/ν not done) — several fixable from EXISTING raw curves (`data/experiments/raw/
kirigami_20260723/`), others need new coupon tests. `imp_amp` (fold) is the one allowed hinge-informed
seed (numerical, not material).

## ★★ MULTI-POINT *PLASTIC FROM COUPONS → CLEAN VALIDATION 379→319N (2026-07-26)
Rebuilt the `*PLASTIC` table from the 9 raw cold-draw coupon curves (`data/experiments/raw/
kirigami_20260723/`), COUPON-ONLY per [[feedback_hinge_validates_not_calibrates]]. Method (`scratchpad/
build_plastic.py`): neck propagates at ~constant ENG force ⇒ by plastic incompressibility each shoulder
point carries true stress **σ_true(ε)=σ_p·exp(ε)** up to ε=ln λ — the intrinsic flow curve, no DIC needed.
Extracted per-specimen: **upper-yield peak mean 44.0MPa** (NB: LOWER than the 48.8 baked in pet.py — 48.8
was likely 0.2%-offset; the peak-yield is 44), plateau 33.6, λ=3.28, true-draw 110@ε1.19. Table (monotonic,
ccx needs non-decreasing so the real post-yield DIP to ~34 is FLATTENED to 44): `[(44,0),(44,0.27),(45.3,
0.3),(55.3,0.5),(67.6,0.7),(82.5,0.9),(100.8,1.1),(110.2,1.19)]`. **17-30% SOFTER than the old 2-pt linear
at ε_p 0.3-0.8** (the hinge's operating range).
- **VALIDATION (opening, E=3.0, NO hinge fit, `pet_w18_open_multipoint.npz`, `scratchpad/validate_
  multipoint.py`): 319N** vs old-2pt 379N vs **exp 287N**. ⇒ coupon material fix removed **~65% of the
  discrepancy (gap 32%→11%)** with zero hinge input. The 2-pt straight line (discarding the yield-drop) WAS
  the dominant over-prediction cause — user's instinct confirmed. **Clean validation at ~11%.**
- Residual +11%: (a) coupon yield scatter 36-50MPa (~±10%, near noise floor); (b) MONOTONICITY — the real
  ~34MPa yield dip had to be flattened to 44 (true dip would drop force further toward 287); (c) maybe small
  real HAZ. None justify a hinge fudge. PEEQ=1.35 at a=5.9mm no-tear ⇒ eps_f≥1.35 (needs run-to-break).
- ✅ **COMMITTED to `pet.py` (2026-07-26):** `PET_PLASTIC` = the 9-point coupon table above; docstring
  updated (yield-peak 44 vs 0.2%-offset 48.8); `eps_f0` default **1.2→1.5** (+ CLI `--eps-f` 1.5). 18
  material golden tests pass; PET deck verified emitting the new *PLASTIC. E=3.0 still lit. placeholder.

## ★ THREE-MODE SUMMARY FIGURE + calibrated shear prediction (2026-07-26)
`nff/scripts/figures/plot_pet_three_modes.py` → `data/experiments/processed/pet_w18_three_modes.png`:
A buckling (sim+real bh, VALIDATED), B shear (PREDICTION only, no data), C tension (sim 319 + real 287
crosshead, VALIDATED ~11%). All w=18mm, committed calibrated PET (9-pt *PLASTIC, eps_f0=1.5, E=3.0).
- **SHEAR PREDICTION (calibrated, r_win=25 = 50mm grip, `pet_w18_shear_calibrated.npz`):** elastic → yield
  knee ~100N @ s≈1.5mm → harden → **tear s≈14mm ~145N**; uz_max 6mm (shear DOES buckle, d-cone). ⚠ shear
  force is GRIP-DISTANCE-sensitive (long-range) — physical shear test MUST grip at 50mm to match.
- ⚠ Panel C x-axes differ (sim=ligament opening, real=crosshead+tile stretch) → compare FORCE, not disp.
- NEXT open validation = a real SHEAR pull (fills panel B; run-to-break also helps pin eps_f/k).

## ★★ FILLET-HOLE VIDEO EXTENSOMETER WORKS → STIFFNESS VALIDATED, E≈3.0 CONSISTENT (2026-07-26)
The user's fillet-hole idea WORKS after all — the earlier auto-track failures were method (drift/background),
not the footage. **Fix: SEED in a TIGHT BOX on the hole path** (`fillet_crop_800x600` y[150:430] x[370:525],
per-frame adaptive thr=min+45, max vertical dark run) → 94% monotonic, scale 0.056mm/px (matches grip-derived
0.060), rest hole ~4mm, **true opening a_max 3.9mm < crosshead 8.2mm** (⇒ HALF the crosshead was tile stretch+
machine compliance, incl. a visible seating toe). Script `scratchpad/fillet_seed.py`, fig `data/experiments/
processed/w18_fillet_seed.png`.
- **★ RESULT: on the TRUE ligament-opening axis the calibrated sim & experiment ELASTIC SLOPES OVERLAY**
  (both reach plateau by ~0.7-1mm; exp plateau ~280N, sim peak 319→280). ⇒ **the "model much stiffer in
  tension" was ENTIRELY the crosshead axis (tile+machine compliance), NOT real.** Stiffness VALIDATED.
- **E≈3.0GPa now CONSISTENT/validated** (elastic slope matches on the true axis) — the unmeasured-placeholder
  worry is resolved for opening; per [[feedback_hinge_validates_not_calibrates]] this VALIDATES E, doesn't set
  it. Remaining sim-vs-exp difference = the ~11% force LEVEL only (plastic/yield), not stiffness.
- ⚠ caveats: compliance toe + pixel quantization in the extracted opening (slope match is "clearly
  consistent", not sub-%); exp only reached ~4mm opening (video ended) so the sim's later geometric softening
  is unobserved. Method now proven → future videos trackable via fillet-seed even without ink dots (though
  ink dots straddling the cut would still be cleaner).

## ★★ LOAD-TRAIN COMPLIANCE QUANTIFIED: C = 7.77 µm/N — halves the E gap, doesn't close it (2026-07-27)
Turned the qualitative "crosshead is inflated" into a **measured constant**, then pushed it back onto the
coupons. New module `nff/calibration/compliance.py` (`C_HINGE_W18`, `fit_compliance`,
`correct_displacement`, `modulus_from_slope`) + figure `nff/scripts/figures/plot_machine_compliance.py`
→ `data/experiments/processed/pet_machine_compliance.png` (A two disp axes, B the C fit, C coupon E at
the origin, D tangent-E roll-off vs stress).
- **C = 7.77 µm/N**, offset −15 µm, **residual 63 µm rms** over F=30–278 N ⇒ the excess `x_crosshead −
  a_video` is *beautifully linear* in force (a true series compliance, not slip). At F=278 N: crosshead
  3.43 mm vs true opening 0.96 mm → **3.55×, i.e. 72% of the crosshead travel is the machine.**
  `crosshead − C·F` collapses onto the video curve (panel A) → the linear model is sufficient.
- **Tile stretch is NOT the culprit** (earlier guess): from video geometry (free span ≈39 mm, film ≈77 mm
  wide, t=0.5) the tiles contribute ≈0.34 µm/N ≈ 4% of C. So C is essentially **frame + grips + tab/film
  seating**, i.e. genuinely machine, and it transfers between specimens (approximately).
- **★★ Coupon E, series-compliance bookkeeping** (`dx/dF = C + L/(EA)`, n=10) → **E = 3.02 ± 0.91 GPa**
  from raw 1.03 ± 0.12. **The compliance correction alone recovers the literature 3.0 GPa** — the C that
  would land exactly on 3.0 is 8.14 ± 1.42 µm/N = **1.05× the measured C**. No extra hypothesis needed.
- **⚠ THE WINDOW IS EVERYTHING (user's correction, and the whole point):** E must be the **INITIAL
  TANGENT**, `TANGENT_WINDOW_MPA = (1, 5) MPa`. PET bends away from linear within a few MPa (anelastic
  pre-yield), so corrected E rolls off **3.4 (σ~2) → 3.0 (σ~3) → 1.8 (σ~8) → 1.35 GPa (σ~12)** — a chord
  at 5–20 MPa understates E by ~2× *even after* the correction, and was my first (wrong) answer.
  Below ~1 MPa the specimen compliance drops under C and the subtraction goes singular. Panel D plots this.
- ⚠ **Leverage caveat:** in the 1–5 MPa window the specimen carries only **37%** of the measured slope
  (12.33 µm/N total, C=7.77), so a relative error in C is amplified **~2.7×** in E — hence ±0.91 GPa
  scatter. Central value is right on 3.0; a coupon-gauge video extensometer would still tighten it.
- **Nothing in `pet.py` changes.** Yield (44) and the *PLASTIC cold-draw anchors come from force/area and
  the measured λ — both compliance-immune. Only E was ever crosshead-derived, and E barely moves the hinge
  (3.0→1.2 GPa = −2% force). Per [[feedback_hinge_validates_not_calibrates]] this is validation, not fitting.
- **The residual ~11% hinge force gap is untouched by any of this** — compliance is a *displacement-axis*
  artifact; the gap is a *force level*. Candidates stay: coupon yield scatter (36–50 MPa), the monotonicity
  constraint flattening the real ~34 MPa post-yield dip up to 44, possibly a small real laser-HAZ effect.
- Fillet-seed extraction hardened: `V = np.maximum.accumulate(medf(V,7))` — opening is monotonic under a
  monotonic pull, so glitch dips (reflection/whitening shrinking the detected dark span for 1–2 frames)
  no longer fold the F-vs-a curve backwards. Video geometry verified: the 800×600 crop is at NATIVE 4K res
  (best match at full-frame (1900,1000), MAD 0.51) ⇒ the `scale_960/4` factor is right.
- Panel C of the three-mode figure now shows **`crosshead − C·F`** (dashed) instead of the raw crosshead,
  alongside the video true-opening curve.
- **Fit figure:** `nff/scripts/figures/plot_pet_coupon_modulus.py` → `pet_coupon_modulus.png`
  (A the corrected curves + per-specimen tangent fits in the window; B out to yield, showing the
  tangent only holds for the first ~0.17% strain). Cross-check: **ISO 527-1 secant (ε 0.05–0.25%) on
  the corrected axis = 2.64 ± 0.56 GPa** — our 1–5 MPa window IS essentially the ISO window.
- **WHY it bends so early (physics, worth keeping):** PET at RT is ~50°C below Tg but its amorphous
  β-relaxation is active ⇒ **viscoelastic from zero load**, no true elastic limit; the anelastic strain
  accumulates with TIME, so at a constant crosshead rate the tangent decays continuously. Above a few
  MPa it goes **stress-activated (Eyring)** — stress biases the segmental-jump barrier, anelastic rate
  grows ~exponentially ⇒ tangent collapses 3.0→1.4 GPa between 5 and 15 MPa, then flattens until yield.
  BOPET's stiff-crystallite / compliant-amorphous composite gives the high initial value. **Rate helps
  us:** 1 mm/min on 114 mm = 1.4e-4 /s is the slowest, most creep-favourable case; at deployment rates
  (cm/s) the linear window extends and PET looks stiffer.
- **Irrelevant to the hinge force:** our ccx model is linear-elastic + J2, no viscoelasticity, so it is
  stiffer than real PET between ~5-44 MPa — but the ligament crosses that band instantly and the measured
  sensitivity is tiny (E 3.0→1.2 GPa = −2% force). E is for stiffness reporting; *PLASTIC carries the force.

## ★★ DAMAGE MEASUREMENT — the fracture ruler vs the FUNCTIONAL one (2026-07-27)
Figure `nff/scripts/figures/plot_pet_damage_ruler.py` → `pet_damage_ruler.png` (A fold PEEQ/D vs θ,
B all three modes vs deployment fraction, C the "damage ruler" of what experiments actually pin).
- **Model:** `D = PEEQ/eps_f(η)`, `eps_f = eps_f0 exp(-k(η-1/3))`, p99 over ligament elements. Three
  buried choices: plastic strain only (elastic ignored), **memoryless** (valid only on monotonic
  proportional rays), and a **p99** aggregate (never checked for percentile sensitivity).
- **Numbers:** fold to 57° → PEEQ 0.12, **D = 0.08** (folding is nearly damage-free in the tear sense).
  Opening crosses D=1 at **a ≈ 0.37 w_lig ≈ 6.7 mm**; shear at **s ≈ 0.78 w_lig ≈ 14 mm** ⇒ opening is
  ~2× more damaging per unit displacement than shear; folding is a different league.
- **★ THE KEY MISMATCH:** the real fold showed **visible stress-whitening from θ≈20°**, where the model
  reads PEEQ 0.043 ⇒ **D = 0.029**. Not a model error — a **definition** mismatch. D measures *distance
  to fracture*; whitening measures *onset of irreversible change* (crazing/lamellar unfolding). For PET
  those are ~30× apart in strain. ⇒ **we need a SECOND, functional damage measure** (residual fold angle
  after release + stiffness on reload), simulable by adding an unload step. ⚠ For the closed pipeline
  (deploy and HOLD a shape) plastic set is a FEATURE, not damage — be explicit which regime is meant.
- **★ k is nearly unidentifiable AND nearly irrelevant here:** backing η out of the runs via
  `peeq_p99/D_p99` gives η ≈ **0.33 (fold)** and **0.41 (shear)** — a narrow band around uniaxial
  tension where `exp(-k(η-1/3))` moves only ~10%. **eps_f0 alone carries the prediction** inside the
  hinge's operating envelope. One number to measure, not two (unless we stray to high triaxiality).
- **Experiments only give LOWER BOUNDS so far:** coupons drew to ε_true 1.19 UNBROKEN (none was run to
  break — all 10 were stopped mid-plateau at ~16-21% crosshead strain; 4.2 broke prematurely at 6% =
  edge defect, not material), hinge opening survived PEEQ 1.35 with no tear. Nothing controlled has
  broken ⇒ eps_f0=1.5 is a floor and k=1.5 is a steel value with ZERO PET evidence.
- **Ranked experiments (each pins one thing):** (1) **coupon run-to-break** → eps_f0 at η=1/3 [highest
  value, nearly free — same strips, just don't stop]; (2) **hinge shear to break** → 2nd (η,ε_f) point
  ⇒ k, and fills panel B; (3) **hinge opening to break** → tests D=1 at 0.37 w_lig; (4) **fold-unload
  cycles** (fold to θ, release, photograph residual angle, θ=10…90°) → the permanent-set curve = the
  functional damage; (5) **whitening as a FREE optical PEEQ gauge** — calibrate greyscale vs known
  ε_true on a coupon (λ gives ε_true), then read the hinge video as a plastic-strain FIELD. That would
  spatially validate the PEEQ field, which we have only ever checked through a p99 scalar.
- **Two open method risks:** percentile sensitivity of the p99 aggregate (never checked, cheap to run),
  and the memoryless assumption — [[project_hinge_path_observation]] found shear REVERSES in 83% of
  hinges on a real deployment, and under reversal PEEQ keeps accumulating while (a,s,θ) returns, so a
  state-function D mis-scores reversing paths.

## ★★ DAMAGE DEFINITION CHOSEN + RUN-TO-BREAK #1 EXECUTED (2026-07-27)

### The definition (user's pick, locked)
**Primary, always-active, smooth: normalized plastic dissipation**
`Δp = ∫σ:dεp dV / (σ_y ε_ref V_lig)`, which for PET's flat 44 MPa plateau collapses to
**`Δp ≈ ⟨PEEQ⟩_lig / ε_ref`** — a VOLUME AVERAGE, not the old p99 max. Secondary: a rarely-active
KS-aggregated fracture barrier `D_frac ≤ 1`.
- **Why max→integral:** p99 puts all gradient on one element, is mesh-sensitive, jumps on rank change,
  and sits at 0.03–0.1 (flat, no signal) because it is anchored on an event that never happens.
- **★ INTENSIVE, NOT EXTENSIVE — the trap:** bending thickness `t` through θ over ligament width `w`
  gives PEEQ ~ `θt/4w` over volume `w t L`, so TOTAL plastic work ≈ `σ_y θ t² L/4` is **independent of
  w** (useless as an objective), while the VOLUME AVERAGE ~ `θt/4w` correctly penalizes narrow hinges.
  Matches the user's intuition ("small hinges folded hard = strong damage"). Use the average.
- **`ε_ref` is a pure scale factor** — divides out, cannot change design ranking. So it is NOT blocking
  the oracle run; the only commitment before generating data is EMITTING `⟨PEEQ⟩`.
- **Whitening is NOT a 30× mismatch after all:** model says PEEQ ≈ 0.043 at whitening onset; polymer
  literature says 5–20% strain. Those AGREE. The 30× was against the *fracture* ruler only.
- Damage IS separable from material assumptions — material model = falsifiable physics producing the
  fields; damage = an uncoupled read-out functional, a design choice. Separation holds ONLY while
  uncoupled (no Lemaitre stiffness feedback) — keep it uncoupled. See [[feedback_hinge_validates_not_calibrates]].
- Literature: KS/p-norm aggregation is standard in stress-constrained topopt; uncoupled ductile
  criteria (Cockcroft-Latham, Oyane, Brozzo) are the forming-industry norm; plastic work as
  objective/constraint = Ivarsson IJNME 2021 (10.1002/nme.6706) + Mathematics 2020 (10.3390/math8112062);
  origami crease metrics = "hinge index" + "residual angle" (Francis/Delimont/Howell, Mech.Sci. 4, 371, 2013).
- **NEW validation burden:** a volume average is wrong if the plastic band has the right peak but the
  wrong WIDTH. Nothing measured so far constrains plastic-zone size. Cheapest check = photograph the
  whitened band width on the already-folded hinges and compare to the ccx `PEEQ > ε_white` band.

### Run-to-break #1 — data stored, IT TORE, but the tear is not on video
`data/experiments/raw/tensile_break_20260727/` (gitignored). Source `~/Downloads/IMG_3630.MOV`
(1920×1080 HEVC 30fps, 1403.7 s, 1.54 GB) transcoded to **139 MB total**:
`video_strip_5fps.mp4` (440×1080 native crop `crop=440:1080:900:0`, CRF 18 — THE analysis asset),
`video_proxy_5fps.mp4` (960×540 CRF 28, keeps the Bluehill screen for coarse sync),
`tensile_with_video.csv`, `README.md`. 5 fps = 0.017 mm crosshead/frame — nothing lost.
- **CSV:** 5.0 mm/min (0.0833 mm/s — **12× slower than the 20260723 batch's 1 mm/s** ⇒ free rate
  comparison), 50 Hz, 27.7 min. Upper yield **463.2 N @ x=7.12 mm**; post-peak dip 279.3 N @ 8.07 mm;
  draw plateau **297 → 313 N, flat but mildly RISING** (the current `PET_PLASTIC` models it as flat);
  record ends with force in free-fall 307→204 N over 2 s at **x = 138.7 mm ⇒ IT TORE**, mid-plateau,
  with the neck having eaten only ~half the ~116 mm gauge (never reached post-draw re-hardening).
- **⚠ The tear is NOT filmed** — video ends at 1403.7 s, tear at ~1662 s (user cut it when the drawn
  material left frame). ⇒ **post-mortem caliper on the broken piece is the ONLY route to eps_f0**
  (`ε_f = ln(A₀/A_f)` at the tear, plus a second reading in mid-draw away from it).
- **★ THE INK LADDER WORKS.** 10 mm pitch, ticks 0…60 drawn on the gauge; scale **0.159 mm/px** in the
  strip crop, near-uniform (fit perspective from the t≈0 ladder anyway). Neck initiates ABOVE the 60
  mark and propagates DOWNWARD (past 40 by t≈1000 s, past 30 by t≈1400 s). At t=1400 s the 30–40
  interval spans ~33 mm from 10 mm ⇒ **λ ≈ 3.3 measured IN SITU**, independently confirming the
  post-mortem λ = 3.28. The cold-draw anchor in `PET_PLASTIC` is corroborated by a second method.
- **Whitening contrast is LOW in transmission** — drawn PET reads hazy/streaked, not white. Use
  oblique REFLECTED light for the optical PEEQ gauge, not a backlight.
- **Best video↔CSV sync:** track the upper grip in the strip video, cross-correlate against the
  `Displacement` column (also re-checks the pixel scale). Bluehill screen in the proxy = fallback.
- **Still missing (asked):** pre-test w and t, grip separation L0, MD/CD, confirm 5 mm/min, WHERE the
  tear started (grip failure would invalidate it), post-mortem drawn w and t, photo. n=1 so far —
  2 more, and a SHORTER gauge would reach full draw + re-hardening within the travel.

### ★★ eps_f0 MEASURED = 1.78 (run-to-break #1 post-mortem, 2026-07-27)
Operator caliper values → `data/experiments/raw/tensile_break_20260727/specimen_summary.csv`
(same layout as the 20260723 sheet, parses with `summary_io.load_summary`; full working in that
folder's `README.md`). **L0 = 114.12 mm, w0 = 18.34 mm, t0 = 0.50 mm NOMINAL (not measured — every
stress scales with it), A0 = 9.170 mm²; at the tear w_f = 8.56, t_f = 0.18 ⇒ A_f = 1.5408 mm².**
Rate 5 mm/min confirmed. Direction (MD/CD) NOT recorded.
- **`eps_f = ln(A0/A_f) = 1.784`** — the first REAL fracture strain for PET. Replaces the
  `eps_f0 = 1.5` floor in `PETIsotropic` (n=1; NOT yet applied to `pet.py` — pending user call).
- Stress axis now real: **upper yield 50.5 MPa eng** (463.2 N), neck initiation 30.5 MPa
  (279.3 N), plateau 33.6 MPa eng, and **199.1 MPa TRUE on the tear section just before failure** —
  the flow curve keeps rising far past `PET_PLASTIC`'s last point (110.2 MPa @ ε 1.189). Two new
  anchors available if the table is extended.
- ⚠ **The tear section drew FURTHER than natural draw** (A0/A_f = 5.95 vs λ ≈ 3.3), and the extra
  is almost entirely in WIDTH (18.34→8.56) while thickness matches natural draw (0.18, same as the
  20260723 batch). It tore in the material that necked FIRST (upper tier, between the drifted 50 and
  60 marks) and sat ~26 min under plateau load ⇒ **part of that extra draw is creep, not monotonic
  hardening.** Hence the "Drawn width/thickness" columns of the summary CSV are deliberately LEFT
  EMPTY (so `SummaryRow.lam` returns None instead of silently reporting 5.95 as the natural λ);
  the tear section lives in separate `Tear *` columns that `summary_io` ignores.
- **★ VIDEO↔CSV SYNC PROVEN TO ~1 s:** operator saw necking at 1m36 = 96 s; CSV post-peak load dip
  (= neck initiation) is at t = 97.0 s. The two timebases start together.
- **Gauge fraction drawn at the tear = 52%** (at λ=3.3) — never reached post-draw re-hardening
  across the full gauge. A SHORTER gauge would fit full draw + re-hardening inside the travel.
- **This coupon cannot pin E** and does not contradict 3.0 GPa: dx/dF = 9.80 µm/N over σ 0.5–5.5 MPa,
  so with C = 7.77 the specimen carries only **21%** of the slope ⇒ E "=" 6.1 GPa is an amplified
  division by a small number. E = 3.0 would need C = 5.65 µm/N (0.73× the hinge value, plausible for
  different grips). The 3.02 ± 0.91 GPa from the 20260723 batch stands.

## ★★ VIDEO LADDER EXTENSOMETER → E IS ~2.3 GPa, NOT 3.0 (2026-07-27)
New code: **`nff/calibration/ladder.py`** (`decode_gray` / `track_ladder` / `gauge_strain` /
`sync_by_grip`; pure numpy + ffmpeg pipe, no OpenCV) and
**`nff/scripts/figures/plot_pet_video_stress_strain.py`** → `pet_video_stress_strain.png`
(A whole test, B flow curve + its 2 anchors, C elastic region on 3 strain axes, D modulus vs window).
Tracking cache: `data/experiments/processed/pet_video_ladder.npz`.
- **Method that works:** 2D template NCC on each ladder tick (digit+tick patch, ±7 px search,
  parabolic subpixel) then an AFFINE fit `y_i = a + b·y_i(0)` over all 6 ticks ⇒ strain = b−1.
  **Residual 0.20 px median / 0.43 max** ⇒ the strain field is uniform and tracking is subpixel.
  ⚠ Two things that did NOT work: independent per-tick peak search (2 ticks jump to neighbours,
  segment strains ±25%) and 1-D profile registration (band-dependent, 20% spread).
- **★ SYNC SOLVED — track the moving GRIP, not the specimen.** Its travel IS the crosshead
  displacement (immune to slip). Fit gives **offset −0.70 s, 0.1417 mm/px, 8 µm rms.** Correlating
  the two smooth ramps directly is DEGENERATE (gave a bogus +4.9 s). Ladder scale (0.1558 mm/px from
  the hand-drawn 10 mm pitch) disagrees ~10% with the grip scale — irrelevant, strain is a ratio.
- **★★ E = 2.05 GPa over σ 0.5–15 MPa, 2.26 ± 0.30 over 0.5–5 (compliance-free, no gauge assumption).**
  Across every window the ladder reads **1.8–2.7 GPa**. Noise floor 0.033 % strain, so the first
  0.2 % (where the true initial tangent lives) is NOT resolvable — the tangent may be a bit higher.
- **★★ C IS NOT A MACHINE CONSTANT.** Fitting `x = ε_ladder·L0 + C·F` on this coupon gives
  **C = 4.81 µm/N**, vs 7.77 measured on the hinge fixture. And the 20260723 batch's TOTAL slope is
  12.33 ± 1.40 µm/N vs 9.80 here — specimen-to-specimen seating varies by MORE than the effect being
  measured. ⇒ **the "E = 3.02 ± 0.91 GPa" from the batch was an artifact of applying the HINGE's
  compliance to COUPON data** (with C=4.81 the same batch gives 1.73 ± 0.32). Retire that number.
- **RECOMMENDATION: `PET['E']` 3000 → ~2500 MPa.** Low stakes (E 3.0→1.2 GPa moves hinge force ~2%,
  the response is plasticity-dominated) but 3.0 is no longer the best estimate. NOT yet changed.
- ν = 0.40 still UNMEASURED: the transverse-strain route failed here because the specimen is
  TRANSPARENT against a structured background (window + monitor) — edge detection garbage. Next
  time: matte backdrop, then width tracking gives ν and an out-of-plane check for free.
- **Open systematic:** an initially-bowed strip straightening under load moves through the depth of
  field and mimics strain (~0.3 %/mm of bow at this working distance). Mostly absorbed by the free
  fit intercept (which came out −0.10 %), but unquantified.

## ★★ pet.py UPDATED (2026-07-27, user-approved): eps_f0 1.5 → 1.784, *PLASTIC extended to 200 MPa
`PET_PLASTIC` gained (123.3, 1.300), (143.2, 1.450), (166.4, 1.600), (200.0, 1.784) — same
constant-force construction `σ_true = 33.6·exp(ε)` as the existing 0.272–1.189 branch.
`PETIsotropic.__init__(eps_f0=1.784)`. 155 tests pass.
- **Why it's legitimate:** the run-to-break reproduced the batch plateau (33.58 vs 33.6 MPa eng, 0.1%)
  **at a 12× slower rate**, and the in-situ λ (3.3) matched the post-mortem λ (3.28). Two independent
  consistency checks on the construction the whole table rests on.
- **⚠ What it is NOT:** four measurements. Only the plateau force and the final cross-section were
  observed; ε = 1.30/1.45/1.60 are interpolation between two measured endpoints.
- **⚠ ACTION REQUIRED — the hinge validation is now stale.** Before the extension CalculiX held
  110.2 MPa FLAT above ε 1.189 (perfectly plastic); it now hardens to 200. The w=18 mm hinge opening
  reaches PEEQ 1.35, i.e. INSIDE the extended region ⇒ the predicted force will RISE and the existing
  319 N sim vs 287 N exp (+11%) over-prediction will get WORSE. **`ccx` is NOT installed on this
  machine** — cannot re-run here. This is the top open item.
- n: the value is n=1. The other 9 coupons corroborate the REGIME (all drew past ε 1.19 unbroken) but
  none was observed to fracture, so they bound eps_f from below, they don't measure it. Also the tear
  came after ~26 min under plateau load, so 1.784 is a slow-rate/creep-inclusive fracture strain.

### CORRECTIONS + IDENTIFIABILITY CHECKS (2026-07-27, same session)
- **✗ RETRACTED: the hinge validation is NOT stale.** Checked `pet_w18_open_multipoint.npz`:
  **peak F = 319.2 N occurs at a = 0.71 mm where PEEQ_p99 = 0.188.** PEEQ only crosses 1.189 at
  a = 4.80 mm, by which point F has already fallen to 275 N. The validated quantity sits entirely
  inside the ORIGINAL table ⇒ the ε>1.189 extension changes nothing about 319-vs-287 N. No re-run
  needed. Docstrings in `pet.py` corrected.
- **Plateau and upper yield are COMPLIANCE-IMMUNE.** Both are `F/A0` read off the FORCE axis at a
  landmark identified by the SHAPE of the force curve. Compliance only ever distorts the
  DISPLACEMENT axis. The `*PLASTIC` strain axis comes from λ (geometry, post-mortem), never from
  crosshead. **E is the ONLY quantity in the material model that compliance touches.**
- **λ is not a separate FEA input** — it appears only in `pet.py`, as the derivation of the
  (110.2 MPa, 1.189) point (ε = ln λ). It is baked into `PET_PLASTIC`, nothing else reads it.
- **★ E IS NOT IDENTIFIABLE FROM THE 11 COUPONS.** Regressing total slope on L0/A0 (which would give
  1/E as the regression slope and mean C as the intercept): **slope = −0.770 ± 2.087 µm/N per 1/mm**,
  where E = 2.5 GPa needs +0.400. Signal/noise **0.19**, and the fit even comes out the wrong sign.
  Cause: L0/A0 spans only 12.17–12.99 (6.5%) while the slope scatter is ±1.67 µm/N. **The fix is a
  VARIABLE-GAUGE-LENGTH batch: gauge lengths ~46 to ~229 mm at this cross-section** would give a
  spread of ~17 1/mm and make E and mean C jointly identifiable — the classic machine-compliance
  calibration. Otherwise just run the ladder on 2–3 more coupons (direct, assumption-free).
- **ν from video: fixable, and the fix is trivial** — draw a TRANSVERSE ladder too (a cross), and
  track it exactly like the axial one. Edge detection failed only because the specimen is
  TRANSPARENT against a structured background AND the elastic transverse strain is ~1 px at this
  framing. Note the PLASTIC contraction is strongly ANISOTROPIC (w ×0.467, t ×0.36 at the tear),
  which J2 cannot represent — a real model limitation, but only in the deep-draw regime.
- **k (triaxiality sensitivity): no defensible PET literature value exists.** Polymer fracture loci
  are calibrated per-material from NOTCHED specimen sets + inverse FE (HDPE/PP/PA6 studies), and the
  locus is not even monotonic in η — Lode angle matters too. Two options: (a) **set k = 0**, making
  ε_f = ε_f0 = 1.784 stress-state-independent — honest, and inside the hinge envelope (η 0.33–0.41)
  it moves ε_f by <10% vs k=1.5 anyway; (b) measure it with ONE laser-cut double-edge-notched coupon
  (η ~0.5–0.6) run on the same protocol. Also note k now only enters the SECONDARY fracture barrier
  of the new damage definition, not the primary plastic-dissipation objective ⇒ lower stakes still.
- **Rate: the PLATEAU is rate-insensitive over 12×** (33.58 MPa at 5 mm/min vs 33.6 at 60 mm/min) —
  which retro-justifies having no rate term for the draw. The upper yield differs (50.5 vs 44) but in
  the wrong direction for rate, so that is specimen scatter and/or the assumed t0 = 0.50.

### ★ k = 0 FOR PET (user decision, 2026-07-27): "if it's a standard model for steel, don't use it for PET"
`PETIsotropic.__init__(k=0.0)` ⇒ `eps_f = eps_f0 = 1.784` regardless of stress state. The
triaxiality locus `eps_f0·exp(-k(η-1/3))` is a METALS construction (models void growth under
hydrostatic tension) and `k=1.5` was a mild-steel number carried over from `SteelJ2`; PET fails by
crazing/fibrillation instead. No PET literature value exists — polymer loci are calibrated per
material from notched sets and are often non-monotonic in η (Lode angle matters).
- **Costs nothing:** measured η is 0.33 (fold) to 0.41 (shear) — both ~uniaxial tension, because a
  fold's critical fibre is the OUTER SURFACE IN BENDING, not shear. So the term never did the job it
  was added for (steel's constant eps_f=0.25 condemning shear-dominated folds). Across that η band
  the exponential moves <10%, under the n=1 uncertainty on eps_f0 itself.
- **Downstream:** figures no longer hardcode 1.5 — `plot_pet_damage_ruler.py` and
  `plot_pet_three_modes.py` now import `PETIsotropic().eps_f0`. Regenerated ruler: fold D 0.080→
  **0.067**, opening D 1.326→**1.115**, whitening D 0.029→**0.024**. 155 tests pass.
- ⚠ `paper.py` still carries the same borrowed `k=1.5` (PARKED material, its docstring already flags
  "calibrate eps_tear0/k against the physical print"). `steel.py` keeps k=1.5 correctly.
- To turn it back on: ONE double-edge-notched coupon (η ~0.5–0.6), same protocol + ladder.
