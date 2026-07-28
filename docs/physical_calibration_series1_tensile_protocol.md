# Series-1 Tensile Protocol — PET Sheet Characterization

**Instrument:** Instron 34SC-5 (3400-series single-column, 5 kN frame) + Bluehill Universal software
**Standard:** ASTM D882 (tensile properties of thin plastic sheeting, thickness ≤ 1.0 mm)
**Purpose:** first physical calibration of the hinge-energy surrogate / CalculiX FEA to the real sheet
**Date drafted:** 2026-07-20 · **Related:** `memory/project_physical_calibration.md`, `docs/hinge_material_hypotheses.md`

---

## 1. Objective & scope

This series answers four questions that feed directly into the material model behind the hinge surrogate:

| Measured quantity | Calibrates (code target) |
|---|---|
| **Young's modulus `E`, per direction** | the isotropic-vs-orthotropic fork → which `Material` subclass we write (`materials/pet.py`) |
| **Yield stress + post-yield σ–ε (hardening)** | the multi-point `*PLASTIC` table in `Material.constitutive_cards()` |
| **Strain-at-break (rupture)** | `eps_f` / `eps_f0` in `nff/rve/damage.py`; confirms `failure_mode = ductile_D` |
| **Stress-whitening onset (semi-quantitative)** | the physical damage level `D` for "functional failure" (the open *is failure D=1 or D≈5?* question) |

**Primary deliverable:** the MD/CD modulus ratio (anisotropy decision). Everything else is secondary but captured in the same pull.

> **Material ID so far (Phase 0a):** ρ ≈ 1390 kg/m³ + rainbow iridescence + ductile + folds 180° without breaking ⇒ almost certainly **PET, likely biaxially oriented (BOPET)**. A copper-wire (Beilstein) flame test — green flame = PVC — is still recommended to rule out PVC before committing the material class.

---

## 2. Equipment & consumables

- Instron 34SC-5, **5 kN load cell** (confirmed installed).
- **Large flat grip faces** spanning ≥ 37 mm (confirmed). Serrated grips reserved as slip fallback only.
- **Calipers** (width) + **micrometer** (thickness) — 0.5 mm thickness needs the micrometer.
- **Cloth / duct tape** for grip tabs (no sandpaper available — tape is the friction aid).
- **Phone + tripod** for a poor-man's video extensometer + whitening observation.
- Fine marker, ruler, specimen labels, this log sheet.

---

## 3. Specimen inventory

10 hand-cut rectangular strips, nominal **202.5 × 36.8 × 0.50 mm**, 5 per direction.

- **`MD1–5`** = strips cut along the **8 ft roll length** (presumptive machine direction).
- **`CD1–5`** = strips cut across (presumptive cross direction).
- **Record which physical set is which before testing** — the anisotropy result is directional and worthless unlabeled.
- Cut **1–2 extra scrap strips** for grip/slip rehearsal (do not spend a real specimen on setup).

> **Width caveat:** 36.8 mm is wider than D882's nominal 5–25.4 mm. Acceptable for an anisotropy screen; it raises the force and makes the test more sensitive to edge nicks and misalignment (see §9). If grip slip or force becomes a problem, slit future strips to ~15 mm.

---

## 4. Dimensional measurement & the area convention

Cross-section area `A = w·t` sets stress (`σ = F/A`) directly — it is the **single largest controllable uncertainty** for hand-cut strips. For **each** specimen, before testing:

1. **Width** with calipers at **3 locations** (near each grip line + mid-gauge). Record all three.
2. **Thickness** with the **micrometer** at **3 locations** (light, consistent anvil pressure — don't crush). Record all three.
3. Compute **mean area** `A_mean` (used for modulus) and **minimum area** `A_min = w_min·t_min` (governs strength; failure happens at the thinnest section).
4. Log the **actual free length** between grips *after clamping* (§7) as the gauge length — hand-cut lengths vary; never assume nominal.

> Bluehill can import caliper/micrometer readings directly if the gauges are connected; otherwise type the per-specimen values into the specimen sheet at test time. **Never run all 10 on a single nominal 36.8 × 0.50.**

---

## 5. Machine setup (frame)

> Control names below follow the Instron 3400-series operator's guide and 34SC front panel. Exact labels can vary slightly by firmware/Bluehill version — confirm against the on-machine guide. See §12 references.

1. **Power on** the frame, then the PC; launch **Bluehill Universal**. Let the load cell warm up a few minutes.
2. **Confirm the load cell**: Bluehill should auto-identify the 5 kN cell (transducer info in the system/hardware panel). Verify capacity reads **5 kN**.
3. **Install the large flat grips** top and bottom. Check faces are **parallel and axially aligned** (misalignment bends the specimen and fakes a low modulus).
4. **Set travel/soft limits**: raise the **limit stops** on the measurement column so the crosshead cannot drive the grips together or overrun. Set an upper travel limit above the expected break extension (grip separation + generous elongation).
5. **Set a force limit** (operational limit) at **≈ 4.8 kN** so the system suspends motion before overloading the cell.
6. **Balance / zero the load** (tare) with the grips installed but **no specimen loaded**, so grip weight is excluded.
7. **Positioning controls** (for loading specimens): use the **jog** buttons for coarse crosshead motion and the **thumbwheel / fine control** for slow positioning. **`RESET GL`** (reset gauge length) sets the zero-extension datum / precise grip position.

**Safety:** rapid crosshead motion is a crush hazard — keep hands clear of the grip region whenever jogging, returning, or running. Never jog with fingers between the jaws.

---

## 6. Bluehill method setup (once)

Create the method once; reuse for all specimens.

1. **New Method → test type `Tensile`.** If present, start from the **pre-configured ASTM D882 template** (Bluehill ships one) and adjust — it pre-loads compliant calculations and speed logic.
2. **Specimen tab:** geometry = *rectangular*; define per-specimen inputs for **width, thickness, gauge length** (entered at test time, §4). Set units (SI: N, mm, MPa).
3. **Control / test speed:**
   - D882 crosshead speed = *initial grip separation × initial strain rate*.
   - Initial strain rate is chosen from expected elongation-at-break: **< 20 % → 0.1 min⁻¹**, **20–100 % → 0.5 min⁻¹**, **> 100 % → 10 min⁻¹**.
   - **Series-1 choice:** run **0.1 min⁻¹** → at L₀ ≈ 150 mm that is **≈ 15 mm/min**. This gives a clean modulus and yield. Keep it **identical for all 10** for a fair MD/CD comparison. (Elongation-based rate refinement is a later concern; do not vary speed here.)
4. **Pre-test:** set a **preload of ~3 N** and enable **"zero extension at preload."** This removes hand-cut slack/waviness so the toe region does not corrupt the modulus.
5. **Data acquisition rate:** set **high (≥ 50 Hz)** so the elastic region is well resolved.
6. **Strain source — compliance correction:** because there is **no extensometer**, configure **Corrected Displacement** (Bluehill's virtual measurement that subtracts machine/system compliance from crosshead position). Use it as the strain channel for modulus. *(The phone-dot video, §7, is the independent cross-check and is preferred where available.)*
7. **End-of-test / break detection:** detect break at a **≥ 40 % drop in load**, plus the travel limit as a backstop.
8. **Calculations / results:** enable
   - **Modulus** — Bluehill **Automatic Young's Modulus** (finds the steepest linear segment; robust to the toe), or a fixed **chord modulus** over a clean window (e.g. 5–25 MPa).
   - **Yield** — 0.2 % offset (or yield-point if a distinct peak appears).
   - **Tensile strength** — maximum stress.
   - **Strain at break** / elongation at break.
9. **Save the method.** Name it e.g. `PET_D882_series1_15mmmin`.

---

## 7. Running one specimen (frame + software)

For each of the 10 (rehearse on scrap first):

1. **Prep the specimen:** fold-back or cloth-tape-wrap the last ~25 mm of **each end** (grip tabs — friction + moves the stress riser off the jaw edge). Mark a **centerline**. Add **two gauge dots ~50 mm apart** mid-gauge for the video; note their spacing.
2. **Clamp the top grip** first; hang the strip; align the centerline vertical; **clamp the bottom grip.** Clamp **just firmly enough** not to slip — over-clamping worsens jaw-edge stress concentration.
3. **Measure the actual free length** between jaws → enter as this specimen's **gauge length**; enter its **width & thickness** (§4).
4. **`RESET GL`** / zero extension datum at the loaded-but-slack position.
5. **Start the phone video** (tripod; put a ruler and, if possible, the Bluehill live load readout in frame; a hand-clap or LED flash at "start" syncs t=0 for whitening timing).
6. In Bluehill, **start the test.** The method runs pre-test (preload → zero), then pulls at 15 mm/min **to break.**
7. **Watch and annotate live:**
   - **Slip** = a sudden slope drop / plateau, or divergence between crosshead and dot-video → note it.
   - **Whitening** — watch the **central third only** (see §9); log the timestamp/strain of first diffuse whitening and of neck localization.
   - **Break location** — central-third neck = valid; **at/inside the jaw = invalid** (§8).
8. **Save the run.** Record break location and any slip in the log.
9. **Return the crosshead** (jog/return, hands clear), remove the specimen, reset for the next.

---

## 8. Validity rules (what to keep, what to discard)

- **Break at or inside the grip line → INVALID.** The jaw is a stress raiser; that number is about the grips, not the material. Re-run a spare.
- **Visible slip before break → strength/elongation invalid**, but **modulus and yield (low-strain) are still valid** — keep them, flag the run.
- **Obvious edge-nick-initiated early break** (break at a visible flaw well below the group) → treat as an outlier; note and consider re-running.
- **If most specimens fail at the jaw**, the grips are too aggressive or the tabs insufficient — add tape, clamp lighter — *before* trusting any UTS / strain-at-break number.
- Aim for **≥ 4 valid specimens per direction** for the anisotropy statistic.

---

## 9. Uncertainty control (dedicated)

Sources, magnitude, and mitigation — hand-cut strips + crosshead-only strain are the two big ones.

| Source | Effect | Mitigation |
|---|---|---|
| **Cross-section area (hand-cut w, t)** | direct on stress → E, yield, UTS | per-specimen 3-point caliper+micrometer; `A_mean` for E, `A_min` for strength (§4) |
| **Machine/grip compliance (no extensometer)** | crosshead overstates strain → **understates E** | Bluehill **Corrected Displacement** + **phone-dot video** as independent strain; cross-check the two |
| **Grip slip** | fake low modulus / lost strength | tape tabs, adequate clamp, watch dot-video vs crosshead |
| **Jaw-edge stress concentration** | premature whitening/break at grips | tabbing; read **central third only** (Saint-Venant: uniform ~1 width ≈ 37 mm from each jaw); distinguish jaw-band whitening (thin, symmetric, at the line) from real central necking |
| **Edge nicks (hand-cut)** | scatter UTS & elongation badly; **little effect on E/yield** | **weight modulus/yield, not UTS, for the anisotropy decision**; ≥ 4 valid/direction |
| **Toe / slack (wavy strips)** | corrupts initial slope | 3 N preload + zero; Automatic Young's Modulus over the linear segment |
| **Misalignment (non-square ends)** | bending → low apparent E, early break | centerline, careful clamping, preload to straighten |
| **Rate sensitivity (PET is viscoelastic)** | E, yield depend on speed | fixed 15 mm/min for all; rate study is a separate experiment |

**Statistics & decision rule:**
- Per direction, report **mean ± standard deviation** and CV for `E`, yield, UTS, strain-at-break.
- **Anisotropy is real if** `|E_MD − E_CD|` exceeds the combined scatter — rule of thumb **> 2 × pooled std**, or non-overlapping mean ± std bands.
- **Decision:** ratio within ~10 % (overlapping) → **`PETIsotropic`** class (isotropic `*ELASTIC`, ductile-D failure — the `SteelJ2` template). Difference > ~10–15 % → **orthotropic** route (paper-style `*ELASTIC, TYPE=ENGINEERING CONSTANTS`) and **cut-angle-to-MD (`orientation_deg`) becomes a real design variable**; stiffer direction = MD.

---

## 10. Data recording template

One row per specimen (spreadsheet or the log sheet):

```
ID | dir(MD/CD) | w1,w2,w3 [mm] | t1,t2,t3 [mm] | A_mean [mm²] | A_min [mm²] |
free_length L0 [mm] | speed [mm/min] | E [MPa] | yield σ_y [MPa] | UTS [MPa] |
strain_at_break [%] | whitening_onset_strain [%] | break_location | slip? | valid? | notes
```

Also archive, per specimen: the **Bluehill raw σ–ε curve** (export) and the **phone video** (named by ID). One representative σ–ε curve per direction becomes the CalculiX `*PLASTIC` input.

---

## 11. What to send back for modeling

After ≥ 2–3 valid runs, report per specimen: **E, yield, UTS, strain-at-break, and whitening-onset strain**, plus the raw σ–ε export. That is enough to (a) make the isotropic-vs-orthotropic call and (b) stub `materials/pet.py` with a real `*PLASTIC` table and `eps_f`.

---

## 12. References

- [Instron 3400-series universal testing systems](https://www.instron.com/en/products/testing-systems/universal-testing-systems/low-force-universal-testing-systems/3400-series/)
- [Instron 3400 Single Column Table Model Operator's Guide (PDF)](https://www.instron.com/wp-content/uploads/2024/06/3400-Single-Column-Table-Model-Operator-Guide.pdf)
- [Bluehill Universal software](https://www.instron.com/en/products/materials-testing-software/bluehill-universal/) · [Method templates](https://www.instron.com/en/products/materials-testing-software/bluehill-universal/method-templates/)
- [Bluehill Universal Test Method Development training manual (PDF)](https://www.chim.unifi.it/upload/sub/ricerca/strumentaz/lista%20strumenti/UTM-Instron-6800/M18-17146-EN%20Rev%20C.pdf)
- [Instron — ASTM D882 tensile testing of thin plastic film](https://www.instron.com/en/testing-solutions/astm-standards/astm-d882/) · [ZwickRoell D882](https://www.zwickroell.com/industries/plastics/thin-sheeting-and-plastic-films/astm-d882-film-tensile-test/)
- [Bluehill system-compliance correction app note (PDF)](https://www.instron.com/wp-content/uploads/2024/07/compliance-correction.pdf)
- [UC Riverside MSE Instron training notebook (PDF)](https://mse.ucr.edu/sites/default/files/2019-02/ucr_mse_instron_notebook_training_rev_1.1.pdf)

> **Note on machine/software step names:** the official 3400 operator guide and Bluehill training manual are distributed as scanned/image PDFs, so the exact on-screen labels above are reconstructed from Instron product pages, the compliance app-note, and third-party SOPs. Confirm precise button/menu names against the on-machine operator guide the first time through; the *workflow* (method → specimen → control → preload → compliance → calculations → run) is standard Bluehill Universal.

---

## 13. Field log & live findings (2026-07-21)

First real pulls on scrap/trial strips (crosshead-only strain, no extensometer yet). These update the plan.

### 13.1 Gripping — resolved recipe
- **36.8 mm strips slipped** in flat grips even taped (too wide → too much force). **Serrated grips then notched** the thin PET → premature break at the jaw base (~0.8 kN ≈ 87 MPa on ~9 mm², far below PET's ~200 MPa ⇒ **invalid grip failure**, not material strength).
- **Winning recipe:** **slit to ~half width + cloth-tape tabs on both faces extending just past the jaw + FLAT grips + slower rate** (lower peak force eases the grip). Serrated = last resort only (notches thin film). Confirms §3's "slit to ~15 mm" mitigation.

### 13.2 Cold drawing / neck propagation (the big observation)
At **5 mm/min** a thinner taped strip showed: stiff elastic rise to **~0.6 kN @ ~3 mm** → small **yield drop** → **~60 mm quasi-constant plateau** = a **neck shoulder propagating** up the strip. Below the front the material thinned ~2× (concave/hourglass), oriented.
- **Front speed ≈ 2× crosshead ⇒ natural draw ratio λ ≈ 2** (front velocity = λ/(λ−1)·v). Area ~halves — matches "twice as thin."
- **This is plastic & permanent** (not elastic) — drawn region stays thin = permanent set = the shape-holding ("holds the fold") behavior we want in a deployable.
- Mechanism = chain unfolding/alignment + **strain-induced orientation/crystallization**. Drawn region is **birefringent** (bright colors between crossed polarizers) — same physics as the sheet's rainbow; a free orientation gauge.

### 13.3 Material ID — PET confirmed (burn test, 2026-07-22)
Cold-draw + strain-crystallization + ρ≈1.39 + iridescence + ductility already pointed to **semi-crystalline / partially-oriented PET.** **Burn test closes it:** the sample **melted, dripped, and drew into long filaments, without sustained burning** — the hallmark of a thermoplastic polyester (PET is a fiber-former). This **excludes PVC** (which chars/blackens, self-extinguishes with acrid HCl, and does *not* form filaments) and the brittle amorphous polymers. **Material = PET (~97% confident).** Optional 100% nail: copper-wire green-flame test (green = PVC) — but the filaments already rule PVC out. **Decision:** commit to the PET material class; the isotropic-vs-orthotropic fork still depends on the E(MD)/E(CD) result.

### 13.4 Rate embrittlement — design-critical
15 mm/min broke early with no draw (partly grips); 5 mm/min drew fully. **Deployment target ≈ a few cm/s (~20–50 mm/s) is ~2–3 decades faster** than the 5 mm/min test ⇒ **stiffer, more brittle, less draw** ⇒ **`eps_f` calibrated slowly is optimistic/unconservative for fast deployment.** Mitigate with a rate margin on `eps_f` and a **bounding rate study** (5 / 50 / 500 mm/min). ⚠ The 34SC max crosshead (~1000 mm/min) **cannot reach cm/s** → bound + margin, don't try to match. (Hinge fails in bending and local strain-rate is geometry-set — don't over-attribute.)

### 13.5 Provisional Young's modulus — E ≈ 3 GPa (lower bound)
From 0.6 kN @ ~3 mm, A ≈ 9 mm², L₀ ≈ 150 mm → σ_yield ≈ 65 MPa, ε ≈ 2 %, **E ≈ 3.2 GPa.** Crosshead-only strain (compliance + slip inflate displacement) ⇒ **true E is higher**; consistent with PET's 2–4 GPa. Firm up with phone-DIC / TestCam / compliance-corrected strain and the exact width & free length.

### 13.6 True-stress caveat for the `*PLASTIC` table
The neck localizes strain, so the **engineering plateau ≠ the true material law.** Extract yield + draw stress + λ as **true stress–true strain**, not the raw engineering plateau.

### 13.7 Recommended order before testing the precut/hinge strips
1. **Fix strain measurement** (phone-DIC or compliance correction) so E and the σ-ε curve are trustworthy — biggest current gap.
2. **Clean valid coupon set at ONE fixed rate (5 mm/min):** confirm the grip recipe reproducibly breaks in the central third; capture E, yield, full σ-ε to break for MD and CD.
3. **Short bounding rate study** (a faster run, e.g. 50 & 500 mm/min) to quantify rate embrittlement.
4. **Then** the cut-hinge coupons for the inverse model calibration.
