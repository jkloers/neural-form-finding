# Video Extensometer — dot protocol + analysis software

**Why:** crosshead strain at L≈103 mm with tape tabs is ~75% machine/tab/toe, so it reads
E ≈ 1 GPa vs the true ~3 GPa (protocol §13.5). A two-dot video extensometer measures **gauge
strain directly**, immune to that compliance. Force-based quantities (yield, draw, UTS) are already
reliable; this fixes **E** and gives **Poisson's ratio** for free.

---

## Part A — How to put down the marks

**Layout (2 axial marks minimum; 4 to also get Poisson):**
- **Two axial dots** on the specimen centreline, in the **uniform central third**, spaced **as far
  apart as fits the uniform region (~40–60 mm)**. Larger spacing → smaller % tracking error.
  Their separation over time gives axial strain; the initial spacing sets the true gauge length.
- **Two lateral dots** (optional) straddling the width mid-gauge → transverse strain → Poisson ν.
- Keep dots **off** the region where the neck will localise if you want a clean *elastic* reading —
  for E only the early uniform stretch matters, so mid-gauge is fine.

**Making good marks (PET is shiny/translucent — contrast is the enemy):**
- Use a **fine paint pen / acrylic marker**: **white dots** if you back the strip with a matte-black
  card, or **black dots** on the bare translucent sheet against a light matte background.
- **Small, round, sharp-edged** (~1–2 mm). Round blobs track to sub-pixel; ragged marks don't.
- **Matte, not glossy** — kill glare. Diffuse, even lighting; camera perpendicular; no hotspots.
- Let them **dry** so they don't smear during the draw.

**Scale & sync (do once per setup):**
- Put a **ruler in the frame** (or measure the initial dot spacing with a caliper) → mm/pixel.
- **Sync t=0**: an LED flash or hand-clap visible in-frame at test start, matched to the Bluehill
  load onset → the video↔force time offset.
- Camera **fixed on a tripod**, fills the frame with the gauge + dots, in focus, ≥ the Bluehill data
  rate (30 fps is plenty at 1 mm/min).

**Measure per specimen (still needed):** free length L (caliper, ~103 mm), and the width×thickness
for area (from the manual summary — the source of truth).

---

## Part B — Analysis software (`nff/calibration/`)

Standalone from the JAX pipeline (numpy; optional opencv + matplotlib).

| Module | Role |
|---|---|
| `bluehill_io.py` | Parse the Bluehill results+raw CSV → per-specimen `SpecimenRun` (time, disp, force). |
| `summary_io.py` | Parse the operator's manual summary sheet → authoritative area / L / direction / speed, matched to curves **by order** of the `included` rows. |
| `stress_strain.py` | Stress-strain curves; chord E, 0.2%-offset yield, UTS, draw plateau, strain-at-break; compliance correction; engineering→true and the cold-draw `*PLASTIC` anchor (`draw_true_point`). |
| `video_extensometer.py` | OpenCV two-dot tracker → axial/lateral strain (`track_marks`→`marks_to_strain`), Poisson, and force-sync (`sync_to_force`). |

CLI: `nff/scripts/calibration/analyze_run.py`

```bash
python -m nff.scripts.calibration.analyze_run \
  --csv data/experiments/raw/<sample>.csv \
  --summary "data/experiments/raw/<summary>.csv" \
  --draw-window 12 22 \
  --plot data/experiments/processed/<sample>.png \
  --processed-dir data/experiments/processed
```

### Strain sources
- **Crosshead** (default): compliance-corrupted → E is a **lower bound**. `--compliance <C_mm/N>`
  subtracts a machine compliance measured on a known-E reference strip.
- **Video**: `video_extensometer.track_marks(...)` → `marks_to_strain(...)` → `sync_to_force(...)`,
  then feed the resulting strain array into `stress_strain.analyze(...)`. This is the trustworthy E.

---

## Part C — Folder layout

```
nff/calibration/            analysis library (this doc, Part B)
nff/scripts/calibration/    CLI entry points (analyze_run.py)
data/experiments/           (gitignored)
├── raw/          Bluehill CSV exports + manual summary sheets
├── video/        test videos for the extensometer
└── processed/    extracted stress-strain CSVs, plots, fitted properties
docs/physical_calibration_series1_tensile_protocol.md   experimental protocol (§13 = field log)
docs/physical_calibration_video_extensometer_plan.md    this file
```

---

## Part D — Roadmap

1. **[done]** Bluehill/summary parsing, stress-strain extraction, crosshead analysis CLI.
2. **[done, needs a real video]** Two-dot tracker (`video_extensometer.py`) — validate on a test clip,
   tune blob area / polarity / ROI.
3. **Compliance calibration** — pull a known-E metal strip, fit `C_machine`; cross-check vs the video E.
4. **True stress-strain + `*PLASTIC`** — combine video strain + force + measured draw ratio λ
   (`draw_true_point`) → the multi-point true-stress table for the PET material class.
5. **PET material class** — `nff/rve/materials/pet.py` (SteelJ2 template + the measured `*PLASTIC`
   table + `eps_f`), once E and the anisotropy call are in.
