# PET 3×3 deployment, measured from photographs — 2026-08-14

The sheet designed in `run_20260813_151917_pet45_3x3_mid_phys` was hand-cut from 4 ft × 8 ft PET,
taped (orange on every tile edge, green on every hinge), laid on a floor inside a taped target
rectangle, and deployed by hand: **tile 3 held, tile 5 pulled at its centroid** — which is exactly
the boundary condition the design was optimised under.

Three photographs (closed / deploying / deployed) are rectified into the simulation's own
millimetre frame and the nine known rigid tiles are posed against the tape, giving a measured
`(dx, dy, dθ)` per tile: the same degrees of freedom Stage 2 solves for.

## Reproduce

```bash
# 1. deployed geometry at every load step (one solve, ~4 min)
JAX_PLATFORMS=cpu PYTHONPATH=$(pwd) python3 nff/scripts/diagnostics/dump_load_path_steps.py \
    --run-dir data/outputs/runs/run_20260813_151917_pet45_3x3_mid_phys

# 2. rectify, fit, measure  (~12 min with the jitter ensemble)
PYTHONPATH=$(pwd) python3 nff/scripts/calibration/analyze_deployment_photos.py \
    --photo-dir data/experiments/pet45_3x3_deployment_20260814/raw \
    --sim data/outputs/runs/run_20260813_151917_pet45_3x3_mid_phys/load_path_steps.npz \
    --out-dir data/experiments/pet45_3x3_deployment_20260814 --jitter-n 16

# 3. the paper figure
PYTHONPATH=$(pwd) python3 nff/scripts/figures/plot_deployment_validation.py
```

## The metric anchor

The **closed sheet is the standard, not the taped target**: it is a known 1219.2 × 2438.4 mm
rectangle cut from squared stock, whereas the target is tape laid on a floor by hand. So the closed
frame is rectified onto the sheet's own four corners — a homography with no free parameters, its
labelling resolved by scoring the chiral cut pattern under all eight rotations/reflections — and the
target rectangle is then *measured* in that frame. The other two frames, where the sheet is no
longer a rectangle, are rectified onto that measured target, which is static across the series.

**Target rectangle, as taped:** 1716.1 × 2719.3 mm at −14.88°, centre (785.3, 1258.7) mm.
**As designed:** 1739.8 × 2714.3 mm at −15.39°, centre (789.0, 1312.3) mm.
→ laid down to within **24 mm** on the short side, **5 mm** on the long, **0.51°** in rotation, and
**54 mm** in centre placement.

## The measurement floor

The closed photograph is compared against the FLAT design directly -- nothing is matched, because
the operator reports the sheet closed and its geometry is known exactly. What the fit returns there
is the combined cost of hand-cutting, tape placement and rectification.
**No deployed number below this is real.**

| | closed (truth = 0) |
|---|---|
| tile centroid error, RMS | **7.6 +- 5.3 mm** |
| after removing rigid motion (Kabsch, no scale) | 4.4 +- 4.8 mm |
| tile rotation, RMS | 0.25 deg |
| hinge opening RMSE | 0.34 deg |

7.6 mm on a 2.44 m sheet is 0.31%. Uncertainties are the s.d. of a 16-draw ensemble in which the
fiducial corners are perturbed by the tape half-width -- a systematic, deliberately pessimistic
proxy, not the fit's own residual.

## Results

Each photograph is matched to the load step by whichever seed-free descriptor best resolves it
THERE, chosen by signal-to-noise rather than fixed in advance: hinge opening moves ten times faster
per unit load at the start of the path than at the end, while extension keeps moving after opening
has saturated. Both deployed frames matched on extension, which is also the quantity the operator
controlled directly by pulling.

| | closed | deploying | deployed |
|---|---|---|---|
| matched at | flat, by assumption | 27.5% of load | **72.5% of load** |
| hinge opening RMSE | 0.34 deg | 2.60 deg | **2.45 deg** |
| shape only, rigid motion removed | 4.4 mm | 16.7 mm | **17.1 +- 3.6 mm** |
| raw tile centroid error, RMS | 7.6 mm | 49.2 mm | 44.1 mm |
| clamp tile slip | 9.7 mm | 18.0 mm | 26.6 mm |
| pull tile travel, measured / simulated | 2 / 2 mm | 104 / 167 mm | **324 / 265 mm** |
| residual swing | +0.05 deg | -2.58 deg | +2.66 deg |

### What the numbers say

1. **The kinematics transfer.** At matched extension the twelve hinge openings are reproduced to
   **2.45 deg RMSE** on angles reaching 20 deg, and the shape to **17 mm RMS** on a 2.4 m sheet --
   0.7% -- against a 7.6 mm floor. The disagreement is real but small.
2. **Raw and shape errors differ by a factor of 2.6, and the difference is the hand.** The
   simulation holds tile 3 rigidly; the lab held it by hand and it slipped 27 mm. Quote the
   shape number for the model and the raw number for the experiment, never one for the other.
3. **The sheet was NOT pulled past the end of the simulated path** -- the hypothesis was tested
   directly by re-solving at 1.25x the design load (`load_path_steps_x125.npz`). Matching against
   the extended path puts the deployed frame at the same place (72.6% against 72.7%), leaves the
   opening RMSE unchanged at 2.45 deg, and makes the width agreement slightly WORSE. Extension is
   not the thing that ran out.
4. **What did run past the end is the GRIP.** The pull tile travelled 324 mm where the simulation
   reaches 265 mm at the matched step and only 286 mm at 1.25x load -- the mechanism saturates and
   cannot travel further at any defensible load. So the grip moved further than the sheet spread.
   With the clamp slipping and the sheet dragging on a floor, the final state is friction-locked
   rather than the frictionless equilibrium the model computes.
5. **The residual is a WIDTH deficit, and it points out of the plane.** At matched extension the
   sheet is **42.4 mm (2.5%) narrower** than predicted. Read as excess arc length, that is a ripple
   of 20-45 mm amplitude for wavelengths of 400-900 mm -- which is what was observed in the lab.
   Independently, the model's own guard says the hinges are in compression **1.25x past the
   surrogate's trained floor** at the design load (`eta_a` reaches -0.1746 against a box floor of
   -0.1399, 1 of 12 hinges outside at the endpoint). Compression is what buckles a 0.5 mm sheet.
   The measurement and the model's out-of-domain flag agree on the same mechanism: the sheet
   relieves hinge compression by leaving the plane instead of spreading.
6. **`deploying` is not an equilibrium.** It was shot mid-pull, so its positions carry a friction
   and motion lag and are flagged `equilibrium: false` in `metrics.json`. Its hinge openings are
   still comparable (2.60 deg RMSE); its centroids are not.

## Caveats that belong in any write-up

- **Out-of-plane buckling is measured, not hypothetical**, and it is the leading explanation for the
  width deficit. A homography measures the floor projection only.
- **Lens distortion is NOT a factor** -- measured: the fiducial's tape lines bow ~3 px over 3800 px
  (about 3 mm), well under the tape width. Tested, not assumed.
- **The camera moved between shots**, so the three homographies are independent and their systematic
  errors add rather than cancel for any change-between-frames quantity.
- **Comparison is at matched kinematics, not matched force.** The lab used a hand, not a calibrated
  27.85 N, and the load fractions above are where the SHAPE sits on the path, not a force reading.
- Tape width (~19-24 mm) is the hard resolution limit on any edge position.
- Nine tiles are fitted as a hinged mechanism, not independently. Fitted independently, a long thin
  tile slides along its own tape and scores almost as well -- one did, by 122 mm -- which is why the
  hinge-connectivity tie is on by default (`--link-weight`, 0 disables).

## Files

```
raw/         the three photographs as shot
rectified/   <state>_rect.png    perspective removed, colour as shot
             <state>_clean.png   perspective removed, floor greyed, tape only  <- the readable one
             <state>_overlay.png simulated (blue) vs measured (black dashed)
derived/     metrics.json   every number above, per tile, per hinge, per descriptor
             measured.npz   measured tile polygons, poses, homographies
             deployment_validation.pdf/.png/.txt   the paper figure and its caption sidecar.
             Pure VECTOR, no photographs (those live in rectified/). The measured sheet is the
             orange body, the simulation the thin black outline over it, and the RED AREA between
             them is the disagreement; the dashed BLACK rectangle is the target, matching the
             manuscript's convention. Tiles are numbered in (a), which is also dimensioned.
             ⚠ Authored at 1.18 x \textwidth: the manuscript must wrap it in
             \makebox[\textwidth][c]{} and set \includegraphics to 1.18\textwidth, or
             audit_paper_type.py fails it.
             Flags: `--width-frac`, `--aspect`, `--no-rotate` (portrait), `--raw-sim` (leave the
             hand grip's rigid offset in), `--sim-style`, `--photos`.
manifest.json  the lab facts that cannot be regenerated (tracked in git)
```

The 1.25x over-load solve lives beside the design run as
`data/outputs/runs/run_20260813_151917_pet45_3x3_mid_phys/load_path_steps_x125.npz`.
