#!/usr/bin/env bash
# Rebuild every plate the deck needs that paper/Images does not already hold.
#
# Run this once after a clean clone, then slides/build.sh. It is separate from build.sh because
# the matrix recut re-solves two designs and takes minutes, while build.sh takes seconds.
#
#   bash slides/make_slide_figures.sh
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/.." && pwd)"
cd "$repo"

mkdir -p "$here/figures"

# The trade-off plot and the photograph crops: seconds, pure numpy and PIL.
conda run -n kgnn_mac python "$here/make_slide_figures.py"

# ── The de-papered plates ─────────────────────────────────────────────────────────────────
# A slide has a speaker. Panel letters exist so running prose can write "see (b)", and a rule
# round a plate exists because a page has no one to say "these three are the baselines", so both
# are furniture here. NFF_SLIDE_FIGURES=1 is read by figstyle.panel_letter and drops every letter
# in one place; --no-boxes is per generator, because a box is not always furniture (the cards in
# pipeline_overview ARE the figure, and it is deliberately not rebuilt).
#
# ⚠ EVERY BUILD BELOW PASSES --out INTO slides/figures/. These generators default their --out to
#   paper/Images/, and paper/ is Overleaf-synced: a build that omits it overwrites the manuscript's
#   own figure with a lettterless one and syncs it out. Do not remove an --out from this block.
depaper() {  # depaper <script> <out-stem> [extra args...]
  local script="$1" stem="$2"; shift 2
  NFF_SLIDE_FIGURES=1 PYTHONPATH="$repo" JAX_PLATFORMS=cpu conda run -n kgnn_mac \
    python "nff/scripts/figures/$script" --out "$here/figures/$stem" "$@"
}

# Authored FLAT and wide: 432 x 216 bp, aspect 2.0. The page plate is aspect 1.35 against a body
# of 2.37, so it was height-bound and reached 57 per cent of the measure. Four columns of square
# sheets in two rows caps the aspect near 2.0, which is the useful end of the range.
# --numbers-array puts the three numbers in a table, one row per quantity labelled once at the left,
# and the row's height goes to the drawings. --scale is the print scale at that canvas: height still
# binds, at 181.6 / 216.
# The four column names are KEPT. They are what the comparison is: four sheets in a row with no
# names is four similar squares. They were dropped while the plate was authored for a page column,
# where they cramped it; on the flat 432 x 216 canvas there is room for them.
depaper plot_validity_ladder.py     validity_ladder_slide     --no-boxes \
        --width-bp 432 --aspect 2.0 --numbers-array --scale 0.84
depaper plot_hinge_overview.py      hinge_layout_slide.pdf    --no-boxes
depaper plot_path_dependence.py     path_dependence_slide     --no-boxes
depaper plot_taylor.py              taylor_convergence_slide  --no-boxes
depaper plot_surrogate_validity.py  surrogate_validity_slide
# NO --scale. Enlarging its type forced a proportional left gutter for the PC2 label and the
# proportions read worse, so this is the composition the manuscript has, shown large instead.
depaper plot_design_cloud.py        solution_set_slide --width-bp 432 --aspect 2.0 \
        --run data/outputs/design_cloud/circle_7x7_n200_s1

# The 'by construction' rung on its own: a flat cut pattern and the same sheet opened, with no
# equilibrium anywhere in it. The deck shows the two cells side by side on the cut-pattern frame so
# the audience can see what cutting a sheet with this parameterisation and opening it produces.
# One arm means no baselines, so the group ground and the caption's baseline sentences drop out.
depaper plot_validity_ladder.py     kinematic_pair            --arms chart --no-boxes --no-titles \
        --one-scale \
        --numbers-array --width-bp 130 --aspect 0.45 --scale 0.85

# The inverse-design matrix, all five cases in ONE build at slide width. The deck walks through
# them one slide at a time, cutting a row out of this file with trim/clip rather than running the
# generator five times: one build means one panel size and one type size across the whole
# sequence, so only the row changes as the audience clicks.
# 432 bp is \textwidth + 2\charterbleed in the deck, so a row prints at 1:1 and the authored type
# size is the projected one. --no-headers drops the four column titles: the deck shows one row per
# frame, so they were printed five times over, and fig_h is derived from the header band, so the
# canvas shortens by 52 bp with them.
# ⚠ The label gutter is sized to the longest case name IN THE BUILD, so a build with fewer cases
# puts its columns elsewhere. The trims in pfe_defense.tex were measured on this five-case file.
NFF_SLIDE_FIGURES=1 PYTHONPATH="$repo" JAX_PLATFORMS=cpu conda run -n kgnn_mac \
  python nff/scripts/figures/plot_inverse_design_matrix.py \
    --row "data/outputs/runs/run_20260806_161121_figmat_1_deploy:Concentrated axial load" \
    --row "data/outputs/runs/run_20260806_115352_figmat_2_cantilever:Axial and transverse load" \
    --row "data/outputs/runs/run_20260806_161121_figmat_3_moment:Axial load and couple" \
    --row "data/outputs/runs/run_20260806_161121_figmat_4_asym:Eccentric axial load" \
    --row "data/outputs/runs/run_20260806_115352_figmat_5_vertical:Self-equilibrated shear couple" \
    --width-bp 432 --no-headers --no-row-labels --no-png --skip-check \
    --out "$here/figures/matrix_all"

# The same five cases again, with ONLY the loading diagram and the force transmission. The deck
# gives each case its own frame, and two columns share the whole measure, so the force panel is
# about 246 bp wide against 110 in the four-column build. ws inside the generator is tied to
# panel_w, so the load-path weights grow with it and print at the size they were tuned for; a crop
# of the four-column build blown up to the same size would print them 2.3x too heavy.
NFF_SLIDE_FIGURES=1 PYTHONPATH="$repo" JAX_PLATFORMS=cpu conda run -n kgnn_mac \
  python nff/scripts/figures/plot_inverse_design_matrix.py \
    --row "data/outputs/runs/run_20260806_161121_figmat_1_deploy:Concentrated axial load" \
    --row "data/outputs/runs/run_20260806_115352_figmat_2_cantilever:Axial and transverse load" \
    --row "data/outputs/runs/run_20260806_161121_figmat_3_moment:Axial load and couple" \
    --row "data/outputs/runs/run_20260806_161121_figmat_4_asym:Eccentric axial load" \
    --row "data/outputs/runs/run_20260806_115352_figmat_5_vertical:Self-equilibrated shear couple" \
    --columns ad --width-bp 432 --no-headers --no-row-labels --no-png --skip-check \
    --out "$here/figures/matrix_detail"

# The pipeline plate with its five cards in ONE ROW. Same generator, same data spine, same panel
# drawings; --slide is a second COMPOSITION, not a second figure. The page plate is near enough
# square that a 16:9 body is height-limited by it and prints it at 47 per cent; the row is
# authored at 408 bp, so its type is projected at the size it was set in. What the row drops is
# the formula and detail band under every card: five cards across 408 bp leaves each 68 bp
# against the page's 110, and a formula sized to fit that would be 5 pt on a projector.
NFF_SLIDE_FIGURES=1 PYTHONPATH="$repo" JAX_PLATFORMS=cpu conda run -n kgnn_mac \
  python nff/scripts/figures/plot_pipeline_overview.py --slide \
    --out "$here/figures/pipeline_row"

# The same row with the hinge-energy card dropped and the deployment drawn with no mechanics on
# it, for the frame that meets the pipeline first. Four cards absorb the fifth's width, so each
# one comes out 81.8 bp wide against 63.4.
NFF_SLIDE_FIGURES=1 PYTHONPATH="$repo" JAX_PLATFORMS=cpu conda run -n kgnn_mac \
  python nff/scripts/figures/plot_pipeline_overview.py --slide --kinematic \
    --out "$here/figures/pipeline_row_kin"

# The five cut-pattern thumbnails the footline strip is built from, sharp and lightened.
conda run -n kgnn_mac python "$here/make_slide_figures.py" design-thumbs

# The merged physical-validation plate, rebuilt WITHOUT its two grouping frames. The plate is a
# full page of eight panels and is illegible at 16:9, so the deck cuts it into rows with
# \includegraphics trim/clip. The frames are what make that impossible: a cut between rows would
# slice a box in half and leave it open-ended, so --no-boxes is not cosmetic here.
NFF_SLIDE_FIGURES=1 PYTHONPATH="$repo" JAX_PLATFORMS=cpu conda run -n kgnn_mac \
  python nff/scripts/figures/plot_physical_validation.py --no-boxes \
    --out "$here/figures/physical_validation_slide.pdf"

# The deck no longer builds hinge_buckle.pdf. plot_hinge_buckle_figure.py is the pre-merge half
# of fig:physical-validation and its panel (a) predates the unmirroring, the squaring of the taped
# target and the 500 mm bar. The slide that shows a ligament out of plane now cuts panels (a,b,c)
# out of physical_validation_slide.pdf above, so the deck has one source for that content.

echo
echo "slide figures rebuilt in $here/figures"
