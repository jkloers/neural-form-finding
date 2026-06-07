#!/usr/bin/env bash
# sofa/run_viz.sh — Two-step SOFA visualization.
#
# Step 1: run the SOFA simulation (Homebrew Python 3.12, no matplotlib)
#         and save results to a .npz file.
# Step 2: load the .npz and render figures (kgnn_mac conda env, matplotlib).
#
# Usage:
#   ./sofa/run_viz.sh                           # baseline params, display figure
#   ./sofa/run_viz.sh --save                    # save PNG to sofa/output/
#   ./sofa/run_viz.sh --arm-width 0.003 --save  # custom params + save
#
# All flags are forwarded to dump_results.py.  Flags understood by dump_results:
#   --arm-width    [m]   hinge arm width      (default: 0.005)
#   --fold-length  [m]   hinge fold length    (default: 0.020)
#   --displacement [m]   applied x-disp F2   (default: 0.010)
#   --face-size    [m]   square face panel    (default: 0.100)
#   --thickness    [m]   sheet thickness      (default: 0.001)
#   --save               save PNG instead of displaying

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
NPZ="$SCRIPT_DIR/output/sofa_result.npz"

# Separate --save flag from simulation flags
SAVE_FLAG=""
DUMP_ARGS=()
for arg in "$@"; do
    if [ "$arg" = "--save" ]; then
        SAVE_FLAG="--save"
    else
        DUMP_ARGS+=("$arg")
    fi
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 1 — SOFA simulation  (Homebrew Python 3.12)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
"$SCRIPT_DIR/run_sofa.sh" "$SCRIPT_DIR/dump_results.py" \
    --out "$NPZ" "${DUMP_ARGS[@]}"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 2 — Visualization  (kgnn_mac conda env)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
conda run -n kgnn_mac python "$SCRIPT_DIR/visualize.py" \
    --npz "$NPZ" $SAVE_FLAG

echo ""
echo "Done."
