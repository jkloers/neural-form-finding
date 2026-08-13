#!/usr/bin/env bash
# Fan the paired-route audit out across shards, one seed range each.
#
# The audit drives the RVE twice per path (measured polyline, then a straight ray to the same
# endpoint), so a shard of N pairs is 2N CalculiX solves. Seeds namespace both the scratch
# directories and the output files, so shards never collide and the merge is order-independent.
#
# One shard per core, not more. A first attempt at 15 shards on 10 cores gave each ccx about half
# a core at load average 71, and the solves it drew were the prior's extreme tail (an axial ratio of
# 3.25, 58 mm of opening on an 18 mm ligament), so a pair was heading for six hours. The endpoint
# band in audit_path_dependence.endpoint_reason removes that tail before any solve runs.
#
#   bash nff/scripts/calibration/run_path_dependence_fanout.sh
#   conda run -n kgnn_mac python -m nff.scripts.calibration.merge_path_dependence \
#       data/surrogates/path_dependence_shards/shard_*.json \
#       --out data/surrogates/path_dependence_pet_v3.json
set -uo pipefail

PRIOR=${PRIOR:-data/fea/path_priors/envelope_v3_w18}
W_LIG=${W_LIG:-18}
SHARDS=${SHARDS:-10}
PER_SHARD=${PER_SHARD:-3}
# The campaign's own solves carry a MEDIAN of 10 states (mean 11.6) per job, so a
# 30-state replay would compare against a ray finer than any the surrogate was fitted
# on, at three times the cost. Match the campaign.
RESAMPLE=${RESAMPLE:-12}
TIMEOUT=${TIMEOUT:-1800}          # the campaign's value; deep w18 solves have a ~1710 s median
OUT_DIR=${OUT_DIR:-data/surrogates/path_dependence_shards}
SCRATCH=${SCRATCH:-${TMPDIR:-/tmp}/hinge_audit}
LOG_DIR=$OUT_DIR/logs

mkdir -p "$OUT_DIR" "$LOG_DIR" "$SCRATCH"
echo "fanning out $SHARDS shards x $PER_SHARD pairs = $((SHARDS * PER_SHARD)) pairs "\
"($((SHARDS * PER_SHARD * 2)) ccx solves), prior $PRIOR, w_lig $W_LIG"

pids=()
for ((i = 0; i < SHARDS; i++)); do
  seed0=$i                     # interleaved: shard i takes seeds i, i+SHARDS, i+2*SHARDS, ...
  conda run -n ccx --no-capture-output python -m nff.scripts.calibration.audit_path_dependence \
    --prior "$PRIOR" --n "$PER_SHARD" --seed0 "$seed0" --seed-stride "$SHARDS" --w-lig "$W_LIG" \
    --resample "$RESAMPLE" --ncpus 1 --timeout "$TIMEOUT" --workdir-root "$SCRATCH" \
    --out "$OUT_DIR/shard_$(printf '%03d' "$seed0").json" \
    > "$LOG_DIR/shard_$(printf '%03d' "$seed0").log" 2>&1 &
  pids+=($!)
done

echo "launched ${#pids[@]} shards; logs in $LOG_DIR"
fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=$((fail + 1))
done

# A shard that exits non-zero has still written every pair it finished, because the audit dumps
# after each one. Report the count and let the merge pool whatever landed.
echo "all shards done; $fail exited non-zero"
ls -1 "$OUT_DIR"/shard_*.json 2>/dev/null | wc -l | xargs echo "shard files written:"
