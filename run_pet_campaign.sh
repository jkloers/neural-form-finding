#!/usr/bin/env bash
# Control the PET hinge campaign: start | resume | stop | status.
#
# The campaign checkpoints after every batch, so it is safe to stop at any moment and pick up where
# it left off -- `resume` skips the jobs already on disk. `start` refuses to clobber an existing
# dataset; use `resume` for that. Runs under `caffeinate` so the machine cannot sleep mid-run.
set -uo pipefail

REPO="/Users/julienkloers/Documents/Code2/princeton/neural-form-finding"
WORK="$REPO/.claude/worktrees/pet-surrogate-campaign"
OUT="$REPO/data/fea/hinge_dataset_pet_v2"
PRIOR="$REPO/data/fea/path_priors/envelope_v3_w18"
LOG="$REPO/data/fea/hinge_dataset_pet_v2.log"
PIDFILE="/tmp/pet_campaign.pid"

# n/seed MUST stay identical between start and resume, or the job list regenerates differently and
# the finished prefix no longer lines up.
# STEPS: 15 states per job, not 30. Rows/hour is unchanged but the JOB count roughly doubles,
# and jobs are what generalisation is measured over -- the train/val split is by job, and rows
# within one job share a geometry and a ray, so they are far from independent.
N=1200; SEED=0; STEPS=30; PARALLEL=9; BATCH=50; TIMEOUT=3000
# w_lig capped at 25 mm, not 50: r_win is 100 mm, so at 50 mm the Saint-Venant window is
# only 2x the ligament and the locality the RVE rests on stops holding. At 25 mm it is 4x.
W_LIG_MIN=5; W_LIG_MAX=25

run() {
  cd "$WORK" || exit 1
  CCX_BIN=/opt/miniconda3/envs/ccx/bin/ccx caffeinate -dimsu \
    conda run -n ccx --no-capture-output python -u -m nff.scripts.generate_hinge_dataset \
      --material pet --thickness 0.5 --r-win 100 --w-lig-min "$W_LIG_MIN" --w-lig-max "$W_LIG_MAX" \
      --path-prior "$PRIOR" --max-load 300 \
      --n "$N" --seed "$SEED" --steps "$STEPS" \
      --parallel "$PARALLEL" --batch-size "$BATCH" --timeout "$TIMEOUT" \
      --out "$OUT" "$@" >> "$LOG" 2>&1 &
  echo $! > "$PIDFILE"
  echo "launched pid $(cat $PIDFILE)  ->  $LOG"
}

case "${1:-status}" in
  start)
    [ -f "$OUT.npz" ] && { echo "refusing: $OUT.npz exists -- use 'resume' (or move it aside)"; exit 1; }
    : > "$LOG"; run ;;
  resume)
    run --resume ;;
  stop)
    [ -f "$PIDFILE" ] && kill "$(cat $PIDFILE)" 2>/dev/null
    pkill -f "generate_hinge_dataset" 2>/dev/null
    pkill -x ccx 2>/dev/null
    # `caffeinate` is a separate process holding the no-sleep assertion; without this a stopped
    # campaign leaves the machine unable to sleep forever, and repeated start/stop cycles stack them.
    pkill -f "caffeinate -dimsu" 2>/dev/null
    rm -f "$PIDFILE"
    echo "stopped -- everything through the last completed batch is on disk; 'resume' to continue" ;;
  status)
    if pgrep -f generate_hinge_dataset >/dev/null; then
      echo "RUNNING ($(pgrep -x ccx | wc -l | tr -d ' ') ccx procs)"
    else
      echo "not running"
    fi
    [ -f "$OUT.json" ] && python3 -c "
import json;d=json.load(open('$OUT.json'))
print('jobs %d  usable %d  samples %d  stops %s'%(d['n_jobs'],d['n_usable'],d['n_samples'],d.get('stop_reasons')))
print('Delta_tear', d.get('delta_tear'), ' from', d.get('n_tear_observations'), 'torn jobs')"
    tail -6 "$LOG" 2>/dev/null ;;
  *) echo "usage: $0 {start|resume|stop|status}"; exit 1 ;;
esac
