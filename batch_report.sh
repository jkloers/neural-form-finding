#!/usr/bin/env bash
# One-shot campaign report: progress, throughput, and the divergence split by fold depth.
LOG="${1:-/Users/julienkloers/Documents/Code2/princeton/neural-form-finding/data/fea/hinge_dataset_pet_v2.log}"
grep -E "^  batch" "$LOG" | tail -1
grep -E "^      \[" "$LOG" | awk '
  {for(i=1;i<=NF;i++) if($i ~ /s$/ && $i+0>0) t=$i+0
   for(i=1;i<=NF;i++) if($i=="th") th=$(i+1)+0
   r=$NF; s+=t; n++
   if(r=="completed") comp++; else if(r=="diverged") dv++; else other++
   if(th>60){ if(r=="diverged") d6++; else c6++ } else { if(r=="diverged") dl++; else cl++ }}
  END{printf "  %d jobs | mean %.0fs -> %.0f jobs/h | %d completed, %d diverged, %d other\n", n, s/n, 9*3600/(s/n), comp, dv, other
      printf "  divergence  theta>60: %d/%d (%.0f%%)   theta<=60: %d/%d (%.0f%%)\n", d6, c6+d6, 100*d6/(c6+d6), dl, cl+dl, 100*dl/(cl+dl)}'
python3 - "$LOG" << 'PY'
import json, os, sys
j = sys.argv[1].replace(".log", ".json")
if os.path.exists(j):
    d = json.load(open(j))
    print(f"  checkpoint: {d['n_jobs']} jobs, {d['n_usable']} usable, {d['n_samples']} samples"
          f" | Delta_tear {d.get('delta_tear')} from {d.get('n_tear_observations',0)} torn")
    r = d.get("regime_names", {})
    print(f"  regimes: elastic {d.get('n_elastic')} / plastic {d.get('n_plastic')} / failed {d.get('n_failed')}")
PY
echo "  disk $(df -h /tmp | tail -1 | awk '{print $4}') free | $(pgrep -x ccx | wc -l | tr -d ' ') ccx | $(pmset -g batt 2>/dev/null | head -1 | grep -o "'.*'")"
