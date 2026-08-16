#!/bin/bash
# CDT eval stage rerun (fixed driver): 45 new checkpoints + 15 bullet, CPU.
set -u
S=/tmp/claude-1001/-home-omniverse-workspace-safevlmcpl/cbe3ff25-bd02-4cf4-9f36-173bf5fa270c/scratchpad
LOGDIR=$S/cdtsweep_logs
eval_one () {
  local rd=$1 targets=$2
  local base=$(basename $(dirname "$rd"))_$(basename "$rd")
  local out="$LOGDIR/evals/${base}.json"
  [ -f "$out" ] && return 0
  conda run -n safevlmcpl --no-capture-output python $S/cdt_eval_sweep.py \
    "$rd" "$targets" "$out" >> "$LOGDIR/evals.log" 2>&1 \
    || echo "FAIL EVAL3 $base" >> "$LOGDIR/progress.log"
}
export -f eval_one
export LOGDIR S
JE=$S/cdtsw_evals2.txt; : > "$JE"
for d in "$LOGDIR"/runs/*-cost-25/CDT*; do [ -d "$d" ] && echo "$d 5,10,15,25" >> "$JE"; done
for d in "$LOGDIR"/runs/*-cost-20/CDT*; do [ -d "$d" ] && echo "$d 5,10,20" >> "$JE"; done
for d in $S/round2_logs/runs/*-cost-10/CDT*; do [ -d "$d" ] && echo "$d 2,5,10" >> "$JE"; done
echo "[$(date +%m/%d-%H:%M:%S)] eval2 jobs: $(wc -l < $JE)" >> "$LOGDIR/progress.log"
xargs -a "$JE" -L1 -P 12 bash -c 'eval_one "$@"' _
echo "[$(date +%m/%d-%H:%M:%S)] CDT EVAL3 ALL DONE ($(ls $LOGDIR/evals/*.json 2>/dev/null | wc -l) jsons)" >> "$LOGDIR/progress.log"
