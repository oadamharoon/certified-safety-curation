#!/bin/bash
# T1.1: CDT sub-budget conditioning sweep.
# Stage 1 (GPU): retrain full-data CDT, 15 DSRL tasks x 3 seeds, cost_limit = budget.
# Stage 2 (CPU): eval each checkpoint at sub-budget targets, 100 eps/target.
# Stage 3 (CPU): bullet eval-only on existing checkpoints at {2,5,10}.
set -u
S=/tmp/claude-1001/-home-omniverse-workspace-safevlmcpl/cbe3ff25-bd02-4cf4-9f36-173bf5fa270c/scratchpad
LOGDIR=$S/cdtsweep_logs
mkdir -p "$LOGDIR/runs" "$LOGDIR/evals"
log_run () { echo "[$(date +%m/%d-%H:%M:%S)] $1" | tee -a "$LOGDIR/progress.log"; }

NAV="OfflineCarGoal1Gymnasium-v0 OfflineCarGoal2Gymnasium-v0 OfflinePointGoal1Gymnasium-v0 OfflinePointGoal2Gymnasium-v0 OfflinePointButton1Gymnasium-v0 OfflinePointButton2Gymnasium-v0 OfflineCarButton1Gymnasium-v0 OfflineCarButton2Gymnasium-v0 OfflinePointCircle1Gymnasium-v0 OfflinePointCircle2Gymnasium-v0"
VEL="OfflineHalfCheetahVelocityGymnasium-v1 OfflineWalker2dVelocityGymnasium-v1 OfflineAntVelocityGymnasium-v1 OfflineHopperVelocityGymnasium-v1 OfflineSwimmerVelocityGymnasium-v1"

train_one () {
  local env=$1 seed=$2 lim=$3
  local tag="cdtsw_${env}_s${seed}"
  [ -f "$LOGDIR/done_${tag}" ] && return 0
  cd /home/omniverse/workspace/safevlmcpl/osrl
  env PYTHONNOUSERSITE=1 PYTHONPATH=/home/omniverse/workspace/safevlmcpl/osrl \
    conda run -n safevlmcpl --no-capture-output \
    python examples/train/train_cdt.py --task "$env" --seed "$seed" \
    --cost_limit "$lim" --device cuda --logdir "$LOGDIR/runs" \
    > "$LOGDIR/${tag}.log" 2>&1 \
    && { touch "$LOGDIR/done_${tag}"; echo "[$(date +%m/%d-%H:%M:%S)] DONE $tag" >> "$LOGDIR/progress.log"; } \
    || echo "[$(date +%m/%d-%H:%M:%S)] FAIL $tag" >> "$LOGDIR/progress.log"
}
export -f train_one
export LOGDIR S

J=$S/cdtsw_jobs.txt; : > "$J"
for seed in 0 1 2; do
  for e in $NAV; do echo "$e $seed 25" >> "$J"; done
  for e in $VEL; do echo "$e $seed 20" >> "$J"; done
done
log_run "CDT sweep trainings: $(wc -l < $J)"
xargs -a "$J" -L1 -P 5 bash -c 'train_one "$@"' _
log_run "TRAIN STAGE DONE"

# Stage 2: eval every new run dir at its target grid
eval_one () {
  local rd=$1 targets=$2
  local base=$(basename $(dirname "$rd"))_$(basename "$rd")
  local out="$LOGDIR/evals/${base}.json"
  [ -f "$out" ] && return 0
  conda run -n safevlmcpl --no-capture-output python $S/cdt_eval_sweep.py \
    "$rd" "$targets" "$out" >> "$LOGDIR/evals.log" 2>&1 \
    || echo "FAIL EVAL $base" >> "$LOGDIR/progress.log"
}
export -f eval_one
JE=$S/cdtsw_evals.txt; : > "$JE"
for d in "$LOGDIR"/runs/*-cost-25/CDT*; do
  [ -d "$d" ] && echo "$d 5,10,15,25" >> "$JE"
done
for d in "$LOGDIR"/runs/*-cost-20/CDT*; do
  [ -d "$d" ] && echo "$d 5,10,20" >> "$JE"
done
for d in $S/round2_logs/runs/*-cost-10/CDT*; do
  [ -d "$d" ] && echo "$d 2,5,10" >> "$JE"
done
log_run "eval jobs: $(wc -l < $JE)"
xargs -a "$JE" -L1 -P 8 bash -c 'eval_one "$@"' _
log_run "CDT SWEEP ALL DONE"
