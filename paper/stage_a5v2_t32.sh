#!/bin/bash
# A5v2: full T3.2 AWR temperature/clip sweep, rerun after the SEED_OVERRIDE fix.
# Prereg scope (preregistration_labels_split.md, T3.2 2026-08-10):
#   nine analysis tasks x 5 configs x 3 seeds = 135 runs.
# The previous pass is void: 04f_train_v_awr.py ignored SEED_OVERRIDE, so all
# three seeds trained bit-identical weights. Its outputs are quarantined under
# runs/logs/_degenerate_awr. Nothing is reused, including seed 0, because
# HalfCheetah showed partial seed variation that is still unexplained.
set -u
W=/home/omniverse/workspace/safevlmcpl/runs
D=/home/omniverse/workspace/safevlmcpl/vlm-with-cpl/new_data
LOGDIR=$W/logs/a5v2
mkdir -p "$LOGDIR"
log_run () { echo "[$(date +%m/%d-%H:%M:%S)] $*" >> "$LOGDIR/progress.log"; }
export -f log_run          # was missing last time, so FAIL lines were never written
export LOGDIR W D

run_awr () {
  local task=$1 tag=$2 beta=$3 clip=$4 seed=$5
  local full="${tag}_seed${seed}"
  local key="${task}_${full}"
  [ -f "$LOGDIR/done_${key}" ] && return 0
  cd $D
  env SAFETY_VLM_TASK=$task WANDB_MODE=disabled OMP_NUM_THREADS=3 \
    AWR_BETA=$beta AWR_WEIGHT_CLIP=$clip SEED_OVERRIDE=$seed OUT_TAG=$full \
    conda run -n safevlmcpl --no-capture-output python scripts/04f_train_v_awr.py \
    > "$LOGDIR/${key}.log" 2>&1 || { log_run "FAIL TRAIN $key"; return 1; }
  env SAFETY_VLM_TASK=$task WANDB_MODE=disabled OMP_NUM_THREADS=3 SEED_OVERRIDE=$seed \
    conda run -n safevlmcpl --no-capture-output python scripts/05_evaluate.py \
    --policy_file "bc_v_awr_${full}_policy.pt" --results_suffix "$full" \
    >> "$LOGDIR/${key}.log" 2>&1 || { log_run "FAIL EVAL $key"; return 1; }
  touch "$LOGDIR/done_${key}"; log_run "DONE $key"
}
export -f run_awr

TASKS="halfcheetah_velocity walker2d_velocity ant_velocity hopper_velocity swimmer_velocity cargoal1_dsrl cargoal2 pointgoal1_dsrl pointgoal2"
J=$W/a5v2_jobs.txt; : > "$J"
for t in $TASKS; do
  for s in 0 1 2; do
    echo "$t vawr_b01  0.1 20  $s" >> "$J"
    echo "$t vawr_b03  0.3 20  $s" >> "$J"
    echo "$t vawr_b3   3.0 20  $s" >> "$J"
    echo "$t vawr_c5   1.0 5   $s" >> "$J"
    echo "$t vawr_c100 1.0 100 $s" >> "$J"
  done
done
log_run "A5v2 start: $(wc -l < $J) jobs"
xargs -a "$J" -L1 -P 7 bash -c 'run_awr "$@"' _
log_run "A5v2 DONE ($(ls $LOGDIR/done_* 2>/dev/null | wc -l)/135)"
