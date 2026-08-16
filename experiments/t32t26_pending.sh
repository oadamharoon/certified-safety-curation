#!/bin/bash
# T3.2 AWR temperature/clip sweep (5 cfg x 9 x 3 = 135) + T2.6 aggregators (2 x 9 x 3 = 54). CPU.
set -u
S=/tmp/claude-1001/-home-omniverse-workspace-safevlmcpl/cbe3ff25-bd02-4cf4-9f36-173bf5fa270c/scratchpad
LOGDIR=$S/t32t26_logs
mkdir -p "$LOGDIR"
log_run () { echo "[$(date +%m/%d-%H:%M:%S)] $1" | tee -a "$LOGDIR/progress.log"; }
LIMOF () { case $1 in *velocity*) echo 20;; *) echo 25;; esac; }
export -f LIMOF

run_awr () {
  local task=$1 tag=$2 beta=$3 clip=$4 seed=$5
  local full="${tag}_seed${seed}"
  local key="${task}_${full}"
  [ -f "$LOGDIR/done_${key}" ] && return 0
  cd /home/omniverse/workspace/safevlmcpl/vlm-with-cpl/new_data
  env SAFETY_VLM_TASK=$task WANDB_MODE=disabled OMP_NUM_THREADS=3 \
    AWR_BETA=$beta AWR_WEIGHT_CLIP=$clip SEED_OVERRIDE=$seed OUT_TAG=$full \
    conda run -n safevlmcpl --no-capture-output python scripts/04f_train_v_awr.py \
    > "$LOGDIR/${key}.log" 2>&1 || { echo "FAIL $key" >> "$LOGDIR/progress.log"; return 1; }
  env SAFETY_VLM_TASK=$task WANDB_MODE=disabled OMP_NUM_THREADS=3 \
    conda run -n safevlmcpl --no-capture-output python scripts/05_evaluate.py \
    --policy_file "bc_v_awr_${full}_policy.pt" --results_suffix "$full" \
    >> "$LOGDIR/${key}.log" 2>&1 || { echo "FAIL EVAL $key" >> "$LOGDIR/progress.log"; return 1; }
  touch "$LOGDIR/done_${key}"
  echo "[$(date +%m/%d-%H:%M:%S)] DONE $key" >> "$LOGDIR/progress.log"
}
run_agg () {
  local task=$1 agg=$2 seed=$3
  local lim=$(LIMOF $task)
  local tag="xagg_${agg}_seed${seed}"
  local key="${task}_${tag}"
  [ -f "$LOGDIR/done_${key}" ] && return 0
  cd /home/omniverse/workspace/safevlmcpl/vlm-with-cpl/new_data
  local frac=$(conda run -n safevlmcpl --no-capture-output python $S/t13_frac.py $task $seed | tail -1)
  env SAFETY_VLM_TASK=$task WANDB_MODE=disabled OMP_NUM_THREADS=3 \
    AGG_MODE=$agg FILTER_FRAC=$frac COST_LIMIT=$lim \
    V_ENSEMBLE_FILE=v_ensemble_pess_seed${seed}.pt SEED_OVERRIDE=$seed OUT_TAG=$tag \
    conda run -n safevlmcpl --no-capture-output python scripts/04p_vfilter_bc.py \
    > "$LOGDIR/${key}.log" 2>&1 || { echo "FAIL $key" >> "$LOGDIR/progress.log"; return 1; }
  env SAFETY_VLM_TASK=$task WANDB_MODE=disabled OMP_NUM_THREADS=3 \
    conda run -n safevlmcpl --no-capture-output python scripts/05_evaluate.py \
    --policy_file "bc_${tag}_policy.pt" --results_suffix "$tag" \
    >> "$LOGDIR/${key}.log" 2>&1 || { echo "FAIL EVAL $key" >> "$LOGDIR/progress.log"; return 1; }
  touch "$LOGDIR/done_${key}"
  echo "[$(date +%m/%d-%H:%M:%S)] DONE $key" >> "$LOGDIR/progress.log"
}
export -f run_awr run_agg
export LOGDIR S

TASKS="halfcheetah_velocity walker2d_velocity ant_velocity hopper_velocity swimmer_velocity cargoal1_dsrl cargoal2 pointgoal1_dsrl pointgoal2"
J=$S/pending_jobs.txt
log_run "t32t26 jobs: $(wc -l < $J)"
dispatch () { local k=$1; shift; if [ "$k" = awr ]; then run_awr "$@"; else run_agg "$@"; fi; }
export -f dispatch
xargs -a "$J" -L1 -P 5 bash -c 'dispatch "$@"' _
cd /home/omniverse/workspace/safevlmcpl/iclr2027
conda run -n safevlmcpl --no-capture-output python scripts/collect_results.py >> "$LOGDIR/progress.log" 2>&1
log_run "T32T26 PENDING-QUEUE DONE"
