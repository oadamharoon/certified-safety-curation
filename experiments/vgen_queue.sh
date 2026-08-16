#!/bin/bash
set -u
S=/tmp/claude-1001/-home-omniverse-workspace-safevlmcpl/cbe3ff25-bd02-4cf4-9f36-173bf5fa270c/scratchpad
LOGDIR=$S/vgen_logs
mkdir -p "$LOGDIR"
cd /home/omniverse/workspace/safevlmcpl/vlm-with-cpl/new_data

run_chain () {
  local task=$1 lim=$2 arm=$3 seed=$4
  local vout="v_ensemble_vgen${arm}_seed${seed}.pt"
  local tag="calfilt_vgen${arm}_seed${seed}"
  local key="${task}_${arm}_s${seed}"
  local log="$LOGDIR/${key}.log"
  [ -f "$LOGDIR/done_${key}" ] && return 0
  env SAFETY_VLM_TASK=$task WANDB_MODE=disabled OMP_NUM_THREADS=3 \
    PREF_LABELS_FILE="gt_labels_vgen${arm}.json" SEED_OVERRIDE=$seed V_OUT="$vout" \
    conda run -n safevlmcpl --no-capture-output python scripts/04n_train_v_only.py \
    > "$log" 2>&1 || { echo "FAIL V $key" >> "$LOGDIR/progress.log"; return 1; }
  env SAFETY_VLM_TASK=$task WANDB_MODE=disabled OMP_NUM_THREADS=3 \
    MODE=ltt CAL_N=200 ALPHA=0.25 DELTA=0.1 COST_LIMIT=$lim \
    V_ENSEMBLE_FILE="$vout" SEED_OVERRIDE=$seed OUT_TAG="$tag" \
    conda run -n safevlmcpl --no-capture-output python scripts/04q_calibrated_vfilter.py \
    >> "$log" 2>&1 || { echo "FAIL CALFILT $key" >> "$LOGDIR/progress.log"; return 1; }
  env SAFETY_VLM_TASK=$task WANDB_MODE=disabled OMP_NUM_THREADS=3 CUDA_VISIBLE_DEVICES="" \
    conda run -n safevlmcpl --no-capture-output python scripts/05_evaluate.py \
    --policy_file "bc_${tag}_policy.pt" --results_suffix "$tag" \
    >> "$log" 2>&1 || { echo "FAIL EVAL $key" >> "$LOGDIR/progress.log"; return 1; }
  touch "$LOGDIR/done_${key}"
  echo "[$(date +%m/%d-%H:%M:%S)] DONE $key" >> "$LOGDIR/progress.log"
}
export -f run_chain
export LOGDIR S

JOBS=$S/vgen_jobs.txt
: > "$JOBS"
for tl in "halfcheetah_velocity 20" "cargoal1_dsrl 25" "pointgoal1_dsrl 25" "pointgoal2 25"; do
  for arm in 1k 5k; do
    for seed in 0 1 2; do
      echo "$tl $arm $seed" >> "$JOBS"
    done
  done
done
xargs -a "$JOBS" -L1 -P 4 bash -c 'run_chain "$@"' _
echo "[$(date +%m/%d-%H:%M:%S)] VGEN ALL DONE" >> "$LOGDIR/progress.log"
