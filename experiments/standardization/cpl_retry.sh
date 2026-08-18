#!/bin/bash
# Retry pass 3: CPL arms (script uses POLICY_OUT, not OUT_TAG).
set -u
W=/home/omniverse/workspace/safevlmcpl/runs
D=/home/omniverse/workspace/safevlmcpl/vlm-with-cpl/new_data
LOGDIR=$W/logs/stage3b
one () {
  local task=$1 seed=$2 tag="cpl_gt_seed${2}"
  local key="${task}_${tag}" pol="cpl_${tag}_policy.pt"
  [ -f "$LOGDIR/done_${key}" ] && return 0
  cd $D
  env SAFETY_VLM_TASK=$task WANDB_MODE=disabled OMP_NUM_THREADS=3 SEED_OVERRIDE=$seed \
      CPL_LAMBDA=0.5 POLICY_OUT="$pol" \
    conda run -n safevlmcpl --no-capture-output python scripts/04c_train_cpl_gt.py \
    > "$LOGDIR/${key}.log" 2>&1 || { echo "[$(date +%m/%d-%H:%M:%S)] FAIL4 $key" >> "$LOGDIR/progress.log"; return 1; }
  env SAFETY_VLM_TASK=$task WANDB_MODE=disabled OMP_NUM_THREADS=3 CUDA_VISIBLE_DEVICES="" \
    conda run -n safevlmcpl --no-capture-output python scripts/05_evaluate.py \
    --policy_file "$pol" --results_suffix "$tag" >> "$LOGDIR/${key}.log" 2>&1 \
    || { echo "[$(date +%m/%d-%H:%M:%S)] FAIL4 EVAL $key" >> "$LOGDIR/progress.log"; return 1; }
  touch "$LOGDIR/done_${key}"; echo "[$(date +%m/%d-%H:%M:%S)] DONE $key" >> "$LOGDIR/progress.log"
}
export -f one; export LOGDIR W D
J=$W/cpl_retry_jobs.txt; : > "$J"
for t in cargoal2 pointgoal1_dsrl pointgoal2; do for s in 0 1 2; do echo "$t $s" >> "$J"; done; done
echo "[$(date +%m/%d-%H:%M:%S)] CPL retry: $(wc -l < $J) jobs" >> "$LOGDIR/progress.log"
xargs -a "$J" -L1 -P 6 bash -c 'one "$@"' _
cd /home/omniverse/workspace/safevlmcpl/iclr2027
conda run -n safevlmcpl --no-capture-output python scripts/collect_results.py >> "$LOGDIR/progress.log" 2>&1
echo "[$(date +%m/%d-%H:%M:%S)] STAGE3B FULLY COMPLETE" >> "$LOGDIR/progress.log"
