#!/bin/bash
# Review-response BC queue: 75 bottom-return jobs + 45 matched-stability jobs.
# BC trains on GPU (small MLPs, coexists with CDT); eval steps envs on CPU.
set -u
S=/tmp/claude-1001/-home-omniverse-workspace-safevlmcpl/cbe3ff25-bd02-4cf4-9f36-173bf5fa270c/scratchpad
LOGDIR=$S/review_logs
mkdir -p "$LOGDIR"
cd /home/omniverse/workspace/safevlmcpl/vlm-with-cpl/new_data

JOBS=$S/review_jobs.txt
: > "$JOBS"
while read -r task lim frac; do
  for seed in 0 1 2 3 4; do
    echo "retbot $task $lim $frac $seed" >> "$JOBS"
  done
done < "$S/fracs.txt"
while read -r task lim frac; do
  for seed in 10 11 12; do
    echo "stab $task $lim $frac $seed" >> "$JOBS"
  done
done < "$S/fracs.txt"
echo "$(wc -l < "$JOBS") jobs queued"

run_job () {
  local kind=$1 task=$2 lim=$3 frac=$4 seed=$5
  local tag log
  if [ "$kind" = retbot ]; then
    tag="vfilt_retbot_seed${seed}"
  else
    tag="lblonly_fixdraw_seed${seed}"
  fi
  log="$LOGDIR/${task}_${tag}.log"
  [ -f "$LOGDIR/done_${task}_${tag}" ] && return 0
  if [ "$kind" = retbot ]; then
    env SAFETY_VLM_TASK="$task" FILTER_FRAC="$frac" COST_LIMIT="$lim" \
      V_ENSEMBLE_FILE=v_ensemble_pess_seed0.pt SEED_OVERRIDE="$seed" \
      SCORE_MODE=return_bottom OUT_TAG="$tag" \
      OMP_NUM_THREADS=3 WANDB_MODE=disabled \
      conda run -n safevlmcpl --no-capture-output \
      python scripts/04p_vfilter_bc.py > "$log" 2>&1 || { echo "TRAIN FAIL $task $tag" >> "$LOGDIR/progress.log"; return 1; }
  else
    env SAFETY_VLM_TASK="$task" FILTER_FRAC="$frac" COST_LIMIT="$lim" \
      CAL_N=200 SPLIT_CAL=0 LABEL_DRAW_SEED=0 SEED_OVERRIDE="$seed" \
      OUT_TAG="$tag" OMP_NUM_THREADS=3 WANDB_MODE=disabled \
      conda run -n safevlmcpl --no-capture-output \
      python scripts/04s_labels_only_filter.py > "$log" 2>&1 || { echo "TRAIN FAIL $task $tag" >> "$LOGDIR/progress.log"; return 1; }
  fi
  env SAFETY_VLM_TASK="$task" OMP_NUM_THREADS=3 WANDB_MODE=disabled CUDA_VISIBLE_DEVICES="" \
    conda run -n safevlmcpl --no-capture-output \
    python scripts/05_evaluate.py --policy_file "bc_${tag}_policy.pt" \
    --results_suffix "$tag" >> "$log" 2>&1 || { echo "EVAL FAIL $task $tag" >> "$LOGDIR/progress.log"; return 1; }
  touch "$LOGDIR/done_${task}_${tag}"
  echo "[$(date +%m/%d-%H:%M:%S)] DONE $task $tag" >> "$LOGDIR/progress.log"
}
export -f run_job
export LOGDIR S

xargs -a "$JOBS" -L1 -P 8 bash -c 'run_job "$@"' _
echo "[$(date +%m/%d-%H:%M:%S)] REVIEW QUEUE ALL DONE" >> "$LOGDIR/progress.log"
