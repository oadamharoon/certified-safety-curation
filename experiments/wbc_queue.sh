#!/bin/bash
# Certified return-weighted cloning feasibility: 9 a25 selections x 3 seeds
# + CarRun certified selection x 3 seeds. CPU only.
set -u
S=/tmp/claude-1001/-home-omniverse-workspace-safevlmcpl/cbe3ff25-bd02-4cf4-9f36-173bf5fa270c/scratchpad
LOGDIR=$S/wbc_logs
mkdir -p "$LOGDIR"
log_run () { echo "[$(date +%m/%d-%H:%M:%S)] $1" | tee -a "$LOGDIR/progress.log"; }

run_wbc () {
  local task=$1 kj=$2 tag_base=$3 seed=$4
  local tag="${tag_base}_seed${seed}"
  local key="${task}_${tag}"
  [ -f "$LOGDIR/done_${key}" ] && return 0
  cd /home/omniverse/workspace/safevlmcpl/vlm-with-cpl/new_data
  env SAFETY_VLM_TASK=$task KEPT_JSON="$kj" SEED_OVERRIDE=$seed OUT_TAG="$tag" \
    RETURN_WEIGHTED=1 WANDB_MODE=disabled OMP_NUM_THREADS=4 \
    conda run -n safevlmcpl --no-capture-output python $S/bc_on_subset.py \
    > "$LOGDIR/${key}.log" 2>&1 || { echo "FAIL $key" >> "$LOGDIR/progress.log"; return 1; }
  env SAFETY_VLM_TASK=$task WANDB_MODE=disabled OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" \
    conda run -n safevlmcpl --no-capture-output python scripts/05_evaluate.py \
    --policy_file "bc_${tag}_policy.pt" --results_suffix "$tag" \
    >> "$LOGDIR/${key}.log" 2>&1 || { echo "FAIL EVAL $key" >> "$LOGDIR/progress.log"; return 1; }
  touch "$LOGDIR/done_${key}"
  echo "[$(date +%m/%d-%H:%M:%S)] DONE $key" >> "$LOGDIR/progress.log"
}
export -f run_wbc
export LOGDIR S

J=$S/wbc_jobs.txt; : > "$J"
for spec in \
  "halfcheetah_velocity a25wq85 wbcq85" "halfcheetah_velocity a25wq80 wbcq80" "halfcheetah_velocity a25wq75 wbcq75" \
  "cargoal1_dsrl a25wq85 wbcq85" "cargoal1_dsrl a25wq80 wbcq80" "cargoal1_dsrl a25wq75 wbcq75" \
  "pointgoal1_dsrl a25wq70 wbcq70" "pointgoal1_dsrl a25wq85 wbcq85" "pointgoal1_dsrl a25wq65 wbcq65"; do
  set -- $spec
  for seed in 0 1 2; do
    echo "$1 $S/certified_h5/$1_$2_kept.json $3 $seed" >> "$J"
  done
done
for seed in 0 1 2; do
  echo "carrun_b $S/certified_h5/carrun_b_echocert_seed0_kept.json wbcecho $seed" >> "$J"
done
log_run "wbc jobs: $(wc -l < $J)"
xargs -a "$J" -L1 -P 6 bash -c 'run_wbc "$@"' _
cd /home/omniverse/workspace/safevlmcpl/iclr2027
conda run -n safevlmcpl --no-capture-output python scripts/collect_results.py >> "$LOGDIR/progress.log" 2>&1
log_run "WBC QUEUE ALL DONE"
