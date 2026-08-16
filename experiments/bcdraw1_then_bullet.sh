#!/bin/bash
# 1) Reconstruct + verify draw-1 selections; 2) 24 BC runs; 3) chain BulletGym.
set -u
S=/tmp/claude-1001/-home-omniverse-workspace-safevlmcpl/cbe3ff25-bd02-4cf4-9f36-173bf5fa270c/scratchpad
LOGDIR=$S/bcdraw1_logs
mkdir -p "$LOGDIR"
log_run () { echo "[$(date +%m/%d-%H:%M:%S)] $1" | tee -a "$LOGDIR/progress.log"; }

cd /home/omniverse/workspace/safevlmcpl/vlm-with-cpl/new_data
if ! ls $S/certified_h5/*_a40d1_*_kept.json >/dev/null 2>&1; then
  log_run "RECONSTRUCT+VERIFY START"
  env CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 \
    conda run -n safevlmcpl --no-capture-output python $S/build_bcdraw1.py \
    > "$LOGDIR/build.log" 2>&1 || { log_run "RECONSTRUCTION FAILED - ABORT"; exit 1; }
  log_run "RECONSTRUCT+VERIFY DONE"
fi

run_bc () {
  local task=$1 kj=$2 seed=$3
  local tag="bca40d1_seed${seed}"
  local key="${task}_${tag}"
  [ -f "$LOGDIR/done_${key}" ] && return 0
  cd /home/omniverse/workspace/safevlmcpl/vlm-with-cpl/new_data
  env SAFETY_VLM_TASK=$task KEPT_JSON="$kj" SEED_OVERRIDE=$seed OUT_TAG="$tag" \
    WANDB_MODE=disabled OMP_NUM_THREADS=3 CUDA_VISIBLE_DEVICES="" \
    conda run -n safevlmcpl --no-capture-output python $S/bc_on_subset.py \
    > "$LOGDIR/${key}.log" 2>&1 || { echo "FAIL BC $key" >> "$LOGDIR/progress.log"; return 1; }
  env SAFETY_VLM_TASK=$task WANDB_MODE=disabled OMP_NUM_THREADS=3 CUDA_VISIBLE_DEVICES="" \
    conda run -n safevlmcpl --no-capture-output python scripts/05_evaluate.py \
    --policy_file "bc_${tag}_policy.pt" --results_suffix "$tag" \
    >> "$LOGDIR/${key}.log" 2>&1 || { echo "FAIL EVAL $key" >> "$LOGDIR/progress.log"; return 1; }
  touch "$LOGDIR/done_${key}"
  echo "[$(date +%m/%d-%H:%M:%S)] DONE $key" >> "$LOGDIR/progress.log"
}
export -f run_bc
export LOGDIR S

JOBS=$S/bcdraw1_jobs.txt
: > "$JOBS"
for kj in $S/certified_h5/*_a40d1_*_kept.json; do
  base=$(basename "$kj" _kept.json)
  task=$(echo "$base" | sed -E 's/_a40d1_seed[0-9]+//')
  for seed in 0 1 2; do
    echo "$task $kj $seed" >> "$JOBS"
  done
done
log_run "BC jobs: $(wc -l < "$JOBS")"
xargs -a "$JOBS" -L1 -P 6 bash -c 'run_bc "$@"' _
log_run "BC DRAW-1 ARM DONE"

# 3) chain BulletGym: wait (up to 48h) for the pipeline script + ready flag
waited=0
while [ ! -f "$S/bullet_pipeline_ready" ] && [ $waited -lt 172800 ]; do
  sleep 300; waited=$((waited+300))
done
if [ -f "$S/bullet_pipeline_ready" ]; then
  log_run "CHAINING BULLETGYM"
  bash $S/bullet_pipeline.sh > $S/bullet_pipeline.out 2>&1
  log_run "BULLETGYM PIPELINE EXITED"
else
  log_run "bullet_pipeline_ready never appeared; not chained"
fi
