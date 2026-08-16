#!/bin/bash
# Margin-regularized certified operator: lambda=1 x c{1,2,3} + lambda{0.5,2} at c=2,
# identical 10 selections x 3 seeds. GPU training, CPU evals.
set -u
S=/tmp/claude-1001/-home-omniverse-workspace-safevlmcpl/cbe3ff25-bd02-4cf4-9f36-173bf5fa270c/scratchpad
LOGDIR=$S/mwbc_logs
mkdir -p "$LOGDIR"
log_run () { echo "[$(date +%m/%d-%H:%M:%S)] $1" | tee -a "$LOGDIR/progress.log"; }

run_arm () {
  local task=$1 kj=$2 mj=$3 tag=$4 clip=$5 lam=$6 seed=$7
  local full="${tag}_seed${seed}"
  local key="${task}_${full}"
  [ -f "$LOGDIR/done_${key}" ] && return 0
  cd /home/omniverse/workspace/safevlmcpl/vlm-with-cpl/new_data
  env SAFETY_VLM_TASK=$task KEPT_JSON="$kj" SEED_OVERRIDE=$seed OUT_TAG="$full" \
    RETURN_WEIGHTED=1 WEIGHT_CLIP=$clip MARGIN_JSON="$mj" MARGIN_LAMBDA=$lam \
    WANDB_MODE=disabled OMP_NUM_THREADS=4 \
    conda run -n safevlmcpl --no-capture-output python $S/bc_on_subset.py \
    > "$LOGDIR/${key}.log" 2>&1 || { echo "FAIL $key" >> "$LOGDIR/progress.log"; return 1; }
  env SAFETY_VLM_TASK=$task WANDB_MODE=disabled OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" \
    conda run -n safevlmcpl --no-capture-output python scripts/05_evaluate.py \
    --policy_file "bc_${full}_policy.pt" --results_suffix "$full" \
    >> "$LOGDIR/${key}.log" 2>&1 || { echo "FAIL EVAL $key" >> "$LOGDIR/progress.log"; return 1; }
  touch "$LOGDIR/done_${key}"
  echo "[$(date +%m/%d-%H:%M:%S)] DONE $key" >> "$LOGDIR/progress.log"
}
export -f run_arm
export LOGDIR S

J=$S/mwbc_jobs.txt; : > "$J"
SELS="halfcheetah_velocity:a25wq85:q85 halfcheetah_velocity:a25wq80:q80 halfcheetah_velocity:a25wq75:q75 \
cargoal1_dsrl:a25wq85:q85 cargoal1_dsrl:a25wq80:q80 cargoal1_dsrl:a25wq75:q75 \
pointgoal1_dsrl:a25wq85:q85 pointgoal1_dsrl:a25wq70:q70 pointgoal1_dsrl:a25wq65:q65 \
carrun_b:echocert_seed0:echo"
for spec in $SELS; do
  IFS=: read task sel q <<< "$spec"
  kj="$S/certified_h5/${task}_${sel}_kept.json"
  mj="$S/certified_h5/${task}_${sel}_margins.json"
  for seed in 0 1 2; do
    echo "$task $kj $mj mc1${q} 1.0 1.0 $seed" >> "$J"
    echo "$task $kj $mj mc2${q} 2.0 1.0 $seed" >> "$J"
    echo "$task $kj $mj mc3${q} 3.0 1.0 $seed" >> "$J"
    echo "$task $kj $mj ms05${q} 2.0 0.5 $seed" >> "$J"
    echo "$task $kj $mj ms2${q} 2.0 2.0 $seed" >> "$J"
  done
done
log_run "mwbc jobs: $(wc -l < $J)"
xargs -a "$J" -L1 -P 6 bash -c 'run_arm "$@"' _
cd /home/omniverse/workspace/safevlmcpl/iclr2027
conda run -n safevlmcpl --no-capture-output python scripts/collect_results.py >> "$LOGDIR/progress.log" 2>&1
log_run "MWBC QUEUE ALL DONE"
