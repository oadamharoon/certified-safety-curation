#!/bin/bash
set -u
S=/tmp/claude-1001/-home-omniverse-workspace-safevlmcpl/cbe3ff25-bd02-4cf4-9f36-173bf5fa270c/scratchpad
LOGDIR=$S/distinct_logs
mkdir -p "$LOGDIR"
log_run () { echo "[$(date +%m/%d-%H:%M:%S)] $1" | tee -a "$LOGDIR/progress.log"; }
T2ENV () {
  case $1 in
    halfcheetah_velocity) echo "OfflineHalfCheetahVelocityGymnasium-v1:20";;
    walker2d_velocity) echo "OfflineWalker2dVelocityGymnasium-v1:20";;
    ant_velocity) echo "OfflineAntVelocityGymnasium-v1:20";;
    swimmer_velocity) echo "OfflineSwimmerVelocityGymnasium-v1:20";;
    cargoal1_dsrl) echo "OfflineCarGoal1Gymnasium-v0:25";;
    pointgoal1_dsrl) echo "OfflinePointGoal1Gymnasium-v0:25";;
    pointgoal2) echo "OfflinePointGoal2Gymnasium-v0:25";;
  esac
}
export -f T2ENV
run_cdt () {
  local h5=$1 seed=$2
  local base=$(basename "$h5" .hdf5)
  local task=$(echo "$base" | sed -E 's/_(a25|a40)selq[0-9]+_seed[0-9]+//')
  IFS=: read e lim <<< "$(T2ENV $task)"
  local tag="cdt_${base}_s${seed}"
  [ -f "$LOGDIR/done_${tag}" ] && return 0
  cd /home/omniverse/workspace/safevlmcpl/osrl
  env PYTHONNOUSERSITE=1 PYTHONPATH=/home/omniverse/workspace/safevlmcpl/osrl \
    conda run -n safevlmcpl --no-capture-output \
    python examples/train/train_cdt.py --task "$e" --seed "$seed" \
    --cost_limit "$lim" --device cuda --augment_percent 0.0 --random_aug 0.0 \
    --subset_h5 "$h5" --logdir "$LOGDIR/runs" > "$LOGDIR/${tag}.log" 2>&1 \
    && { touch "$LOGDIR/done_${tag}"; echo "[$(date +%m/%d-%H:%M:%S)] DONE $tag" >> "$LOGDIR/progress.log"; } \
    || echo "[$(date +%m/%d-%H:%M:%S)] FAIL $tag" >> "$LOGDIR/progress.log"
}
run_bc () {
  local kj=$1 seed=$2
  local base=$(basename "$kj" _kept.json)
  local task=$(echo "$base" | sed -E 's/_(a25|a40)selq[0-9]+_seed[0-9]+//')
  local q=$(echo "$base" | grep -oE 'a40selq[0-9]+')
  local tag="bc${q}_seed${seed}"
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
export -f run_cdt run_bc
export LOGDIR S
CJ=$S/distinct_cdt_jobs.txt; : > "$CJ"
for h5 in $S/certified_h5/*selq*.hdf5; do
  for seed in 0 1 2; do echo "$h5 $seed" >> "$CJ"; done
done
BJ=$S/distinct_bc_jobs.txt; : > "$BJ"
for kj in $S/certified_h5/*a40selq*_kept.json; do
  for seed in 0 1 2; do echo "$kj $seed" >> "$BJ"; done
done
log_run "CDT jobs: $(wc -l < $CJ) | BC jobs: $(wc -l < $BJ)"
xargs -a "$BJ" -L1 -P 6 bash -c 'run_bc "$@"' _ &
BCPID=$!
xargs -a "$CJ" -L1 -P 5 bash -c 'run_cdt "$@"' _
wait $BCPID
log_run "DISTINCT-SELECTION RUNS ALL DONE"
