#!/bin/bash
# Symmetric three-draw program: build 19 subsets, then run
#   CDT: 9 (draw3 @ .25) + 48 (a40 draws 2-3), 5 concurrent on GPU
#   BC : 48 identical-selection runs (a40 draws 2-3), CPU, 6 concurrent
set -u
S=/tmp/claude-1001/-home-omniverse-workspace-safevlmcpl/cbe3ff25-bd02-4cf4-9f36-173bf5fa270c/scratchpad
LOGDIR=$S/draws3_logs
mkdir -p "$LOGDIR"
log_run () { echo "[$(date +%m/%d-%H:%M:%S)] $1" | tee -a "$LOGDIR/progress.log"; }

# --- stage 0: build subsets (idempotent-ish: skip if all 19 h5s exist) ---
n_h5=$(ls $S/certified_h5/*_draw3_*.hdf5 $S/certified_h5/*_a40d2_*.hdf5 $S/certified_h5/*_a40d3_*.hdf5 2>/dev/null | wc -l)
if [ "$n_h5" -lt 19 ]; then
  log_run "BUILD SUBSETS START"
  cd /home/omniverse/workspace/safevlmcpl/vlm-with-cpl/new_data
  env CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 \
    conda run -n safevlmcpl --no-capture-output python $S/build_subsets2.py \
    > "$LOGDIR/build.log" 2>&1 || { log_run "BUILD FAILED"; exit 1; }
  log_run "BUILD SUBSETS DONE"
fi

T2ENV () {
  case $1 in
    halfcheetah_velocity) echo "OfflineHalfCheetahVelocityGymnasium-v1:20";;
    walker2d_velocity) echo "OfflineWalker2dVelocityGymnasium-v1:20";;
    ant_velocity) echo "OfflineAntVelocityGymnasium-v1:20";;
    swimmer_velocity) echo "OfflineSwimmerVelocityGymnasium-v1:20";;
    cargoal1_dsrl) echo "OfflineCarGoal1Gymnasium-v0:25";;
    cargoal2) echo "OfflineCarGoal2Gymnasium-v0:25";;
    pointgoal1_dsrl) echo "OfflinePointGoal1Gymnasium-v0:25";;
    pointgoal2) echo "OfflinePointGoal2Gymnasium-v0:25";;
  esac
}
export -f T2ENV

# --- CDT job list: every new subset x 3 seeds ---
CJOBS=$S/draws3_cdt_jobs.txt
: > "$CJOBS"
for h5 in $S/certified_h5/*_draw3_*.hdf5 $S/certified_h5/*_a40d2_*.hdf5 $S/certified_h5/*_a40d3_*.hdf5; do
  base=$(basename "$h5" .hdf5)
  task=$(echo "$base" | sed -E 's/_(draw3|a40d2|a40d3)_seed[0-9]+//')
  for seed in 0 1 2; do
    echo "$task $h5 $seed" >> "$CJOBS"
  done
done
log_run "CDT jobs: $(wc -l < $CJOBS)"

run_cdt () {
  local task=$1 h5=$2 seed=$3
  IFS=: read e lim <<< "$(T2ENV $task)"
  local tag="cdt_$(basename $h5 .hdf5)_s${seed}"
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
export -f run_cdt
export LOGDIR S

# --- BC job list: a40 draws 2-3 identical selections x 3 seeds ---
BJOBS=$S/draws3_bc_jobs.txt
: > "$BJOBS"
for kj in $S/certified_h5/*_a40d2_*_kept.json $S/certified_h5/*_a40d3_*_kept.json; do
  base=$(basename "$kj" _kept.json)
  task=$(echo "$base" | sed -E 's/_(a40d2|a40d3)_seed[0-9]+//')
  arm=$(echo "$base" | grep -oE 'a40d[23]')
  for seed in 0 1 2; do
    echo "$task $kj $arm $seed" >> "$BJOBS"
  done
done
log_run "BC jobs: $(wc -l < $BJOBS)"

run_bc () {
  local task=$1 kj=$2 arm=$3 seed=$4
  IFS=: read e lim <<< "$(T2ENV $task)"
  local tag="bc${arm}_seed${seed}"
  local key="${task}_${tag}"
  [ -f "$LOGDIR/done_${key}" ] && return 0
  cd /home/omniverse/workspace/safevlmcpl/vlm-with-cpl/new_data
  env SAFETY_VLM_TASK=$task KEPT_JSON="$kj" SEED_OVERRIDE=$seed OUT_TAG="$tag" \
    WANDB_MODE=disabled OMP_NUM_THREADS=3 CUDA_VISIBLE_DEVICES="" \
    conda run -n safevlmcpl --no-capture-output python $S/bc_on_subset.py \
    > "$LOGDIR/${key}.log" 2>&1 || { echo "[$(date +%m/%d-%H:%M:%S)] FAIL BC $key" >> "$LOGDIR/progress.log"; return 1; }
  env SAFETY_VLM_TASK=$task WANDB_MODE=disabled OMP_NUM_THREADS=3 CUDA_VISIBLE_DEVICES="" \
    conda run -n safevlmcpl --no-capture-output python scripts/05_evaluate.py \
    --policy_file "bc_${tag}_policy.pt" --results_suffix "$tag" \
    >> "$LOGDIR/${key}.log" 2>&1 || { echo "[$(date +%m/%d-%H:%M:%S)] FAIL EVAL $key" >> "$LOGDIR/progress.log"; return 1; }
  touch "$LOGDIR/done_${key}"
  echo "[$(date +%m/%d-%H:%M:%S)] DONE $key" >> "$LOGDIR/progress.log"
}
export -f run_bc

xargs -a "$BJOBS" -L1 -P 6 bash -c 'run_bc "$@"' _ &
BCPID=$!
xargs -a "$CJOBS" -L1 -P 5 bash -c 'run_cdt "$@"' _
wait $BCPID
log_run "THREE-DRAW PROGRAM ALL DONE"
