#!/bin/bash
# T1.3 safe-mass V-filter (20 tasks x 5 seeds) + T2.2 calsafe fallback (16 x 5). CPU.
set -u
S=/tmp/claude-1001/-home-omniverse-workspace-safevlmcpl/cbe3ff25-bd02-4cf4-9f36-173bf5fa270c/scratchpad
LOGDIR=$S/t13t22_logs
mkdir -p "$LOGDIR"
log_run () { echo "[$(date +%m/%d-%H:%M:%S)] $1" | tee -a "$LOGDIR/progress.log"; }
LIMOF () { case $1 in *velocity*) echo 20;; *_b) echo 10;; *) echo 25;; esac; }
export -f LIMOF

run_t13 () {
  local task=$1 seed=$2
  local lim=$(LIMOF $task)
  local tag="vfilt_calsafe_seed${seed}"
  local key="${task}_${tag}"
  [ -f "$LOGDIR/done_${key}" ] && return 0
  cd /home/omniverse/workspace/safevlmcpl/vlm-with-cpl/new_data
  local frac=$(conda run -n safevlmcpl --no-capture-output python $S/t13_frac.py $task $seed | tail -1)
  env SAFETY_VLM_TASK=$task WANDB_MODE=disabled OMP_NUM_THREADS=3 CUDA_VISIBLE_DEVICES="" \
    FILTER_FRAC=$frac COST_LIMIT=$lim \
    V_ENSEMBLE_FILE=v_ensemble_pess_seed${seed}.pt SEED_OVERRIDE=$seed OUT_TAG=$tag \
    conda run -n safevlmcpl --no-capture-output python scripts/04p_vfilter_bc.py \
    > "$LOGDIR/${key}.log" 2>&1 || { echo "FAIL $key" >> "$LOGDIR/progress.log"; return 1; }
  env SAFETY_VLM_TASK=$task WANDB_MODE=disabled OMP_NUM_THREADS=3 CUDA_VISIBLE_DEVICES="" \
    conda run -n safevlmcpl --no-capture-output python scripts/05_evaluate.py \
    --policy_file "bc_${tag}_policy.pt" --results_suffix "$tag" \
    >> "$LOGDIR/${key}.log" 2>&1 || { echo "FAIL EVAL $key" >> "$LOGDIR/progress.log"; return 1; }
  touch "$LOGDIR/done_${key}"
  echo "[$(date +%m/%d-%H:%M:%S)] DONE $key" >> "$LOGDIR/progress.log"
}
run_t22 () {
  local task=$1 seed=$2
  local lim=$(LIMOF $task)
  local tag="calfilt_csf_seed${seed}"
  local key="${task}_${tag}"
  [ -f "$LOGDIR/done_${key}" ] && return 0
  cd /home/omniverse/workspace/safevlmcpl/vlm-with-cpl/new_data
  env SAFETY_VLM_TASK=$task WANDB_MODE=disabled OMP_NUM_THREADS=3 CUDA_VISIBLE_DEVICES="" \
    MODE=ltt CAL_N=200 ALPHA=0.25 DELTA=0.1 FALLBACK_MODE=calsafe COST_LIMIT=$lim \
    V_ENSEMBLE_FILE=v_ensemble_pess_seed${seed}.pt SEED_OVERRIDE=$seed OUT_TAG=$tag \
    conda run -n safevlmcpl --no-capture-output python scripts/04q_calibrated_vfilter.py \
    > "$LOGDIR/${key}.log" 2>&1 || { echo "FAIL $key" >> "$LOGDIR/progress.log"; return 1; }
  env SAFETY_VLM_TASK=$task WANDB_MODE=disabled OMP_NUM_THREADS=3 CUDA_VISIBLE_DEVICES="" \
    conda run -n safevlmcpl --no-capture-output python scripts/05_evaluate.py \
    --policy_file "bc_${tag}_policy.pt" --results_suffix "$tag" \
    >> "$LOGDIR/${key}.log" 2>&1 || { echo "FAIL EVAL $key" >> "$LOGDIR/progress.log"; return 1; }
  touch "$LOGDIR/done_${key}"
  echo "[$(date +%m/%d-%H:%M:%S)] DONE $key" >> "$LOGDIR/progress.log"
}
export -f run_t13 run_t22
export LOGDIR S

DSRL="halfcheetah_velocity walker2d_velocity ant_velocity hopper_velocity swimmer_velocity cargoal1_dsrl cargoal2 pointgoal1_dsrl pointgoal2 pointbutton1 pointbutton2 carbutton1_t3 carbutton2 pointcircle1 pointcircle2"
BULLET="ballrun_b ballcircle_b carcircle_b carrun_b dronerun_b"
UNCERT="walker2d_velocity ant_velocity hopper_velocity swimmer_velocity cargoal2 pointgoal2 pointbutton1 pointbutton2 carbutton1_t3 carbutton2 pointcircle1 pointcircle2 ballrun_b ballcircle_b carcircle_b dronerun_b"
J=$S/t13t22_jobs.txt; : > "$J"
for t in $DSRL $BULLET; do for s in 0 1 2 3 4; do echo "t13 $t $s" >> "$J"; done; done
for t in $UNCERT; do for s in 0 1 2 3 4; do echo "t22 $t $s" >> "$J"; done; done
log_run "t13t22 jobs: $(wc -l < $J)"
dispatch () { local k=$1; shift; if [ "$k" = t13 ]; then run_t13 "$@"; else run_t22 "$@"; fi; }
export -f dispatch
xargs -a "$J" -L1 -P 4 bash -c 'dispatch "$@"' _
cd /home/omniverse/workspace/safevlmcpl/iclr2027
conda run -n safevlmcpl --no-capture-output python scripts/collect_results.py >> "$LOGDIR/progress.log" 2>&1
log_run "T13T22 QUEUE ALL DONE"
