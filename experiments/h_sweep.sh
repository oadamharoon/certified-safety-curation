#!/bin/bash
# Segment-length sweep: H in {10, 50}, 9 analysis tasks, full pipeline.
# Per (task, H): 01 segment -> 03b label -> 04n V x3 -> 04q calfilt+BC x3 -> 05 eval.
# Runs alongside the no-aug CDT waves; BC/V nets are small.
set -u
S=/tmp/claude-1001/-home-omniverse-workspace-safevlmcpl/cbe3ff25-bd02-4cf4-9f36-173bf5fa270c/scratchpad
LOGDIR=$S/hsweep_logs
mkdir -p "$LOGDIR"
cd /home/omniverse/workspace/safevlmcpl/vlm-with-cpl/new_data
log_run () { echo "[$(date +%m/%d-%H:%M:%S)] $1" | tee -a "$LOGDIR/progress.log"; }

TASKS="halfcheetah_velocity:20 walker2d_velocity:20 ant_velocity:20 hopper_velocity:20 swimmer_velocity:20 cargoal1_dsrl:25 cargoal2:25 pointgoal1_dsrl:25 pointgoal2:25"

run_chain () {
  local task=$1 lim=$2 h=$3
  local cfgf="config_h${h}.yaml"
  local key="${task}_h${h}"
  local B="env SAFETY_VLM_CONFIG=$cfgf SAFETY_VLM_TASK=$task WANDB_MODE=disabled OMP_NUM_THREADS=3"
  if [ ! -f "$LOGDIR/done_seg_${key}" ]; then
    log_run "SEG $key"
    $B conda run -n safevlmcpl --no-capture-output python scripts/01_segment_and_filter.py \
      > "$LOGDIR/${key}_01.log" 2>&1 || { log_run "FAIL seg $key"; return 1; }
    $B conda run -n safevlmcpl --no-capture-output python scripts/03b_label_by_cost.py \
      >> "$LOGDIR/${key}_01.log" 2>&1 || { log_run "FAIL label $key"; return 1; }
    touch "$LOGDIR/done_seg_${key}"
  fi
  for seed in 0 1 2; do
    if [ ! -f "$LOGDIR/done_v_${key}_s${seed}" ]; then
      log_run "V $key s$seed"
      $B SEED_OVERRIDE=$seed conda run -n safevlmcpl --no-capture-output \
        python scripts/04n_train_v_only.py > "$LOGDIR/${key}_v${seed}.log" 2>&1 \
        || { log_run "FAIL V $key s$seed"; return 1; }
      touch "$LOGDIR/done_v_${key}_s${seed}"
    fi
  done
  for seed in 0 1 2; do
    local tag="calfilt_h${h}_seed${seed}"
    if [ ! -f "$LOGDIR/done_bc_${key}_s${seed}" ]; then
      log_run "CALFILT $key s$seed"
      $B MODE=ltt CAL_N=200 ALPHA=0.25 DELTA=0.1 COST_LIMIT=$lim \
        V_ENSEMBLE_FILE=v_ensemble_pess_seed${seed}.pt SEED_OVERRIDE=$seed \
        OUT_TAG=$tag conda run -n safevlmcpl --no-capture-output \
        python scripts/04q_calibrated_vfilter.py > "$LOGDIR/${key}_bc${seed}.log" 2>&1 \
        || { log_run "FAIL calfilt $key s$seed"; return 1; }
      $B CUDA_VISIBLE_DEVICES="" conda run -n safevlmcpl --no-capture-output \
        python scripts/05_evaluate.py --policy_file "bc_${tag}_policy.pt" \
        --results_suffix "$tag" >> "$LOGDIR/${key}_bc${seed}.log" 2>&1 \
        || { log_run "FAIL eval $key s$seed"; return 1; }
      touch "$LOGDIR/done_bc_${key}_s${seed}"
    fi
  done
  log_run "CHAIN DONE $key"
}
export -f run_chain log_run
export LOGDIR S

JOBS=$S/hsweep_jobs.txt
: > "$JOBS"
for h in 10 50; do
  for tl in $TASKS; do
    echo "${tl%%:*} ${tl##*:} $h" >> "$JOBS"
  done
done
xargs -a "$JOBS" -L1 -P 4 bash -c 'run_chain "$@"' _
log_run "H SWEEP ALL DONE"
