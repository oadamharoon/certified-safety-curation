#!/bin/bash
# Stage 2 of 1000-pair standardization: retrain all affected V ensembles.
#   12 main tasks x 5 seeds (60) + 18 H-variants x 3 seeds (54) = 114
set -u
S=/tmp/claude-1001/-home-omniverse-workspace-safevlmcpl/cbe3ff25-bd02-4cf4-9f36-173bf5fa270c/scratchpad
LOGDIR=$S/restd_v_logs
mkdir -p "$LOGDIR"
log_run () { echo "[$(date +%m/%d-%H:%M:%S)] $1" | tee -a "$LOGDIR/progress.log"; }

train_v () {
  local task=$1 seed=$2 cfgfile=$3 key_override=${4:-}
  local key="${key_override:-${task}}_s${seed}"
  [ -f "$LOGDIR/done_${key}" ] && return 0
  cd /home/omniverse/workspace/safevlmcpl/vlm-with-cpl/new_data
  local CFG=""
  [ "$cfgfile" != "-" ] && CFG="SAFETY_VLM_CONFIG=$cfgfile"
  env SAFETY_VLM_TASK=$task $CFG WANDB_MODE=disabled OMP_NUM_THREADS=3 \
    SEED_OVERRIDE=$seed \
    conda run -n safevlmcpl --no-capture-output python scripts/04n_train_v_only.py \
    > "$LOGDIR/${key}.log" 2>&1 \
    && { touch "$LOGDIR/done_${key}"; echo "[$(date +%m/%d-%H:%M:%S)] DONE $key" >> "$LOGDIR/progress.log"; } \
    || echo "[$(date +%m/%d-%H:%M:%S)] FAIL $key" >> "$LOGDIR/progress.log"
}
export -f train_v
export LOGDIR S

J=$S/restd_v_jobs.txt; : > "$J"
for t in pointgoal2 pointbutton1 pointbutton2 carbutton1_t3 carbutton2 pointcircle1 pointcircle2 \
         ballrun_b ballcircle_b carcircle_b carrun_b dronerun_b; do
  for s in 0 1 2 3 4; do echo "$t $s -" >> "$J"; done
done
for t in halfcheetah_velocity walker2d_velocity ant_velocity hopper_velocity swimmer_velocity \
         cargoal1_dsrl cargoal2 pointgoal1_dsrl pointgoal2; do
  for h in 10 50; do
    for s in 0 1 2; do echo "$t $s config_h${h}.yaml ${t}_h${h}" >> "$J"; done
  done
done
log_run "stage2 V retrainings: $(wc -l < $J)"
xargs -a "$J" -L1 -P 6 bash -c 'train_v "$@"' _
log_run "STAGE2 V RETRAIN DONE ($(ls $LOGDIR/done_* 2>/dev/null | wc -l) ok)"
