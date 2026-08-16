#!/bin/bash
set -u
S=/tmp/claude-1001/-home-omniverse-workspace-safevlmcpl/cbe3ff25-bd02-4cf4-9f36-173bf5fa270c/scratchpad
LOGDIR=$S/draw2_logs
mkdir -p "$LOGDIR"
cd /home/omniverse/workspace/safevlmcpl/osrl
log_run () { echo "[$(date +%m/%d-%H:%M:%S)] $1" | tee -a "$LOGDIR/progress.log"; }
JOBS=(
  "OfflineHalfCheetahVelocityGymnasium-v1:20:halfcheetah_velocity_draw2_seed1.hdf5"
  "OfflineCarGoal1Gymnasium-v0:25:cargoal1_dsrl_draw2_seed3.hdf5"
  "OfflinePointGoal1Gymnasium-v0:25:pointgoal1_dsrl_draw2_seed0.hdf5"
)
for seed in 0 1 2; do
  while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)" -lt 10000 ]; do sleep 300; done
  log_run "WAVE seed=$seed START"
  for job in "${JOBS[@]}"; do
    IFS=: read e lim h5 <<< "$job"
    tag="cdtdraw2_${e}_s${seed}"
    [ -f "$LOGDIR/done_${tag}" ] && continue
    log_run "START $tag"
    env PYTHONNOUSERSITE=1 PYTHONPATH=/home/omniverse/workspace/safevlmcpl/osrl \
      conda run -n safevlmcpl --no-capture-output \
      python examples/train/train_cdt.py --task "$e" --seed "$seed" \
      --cost_limit "$lim" --device cuda --augment_percent 0.0 --random_aug 0.0 \
      --subset_h5 "$S/certified_h5/$h5" \
      --logdir "$LOGDIR/runs" > "$LOGDIR/${tag}.log" 2>&1 \
      && touch "$LOGDIR/done_${tag}" && log_run "DONE $tag" || log_run "FAIL $tag" &
    sleep 15
  done
  wait
  log_run "WAVE seed=$seed COMPLETE"
done
log_run "DRAW2 ALL DONE"
