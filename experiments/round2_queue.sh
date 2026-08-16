#!/bin/bash
# Review-round-2 remaining runs, compute-minded:
#  now (CPU, P=4 alongside distinct queue): tier-2 fallback (8 tasks x 3 seeds)
#                                           + bullet random control (5 x 5)
#  after distinct queue drains (GPU): full-data CDT on 5 bullet tasks x 3 seeds
#  then: final harvest.
set -u
S=/tmp/claude-1001/-home-omniverse-workspace-safevlmcpl/cbe3ff25-bd02-4cf4-9f36-173bf5fa270c/scratchpad
LOGDIR=$S/round2_logs
mkdir -p "$LOGDIR"
log_run () { echo "[$(date +%m/%d-%H:%M:%S)] $1" | tee -a "$LOGDIR/progress.log"; }

LIMOF () {
  case $1 in
    hopper_velocity|swimmer_velocity) echo 20;;
    pointbutton2|pointcircle2) echo 25;;
    *_b) echo 10;;
  esac
}
FRACOF () {
  case $1 in
    ballrun_b) echo 0.090;; ballcircle_b) echo 0.097;; carcircle_b) echo 0.114;;
    carrun_b) echo 0.462;; dronerun_b) echo 0.280;;
  esac
}
export -f LIMOF FRACOF

run_t2 () {
  local task=$1 seed=$2
  local lim=$(LIMOF $task)
  local tag="calfilt_t2_seed${seed}"
  local key="${task}_${tag}"
  [ -f "$LOGDIR/done_${key}" ] && return 0
  cd /home/omniverse/workspace/safevlmcpl/vlm-with-cpl/new_data
  env SAFETY_VLM_TASK=$task WANDB_MODE=disabled OMP_NUM_THREADS=3 CUDA_VISIBLE_DEVICES="" \
    MODE=ltt CAL_N=200 ALPHA=0.25 DELTA=0.1 TIER2_DELTA=0.5 COST_LIMIT=$lim \
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
run_rand () {
  local task=$1 seed=$2
  local lim=$(LIMOF $task); local frac=$(FRACOF $task)
  local tag="vfilt_random_matchgt_seed${seed}"
  local key="${task}_${tag}"
  [ -f "$LOGDIR/done_${key}" ] && return 0
  cd /home/omniverse/workspace/safevlmcpl/vlm-with-cpl/new_data
  env SAFETY_VLM_TASK=$task WANDB_MODE=disabled OMP_NUM_THREADS=3 CUDA_VISIBLE_DEVICES="" \
    SCORE_MODE=random FILTER_FRAC=$frac COST_LIMIT=$lim \
    V_ENSEMBLE_FILE=v_ensemble_pess_seed0.pt SEED_OVERRIDE=$seed OUT_TAG=$tag \
    conda run -n safevlmcpl --no-capture-output python scripts/04p_vfilter_bc.py \
    > "$LOGDIR/${key}.log" 2>&1 || { echo "FAIL $key" >> "$LOGDIR/progress.log"; return 1; }
  env SAFETY_VLM_TASK=$task WANDB_MODE=disabled OMP_NUM_THREADS=3 CUDA_VISIBLE_DEVICES="" \
    conda run -n safevlmcpl --no-capture-output python scripts/05_evaluate.py \
    --policy_file "bc_${tag}_policy.pt" --results_suffix "$tag" \
    >> "$LOGDIR/${key}.log" 2>&1 || { echo "FAIL EVAL $key" >> "$LOGDIR/progress.log"; return 1; }
  touch "$LOGDIR/done_${key}"
  echo "[$(date +%m/%d-%H:%M:%S)] DONE $key" >> "$LOGDIR/progress.log"
}
export -f run_t2 run_rand
export LOGDIR S

J=$S/round2_cpu_jobs.txt; : > "$J"
for t in hopper_velocity swimmer_velocity pointbutton2 pointcircle2 ballrun_b ballcircle_b carcircle_b dronerun_b; do
  for s in 0 1 2; do echo "t2 $t $s" >> "$J"; done
done
for t in ballrun_b ballcircle_b carcircle_b carrun_b dronerun_b; do
  for s in 0 1 2 3 4; do echo "rand $t $s" >> "$J"; done
done
log_run "CPU jobs: $(wc -l < $J) (tier2 24 + random 25)"
dispatch () { local kind=$1; shift; if [ "$kind" = t2 ]; then run_t2 "$@"; else run_rand "$@"; fi; }
export -f dispatch
xargs -a "$J" -L1 -P 2 bash -c 'dispatch "$@"' _
log_run "CPU STAGE DONE"

# wait for the distinct-selection GPU queue to drain, then bullet full-data CDT
while ! grep -q "DISTINCT-SELECTION RUNS ALL DONE" $S/distinct_logs/progress.log 2>/dev/null; do
  sleep 600
done
log_run "GPU FREE - BULLET CDT START"
BENVS="OfflineBallRun-v0 OfflineBallCircle-v0 OfflineCarCircle-v0 OfflineCarRun-v0 OfflineDroneRun-v0"
for seed in 0 1 2; do
  for e in $BENVS; do
    tag="cdtbullet_${e}_s${seed}"
    [ -f "$LOGDIR/done_${tag}" ] && continue
    cd /home/omniverse/workspace/safevlmcpl/osrl
    env PYTHONNOUSERSITE=1 PYTHONPATH=/home/omniverse/workspace/safevlmcpl/osrl \
      conda run -n safevlmcpl --no-capture-output \
      python examples/train/train_cdt.py --task "$e" --seed "$seed" \
      --cost_limit 10 --device cuda --logdir "$LOGDIR/runs" \
      > "$LOGDIR/${tag}.log" 2>&1 \
      && { touch "$LOGDIR/done_${tag}"; echo "[$(date +%m/%d-%H:%M:%S)] DONE $tag" >> "$LOGDIR/progress.log"; } \
      || echo "[$(date +%m/%d-%H:%M:%S)] FAIL $tag" >> "$LOGDIR/progress.log" &
    sleep 15
  done
  wait
  log_run "BULLET CDT WAVE seed=$seed COMPLETE"
done
cd /home/omniverse/workspace/safevlmcpl/iclr2027
conda run -n safevlmcpl --no-capture-output python scripts/harvest_osrl.py >> "$LOGDIR/progress.log" 2>&1
conda run -n safevlmcpl --no-capture-output python scripts/collect_results.py >> "$LOGDIR/progress.log" 2>&1
log_run "ROUND-2 QUEUE ALL DONE"
