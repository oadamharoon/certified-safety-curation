#!/bin/bash
# Retry pass for bottom-return control arms (SCORE_MODE name was wrong).
set -u
W=/home/omniverse/workspace/safevlmcpl/runs
D=/home/omniverse/workspace/safevlmcpl/vlm-with-cpl/new_data
LOGDIR=$W/logs/stage3b
while pgrep -f "scripts/stage3b.sh" > /dev/null; do sleep 300; done
LIMOF () { case $1 in *velocity*) echo 20;; *_b) echo 10;; *) echo 25;; esac; }
GTF () { conda run -n safevlmcpl --no-capture-output python -c "import json;print(json.load(open('$W/gtfracs.json'))['$1']['gt_frac'])"; }
export -f LIMOF GTF
one () {
  local task=$1 seed=$2 lim=$(LIMOF $1) tag="vfilt_retbot_seed${2}"
  local key="${task}_${tag}"
  [ -f "$LOGDIR/done_${key}" ] && return 0
  cd $D
  env SAFETY_VLM_TASK=$task WANDB_MODE=disabled OMP_NUM_THREADS=3 SEED_OVERRIDE=$seed \
      SCORE_MODE=return_bottom FILTER_FRAC=$(GTF $task) COST_LIMIT=$lim \
      V_ENSEMBLE_FILE=v_ensemble_pess_seed${seed}.pt OUT_TAG=$tag \
    conda run -n safevlmcpl --no-capture-output python scripts/04p_vfilter_bc.py \
    > "$LOGDIR/${key}.log" 2>&1 || { echo "[$(date +%m/%d-%H:%M:%S)] FAIL2 $key" >> "$LOGDIR/progress.log"; return 1; }
  env SAFETY_VLM_TASK=$task WANDB_MODE=disabled OMP_NUM_THREADS=3 CUDA_VISIBLE_DEVICES="" \
    conda run -n safevlmcpl --no-capture-output python scripts/05_evaluate.py \
    --policy_file "bc_${tag}_policy.pt" --results_suffix "$tag" >> "$LOGDIR/${key}.log" 2>&1 \
    || { echo "[$(date +%m/%d-%H:%M:%S)] FAIL2 EVAL $key" >> "$LOGDIR/progress.log"; return 1; }
  touch "$LOGDIR/done_${key}"; echo "[$(date +%m/%d-%H:%M:%S)] DONE $key" >> "$LOGDIR/progress.log"
}
export -f one; export LOGDIR W D
J=$W/retbot_jobs.txt; : > "$J"
for t in pointgoal2 pointbutton1 pointcircle1 pointcircle2 ballrun_b ballcircle_b carcircle_b carrun_b dronerun_b; do
  for s in 0 1 2 3 4; do echo "$t $s" >> "$J"; done; done
echo "[$(date +%m/%d-%H:%M:%S)] RETBOT retry: $(wc -l < $J) jobs" >> "$LOGDIR/progress.log"
xargs -a "$J" -L1 -P 8 bash -c 'one "$@"' _
cd /home/omniverse/workspace/safevlmcpl/iclr2027
conda run -n safevlmcpl --no-capture-output python scripts/collect_results.py >> "$LOGDIR/progress.log" 2>&1
echo "[$(date +%m/%d-%H:%M:%S)] RETBOT RETRY DONE" >> "$LOGDIR/progress.log"
