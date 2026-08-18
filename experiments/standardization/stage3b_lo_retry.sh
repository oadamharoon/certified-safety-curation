#!/bin/bash
# Retry pass 2: labels-only arms that failed for missing FILTER_FRAC.
set -u
W=/home/omniverse/workspace/safevlmcpl/runs
D=/home/omniverse/workspace/safevlmcpl/vlm-with-cpl/new_data
LOGDIR=$W/logs/stage3b
while pgrep -f "scripts/stage3b.sh" > /dev/null || pgrep -f "stage3b_retbot.sh" > /dev/null; do sleep 300; done
LIMOF () { case $1 in *velocity*) echo 20;; *_b) echo 10;; *) echo 25;; esac; }
GTF () { conda run -n safevlmcpl --no-capture-output python -c "import json;print(json.load(open('$W/gtfracs.json'))['$1']['gt_frac'])"; }
export -f LIMOF GTF
one () {
  local task=$1 kind=$2 seed=$3 lim=$(LIMOF $1) tag extra
  case $kind in
    lo)     tag="labels_only_seed${seed}";          extra="CAL_N=200";;
    lo50)   tag="labels_only_n50_seed${seed}";      extra="CAL_N=50";;
    lo100)  tag="labels_only_n100_seed${seed}";     extra="CAL_N=100";;
    lo400)  tag="labels_only_n400_seed${seed}";     extra="CAL_N=400";;
    losf1)  tag="labels_only_sf1_seed${seed}";      extra="CAL_N=200 STATE_FRAC=0.01";;
    losf10) tag="labels_only_sf10_seed${seed}";     extra="CAL_N=200 STATE_FRAC=0.1";;
    lofix)  tag="labels_only_fixdraw_seed${seed}";  extra="CAL_N=200 LABEL_DRAW_SEED=0";;
  esac
  local key="${task}_${tag}"
  [ -f "$LOGDIR/done_${key}" ] && return 0
  cd $D
  env SAFETY_VLM_TASK=$task WANDB_MODE=disabled OMP_NUM_THREADS=3 SEED_OVERRIDE=$seed \
      COST_LIMIT=$lim FILTER_FRAC=$(GTF $task) OUT_TAG=$tag $extra \
    conda run -n safevlmcpl --no-capture-output python scripts/04s_labels_only_filter.py \
    > "$LOGDIR/${key}.log" 2>&1 || { echo "[$(date +%m/%d-%H:%M:%S)] FAIL3 $key" >> "$LOGDIR/progress.log"; return 1; }
  env SAFETY_VLM_TASK=$task WANDB_MODE=disabled OMP_NUM_THREADS=3 CUDA_VISIBLE_DEVICES="" \
    conda run -n safevlmcpl --no-capture-output python scripts/05_evaluate.py \
    --policy_file "bc_${tag}_policy.pt" --results_suffix "$tag" >> "$LOGDIR/${key}.log" 2>&1 \
    || { echo "[$(date +%m/%d-%H:%M:%S)] FAIL3 EVAL $key" >> "$LOGDIR/progress.log"; return 1; }
  touch "$LOGDIR/done_${key}"; echo "[$(date +%m/%d-%H:%M:%S)] DONE $key" >> "$LOGDIR/progress.log"
}
export -f one; export LOGDIR W D
J=$W/lo_retry_jobs.txt; : > "$J"
for t in cargoal2 pointgoal1_dsrl pointgoal2; do for s in 0 1 2; do
  for k in lo lo50 lo100 lo400 losf1 losf10 lofix; do echo "$t $k $s" >> "$J"; done; done; done
echo "[$(date +%m/%d-%H:%M:%S)] LO retry: $(wc -l < $J) jobs" >> "$LOGDIR/progress.log"
xargs -a "$J" -L1 -P 8 bash -c 'one "$@"' _
cd /home/omniverse/workspace/safevlmcpl/iclr2027
conda run -n safevlmcpl --no-capture-output python scripts/collect_results.py >> "$LOGDIR/progress.log" 2>&1
echo "[$(date +%m/%d-%H:%M:%S)] LO RETRY DONE" >> "$LOGDIR/progress.log"
