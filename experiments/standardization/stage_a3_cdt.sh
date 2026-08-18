#!/bin/bash
# A3: CDT retrained on the regenerated certified selections (composability).
set -u
W=/home/omniverse/workspace/safevlmcpl/runs
LOGDIR=$W/logs/a3cdt
mkdir -p "$LOGDIR/runs"
T2ENV () { case $1 in
  pointgoal1_dsrl) echo "OfflinePointGoal1Gymnasium-v0:25";;
  pointgoal2)      echo "OfflinePointGoal2Gymnasium-v0:25";;
  cargoal2)        echo "OfflineCarGoal2Gymnasium-v0:25";;
  carrun_b)        echo "OfflineCarRun-v0:10";; esac; }
export -f T2ENV
run_cdt () {
  local h5=$1 seed=$2
  local base=$(basename "$h5" .hdf5)
  local task=$(echo "$base" | sed -E 's/_a(25|40)_q[0-9]+$//')
  IFS=: read e lim <<< "$(T2ENV $task)"
  [ -z "${e:-}" ] && { echo "no env for $task" >> "$LOGDIR/progress.log"; return 1; }
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
export -f run_cdt; export LOGDIR W
J=$W/a3_jobs.txt; : > "$J"
for h5 in $W/selections/*.hdf5; do for s in 0 1 2; do echo "$h5 $s" >> "$J"; done; done
echo "[$(date +%m/%d-%H:%M:%S)] A3 CDT: $(wc -l < $J) jobs" >> "$LOGDIR/progress.log"
xargs -a "$J" -L1 -P 5 bash -c 'run_cdt "$@"' _
cd /home/omniverse/workspace/safevlmcpl/iclr2027
conda run -n safevlmcpl --no-capture-output python scripts/harvest_osrl.py >> "$LOGDIR/progress.log" 2>&1
echo "[$(date +%m/%d-%H:%M:%S)] A3 CDT DONE ($(ls $LOGDIR/done_* 2>/dev/null | wc -l))" >> "$LOGDIR/progress.log"
