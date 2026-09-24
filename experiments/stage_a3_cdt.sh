#!/bin/bash
# A3: CDT retrained on the regenerated certified selections (composability).
# --- paths: set CSC_WORKSPACE or the individual roots; see the README ---
_csc_root () { local d; d="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  while [ "$d" != "/" ]; do [ -e "$d/.csc-root" ] && { printf %s "$d"; return; }; d="$(dirname "$d")"; done
  (cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); }
CSC_REPO="${CSC_REPO:-$(_csc_root)}"
CSC_WORKSPACE="${CSC_WORKSPACE:-$(dirname "$CSC_REPO")}"
CSC_WORK="${CSC_WORK:-$CSC_WORKSPACE/datasets}"
CSC_RUNS="${CSC_RUNS:-$([ -d "$CSC_WORKSPACE/runs" ] && printf %s "$CSC_WORKSPACE/runs" || printf %s "$CSC_REPO/runs")}"
CSC_OSRL="${CSC_OSRL:-$CSC_WORKSPACE/osrl}"
CSC_PAPER="${CSC_PAPER:-$CSC_REPO/paper}"
CSC_PAPER_DATA="${CSC_PAPER_DATA:-$CSC_PAPER/data}"
CSC_CONFIG="${CSC_CONFIG:-$CSC_REPO/configs}"
PYTHON="${PYTHON:-python}"
# ------------------------------------------------------------------------

set -u
W=${CSC_RUNS}
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
  # Longest-prefix match against the known tasks. The previous regex stripped
  # only _a25_qNN / _a40_qNN, so every a40new_/a40grid_/echonew_ subset resolved
  # to a task name that T2ENV does not know and was silently skipped as "no env"
  # -- which is exactly the set of cells the composability grids still need.
  local task=""
  for t in pointgoal1_dsrl pointgoal2 cargoal2 carrun_b; do
    case "$base" in ${t}_*) [ ${#t} -gt ${#task} ] && task=$t;; esac
  done
  IFS=: read e lim <<< "$(T2ENV $task)"
  [ -z "${e:-}" ] && { echo "no env for $task" >> "$LOGDIR/progress.log"; return 1; }
  local tag="cdt_${base}_s${seed}"
  [ -f "$LOGDIR/done_${tag}" ] && return 0
  cd ${CSC_OSRL}
  env PYTHONNOUSERSITE=1 PYTHONPATH=${CSC_OSRL} \
    ${PYTHON} \
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
cd ${CSC_PAPER}
${PYTHON} scripts/harvest_osrl.py >> "$LOGDIR/progress.log" 2>&1
echo "[$(date +%m/%d-%H:%M:%S)] A3 CDT DONE ($(ls $LOGDIR/done_* 2>/dev/null | wc -l))" >> "$LOGDIR/progress.log"
