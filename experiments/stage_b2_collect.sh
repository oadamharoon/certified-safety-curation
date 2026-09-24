#!/bin/bash
# B2 (corrected): distil a DSRL BC policy, then collect fresh trajectories whose
# recorded seed IS the reset seed.
#
# Why: the DSRL pickles store a placeholder seed, so replaying the logged actions
# lands in a different layout and every rendered clip shows hazards that did not
# produce the logged cost. See preregistration_labels_split.md, B2 addendum.
# --- paths: set CSC_WORKSPACE or the individual roots; see the README ---
_csc_root () { local d; d="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  while [ "$d" != "/" ]; do [ -e "$d/.csc-root" ] && { printf %s "$d"; return; }; d="$(dirname "$d")"; done
  (cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); }
CSC_REPO="${CSC_REPO:-$(_csc_root)}"
CSC_WORKSPACE="${CSC_WORKSPACE:-$(dirname "$CSC_REPO")}"
CSC_WORK="${CSC_WORK:-$CSC_WORKSPACE/vlm-with-cpl/new_data}"
CSC_RUNS="${CSC_RUNS:-$([ -d "$CSC_WORKSPACE/runs" ] && printf %s "$CSC_WORKSPACE/runs" || printf %s "$CSC_REPO/runs")}"
CSC_OSRL="${CSC_OSRL:-$CSC_WORKSPACE/osrl}"
CSC_PAPER="${CSC_PAPER:-$CSC_REPO/paper}"
CSC_PAPER_DATA="${CSC_PAPER_DATA:-$CSC_PAPER/data}"
CSC_CONFIG="${CSC_CONFIG:-$CSC_REPO/configs}"
PYTHON="${PYTHON:-python}"
# ------------------------------------------------------------------------

set -u
W=${CSC_RUNS}
D=${CSC_WORK}
LOGDIR=$W/logs/b2c
mkdir -p "$LOGDIR"
log_run () { echo "[$(date +%m/%d-%H:%M:%S)] $*" >> "$LOGDIR/progress.log"; }
export -f log_run
export LOGDIR W D

bc_one () {
  local task=$1
  [ -f "$LOGDIR/done_bc_${task}" ] && return 0
  cd "$D"
  env SAFETY_VLM_TASK=$task WANDB_MODE=disabled PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 \
    conda run -n safevlmcpl --no-capture-output \
    python scripts/00a_train_bc_from_dsrl.py --epochs 50 \
    > "$LOGDIR/bc_${task}.log" 2>&1 || { log_run "FAIL BC $task"; return 1; }
  touch "$LOGDIR/done_bc_${task}"; log_run "DONE BC $task"
}
export -f bc_one

TASKS="cargoal1_dsrl pointgoal1_dsrl"
log_run "B2 BC distillation start: $TASKS"
printf '%s\n' $TASKS > "$W/b2c_bc_jobs.txt"
xargs -a "$W/b2c_bc_jobs.txt" -L1 -P 2 bash -c 'bc_one "$@"' _
log_run "B2 BC done ($(ls $LOGDIR/done_bc_* 2>/dev/null | wc -l)/2)"

# --- stage 2: collect fresh trajectories with real, reproducible seeds --------
collect_one () {
  local task=$1
  [ -f "$LOGDIR/done_collect_${task}" ] && return 0
  cd "$D"
  env SAFETY_VLM_TASK=$task WANDB_MODE=disabled PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 \
    conda run -n safevlmcpl --no-capture-output python scripts/00_collect_online_data.py \
    > "$LOGDIR/collect_${task}.log" 2>&1 || { log_run "FAIL COLLECT $task"; return 1; }
  touch "$LOGDIR/done_collect_${task}"; log_run "DONE COLLECT $task"
}
export -f collect_one
log_run "B2 collection start: $TASKS"
xargs -a "$W/b2c_bc_jobs.txt" -L1 -P 2 bash -c 'collect_one "$@"' _
log_run "B2 collection done ($(ls $LOGDIR/done_collect_* 2>/dev/null | wc -l)/2)"
