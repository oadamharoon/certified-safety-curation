#!/bin/bash
# B2 stage 1: render segment frames at camera 3 for the three VLM-arm tasks.
#
# Camera 3 is the third-person track view. The default (egocentric) shows floor
# and chassis with hazards out of frame, which is why every archived VLM run
# scored near chance. Renders ALL active segments, matching how the archived
# runs were produced; rendering only sampled segments risks silently reshaping
# the pair distribution, since 03_vlm_query skips pairs whose frames are absent.
#
# Prereg: preregistration_labels_split.md, B2/T3.5 (2026-08-25).
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
LOGDIR=$W/logs/b2
mkdir -p "$LOGDIR"
log_run () { echo "[$(date +%m/%d-%H:%M:%S)] $*" >> "$LOGDIR/progress.log"; }
export -f log_run
export LOGDIR W D

render_one () {
  local task=$1
  [ -f "$LOGDIR/done_render_${task}" ] && return 0
  cd "$D"
  env SAFETY_VLM_TASK=$task WANDB_MODE=disabled PYTHONNOUSERSITE=1 \
      OMP_NUM_THREADS=4 \
    conda run -n safevlmcpl --no-capture-output python scripts/02_render_frames.py \
    > "$LOGDIR/render_${task}.log" 2>&1 \
    || { log_run "FAIL RENDER $task"; return 1; }
  touch "$LOGDIR/done_render_${task}"; log_run "DONE RENDER $task"
}
export -f render_one

TASKS="cargoal1_dsrl pointgoal1_dsrl cargoal2"
log_run "B2 render start: $TASKS"
printf '%s\n' $TASKS > "$W/b2_render_jobs.txt"
xargs -a "$W/b2_render_jobs.txt" -L1 -P 3 bash -c 'render_one "$@"' _
log_run "B2 render done ($(ls $LOGDIR/done_render_* 2>/dev/null | wc -l)/3)"
