#!/bin/bash
# A5v2: full T3.2 AWR temperature/clip sweep, rerun after the SEED_OVERRIDE fix.
# Prereg scope (preregistration_labels_split.md, T3.2 2026-08-10):
#   nine analysis tasks x 5 configs x 3 seeds = 135 runs, of which 45 already
#   exist and are reusable, so this runs the 90 missing ones (seeds 1 and 2).
#
# Why seed 0 is kept: the 45 seed-0 runs postdate every task's
# active_segments.pkl, utils/common.py has not changed since 2026-07-29, and no
# commit touched 04f in the window. The SEED_OVERRIDE patch cannot alter them
# either, since the config default is seed: 0, so the override sets the value it
# already had. Seeds 1 and 2 were bit-identical copies of seed 0 and are void.
# The previous pass is void: 04f_train_v_awr.py ignored SEED_OVERRIDE, so all
# three seeds trained bit-identical weights. Its outputs are quarantined under
# runs/logs/_degenerate_awr. Nothing is reused, including seed 0, because
# HalfCheetah showed partial seed variation that is still unexplained.
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
D=${CSC_WORK}
LOGDIR=$W/logs/a5v2
mkdir -p "$LOGDIR"
log_run () { echo "[$(date +%m/%d-%H:%M:%S)] $*" >> "$LOGDIR/progress.log"; }
export -f log_run          # was missing last time, so FAIL lines were never written
export LOGDIR W D

run_awr () {
  local task=$1 tag=$2 beta=$3 clip=$4 seed=$5
  local full="${tag}_seed${seed}"
  local key="${task}_${full}"
  [ -f "$LOGDIR/done_${key}" ] && return 0
  cd $D
  env SAFETY_VLM_TASK=$task WANDB_MODE=disabled OMP_NUM_THREADS=3 \
    AWR_BETA=$beta AWR_WEIGHT_CLIP=$clip SEED_OVERRIDE=$seed OUT_TAG=$full \
    conda run -n safevlmcpl --no-capture-output python scripts/04f_train_v_awr.py \
    > "$LOGDIR/${key}.log" 2>&1 || { log_run "FAIL TRAIN $key"; return 1; }
  env SAFETY_VLM_TASK=$task WANDB_MODE=disabled OMP_NUM_THREADS=3 \
    conda run -n safevlmcpl --no-capture-output python scripts/05_evaluate.py \
    --policy_file "bc_v_awr_${full}_policy.pt" --results_suffix "$full" \
    >> "$LOGDIR/${key}.log" 2>&1 || { log_run "FAIL EVAL $key"; return 1; }
  touch "$LOGDIR/done_${key}"; log_run "DONE $key"
}
export -f run_awr

TASKS="halfcheetah_velocity walker2d_velocity ant_velocity hopper_velocity swimmer_velocity cargoal1_dsrl cargoal2 pointgoal1_dsrl pointgoal2"
J=$W/a5v2_jobs.txt; : > "$J"
for t in $TASKS; do
  for s in 1 2; do          # seed 0 is reusable, see below
    echo "$t vawr_b01  0.1 20  $s" >> "$J"
    echo "$t vawr_b03  0.3 20  $s" >> "$J"
    echo "$t vawr_b3   3.0 20  $s" >> "$J"
    echo "$t vawr_c5   1.0 5   $s" >> "$J"
    echo "$t vawr_c100 1.0 100 $s" >> "$J"
  done
done
log_run "A5v2 start: $(wc -l < $J) jobs (seeds 1,2; seed 0 reused)"
xargs -a "$J" -L1 -P 7 bash -c 'run_awr "$@"' _
log_run "A5v2 DONE ($(ls $LOGDIR/done_* 2>/dev/null | wc -l)/90)"
