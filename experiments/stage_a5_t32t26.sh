#!/bin/bash
# A5: rerun the T3.2 AWR temperature/clip sweep and the T2.6 aggregators on the
# three navigation tasks whose selections were regenerated.
#   3 tasks x (5 AWR configs + 2 aggregators) x 3 seeds = 63 runs
# Prereg: iclr2027/preregistration_labels_split.md, "T3.2 + T2.6 prereg (2026-08-10)".
# Grid copied verbatim from experiments/t32t26_gpu8.sh; the only changes are the
# task list, a durable LOGDIR (the original wrote done-markers into a session
# scratchpad, so reruns could not resume), and t13_frac.py from runs/scripts.
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
LOGDIR=$W/logs/a5
mkdir -p "$LOGDIR"
log_run () { echo "[$(date +%m/%d-%H:%M:%S)] $*" >> "$LOGDIR/progress.log"; }

LIMOF () { case $1 in *velocity*) echo 20;; *) echo 25;; esac; }
export -f LIMOF

run_awr () {
  local task=$1 tag=$2 beta=$3 clip=$4 seed=$5
  local full="${tag}_seed${seed}"
  local key="${task}_${full}"
  [ -f "$LOGDIR/done_${key}" ] && return 0
  cd $D
  env SAFETY_VLM_TASK=$task WANDB_MODE=disabled OMP_NUM_THREADS=3 \
    AWR_BETA=$beta AWR_WEIGHT_CLIP=$clip SEED_OVERRIDE=$seed OUT_TAG=$full \
    ${PYTHON} scripts/04f_train_v_awr.py \
    > "$LOGDIR/${key}.log" 2>&1 || { log_run "FAIL $key"; return 1; }
  env SAFETY_VLM_TASK=$task WANDB_MODE=disabled OMP_NUM_THREADS=3 \
    ${PYTHON} scripts/05_evaluate.py \
    --policy_file "bc_v_awr_${full}_policy.pt" --results_suffix "$full" \
    >> "$LOGDIR/${key}.log" 2>&1 || { log_run "FAIL EVAL $key"; return 1; }
  touch "$LOGDIR/done_${key}"; log_run "DONE $key"
}
run_agg () {
  local task=$1 agg=$2 seed=$3
  local lim=$(LIMOF $task)
  local tag="xagg_${agg}_seed${seed}"
  local key="${task}_${tag}"
  [ -f "$LOGDIR/done_${key}" ] && return 0
  cd $D
  local frac=$(${PYTHON} $W/scripts/t13_frac.py $task $seed | tail -1)
  env SAFETY_VLM_TASK=$task WANDB_MODE=disabled OMP_NUM_THREADS=3 \
    AGG_MODE=$agg FILTER_FRAC=$frac COST_LIMIT=$lim \
    V_ENSEMBLE_FILE=v_ensemble_pess_seed${seed}.pt SEED_OVERRIDE=$seed OUT_TAG=$tag \
    ${PYTHON} scripts/04p_vfilter_bc.py \
    > "$LOGDIR/${key}.log" 2>&1 || { log_run "FAIL $key"; return 1; }
  env SAFETY_VLM_TASK=$task WANDB_MODE=disabled OMP_NUM_THREADS=3 \
    ${PYTHON} scripts/05_evaluate.py \
    --policy_file "bc_${tag}_policy.pt" --results_suffix "$tag" \
    >> "$LOGDIR/${key}.log" 2>&1 || { log_run "FAIL EVAL $key"; return 1; }
  touch "$LOGDIR/done_${key}"; log_run "DONE $key"
}
export -f run_awr run_agg
export LOGDIR W D

TASKS="cargoal2 pointgoal1_dsrl pointgoal2"
J=$W/a5_jobs.txt; : > "$J"
for t in $TASKS; do
  for s in 0 1 2; do
    echo "awr $t vawr_b01 0.1 20 $s"  >> "$J"
    echo "awr $t vawr_b03 0.3 20 $s"  >> "$J"
    echo "awr $t vawr_b3  3.0 20 $s"  >> "$J"
    echo "awr $t vawr_c5  1.0 5   $s" >> "$J"
    echo "awr $t vawr_c100 1.0 100 $s" >> "$J"
    echo "agg $t min $s" >> "$J"
    echo "agg $t p10 $s" >> "$J"
  done
done
log_run "A5 start: $(wc -l < $J) jobs"
dispatch () { local k=$1; shift; if [ "$k" = awr ]; then run_awr "$@"; else run_agg "$@"; fi; }
export -f dispatch
# log_run is called inside the xargs subshells, so it must be exported
export -f log_run
xargs -a "$J" -L1 -P 6 bash -c 'dispatch "$@"' _
log_run "A5 DONE ($(ls $LOGDIR/done_* 2>/dev/null | wc -l)/63)"
