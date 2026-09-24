#!/bin/bash
# Re-run the CPL baseline on the 7 tasks that consumed 3000 preference pairs,
# using the current 1000-pair gt_labels.json and the current batch_size (512).
#
# Table tab:baselines calls this arm "CPL on our exact preference data". Our own
# pipeline consumes 1000 pairs per task, so on these 7 tasks the claim was not
# true and CPL sat in the wrong place on the supervision axis of fig:pareto.
# Four of them additionally ran at batch 64, an artifact of the global default
# at the time rather than a deliberate per-task choice.
# Superseded results archived under runs/archive/cpl_gt_3000pair_2026-07-19.
# Policy filename carries the seed: 04c's default name is shared, and running
# seeds concurrently made both evals read whichever policy saved last.
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
D=${CSC_WORK}
LOGDIR=${CSC_RUNS}/logs/cpl_rerun1000
PY=${PYTHON}
mkdir -p "$LOGDIR"; cd "$D"

TASKS="carbutton1_t3 carbutton2 pointbutton1 pointbutton2 pointcircle1 pointcircle2 pointgoal2"

run_one () {
  local task=$1 seed=$2
  local tag="${task}_s${seed}"
  [ -f "$LOGDIR/done_$tag" ] && { echo "skip $tag"; return 0; }
  env PYTHONNOUSERSITE=1 SAFETY_VLM_TASK=$task SEED_OVERRIDE=$seed \
    POLICY_OUT=bc_gt_cpl_policy_seed$seed.pt \
    $PY scripts/04c_train_cpl_gt.py > "$LOGDIR/train_$tag.log" 2>&1 || {
      echo "[$(date +%H:%M:%S)] TRAIN FAIL $tag"; return 1; }
  env PYTHONNOUSERSITE=1 SAFETY_VLM_TASK=$task SEED_OVERRIDE=$seed \
    $PY scripts/05_evaluate.py --policy_file bc_gt_cpl_policy_seed$seed.pt \
    --results_suffix cplgt_seed$seed > "$LOGDIR/eval_$tag.log" 2>&1 && {
      touch "$LOGDIR/done_$tag"; echo "[$(date +%H:%M:%S)] DONE $tag"; } || {
      echo "[$(date +%H:%M:%S)] EVAL FAIL $tag"; }
}
export -f run_one; export LOGDIR PY D

L=$LOGDIR/jobs.txt; : > "$L"
for t in $TASKS; do for s in 0 1 2; do echo "$t $s" >> "$L"; done; done
echo "cpl re-run jobs: $(wc -l < $L)"
xargs -a "$L" -L1 -P 2 bash -c 'run_one $0 $1'
echo "CPL RERUN (1000 pairs / batch 512) COMPLETE"
