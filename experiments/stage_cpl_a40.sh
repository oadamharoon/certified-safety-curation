#!/bin/bash
# CPL on the alpha=0.40 deployed certified selection, for the eight tasks that
# certify at that level. Mirrors the alpha=0.25 arm exactly, so CPL appears in
# BOTH composability tables rather than only at the level where it helps most.
# The deployed selection is one of the three in the distinct-selection grid, so
# this is a strict subset of a full-grid run and extends without rework.
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
D=${CSC_WORK}
LOGDIR=${CSC_RUNS}/logs/cpl_a40
PY=${PYTHON}
mkdir -p "$LOGDIR"; cd "$D"

# task:deployed-pipeline-seed (lowest certifying seed of calfilt_a40)
JOBS="halfcheetah_velocity:1 walker2d_velocity:0 ant_velocity:0 swimmer_velocity:1 cargoal1_dsrl:0 cargoal2:0 pointgoal1_dsrl:0 pointgoal2:1"

for spec in $JOBS; do
  task=${spec%%:*}; vs=${spec##*:}
  KEPT=$D/outputs/$task/kept_calfilt_a40_seed$vs.json
  LBL=gt_labels_a40_seed$vs.json
  if [ ! -f "$D/outputs/$task/$LBL" ]; then
    echo "[$(date +%H:%M:%S)] LABELS $task"
    env PYTHONNOUSERSITE=1 SAFETY_VLM_TASK=$task SUBSET_KEPT=$KEPT LABELS_OUT=$LBL \
      $PY scripts/03b_label_by_cost.py > "$LOGDIR/labels_$task.log" 2>&1 || {
        echo "[$(date +%H:%M:%S)] LABEL FAIL $task"; continue; }
  fi
  for seed in 0 1 2; do
    tag="${task}_s${seed}"
    [ -f "$LOGDIR/done_$tag" ] && { echo "skip $tag"; continue; }
    echo "[$(date +%H:%M:%S)] TRAIN $tag"
    env PYTHONNOUSERSITE=1 SAFETY_VLM_TASK=$task SEED_OVERRIDE=$seed \
      SUBSET_KEPT=$KEPT LABELS_IN=$LBL POLICY_OUT=cpl_a40_seed$seed.pt \
      $PY scripts/04c_train_cpl_gt.py > "$LOGDIR/train_$tag.log" 2>&1 || {
        echo "[$(date +%H:%M:%S)] TRAIN FAIL $tag"; continue; }
    env PYTHONNOUSERSITE=1 SAFETY_VLM_TASK=$task SEED_OVERRIDE=$seed \
      $PY scripts/05_evaluate.py --policy_file cpl_a40_seed$seed.pt \
      --results_suffix cplgt_a40_seed$seed > "$LOGDIR/eval_$tag.log" 2>&1 && {
        touch "$LOGDIR/done_$tag"; echo "[$(date +%H:%M:%S)] DONE $tag"; } || {
        echo "[$(date +%H:%M:%S)] EVAL FAIL $tag"; }
  done
done
echo "CPL A40 STAGE COMPLETE"
