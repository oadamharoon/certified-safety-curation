#!/bin/bash
# Extend the CPL composability arm from the deployed selection to all NINE
# alpha=0.25 certified selections, matching CDT's coverage (Table tab:a25draws).
# The deployed selection (q85) is covered by stage_cpl_compose.sh; this adds the
# six remaining distinct selections. Same protocol throughout: one fixed
# selection, three learner seeds, 1000-pair label budget re-drawn inside the
# selection.
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
SEL=${CSC_RUNS}/selections
LOGDIR=${CSC_RUNS}/logs/cpl_nine
PY=${PYTHON}
mkdir -p "$LOGDIR"; cd "$D"

JOBS="halfcheetah_velocity:80 halfcheetah_velocity:75 cargoal1_dsrl:80 cargoal1_dsrl:75 pointgoal1_dsrl:80 pointgoal1_dsrl:55"

for spec in $JOBS; do
  task=${spec%%:*}; q=${spec##*:}
  for seed in 0 1 2; do
    tag="${task}_q${q}_s${seed}"
    [ -f "$LOGDIR/done_$tag" ] && { echo "skip $tag"; continue; }
    echo "[$(date +%H:%M:%S)] TRAIN $tag"
    env PYTHONNOUSERSITE=1 SAFETY_VLM_TASK=$task SEED_OVERRIDE=$seed \
      SUBSET_KEPT=$SEL/${task}_a25_q${q}_kept.json \
      LABELS_IN=gt_labels_a25q${q}.json \
      POLICY_OUT=cpl_a25q${q}_seed$seed.pt \
      $PY scripts/04c_train_cpl_gt.py > "$LOGDIR/train_$tag.log" 2>&1 || {
        echo "[$(date +%H:%M:%S)] TRAIN FAIL $tag"; continue; }
    echo "[$(date +%H:%M:%S)] EVAL  $tag"
    env PYTHONNOUSERSITE=1 SAFETY_VLM_TASK=$task SEED_OVERRIDE=$seed \
      $PY scripts/05_evaluate.py --policy_file cpl_a25q${q}_seed$seed.pt \
      --results_suffix cplgt_a25q${q}_seed$seed > "$LOGDIR/eval_$tag.log" 2>&1 && {
        touch "$LOGDIR/done_$tag"; echo "[$(date +%H:%M:%S)] DONE $tag"; } || {
        echo "[$(date +%H:%M:%S)] EVAL FAIL $tag"; }
  done
done
echo "CPL NINE-SELECTION STAGE COMPLETE"
