#!/bin/bash
# CPL on the two NON-deployed alpha=0.40 grid selections per task, for the five
# tasks whose grid cells lack CPL. The deployed selection of each task already
# has CPL as cpl_gt_a40, so only 2 of 3 cells per task are run: 10 cells x 3
# learner seeds = 30 runs.
#
# Parity with the CDT and BC columns of the same table row:
#   - identical selection (rebuilt grid selection, verified set-identical to the
#     pipeline's recorded deployed selection on all five tasks)
#   - three learner seeds, CDT's convention
#   - CPL's 1000-pair label budget redrawn WITHIN the selection, exactly as
#     stage_cpl_compose.sh and stage_cpl_a40.sh do
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
SEL=${CSC_RUNS}/selections
LOGDIR=${CSC_RUNS}/logs/cpl_a40grid
PY=${PYTHON}
mkdir -p "$LOGDIR"; cd "$D"

# task|arm|selection-file   (deployed cell omitted: it is cpl_gt_a40)
CELLS="
halfcheetah_velocity|a40d3|halfcheetah_velocity_a40grid_q85_kept.json
halfcheetah_velocity|a40d2|halfcheetah_velocity_a40grid_q80_kept.json
walker2d_velocity|a40d2|walker2d_velocity_a40grid_q60_kept.json
walker2d_velocity|a40selq55|walker2d_velocity_a40grid_q55_kept.json
ant_velocity|a40selq85|ant_velocity_a40grid_q85_kept.json
ant_velocity|a40d2|ant_velocity_a40grid_q80_kept.json
swimmer_velocity|a40selq80|swimmer_velocity_a40grid_q80_kept.json
swimmer_velocity|a40selq75|swimmer_velocity_a40grid_q75_kept.json
cargoal1_dsrl|a40d3|cargoal1_dsrl_a40grid_q60_kept.json
cargoal1_dsrl|a40d2|cargoal1_dsrl_a40grid_q55_kept.json
"

run_task () {                      # all cells belonging to one task
  local want=$1
  echo "$CELLS" | while IFS='|' read -r task arm self; do
    [ -z "${task:-}" ] && continue
    [ "$task" != "$want" ] && continue
    local LBL="gt_labels_${arm}.json"
    if [ ! -f "$D/outputs/$task/$LBL" ]; then
      env PYTHONNOUSERSITE=1 SAFETY_VLM_TASK=$task SUBSET_KEPT=$SEL/$self LABELS_OUT=$LBL \
        $PY scripts/03b_label_by_cost.py > "$LOGDIR/labels_${task}_${arm}.log" 2>&1 || {
          echo "[$(date +%H:%M:%S)] LABEL FAIL $task $arm"; continue; }
    fi
    for seed in 0 1 2; do
      local tag="${task}_${arm}_s${seed}"
      [ -f "$LOGDIR/done_$tag" ] && { echo "skip $tag"; continue; }
      echo "[$(date +%H:%M:%S)] TRAIN $tag"
      env PYTHONNOUSERSITE=1 SAFETY_VLM_TASK=$task SEED_OVERRIDE=$seed \
        SUBSET_KEPT=$SEL/$self LABELS_IN=$LBL POLICY_OUT=cpl_${arm}_seed${seed}.pt \
        $PY scripts/04c_train_cpl_gt.py > "$LOGDIR/train_$tag.log" 2>&1 || {
          echo "[$(date +%H:%M:%S)] TRAIN FAIL $tag"; continue; }
      env PYTHONNOUSERSITE=1 SAFETY_VLM_TASK=$task SEED_OVERRIDE=$seed \
        $PY scripts/05_evaluate.py --policy_file cpl_${arm}_seed${seed}.pt \
        --results_suffix cplgt_${arm}_seed${seed} > "$LOGDIR/eval_$tag.log" 2>&1 && {
          touch "$LOGDIR/done_$tag"; echo "[$(date +%H:%M:%S)] DONE $tag"; } || {
          echo "[$(date +%H:%M:%S)] EVAL FAIL $tag"; }
    done
  done
}

# three tasks at a time; GPU is idle and each run is small
for t in halfcheetah_velocity walker2d_velocity ant_velocity; do
  run_task "$t" > "$LOGDIR/task_$t.log" 2>&1 &
done
wait
for t in swimmer_velocity cargoal1_dsrl; do
  run_task "$t" > "$LOGDIR/task_$t.log" 2>&1 &
done
wait
echo "CPL A40 GRID STAGE COMPLETE"
