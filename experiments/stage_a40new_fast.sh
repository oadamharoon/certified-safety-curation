#!/bin/bash
# Regenerated alpha=0.40 rows for cargoal2 / pointgoal1 / pointgoal2, whose
# published grid selections came from V ensembles overwritten on 2026-08-16.
# BC and CPL first (minutes to tens of minutes); CDT is a separate stage run
# last because it is ~4.6h per run.
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
LOGDIR=${CSC_RUNS}/logs/a40new_fast
PY=${PYTHON}
mkdir -p "$LOGDIR"; cd "$D"

SELS="cargoal2:75 cargoal2:70 cargoal2:65 pointgoal1_dsrl:40 pointgoal1_dsrl:35 pointgoal1_dsrl:30 pointgoal2:85 pointgoal2:80 pointgoal2:75"

# ---- stage 1: BC on each selection (fastest) ----
run_bc () {
  local task=$1 q=$2 seed=$3
  local tag="a40new_q${q}"; local key="${task}_${tag}_s${seed}"
  [ -f "$LOGDIR/done_bc_$key" ] && { echo "skip bc $key"; return 0; }
  env PYTHONNOUSERSITE=1 SAFETY_VLM_TASK=$task SEED_OVERRIDE=$seed \
    KEPT_JSON="$SEL/${task}_${tag}_kept.json" OUT_TAG="${tag}_s${seed}" \
    $PY scripts/04s_bc_on_subset.py > "$LOGDIR/bc_$key.log" 2>&1 || {
      echo "[$(date +%H:%M:%S)] BC FAIL $key"; return 1; }
  env PYTHONNOUSERSITE=1 SAFETY_VLM_TASK=$task SEED_OVERRIDE=$seed \
    $PY scripts/05_evaluate.py --policy_file "bc_${tag}_s${seed}_policy.pt" \
    --results_suffix "bc${tag}_seed${seed}" >> "$LOGDIR/bc_$key.log" 2>&1 && {
      touch "$LOGDIR/done_bc_$key"; echo "[$(date +%H:%M:%S)] DONE bc $key"; } || {
      echo "[$(date +%H:%M:%S)] BC EVAL FAIL $key"; }
}
# ---- stage 2: CPL on each selection, same protocol as the other CPL arms ----
run_cpl () {
  local task=$1 q=$2 seed=$3
  local tag="a40new_q${q}"; local key="${task}_${tag}_s${seed}"
  [ -f "$LOGDIR/done_cpl_$key" ] && { echo "skip cpl $key"; return 0; }
  local LBL="gt_labels_${tag}.json"
  if [ ! -f "$D/outputs/$task/$LBL" ]; then
    env PYTHONNOUSERSITE=1 SAFETY_VLM_TASK=$task \
      SUBSET_KEPT="$SEL/${task}_${tag}_kept.json" LABELS_OUT="$LBL" \
      $PY scripts/03b_label_by_cost.py > "$LOGDIR/labels_${task}_${tag}.log" 2>&1 || {
        echo "[$(date +%H:%M:%S)] LABEL FAIL $key"; return 1; }
  fi
  env PYTHONNOUSERSITE=1 SAFETY_VLM_TASK=$task SEED_OVERRIDE=$seed \
    SUBSET_KEPT="$SEL/${task}_${tag}_kept.json" LABELS_IN="$LBL" \
    POLICY_OUT="cpl_${tag}_seed${seed}.pt" \
    $PY scripts/04c_train_cpl_gt.py > "$LOGDIR/cpl_$key.log" 2>&1 || {
      echo "[$(date +%H:%M:%S)] CPL FAIL $key"; return 1; }
  env PYTHONNOUSERSITE=1 SAFETY_VLM_TASK=$task SEED_OVERRIDE=$seed \
    $PY scripts/05_evaluate.py --policy_file "cpl_${tag}_seed${seed}.pt" \
    --results_suffix "cplgt${tag}_seed${seed}" >> "$LOGDIR/cpl_$key.log" 2>&1 && {
      touch "$LOGDIR/done_cpl_$key"; echo "[$(date +%H:%M:%S)] DONE cpl $key"; } || {
      echo "[$(date +%H:%M:%S)] CPL EVAL FAIL $key"; }
}
export -f run_bc run_cpl; export LOGDIR PY D SEL

BJ=$LOGDIR/bc_jobs.txt; : > "$BJ"
CJ=$LOGDIR/cpl_jobs.txt; : > "$CJ"
for s in $SELS; do
  t=${s%%:*}; q=${s##*:}
  for seed in 0 1 2; do echo "$t $q $seed" >> "$BJ"; echo "$t $q $seed" >> "$CJ"; done
done
echo "BC jobs: $(wc -l < $BJ) | CPL jobs: $(wc -l < $CJ)"
xargs -a "$BJ" -L1 -P 4 bash -c 'run_bc $0 $1 $2'
echo "BC STAGE COMPLETE"
xargs -a "$CJ" -L1 -P 4 bash -c 'run_cpl $0 $1 $2'
echo "A40NEW FAST STAGES COMPLETE"
