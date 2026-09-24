#!/bin/bash
# CPL composability arm: CPL trained on the SAME deployed certified selection
# CDT receives, for parity with the cdt_cert column.
#
# Parity with CDT:
#   - identical selection (the deployed draw-1 certified selection, recovered
#     and verified against the recorded n_kept by recover_kept.py)
#   - one fixed selection per task, three learner seeds (CDT's convention)
#   - CPL's label budget (1000 pairs) re-drawn WITHIN the selection, because
#     the full-pool pairs are cost-contrast sampled and only 0-3 of 1000 have
#     both endpoints inside a certified selection.
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
LOGDIR=${CSC_RUNS}/logs/cpl_compose
PY=${PYTHON}
mkdir -p "$LOGDIR"
cd "$D"

# task:pipeline-seed = the lowest certifying seed (the deployed selection)
JOBS="halfcheetah_velocity:1 cargoal1_dsrl:3 pointgoal1_dsrl:0"

for spec in $JOBS; do
  task=${spec%%:*}; vs=${spec##*:}
  for seed in 0 1 2; do
    tag="${task}_s${seed}"
    [ -f "$LOGDIR/done_$tag" ] && { echo "skip $tag (done)"; continue; }
    echo "[$(date +%H:%M:%S)] TRAIN $tag"
    env PYTHONNOUSERSITE=1 SAFETY_VLM_TASK=$task SEED_OVERRIDE=$seed \
      SUBSET_KEPT=$D/outputs/$task/kept_calfilt_csf_seed$vs.json \
      LABELS_IN=gt_labels_cert_seed$vs.json \
      POLICY_OUT=cpl_cert_seed$seed.pt \
      $PY scripts/04c_train_cpl_gt.py > "$LOGDIR/train_$tag.log" 2>&1 || {
        echo "[$(date +%H:%M:%S)] TRAIN FAIL $tag"; continue; }
    echo "[$(date +%H:%M:%S)] EVAL  $tag"
    env PYTHONNOUSERSITE=1 SAFETY_VLM_TASK=$task SEED_OVERRIDE=$seed \
      $PY scripts/05_evaluate.py --policy_file cpl_cert_seed$seed.pt \
      --results_suffix cplgt_cert_seed$seed > "$LOGDIR/eval_$tag.log" 2>&1 && {
        touch "$LOGDIR/done_$tag"; echo "[$(date +%H:%M:%S)] DONE $tag"; } || {
        echo "[$(date +%H:%M:%S)] EVAL FAIL $tag"; }
  done
done
echo "CPL COMPOSABILITY STAGE COMPLETE"
