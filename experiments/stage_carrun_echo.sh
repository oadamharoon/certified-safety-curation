#!/bin/bash
# Regenerate the CarRun echo arms. Their certified selection lived only in a
# session scratchpad (carrun_b_echocert_seed0.hdf5) and matches no current
# (V seed, quantile): the paper's selection is ahat=0.138 at half the pool,
# while the current pipeline's alpha=0.25 certification yields q85 (ahat=0.051,
# 15% of pool) and q80 (ahat=0.221). Rebuilt as carrun_b_echonew_q85.
#
# Covers BC and the four operator variants; CDT echo runs with the CDT block.
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
LOGDIR=${CSC_RUNS}/logs/carrun_echo
PY=${PYTHON}
KJ="$SEL/carrun_b_echonew_q85_kept.json"
mkdir -p "$LOGDIR"; cd "$D"
[ -f "$KJ" ] || { echo "missing $KJ"; exit 1; }

run_arm () {
  local tag=$1 mode=$2 clip=$3 seed=$4
  local full="${tag}_seed${seed}"; local key="carrun_b_${full}"
  [ -f "$LOGDIR/done_$key" ] && { echo "skip $key"; return 0; }
  local EXTRA_A="" EXTRA_B=""
  if [ "$mode" = w ]; then EXTRA_A="RETURN_WEIGHTED=1"; EXTRA_B="WEIGHT_CLIP=$clip";
  elif [ "$mode" = t ]; then EXTRA_A="RETURN_TOPHALF=1"; fi
  env PYTHONNOUSERSITE=1 SAFETY_VLM_TASK=carrun_b KEPT_JSON="$KJ" \
    SEED_OVERRIDE=$seed OUT_TAG="$full" $EXTRA_A $EXTRA_B \
    $PY scripts/04s_bc_on_subset.py > "$LOGDIR/$key.log" 2>&1 || {
      echo "[$(date +%H:%M:%S)] FAIL $key"; return 1; }
  env PYTHONNOUSERSITE=1 SAFETY_VLM_TASK=carrun_b SEED_OVERRIDE=$seed \
    $PY scripts/05_evaluate.py --policy_file "bc_${full}_policy.pt" \
    --results_suffix "$full" >> "$LOGDIR/$key.log" 2>&1 && {
      touch "$LOGDIR/done_$key"; echo "[$(date +%H:%M:%S)] DONE $key"; } || {
      echo "[$(date +%H:%M:%S)] EVAL FAIL $key"; }
}
export -f run_arm; export LOGDIR PY D KJ SEL

J=$LOGDIR/jobs.txt; : > "$J"
for seed in 0 1 2; do
  echo "bcechonew b - $seed"   >> "$J"   # plain clone of the selection
  echo "wbc1echonew w 1.0 $seed" >> "$J" # return-weighted, kappa 1
  echo "wbcechonew  w 2.0 $seed" >> "$J" # kappa 2 (the reported setting)
  echo "wbc3echonew w 3.0 $seed" >> "$J" # kappa 3
  echo "tophechonew t - $seed"  >> "$J"  # reward-aware top half
done
echo "carrun echo jobs: $(wc -l < $J)"
xargs -a "$J" -L1 -P 3 bash -c 'run_arm $0 $1 $2 $3'
echo "CARRUN ECHO STAGE COMPLETE"
