#!/bin/bash
# Certified return-weighted cloning feasibility: 9 a25 selections x 3 seeds
# + CarRun certified selection x 3 seeds. CPU only.
# --- paths: set CSC_WORKSPACE or the individual roots; see the README ---
_csc_root () { local d; d="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  while [ "$d" != "/" ]; do [ -e "$d/.csc-root" ] && { printf %s "$d"; return; }; d="$(dirname "$d")"; done
  (cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); }
CSC_REPO="${CSC_REPO:-$(_csc_root)}"
CSC_WORKSPACE="${CSC_WORKSPACE:-$(dirname "$CSC_REPO")}"
CSC_WORK="${CSC_WORK:-$CSC_WORKSPACE/datasets}"
CSC_RUNS="${CSC_RUNS:-$CSC_WORKSPACE/runs}"
CSC_OSRL="${CSC_OSRL:-$CSC_WORKSPACE/osrl}"
CSC_PAPER="${CSC_PAPER:-$CSC_REPO/paper}"
CSC_PAPER_DATA="${CSC_PAPER_DATA:-$CSC_PAPER/data}"
PYTHON="${PYTHON:-python}"
# ------------------------------------------------------------------------

set -u
S=${TMPDIR:-/tmp}
LOGDIR=$S/wbc_logs
mkdir -p "$LOGDIR"
log_run () { echo "[$(date +%m/%d-%H:%M:%S)] $1" | tee -a "$LOGDIR/progress.log"; }

run_wbc () {
  local task=$1 kj=$2 tag_base=$3 seed=$4
  local tag="${tag_base}_seed${seed}"
  local key="${task}_${tag}"
  [ -f "$LOGDIR/done_${key}" ] && return 0
  cd ${CSC_WORK}
  env SAFETY_VLM_TASK=$task KEPT_JSON="$kj" SEED_OVERRIDE=$seed OUT_TAG="$tag" \
    RETURN_WEIGHTED=1 WANDB_MODE=disabled OMP_NUM_THREADS=4 \
    conda run -n safevlmcpl --no-capture-output python $S/bc_on_subset.py \
    > "$LOGDIR/${key}.log" 2>&1 || { echo "FAIL $key" >> "$LOGDIR/progress.log"; return 1; }
  env SAFETY_VLM_TASK=$task WANDB_MODE=disabled OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" \
    conda run -n safevlmcpl --no-capture-output python scripts/05_evaluate.py \
    --policy_file "bc_${tag}_policy.pt" --results_suffix "$tag" \
    >> "$LOGDIR/${key}.log" 2>&1 || { echo "FAIL EVAL $key" >> "$LOGDIR/progress.log"; return 1; }
  touch "$LOGDIR/done_${key}"
  echo "[$(date +%m/%d-%H:%M:%S)] DONE $key" >> "$LOGDIR/progress.log"
}
export -f run_wbc
export LOGDIR S

J=$S/wbc_jobs.txt; : > "$J"
for spec in \
  "halfcheetah_velocity a25wq85 wbcq85" "halfcheetah_velocity a25wq80 wbcq80" "halfcheetah_velocity a25wq75 wbcq75" \
  "cargoal1_dsrl a25wq85 wbcq85" "cargoal1_dsrl a25wq80 wbcq80" "cargoal1_dsrl a25wq75 wbcq75" \
  "pointgoal1_dsrl a25wq70 wbcq70" "pointgoal1_dsrl a25wq85 wbcq85" "pointgoal1_dsrl a25wq65 wbcq65"; do
  set -- $spec
  for seed in 0 1 2; do
    echo "$1 $S/certified_h5/$1_$2_kept.json $3 $seed" >> "$J"
  done
done
for seed in 0 1 2; do
  echo "carrun_b $S/certified_h5/carrun_b_echocert_seed0_kept.json wbcecho $seed" >> "$J"
done
log_run "wbc jobs: $(wc -l < $J)"
# log_run is called inside the xargs subshells, so it must be exported
export -f log_run
xargs -a "$J" -L1 -P 6 bash -c 'run_wbc "$@"' _
cd ${CSC_PAPER}
conda run -n safevlmcpl --no-capture-output python scripts/collect_results.py >> "$LOGDIR/progress.log" 2>&1
log_run "WBC QUEUE ALL DONE"
