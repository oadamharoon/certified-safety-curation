#!/bin/bash
# Operator completion sets: clip sensitivity (+/-1, +/-3) + hard top-half,
# on the identical 10 selections x 3 seeds. GPU training, CPU evals.
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
LOGDIR=$S/wbc2_logs
mkdir -p "$LOGDIR"
log_run () { echo "[$(date +%m/%d-%H:%M:%S)] $1" | tee -a "$LOGDIR/progress.log"; }

run_arm () {
  local task=$1 kj=$2 tag=$3 mode=$4 clip=$5 seed=$6
  local full="${tag}_seed${seed}"
  local key="${task}_${full}"
  [ -f "$LOGDIR/done_${key}" ] && return 0
  cd ${CSC_WORK}
  local EXTRA=""
  if [ "$mode" = w ]; then EXTRA="RETURN_WEIGHTED=1 WEIGHT_CLIP=$clip"; else EXTRA="RETURN_TOPHALF=1"; fi
  env SAFETY_VLM_TASK=$task KEPT_JSON="$kj" SEED_OVERRIDE=$seed OUT_TAG="$full" $EXTRA \
    WANDB_MODE=disabled OMP_NUM_THREADS=4 \
    ${PYTHON} $S/bc_on_subset.py \
    > "$LOGDIR/${key}.log" 2>&1 || { echo "FAIL $key" >> "$LOGDIR/progress.log"; return 1; }
  env SAFETY_VLM_TASK=$task WANDB_MODE=disabled OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" \
    ${PYTHON} scripts/05_evaluate.py \
    --policy_file "bc_${full}_policy.pt" --results_suffix "$full" \
    >> "$LOGDIR/${key}.log" 2>&1 || { echo "FAIL EVAL $key" >> "$LOGDIR/progress.log"; return 1; }
  touch "$LOGDIR/done_${key}"
  echo "[$(date +%m/%d-%H:%M:%S)] DONE $key" >> "$LOGDIR/progress.log"
}
export -f run_arm
export LOGDIR S

J=$S/wbc2_jobs.txt; : > "$J"
SELS="halfcheetah_velocity:a25wq85:q85 halfcheetah_velocity:a25wq80:q80 halfcheetah_velocity:a25wq75:q75 \
cargoal1_dsrl:a25wq85:q85 cargoal1_dsrl:a25wq80:q80 cargoal1_dsrl:a25wq75:q75 \
pointgoal1_dsrl:a25wq70:q70 pointgoal1_dsrl:a25wq85:q85 pointgoal1_dsrl:a25wq65:q65"
for spec in $SELS; do
  IFS=: read task sel q <<< "$spec"
  kj="$S/certified_h5/${task}_${sel}_kept.json"
  for seed in 0 1 2; do
    echo "$task $kj wc1${q} w 1.0 $seed" >> "$J"
    echo "$task $kj wc3${q} w 3.0 $seed" >> "$J"
    echo "$task $kj th${q} t - $seed" >> "$J"
  done
done
KJE="$S/certified_h5/carrun_b_echocert_seed0_kept.json"
for seed in 0 1 2; do
  echo "carrun_b $KJE wc1echo w 1.0 $seed" >> "$J"
  echo "carrun_b $KJE wc3echo w 3.0 $seed" >> "$J"
  echo "carrun_b $KJE thecho t - $seed" >> "$J"
done
log_run "wbc2 jobs: $(wc -l < $J)"
# log_run is called inside the xargs subshells, so it must be exported
export -f log_run
xargs -a "$J" -L1 -P 6 bash -c 'run_arm "$@"' _
cd ${CSC_PAPER}
${PYTHON} scripts/collect_results.py >> "$LOGDIR/progress.log" 2>&1
log_run "WBC2 QUEUE ALL DONE"
