#!/bin/bash
# Complete calfilt_csf (the deployed procedure: LTT with the Clopper-Pearson
# calsafe fallback) on the three tasks that lack it, at the same five seeds
# used everywhere else. Contract copied verbatim from stage_hcalsafe.sh.
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
W=${CSC_RUNS}
D=${CSC_WORK}
LOGDIR=$W/logs/csfcomplete
mkdir -p "$LOGDIR"
one () {
  local t=$1 seed=$2
  local tag="calfilt_csf_seed${seed}"
  local key="${t}_${tag}"
  [ -f "$LOGDIR/done_${key}" ] && return 0
  local lim; case $t in *velocity*) lim=20;; *) lim=25;; esac
  cd $D
  env SAFETY_VLM_TASK=$t WANDB_MODE=disabled OMP_NUM_THREADS=3 SEED_OVERRIDE=$seed \
      MODE=ltt CAL_N=200 ALPHA=0.25 DELTA=0.1 FALLBACK_MODE=calsafe COST_LIMIT=$lim \
      V_ENSEMBLE_FILE=v_ensemble_pess_seed${seed}.pt OUT_TAG=$tag \
    conda run -n safevlmcpl --no-capture-output python scripts/04q_calibrated_vfilter.py \
    > "$LOGDIR/${key}.log" 2>&1 \
    || { echo "[$(date +%m/%d-%H:%M:%S)] FAIL $key" >> "$LOGDIR/progress.log"; return 1; }
  env SAFETY_VLM_TASK=$t WANDB_MODE=disabled OMP_NUM_THREADS=3 CUDA_VISIBLE_DEVICES="" \
    conda run -n safevlmcpl --no-capture-output python scripts/05_evaluate.py \
    --policy_file "bc_${tag}_policy.pt" --results_suffix "$tag" >> "$LOGDIR/${key}.log" 2>&1 \
    || { echo "[$(date +%m/%d-%H:%M:%S)] FAIL EVAL $key" >> "$LOGDIR/progress.log"; return 1; }
  touch "$LOGDIR/done_${key}"
  echo "[$(date +%m/%d-%H:%M:%S)] DONE $key" >> "$LOGDIR/progress.log"
}
export -f one; export LOGDIR W D
J=$W/csf_jobs.txt; : > "$J"
for t in halfcheetah_velocity cargoal1_dsrl pointgoal1_dsrl; do
  for s in 0 1 2 3 4; do echo "$t $s" >> "$J"; done
done
echo "[$(date +%m/%d-%H:%M:%S)] CSF COMPLETE: $(wc -l < $J) jobs" >> "$LOGDIR/progress.log"
xargs -a "$J" -L1 -P 5 bash -c 'one "$@"' _
echo "[$(date +%m/%d-%H:%M:%S)] CSF COMPLETE DONE ($(ls $LOGDIR/done_* 2>/dev/null | wc -l)/15)" >> "$LOGDIR/progress.log"
