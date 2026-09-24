#!/bin/bash
# H-sweep arms under the DEPLOYED procedure (calibration-estimated fallback),
# so the segment-length comparison matches the method the paper ships.
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
LOGDIR=$W/logs/hcalsafe
mkdir -p "$LOGDIR"
one () {
  local t=$1 h=$2 seed=$3
  local tag="calfilt_hcsf_seed${seed}"
  local key="${t}_h${h}_${tag}"
  [ -f "$LOGDIR/done_${key}" ] && return 0
  local lim; case $t in *velocity*) lim=20;; *) lim=25;; esac
  cd $D
  env SAFETY_VLM_TASK=$t SAFETY_VLM_CONFIG=config_h${h}.yaml WANDB_MODE=disabled \
      OMP_NUM_THREADS=3 SEED_OVERRIDE=$seed MODE=ltt CAL_N=200 ALPHA=0.25 DELTA=0.1 \
      FALLBACK_MODE=calsafe COST_LIMIT=$lim \
      V_ENSEMBLE_FILE=v_ensemble_pess_seed${seed}.pt OUT_TAG=$tag \
    conda run -n safevlmcpl --no-capture-output python scripts/04q_calibrated_vfilter.py \
    > "$LOGDIR/${key}.log" 2>&1 || { echo "[$(date +%m/%d-%H:%M:%S)] FAIL $key" >> "$LOGDIR/progress.log"; return 1; }
  env SAFETY_VLM_TASK=$t SAFETY_VLM_CONFIG=config_h${h}.yaml WANDB_MODE=disabled \
      OMP_NUM_THREADS=3 CUDA_VISIBLE_DEVICES="" \
    conda run -n safevlmcpl --no-capture-output python scripts/05_evaluate.py \
    --policy_file "bc_${tag}_policy.pt" --results_suffix "$tag" >> "$LOGDIR/${key}.log" 2>&1 \
    || { echo "[$(date +%m/%d-%H:%M:%S)] FAIL EVAL $key" >> "$LOGDIR/progress.log"; return 1; }
  touch "$LOGDIR/done_${key}"
  echo "[$(date +%m/%d-%H:%M:%S)] DONE $key" >> "$LOGDIR/progress.log"
}
export -f one; export LOGDIR W D
J=$W/hcalsafe_jobs.txt; : > "$J"
for t in halfcheetah_velocity walker2d_velocity ant_velocity hopper_velocity swimmer_velocity cargoal1_dsrl cargoal2 pointgoal1_dsrl pointgoal2; do
  for h in 10 50; do for s in 0 1 2; do echo "$t $h $s" >> "$J"; done; done
done
echo "[$(date +%m/%d-%H:%M:%S)] HCALSAFE: $(wc -l < $J) jobs" >> "$LOGDIR/progress.log"
xargs -a "$J" -L1 -P 8 bash -c 'one "$@"' _
cd ${CSC_PAPER}
conda run -n safevlmcpl --no-capture-output python scripts/collect_results.py >> "$LOGDIR/progress.log" 2>&1
echo "[$(date +%m/%d-%H:%M:%S)] HCALSAFE DONE ($(ls $LOGDIR/done_* 2>/dev/null | wc -l))" >> "$LOGDIR/progress.log"
