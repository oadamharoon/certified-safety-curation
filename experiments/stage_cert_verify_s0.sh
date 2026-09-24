#!/bin/bash
# Rescoped cert-verify: ONE learner seed per task. Reproducing the lost
# certified selections is a data question, so a single seed per task is enough
# to confirm the rebuilt h5 is faithful; three seeds only re-measures CDT's
# seed variance, which the published cdt_cert column already reports.
# HalfCheetah seed 0 was already in flight and is left running.
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
SEL=${CSC_RUNS}/selections
LOGDIR=${CSC_RUNS}/logs/cert_verify
PY=${PYTHON}
mkdir -p "$LOGDIR"; cd ${CSC_OSRL}

run_one () {
  local task=$1 env=$2 lim=$3 h5=$4 seed=$5
  local tag="${task}_s${seed}"
  [ -f "$LOGDIR/done_$tag" ] && { echo "skip $tag"; return 0; }
  env PYTHONNOUSERSITE=1 PYTHONPATH=${CSC_OSRL} \
    $PY examples/train/train_cdt.py --task "$env" --seed "$seed" \
    --cost_limit "$lim" --device cuda --augment_percent 0.0 --random_aug 0.0 \
    --subset_h5 "$h5" --logdir "$LOGDIR/runs" > "$LOGDIR/$tag.log" 2>&1 \
    && { touch "$LOGDIR/done_$tag"; echo "[$(date +%H:%M:%S)] DONE $tag"; } \
    || echo "[$(date +%H:%M:%S)] FAIL $tag"
}
export -f run_one; export LOGDIR PY

L=$LOGDIR/jobs_s0.txt; : > "$L"
echo "cargoal1_dsrl OfflineCarGoal1Gymnasium-v0 25 $SEL/cargoal1_dsrl_cert_seed3.hdf5 0" >> "$L"
echo "pointgoal1_dsrl OfflinePointGoal1Gymnasium-v0 25 $SEL/pointgoal1_dsrl_cert_seed0.hdf5 0" >> "$L"
echo "rescoped cert-verify jobs: $(wc -l < $L)"
xargs -a "$L" -L1 -P 2 bash -c 'run_one $0 $1 $2 $3 $4'
echo "CERT VERIFY (S0) COMPLETE"
