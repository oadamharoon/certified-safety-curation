#!/bin/bash
# Verify the REBUILT certified selections reproduce the published cdt_cert
# numbers. The originals were written to a session scratchpad and lost, so
# membership was recovered from the stored tau (verified against the recorded
# n_kept) and the h5 rebuilt on the raw-DSRL span route used by build_draw2.py.
# If these runs match the published cdt_cert column, the composability result
# is reproducible again. Logs to a SEPARATE dir so nothing overwrites the
# published arm during harvest.
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
SEL=${CSC_RUNS}/selections
LOGDIR=${CSC_RUNS}/logs/cert_verify
PY=${PYTHON}
mkdir -p "$LOGDIR"
cd ${CSC_OSRL}

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

JOBS=$LOGDIR/jobs.txt; : > "$JOBS"
echo "halfcheetah_velocity OfflineHalfCheetahVelocityGymnasium-v1 20 $SEL/halfcheetah_velocity_cert_seed1.hdf5" >> "$JOBS"
echo "cargoal1_dsrl OfflineCarGoal1Gymnasium-v0 25 $SEL/cargoal1_dsrl_cert_seed3.hdf5" >> "$JOBS"
echo "pointgoal1_dsrl OfflinePointGoal1Gymnasium-v0 25 $SEL/pointgoal1_dsrl_cert_seed0.hdf5" >> "$JOBS"

ALL=$LOGDIR/all.txt; : > "$ALL"
while read -r task env lim h5; do
  for s in 0 1 2; do echo "$task $env $lim $h5 $s" >> "$ALL"; done
done < "$JOBS"
echo "cert-verify jobs: $(wc -l < $ALL)"
xargs -a "$ALL" -L1 -P 2 bash -c 'run_one $0 $1 $2 $3 $4'
echo "CERT VERIFY STAGE COMPLETE"
