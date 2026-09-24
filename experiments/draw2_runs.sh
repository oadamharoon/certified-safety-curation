#!/bin/bash
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
LOGDIR=$S/draw2_logs
mkdir -p "$LOGDIR"
cd ${CSC_OSRL}
log_run () { echo "[$(date +%m/%d-%H:%M:%S)] $1" | tee -a "$LOGDIR/progress.log"; }
JOBS=(
  "OfflineHalfCheetahVelocityGymnasium-v1:20:halfcheetah_velocity_draw2_seed1.hdf5"
  "OfflineCarGoal1Gymnasium-v0:25:cargoal1_dsrl_draw2_seed3.hdf5"
  "OfflinePointGoal1Gymnasium-v0:25:pointgoal1_dsrl_draw2_seed0.hdf5"
)
for seed in 0 1 2; do
  while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)" -lt 10000 ]; do sleep 300; done
  log_run "WAVE seed=$seed START"
  for job in "${JOBS[@]}"; do
    IFS=: read e lim h5 <<< "$job"
    tag="cdtdraw2_${e}_s${seed}"
    [ -f "$LOGDIR/done_${tag}" ] && continue
    log_run "START $tag"
    env PYTHONNOUSERSITE=1 PYTHONPATH=${CSC_OSRL} \
      conda run -n safevlmcpl --no-capture-output \
      python examples/train/train_cdt.py --task "$e" --seed "$seed" \
      --cost_limit "$lim" --device cuda --augment_percent 0.0 --random_aug 0.0 \
      --subset_h5 "$S/certified_h5/$h5" \
      --logdir "$LOGDIR/runs" > "$LOGDIR/${tag}.log" 2>&1 \
      && touch "$LOGDIR/done_${tag}" && log_run "DONE $tag" || log_run "FAIL $tag" &
    sleep 15
  done
  wait
  log_run "WAVE seed=$seed COMPLETE"
done
log_run "DRAW2 ALL DONE"
