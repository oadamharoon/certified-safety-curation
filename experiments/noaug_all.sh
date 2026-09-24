#!/bin/bash
# No-aug full-data CDT control: 5 remaining a40 tasks x 3 seeds, GPU.
# Waves of 5 (one trainer per task); seeds sequential across waves.
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
LOGDIR=$S/noaug_all_logs
mkdir -p "$LOGDIR"
cd ${CSC_OSRL}
log_run () { echo "[$(date +%m/%d-%H:%M:%S)] $1" | tee -a "$LOGDIR/progress.log"; }

TASKS=(
  "OfflineWalker2dVelocityGymnasium-v1:20"
  "OfflineAntVelocityGymnasium-v1:20"
  "OfflineSwimmerVelocityGymnasium-v1:20"
  "OfflineCarGoal2Gymnasium-v0:25"
  "OfflinePointGoal2Gymnasium-v0:25"
)

for seed in 0 1 2; do
  # wait for enough free GPU memory before each wave (5 trainers ~= 15-18 GB)
  while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)" -lt 16000 ]; do
    sleep 300
  done
  log_run "WAVE seed=$seed START"
  for env in "${TASKS[@]}"; do
    IFS=: read e lim <<< "$env"
    tag="cdtnoaug_${e}_s${seed}"
    [ -f "$LOGDIR/done_${tag}" ] && { log_run "SKIP $tag (done)"; continue; }
    log_run "START $tag"
    env PYTHONNOUSERSITE=1 PYTHONPATH=${CSC_OSRL} \
      conda run -n safevlmcpl --no-capture-output \
      python examples/train/train_cdt.py --task "$e" --seed "$seed" \
      --cost_limit "$lim" --device cuda --augment_percent 0.0 --random_aug 0.0 \
      --logdir "$LOGDIR/runs" > "$LOGDIR/${tag}.log" 2>&1 \
      && touch "$LOGDIR/done_${tag}" && log_run "DONE $tag" || log_run "FAIL $tag" &
    sleep 20
  done
  wait
  log_run "WAVE seed=$seed COMPLETE"
done
log_run "NOAUG ALL DONE"
