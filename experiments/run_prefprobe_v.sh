#!/bin/bash
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
D=${CSC_WORK}
L=${CSC_RUNS}/logs/prefprobe
PY=${PYTHON}
mkdir -p "$L"; cd "$D"
for task in pointgoal1_dsrl halfcheetah_velocity; do
  for arm in base hicontrast retmatched; do
    [ -f "$D/outputs/$task/v_probe_${arm}.pt" ] && continue
    env PYTHONNOUSERSITE=1 SAFETY_VLM_TASK=$task SEED_OVERRIDE=0 WANDB_MODE=disabled \
      PREF_LABELS_FILE=gt_labels_probe_${arm}.json V_OUT=v_probe_${arm}.pt \
      $PY scripts/04n_train_v_only.py > "$L/${task}_${arm}.log" 2>&1 \
      && echo "[$(date +%H:%M:%S)] done $task $arm" || echo "[$(date +%H:%M:%S)] FAIL $task $arm"
  done
done
echo PREFPROBE V DONE
