#!/bin/bash
# B1 (V2_REMEDIATION): retrain the six old-cohort value ensembles at the paper's STATED
# protocol (04n_train_v_only: v_epochs = cpl_epochs = 300, batch 512, 1000 pairs, K = 3),
# seeds 0-4, into v_ensemble_pess300_seed{s}.pt so the existing checkpoints are untouched
# until the B2 comparison and the B3 decision. Uses the archived gt_labels.json of each task.
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
D=${CSC_WORK}; L=${CSC_RUNS}/logs/pess300
PY=${PYTHON}; mkdir -p $L; cd $D
for t in halfcheetah_velocity cargoal1_dsrl walker2d_velocity ant_velocity hopper_velocity swimmer_velocity; do
  for s in 0 1 2 3 4; do
    [ -f outputs/$t/v_ensemble_pess300_seed$s.pt ] && continue
    env PYTHONNOUSERSITE=1 SAFETY_VLM_TASK=$t WANDB_MODE=disabled SEED_OVERRIDE=$s V_OUT=v_ensemble_pess300_seed$s.pt \
      $PY scripts/04n_train_v_only.py > $L/${t}_s$s.log 2>&1 && echo "[$(date +%H:%M)] done $t s$s $(grep -m1 'V-only:' $L/${t}_s$s.log)" || echo "FAIL $t s$s"
  done
done
echo PESS300 DONE
