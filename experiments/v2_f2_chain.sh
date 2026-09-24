#!/bin/bash
# Waits for F1 (all six tasks' P1 cells, so hopper/swimmer metas exist), builds their selections,
# then runs F2 once the GPU has room for CDT.
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

L=${CSC_RUNS}/logs; cd ${CSC_WORKSPACE}
until grep -q "V2 REGEN F1 COMPLETE" $L/v2regen/progress.log 2>/dev/null; do sleep 300; done
env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES="" ${PYTHON} runs/scripts/build_v2_selections.py hopper_velocity,swimmer_velocity > $L/v2f2/build_hs.log 2>&1
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 14000 ]; do sleep 300; done
CDT_PAR=2 BC_PAR=3 bash runs/scripts/stage_v2_f2.sh > $L/v2f2/driver.out 2>&1
echo "[$(date +%m/%d-%H:%M)] F2 CHAIN DONE" >> $L/v2f2/progress.log
