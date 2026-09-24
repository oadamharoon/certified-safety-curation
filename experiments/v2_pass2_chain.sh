#!/bin/bash
# After the first F1 pass ends, rerun the idempotent drivers to fill cells that failed
# (04p env bug for random/return controls, fixed in pass2; OOMs while sharing the GPU).
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

L=${CSC_RUNS}/logs
until grep -q "V2 REGEN F1 COMPLETE" $L/v2regen/progress.log 2>/dev/null; do sleep 300; done
cd ${CSC_WORKSPACE}
PAR=${PAR:-3} bash runs/scripts/stage_v2_regen_pass2.sh > $L/v2regen/driver_pass2.out 2>&1
bash runs/scripts/stage_v2_variants.sh >> $L/v2variants/progress.log 2>&1
echo "[$(date +%m/%d-%H:%M)] PASS2 CHAIN DONE: $(ls $L/v2regen/done_* | wc -l) cells done, variants $(ls datasets/outputs/*/v_ensemble_{noise05,noise10,noise20,noise30,n100,n300,boltz}_seed[012].pt 2>/dev/null | wc -l)/189" >> $L/v2regen/progress.log
