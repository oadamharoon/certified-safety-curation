#!/bin/bash
# Retry of the 13 BC/CPL cells that failed on 09-17 (cudaErrorMemoryAllocation beside the LLM
# queue). Same run functions as stage_v2_f2.sh (sourced up to its job build), BC_PAR=2, beside the
# running CDT cells (a BC/CPL run needs ~2-4 GB; the GPU has ~12 GB free with three CDT cells).
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

W=${CSC_WORKSPACE}; L=$W/runs/logs/v2f2
eval "$(sed -n '/^set -u/,/^export -f run_cdt run_cpl run_bc/p' $W/runs/scripts/stage_v2_f2.sh)"
dispatch () { local k=$1; shift; if [ "$k" = cpl ]; then run_cpl "$1" "$2" "$3" "$4"; else run_bc "$1" "$2" "$3" "$4" "${5:-} ${6:-}"; fi; }
export -f dispatch
echo "[$(date +%m/%d-%H:%M)] BC/CPL retry: $(wc -l < $L/jobs_bc_retry.txt) cells" >> $L/progress.log
xargs -a "$L/jobs_bc_retry.txt" -L1 -P 2 bash -c 'dispatch "$@"' _
echo "[$(date +%m/%d-%H:%M)] BC/CPL retry finished ($(ls $L/done_* | grep -v done_cdt_ | wc -l)/249 done)" >> $L/progress.log
