#!/bin/bash
# Resume F2's CDT cells after the LLM paper's GPU queue (certified-data-curation-llm/scripts/
# run_llm_rest.sh) finishes; stage_v2_f2.sh is idempotent (done markers), so the BC/CPL cells
# already done are skipped. CDT_PAR=3: the GPU's CDT throughput is ~0.6 cells/h regardless.
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
until grep -q "E15B DONE" $W/certified-data-curation-llm/logs/run_e15b.out 2>/dev/null; do sleep 300; done
while pgrep -x xargs -a | grep -q "jobs_bc.txt\|jobs_cdt.txt"; do sleep 300; done
echo "[$(date +%m/%d-%H:%M)] F1 PASS 3 (32 vfilt_random/return cells that OOM'd beside the 8B SFTs on 09/16), then F2 RESUME" >> $L/progress.log
PAR=3 bash $W/runs/scripts/stage_v2_regen_pass2.sh > $W/runs/logs/v2regen/driver_pass3.out 2>&1
CDT_PAR=3 BC_PAR=3 bash $W/runs/scripts/stage_v2_f2.sh
