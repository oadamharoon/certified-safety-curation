#!/bin/bash
# CDT block, chained behind the verification queue. Detached (setsid) so it
# survives session teardown; the earlier chains were session-bound and died.
#
# Two fixes were required before this could do anything useful:
#  - the 9 a40new + 2 echonew selections had NO hdf5, so the *.hdf5 glob skipped
#    exactly the 33 cells the composability grids need;
#  - the task regex stripped only _a25_qNN/_a40_qNN, so a40new/echonew names
#    resolved to unknown tasks and were dropped as "no env".
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

cd ${CSC_WORKSPACE}
while pgrep -x -f "bash runs/scripts/run_all_remaining.sh" >/dev/null 2>&1 \
   || pgrep -x -f "bash runs/scripts/run_cfoctg.sh" >/dev/null 2>&1; do sleep 120; done
echo "=== [$(date +%m/%d-%H:%M:%S)] verification queue clear; starting CDT block ==="
free -g | sed -n '2p'; nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
bash runs/scripts/stage_a3_cdt.sh
echo "=== [$(date +%m/%d-%H:%M:%S)] CDT block finished: $(ls runs/logs/a3cdt/done_* 2>/dev/null | wc -l) done markers ==="
