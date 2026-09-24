#!/bin/bash
# Wait out the 38-cell pess-refresh batch, then run the 25 calfilt_pref cells.
# Chained rather than concurrent: cargoal2 cells are the memory-heavy ones, and an
# earlier 8-way overlap OOM-killed a legitimate cell.
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
WAIT_PID=$1
while [ -d /proc/$WAIT_PID ]; do sleep 60; done
n=$(ls runs/logs/reverify/done_*xlab_learned* 2>/dev/null | wc -l)
echo "pess-refresh batch finished: $n/38 done"
if [ "$n" -lt 38 ]; then
  echo "WARNING: only $n/38 complete - NOT starting calfilt_pref, investigate first"
  exit 1
fi
free -g | sed -n '2p'
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
echo "starting calfilt_pref (25 cells)"
env JOBS_FILE=${CSC_RUNS}/logs/reverify/jobs_calfilt_pref.txt \
  bash runs/scripts/stage_reverify.sh > runs/logs/reverify_calfilt_pref.log 2>&1
echo "calfilt_pref stage exited"
d=$(ls runs/logs/reverify/done_*calfilt_pref* 2>/dev/null | wc -l)
f=$(grep -cE "TRAIN FAIL|EVAL FAIL" runs/logs/reverify_calfilt_pref.log 2>/dev/null) || f=0
echo "calfilt_pref: $d/25 done, $f failed"
