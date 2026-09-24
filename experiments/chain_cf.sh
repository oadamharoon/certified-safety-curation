#!/bin/bash
# Run the 33 cf reruns + 1 cf_octg reproduction probe once calfilt_pref is done.
# Chained, not concurrent: 04o holds the dataset, a dynamics ensemble and M
# candidate rollouts in memory, and an 8-way overlap OOM-killed a cell earlier.
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

cd ${CSC_WORKSPACE}
WAIT_PID=$1
if [ -n "$WAIT_PID" ]; then while [ -d /proc/$WAIT_PID ]; do sleep 60; done; fi
n=$(ls runs/logs/reverify/done_*calfilt_pref* 2>/dev/null | wc -l)
echo "calfilt_pref finished: $n/25"
free -g | sed -n '2p'
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
echo "starting cf stage (34 cells)"
env JOBS_FILE=${CSC_RUNS}/logs/reverify/jobs_cf.txt \
  bash runs/scripts/stage_reverify.sh > runs/logs/reverify_cf.log 2>&1
d=$(ls runs/logs/reverify/done_*cf_M8* runs/logs/reverify/done_*cfv2_M16* 2>/dev/null | wc -l)
f=$(grep -cE "TRAIN FAIL|EVAL FAIL" runs/logs/reverify_cf.log 2>/dev/null) || f=0
echo "cf stage done: $d/33 reruns complete, $f failed"
