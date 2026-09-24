#!/bin/bash
# Run the 9 qfilt cells after the cf stage finishes.
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
while pgrep -x -f "bash runs/scripts/stage_reverify.sh" >/dev/null 2>&1; do sleep 60; done
echo "cf stage finished; starting qfilt (9 cells)"
free -g | sed -n '2p'
env JOBS_FILE=${CSC_RUNS}/logs/reverify/jobs_qfilt.txt \
  bash runs/scripts/stage_reverify.sh > runs/logs/reverify_qfilt.log 2>&1
d=$(ls runs/logs/reverify/done_*qfilt_seed* 2>/dev/null | wc -l)
f=$(grep -cE "TRAIN FAIL|EVAL FAIL" runs/logs/reverify_qfilt.log 2>/dev/null) || f=0
echo "qfilt stage done: $d/9, $f failed"
