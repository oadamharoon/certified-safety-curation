#!/bin/bash
# Run the 12 preference-dependent cells (cpl_gt, vawr_from_bcsafe) after qfilt.
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
sleep 120
while pgrep -x -f "bash runs/scripts/stage_reverify.sh" >/dev/null 2>&1; do sleep 60; done
echo "prior stages finished; starting prefdeps (12 cells)"
free -g | sed -n '2p'
env JOBS_FILE=${CSC_RUNS}/logs/reverify/jobs_prefdeps.txt \
  bash runs/scripts/stage_reverify.sh > runs/logs/reverify_prefdeps.log 2>&1
d=$(ls runs/logs/reverify/done_*cplgt_seed* runs/logs/reverify/done_*vawr_from_bcsafe* 2>/dev/null | wc -l)
f=$(grep -cE "TRAIN FAIL|EVAL FAIL" runs/logs/reverify_prefdeps.log 2>/dev/null) || f=0
echo "prefdeps done: $d/12, $f failed"
