#!/bin/bash
# Single detached driver for every outstanding verification stage.
#
# The previous run lost its chaining when the controlling process exited: the
# stages were started as session-bound background jobs, so they died with it at
# cf 24/33. This script is launched under setsid, in its own session and process
# group, so it survives that. Stages run sequentially; concurrency within a stage
# is PAR.
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
export PAR=${PAR:-6}
L=runs/logs
run_stage () {
  local name=$1 jobs=$2
  echo "=== [$(date +%H:%M:%S)] stage $name (PAR=$PAR) ==="
  free -g | sed -n '2p'
  env JOBS_FILE="$PWD/$jobs" bash runs/scripts/stage_reverify.sh > "$L/reverify_${name}.log" 2>&1
  local f; f=$(grep -cE "TRAIN FAIL|EVAL FAIL" "$L/reverify_${name}.log" 2>/dev/null) || f=0
  local k; k=$(grep -c "Killed" "$L/reverify_${name}.log" 2>/dev/null) || k=0
  echo "[$(date +%H:%M:%S)] stage $name finished: $f failed, $k killed"
}
run_stage cf2      runs/logs/reverify/jobs_cf.txt
run_stage qfilt    runs/logs/reverify/jobs_qfilt.txt
run_stage prefdeps runs/logs/reverify/jobs_prefdeps.txt
run_stage expk1    runs/logs/reverify/jobs_expk1.txt
echo "[$(date +%H:%M:%S)] ALL STAGES COMPLETE"
