#!/bin/bash
# cf_octg restated at documented defaults, chained behind the main driver.
# The probe proved cf_octg does not reproduce even with AWR_EPOCHS/BATCH_SIZE
# corrected (R 11.64->19.72, C 11.55->32.72), so its unrecorded env knobs
# (CF_BETA, LAMBDA_V, LAMBDA_DYN, CF_PERSTATE_NORM, CAND_SIGMA_SCALE, DYN_K) are
# the blocker. Restating the whole arm keeps the cf family one vintage: leaving
# cf_octg alone while cf_v1/cf_v2 are restated is the mixing being eliminated.
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
while pgrep -x -f "bash runs/scripts/run_all_remaining.sh" >/dev/null 2>&1; do sleep 60; done
echo "=== [$(date +%H:%M:%S)] cf_octg stage (27 cells, PAR=$PAR) ==="
free -g | sed -n '2p'
env JOBS_FILE="$PWD/runs/logs/reverify/jobs_cfoctg.txt" \
  bash runs/scripts/stage_reverify.sh > runs/logs/reverify_cfoctg.log 2>&1
d=0; while read t a tag s; do [ -f "runs/logs/reverify/done_${t}_${tag}" ] && d=$((d+1)); done < runs/logs/reverify/jobs_cfoctg.txt
f=$(grep -cE "TRAIN FAIL|EVAL FAIL" runs/logs/reverify_cfoctg.log 2>/dev/null) || f=0
echo "[$(date +%H:%M:%S)] cf_octg done: $d/27, $f failed"
