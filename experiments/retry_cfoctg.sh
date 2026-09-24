#!/bin/bash
# Retry any cf_octg cell that did not produce a done-marker, at low concurrency.
# cargoal2_cf_octg_seed2 was OOM-killed at PAR=6: CarGoal2 is 4.1M transitions and
# 04o holds M candidate rollouts per cell, so the heavy tasks cannot share 6 slots.
# PAR=2 trades wall-clock for headroom on exactly the cells that need it.
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
while pgrep -x -f "bash runs/scripts/run_cfoctg.sh" >/dev/null 2>&1; do sleep 60; done
J=runs/logs/reverify/jobs_cfoctg_retry.txt; : > "$J"
while read t a tag s; do
  [ -f "runs/logs/reverify/done_${t}_${tag}" ] || echo "$t $a $tag $s" >> "$J"
done < runs/logs/reverify/jobs_cfoctg.txt
n=$(wc -l < "$J")
echo "=== [$(date +%m/%d-%H:%M:%S)] cf_octg retry: $n cell(s) outstanding ==="
if [ "$n" -eq 0 ]; then echo "nothing to retry"; exit 0; fi
cat "$J"
free -g | sed -n '2p'
env PAR=2 JOBS_FILE="$PWD/$J" bash runs/scripts/stage_reverify.sh > runs/logs/reverify_cfoctg_retry.log 2>&1
d=0; while read t a tag s; do [ -f "runs/logs/reverify/done_${t}_${tag}" ] && d=$((d+1)); done < "$J"
echo "[$(date +%m/%d-%H:%M:%S)] retry finished: $d/$n recovered"
