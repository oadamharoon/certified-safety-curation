#!/bin/bash
# Counts cells as children of the driver's xargs pool, not by pgrep on the script
# name: the monitor's own command line contains that name, so pgrep -fc counted
# itself and reported more concurrency than -P allows. Twice.
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
LOG=runs/logs/reverify_pess_refresh.log
prev=""
while true; do
  done_n=$(ls runs/logs/reverify/done_*xlab_learned* 2>/dev/null | wc -l)
  drv=$(pgrep -x -f "bash runs/scripts/stage_reverify.sh" 2>/dev/null | head -1)
  live=0
  if [ -n "$drv" ]; then
    xp=$(pgrep -P "$drv" 2>/dev/null | head -1)
    [ -n "$xp" ] && for w in $(pgrep -P "$xp" 2>/dev/null); do
      live=$((live + $(pgrep -P "$w" 2>/dev/null | wc -l)))
    done
  fi
  fails=$(grep -Ec "TRAIN FAIL|EVAL FAIL" "$LOG" 2>/dev/null) || fails=0
  oom=$(grep -c "Killed" "$LOG" 2>/dev/null) || oom=0
  cur="pess_refresh: $done_n/38 done, $live running, $fails failed, $oom killed"
  [ "$cur" != "$prev" ] && { echo "$cur"; prev="$cur"; }
  if [ "$fails" -gt 0 ] || [ "$oom" -gt 0 ]; then
    echo "ATTENTION: $fails failed, $oom killed - see $LOG"; fi
  if [ "$done_n" -ge 38 ]; then echo "ALL 38 pess-refresh cells complete"; break; fi
  if [ -z "$drv" ] && [ "$done_n" -lt 38 ]; then
    echo "WARNING: driver gone with only $done_n/38 done"; break; fi
  sleep 240
done
