#!/bin/bash
# Counts cells as grandchildren of the live driver's xargs pool. Counting by
# pgrep on the script name matched the monitor's own command line twice today
# and reported more concurrency than -P allows.
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
prev=""
gone=0
while true; do
  cp=$(ls runs/logs/reverify/done_*calfilt_pref* 2>/dev/null | wc -l)
  cf=$(ls runs/logs/reverify/done_*cf_M8* runs/logs/reverify/done_*cfv2_M16* 2>/dev/null | wc -l)
  drv=$(pgrep -x -f "bash runs/scripts/stage_reverify.sh" 2>/dev/null | head -1)
  live=0
  if [ -n "$drv" ]; then
    xp=$(pgrep -P "$drv" 2>/dev/null | head -1)
    [ -n "$xp" ] && for w in $(pgrep -P "$xp" 2>/dev/null); do
      live=$((live + $(pgrep -P "$w" 2>/dev/null | wc -l)))
    done
  fi
  f1=$(grep -cE "TRAIN FAIL|EVAL FAIL" runs/logs/reverify_calfilt_pref.log 2>/dev/null) || f1=0
  f2=$(grep -cE "TRAIN FAIL|EVAL FAIL" runs/logs/reverify_cf.log 2>/dev/null) || f2=0
  k=$(grep -ch "Killed" runs/logs/reverify_calfilt_pref.log runs/logs/reverify_cf.log 2>/dev/null | paste -sd+ | bc) || k=0
  qf=$(ls runs/logs/reverify/done_*qfilt_seed* 2>/dev/null | wc -l)
  cur="calfilt_pref $cp/25 | cf $cf/33 | qfilt $qf/9 | $live running | fails $((f1+f2)) | killed ${k:-0}"
  [ "$cur" != "$prev" ] && { echo "$cur"; prev="$cur"; }
  [ $((f1+f2)) -gt 0 ] && echo "ATTENTION: failures - see runs/logs/reverify_cf.log"
  [ "${k:-0}" -gt 0 ] && echo "ATTENTION: OOM kill detected"
  if [ "$cp" -ge 25 ] && [ "$cf" -ge 33 ] && [ "$qf" -ge 9 ]; then echo "ALL REMAINING VERIFICATION CELLS COMPLETE"; break; fi
  # A stage handing off to the next leaves a gap with no driver; only treat a
  # missing driver as terminal after it has persisted across two checks.
  if [ -z "$drv" ]; then gone=$((gone+1)); else gone=0; fi
  if [ "$gone" -ge 3 ] && [ "$cf" -lt 33 ]; then
    echo "WARNING: no driver for 3 checks, cf only $cf/33"; break; fi
  sleep 240
done
