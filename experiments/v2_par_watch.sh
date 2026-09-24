#!/bin/bash
# Raise the regeneration drivers' parallelism from 3 to TARGET by sending SIGUSR1 to their GNU
# xargs (each signal adds one worker), only while the GPU has >3 GB free and the CPUs are not
# saturated. Tracks each xargs PID so a later driver (P1, pass 2) is raised too. Exits when
# the pass-2 chain reports done.
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

TARGET=${TARGET:-6}; declare -A sent
while ! grep -q "PASS2 CHAIN DONE" ${CSC_RUNS}/logs/v2regen/progress.log 2>/dev/null; do
  for pid in $(ps -eo pid,args | grep "[x]args -a ${CSC_RUNS}/logs/v2regen/jobs" | awk '{print $1}'); do
    n=${sent[$pid]:-0}
    free=$(( $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits) - $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) ))
    idle=$(top -bn1 | awk '/Cpu\(s\)/ {print int($8)}')
    if [ $n -lt $((TARGET-3)) ] && [ $free -gt 3000 ] && [ ${idle:-0} -gt 15 ]; then
      kill -USR1 $pid && sent[$pid]=$((n+1)) && echo "[$(date +%m/%d-%H:%M)] xargs $pid -> PAR $((3+n+1)) (free ${free}MiB, idle ${idle}%)"
    fi
  done
  sleep 180
done
echo "[$(date +%m/%d-%H:%M)] par watch exit"
