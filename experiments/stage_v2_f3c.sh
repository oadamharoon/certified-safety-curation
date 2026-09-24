#!/bin/bash
# F3c: the chain the recency gate found after its false-clean bug was fixed.
# guarantee_validation writes guarantee_stats_2000.json, which margin_vs_yield consumes,
# which label_complexity_ext consumes, which the fit and the CI consume. Strictly ordered.
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

set -u
W=${CSC_WORKSPACE}
PY=${PYTHON}
L=$W/runs/logs/v2f3c; mkdir -p $L
A=$W/certified-safety-curation/analysis
run () {
  local tag=$1 path=$2
  [ -f "$L/done_$tag" ] && return 0
  echo "[$(date +%m/%d-%H:%M)] START $tag" >> $L/progress.log
  ( cd "$W/iclr2027" && env PYTHONNOUSERSITE=1 PYTHONPATH=$W/datasets "$PY" "$path" > "$L/$tag.log" 2>&1 ) \
    && { touch "$L/done_$tag"; echo "[$(date +%m/%d-%H:%M)] DONE $tag" >> $L/progress.log; } \
    || { echo "[$(date +%m/%d-%H:%M)] FAIL $tag" >> $L/progress.log; return 1; }
}
echo "[$(date +%m/%d-%H:%M)] F3c START" >> $L/progress.log
run guarantee_validation "$A/guarantee_validation.py" || exit 1
run margin_vs_yield      "$A/margin_vs_yield.py"      || exit 1
run label_complexity_ext "$A/label_complexity_ext.py" || exit 1
run label_complexity_fit "$A/label_complexity_fit.py" || exit 1
run label_complexity_ci  "$A/label_complexity_ci.py"  || exit 1
echo "[$(date +%m/%d-%H:%M)] F3c COMPLETE" >> $L/progress.log
