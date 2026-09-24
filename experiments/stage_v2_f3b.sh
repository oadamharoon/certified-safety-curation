#!/bin/bash
# F3b: the artifacts the first F3 pass missed. These live in certified-safety-curation/analysis and
# in iclr2027/scripts, and all of them read the six regenerated tasks' value ensembles, so every
# number the paper quotes from them was computed before the v2 rewrite.
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
L=$W/runs/logs/v2f3b; mkdir -p $L
run () {   # label script-path [script-args...]
  local tag=$1 path=$2; shift 2
  local cwd=$W/iclr2027
  [ -f "$path" ] || { echo "[$(date +%m/%d-%H:%M)] MISSING $tag" >> $L/progress.log; return; }
  [ -f "$L/done_$tag" ] && return 0
  ( cd "$cwd" && env PYTHONNOUSERSITE=1 PYTHONPATH=$W/datasets "$PY" "$path" "$@" > "$L/$tag.log" 2>&1 ) \
    && { touch "$L/done_$tag"; echo "[$(date +%m/%d-%H:%M)] DONE $tag" >> $L/progress.log; } \
    || echo "[$(date +%m/%d-%H:%M)] FAIL $tag" >> $L/progress.log
}
A=$W/certified-safety-curation/analysis
echo "[$(date +%m/%d-%H:%M)] F3b START" >> $L/progress.log
for s in margin_probe margin_by_H score_correlations weighted_cert seq_cert_sim eprocess_sim; do
  run "$s" "$A/$s.py"
done
run profile_by_H "$W/runs/scripts/profile_by_H.py"   # lives here, not in analysis/
run method_figure    "$W/iclr2027/scripts/method_figure.py"
run landscape_visited "$W/iclr2027/scripts/landscape_visited.py" --task pointgoal1_dsrl
echo "[$(date +%m/%d-%H:%M)] F3b COMPLETE ($(ls $L/done_* 2>/dev/null | wc -l))" >> $L/progress.log
