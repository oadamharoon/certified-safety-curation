#!/bin/bash
# F3f: the pre-F0 SCORE CACHE and everything built on it.
#
# data/e_scores/<task>_seed<s>.npz is dated 2026-09-03 and F0 retrained the six tasks'
# ensembles on 09-14, so every artifact reading the cache describes ensembles that no longer
# exist on those six. The recency gate could not see it: it globs data/**/*.json and these
# are .npz. certificate_triple's realized certification rate for HalfCheetah reads 0.13
# where the 2000-draw validation on the current ensembles gives 0.235, which is the symptom.
# Order matters: the cache first, then the two artifacts that consume it.
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

set -u
W=${CSC_WORKSPACE}
PY=${PYTHON}
L=$W/runs/logs/v2f3f; mkdir -p $L
run () {
  local tag=$1 path=$2
  echo "[$(date +%m/%d-%H:%M)] START $tag" >> $L/progress.log
  ( cd "$W/iclr2027" && env PYTHONNOUSERSITE=1 PYTHONPATH=$W/vlm-with-cpl/new_data "$PY" "$path" > "$L/$tag.log" 2>&1 ) \
    && echo "[$(date +%m/%d-%H:%M)] DONE $tag" >> $L/progress.log \
    || { echo "[$(date +%m/%d-%H:%M)] FAIL $tag" >> $L/progress.log; return 1; }
}
echo "[$(date +%m/%d-%H:%M)] F3f START" >> $L/progress.log
run e_cache_scores        "$W/iclr2027/scripts/e_cache_scores.py"        || exit 1
run e_contamination_sweep "$W/iclr2027/scripts/e_contamination_sweep.py" || exit 1
run certificate_triple    "$W/iclr2027/scripts/certificate_triple.py"    || exit 1
echo "[$(date +%m/%d-%H:%M)] F3f COMPLETE" >> $L/progress.log
