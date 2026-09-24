#!/bin/bash
# Everything that must be rebuilt once F3e's 2000-episode re-evaluations land, in dependency order.
# h_seed_failures joins the per-run calibration meta with the certn2k evals; the mean-cost and
# policy certificates read those evals directly; make_new_tables emits their two tables.
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
L=$W/runs/logs/v2f3e_post; mkdir -p $L
run () {
  local tag=$1; shift
  echo "[$(date +%m/%d-%H:%M)] START $tag" >> $L/progress.log
  ( cd "$W/iclr2027" && env PYTHONNOUSERSITE=1 PYTHONPATH=$W/datasets "$PY" "$@" > "$L/$tag.log" 2>&1 ) \
    && echo "[$(date +%m/%d-%H:%M)] DONE $tag" >> $L/progress.log \
    || { echo "[$(date +%m/%d-%H:%M)] FAIL $tag" >> $L/progress.log; return 1; }
}
echo "[$(date +%m/%d-%H:%M)] F3e-post START" >> $L/progress.log
run h_seed_failures    "$W/runs/scripts/build_h_seed_failures.py" || exit 1
run mean_cost_cert     "$W/iclr2027/scripts/mean_cost_certificate.py" --write || exit 1
run policy_cert        "$W/iclr2027/scripts/policy_certificate.py" --write || exit 1
run make_new_tables    "$W/iclr2027/scripts/make_new_tables.py" || exit 1
run make_tables        "$W/iclr2027/scripts/make_tables.py" || exit 1
echo "[$(date +%m/%d-%H:%M)] F3e-post COMPLETE" >> $L/progress.log
