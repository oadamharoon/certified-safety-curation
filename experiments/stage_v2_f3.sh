#!/bin/bash
# F3 (V2_REMEDIATION): regenerate every data artifact that reads the six regenerated tasks'
# value ensembles. Each generator is deterministic, so a task whose ensemble did not change must
# reproduce bit-exactly -- that is the control that validates each rerun. Cheap CPU stats first,
# then the certificate artifacts, then the 2000-episode policy evaluations, which are the long pole.
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
[ -x "$PY" ] || PY=${PYTHON}
S=$W/iclr2027/scripts; L=$W/runs/logs/v2f3
run () {   # script [args]
  local s=$1; shift
  [ -f "$S/$s.py" ] || { echo "[$(date +%m/%d-%H:%M)] MISSING $s" >> $L/progress.log; return; }
  [ -f "$L/done_$s" ] && return 0
  cd $W/iclr2027 && env PYTHONNOUSERSITE=1 "$PY" "$S/$s.py" "$@" > "$L/$s.log" 2>&1 \
    && { touch "$L/done_$s"; echo "[$(date +%m/%d-%H:%M)] DONE $s" >> $L/progress.log; } \
    || echo "[$(date +%m/%d-%H:%M)] FAIL $s" >> $L/progress.log
}
echo "[$(date +%m/%d-%H:%M)] F3 START" >> $L/progress.log
for s in cert_rate_theory stability_stats cardinal_control recover_q1_stats \
         aggregation_ablation transfer_and_kablation robustness_stats obstacle2_stats \
         interpret_v purity_vs_violation e_contamination_sweep certificate_triple \
         alpha_cert_stats meancost_cert_stats guarantee_stats; do run "$s"; done
for s in policy_certificate mean_cost_certificate; do run "$s"; done
echo "[$(date +%m/%d-%H:%M)] F3 COMPLETE ($(ls $L/done_* 2>/dev/null | wc -l) of 17)" >> $L/progress.log
