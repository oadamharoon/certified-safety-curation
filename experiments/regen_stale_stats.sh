#!/usr/bin/env bash
# Regenerate the data artifacts that were computed BEFORE the 2026-08-16 15:32-15:34
# v_ensemble_pess rewrite and that load those checkpoints. guarantee_stats.json hit
# this first; the sweep over iclr2027/data found five more behind the same event.
# Each generator is deterministic, so tasks whose ensembles did not change must
# reproduce bit-exactly -- that is the control that validates each rerun.
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
PY=${PYTHON}
P=${CSC_REPO}/paper
L=${CSC_RUNS}/logs/guarantee
for s in aggregation_ablation transfer_and_kablation meancost_cert_stats robustness_stats; do
  echo "=== $s ==="
  env PYTHONNOUSERSITE=1 "$PY" "$P/$s.py" > "$L/$s.log" 2>&1 \
    && echo "  ok" || echo "  FAILED (see $L/$s.log)"
done
echo "=== all done ==="
