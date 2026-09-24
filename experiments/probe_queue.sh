#!/bin/bash
# --- paths: set CSC_WORKSPACE or the individual roots; see the README ---
_csc_root () { local d; d="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  while [ "$d" != "/" ]; do [ -e "$d/.csc-root" ] && { printf %s "$d"; return; }; d="$(dirname "$d")"; done
  (cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); }
CSC_REPO="${CSC_REPO:-$(_csc_root)}"
CSC_WORKSPACE="${CSC_WORKSPACE:-$(dirname "$CSC_REPO")}"
CSC_WORK="${CSC_WORK:-$CSC_WORKSPACE/datasets}"
CSC_RUNS="${CSC_RUNS:-$CSC_WORKSPACE/runs}"
CSC_OSRL="${CSC_OSRL:-$CSC_WORKSPACE/osrl}"
CSC_PAPER="${CSC_PAPER:-$CSC_REPO/paper}"
CSC_PAPER_DATA="${CSC_PAPER_DATA:-$CSC_PAPER/data}"
PYTHON="${PYTHON:-python}"
# ------------------------------------------------------------------------

set -u
cd ${CSC_WORK}
env CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=6 SCRATCH=${TMPDIR:-/tmp} ${PYTHON} ${TMPDIR:-/tmp}/eprocess_sim.py > ${TMPDIR:-/tmp}/eprocess_sim.log 2>&1 || echo "FAIL eprocess" >> ${TMPDIR:-/tmp}/probe_progress.log
env CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=6 SCRATCH=${TMPDIR:-/tmp} ${PYTHON} ${TMPDIR:-/tmp}/margin_probe.py > ${TMPDIR:-/tmp}/margin_probe.log 2>&1 || echo "FAIL margin" >> ${TMPDIR:-/tmp}/probe_progress.log
echo "PROBES ALL DONE" >> ${TMPDIR:-/tmp}/probe_progress.log
