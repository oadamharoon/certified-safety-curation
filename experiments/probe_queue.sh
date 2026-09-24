#!/bin/bash
# --- paths: set CSC_WORKSPACE or the individual roots; see the README ---
_csc_root () { local d; d="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  while [ "$d" != "/" ]; do [ -e "$d/.csc-root" ] && { printf %s "$d"; return; }; d="$(dirname "$d")"; done
  (cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); }
CSC_REPO="${CSC_REPO:-$(_csc_root)}"
CSC_WORKSPACE="${CSC_WORKSPACE:-$(dirname "$CSC_REPO")}"
CSC_WORK="${CSC_WORK:-$CSC_WORKSPACE/vlm-with-cpl/new_data}"
CSC_RUNS="${CSC_RUNS:-$CSC_WORKSPACE/runs}"
CSC_OSRL="${CSC_OSRL:-$CSC_WORKSPACE/osrl}"
CSC_PAPER="${CSC_PAPER:-$CSC_REPO/paper}"
CSC_PAPER_DATA="${CSC_PAPER_DATA:-$CSC_PAPER/data}"
PYTHON="${PYTHON:-python}"
# ------------------------------------------------------------------------

set -u
cd ${CSC_WORK}
env CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=6 SCRATCH=/tmp/claude-1001/-home-omniverse-workspace-safevlmcpl/cbe3ff25-bd02-4cf4-9f36-173bf5fa270c/scratchpad conda run -n safevlmcpl --no-capture-output python /tmp/claude-1001/-home-omniverse-workspace-safevlmcpl/cbe3ff25-bd02-4cf4-9f36-173bf5fa270c/scratchpad/eprocess_sim.py > /tmp/claude-1001/-home-omniverse-workspace-safevlmcpl/cbe3ff25-bd02-4cf4-9f36-173bf5fa270c/scratchpad/eprocess_sim.log 2>&1 || echo "FAIL eprocess" >> /tmp/claude-1001/-home-omniverse-workspace-safevlmcpl/cbe3ff25-bd02-4cf4-9f36-173bf5fa270c/scratchpad/probe_progress.log
env CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=6 SCRATCH=/tmp/claude-1001/-home-omniverse-workspace-safevlmcpl/cbe3ff25-bd02-4cf4-9f36-173bf5fa270c/scratchpad conda run -n safevlmcpl --no-capture-output python /tmp/claude-1001/-home-omniverse-workspace-safevlmcpl/cbe3ff25-bd02-4cf4-9f36-173bf5fa270c/scratchpad/margin_probe.py > /tmp/claude-1001/-home-omniverse-workspace-safevlmcpl/cbe3ff25-bd02-4cf4-9f36-173bf5fa270c/scratchpad/margin_probe.log 2>&1 || echo "FAIL margin" >> /tmp/claude-1001/-home-omniverse-workspace-safevlmcpl/cbe3ff25-bd02-4cf4-9f36-173bf5fa270c/scratchpad/probe_progress.log
echo "PROBES ALL DONE" >> /tmp/claude-1001/-home-omniverse-workspace-safevlmcpl/cbe3ff25-bd02-4cf4-9f36-173bf5fa270c/scratchpad/probe_progress.log
