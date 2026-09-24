#!/bin/bash
# F0 (V2_REMEDIATION): archive the six old-cohort task directories (every policy, eval,
# meta, kept file and ensemble; the immutable .pkl inputs are excluded) to
# runs/archive/v2_oldcohort_2026-09-14/<task>/, then install the ensembles retrained at the
# stated protocol (v_ensemble_pess300_seed{s}.pt, stage_pess300.sh) as v_ensemble_pess_seed{s}.pt.
# pointbutton1 is archived too because its bc_policy.pt is regenerated (R6). Refuses to run
# twice: the archive directory must not already hold a task copy.
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

set -eu
D=${CSC_WORK}/outputs
A=${CSC_RUNS}/archive/v2_oldcohort_2026-09-14
for t in halfcheetah_velocity cargoal1_dsrl walker2d_velocity ant_velocity hopper_velocity swimmer_velocity pointbutton1; do
  [ -f $A/$t/ARCHIVED ] && { echo "$t already archived"; continue; }
  mkdir -p $A/$t
  rsync -a --exclude='*.pkl' $D/$t/ $A/$t/
  date > $A/$t/ARCHIVED; echo "archived $t ($(ls $A/$t | wc -l) files)"
done
for t in halfcheetah_velocity cargoal1_dsrl walker2d_velocity ant_velocity hopper_velocity swimmer_velocity; do
  for s in 0 1 2 3 4; do
    [ -f $D/$t/v_ensemble_pess300_seed$s.pt ] || { echo "MISSING pess300 $t s$s"; exit 1; }
    cp -p $D/$t/v_ensemble_pess300_seed$s.pt $D/$t/v_ensemble_pess_seed$s.pt
  done
  echo "installed pess300 -> pess on $t"
done
echo INSTALL DONE
