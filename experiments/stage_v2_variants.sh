#!/bin/bash
# F0b (V2_REMEDIATION, R4): retrain every variant value ensemble the robustness table reads
# (noise05/10/20/30, n100, n300, boltz) on the nine analysis tasks, seeds 0-2, at the paper's
# stated protocol (04n_train_v_only: 300 epochs, batch 512, K=3, current gt_labels.json).
# Boltzmann temperature T_lab = 3 on every task (the 07-14 originals used 5 on velocity).
# The 07-14 originals are archived (moved) before anything is written. Idempotent: a variant
# whose archive copy exists and whose new file exists is skipped.
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
D=${CSC_WORK}
A=${CSC_RUNS}/archive/v2_oldcohort_2026-09-14
L=${CSC_RUNS}/logs/v2variants
PY=${PYTHON}
F0=$(date -d '2026-09-14 21:50' +%s)
mkdir -p $L; cd $D
TASKS="halfcheetah_velocity walker2d_velocity ant_velocity hopper_velocity swimmer_velocity cargoal1_dsrl cargoal2 pointgoal1_dsrl pointgoal2"
for t in $TASKS; do
  mkdir -p $A/$t
  for v in noise05 noise10 noise20 noise30 n100 n300 boltz; do
    case $v in
      noise05) X="PREF_NOISE=0.05";; noise10) X="PREF_NOISE=0.10";; noise20) X="PREF_NOISE=0.20";; noise30) X="PREF_NOISE=0.30";;
      n100) X="PREF_SUBSET=100";; n300) X="PREF_SUBSET=300";; boltz) X="PREF_BOLTZ=3";;
    esac
    for s in 0 1 2; do
      f=outputs/$t/v_ensemble_${v}_seed$s.pt
      # Skip only a file written by THIS stage (newer than the F0 install, 2026-09-14 21:50). The
      # first run (09-14) skipped 103 variants on walker2d/ant/hopper/swimmer/cargoal1 because F0
      # had already copied those task dirs to the archive, so "archive copy exists" was mistaken
      # for "already retrained"; found 2026-09-18 while confirming completeness (ledger F0b).
      if [ -f $f ] && [ $(stat -c %Y $f) -gt $F0 ]; then continue; fi
      if [ -f $f ] && [ ! -f $A/$t/v_ensemble_${v}_seed$s.pt ]; then mv $f $A/$t/; fi
      rm -f $f
      env PYTHONNOUSERSITE=1 SAFETY_VLM_TASK=$t WANDB_MODE=disabled SEED_OVERRIDE=$s V_OUT=v_ensemble_${v}_seed$s.pt $X \
        $PY scripts/04n_train_v_only.py > $L/${t}_${v}_s$s.log 2>&1 \
        && echo "[$(date +%H:%M)] done $t $v s$s $(grep -m1 'V-only:' $L/${t}_${v}_s$s.log) $(grep -m1 '\[boltz\]\|\[noise\]\|\[subset\]' $L/${t}_${v}_s$s.log)" || echo "FAIL $t $v s$s"
    done
  done
done
echo VARIANTS DONE
