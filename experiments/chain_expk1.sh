#!/bin/bash
# Regenerate tab:oracle's learned-value column as a uniform exp-mode k=1 arm on all
# nine analysis tasks (option B). vawr_5seed dates to 2026-06-08, only an eval log
# survives, and its June training script is not identifiable, so the column could not
# be re-derived; it also silently fell back to xlab_exp_k1 on pointgoal2, mixing
# vintages within one row. PointGoal2 already has exp_k1, so only 8 tasks x 3 seeds
# run here. Three seeds matches the other analysis columns in the same table.
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

cd ${CSC_WORKSPACE}
sleep 180
while pgrep -x -f "bash runs/scripts/stage_reverify.sh" >/dev/null 2>&1; do sleep 60; done
echo "prior stages finished; starting exp_k1 column (24 new cells)"
free -g | sed -n '2p'
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
env JOBS_FILE=${CSC_RUNS}/logs/reverify/jobs_expk1.txt \
  bash runs/scripts/stage_reverify.sh > runs/logs/reverify_expk1.log 2>&1
d=$(ls runs/logs/reverify/done_*xlab_learned_exp_k1* 2>/dev/null | wc -l)
f=$(grep -cE "TRAIN FAIL|EVAL FAIL" runs/logs/reverify_expk1.log 2>/dev/null) || f=0
echo "exp_k1 column done: $d/24, $f failed"
