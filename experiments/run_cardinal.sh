#!/usr/bin/env bash
# Item G: the CARDINAL-supervision control.
#
# The paper claimed labels-only "presumes a cost function that returns a number".
# It does not: 04s line 59 forms a binary over-budget indicator, the same judgment our
# calibration sample uses. That claim has been corrected, but it described a baseline
# worth actually having -- one that consumes the numeric episodic cost. This is it.
#
# Everything is held identical to the labels-only arm it must be compared against:
# same script, same CAL_N=200, same FILTER_FRAC from gtfracs.json, same architecture,
# same seeds, same clone and evaluation. The ONLY change is LABEL_MODE=cardinal, which
# swaps the binary target for the standardised negative episodic cost and BCE for MSE.
#
# Pre-registered prediction (recorded before the first run): if safety here is
# decodable from the ordering alone, cardinal buys no additional safe tasks at matched
# budget, and the labels-only result is about curation rather than label richness.
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
REPO=${CSC_WORKSPACE}
PY=${PYTHON}
CD=$REPO/vlm-with-cpl/new_data
LOGD=$REPO/runs/logs/cardinal
FRACS=$REPO/runs/logs/cardinal/fracs.json
PAR="${PAR:-5}"

run_one() {
  set -u
  local task=$1 seed=$2
  local tag="labels_cardinal_seed${seed}"
  local key="${task}_${tag}"
  [ -f "$LOGD/done_${key}" ] && { echo "skip (done): $key"; return 0; }
  local frac lim
  # the fraction the matched labels-only n=200 run actually used, read from its own
  # meta file rather than from gtfracs.json (which covers only 14 tasks)
  frac=$(env PYTHONNOUSERSITE=1 "$PY" -c "import json;print(json.load(open('$FRACS'))['$task'])") || return 1
  lim=$(env PYTHONNOUSERSITE=1 "$PY" -c "
import yaml;c=yaml.safe_load(open('$CD/config.yaml'))
t=c['tasks']['$task'];print(t.get('cost_limit',c['cost_limit']))") || return 1
  ( cd "$CD" && env PYTHONNOUSERSITE=1 SAFETY_VLM_TASK="$task" WANDB_MODE=disabled \
      OMP_NUM_THREADS=3 SEED_OVERRIDE="$seed" COST_LIMIT="$lim" \
      LABEL_MODE=cardinal CAL_N=200 FILTER_FRAC="$frac" OUT_TAG="$tag" \
      "$PY" scripts/04s_labels_only_filter.py ) > "$LOGD/${key}.log" 2>&1 \
    || { echo "FAIL TRAIN: $key"; return 1; }
  ( cd "$CD" && env PYTHONNOUSERSITE=1 SAFETY_VLM_TASK="$task" WANDB_MODE=disabled \
      OMP_NUM_THREADS=3 CUDA_VISIBLE_DEVICES="" \
      "$PY" scripts/05_evaluate.py --policy_file "bc_${tag}_policy.pt" \
      --results_suffix "$tag" ) >> "$LOGD/${key}.log" 2>&1 \
    || { echo "FAIL EVAL: $key"; return 1; }
  touch "$LOGD/done_${key}"; echo "ok: $key"
}
export -f run_one; export REPO PY CD LOGD FRACS

awk '{print $1" "$2}' "$LOGD/jobs.txt" \
  | xargs -P "$PAR" -I{} bash -c 'run_one $0' {} | tee "$LOGD/driver.log"
echo "=== done: $(grep -c '^ok:' "$LOGD/driver.log") ok, $(grep -c '^FAIL' "$LOGD/driver.log") failed ==="
