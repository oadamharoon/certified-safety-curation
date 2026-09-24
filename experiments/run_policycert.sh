#!/usr/bin/env bash
# Extra evaluation rollouts for the policy-level MEAN-cost certificate (item A2).
#
# The published evaluations use 100 episodes, which supports a Clopper-Pearson bound
# on Pr[cost > budget] but NOT a bound on E[cost]: with episodic cost in [0, 1000] the
# range term of any empirical-Bernstein bound swamps the deviation term at n = 100.
# 2000 episodes is pre-registered here for every cell, uniformly, so the budget is not
# chosen per task from the numbers it will certify.
#
# Writes eval_results_certn2k_calfilt_csf_seed<S>.json, a NEW filename: nothing
# published is touched, and the harvest globs for the published arms cannot match it.
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
JOBS=$REPO/runs/logs/policycert/jobs.txt
LOGD=$REPO/runs/logs/policycert
PAR="${PAR:-24}"
EPS="${EPS:-2000}"

run_one() {
  set -u
  local task=$1 seed=$2 pf=$3
  local tag="certn2k_calfilt_csf_seed${seed}"
  local log="$LOGD/${task}_seed${seed}.log"
  if [ -f "$CD/outputs/$task/eval_results_${tag}.json" ] && [ -z "${FORCE:-}" ]; then
    echo "skip (exists): $task seed$seed"; return 0
  fi
  ( cd "$CD" && env PYTHONNOUSERSITE=1 SAFETY_VLM_TASK="$task" EVAL_EPISODES="$EPS" \
      WANDB_MODE=disabled "$PY" scripts/05_evaluate.py \
      --policy_file "$pf" --results_suffix "$tag" ) >"$log" 2>&1
  if [ -f "$CD/outputs/$task/eval_results_${tag}.json" ]; then
    echo "ok: $task seed$seed"
  else
    echo "FAIL: $task seed$seed (see $log)"
  fi
}
export -f run_one
export REPO PY CD LOGD EPS

awk '{print $1" "$2" "$3}' "$JOBS" \
  | xargs -P "$PAR" -I{} bash -c 'run_one $0' {} \
  | tee "$LOGD/driver.log"
echo "=== done: $(grep -c '^ok:' "$LOGD/driver.log") ok, $(grep -c '^FAIL:' "$LOGD/driver.log") failed ==="
