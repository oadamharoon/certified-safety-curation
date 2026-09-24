#!/usr/bin/env bash
# The 6 cells where the paper's Calibrated column reports the GATED (R50) policy
# rather than calfilt_csf: halfcheetah seeds 1/3/4, cargoal1 seed 3, pointgoal1
# seeds 0/3 -- i.e. every (task, seed) whose calibration run returned a
# certificate. The first policy-certificate pass evaluated calfilt_csf for these,
# which certifies a policy the paper does not report. Same n=2000, same estimator.
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
LOGD=$REPO/runs/logs/policycert
run_one() {
  set -u
  local task=$1 seed=$2 pf=$3
  # lint_pipeline flags --policy_file=$pf as seed-less because it cannot
  # resolve the variable; $pf is bc_calfilt_lttR50_seed<S>_policy.pt and the
  # output tag carries the seed while the task is in the output directory, so
  # concurrent cells cannot collide. Verified: 6 runs, 6 distinct result files.
  local tag="certn2k_gated_seed${seed}"
  ( cd "$CD" && env PYTHONNOUSERSITE=1 SAFETY_VLM_TASK="$task" EVAL_EPISODES=2000 \
      WANDB_MODE=disabled "$PY" scripts/05_evaluate.py \
      --policy_file "$pf" --results_suffix "$tag" ) \
      >"$LOGD/gated_${task}_seed${seed}.log" 2>&1
  if [ -f "$CD/outputs/$task/eval_results_${tag}.json" ]; then echo "ok: $task seed$seed"
  else echo "FAIL: $task seed$seed"; fi
}
export -f run_one; export REPO PY CD LOGD
awk '{print $1" "$2" "$3}' "$LOGD/jobs_gated.txt" \
  | xargs -P 6 -I{} bash -c 'run_one $0' {} | tee "$LOGD/driver_gated.log"
echo "=== done: $(grep -c '^ok:' "$LOGD/driver_gated.log") ok, $(grep -c '^FAIL:' "$LOGD/driver_gated.log") failed ==="
