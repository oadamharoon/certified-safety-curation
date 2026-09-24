#!/usr/bin/env bash
# F3e: re-evaluate the 2000-episode policy-certificate cells whose POLICY was retrained
# after the F0 install.
#
# The first pass ran 2026-09-01. F0/F1 then retrained calfilt_csf (and its gated R50
# counterpart) on the six regenerated tasks on 09-15/09-16, so on those six tasks the
# published 2000-episode numbers -- the mean-cost certificate, the policy certificate and
# the 75-cell seed-failure analysis -- described policies that no longer exist. The
# completeness gate's R3 could not see this: it globs the PUBLISHED eval filenames, and
# certn2k_* was deliberately named so those globs cannot match it.
#
# 30 calfilt_csf cells (six tasks x five seeds) plus the four gated cells whose certified
# status or policy changed (halfcheetah s4, walker2d s2/s3, cargoal1 s3). PointGoal1's two
# gated cells keep their 09-01 evaluation: its policies are unchanged since 08-19.
# Per-episode costs stay in the logs, which is what the episode-median claims are read from.
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
REPO=${CSC_WORKSPACE}
PY=${PYTHON}
CD=$REPO/datasets
LOGD=$REPO/runs/logs/policycert_v2
JOBS=$LOGD/jobs.txt
PAR="${PAR:-6}"
EPS="${EPS:-2000}"
run_one() {
  set -u
  local task=$1 seed=$2 pf=$3 tag=$4
  local log="$LOGD/${task}_${tag}.log"
  ( cd "$CD" && env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 SAFETY_VLM_TASK="$task" EVAL_EPISODES="$EPS" \
      WANDB_MODE=disabled "$PY" scripts/05_evaluate.py \
      --policy_file "$pf" --results_suffix "$tag" ) >"$log" 2>&1
  if [ -f "$CD/outputs/$task/eval_results_${tag}.json" ]; then echo "ok: $task $tag"
  else echo "FAIL: $task $tag (see $log)"; fi
}
export -f run_one; export REPO PY CD LOGD EPS
awk '{print $1" "$2" "$3" "$4}' "$JOBS" \
  | xargs -P "$PAR" -I{} bash -c 'run_one $0 $1 $2 $3' {} \
  | tee "$LOGD/driver.log"
echo "=== done: $(grep -c '^ok:' "$LOGD/driver.log") ok, $(grep -c '^FAIL:' "$LOGD/driver.log") failed ==="
