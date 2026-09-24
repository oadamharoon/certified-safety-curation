#!/bin/bash
# BulletGym extension: full pipeline on 5 tasks, budget 10, zero tuning.
# Per task: segment -> label -> V x3 -> {calfilt x5, vfilt x5, BC-All, BC-Safe x5} -> evals
# Then: 2000-draw guarantee resampling over the bullet scores.
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
S=${TMPDIR:-/tmp}
LOGDIR=$S/bullet_logs
mkdir -p "$LOGDIR"
cd ${CSC_WORK}
log_run () { echo "[$(date +%m/%d-%H:%M:%S)] $1" | tee -a "$LOGDIR/progress.log"; }

TASKS="ballrun_b:0.090 ballcircle_b:0.097 carcircle_b:0.114 carrun_b:0.462 dronerun_b:0.280"
LIM=10

prep_task () {
  local task=$1
  local B="env SAFETY_VLM_TASK=$task WANDB_MODE=disabled OMP_NUM_THREADS=4"
  if [ ! -f "$LOGDIR/done_seg_${task}" ]; then
    log_run "SEG $task"
    $B ${PYTHON} scripts/01_segment_and_filter.py \
      > "$LOGDIR/${task}_01.log" 2>&1 || { log_run "FAIL seg $task"; return 1; }
    $B ${PYTHON} scripts/03b_label_by_cost.py \
      >> "$LOGDIR/${task}_01.log" 2>&1 || { log_run "FAIL label $task"; return 1; }
    touch "$LOGDIR/done_seg_${task}"
  fi
  for seed in 0 1 2 3 4; do
    if [ ! -f "$LOGDIR/done_v_${task}_s${seed}" ]; then
      log_run "V $task s$seed"
      $B SEED_OVERRIDE=$seed ${PYTHON} \
        python scripts/04n_train_v_only.py > "$LOGDIR/${task}_v${seed}.log" 2>&1 \
        || { log_run "FAIL V $task s$seed"; return 1; }
      touch "$LOGDIR/done_v_${task}_s${seed}"
    fi
  done
  # kept-index files for BC-All / BC-Safe
  $B ${PYTHON} - <<PYEOF
import json, pickle, numpy as np, yaml, os
cfg = yaml.safe_load(open("config.yaml"))
t = "$task"
trajs = pickle.load(open(cfg["tasks"][t]["data_pickle"], "rb"))
c = np.array([float(np.sum(x["costs"])) for x in trajs])
os.makedirs("outputs/" + t, exist_ok=True)
json.dump({"kept": list(range(len(trajs)))}, open(f"outputs/{t}/kept_all.json", "w"))
json.dump({"kept": np.where(c <= $LIM)[0].tolist()}, open(f"outputs/{t}/kept_safe.json", "w"))
PYEOF
}
export -f prep_task log_run
export LOGDIR S LIM

run_arm () {
  local task=$1 kind=$2 seed=$3 frac=$4
  local tag key
  cd ${CSC_WORK}
  local B="env SAFETY_VLM_TASK=$task WANDB_MODE=disabled OMP_NUM_THREADS=3 CUDA_VISIBLE_DEVICES="
  case $kind in
    calfilt) tag="calfilt_lttv2_seed${seed}"
      cmd="$B MODE=ltt CAL_N=200 ALPHA=0.25 DELTA=0.1 COST_LIMIT=$LIM \
        V_ENSEMBLE_FILE=v_ensemble_pess_seed${seed}.pt SEED_OVERRIDE=$seed OUT_TAG=$tag \
        ${PYTHON} scripts/04q_calibrated_vfilter.py";;
    vfilt) tag="vfilt_matchgt_seed${seed}"
      cmd="$B FILTER_FRAC=$frac COST_LIMIT=$LIM \
        V_ENSEMBLE_FILE=v_ensemble_pess_seed${seed}.pt SEED_OVERRIDE=$seed OUT_TAG=$tag \
        ${PYTHON} scripts/04p_vfilter_bc.py";;
    bcall) tag="bc"
      cmd="$B KEPT_JSON=outputs/${task}/kept_all.json SEED_OVERRIDE=0 OUT_TAG=$tag \
        ${PYTHON} $S/bc_on_subset.py";;
    bcsafe) tag="bcsafe_seed${seed}"
      cmd="$B KEPT_JSON=outputs/${task}/kept_safe.json SEED_OVERRIDE=$seed OUT_TAG=$tag \
        ${PYTHON} $S/bc_on_subset.py";;
  esac
  key="${task}_${tag}"
  [ -f "$LOGDIR/done_${key}" ] && return 0
  eval "$cmd" > "$LOGDIR/${key}.log" 2>&1 || { echo "FAIL $key" >> "$LOGDIR/progress.log"; return 1; }
  $B CUDA_VISIBLE_DEVICES="" ${PYTHON} \
    python scripts/05_evaluate.py --policy_file "bc_${tag}_policy.pt" \
    --results_suffix "$tag" >> "$LOGDIR/${key}.log" 2>&1 \
    || { echo "FAIL EVAL $key" >> "$LOGDIR/progress.log"; return 1; }
  touch "$LOGDIR/done_${key}"
  echo "[$(date +%m/%d-%H:%M:%S)] DONE $key" >> "$LOGDIR/progress.log"
}
export -f run_arm

# stage 1: prep all tasks (parallel over tasks)
for tl in $TASKS; do
  echo "${tl%%:*}"
done | xargs -L1 -P 5 bash -c 'prep_task "$@"' _
log_run "PREP STAGE COMPLETE"

# stage 2: policy arms
JOBS=$S/bullet_jobs.txt
: > "$JOBS"
for tl in $TASKS; do
  task="${tl%%:*}"; frac="${tl##*:}"
  for seed in 0 1 2 3 4; do
    echo "$task calfilt $seed $frac" >> "$JOBS"
    echo "$task vfilt $seed $frac" >> "$JOBS"
    echo "$task bcsafe $seed $frac" >> "$JOBS"
  done
  echo "$task bcall 0 $frac" >> "$JOBS"
done
log_run "policy jobs: $(wc -l < "$JOBS")"
xargs -a "$JOBS" -L1 -P 6 bash -c 'run_arm "$@"' _
log_run "POLICY STAGE COMPLETE"

# stage 3: guarantee resampling on bullet scores
env CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 ${PYTHON} \
  python $S/bullet_guarantee.py > "$LOGDIR/guarantee.log" 2>&1 \
  && log_run "GUARANTEE STAGE COMPLETE" || log_run "FAIL guarantee stage"
cd ${CSC_PAPER}
${PYTHON} scripts/collect_results.py \
  >> "$LOGDIR/progress.log" 2>&1 && log_run "HARVEST COMPLETE"
log_run "BULLETGYM ALL DONE"
