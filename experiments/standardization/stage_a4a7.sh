#!/bin/bash
# A4: operator arms on regenerated certified selections (PointGoal1 a25, CarRun a25)
# A7: H-sweep calibrated arms (18 configs x 3 seeds)
set -u
W=/home/omniverse/workspace/safevlmcpl/runs
D=/home/omniverse/workspace/safevlmcpl/vlm-with-cpl/new_data
LOGDIR=$W/logs/a4a7
mkdir -p "$LOGDIR"
log_run () { echo "[$(date +%m/%d-%H:%M:%S)] $1" | tee -a "$LOGDIR/progress.log"; }

op_arm () {   # task selfile qtag variant seed
  local task=$1 selfile=$2 q=$3 var=$4 seed=$5
  local tag extra
  case $var in
    c1) tag="wc1${q}_seed${seed}";  extra="RETURN_WEIGHTED=1 WEIGHT_CLIP=1.0";;
    c2) tag="wbc${q}_seed${seed}";  extra="RETURN_WEIGHTED=1 WEIGHT_CLIP=2.0";;
    c3) tag="wc3${q}_seed${seed}";  extra="RETURN_WEIGHTED=1 WEIGHT_CLIP=3.0";;
    th) tag="th${q}_seed${seed}";   extra="RETURN_TOPHALF=1";;
  esac
  local key="${task}_${tag}"
  [ -f "$LOGDIR/done_${key}" ] && return 0
  cd $D
  env SAFETY_VLM_TASK=$task WANDB_MODE=disabled OMP_NUM_THREADS=3 SEED_OVERRIDE=$seed \
      KEPT_JSON="$W/selections/$selfile" OUT_TAG=$tag $extra \
    conda run -n safevlmcpl --no-capture-output python $W/scripts/bc_on_subset.py \
    > "$LOGDIR/${key}.log" 2>&1 || { echo "[$(date +%m/%d-%H:%M:%S)] FAIL $key" >> "$LOGDIR/progress.log"; return 1; }
  env SAFETY_VLM_TASK=$task WANDB_MODE=disabled OMP_NUM_THREADS=3 CUDA_VISIBLE_DEVICES="" \
    conda run -n safevlmcpl --no-capture-output python scripts/05_evaluate.py \
    --policy_file "bc_${tag}_policy.pt" --results_suffix "$tag" >> "$LOGDIR/${key}.log" 2>&1 \
    || { echo "[$(date +%m/%d-%H:%M:%S)] FAIL EVAL $key" >> "$LOGDIR/progress.log"; return 1; }
  touch "$LOGDIR/done_${key}"; echo "[$(date +%m/%d-%H:%M:%S)] DONE $key" >> "$LOGDIR/progress.log"
}
h_arm () {    # basetask h seed
  local t=$1 h=$2 seed=$3
  local tag="calfilt_ltt_seed${seed}"
  local key="${t}_h${h}_${tag}"
  [ -f "$LOGDIR/done_${key}" ] && return 0
  cd $D
  local lim; case $t in *velocity*) lim=20;; *) lim=25;; esac
  env SAFETY_VLM_TASK=$t SAFETY_VLM_CONFIG=config_h${h}.yaml WANDB_MODE=disabled \
      OMP_NUM_THREADS=3 SEED_OVERRIDE=$seed MODE=ltt CAL_N=200 ALPHA=0.25 DELTA=0.1 \
      COST_LIMIT=$lim V_ENSEMBLE_FILE=v_ensemble_pess_seed${seed}.pt OUT_TAG=$tag \
    conda run -n safevlmcpl --no-capture-output python scripts/04q_calibrated_vfilter.py \
    > "$LOGDIR/${key}.log" 2>&1 || { echo "[$(date +%m/%d-%H:%M:%S)] FAIL $key" >> "$LOGDIR/progress.log"; return 1; }
  env SAFETY_VLM_TASK=$t SAFETY_VLM_CONFIG=config_h${h}.yaml WANDB_MODE=disabled \
      OMP_NUM_THREADS=3 CUDA_VISIBLE_DEVICES="" \
    conda run -n safevlmcpl --no-capture-output python scripts/05_evaluate.py \
    --policy_file "bc_${tag}_policy.pt" --results_suffix "$tag" >> "$LOGDIR/${key}.log" 2>&1 \
    || { echo "[$(date +%m/%d-%H:%M:%S)] FAIL EVAL $key" >> "$LOGDIR/progress.log"; return 1; }
  touch "$LOGDIR/done_${key}"; echo "[$(date +%m/%d-%H:%M:%S)] DONE $key" >> "$LOGDIR/progress.log"
}
dispatch () { if [ "$1" = op ]; then shift; op_arm "$@"; else shift; h_arm "$@"; fi; }
export -f op_arm h_arm dispatch; export LOGDIR W D

J=$W/a4a7_jobs.txt; : > "$J"
for s in 0 1 2; do
  for spec in "pointgoal1_dsrl_a25_q80_kept.json q80" "pointgoal1_dsrl_a25_q70_kept.json q70" "pointgoal1_dsrl_a25_q65_kept.json q65"; do
    set -- $spec; for v in c1 c2 c3 th; do echo "op pointgoal1_dsrl $1 $2 $v $s" >> "$J"; done
  done
  for spec in "carrun_b_a25_q85_kept.json q85" "carrun_b_a25_q80_kept.json q80"; do
    set -- $spec; for v in c1 c2 c3 th; do echo "op carrun_b $1 $2 $v $s" >> "$J"; done
  done
  for t in halfcheetah_velocity walker2d_velocity ant_velocity hopper_velocity swimmer_velocity cargoal1_dsrl cargoal2 pointgoal1_dsrl pointgoal2; do
    for h in 10 50; do echo "h $t $h $s" >> "$J"; done
  done
done
log_run "A4+A7: $(wc -l < $J) jobs"
xargs -a "$J" -L1 -P 8 bash -c 'dispatch "$@"' _
cd /home/omniverse/workspace/safevlmcpl/iclr2027
conda run -n safevlmcpl --no-capture-output python scripts/collect_results.py >> "$LOGDIR/progress.log" 2>&1
log_run "A4A7 DONE ($(ls $LOGDIR/done_* 2>/dev/null | wc -l) arms)"
