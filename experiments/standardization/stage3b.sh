#!/bin/bash
# Stage 3b (persistent-path rebuild): ablation + BC-family arms, 14 affected tasks.
set -u
W=/home/omniverse/workspace/safevlmcpl/runs
D=/home/omniverse/workspace/safevlmcpl/vlm-with-cpl/new_data
LOGDIR=$W/logs/stage3b
mkdir -p "$LOGDIR"
log_run () { echo "[$(date +%m/%d-%H:%M:%S)] $1" | tee -a "$LOGDIR/progress.log"; }
LIMOF () { case $1 in *velocity*) echo 20;; *_b) echo 10;; *) echo 25;; esac; }
GTF () { conda run -n safevlmcpl --no-capture-output python -c "import json;print(json.load(open('$W/gtfracs.json'))['$1']['gt_frac'])"; }
export -f LIMOF GTF

run_arm () {
  local task=$1 kind=$2 seed=$3
  local lim=$(LIMOF $task) tag script env_extra=""
  cd $D
  case $kind in
    lttR50) tag="calfilt_lttR50_seed${seed}"; script=scripts/04q_calibrated_vfilter.py
            env_extra="MODE=ltt CAL_N=200 ALPHA=0.25 DELTA=0.1 REWARD_FRAC=0.5";;
    a10)    tag="calfilt_a10_seed${seed}";    script=scripts/04q_calibrated_vfilter.py
            env_extra="MODE=ltt CAL_N=200 ALPHA=0.10 DELTA=0.1";;
    a40)    tag="calfilt_a40_seed${seed}";    script=scripts/04q_calibrated_vfilter.py
            env_extra="MODE=ltt CAL_N=200 ALPHA=0.40 DELTA=0.1";;
    pref)   tag="calfilt_pref_seed${seed}";   script=scripts/04q_calibrated_vfilter.py
            env_extra="MODE=pref PREF_Q=0.9";;
    tier2)  tag="calfilt_tier2_seed${seed}";  script=scripts/04q_calibrated_vfilter.py
            env_extra="MODE=ltt CAL_N=200 ALPHA=0.25 DELTA=0.1 TIER2_DELTA=0.5";;
    lo)     tag="labels_only_seed${seed}";    script=scripts/04s_labels_only_filter.py; env_extra="CAL_N=200 FILTER_FRAC=$(GTF $task)";;
    lo50)   tag="labels_only_n50_seed${seed}";  script=scripts/04s_labels_only_filter.py; env_extra="CAL_N=50 FILTER_FRAC=$(GTF $task)";;
    lo100)  tag="labels_only_n100_seed${seed}"; script=scripts/04s_labels_only_filter.py; env_extra="CAL_N=100 FILTER_FRAC=$(GTF $task)";;
    lo400)  tag="labels_only_n400_seed${seed}"; script=scripts/04s_labels_only_filter.py; env_extra="CAL_N=400 FILTER_FRAC=$(GTF $task)";;
    losf1)  tag="labels_only_sf1_seed${seed}";  script=scripts/04s_labels_only_filter.py; env_extra="CAL_N=200 STATE_FRAC=0.01 FILTER_FRAC=$(GTF $task)";;
    losf10) tag="labels_only_sf10_seed${seed}"; script=scripts/04s_labels_only_filter.py; env_extra="CAL_N=200 STATE_FRAC=0.1 FILTER_FRAC=$(GTF $task)";;
    lofix)  tag="labels_only_fixdraw_seed${seed}"; script=scripts/04s_labels_only_filter.py; env_extra="CAL_N=200 LABEL_DRAW_SEED=0 FILTER_FRAC=$(GTF $task)";;
    ls50)   tag="labels_split_s50_seed${seed}";  script=scripts/04s_labels_only_filter.py; env_extra="CAL_N=50 SPLIT_CAL=1";;
    ls100)  tag="labels_split_s100_seed${seed}"; script=scripts/04s_labels_only_filter.py; env_extra="CAL_N=100 SPLIT_CAL=1";;
    ls200)  tag="labels_split_s200_seed${seed}"; script=scripts/04s_labels_only_filter.py; env_extra="CAL_N=200 SPLIT_CAL=1";;
    cpl)    tag="cpl_gt_seed${seed}";  script=scripts/04c_train_cpl_gt.py; env_extra="CPL_LAMBDA=0.5 POLICY_OUT=cpl_cpl_gt_seed${seed}_policy.pt";;
    rand)   tag="vfilt_random_seed${seed}"; script=scripts/04p_vfilter_bc.py; env_extra="SCORE_MODE=random FILTER_FRAC=$(GTF $task)";;
    ret)    tag="vfilt_return_seed${seed}"; script=scripts/04p_vfilter_bc.py; env_extra="SCORE_MODE=return FILTER_FRAC=$(GTF $task)";;
    retbot) tag="vfilt_retbot_seed${seed}"; script=scripts/04p_vfilter_bc.py; env_extra="SCORE_MODE=return_bottom FILTER_FRAC=$(GTF $task)";;
    bcall)  tag="bc";                  script=$W/scripts/bc_on_subset.py; env_extra="KEPT_JSON=outputs/${task}/kept_all.json";;
    bcsafe) tag="bcsafe_seed${seed}";  script=$W/scripts/bc_on_subset.py; env_extra="KEPT_JSON=outputs/${task}/kept_safe.json";;
  esac
  local key="${task}_${tag}"
  [ -f "$LOGDIR/done_${key}" ] && return 0
  env SAFETY_VLM_TASK=$task WANDB_MODE=disabled OMP_NUM_THREADS=3 SEED_OVERRIDE=$seed \
      COST_LIMIT=$lim V_ENSEMBLE_FILE=v_ensemble_pess_seed${seed}.pt OUT_TAG=$tag $env_extra \
    conda run -n safevlmcpl --no-capture-output python "$script" > "$LOGDIR/${key}.log" 2>&1 \
    || { echo "[$(date +%m/%d-%H:%M:%S)] FAIL $key" >> "$LOGDIR/progress.log"; return 1; }
  local pol="bc_${tag}_policy.pt"; [ "$kind" = cpl ] && pol="cpl_${tag}_policy.pt"
  env SAFETY_VLM_TASK=$task WANDB_MODE=disabled OMP_NUM_THREADS=3 CUDA_VISIBLE_DEVICES="" \
    conda run -n safevlmcpl --no-capture-output python scripts/05_evaluate.py \
    --policy_file "$pol" --results_suffix "$tag" >> "$LOGDIR/${key}.log" 2>&1 \
    || { echo "[$(date +%m/%d-%H:%M:%S)] FAIL EVAL $key" >> "$LOGDIR/progress.log"; return 1; }
  touch "$LOGDIR/done_${key}"; echo "[$(date +%m/%d-%H:%M:%S)] DONE $key" >> "$LOGDIR/progress.log"
}
export -f run_arm; export LOGDIR W D

ALL14="pointgoal2 pointbutton1 pointbutton2 carbutton1_t3 carbutton2 pointcircle1 pointcircle2 ballrun_b ballcircle_b carcircle_b carrun_b dronerun_b cargoal2 pointgoal1_dsrl"
ANA3="cargoal2 pointgoal1_dsrl pointgoal2"
NINE="pointgoal2 pointbutton1 pointcircle1 pointcircle2 ballrun_b ballcircle_b carcircle_b carrun_b dronerun_b"
J=$W/stage3b_jobs.txt; : > "$J"
for t in $ALL14; do for s in 0 1 2 3 4; do echo "$t bcsafe $s"; echo "$t rand $s"; done; echo "$t bcall 0"; done >> "$J"
for t in $NINE; do for s in 0 1 2 3 4; do for k in lttR50 pref ret retbot; do echo "$t $k $s"; done; done; done >> "$J"
for t in $ANA3; do for s in 0 1 2; do for k in a10 a40 lo lo50 lo100 lo400 losf1 losf10 lofix ls50 ls100 ls200 cpl; do echo "$t $k $s"; done; done; done >> "$J"
for t in pointgoal2 ballrun_b ballcircle_b carcircle_b dronerun_b; do for s in 0 1 2; do echo "$t tier2 $s"; done; done >> "$J"
log_run "STAGE3b: $(wc -l < $J) jobs (persistent logs at $LOGDIR)"
xargs -a "$J" -L1 -P 8 bash -c 'run_arm "$@"' _
cd /home/omniverse/workspace/safevlmcpl/iclr2027
conda run -n safevlmcpl --no-capture-output python scripts/collect_results.py >> "$LOGDIR/progress.log" 2>&1
log_run "STAGE3B DONE ($(ls $LOGDIR/done_* 2>/dev/null | wc -l) arms)"
