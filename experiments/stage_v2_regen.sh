#!/bin/bash
# F1 (V2_REMEDIATION, B3 second branch): regenerate every V-dependent policy arm on the six
# old-cohort tasks from the ensembles retrained at the stated protocol (installed by
# v2_install_pess300.sh as v_ensemble_pess_seed{s}.pt). Each family's invocation is the one
# that produced the arm on the nine refreshed tasks (t13t22_queue.sh, stage_r50_complete.sh,
# stage3b.sh, stage_a5_t32t26.sh, stage_reverify.sh), so the six tasks end on the same
# contract as the nine. BC batch is the task config's (512, the stated protocol) unless
# BC_BATCH is set (B4). xlab/cf pin AWR_EPOCHS=50 BATCH_SIZE=1024 AWR_BETA=0.1 (stated in
# App. hyper after C19). Idempotent via done markers; usage: [PAR=n] [ONLY=regex] bash stage_v2_regen.sh
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
W=${CSC_RUNS}
LOGDIR=$W/logs/v2regen
PY=${PYTHON}
mkdir -p "$LOGDIR"; cd "$D"
GTF () { $PY -c "import json;print(json.load(open('$W/gtfracs.json'))['$1']['gt_frac'])"; }
CL () { case $1 in *velocity*) echo 20;; *_b) echo 10;; *) echo 25;; esac; }
T13 () { env PYTHONNOUSERSITE=1 $PY $W/scripts/t13_frac.py $1 $2 | tail -1; }
export -f GTF CL T13; export W PY D LOGDIR
BCB="${BC_BATCH:+BATCH_SIZE=$BC_BATCH}"; export BCB

run_cell () {
  local task=$1 arm=$2 seed=$3
  local lim=$(CL $task) ens="V_ENSEMBLE_FILE=v_ensemble_pess_seed${seed}.pt" script="" extra="" tag=""
  case "$arm" in
    bcall)          tag="bc"; script=$W/scripts/bc_on_subset.py; extra="KEPT_JSON=outputs/${task}/kept_all.json $BCB";;
    bcsafe)         tag="bcsafe_seed$seed"; script=$W/scripts/bc_on_subset.py; extra="KEPT_JSON=outputs/${task}/kept_safe.json $BCB";;
    bcsafeseg)      tag="bcsafeseg_seed$seed"; script=scripts/04b_train_bc.py; extra="BC_SAFE_SEG=1 BC_OUT=bc_bcsafeseg_seed${seed}_policy.pt $BCB";;   # 04b writes BC_OUT, not OUT_TAG (R6)
    vfilt_random)   tag="vfilt_random_matchgt_seed$seed"; script=scripts/04p_vfilter_bc.py; extra="SCORE_MODE=random FILTER_FRAC=$(GTF $task) COST_LIMIT=$lim $BCB";;
    vfilt_return)   tag="vfilt_return_matchgt_seed$seed"; script=scripts/04p_vfilter_bc.py; extra="SCORE_MODE=return FILTER_FRAC=$(GTF $task) COST_LIMIT=$lim $BCB";;
    vfilt_matchgt)  tag="vfilt_matchgt_seed$seed"; script=scripts/04p_vfilter_bc.py; extra="FILTER_FRAC=$(GTF $task) COST_LIMIT=$lim $ens $BCB";;
    vfilt_q25)      tag="vfilt_q25_seed$seed";     script=scripts/04p_vfilter_bc.py; extra="FILTER_FRAC=0.25 COST_LIMIT=$lim $ens $BCB";;
    vfilt_calsafe)  tag="vfilt_calsafe_seed$seed"; script=scripts/04p_vfilter_bc.py; extra="FILTER_FRAC=$(T13 $task $seed) COST_LIMIT=$lim $ens $BCB";;
    xagg_min|xagg_p10) tag="${arm}_seed$seed";     script=scripts/04p_vfilter_bc.py; extra="AGG_MODE=${arm#xagg_} FILTER_FRAC=$(T13 $task $seed) COST_LIMIT=$lim $ens $BCB";;
    calfilt_csf)    tag="calfilt_csf_seed$seed";   script=scripts/04q_calibrated_vfilter.py; extra="MODE=ltt CAL_N=200 ALPHA=0.25 DELTA=0.1 FALLBACK_MODE=calsafe COST_LIMIT=$lim $ens $BCB";;
    calfilt_lttR50) tag="calfilt_lttR50_seed$seed"; script=scripts/04q_calibrated_vfilter.py; extra="MODE=ltt CAL_N=200 ALPHA=0.25 DELTA=0.1 FALLBACK_MODE=calsafe REWARD_FRAC=0.5 COST_LIMIT=$lim $ens $BCB";;
    calfilt_ltt)    tag="calfilt_ltt_seed$seed";   script=scripts/04q_calibrated_vfilter.py; extra="MODE=ltt CAL_N=200 ALPHA=0.25 DELTA=0.1 COST_LIMIT=$lim $ens $BCB";;
    calfilt_a10)    tag="calfilt_a10_seed$seed";   script=scripts/04q_calibrated_vfilter.py; extra="MODE=ltt CAL_N=200 ALPHA=0.10 DELTA=0.1 COST_LIMIT=$lim $ens $BCB";;
    calfilt_a40)    tag="calfilt_a40_seed$seed";   script=scripts/04q_calibrated_vfilter.py; extra="MODE=ltt CAL_N=200 ALPHA=0.40 DELTA=0.1 COST_LIMIT=$lim $ens $BCB";;
    calfilt_tier2)  tag="calfilt_tier2_seed$seed"; script=scripts/04q_calibrated_vfilter.py; extra="MODE=ltt CAL_N=200 ALPHA=0.25 DELTA=0.1 TIER2_DELTA=0.5 COST_LIMIT=$lim $ens $BCB";;
    calfilt_noise20) tag="calfilt_noise20_seed$seed"; script=scripts/04q_calibrated_vfilter.py; extra="MODE=ltt CAL_N=200 ALPHA=0.25 DELTA=0.1 COST_LIMIT=$lim V_ENSEMBLE_FILE=v_ensemble_noise20_seed${seed}.pt $BCB";;
    calfilt_pref)   tag="calfilt_pref_seed$seed";  script=scripts/04q_calibrated_vfilter.py; extra="MODE=pref PREF_Q=0.9 CAL_N=200 ALPHA=0.25 DELTA=0.1 COST_LIMIT=$lim $ens $BCB";;
    xlab_*)         tag="${arm}_seed$seed";        script=scripts/04m_extraction_lab.py
                    local rest=${arm#xlab_learned_}; local mode=${rest%%_*}; local k=${rest#*_k}
                    extra="ADV_SOURCE=learned WEIGHT_MODE=$mode AWR_K_STEP=$k AWR_EPOCHS=50 BATCH_SIZE=1024 AWR_BETA=0.1 $ens";;
    vawr_from_bcsafe) tag="xlab_vawr_from_bcsafe_seed$seed"; script=scripts/04m_extraction_lab.py
                    extra="ADV_SOURCE=learned WEIGHT_MODE=exp AWR_K_STEP=1 AWR_EPOCHS=50 BATCH_SIZE=1024 AWR_BETA=0.1 $ens BC_POLICY=bc_bcsafe_seed${seed}_policy.pt";;
    cf_v2)          tag="cfv2_M16_seed$seed";      script=scripts/04o_cf_extraction.py; extra="M_CANDIDATES=16 AWR_EPOCHS=50 BATCH_SIZE=1024 $ens";;
    *) echo "NO INVOCATION for $arm"; return 1;;
  esac
  case "$tag" in bc|*seed${seed}) ;; *) echo "FATAL tag $tag lacks seed"; return 1;; esac
  local key="${task}_${tag}"
  [ -f "$LOGDIR/done_$key" ] && return 0
  env PYTHONNOUSERSITE=1 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 SAFETY_VLM_TASK=$task WANDB_MODE=disabled \
      SEED_OVERRIDE=$seed OUT_TAG="$tag" $extra $PY "$script" > "$LOGDIR/$key.log" 2>&1 \
    || { echo "[$(date +%m/%d-%H:%M)] TRAIN FAIL $key" >> "$LOGDIR/progress.log"; return 1; }
  env PYTHONNOUSERSITE=1 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 SAFETY_VLM_TASK=$task WANDB_MODE=disabled CUDA_VISIBLE_DEVICES="" \
      SEED_OVERRIDE=$seed $PY scripts/05_evaluate.py --policy_file "bc_${tag}_policy.pt" --results_suffix "$tag" >> "$LOGDIR/$key.log" 2>&1 \
    && { touch "$LOGDIR/done_$key"; echo "[$(date +%m/%d-%H:%M)] DONE $key" >> "$LOGDIR/progress.log"; } \
    || echo "[$(date +%m/%d-%H:%M)] EVAL FAIL $key" >> "$LOGDIR/progress.log"
}
export -f run_cell

SIX="halfcheetah_velocity cargoal1_dsrl walker2d_velocity ant_velocity hopper_velocity swimmer_velocity"
NINE="cargoal2 pointgoal1_dsrl pointgoal2 pointbutton1 pointbutton2 carbutton1_t3 carbutton2 pointcircle1 pointcircle2"
# P0: BC arms that read no V but were trained at batch 1024 (July config) on the six tasks;
# xlab starts from bc_policy.pt and vawr_from_bcsafe from bcsafe, so P0 completes before P1.
J0=$LOGDIR/jobs_p0.txt; : > "$J0"
for t in $SIX; do echo "$t bcall 0"; for s in 0 1 2 3 4; do for a in bcsafe bcsafeseg vfilt_random vfilt_return; do echo "$t $a $s"; done; done; done >> "$J0"
echo "pointbutton1 bcall 0" >> "$J0"; echo "pointbutton1 bcsafeseg 0" >> "$J0"   # R6: its bc_policy.pt was overwritten by the 08-27 bcsafeseg rerun
J=$LOGDIR/jobs.txt; : > "$J"
for t in $SIX; do
  for s in 0 1 2 3 4; do for a in calfilt_csf calfilt_lttR50 vfilt_calsafe vfilt_matchgt vfilt_q25 calfilt_pref; do echo "$t $a $s"; done; done
  for s in 0 1 2; do for a in calfilt_ltt calfilt_a10 calfilt_a40 xagg_min xagg_p10 xlab_learned_rank_k1 xlab_learned_binary_k1 xlab_learned_exp_k1 xlab_learned_exp_k5 vawr_from_bcsafe cf_v2; do echo "$t $a $s"; done; done
done >> "$J"
# calfilt_pref on the nine (its 08-27 reruns were pinned to batch 1024 for reproduction; the
# stated protocol is 512, which B4 showed every other BC arm of the Aug cohort used)
for t in $NINE; do for s in 0 1 2 3 4; do echo "$t calfilt_pref $s"; done; done >> "$J"
for t in halfcheetah_velocity walker2d_velocity hopper_velocity pointgoal1_dsrl; do for s in 0 1 2; do echo "$t calfilt_noise20 $s"; done; done >> "$J"
for t in hopper_velocity swimmer_velocity; do for s in 0 1 2; do echo "$t calfilt_tier2 $s"; done; done >> "$J"
for s in 0 1 2; do echo "halfcheetah_velocity xlab_learned_exp_k3 $s"; echo "halfcheetah_velocity xlab_learned_exp_k10 $s"; done >> "$J"
if [ -n "${ONLY:-}" ]; then grep -E "$ONLY" "$J0" > "$J0.sel"; J0="$J0.sel"; grep -E "$ONLY" "$J" > "$J.sel"; J="$J.sel"; fi
echo "[$(date +%m/%d-%H:%M)] V2 REGEN P0: $(wc -l < $J0) cells, P1: $(wc -l < $J) cells, PAR=${PAR:-3}, BC_BATCH=${BC_BATCH:-config}" >> "$LOGDIR/progress.log"
xargs -a "$J0" -L1 -P "${PAR:-3}" bash -c 'run_cell $0 $1 $2'
echo "[$(date +%m/%d-%H:%M)] V2 REGEN P0 COMPLETE" >> "$LOGDIR/progress.log"
xargs -a "$J" -L1 -P "${PAR:-3}" bash -c 'run_cell $0 $1 $2'
echo "[$(date +%m/%d-%H:%M)] V2 REGEN F1 COMPLETE ($(ls $LOGDIR/done_* 2>/dev/null | wc -l) done)" >> "$LOGDIR/progress.log"
