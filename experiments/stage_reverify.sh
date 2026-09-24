#!/bin/bash
# Re-run the paper-consumed arms whose results predate an input they genuinely
# read, so every published number is reproducible from what is on disk now.
#
# Each family's invocation was recovered from what actually produced the current
# results, not from a runner script's claims:
#   labels_only*  04s_labels_only_filter.py; CAL_N from its meta, FILTER_FRAC
#                 from runs/gtfracs.json (verified to match all 27 metas)
#   xlab_*        04m_extraction_lab.py; the tag IS the config,
#                 xlab_<ADV_SOURCE>_<WEIGHT_MODE>_k<AWR_K_STEP>_seed<n>
#   bcsafeseg     04b_train_bc.py with BC_SAFE_SEG=1 (zero-cost segments)
#   cf_v1         04o at documented defaults (M=8) WITH
#                 V_ENSEMBLE_FILE=v_ensemble_pess_seed<n>.pt. 04o trains a fresh
#                 ensemble when that variable is unset, and every original cf run
#                 loaded one (verified in wandb), so omitting it would silently
#                 substitute a different experiment.
#   cf_octg       04o with V_ENSEMBLE_FILE=v_ensemble_octg_seed<n>.pt
#
#   ALL xlab_*/cf_* additionally pin AWR_EPOCHS=50 BATCH_SIZE=1024 (and
#   AWR_BETA=0.1 for xlab). The task config was edited after these arms ran:
#   awr_epochs and awr_beta were removed and batch_size halved, so a rerun on
#   today's config silently trains 6x longer at a different beta. The original
#   values come from each run's wandb config.yaml, which snapshots cfg after
#   env overrides. Verified identical across all six originals.
#   cf_v2         REDEFINED: identical to cf_v1 except M_CANDIDATES=16, the one
#                 parameter its surviving tag attests. Its original non-default
#                 settings were recorded nowhere (init_wandb logs only the yaml
#                 config, the printed config line never reached output.log, and
#                 no runner survives), so the arm is restated explicitly rather
#                 than guessed. Documented in the hyperparameter appendix.
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
D=${CSC_WORK}
W=${CSC_RUNS}
LOGDIR=$W/logs/reverify
PY=${PYTHON}
mkdir -p "$LOGDIR"; cd "$D"

GTF () { $PY -c "import json;print(json.load(open('$W/gtfracs.json'))['$1']['gt_frac'])" 2>/dev/null; }
CL () { $PY -c "import yaml;c=yaml.safe_load(open('$D/config.yaml'));\
print(c.get('tasks',{}).get('$1',{}).get('cost_limit', c.get('cost_limit')))" 2>/dev/null; }
export -f GTF CL; export W PY D

run_cell () {
  local task=$1 arm=$2 tag=$3 seed=$4
  # Every per-run artifact below is named from $tag, and cells run concurrently
  # under xargs -P. A tag that does not embed its seed makes two seeds write the
  # same policy/results file, which is how an earlier stage had both seeds read
  # whichever policy saved last. Refuse the cell rather than race.
  case "$tag" in
    *seed${seed}*|*_s${seed}|*_s${seed}_*) ;;
    *) echo "FATAL: tag '$tag' does not embed seed '$seed'; refusing" >&2; return 1;;
  esac
  local key="${task}_${tag}"
  [ -f "$LOGDIR/done_$key" ] && { echo "skip $key"; return 0; }
  local script="" extra=""
  case "$arm" in
    labels_only)          script=scripts/04s_labels_only_filter.py; extra="CAL_N=200 FILTER_FRAC=$(GTF $task)";;
    labels_only_n50)      script=scripts/04s_labels_only_filter.py; extra="CAL_N=50 FILTER_FRAC=$(GTF $task)";;
    labels_only_n100)     script=scripts/04s_labels_only_filter.py; extra="CAL_N=100 FILTER_FRAC=$(GTF $task)";;
    labels_only_n400)     script=scripts/04s_labels_only_filter.py; extra="CAL_N=400 FILTER_FRAC=$(GTF $task)";;
    bcsafeseg)            script=scripts/04b_train_bc.py;           extra="BC_SAFE_SEG=1";;
    xlab_*)               script=scripts/04m_extraction_lab.py
                          # the originals LOADED a saved ensemble (162 runs) rather than
                          # training a fresh one (3). Omitting V_ENSEMBLE_FILE makes 04m
                          # train its own 300-epoch ensemble, which is both a different
                          # experiment and the reason the cells ran ~5x longer.
                          # tag: xlab_<src>_<mode>_k<k>_seed<n>
                          local rest=${tag#xlab_}; local src="" mode="" k=""
                          case "$rest" in
                            oracle_step_*) src=oracle_step; rest=${rest#oracle_step_};;
                            oracle_ctg_*)  src=oracle_ctg;  rest=${rest#oracle_ctg_};;
                            learned_*)     src=learned;     rest=${rest#learned_};;
                          esac
                          mode=${rest%%_*}; rest=${rest#*_}; k=${rest%%_*}; k=${k#k}
                          # config.yaml drifted after these runs: awr_epochs (50) and
                          # awr_beta (0.1) were REMOVED from the task config and
                          # batch_size changed 1024 -> 512, so today's resolved cfg
                          # gives 300 epochs at beta 0.5 on half the batch. Recovered
                          # from each original run's wandb config.yaml, which records
                          # cfg AFTER the env overrides are applied. Without these the
                          # rerun is a different experiment, not a failed reproduction.
                          extra="ADV_SOURCE=$src WEIGHT_MODE=$mode AWR_K_STEP=$k AWR_EPOCHS=50 BATCH_SIZE=1024 AWR_BETA=0.1"
                          # oracle sources derive the advantage from ground-truth cost and
                          # never touch the learned value, so only 'learned' needs the file
                          [ "$src" = learned ] && extra="$extra V_ENSEMBLE_FILE=v_ensemble_pess_seed${seed}.pt";;
    calfilt_pref)         script=scripts/04q_calibrated_vfilter.py
                          # Held-out-preference calibration control. Same config
                          # drift as the xlab/cf families: batch_size was 1024 when
                          # these ran and is 512 today, and bc_only_epochs is absent
                          # from some task configs. Both recovered from the originals'
                          # wandb config.yaml; all 15 tasks ran 100 epochs at 1024.
                          # 04q has no default for V_ENSEMBLE_FILE, so it is explicit.
                          extra="MODE=pref PREF_Q=0.9 CAL_N=200 ALPHA=0.25 DELTA=0.1 \
BC_EPOCHS=100 BATCH_SIZE=1024 COST_LIMIT=$(CL $task) \
V_ENSEMBLE_FILE=v_ensemble_pess_seed${seed}.pt";;
    cpl_gt)               script=scripts/04c_train_cpl_gt.py
                          # CPL on our exact preference data. Reads gt_labels.json
                          # directly, which was regenerated 08-16, so these predate
                          # their own labels. No config drift: originals ran at
                          # cpl_epochs=300 bc_pretrain_epochs=50 batch_size=512
                          # lambda_bc=0.1 temperature_cpl=1, all still current, and
                          # gt_labels.json still holds 1000 pairs as it did then.
                          # POLICY_OUT must carry the seed: the default filename is
                          # shared and concurrent seeds would overwrite each other.
                          extra="POLICY_OUT=bc_${tag}_policy.pt";;
    vawr_from_bcsafe)     script=scripts/04m_extraction_lab.py
                          # 04m started from the BC-Safe policy instead of BC-All.
                          # Recovered from wandb run 751r8hzz: mode=exp, k=1,
                          # V_ENSEMBLE_FILE=v_ensemble_pess_seed<n>.pt,
                          # BC_POLICY=bc_bcsafe_seed<n>_policy.pt.
                          extra="ADV_SOURCE=learned WEIGHT_MODE=exp AWR_K_STEP=1 \
AWR_EPOCHS=50 BATCH_SIZE=1024 AWR_BETA=0.1 \
V_ENSEMBLE_FILE=v_ensemble_pess_seed${seed}.pt \
BC_POLICY=bc_bcsafe_seed${seed}_policy.pt";;
    qfilt)                script=scripts/04t_q_ablation.py
                          # Q(s,a) ablation. Trains its own action-conditioned
                          # ensemble from the preference labels, so it reads no
                          # saved checkpoint; FILTER_FRAC is required and is the
                          # task's ground-truth safe fraction, matching the
                          # original (kept 1047/2022 on pointgoal1). No config
                          # drift here: these ran 07-20, after batch_size changed.
                          extra="FILTER_FRAC=$(GTF $task)";;
    cf_v1)                script=scripts/04o_cf_extraction.py;      extra="M_CANDIDATES=8 AWR_EPOCHS=50 BATCH_SIZE=1024 V_ENSEMBLE_FILE=v_ensemble_pess_seed${seed}.pt";;
    cf_v2)                script=scripts/04o_cf_extraction.py;      extra="M_CANDIDATES=16 AWR_EPOCHS=50 BATCH_SIZE=1024 V_ENSEMBLE_FILE=v_ensemble_pess_seed${seed}.pt";;
    cf_octg)              script=scripts/04o_cf_extraction.py;      extra="M_CANDIDATES=8 V_ENSEMBLE_FILE=v_ensemble_octg_seed${seed}.pt AWR_EPOCHS=50 BATCH_SIZE=1024";;
    *) echo "[$(date +%H:%M:%S)] NO INVOCATION for arm=$arm ($key)"; return 1;;
  esac
  # OMP_NUM_THREADS=3 matches every original runner. Left unset, torch spawns
  # ~63 threads per process; four concurrent cells oversubscribed 32 cores and
  # ran 12-30x slower than the originals' 0.44h/1.07h medians. Thread count also
  # changes float reduction order, so matching it is required for reproduction.
  env PYTHONNOUSERSITE=1 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 \
    SAFETY_VLM_TASK=$task SEED_OVERRIDE=$seed OUT_TAG="$tag" $extra \
    $PY "$script" > "$LOGDIR/$key.log" 2>&1 || {
      echo "[$(date +%H:%M:%S)] TRAIN FAIL $key"; return 1; }
  env PYTHONNOUSERSITE=1 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 SAFETY_VLM_TASK=$task SEED_OVERRIDE=$seed \
    $PY scripts/05_evaluate.py --policy_file "bc_${tag}_policy.pt" \
    --results_suffix "$tag" >> "$LOGDIR/$key.log" 2>&1 && {
      touch "$LOGDIR/done_$key"; echo "[$(date +%H:%M:%S)] DONE $key"; } || {
      echo "[$(date +%H:%M:%S)] EVAL FAIL $key"; }
}
export -f run_cell; export LOGDIR D PY

J=${JOBS_FILE:-$LOGDIR/jobs.txt}
echo "reverify jobs: $(wc -l < "$J")"
# Concurrency is a knob because it depends on who else is on the box: 4 while
# sharing, more when idle. Memory is the binding constraint, not the GPU.
xargs -a "$J" -L1 -P "${PAR:-4}" bash -c 'run_cell $0 $1 $2 $3'
echo "REVERIFY STAGE COMPLETE"
