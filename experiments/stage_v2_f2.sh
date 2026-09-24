#!/bin/bash
# F2 (V2_REMEDIATION): composability and operator arms on the regenerated certified selections of
# the six old-cohort tasks (runs/selections/v2_summary.json from build_v2_selections.py).
# Per task with a certifying alpha=.25 seed vs: CDT (3 seeds) on <task>_cert_seed{vs} and on the
# three distinct a25new_q{q} selections; CPL (3 seeds) on the same, with its 1000 pairs re-drawn
# inside each selection (03b SUBSET_KEPT, as stage_cpl_compose.sh); the operator grid
# (wbc/wc1/wc3/th, 3 seeds) on the three a25new selections. Per task with a certifying alpha=.40
# seed: CDT + CPL on <task>_a40_seed{vs} and CDT + CPL + BC on the three a40new_q{q} selections.
# Harvester naming: cert_seed -> cdt_cert, a25new_q -> cdt_a25new_q, a40_seed -> cdt_a40,
# a40new_q -> cdt_a40new_q (iclr2027/scripts/harvest_osrl.py keeps the latest run per key).
# Idempotent via done markers in runs/logs/v2f2. Usage: [CDT_PAR=n] [BC_PAR=n] bash stage_v2_f2.sh
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
W=${CSC_WORKSPACE}; D=$W/vlm-with-cpl/new_data; SEL=$W/runs/selections; L=$W/runs/logs/v2f2
PY=${PYTHON}; mkdir -p $L/runs; cd $D
T2ENV () { case $1 in
  halfcheetah_velocity) echo "OfflineHalfCheetahVelocityGymnasium-v1:20";; walker2d_velocity) echo "OfflineWalker2dVelocityGymnasium-v1:20";;
  ant_velocity) echo "OfflineAntVelocityGymnasium-v1:20";; hopper_velocity) echo "OfflineHopperVelocityGymnasium-v1:20";;
  swimmer_velocity) echo "OfflineSwimmerVelocityGymnasium-v1:20";; cargoal1_dsrl) echo "OfflineCarGoal1Gymnasium-v0:25";; esac; }
export -f T2ENV; export W D SEL L PY

run_cdt () {   # task h5name seed
  local task=$1 h5=$2 seed=$3; IFS=: read e lim <<< "$(T2ENV $task)"; local tag="cdt_${h5}_s${seed}"
  [ -f "$L/done_$tag" ] && return 0
  cd $W/osrl && env PYTHONNOUSERSITE=1 PYTHONPATH=$W/osrl $PY examples/train/train_cdt.py --task "$e" --seed "$seed" --cost_limit "$lim" \
    --device cuda --augment_percent 0.0 --random_aug 0.0 --subset_h5 "$SEL/$h5.hdf5" --logdir "$L/runs" > "$L/$tag.log" 2>&1 \
    && { touch "$L/done_$tag"; echo "[$(date +%m/%d-%H:%M)] DONE $tag" >> $L/progress.log; } || echo "[$(date +%m/%d-%H:%M)] FAIL $tag" >> $L/progress.log
}
run_cpl () {   # task selname resulttag seed   (resulttag e.g. cplgt_cert, cplgt_a25q85, cplgt_a40, cplgta40new_q65)
  local task=$1 sel=$2 rt=$3 seed=$4; local tag="${rt}_seed${seed}"; local key="${task}_${tag}"
  [ -f "$L/done_$key" ] && return 0
  cd $D; local lbl="gt_labels_${sel}.json"
  [ -f "outputs/$task/$lbl" ] || env PYTHONNOUSERSITE=1 SAFETY_VLM_TASK=$task SUBSET_KEPT=$SEL/${sel}_kept.json LABELS_OUT=$lbl $PY scripts/03b_label_by_cost.py > "$L/labels_${task}_${sel}.log" 2>&1 \
    || { echo "[$(date +%m/%d-%H:%M)] LABEL FAIL $task $sel" >> $L/progress.log; return 1; }
  env PYTHONNOUSERSITE=1 OMP_NUM_THREADS=3 SAFETY_VLM_TASK=$task WANDB_MODE=disabled SEED_OVERRIDE=$seed SUBSET_KEPT=$SEL/${sel}_kept.json LABELS_IN=$lbl POLICY_OUT=cpl_${tag}.pt \
    $PY scripts/04c_train_cpl_gt.py > "$L/$key.log" 2>&1 || { echo "[$(date +%m/%d-%H:%M)] TRAIN FAIL $key" >> $L/progress.log; return 1; }
  env PYTHONNOUSERSITE=1 OMP_NUM_THREADS=3 SAFETY_VLM_TASK=$task WANDB_MODE=disabled CUDA_VISIBLE_DEVICES="" SEED_OVERRIDE=$seed \
    $PY scripts/05_evaluate.py --policy_file cpl_${tag}.pt --results_suffix $tag >> "$L/$key.log" 2>&1 \
    && { touch "$L/done_$key"; echo "[$(date +%m/%d-%H:%M)] DONE $key" >> $L/progress.log; } || echo "[$(date +%m/%d-%H:%M)] EVAL FAIL $key" >> $L/progress.log
}
run_bc () {    # task selname resulttag seed extra-env   (bc_on_subset; resulttag e.g. bca40new_q65, wbcq85, wc1q85, wc3q85, thq85)
  local task=$1 sel=$2 rt=$3 seed=$4 extra="${5:-}"; local tag="${rt}_seed${seed}"; local key="${task}_${tag}"
  [ -f "$L/done_$key" ] && return 0
  cd $D; env PYTHONNOUSERSITE=1 OMP_NUM_THREADS=3 SAFETY_VLM_TASK=$task WANDB_MODE=disabled SEED_OVERRIDE=$seed KEPT_JSON=$SEL/${sel}_kept.json OUT_TAG=$tag $extra \
    $PY $W/runs/scripts/bc_on_subset.py > "$L/$key.log" 2>&1 || { echo "[$(date +%m/%d-%H:%M)] TRAIN FAIL $key" >> $L/progress.log; return 1; }
  env PYTHONNOUSERSITE=1 OMP_NUM_THREADS=3 SAFETY_VLM_TASK=$task WANDB_MODE=disabled CUDA_VISIBLE_DEVICES="" SEED_OVERRIDE=$seed \
    $PY scripts/05_evaluate.py --policy_file bc_${tag}_policy.pt --results_suffix $tag >> "$L/$key.log" 2>&1 \
    && { touch "$L/done_$key"; echo "[$(date +%m/%d-%H:%M)] DONE $key" >> $L/progress.log; } || echo "[$(date +%m/%d-%H:%M)] EVAL FAIL $key" >> $L/progress.log
}
export -f run_cdt run_cpl run_bc

# ---- job lists from v2_summary.json
JC=$L/jobs_cdt.txt; JB=$L/jobs_bc.txt; : > $JC; : > $JB
$PY - <<EOF >> /dev/null
import json
S=json.load(open("$SEL/v2_summary.json")); jc=open("$JC","w"); jb=open("$JB","w")
for task,rec in S.items():
    a25=rec.get("a25"); a40=rec.get("a40")
    if a25:
        dep=a25["deployed"]["name"]
        for s in range(3):
            jc.write(f"{task} {dep} {s}\n"); jb.write(f"cpl {task} {dep} cplgt_cert {s}\n")
            for d in a25["distinct"]:
                q=int(d["q"]*100); jc.write(f"{task} {d['name']} {s}\n"); jb.write(f"cpl {task} {d['name']} cplgt_a25q{q} {s}\n")
                for rt,ex in ((f"wbcq{q}","RETURN_WEIGHTED=1 WEIGHT_CLIP=2.0"),(f"wc1q{q}","RETURN_WEIGHTED=1 WEIGHT_CLIP=1.0"),(f"wc3q{q}","RETURN_WEIGHTED=1 WEIGHT_CLIP=3.0"),(f"thq{q}","RETURN_TOPHALF=1")):
                    jb.write(f"bc {task} {d['name']} {rt} {s} {ex}\n")
    if a40:
        dep=a40["deployed"]["name"]
        for s in range(3):
            jc.write(f"{task} {dep} {s}\n"); jb.write(f"cpl {task} {dep} cplgt_a40 {s}\n")
            for d in a40["distinct"]:
                q=int(d["q"]*100); jc.write(f"{task} {d['name']} {s}\n"); jb.write(f"cpl {task} {d['name']} cplgta40new_q{q} {s}\n"); jb.write(f"bc {task} {d['name']} bca40new_q{q} {s}\n")
EOF
echo "[$(date +%m/%d-%H:%M)] F2: $(wc -l < $JC) CDT cells, $(wc -l < $JB) BC/CPL cells; CDT_PAR=${CDT_PAR:-2} BC_PAR=${BC_PAR:-3}" >> $L/progress.log
dispatch () { local k=$1; shift; if [ "$k" = cpl ]; then run_cpl "$1" "$2" "$3" "$4"; else run_bc "$1" "$2" "$3" "$4" "${5:-} ${6:-}"; fi; }
export -f dispatch
xargs -a "$JB" -L1 -P "${BC_PAR:-3}" bash -c 'dispatch "$@"' _ &
xargs -a "$JC" -L1 -P "${CDT_PAR:-2}" bash -c 'run_cdt "$@"' _ &
wait
cd $W/iclr2027 && env PYTHONNOUSERSITE=1 $PY scripts/harvest_osrl.py >> $L/progress.log 2>&1
echo "[$(date +%m/%d-%H:%M)] F2 COMPLETE ($(ls $L/done_* | wc -l) done markers)" >> $L/progress.log
