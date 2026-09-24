#!/bin/bash
# A3 (V2_REMEDIATION): prove, per task, that the deployed value ensemble was trained at the
# stated protocol (04n_train_v_only, 300 epochs, batch 512, K=3, current gt_labels.json), by
# retraining seed 0 into a temporary file and comparing weights bit-exactly. Covers the 14
# tasks refreshed 2026-08-16 (their training logs lived in a scratchpad and are gone), the 18
# H-variant directories, and the six pess300 tasks (control: must match by construction).
# Output: runs/probe/ensemble_protocol_verify.json; temp files are deleted.
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
D=${CSC_WORK}; PY=${PYTHON}
OUT=${CSC_RUNS}/probe/ensemble_protocol_verify.json; L=${CSC_RUNS}/logs/protoverify; mkdir -p $L; cd $D
: > $L/results.txt
check () {  # task cfgfile outdir
  local t=$1 cfg=$2 od=$3 CFG=""
  [ "$cfg" != "-" ] && CFG="SAFETY_VLM_CONFIG=$cfg"
  [ -f outputs/$od/v_ensemble_pess_seed0.pt ] || { echo "$od MISSING" >> $L/results.txt; return; }
  env PYTHONNOUSERSITE=1 OMP_NUM_THREADS=3 SAFETY_VLM_TASK=$t $CFG WANDB_MODE=disabled SEED_OVERRIDE=0 V_OUT=v_ensemble_protoverify_seed0.pt \
    $PY scripts/04n_train_v_only.py > $L/$od.log 2>&1 || { echo "$od TRAINFAIL" >> $L/results.txt; return; }
  local r=$(env PYTHONNOUSERSITE=1 $PY -c "
import torch;a=torch.load('outputs/$od/v_ensemble_pess_seed0.pt',map_location='cpu',weights_only=False);b=torch.load('outputs/$od/v_ensemble_protoverify_seed0.pt',map_location='cpu',weights_only=False)
a=a.get('state_dict',a);b=b.get('state_dict',b);print(max((a[k]-b[k]).abs().max().item() for k in a if hasattr(a[k],'shape')))")
  rm -f outputs/$od/v_ensemble_protoverify_seed0.pt
  echo "$od maxdiff=$r $(grep -m1 'V-only:' $L/$od.log)" >> $L/results.txt
}
for t in pointgoal2 pointbutton1 pointbutton2 carbutton1_t3 carbutton2 pointcircle1 pointcircle2 ballrun_b ballcircle_b carcircle_b carrun_b dronerun_b cargoal2 pointgoal1_dsrl \
         halfcheetah_velocity cargoal1_dsrl walker2d_velocity ant_velocity hopper_velocity swimmer_velocity; do check $t - $t; done
for t in halfcheetah_velocity walker2d_velocity ant_velocity hopper_velocity swimmer_velocity cargoal1_dsrl cargoal2 pointgoal1_dsrl pointgoal2; do
  for h in 10 50; do check $t config_h$h.yaml ${t}_h$h; done; done
env PYTHONNOUSERSITE=1 $PY - <<PY
import json,re
rows=[l.split() for l in open("$L/results.txt")]
out={r[0]:{"maxdiff":(float(r[1].split("=")[1]) if "maxdiff" in r[1] else None),"status":r[1] if "maxdiff" not in r[1] else ("BIT-EXACT" if float(r[1].split("=")[1])==0 else "DIFFERS"),"log":" ".join(r[2:])} for r in rows}
json.dump(out,open("$OUT","w"),indent=1)
print({k:v["status"] for k,v in out.items()})
PY
echo PROTOVERIFY DONE
