#!/bin/bash
# Probe 11: 2 tasks x 4 coverage levels x 3 arms x 5 seeds = 120 BC runs, then summarize.
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
S=${CSC_RUNS}/selections_recon
L=${CSC_RUNS}/logs/cov
PY=${PYTHON}
mkdir -p "$L"; cd "$D"
for task in pointgoal1_dsrl cargoal1_dsrl; do
 for lv in 05 15 30 50; do
  for arm in mix swap thr; do
   for s in 0 1 2 3 4; do
    tag="${task}_cov${lv}_${arm}"; K="$S/${tag}_kept.json"
    [ -f "$K" ] || continue
    if [ ! -f "$D/outputs/$task/bc_cv${lv}${arm}_s${s}_policy.pt" ]; then
      env PYTHONNOUSERSITE=1 SAFETY_VLM_TASK=$task SEED_OVERRIDE=$s WANDB_MODE=disabled \
        KEPT_JSON=$K OUT_TAG=cv${lv}${arm}_s${s} \
        $PY scripts/04s_bc_on_subset.py > "$L/tr_${tag}_$s.log" 2>&1 || continue
    fi
    [ -f "$D/outputs/$task/eval_results_cv${lv}${arm}_s${s}.json" ] && continue
    env PYTHONNOUSERSITE=1 SAFETY_VLM_TASK=$task SEED_OVERRIDE=$s WANDB_MODE=disabled \
      $PY scripts/05_evaluate.py --policy_file bc_cv${lv}${arm}_s${s}_policy.pt \
      --results_suffix cv${lv}${arm}_s${s} > "$L/ev_${tag}_$s.log" 2>&1
   done
  done
 done
done
env PYTHONNOUSERSITE=1 $PY - <<'PYEOF'
import json,glob,os,numpy as np
from scipy import stats
D="${CSC_WORK}/outputs"
S="${CSC_RUNS}/selections_recon"
res={}
print("\n=== PROBE 11: does COVERAGE cause utility? (mix vs swap = the causal test) ===")
for task in ("pointgoal1_dsrl","cargoal1_dsrl"):
    print(f"\n  --- {task} (budget 25) ---")
    print(f"  {'lvl':>4s} {'n':>5s} {'contam':>7s} {'cov(mix)':>9s} | "
          f"{'mix R':>12s} {'swap R':>12s} {'dR':>7s} {'p':>6s} | {'mix C':>12s} {'swap C':>12s} {'dC':>7s}")
    for lv in ("05","15","30","50"):
        row={}
        for arm in ("mix","swap","thr"):
            sp=os.path.join(S,f"{task}_cov{lv}_{arm}_kept.json")
            if not os.path.exists(sp): continue
            sel=json.load(open(sp))
            fs=sorted(glob.glob(os.path.join(D,task,f"eval_results_cv{lv}{arm}_s*.json")))
            if not fs: continue
            row[arm]={"R":[json.load(open(f))["avg_reward"] for f in fs],
                      "C":[json.load(open(f))["avg_cost"] for f in fs],
                      "n":sel["n_kept"],"contam":sel["contamination"],"cov":sel["hi_return_safe_frac"]}
        if "mix" not in row or "swap" not in row: continue
        m,w=row["mix"],row["swap"]
        h=lambda v:1.96*np.std(v,ddof=1)/len(v)**.5 if len(v)>1 else 0.0
        dR=np.mean(m["R"])-np.mean(w["R"]); dC=np.mean(m["C"])-np.mean(w["C"])
        p=stats.ttest_ind(m["R"],w["R"],equal_var=False).pvalue
        print(f"  {lv:>4s} {m['n']:5d} {m['contam']:7.4f} {m['cov']:9.4f} | "
              f"{np.mean(m['R']):7.2f}+-{h(m['R']):4.2f} {np.mean(w['R']):7.2f}+-{h(w['R']):4.2f} "
              f"{dR:+7.2f} {p:6.3f} | {np.mean(m['C']):7.2f}+-{h(m['C']):4.2f} "
              f"{np.mean(w['C']):7.2f}+-{h(w['C']):4.2f} {dC:+7.2f}")
        res.setdefault(task,{})[lv]=row
    # dose-response: does the mix-swap reward gap grow with coverage?
    lv=[l for l in res.get(task,{})]
    if len(lv)>2:
        x=[res[task][l]["mix"]["cov"] for l in lv]
        y=[np.mean(res[task][l]["mix"]["R"])-np.mean(res[task][l]["swap"]["R"]) for l in lv]
        r=stats.spearmanr(x,y)
        print(f"    dose-response Spearman(coverage, mix-swap reward gap) = {r.statistic:+.3f} (p={r.pvalue:.3f})")
json.dump(res,open("${CSC_RUNS}/probe/coverage_doseresponse.json","w"),indent=2)
print("\n  wrote runs/probe/coverage_doseresponse.json")
PYEOF
