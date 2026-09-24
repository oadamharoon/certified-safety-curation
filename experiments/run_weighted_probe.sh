#!/bin/bash
# Probe 10: matched-composition mixture vs threshold. Train, eval, summarize.
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
L=${CSC_RUNS}/logs/wprobe
PY=${PYTHON}
mkdir -p "$L"; cd "$D"
for arm in mix005 thr005 mix010 thr010; do
  for s in 0 1 2; do
    tag="${arm}_s${s}"
    if [ ! -f "$D/outputs/pointgoal1_dsrl/bc_w${tag}_policy.pt" ]; then
      env PYTHONNOUSERSITE=1 SAFETY_VLM_TASK=pointgoal1_dsrl SEED_OVERRIDE=$s WANDB_MODE=disabled \
        KEPT_JSON=$S/pointgoal1_dsrl_${arm}_kept.json OUT_TAG=w${tag} \
        $PY scripts/04s_bc_on_subset.py > "$L/train_$tag.log" 2>&1 || { echo "TRAIN FAIL $tag"; continue; }
    fi
    [ -f "$D/outputs/pointgoal1_dsrl/eval_results_w${tag}.json" ] && continue
    env PYTHONNOUSERSITE=1 SAFETY_VLM_TASK=pointgoal1_dsrl SEED_OVERRIDE=$s WANDB_MODE=disabled \
      $PY scripts/05_evaluate.py --policy_file bc_w${tag}_policy.pt \
      --results_suffix w${tag} > "$L/eval_$tag.log" 2>&1 || echo "EVAL FAIL $tag"
  done
done
env PYTHONNOUSERSITE=1 $PY - <<'PYEOF'
import json,glob,os,numpy as np
D="${CSC_WORK}/outputs/pointgoal1_dsrl"
S="${CSC_RUNS}/selections_recon"
out={}
print("\n=== PROBE 10: does weighting-recovered coverage buy reward at equal cost? ===")
print("    (PointGoal1, budget 25, 3 seeds; each mix/thr pair matched on n AND contamination)")
print(f"  {'arm':8s} {'n':>5s} {'contam':>8s} {'coverage':>9s} {'reward':>15s} {'cost':>15s}")
for arm in ("mix005","thr005","mix010","thr010"):
    sel=json.load(open(os.path.join(S,f"pointgoal1_dsrl_{arm}_kept.json")))
    fs=sorted(glob.glob(os.path.join(D,f"eval_results_w{arm}_s*.json")))
    if not fs: print(f"  {arm:8s} (no results)"); continue
    R=[json.load(open(f))["avg_reward"] for f in fs]; C=[json.load(open(f))["avg_cost"] for f in fs]
    h=lambda v:1.96*np.std(v,ddof=1)/len(v)**.5 if len(v)>1 else 0.0
    out[arm]={"R":R,"C":C,"contam":sel["contamination"],"cov":sel["hi_return_safe_frac"],"n":sel["n_kept"]}
    print(f"  {arm:8s} {sel['n_kept']:5d} {sel['contamination']:8.4f} {sel['hi_return_safe_frac']:9.4f} "
          f"{np.mean(R):8.2f}+-{h(R):4.2f} {np.mean(C):8.2f}+-{h(C):4.2f}")
json.dump(out,open("${CSC_RUNS}/probe/weighted.json","w"),indent=2)
for f in ("005","010"):
    if f"mix{f}" in out and f"thr{f}" in out:
        m,t=out[f"mix{f}"],out[f"thr{f}"]
        dR=np.mean(m["R"])-np.mean(t["R"]); dC=np.mean(m["C"])-np.mean(t["C"])
        print(f"\n  pair {f}: mixture minus threshold  reward {dR:+.2f}  cost {dC:+.2f}")
print("\n  wrote runs/probe/weighted.json")
PYEOF
