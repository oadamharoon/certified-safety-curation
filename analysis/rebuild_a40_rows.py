"""Regenerate the alpha=0.40 distinct-selection rows for the three tasks whose
V ensembles were retrained on 2026-08-16 (cargoal2, pointgoal1_dsrl, pointgoal2).

Their published grid selections came from ensembles that no longer exist, so
4 of the 9 cells cannot be reproduced at all and the rest only by drawing across
mixed V seeds. This re-derives each row the way build_selections.py does: resample
calibration draws, keep the three most probable DISTINCT certified thresholds,
ordered by realized contamination. All three learners then train on these.

Output goes to runs/selections/, NOT a scratchpad.
"""

# --- paths ------------------------------------------------------------------
# The research tree addressed itself by absolute path; these roots replace it. Set
# CSC_WORKSPACE (or the individual roots) to point at your own trees. See the README.
import os as _os


def _csc_root(_p):
    """The repository root, found by the .csc-root marker rather than by depth."""
    _d = _os.path.dirname(_os.path.abspath(_p))
    while True:
        if _os.path.exists(_os.path.join(_d, ".csc-root")):
            return _d
        _up = _os.path.dirname(_d)
        if _up == _d:
            return _os.path.dirname(_os.path.dirname(_os.path.abspath(_p)))
        _d = _up


# __file__ is undefined when a script's source is exec'd in a fresh namespace, which the
# audit does to reuse the table builder's tables; fall back to the working directory, which
# the .csc-root walk resolves from anywhere inside the repository.
_self = globals().get("__file__") or _os.path.join(_os.getcwd(), "_")
CSC_REPO = _os.environ.get("CSC_REPO", _csc_root(_self))
_WS = _os.environ.get("CSC_WORKSPACE", _os.path.dirname(CSC_REPO))
CSC_WORK = _os.environ.get("CSC_WORK", _os.path.join(_WS, "vlm-with-cpl", "new_data"))
_runs = _os.path.join(_WS, "runs")
CSC_RUNS = _os.environ.get("CSC_RUNS", _runs if _os.path.isdir(_runs) else _os.path.join(CSC_REPO, "runs"))
CSC_OSRL = _os.environ.get("CSC_OSRL", _os.path.join(_WS, "osrl"))
CSC_PAPER = _os.environ.get("CSC_PAPER", _os.path.join(CSC_REPO, "paper"))
CSC_PAPER_DATA = _os.path.join(CSC_PAPER, "data")
# the run configs are carried by the repository, so they resolve without a working tree
CSC_CONFIG = _os.environ.get("CSC_CONFIG", _os.path.join(CSC_REPO, "configs"))
# -----------------------------------------------------------------------------

import json, os, pickle, sys
import numpy as np, torch, yaml
from scipy.stats import hypergeom
D=CSC_WORK
OUT=CSC_RUNS + "/selections"
sys.path.insert(0,D); os.chdir(D)
from model.policy import VEnsemble
cfg=yaml.safe_load(open("config.yaml")); dev="cuda" if torch.cuda.is_available() else "cpu"
QS=[0.85,0.80,0.75,0.70,0.65,0.60,0.55,0.50,0.45,0.40,0.35,0.30]
ALPHA,DELTA,NCAL,REPS=0.40,0.1,200,500
# deployed V seed per task (lowest certifying seed of calfilt_a40)
JOBS=[("cargoal2",0,25),("pointgoal1_dsrl",0,25),("pointgoal2",1,25)]
# CarRun echo: its certified selection (paper ahat=0.138) was also a lost
# scratchpad h5 and is not reproducible from any current (V seed, quantile),
# so it is rebuilt here too. ALPHA_BY_TASK overrides the alpha=0.40 default.
ALPHA_BY_TASK={"carrun_b":0.25}
JOBS.append(("carrun_b",0,10))
summary={}
for task,vs,lim in JOBS:
    tc=cfg["tasks"][task]
    trajs=pickle.load(open(tc["data_pickle"],"rb"))
    cost=np.array([float(np.sum(t["costs"])) for t in trajs]); unsafe=(cost>lim).astype(float)
    ens=VEnsemble(trajs[0]["observations"].shape[1],256,K=3).to(dev)
    ck=torch.load(f"outputs/{task}/v_ensemble_pess_seed{vs}.pt",map_location=dev,weights_only=False)
    ens.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck); ens.eval()
    g=np.zeros(len(trajs))
    with torch.no_grad():
        for i,t in enumerate(trajs):
            o=torch.as_tensor(t["observations"],dtype=torch.float32).to(dev)
            g[i]=torch.cat([ens(o[j:j+8192]).cpu() for j in range(0,len(o),8192)]).mean().item()
    A=ALPHA_BY_TASK.get(task,ALPHA)
    taus=[float(np.quantile(g,q)) for q in QS]; Ns=[int((g>=t).sum()) for t in taus]
    rng=np.random.default_rng(500+vs); picks={}
    for _ in range(REPS):
        cal=rng.choice(len(g),NCAL,replace=False); cs,cu=g[cal],unsafe[cal]
        chosen=None
        for j,t in enumerate(taus):
            sel=cs>=t; m,k=int(sel.sum()),int(cu[sel].sum())
            ks=int(A*Ns[j])+1
            p=float(hypergeom.cdf(k,Ns[j],ks,m)) if (m>0 and ks<=Ns[j]) else 1.0
            if m>0 and p<=DELTA: chosen=j
            else: break
        if chosen is not None: picks[chosen]=picks.get(chosen,0)+1
    if not picks:
        print(f"  {task}: NEVER CERTIFIES"); continue
    top=sorted(picks.items(), key=lambda kv:-kv[1])[:3]
    sels=[]
    for j,cnt in top:
        kept=np.where(g>=taus[j])[0]; c=float(unsafe[kept].mean())
        name=f"{task}_a40new_q{int(QS[j]*100)}" if task!="carrun_b" else f"carrun_b_echonew_q{int(QS[j]*100)}"
        json.dump({"kept":[int(i) for i in kept],"quantile":QS[j],"tau":taus[j],
                   "n_kept":int(len(kept)),"contamination":c,"v_seed":vs,"task":task,
                   "prob":cnt/REPS},open(f"{OUT}/{name}_kept.json","w"))
        sels.append({"name":name,"quantile":QS[j],"prob":cnt/REPS,
                     "n_kept":int(len(kept)),"contamination":c})
    sels.sort(key=lambda s:s["contamination"])
    summary[task]={"v_seed":vs,"cert_rate":sum(picks.values())/REPS,"selections":sels}
    print(f"  {task} (V seed {vs}): cert {sum(picks.values())/REPS:.2f} | " +
          " | ".join(f"q{int(s['quantile']*100)} p={s['prob']:.2f} n={s['n_kept']} ahat={s['contamination']:.3f}"
                     for s in sels), flush=True)
json.dump(summary, open(f"{OUT}/a40new_summary.json","w"), indent=1)
print("A40 ROWS REBUILT")
