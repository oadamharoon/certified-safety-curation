"""Locate the 6 alpha=0.40 grid selections that do not match at the deployed V
seed, by scanning every (V seed, quantile) pair for the published contamination."""

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
D=CSC_WORK
OUT=CSC_RUNS + "/selections"
sys.path.insert(0,D); os.chdir(D)
from model.policy import VEnsemble
cfg=yaml.safe_load(open("config.yaml")); dev="cuda" if torch.cuda.is_available() else "cpu"
QS=[round(0.85-0.05*i,2) for i in range(16)]
MISS={"cargoal2":(25,[0.287,0.334]),"pointgoal1_dsrl":(25,[0.345]),
      "pointgoal2":(25,[0.269,0.335,0.382])}
for task,(lim,targets) in MISS.items():
    tc=cfg["tasks"][task]
    trajs=pickle.load(open(tc["data_pickle"],"rb"))
    cost=np.array([float(np.sum(t["costs"])) for t in trajs]); unsafe=(cost>lim).astype(int)
    print(f"  {task}")
    found={}
    for vs in range(5):
        p=f"outputs/{task}/v_ensemble_pess_seed{vs}.pt"
        if not os.path.exists(p): continue
        ens=VEnsemble(trajs[0]["observations"].shape[1],256,K=3).to(dev)
        ck=torch.load(p,map_location=dev,weights_only=False)
        ens.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck); ens.eval()
        g=np.zeros(len(trajs))
        with torch.no_grad():
            for i,t in enumerate(trajs):
                o=torch.as_tensor(t["observations"],dtype=torch.float32).to(dev)
                g[i]=torch.cat([ens(o[j:j+8192]).cpu() for j in range(0,len(o),8192)]).mean().item()
        for q in QS:
            tau=float(np.quantile(g,q)); k=g>=tau; c=float(unsafe[k].mean())
            for tgt in targets:
                if abs(c-tgt)<=0.003 and tgt not in found:
                    kept=np.where(k)[0]
                    fp=os.path.join(OUT,f"{task}_a40grid_s{vs}q{int(q*100)}_kept.json")
                    json.dump({"kept":[int(i) for i in kept],"quantile":q,"tau":tau,
                               "n_kept":int(k.sum()),"contamination":c,"v_seed":vs,
                               "task":task,"target_contamination":tgt},open(fp,"w"))
                    found[tgt]=(vs,q,int(k.sum()),c)
                    print(f"      target {tgt:.3f} -> V seed {vs} q{int(q*100)} n={k.sum()} c={c:.3f} FOUND")
    for tgt in targets:
        if tgt not in found: print(f"      target {tgt:.3f}: NOT FOUND on any seed/quantile")
