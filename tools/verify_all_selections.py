"""Content check for the stale-artifact error class across every paper task.

For each calfilt_* arm, recompute the selection from the CURRENT V ensemble
using the tau recorded in calfilt_meta and compare the resulting size to the
recorded n_kept. A mismatch means the ensemble that produced the published
selection is gone, so that arm's result cannot be regenerated.
"""

# --- paths ------------------------------------------------------------------
# Datasets, checkpoints and run output live outside the repository. Set CSC_WORKSPACE,
# or the individual roots, to point at yours. See the README.
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
CSC_WORK = _os.environ.get("CSC_WORK", _os.path.join(_WS, "datasets"))
_runs = _os.path.join(_WS, "runs")
CSC_RUNS = _os.environ.get("CSC_RUNS", _runs if _os.path.isdir(_runs) else _os.path.join(CSC_REPO, "runs"))
CSC_OSRL = _os.environ.get("CSC_OSRL", _os.path.join(_WS, "osrl"))
CSC_PAPER = _os.environ.get("CSC_PAPER", _os.path.join(CSC_REPO, "paper"))
CSC_PAPER_DATA = _os.path.join(CSC_PAPER, "data")
# the run configs ship with the repository, so they resolve on their own
CSC_CONFIG = _os.environ.get("CSC_CONFIG", _os.path.join(CSC_REPO, "configs"))
# -----------------------------------------------------------------------------

import glob, json, os, pickle, sys
import numpy as np, torch, yaml
D=CSC_WORK
sys.path.insert(0,D); os.chdir(D)
from model.policy import VEnsemble
cfg=yaml.safe_load(open("config.yaml"))
dev="cuda" if torch.cuda.is_available() else "cpu"
import sys as _s
TASKS=(_s.argv[1].split(",") if len(_s.argv)>1 else
       ["halfcheetah_velocity","walker2d_velocity","ant_velocity","hopper_velocity",
        "swimmer_velocity","cargoal1_dsrl","cargoal2","pointgoal1_dsrl","pointgoal2",
        "pointbutton1","pointbutton2","carbutton1_t3","carbutton2","pointcircle1","pointcircle2"])
bad=[]; good=0; skipped=0
for task in TASKS:
    tc=cfg["tasks"].get(task)
    if not tc: print(f"  {task}: not in config"); continue
    metas=sorted(glob.glob(f"outputs/{task}/calfilt_meta_*.json"))
    if not metas: continue
    trajs=pickle.load(open(tc["data_pickle"],"rb"))
    cache={}
    for mp in metas:
        base=os.path.basename(mp)[len("calfilt_meta_"):-len(".json")]
        seed=base.rsplit("_seed",1)[-1]
        ens_p=f"outputs/{task}/v_ensemble_pess_seed{seed}.pt"
        if not os.path.exists(ens_p): skipped+=1; continue
        try: meta=json.load(open(mp))
        except Exception: skipped+=1; continue
        if "tau" not in meta or "n_kept" not in meta: skipped+=1; continue
        if seed not in cache:
            ens=VEnsemble(trajs[0]["observations"].shape[1],256,K=3).to(dev)
            ck=torch.load(ens_p,map_location=dev,weights_only=False)
            ens.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck); ens.eval()
            g=np.zeros(len(trajs))
            with torch.no_grad():
                for i,t in enumerate(trajs):
                    o=torch.as_tensor(t["observations"],dtype=torch.float32).to(dev)
                    g[i]=torch.cat([ens(o[j:j+8192]).cpu() for j in range(0,len(o),8192)]).mean().item()
            cache[seed]=g
        g=cache[seed]
        n=int((g>=float(meta["tau"])).sum())
        if n<50: n=50
        if n!=int(meta["n_kept"]):
            bad.append((task,base,n,int(meta["n_kept"])))
        else: good+=1
    print(f"  {task}: {len(metas)} metas checked", flush=True)
print(f"\n  VERIFIED {good}   MISMATCH {len(bad)}   skipped {skipped}")
for t,b,n,rec in bad:
    print(f"      MISMATCH {t} {b}: recompute={n} recorded={rec}")
